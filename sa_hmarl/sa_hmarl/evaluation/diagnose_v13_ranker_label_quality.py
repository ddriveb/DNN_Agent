"""Lightweight diagnostic of v1.3 ranker label quality and regret.

For a small set of post-warmup decision states, evaluates the *training* candidate
pool (ppo_r_topk_only, top-30 PPO-R proposals) under H=5/20/50 PPO-R continuation,
scores the candidates with the deployed ranker, and reports:
- per-group label range and H-vs-long-horizon Spearman,
- ranker regret vs the H-step oracle,
- KSP-FF regret vs the H-step oracle,
- how these regrets vary with server utilization.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _rollout_future,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import (
    _load_ranker_policy_from_checkpoint,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


@dataclass
class Cand:
    action_idx: int
    return_h5: float
    return_h20: Optional[float] = None
    return_h50: Optional[float] = None
    ranker_score: Optional[float] = None


@dataclass
class State:
    seed: int
    request: int
    server_util: float
    selected_valid_r: int
    raw_r_count: int
    candidates: List[Cand] = field(default_factory=list)
    ksp_action: Optional[int] = None
    ppo_action: Optional[int] = None


def _compute_return(info: Dict[str, Any], future: Dict[str, Any], args) -> float:
    current_blocked = 0.0 if info.get("success", False) else 1.0
    return float(
        -args.return_current_block_coef * current_blocked
        - args.return_future_block_coef * float(future.get("blocked", 0))
        - args.return_future_nsb_coef * float(future.get("no_suitable_block", 0))
        - args.return_future_server_overload_coef * float(future.get("server_overload", 0))
        - args.return_delay_coef * float(future.get("delay_mean", 0.0))
        - args.return_fs_coef * float(future.get("avg_fs", 0.0))
    )


def _ppo_topk(agent_r, obs_r, legal: List[int], top_k: int) -> List[int]:
    features, mask = agent_r.build_action_features(obs_r)
    logits = agent_r.policy_net(
        torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
    ).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    if not legal:
        return []
    local = {int(a): i for i, a in enumerate(legal)}
    scores = np.asarray([logits[local[a]] for a in legal])
    order = np.argsort(-scores, kind="stable")[: min(top_k, len(legal))]
    return [int(legal[i]) for i in order]


def _evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, args.seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        path_sort_strategy=args.path_sort_strategy, k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    ranker_policy = _load_ranker_policy_from_checkpoint(args.ranker_checkpoint, args.device)
    # Force training-time candidate construction: PPO-R top-K only.
    ranker_policy.candidate_mode = "ppo_r_topk_only"
    ranker_policy.max_candidates = args.train_max_candidates
    ranker_policy.ppo_top_k = args.train_ppo_top_k

    rng = np.random.RandomState(args.seed)
    src = int(rng.randint(0, env_proto.net.NUM_NODES))
    requests = generate_requests(
        env_proto, rng, src, args.requests_per_episode,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )

    env = make_env(
        args.topology, args.num_slots, args.num_servers, args.seed,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        path_sort_strategy=args.path_sort_strategy, k=args.k_paths,
    )
    env.reset(requests)

    states: List[State] = []
    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        if step_idx < args.warmup_requests:
            obs_c = build_agent_c_observation(env, req)
            c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            action_r = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), action_r)
            continue
        if len(states) >= args.max_states:
            break

        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        if len(legal) < 2:
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            action_r = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), action_r)
            continue

        snapshot = _snapshot_before_r_decision(env, req.req_id)
        cands = _ppo_topk(agent_r, obs_r, legal, args.train_ppo_top_k)[:args.train_max_candidates]

        st = State(
            seed=args.seed, request=step_idx,
            server_util=float(obs_c["server_utilizations"][server_id]),
            selected_valid_r=int(obs_c["feasible_counts"][split_id][server_id]),
            raw_r_count=len(legal),
            ksp_action=int(ksp_ff_highest_mod_action(obs_r)) if ksp_ff_highest_mod_action(obs_r) is not None else None,
            ppo_action=int(_select_r_action_from_obs(agent_r, obs_r, env.max_blocks)[0]),
        )

        # Ranker scores for the candidate set it would have chosen.
        cands_for_ranker, scores_for_ranker = ranker_policy.score_legal_actions(
            env, req, obs_c, obs_r, agent_r, split_id, server_id
        )
        score_map = {int(a): float(s) for a, s in zip(cands_for_ranker, scores_for_ranker)}

        for r_idx in cands:
            branch = copy.deepcopy(snapshot)
            action_r = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
            _, _, _, info = branch.step((split_id, server_id), action_r)
            future5 = _rollout_future(copy.deepcopy(branch), requests, step_idx + 1, 5, agent_c, agent_r, args.num_servers, args.util_threshold, args.alpha)
            ret5 = _compute_return(info, future5, args)
            future20 = _rollout_future(copy.deepcopy(branch), requests, step_idx + 1, 20, agent_c, agent_r, args.num_servers, args.util_threshold, args.alpha)
            ret20 = _compute_return(info, future20, args)
            future50 = _rollout_future(copy.deepcopy(branch), requests, step_idx + 1, 50, agent_c, agent_r, args.num_servers, args.util_threshold, args.alpha)
            ret50 = _compute_return(info, future50, args)
            st.candidates.append(Cand(
                action_idx=int(r_idx),
                return_h5=ret5,
                return_h20=ret20,
                return_h50=ret50,
                ranker_score=score_map.get(int(r_idx)),
            ))
        states.append(st)

        # Advance real env with PPO-R.
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        action_r = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), action_r)

    return _summarize(states, args)


def _summarize(states: List[State], args) -> Dict[str, Any]:
    ranges_h5, ranges_h20, ranges_h50 = [], [], []
    sp_h5_h20, sp_h5_h50 = [], []
    ranker_regret_h5, ranker_regret_h20, ranker_regret_h50 = [], [], []
    ksp_regret_h5, ksp_regret_h20, ksp_regret_h50 = [], [], []
    ppo_regret_h5 = []

    util_buckets = [(0.7, "<=0.7"), (0.8, "<=0.8"), (0.9, "<=0.9"), (1.0, "<=1.0")]
    bucketed = {label: [] for _, label in util_buckets}

    for st in states:
        cands = [c for c in st.candidates if c.ranker_score is not None]
        if not cands:
            continue
        # Label ranges (all candidates, not only successful).
        rets5 = [c.return_h5 for c in st.candidates]
        rets20 = [c.return_h20 for c in st.candidates]
        rets50 = [c.return_h50 for c in st.candidates]
        ranges_h5.append(float(max(rets5) - min(rets5)))
        ranges_h20.append(float(max(rets20) - min(rets20)))
        ranges_h50.append(float(max(rets50) - min(rets50)))

        # Spearman between H5 and longer horizons.
        if len(rets5) >= 2:
            rho20, _ = stats.spearmanr(rets5, rets20)
            if not np.isnan(rho20):
                sp_h5_h20.append(float(rho20))
            rho50, _ = stats.spearmanr(rets5, rets50)
            if not np.isnan(rho50):
                sp_h5_h50.append(float(rho50))

        def _best(horizon):
            arr = [(getattr(c, f"return_{horizon}"), c.action_idx) for c in st.candidates]
            best = max(arr, key=lambda x: x[0])
            return best[0], best[1]

        def _regret(horizon, action_idx):
            best_ret, _ = _best(horizon)
            for c in st.candidates:
                if c.action_idx == action_idx:
                    return float(best_ret - getattr(c, f"return_{horizon}"))
            return None

        ranker_action = max(cands, key=lambda c: c.ranker_score).action_idx
        r_h5 = _regret("h5", ranker_action)
        r_h20 = _regret("h20", ranker_action)
        r_h50 = _regret("h50", ranker_action)
        if r_h5 is not None:
            ranker_regret_h5.append(r_h5)
        if r_h20 is not None:
            ranker_regret_h20.append(r_h20)
        if r_h50 is not None:
            ranker_regret_h50.append(r_h50)

        if st.ksp_action is not None:
            k_h5 = _regret("h5", st.ksp_action)
            k_h20 = _regret("h20", st.ksp_action)
            k_h50 = _regret("h50", st.ksp_action)
            if k_h5 is not None:
                ksp_regret_h5.append(k_h5)
            if k_h20 is not None:
                ksp_regret_h20.append(k_h20)
            if k_h50 is not None:
                ksp_regret_h50.append(k_h50)
            for threshold, label in util_buckets:
                if st.server_util <= threshold:
                    bucketed[label].append((k_h5 if k_h5 is not None else 0.0, r_h5 if r_h5 is not None else 0.0))
                    break

        if st.ppo_action is not None:
            p_h5 = _regret("h5", st.ppo_action)
            if p_h5 is not None:
                ppo_regret_h5.append(p_h5)

    def _mean(xs):
        return float(np.mean(xs)) if xs else 0.0

    bucket_summary = {}
    for label, pairs in bucketed.items():
        bucket_summary[label] = {
            "states": len(pairs),
            "mean_ksp_regret_h5": _mean([p[0] for p in pairs]),
            "mean_ranker_regret_h5": _mean([p[1] for p in pairs]),
        }

    return {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "states": len(states),
        "summary": {
            "mean_label_range_h5": _mean(ranges_h5),
            "mean_label_range_h20": _mean(ranges_h20),
            "mean_label_range_h50": _mean(ranges_h50),
            "nonzero_range_h5": float(np.mean([r > 1e-6 for r in ranges_h5])) if ranges_h5 else 0.0,
            "spearman_h5_h20": _mean(sp_h5_h20),
            "spearman_h5_h50": _mean(sp_h5_h50),
            "mean_ranker_regret_h5": _mean(ranker_regret_h5),
            "mean_ranker_regret_h20": _mean(ranker_regret_h20),
            "mean_ranker_regret_h50": _mean(ranker_regret_h50),
            "mean_ksp_regret_h5": _mean(ksp_regret_h5),
            "mean_ksp_regret_h20": _mean(ksp_regret_h20),
            "mean_ksp_regret_h50": _mean(ksp_regret_h50),
            "mean_ppo_regret_h5": _mean(ppo_regret_h5),
            "bucket_summary": bucket_summary,
        },
        "elapsed_seconds": time.time() - args._start_time,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranker_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt")
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=50)
    parser.add_argument("--path_sort_strategy", default="hops", choices=["km", "hops"])
    parser.add_argument("--block_sort_strategy", default="start_asc")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seed", type=int, default=3030)
    parser.add_argument("--requests_per_episode", type=int, default=2500)
    parser.add_argument("--warmup_requests", type=int, default=2000)
    parser.add_argument("--max_states", type=int, default=50)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--train_ppo_top_k", type=int, default=50)
    parser.add_argument("--train_max_candidates", type=int, default=30)
    parser.add_argument("--return_current_block_coef", type=float, default=3.0)
    parser.add_argument("--return_future_block_coef", type=float, default=4.0)
    parser.add_argument("--return_future_nsb_coef", type=float, default=3.0)
    parser.add_argument("--return_future_server_overload_coef", type=float, default=0.0)
    parser.add_argument("--return_delay_coef", type=float, default=0.03)
    parser.add_argument("--return_fs_coef", type=float, default=0.05)
    parser.add_argument("--output_json", default="sa_hmarl/experiments/v13_ranker_label_quality_diagnostic.json")
    return parser


def main():
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    report = _evaluate(args)
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"Saved {out}")
    print(json.dumps(report["summary"], indent=2))


if __name__ == "__main__":
    main()
