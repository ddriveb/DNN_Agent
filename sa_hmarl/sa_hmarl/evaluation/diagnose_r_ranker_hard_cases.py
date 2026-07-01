"""Mine hard cases for the counterfactual R-side ranker.

Run a rank-only trajectory. At every R decision point, snapshot the environment,
then counterfactually evaluate what DeepRMSA (and optionally PPO-R) would have
done in the *same* state. A hard case is defined as:

    rank_only blocked  AND  DeepRMSA would have succeeded (counterfactual)

For each hard case we record whether DeepRMSA's action is present in the v1
candidate set, the ranker score distribution, and the action features. This
tells us whether the gap is:

  - candidate missing: DeepRMSA action not in v1 candidates, or
  - ranking error: DeepRMSA action is in candidates but scored below rank-only's
    chosen action.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop_v2 import (
    _select_rank_only_r_action,
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import (
    _select_candidate_actions_v1,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES, _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


R_FEATURE_ORDER = {
    "path_length_km": 0,
    "hop_count": 1,
    "lfb": 2,
    "free_ratio": 3,
    "frag_index": 4,
    "spectral_efficiency": 5,
    "reach_km": 6,
    "required_fs": 7,
    "block_size": 8,
    "block_waste": 9,
    "path_mod_feasible": 10,
}


def _action_features(obs_r: Dict[str, Any], r_features: np.ndarray, action_idx: int) -> Dict[str, Any]:
    if action_idx < 0:
        return None
    num_mods = len(obs_r["mod_names"])
    max_blocks = len(obs_r["agent_r_mask"]) // (len(obs_r["candidate_paths"]) * num_mods)
    path_idx, mod_idx, block_idx = decode_agent_r_action(action_idx, num_mods, max_blocks)
    feat = r_features[action_idx]
    return {
        "path_idx": int(path_idx),
        "mod_idx": int(mod_idx),
        "block_idx": int(block_idx),
        "path_length_km": float(feat[R_FEATURE_ORDER["path_length_km"]]),
        "hop_count": float(feat[R_FEATURE_ORDER["hop_count"]]),
        "spectral_efficiency": float(feat[R_FEATURE_ORDER["spectral_efficiency"]]),
        "required_fs": float(feat[R_FEATURE_ORDER["required_fs"]]),
        "block_size": float(feat[R_FEATURE_ORDER["block_size"]]),
        "block_waste": float(feat[R_FEATURE_ORDER["block_waste"]]),
        "path_mod_feasible": int(feat[R_FEATURE_ORDER["path_mod_feasible"]]),
    }


def _ranker_value_distribution(
    env, req, obs_c, obs_r, agent_r, model, mean, std, split_id, server_id, device
):
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return None, legal
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        values = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    return values, legal


def _counterfactual_success(env_snapshot, req, c_action, r_action):
    env_cf = copy.deepcopy(env_snapshot)
    env_cf.advance_time(req.arrival_time)
    _, _, _, info = env_cf.step(c_action, r_action)
    return bool(info.get("success", False)), info.get("reason", "")


def _candidate_args() -> argparse.Namespace:
    """Minimal args namespace for v1 candidate selection."""
    ns = argparse.Namespace()
    ns.ppo_top_k = 8
    ns.num_random_candidates = 5
    ns.min_candidates = 15
    ns.max_candidates = 30
    ns.candidate_seed = 12345
    return ns


def run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)

    episodes_by_seed: Dict[int, List[List[Any]]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max, args.size_min_mb,
                args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile,
            ))
        episodes_by_seed[seed] = episodes

    cand_args = _candidate_args()
    hard_cases: List[Dict[str, Any]] = []
    all_decisions: List[Dict[str, Any]] = []

    for seed, episodes in episodes_by_seed.items():
        for requests in episodes:
            env = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            for req in requests:
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                c_idx = agent_c.select_action(obs_c, deterministic=True)
                if c_idx is None:
                    c_idx = 0
                split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
                c_action = (split_id, server_id)
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                r_features, r_mask = agent_r.build_action_features(obs_r)
                legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()

                snapshot = _snapshot_before_r_decision(env, req.req_id)

                rank_idx = _select_rank_only_r_action(
                    env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                    split_id, server_id, args.device,
                )
                rank_action = decode_agent_r_action(rank_idx, len(obs_r["mod_names"]), env.max_blocks)

                deep_idx = deep_rmsa.select_action(obs_r)
                deep_action = (0, 0, 0) if deep_idx is None else decode_agent_r_action(
                    deep_idx, len(obs_r["mod_names"]), env.max_blocks
                )

                ppo_idx = agent_r.select_action(obs_r, deterministic=True)
                ppo_action = (0, 0, 0) if ppo_idx is None else decode_agent_r_action(
                    ppo_idx, len(obs_r["mod_names"]), env.max_blocks
                )

                # Counterfactual successes.
                rank_success, rank_reason = _counterfactual_success(snapshot, req, c_action, rank_action)
                deep_success, deep_reason = _counterfactual_success(snapshot, req, c_action, deep_action)
                ppo_success, ppo_reason = _counterfactual_success(snapshot, req, c_action, ppo_action)

                # Candidate set and ranker scores.
                candidate_actions = _select_candidate_actions_v1(
                    agent_r, obs_r, r_features, legal, env.max_blocks, cand_args,
                ) if legal else []
                values, value_legal = _ranker_value_distribution(
                    env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                    split_id, server_id, args.device,
                )
                score_by_action = {}
                if values is not None:
                    score_by_action = {int(a): float(v) for a, v in zip(value_legal, values)}

                rank_score = score_by_action.get(int(rank_idx))
                deep_score = score_by_action.get(int(deep_idx)) if deep_idx is not None else None
                ppo_score = score_by_action.get(int(ppo_idx)) if ppo_idx is not None else None

                decision = {
                    "seed": int(seed),
                    "req_id": int(req.req_id),
                    "legal_count": int(len(legal)),
                    "selected_valid": int(obs_c["feasible_counts"][split_id][server_id]),
                    "rank_idx": int(rank_idx),
                    "deep_idx": int(deep_idx) if deep_idx is not None else -1,
                    "ppo_idx": int(ppo_idx) if ppo_idx is not None else -1,
                    "rank_success": rank_success,
                    "deep_success": deep_success,
                    "ppo_success": ppo_success,
                    "rank_reason": rank_reason,
                    "deep_reason": deep_reason,
                    "ppo_reason": ppo_reason,
                    "candidate_actions": [int(a) for a in candidate_actions],
                    "deep_in_candidates": int(deep_idx) in candidate_actions if deep_idx is not None and candidate_actions else False,
                    "ppo_in_candidates": int(ppo_idx) in candidate_actions if ppo_idx is not None and candidate_actions else False,
                    "rank_score": rank_score,
                    "deep_score": deep_score,
                    "ppo_score": ppo_score,
                    "rank_action_features": _action_features(obs_r, r_features, int(rank_idx)),
                    "deep_action_features": _action_features(obs_r, r_features, int(deep_idx)) if deep_idx is not None else None,
                    "ppo_action_features": _action_features(obs_r, r_features, int(ppo_idx)) if ppo_idx is not None else None,
                }
                all_decisions.append(decision)

                if not rank_success and deep_success:
                    hard_cases.append(decision)

                # Step the real env with rank-only action.
                env.step(c_action, rank_action)

    # Aggregates.
    total = len(all_decisions)
    rank_blocked = sum(1 for d in all_decisions if not d["rank_success"])
    deep_successes = sum(1 for d in all_decisions if d["deep_success"])
    hard_count = len(hard_cases)

    candidate_missing = [d for d in hard_cases if not d["deep_in_candidates"]]
    ranking_error = [d for d in hard_cases if d["deep_in_candidates"]]

    def _avg(features, key):
        vals = [d[key] for d in features if d is not None]
        return float(np.mean(vals)) if vals else None

    report = {
        "config": vars(args),
        "total_decisions": total,
        "rank_blocked": rank_blocked,
        "deep_successes": deep_successes,
        "hard_cases": hard_count,
        "hard_case_rate": hard_count / total if total else 0.0,
        "candidate_missing": len(candidate_missing),
        "ranking_error": len(ranking_error),
        "candidate_missing_rate": len(candidate_missing) / hard_count if hard_count else 0.0,
        "ranking_error_rate": len(ranking_error) / hard_count if hard_count else 0.0,
        "hard_case_legal_count": {
            "mean": _avg(hard_cases, "legal_count"),
            "median": float(np.median([d["legal_count"] for d in hard_cases])) if hard_cases else None,
        },
        "hard_case_selected_valid": {
            "mean": _avg(hard_cases, "selected_valid"),
            "median": float(np.median([d["selected_valid"] for d in hard_cases])) if hard_cases else None,
        },
        "rank_action_features": {
            "mod_idx": _avg([d["rank_action_features"] for d in hard_cases], "mod_idx"),
            "spectral_efficiency": _avg([d["rank_action_features"] for d in hard_cases], "spectral_efficiency"),
            "block_size": _avg([d["rank_action_features"] for d in hard_cases], "block_size"),
            "block_waste": _avg([d["rank_action_features"] for d in hard_cases], "block_waste"),
        },
        "deep_action_features": {
            "mod_idx": _avg([d["deep_action_features"] for d in hard_cases], "mod_idx"),
            "spectral_efficiency": _avg([d["deep_action_features"] for d in hard_cases], "spectral_efficiency"),
            "block_size": _avg([d["deep_action_features"] for d in hard_cases], "block_size"),
            "block_waste": _avg([d["deep_action_features"] for d in hard_cases], "block_waste"),
        },
        "samples": hard_cases[: args.max_sample_cases],
        "elapsed_seconds": time.time() - getattr(args, "_start_time", time.time()),
    }
    return report


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = ["# R-Ranker Hard-Case Diagnostic", ""]
    def _fmt(value, fmt=".3f"):
        return f"{value:{fmt}}" if value is not None else "N/A"

    lines += [
        "## Summary",
        "",
        f"- Total R decisions: {report['total_decisions']}",
        f"- Rank-only blocked: {report['rank_blocked']}",
        f"- DeepRMSA successes: {report['deep_successes']}",
        f"- Hard cases (rank blocked & DeepRMSA success): {report['hard_cases']} "
        f"({report['hard_case_rate']:.2%})",
        "",
        "## Breakdown of hard cases",
        "",
        f"- Candidate missing (DeepRMSA action not in v1 candidates): {report['candidate_missing']} "
        f"({report['candidate_missing_rate']:.2%})",
        f"- Ranking error (DeepRMSA action in v1 candidates but not selected): {report['ranking_error']} "
        f"({report['ranking_error_rate']:.2%})",
        "",
        "## Hard-case state characteristics",
        "",
        f"- Mean legal R actions: {_fmt(report['hard_case_legal_count']['mean'], '.2f')}",
        f"- Median legal R actions: {_fmt(report['hard_case_legal_count']['median'], '.1f')}",
        f"- Mean selected_valid_r_actions: {_fmt(report['hard_case_selected_valid']['mean'], '.2f')}",
        f"- Median selected_valid_r_actions: {_fmt(report['hard_case_selected_valid']['median'], '.1f')}",
        "",
        "## Rank-only selected action vs DeepRMSA action (hard cases)",
        "",
        "| Feature | Rank-only | DeepRMSA |",
        "|---|---|---:|",
        f"| mod_idx | {_fmt(report['rank_action_features']['mod_idx'])} | {_fmt(report['deep_action_features']['mod_idx'])} |",
        f"| spectral_efficiency | {_fmt(report['rank_action_features']['spectral_efficiency'])} | {_fmt(report['deep_action_features']['spectral_efficiency'])} |",
        f"| block_size | {_fmt(report['rank_action_features']['block_size'], '.2f')} | {_fmt(report['deep_action_features']['block_size'], '.2f')} |",
        f"| block_waste | {_fmt(report['rank_action_features']['block_waste'])} | {_fmt(report['deep_action_features']['block_waste'])} |",
        "",
        "## Interpretation",
        "",
        "- If **candidate_missing** dominates, the v1 candidate generator drops the action that would have saved the request. "
        "Fix: expand candidate coverage, but only on hard cases (not globally).",
        "- If **ranking_error** dominates, the ranker sees the right action but scores it below a worse action. "
        "Fix: hard-case mining / pairwise fine-tuning on these specific states.",
        "- If **hard_cases = 0**, DeepRMSA does not succeed in any state where rank-only blocks. "
        "The gap is then caused by earlier decisions changing the state distribution, not by the ranker's choice at failure states.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_k5m10_h5/ranking_model.pt")
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.15)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_sample_cases", type=int, default=100)
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_ranker_hard_cases.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_ranker_hard_cases.md")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    report = run_diagnostic(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(out_md, report)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
