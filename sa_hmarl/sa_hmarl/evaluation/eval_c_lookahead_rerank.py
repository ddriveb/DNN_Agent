"""Closed-loop evaluation of C-side look-ahead reranking.

Agent-C first produces its top-K legal (split, server) candidates.  For each
candidate we build the downstream Agent-R observation, count legal R actions,
and query the frozen v1 ranking model for the best R score.  The C candidate
is then re-ranked by:

    score = PPO-C logit + λ * best_R_rank_score + β * log1p(legal_R_count)

The highest-scoring C candidate is executed, and Agent-R uses the frozen v1
ranker.  This tests whether looking ahead to R feasibility reduces zero-legal
states and blocking.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.agents.post_decision_value import PostDecisionValueNetwork
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
    _get_ppo_c_logits,
    _select_ppo_c_action,
    _zscore,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop_v2 import (
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES, _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _get_r_logits(agent_r, obs_r: Dict[str, Any]) -> np.ndarray:
    features, mask = agent_r.build_action_features(obs_r)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    return logits


def _select_rank_only_r_action(
    env, req, obs_c, obs_r, agent_r, model, mean, std,
    split_id: int, server_id: int, device: str,
) -> int:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        scores = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy()
    return int(legal[int(np.argmax(scores))])


def _lookahead_c_score(
    c_idx: int,
    logit: float,
    env,
    req,
    obs_c,
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    lambda_value: float,
    beta_value: float,
    device: str,
) -> Tuple[float, int]:
    split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    legal_count = len(legal)
    if legal_count == 0:
        best_score = -1e9
    else:
        online = np.stack([
            _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
            for a in legal
        ])
        normalized = (online - rank_mean) / rank_std
        with torch.no_grad():
            scores = rank_model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy()
        best_score = float(scores.max())
    combined = logit + lambda_value * best_score + beta_value * float(np.log1p(legal_count))
    return combined, legal_count


def _select_c_lookahead_action(
    agent_c,
    agent_r,
    env,
    req,
    obs_c: Dict[str, Any],
    num_slots: int,
    rank_model,
    rank_mean,
    rank_std,
    top_k_c: int,
    lambda_value: float,
    beta_value: float,
    device: str,
) -> Tuple[int, int]:
    """Return selected C action and number of candidates evaluated."""
    c_action_ppo, raw_c_mask, risk_c_mask = _select_c_action_from_obs(agent_c, obs_c, num_slots)
    legal = np.flatnonzero(np.asarray(risk_c_mask, dtype=bool)).tolist()
    if not legal:
        return int(c_action_ppo), 0
    logits_all = _get_ppo_c_logits(agent_c, obs_c, np.asarray(risk_c_mask, dtype=bool))

    if top_k_c <= 0 or top_k_c >= len(legal):
        topk_indices = legal
    else:
        order = np.argsort(-logits_all, kind="stable")[:top_k_c]
        topk_indices = [int(legal[i]) for i in order]

    best_c = int(c_action_ppo)
    best_score = -1e18
    for local_idx, c_idx in enumerate(topk_indices):
        logit = float(logits_all[local_idx])
        score, _ = _lookahead_c_score(
            c_idx, logit, env, req, obs_c, agent_r, rank_model, rank_mean, rank_std,
            lambda_value, beta_value, device,
        )
        if score > best_score + 1e-9:
            best_score = score
            best_c = c_idx
    return best_c, len(topk_indices)


def _run_episode(
    env, requests, agent_c, agent_r,
    rank_model, rank_mean, rank_std,
    args, metrics: PerMethodMetrics, mode: str,
    top_k_c: int = 1,
    lambda_value: float = 0.0,
    beta_value: float = 0.0,
    deep_rmsa=None,
) -> None:
    # Track C-action changes for look-ahead mode.
    c_changed_count = 0
    c_evaluated_count = 0
    c_evaluated_n = 0

    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        raw_empty = int(raw_c_mask.sum()) == 0

        if mode == "ppo_r":
            c_idx, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            r_action = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
        elif mode == "counterfactual_rank_only":
            c_idx, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_idx = _select_rank_only_r_action(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        elif mode == "c_lookahead_rerank":
            ppo_c_idx, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
            c_idx, n_eval = _select_c_lookahead_action(
                agent_c, agent_r, env, req, obs_c, env.net.num_slots,
                rank_model, rank_mean, rank_std,
                top_k_c, lambda_value, beta_value, args.device,
            )
            c_evaluated_count += n_eval
            c_evaluated_n += 1
            if int(c_idx) != int(ppo_c_idx):
                c_changed_count += 1
            split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_idx = _select_rank_only_r_action(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, args.device,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        elif mode == "deep_rmsa":
            c_idx, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
            r_action = (0, 0, 0) if r_idx is None else decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
            r_idx = r_idx if r_idx is not None else -1
        elif mode == "ksp_bf":
            c_idx, _ = _select_ppo_c_action(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(int(c_idx), len(env.mec.servers))
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_idx = ksp_bf_action(obs_r)
            r_action = (0, 0, 0) if r_idx is None else decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
            r_idx = r_idx if r_idx is not None else -1
        else:
            raise ValueError(f"Unknown mode: {mode}")

        decision_ms = (time.perf_counter() - started) * 1000.0
        _, _, _, info = env.step((split_id, server_id), r_action)
        same_as_ppo = True
        if r_idx is not None:
            ppo_idx = agent_r.select_action(obs_r, deterministic=True)
            same_as_ppo = int(r_idx) == int(ppo_idx if ppo_idx is not None else 0)
        _record_outcome(
            metrics, info, raw_empty,
            obs_c, split_id, server_id, decision_ms, same_as_ppo,
        )
        metrics.active_connections[-1] = len(env.active_connections)

    # Attach C-change stats to the metrics instance for later retrieval.
    metrics.c_changed_count = c_changed_count
    metrics.c_evaluated_avg = (c_evaluated_count / max(c_evaluated_n, 1)) if c_evaluated_n else 0.0


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
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
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)

    deep_rmsa = None
    if "deep_rmsa" in [item.strip() for item in args.methods.split(",")]:
        deep_rmsa = _load_deep_rmsa(
            args.deep_rmsa_checkpoint,
            env_proto,
            ModulationRegistry.from_profile(args.modulation_profile),
            args.device,
        )

    episodes_by_seed = {}
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

    requested = [item.strip() for item in args.methods.split(",") if item.strip()]
    methods: Dict[str, Dict[str, Any]] = {}
    if "ppo_r" in requested:
        methods["ppo_r"] = {"per_seed": {}, "mode": "ppo_r"}
    if "counterfactual_rank_only" in requested:
        methods["counterfactual_rank_only"] = {"per_seed": {}, "mode": "counterfactual_rank_only"}
    if "c_lookahead_rerank" in requested:
        top_k_values = [int(k.strip()) for k in args.top_k_c_values.split(",") if k.strip()]
        lambda_values = [float(v.strip()) for v in args.lambda_values.split(",") if v.strip()]
        beta_values = [float(v.strip()) for v in args.beta_values.split(",") if v.strip()]
        for k in top_k_values:
            for lam in lambda_values:
                for beta in beta_values:
                    name = f"c_lookahead_top{k}_lambda{lam:g}_beta{beta:g}"
                    methods[name] = {
                        "per_seed": {},
                        "mode": "c_lookahead_rerank",
                        "top_k_c": k,
                        "lambda": lam,
                        "beta": beta,
                    }
    if "deep_rmsa" in requested:
        methods["deep_rmsa"] = {"per_seed": {}, "mode": "deep_rmsa"}
    if "ksp_bf" in requested:
        methods["ksp_bf"] = {"per_seed": {}, "mode": "ksp_bf"}

    for method_name, spec in methods.items():
        for seed in seeds:
            metrics = PerMethodMetrics()
            for requests in episodes_by_seed[seed]:
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                env.reset(requests)
                _run_episode(
                    env, requests, agent_c, agent_r,
                    rank_model, rank_mean, rank_std,
                    args, metrics, spec["mode"],
                    spec.get("top_k_c", 1),
                    spec.get("lambda", 0.0),
                    spec.get("beta", 0.0),
                    deep_rmsa,
                )
            agg = _aggregate_metrics(metrics)
            n = max(metrics.total, 1)
            agg["c_action_changed_rate"] = float(getattr(metrics, "c_changed_count", 0)) / n
            agg["c_evaluated_avg"] = float(getattr(metrics, "c_evaluated_avg", 0.0))
            spec["per_seed"][str(seed)] = agg
            print(f"[{method_name}] completed seed {seed}", flush=True)
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key in next(iter(spec["per_seed"].values()))
            if isinstance(next(iter(spec["per_seed"].values()))[key], (int, float))
        }
    return {
        "config": vars(args),
        "methods": methods,
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# C-Side Look-Ahead Reranking Closed-Loop Evaluation", "",
        "| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Avg FS | Avg waste | C changed | C eval avg |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in report["methods"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {name} | {agg['blocking_rate']:.2%} | "
            f"{agg['raw_mask_empty_rate']:.2%} | {agg['no_suitable_block_rate']:.2%} | "
            f"{agg['server_overload_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{agg['avg_fs']:.2f} | {agg['avg_waste']:.3f} | "
            f"{agg.get('c_action_changed_rate', 0):.2%} | {agg.get('c_evaluated_avg', 0):.1f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_k5m10_h5/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="ppo_r,counterfactual_rank_only,c_lookahead_rerank,deep_rmsa,ksp_bf")
    parser.add_argument("--top_k_c_values", default="3,5")
    parser.add_argument("--lambda_values", default="0.25,0.5,1.0")
    parser.add_argument("--beta_values", default="0.0,0.1,0.3")
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/c_lookahead_rerank_eval.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/c_lookahead_rerank_eval.md")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    result = evaluate(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    _write_markdown(out_md, result)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
