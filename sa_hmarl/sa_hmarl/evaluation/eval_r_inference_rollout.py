"""Inference-time Top-K rollout reranking for the v1.2 R-ranker.

This is a no-training upper-bound probe.  At each R decision:

1. score legal R actions with the frozen v1.2 counterfactual ranker;
2. keep the Top-K actions;
3. for each Top-K action, simulate the current action and roll out a short
   future horizon using the same frozen PPO-C + v1.2 R policy;
4. choose the action with the best simulated future outcome.

If this does not improve over v1.2 rank-only, then "longer sequence reasoning"
is unlikely to be the missing ingredient for further R-side gains.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

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
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    _select_rank_only_r_action,
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.generate_r_post_decision_dataset import _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _score_legal_ranker(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    split_id: int,
    server_id: int,
    device: str,
) -> Tuple[List[int], np.ndarray]:
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return [], np.empty((0,), dtype=np.float32)
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - mean) / std
    with torch.no_grad():
        scores = model(torch.as_tensor(normalized, dtype=torch.float32, device=device)).cpu().numpy().ravel()
    return legal, scores.astype(np.float32)


def _select_rank_action_and_score(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    model,
    mean: np.ndarray,
    std: np.ndarray,
    split_id: int,
    server_id: int,
    device: str,
) -> Tuple[int, float, int]:
    legal, scores = _score_legal_ranker(
        env, req, obs_c, obs_r, agent_r, model, mean, std, split_id, server_id, device
    )
    if not legal:
        return 0, 0.0, 0
    best = int(np.argmax(scores))
    return int(legal[best]), float(scores[best]), len(legal)


def _rollout_future_with_ranker(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
) -> Dict[str, float]:
    blocked = 0
    no_suitable = 0
    overload = 0
    delay_sum = 0.0
    accepted = 0
    steps = min(int(horizon), max(len(requests) - int(start_idx), 0))
    for offset in range(steps):
        req = requests[start_idx + offset]
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_action, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_idx, _, _ = _select_rank_action_and_score(
            env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
            split_id, server_id, args.device,
        )
        r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)
        if not info.get("success", False):
            blocked += 1
            if info.get("reason") == "no_suitable_block":
                no_suitable += 1
            if info.get("reason") in ("server_overload", "server_saturated"):
                overload += 1
        else:
            accepted += 1
            delay_sum += float(info.get("delay_ms", 0.0))
    return {
        "blocked": float(blocked),
        "no_suitable_block": float(no_suitable),
        "server_overload": float(overload),
        "accepted": float(accepted),
        "delay_sum": float(delay_sum),
        "steps": float(steps),
    }


def _select_rollout_action(
    env,
    requests: List[Any],
    request_index: int,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    split_id: int,
    server_id: int,
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[int, Dict[str, Any]]:
    legal, scores = _score_legal_ranker(
        env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
        split_id, server_id, args.device,
    )
    if not legal:
        return 0, {"legal_count": 0, "topk_count": 0, "changed_vs_rank": False}
    if len(legal) == 1 or args.rollout_top_k <= 1 or args.rollout_horizon <= 0:
        best = int(np.argmax(scores))
        return int(legal[best]), {
            "legal_count": len(legal),
            "topk_count": 1,
            "changed_vs_rank": False,
        }

    order = np.argsort(-scores, kind="stable")
    top_indices = order[: min(int(args.rollout_top_k), len(order))]
    top_actions = [int(legal[i]) for i in top_indices]
    top_scores = [float(scores[i]) for i in top_indices]
    rank_best_action = top_actions[0]

    evaluations = []
    for action_idx, base_score in zip(top_actions, top_scores):
        branch = copy.deepcopy(env)
        r_action = decode_agent_r_action(action_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = branch.step((split_id, server_id), r_action)
        current_block = 0 if info.get("success", False) else 1
        future = _rollout_future_with_ranker(
            branch,
            requests,
            request_index + 1,
            args.rollout_horizon,
            agent_c,
            agent_r,
            rank_model,
            rank_mean,
            rank_std,
            args,
        )
        evaluations.append({
            "action_idx": int(action_idx),
            "base_score": float(base_score),
            "current_block": int(current_block),
            "future_blocked": int(future["blocked"]),
            "future_nsb": int(future["no_suitable_block"]),
            "future_overload": int(future["server_overload"]),
            "future_delay_mean": (
                float(future["delay_sum"]) / max(float(future["accepted"]), 1.0)
                if future["accepted"] > 0 else 0.0
            ),
        })

    # Conservative lexicographic selector: only short-rollout outcomes can
    # overturn the ranker, with base_score as final tie-breaker.
    best = min(
        evaluations,
        key=lambda row: (
            row["current_block"],
            row["future_blocked"],
            row["future_nsb"],
            row["future_overload"],
            -row["base_score"],
            row["action_idx"],
        ),
    )
    return int(best["action_idx"]), {
        "legal_count": len(legal),
        "topk_count": len(top_actions),
        "changed_vs_rank": int(best["action_idx"]) != int(rank_best_action),
        "rank_best_action": int(rank_best_action),
        "selected_eval": best,
    }


def _select_r_for_mode(
    mode: str,
    env,
    requests: List[Any],
    request_index: int,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    split_id: int,
    server_id: int,
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    deep_rmsa,
    args: argparse.Namespace,
) -> Tuple[Tuple[int, int, int], Optional[int], Dict[str, Any]]:
    if mode == "rank_only":
        idx, _, legal_count = _select_rank_action_and_score(
            env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
            split_id, server_id, args.device,
        )
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks), int(idx), {
            "legal_count": legal_count,
            "changed_vs_rank": False,
        }

    if mode == "rollout_topk":
        idx, extra = _select_rollout_action(
            env, requests, request_index, req, obs_c, obs_r, split_id, server_id,
            agent_c, agent_r, rank_model, rank_mean, rank_std, args,
        )
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks), int(idx), extra

    if mode == "deep_rmsa":
        idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
        if idx is None:
            return (0, 0, 0), None, {"legal_count": int(np.asarray(obs_r["agent_r_mask"], dtype=bool).sum())}
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks), int(idx), {
            "legal_count": int(np.asarray(obs_r["agent_r_mask"], dtype=bool).sum())
        }

    if mode == "ksp_bf":
        idx = ksp_bf_action(obs_r)
        if idx is None:
            return (0, 0, 0), None, {"legal_count": int(np.asarray(obs_r["agent_r_mask"], dtype=bool).sum())}
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks), int(idx), {
            "legal_count": int(np.asarray(obs_r["agent_r_mask"], dtype=bool).sum())
        }

    raise ValueError(f"Unknown mode: {mode}")


def _run_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    deep_rmsa,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    stats: Dict[str, Any],
    mode: str,
) -> None:
    for request_index, req in enumerate(requests):
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_action, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_action, r_idx, extra = _select_r_for_mode(
            mode, env, requests, request_index, req, obs_c, obs_r, split_id, server_id,
            agent_c, agent_r, rank_model, rank_mean, rank_std, deep_rmsa, args,
        )
        decision_ms = (time.perf_counter() - started) * 1000.0
        _, _, _, info = env.step((split_id, server_id), r_action)
        same_as_ppo = True
        if r_idx is not None:
            ppo_idx = agent_r.select_action(obs_r, deterministic=True)
            same_as_ppo = int(r_idx) == int(ppo_idx if ppo_idx is not None else 0)
        _record_outcome(
            metrics,
            info,
            int(np.asarray(raw_c_mask, dtype=bool).sum()) == 0,
            obs_c,
            split_id,
            server_id,
            decision_ms,
            same_as_ppo,
        )
        metrics.active_connections[-1] = len(env.active_connections)

        stats["legal_counts"].append(int(extra.get("legal_count", 0)))
        stats["rollout_used"] += int(mode == "rollout_topk" and int(extra.get("topk_count", 0)) > 1)
        stats["changed_vs_rank"] += int(bool(extra.get("changed_vs_rank", False)))


def _aggregate_extra(metrics: PerMethodMetrics, stats: Dict[str, Any]) -> Dict[str, Any]:
    agg = _aggregate_metrics(metrics)
    n = max(metrics.total, 1)
    agg.update({
        "avg_legal_r_actions": float(np.mean(stats["legal_counts"])) if stats["legal_counts"] else 0.0,
        "rollout_used_rate": stats["rollout_used"] / n,
        "changed_vs_rank_rate": stats["changed_vs_rank"] / n,
    })
    return agg


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)
    deep_rmsa = None
    if "deep_rmsa" in methods:
        deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

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

    report: Dict[str, Any] = {"config": vars(args), "methods": {}}
    for mode in methods:
        spec = {"per_seed": {}}
        for seed in seeds:
            metrics = PerMethodMetrics()
            stats = {"legal_counts": [], "rollout_used": 0, "changed_vs_rank": 0}
            for requests in episodes_by_seed[seed]:
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks,
                    block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                env.reset(requests)
                _run_episode(
                    env, requests, agent_c, agent_r,
                    rank_model, rank_mean, rank_std, deep_rmsa,
                    args, metrics, stats, mode,
                )
            spec["per_seed"][str(seed)] = _aggregate_extra(metrics, stats)
            print(f"[{mode}] completed seed {seed}", flush=True)
        first = next(iter(spec["per_seed"].values()))
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key, value in first.items()
            if isinstance(value, (int, float))
        }
        report["methods"][mode] = spec
    return report


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Inference-Time Top-K Rollout Reranking",
        "",
        "| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 | Avg legal R | Rollout used | Changed vs rank |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in report["methods"].items():
        a = spec["aggregate"]
        lines.append(
            f"| {name} | {a['blocking_rate']:.2%} | {a['raw_mask_empty_rate']:.2%} | "
            f"{a['no_suitable_block_rate']:.2%} | {a['server_overload_rate']:.2%} | "
            f"{a['mean_delay_ms']:.3f}/{a['p95_delay_ms']:.3f} | "
            f"{a['mean_decision_time_ms']:.3f}/{a['p95_decision_time_ms']:.3f} | "
            f"{a['avg_legal_r_actions']:.2f} | {a['rollout_used_rate']:.2%} | "
            f"{a['changed_vs_rank_rate']:.2%} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_s80_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="rank_only,rollout_topk,deep_rmsa")
    parser.add_argument("--rollout_top_k", type=int, default=3)
    parser.add_argument("--rollout_horizon", type=int, default=2)
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=80)
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
    parser.add_argument("--size_max_mb", type=float, default=40.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_inference_rollout_s80.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_inference_rollout_s80.md")
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
