"""Evaluate C-mask relaxation as a mechanism-level blocking reducer.

This script does not train or modify checkpoints.  It rebuilds Agent-C masks
with relaxed server-utilization thresholds while still requiring downstream R
feasibility (`feasible_count > 0`).  The goal is to test whether the original
0.95 utilization cutoff is leaving useful C candidates on the table.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.env.action_mask import build_agent_c_mask
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
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
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _get_c_logits(agent_c, features: np.ndarray) -> np.ndarray:
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_c.device).unsqueeze(0)
        return agent_c.policy_net(x).squeeze(0).cpu().numpy()


def _relaxed_c_mask(obs_c: Dict[str, Any], util_threshold: float) -> np.ndarray:
    return build_agent_c_mask(
        num_splits=len(obs_c["feasible_counts"]),
        num_servers=len(obs_c["server_utilizations"]),
        server_utilizations=list(obs_c["server_utilizations"]),
        util_threshold=util_threshold,
        feasible_counts=obs_c["feasible_counts"],
    )


def _select_c_action(
    agent_c,
    obs_c: Dict[str, Any],
    policy: str,
    util_threshold: float,
) -> Tuple[int, np.ndarray, Dict[str, Any]]:
    features, strict_mask = agent_c.build_action_features(obs_c)
    strict_mask = np.asarray(strict_mask, dtype=bool)
    mask = strict_mask.copy() if policy == "strict" else _relaxed_c_mask(obs_c, util_threshold)

    stats = {
        "strict_valid": int(strict_mask.sum()),
        "relaxed_valid": int(mask.sum()),
        "relaxed_added": int(max(mask.sum() - strict_mask.sum(), 0)),
        "selected_was_strict_valid": False,
    }

    if not np.any(mask):
        return 0, mask, stats

    if policy in ("strict", "relax_policy"):
        action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
        action = 0 if action is None else int(action)
    elif policy == "relax_max_r":
        logits = _get_c_logits(agent_c, features)
        num_servers = len(obs_c["server_utilizations"])
        candidates = np.flatnonzero(mask).tolist()

        def key(idx: int):
            split_id = idx // num_servers
            server_id = idx % num_servers
            feasible = int(obs_c["feasible_counts"][split_id][server_id])
            # Maximize downstream R action count, then PPO-C logit.
            return (feasible, float(logits[idx]), -idx)

        action = int(max(candidates, key=key))
    else:
        raise ValueError(f"Unknown C policy: {policy}")

    stats["selected_was_strict_valid"] = bool(strict_mask[action])
    return action, mask, stats


def _select_r_action(
    mode: str,
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    split_id: int,
    server_id: int,
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    deep_rmsa,
    args: argparse.Namespace,
) -> Tuple[Tuple[int, int, int], Optional[int]]:
    if mode == "counterfactual_rank_only":
        idx = _select_rank_only_r_action(
            env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
            split_id, server_id, args.device,
        )
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks), int(idx)

    if mode == "deep_rmsa":
        idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
        if idx is None:
            return (0, 0, 0), None
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks), int(idx)

    if mode == "ppo_r":
        idx = agent_r.select_action(obs_r, deterministic=True)
        if idx is None:
            return (0, 0, 0), None
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks), int(idx)

    raise ValueError(f"Unknown R mode: {mode}")


def _run_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    deep_rmsa,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    stats: Dict[str, Any],
    c_policy: str,
    r_mode: str,
    util_threshold: float,
) -> None:
    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_action, c_mask, c_stats = _select_c_action(agent_c, obs_c, c_policy, util_threshold)
        split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_action, r_idx = _select_r_action(
            r_mode, env, req, obs_c, obs_r, split_id, server_id,
            agent_r, rank_model, rank_mean, rank_std, deep_rmsa, args,
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
            int(np.asarray(c_mask, dtype=bool).sum()) == 0,
            obs_c,
            split_id,
            server_id,
            decision_ms,
            same_as_ppo,
        )
        metrics.active_connections[-1] = len(env.active_connections)

        stats["strict_empty"] += int(c_stats["strict_valid"] == 0)
        stats["relaxed_empty"] += int(c_stats["relaxed_valid"] == 0)
        stats["relaxed_added_total"] += int(c_stats["relaxed_added"])
        stats["relaxed_added_request"] += int(c_stats["relaxed_added"] > 0)
        stats["selected_relaxed_only"] += int(
            c_stats["relaxed_valid"] > 0 and not c_stats["selected_was_strict_valid"]
        )


def _aggregate_extra(metrics: PerMethodMetrics, stats: Dict[str, Any]) -> Dict[str, Any]:
    agg = _aggregate_metrics(metrics)
    n = max(metrics.total, 1)
    agg.update({
        "strict_mask_empty_rate": stats["strict_empty"] / n,
        "relaxed_mask_empty_rate": stats["relaxed_empty"] / n,
        "relaxed_added_request_rate": stats["relaxed_added_request"] / n,
        "avg_relaxed_added_actions": stats["relaxed_added_total"] / n,
        "selected_relaxed_only_rate": stats["selected_relaxed_only"] / n,
    })
    return agg


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    thresholds = [float(x.strip()) for x in args.relaxed_util_thresholds.split(",") if x.strip()]
    r_modes = [x.strip() for x in args.r_modes.split(",") if x.strip()]
    c_policies = [x.strip() for x in args.c_policies.split(",") if x.strip()]

    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(
        args.agent_r_checkpoint,
        ModulationRegistry.from_profile(args.modulation_profile),
        args.device,
    )
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)

    deep_rmsa = None
    if "deep_rmsa" in r_modes:
        deep_rmsa = _load_deep_rmsa(
            args.deep_rmsa_checkpoint,
            env_proto,
            ModulationRegistry.from_profile(args.modulation_profile),
            args.device,
        )

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

    methods: Dict[str, Dict[str, Any]] = {}
    for r_mode in r_modes:
        for c_policy in c_policies:
            active_thresholds = [0.95] if c_policy == "strict" else thresholds
            for threshold in active_thresholds:
                method_name = f"{r_mode}__{c_policy}"
                if c_policy != "strict":
                    method_name += f"_u{threshold:g}"
                methods[method_name] = {
                    "r_mode": r_mode,
                    "c_policy": c_policy,
                    "util_threshold": threshold,
                    "per_seed": {},
                }

    for method_name, spec in methods.items():
        for seed in seeds:
            metrics = PerMethodMetrics()
            stats = {
                "strict_empty": 0,
                "relaxed_empty": 0,
                "relaxed_added_total": 0,
                "relaxed_added_request": 0,
                "selected_relaxed_only": 0,
            }
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
                    rank_model, rank_mean, rank_std, deep_rmsa,
                    args, metrics, stats,
                    spec["c_policy"], spec["r_mode"], spec["util_threshold"],
                )
            spec["per_seed"][str(seed)] = _aggregate_extra(metrics, stats)
            print(f"[{method_name}] completed seed {seed}", flush=True)
        first = next(iter(spec["per_seed"].values()))
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key, value in first.items()
            if isinstance(value, (int, float))
        }

    return {"config": vars(args), "methods": methods}


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# C-Mask Relaxation Evaluation",
        "",
        "| Method | Blocking | Raw empty | Strict empty | Relaxed empty | Relax added req | Selected relaxed-only | NSB | Overload | Delay mean/P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in report["methods"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {name} | {agg['blocking_rate']:.2%} | {agg['raw_mask_empty_rate']:.2%} | "
            f"{agg['strict_mask_empty_rate']:.2%} | {agg['relaxed_mask_empty_rate']:.2%} | "
            f"{agg['relaxed_added_request_rate']:.2%} | {agg['selected_relaxed_only_rate']:.2%} | "
            f"{agg['no_suitable_block_rate']:.2%} | {agg['server_overload_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_s80_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--r_modes", default="counterfactual_rank_only,deep_rmsa")
    parser.add_argument("--c_policies", default="strict,relax_policy,relax_max_r")
    parser.add_argument("--relaxed_util_thresholds", default="1.0,1.05")
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/c_mask_relaxation_eval.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/c_mask_relaxation_eval.md")
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
