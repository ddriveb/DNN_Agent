"""Closed-loop evaluation with deadline-aware waiting before R scheduling.

This is an evaluation-only mechanism.  It does not modify training,
checkpoints, or the environment.  When the current request has no raw C action
or the selected C action has no legal R action, the evaluator waits until the
next resource-release event within a bounded budget and retries the same
request.
"""
from __future__ import annotations

import argparse
import heapq
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
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _next_release_time(env) -> Optional[float]:
    future = [
        float(conn["release_time"])
        for conn in env.active_connections
        if float(conn["release_time"]) > float(env.time) + 1e-12
    ]
    return min(future) if future else None


def _r_mask_empty(agent_r, obs_r: Dict[str, Any]) -> bool:
    _, mask = agent_r.build_action_features(obs_r)
    return int(np.asarray(mask, dtype=bool).sum()) == 0


def _select_r_for_mode(
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
    if mode == "ppo_r":
        action_idx = agent_r.select_action(obs_r, deterministic=True)
        if action_idx is None:
            return (0, 0, 0), None
        return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), env.max_blocks), int(action_idx)

    if mode == "counterfactual_rank_only":
        action_idx = _select_rank_only_r_action(
            env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
            split_id, server_id, args.device,
        )
        return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), env.max_blocks), int(action_idx)

    if mode == "deep_rmsa":
        action_idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
        if action_idx is None:
            return (0, 0, 0), None
        return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), env.max_blocks), int(action_idx)

    if mode == "ksp_bf":
        action_idx = ksp_bf_action(obs_r)
        if action_idx is None:
            return (0, 0, 0), None
        return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), env.max_blocks), int(action_idx)

    raise ValueError(f"Unknown mode: {mode}")


def _build_decision(
    mode: str,
    env,
    req,
    agent_c,
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    deep_rmsa,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    obs_c = build_agent_c_observation(env, req)
    c_action, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
    raw_empty = int(np.asarray(raw_c_mask, dtype=bool).sum()) == 0
    split_id, server_id = decode_agent_c_action(c_action, args.num_servers)
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    r_empty = _r_mask_empty(agent_r, obs_r)
    r_action, r_idx = _select_r_for_mode(
        mode, env, req, obs_c, obs_r, split_id, server_id,
        agent_r, rank_model, rank_mean, rank_std, deep_rmsa, args,
    )
    return {
        "obs_c": obs_c,
        "raw_c_mask": raw_c_mask,
        "raw_empty": raw_empty,
        "r_empty": r_empty,
        "c_action": c_action,
        "split_id": split_id,
        "server_id": server_id,
        "r_action": r_action,
        "r_idx": r_idx,
    }


def _run_episode_waiting(
    env,
    requests: List,
    agent_c,
    agent_r,
    rank_model,
    rank_mean,
    rank_std,
    deep_rmsa,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    wait_stats: Dict[str, Any],
    mode: str,
) -> None:
    max_wait_s = max(float(args.max_wait_ms), 0.0) / 1000.0

    for req in requests:
        started = time.perf_counter()

        # Remove this request from the env queue.  We run the same arrival
        # sequence as env.step(), but allow the current request to wait before
        # calling the internal arrival handler.
        if env.event_queue:
            _, _, queued_req = heapq.heappop(env.event_queue)
            if queued_req.req_id != req.req_id:
                raise RuntimeError(
                    f"Request order mismatch: queued={queued_req.req_id}, loop={req.req_id}"
                )

        start_time = max(float(env.time), float(req.arrival_time))
        env.advance_time(start_time)
        wait_until = start_time + max_wait_s
        wait_ms = 0.0
        wait_triggered = False
        retry_count = 0

        decision = _build_decision(
            mode, env, req, agent_c, agent_r, rank_model, rank_mean, rank_std,
            deep_rmsa, args,
        )
        initial_raw_empty = bool(decision["raw_empty"])
        initial_r_empty = bool(decision["r_empty"])

        while max_wait_s > 0.0 and (decision["raw_empty"] or decision["r_empty"]):
            next_release = _next_release_time(env)
            if next_release is None or next_release > wait_until + 1e-12:
                break
            wait_triggered = True
            retry_count += 1
            env.advance_time(next_release)
            wait_ms = max(0.0, (float(env.time) - start_time) * 1000.0)
            decision = _build_decision(
                mode, env, req, agent_c, agent_r, rank_model, rank_mean, rank_std,
                deep_rmsa, args,
            )

        decision_ms = (time.perf_counter() - started) * 1000.0
        _, _, _, info = env._handle_arrival(
            req,
            (decision["split_id"], decision["server_id"]),
            decision["r_action"],
        )

        info = dict(info)
        if info.get("success", False):
            # Report queueing delay in the delay metric for this diagnostic.
            info["delay_ms"] = float(info.get("delay_ms", 0.0)) + wait_ms
        info["wait_ms"] = wait_ms
        info["wait_triggered"] = wait_triggered
        info["initial_raw_empty"] = initial_raw_empty
        info["initial_r_empty"] = initial_r_empty
        info["retry_count"] = retry_count

        same_as_ppo = True
        if decision["r_idx"] is not None:
            ppo_idx = agent_r.select_action(
                build_agent_r_observation(env, req, decision["split_id"], decision["server_id"]),
                deterministic=True,
            )
            same_as_ppo = int(decision["r_idx"]) == int(ppo_idx if ppo_idx is not None else 0)

        _record_outcome(
            metrics,
            info,
            bool(decision["raw_empty"]),
            decision["obs_c"],
            decision["split_id"],
            decision["server_id"],
            decision_ms,
            same_as_ppo,
        )
        metrics.active_connections[-1] = len(env.active_connections)

        wait_stats["initial_raw_empty"] += int(initial_raw_empty)
        wait_stats["initial_r_empty"] += int(initial_r_empty)
        wait_stats["wait_triggered"] += int(wait_triggered)
        wait_stats["wait_success"] += int(wait_triggered and bool(info.get("success", False)))
        wait_stats["retry_count"].append(retry_count)
        wait_stats["wait_ms"].append(wait_ms)


def _aggregate_wait_stats(metrics: PerMethodMetrics, wait_stats: Dict[str, Any]) -> Dict[str, Any]:
    agg = _aggregate_metrics(metrics)
    n = max(metrics.total, 1)
    waits = wait_stats["wait_ms"]
    retries = wait_stats["retry_count"]
    agg.update({
        "initial_raw_empty_rate": wait_stats["initial_raw_empty"] / n,
        "initial_r_empty_rate": wait_stats["initial_r_empty"] / n,
        "wait_trigger_rate": wait_stats["wait_triggered"] / n,
        "wait_success_rate": wait_stats["wait_success"] / n,
        "wait_success_given_wait": (
            wait_stats["wait_success"] / wait_stats["wait_triggered"]
            if wait_stats["wait_triggered"] else 0.0
        ),
        "mean_wait_ms": float(np.mean(waits)) if waits else 0.0,
        "p95_wait_ms": float(np.percentile(waits, 95)) if waits else 0.0,
        "mean_retry_count": float(np.mean(retries)) if retries else 0.0,
    })
    return agg


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    requested = [m.strip() for m in args.methods.split(",") if m.strip()]
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
    if "deep_rmsa" in requested:
        deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

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

    methods: Dict[str, Any] = {m: {"per_seed": {}} for m in requested}
    for method in requested:
        for seed in seeds:
            metrics = PerMethodMetrics()
            wait_stats = {
                "initial_raw_empty": 0,
                "initial_r_empty": 0,
                "wait_triggered": 0,
                "wait_success": 0,
                "retry_count": [],
                "wait_ms": [],
            }
            for episode_requests in episodes_by_seed[seed]:
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks,
                    block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                env.reset(episode_requests)
                _run_episode_waiting(
                    env, episode_requests, agent_c, agent_r,
                    rank_model, rank_mean, rank_std, deep_rmsa,
                    args, metrics, wait_stats, method,
                )
            methods[method]["per_seed"][str(seed)] = _aggregate_wait_stats(metrics, wait_stats)
            print(f"[{method}] completed seed {seed}", flush=True)

        keys = list(next(iter(methods[method]["per_seed"].values())).keys())
        aggregate = {}
        for key in keys:
            vals = [v[key] for v in methods[method]["per_seed"].values() if isinstance(v.get(key), (int, float))]
            if vals:
                aggregate[key] = float(np.mean(vals))
                aggregate[f"{key}_std"] = float(np.std(vals))
        methods[method]["aggregate"] = aggregate

    return {
        "config": vars(args),
        "methods": methods,
    }


def _write_md(report: Dict[str, Any], path: str) -> None:
    lines = [
        "# R-Side Closed-Loop Evaluation with Waiting",
        "",
        f"- Max wait: `{report['config']['max_wait_ms']}` ms",
        "",
        "| Method | Blocking | Delay mean/P95 | Wait trigger | Wait success | Wait mean/P95 | Raw empty init/final | R empty init | NSB | Overload |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for method, payload in report["methods"].items():
        agg = payload["aggregate"]
        lines.append(
            f"| {method} | {agg['blocking_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{agg['wait_trigger_rate']:.2%} | {agg['wait_success_given_wait']:.2%} | "
            f"{agg['mean_wait_ms']:.3f}/{agg['p95_wait_ms']:.3f} ms | "
            f"{agg['initial_raw_empty_rate']:.2%}/{agg['raw_mask_empty_rate']:.2%} | "
            f"{agg['initial_r_empty_rate']:.2%} | "
            f"{agg['no_suitable_block_rate']:.2%} | {agg['server_overload_rate']:.2%} |"
        )
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="counterfactual_rank_only,deep_rmsa,ppo_r,ksp_bf")
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
    parser.add_argument("--max_wait_ms", type=float, default=0.0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_waiting_closed_loop.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_waiting_closed_loop.md")
    args = parser.parse_args()

    report = evaluate(args)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(report, args.output_md)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")


if __name__ == "__main__":
    main()
