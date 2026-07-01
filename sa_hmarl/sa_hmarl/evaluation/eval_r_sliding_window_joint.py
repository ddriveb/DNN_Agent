"""Sliding-window joint RMSA oracle/diagnostic evaluation.

This script tests whether moving beyond per-request greedy R decisions can
reduce blocking.  It buffers a small window of requests, then performs a
limited Top-K tree search with the frozen PPO-C and v1.2 R-ranker:

    request window -> recursively enumerate v1.2 Top-K R actions -> choose the
    sequence with the fewest in-window failures.

No models are trained or modified.  This is a mechanism-level oracle probe.
Queueing delay from waiting for the window to fill is added to reported delay,
but this first diagnostic does not reject a request again if queueing makes it
exceed its deadline.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass
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
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.generate_r_post_decision_dataset import _r_feature_vector
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


@dataclass
class PlannedDecision:
    split_id: int
    server_id: int
    r_action: Tuple[int, int, int]
    r_idx: int
    raw_empty: bool
    obs_c: Dict[str, Any]
    rank_score: float


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


def _select_rank_only(
    env,
    req,
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
) -> PlannedDecision:
    obs_c = build_agent_c_observation(env, req)
    c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
    split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    legal, scores = _score_legal_ranker(
        env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
        split_id, server_id, args.device,
    )
    if not legal:
        r_idx = 0
        score = 0.0
    else:
        best = int(np.argmax(scores))
        r_idx = int(legal[best])
        score = float(scores[best])
    return PlannedDecision(
        split_id=split_id,
        server_id=server_id,
        r_action=decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks),
        r_idx=r_idx,
        raw_empty=bool(int(np.asarray(raw_c_mask, dtype=bool).sum()) == 0),
        obs_c=obs_c,
        rank_score=score,
    )


def _topk_rank_actions(
    env,
    req,
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
) -> Tuple[Dict[str, Any], bool, int, int, List[Tuple[int, Tuple[int, int, int], float]]]:
    obs_c = build_agent_c_observation(env, req)
    c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
    raw_empty = bool(int(np.asarray(raw_c_mask, dtype=bool).sum()) == 0)
    split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    legal, scores = _score_legal_ranker(
        env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
        split_id, server_id, args.device,
    )
    if not legal:
        return obs_c, raw_empty, split_id, server_id, [(0, (0, 0, 0), 0.0)]
    order = np.argsort(-scores, kind="stable")[: min(args.top_k, len(legal))]
    actions = []
    for index in order:
        r_idx = int(legal[int(index)])
        actions.append((
            r_idx,
            decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks),
            float(scores[int(index)]),
        ))
    return obs_c, raw_empty, split_id, server_id, actions


def _simulate_one(env, req, split_id: int, server_id: int, r_action: Tuple[int, int, int]) -> Dict[str, Any]:
    _, _, _, info = env._handle_arrival(req, (split_id, server_id), r_action)
    return dict(info)


def _window_objective(infos: List[Dict[str, Any]], scores: List[float]) -> Tuple:
    blocked = sum(1 for info in infos if not info.get("success", False))
    no_suitable = sum(1 for info in infos if info.get("reason") == "no_suitable_block")
    overload = sum(1 for info in infos if info.get("reason") in ("server_overload", "server_saturated"))
    delay_sum = sum(float(info.get("delay_ms", 0.0)) for info in infos if info.get("success", False))
    # Minimize failures, then delay; maximize rank score as the final tie-breaker.
    return (blocked, no_suitable, overload, delay_sum, -sum(scores))


def _search_window(
    env,
    window: List[Any],
    depth: int,
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
    decisions: List[PlannedDecision],
    infos: List[Dict[str, Any]],
) -> Tuple[Tuple, List[PlannedDecision], List[Dict[str, Any]], int]:
    if depth >= len(window):
        return _window_objective(infos, [d.rank_score for d in decisions]), list(decisions), list(infos), 1

    req = window[depth]
    obs_c, raw_empty, split_id, server_id, actions = _topk_rank_actions(
        env, req, agent_c, agent_r, rank_model, rank_mean, rank_std, args
    )

    best_obj: Optional[Tuple] = None
    best_decisions: List[PlannedDecision] = []
    best_infos: List[Dict[str, Any]] = []
    branches = 0
    for r_idx, r_action, score in actions:
        branch = copy.deepcopy(env)
        info = _simulate_one(branch, req, split_id, server_id, r_action)
        decision = PlannedDecision(
            split_id=split_id,
            server_id=server_id,
            r_action=r_action,
            r_idx=r_idx,
            raw_empty=raw_empty,
            obs_c=obs_c,
            rank_score=score,
        )
        obj, seq, seq_infos, n_branch = _search_window(
            branch, window, depth + 1, agent_c, agent_r,
            rank_model, rank_mean, rank_std, args,
            decisions + [decision], infos + [info],
        )
        branches += n_branch
        if best_obj is None or obj < best_obj:
            best_obj = obj
            best_decisions = seq
            best_infos = seq_infos
    return best_obj or (0, 0, 0, 0.0, 0.0), best_decisions, best_infos, branches


def _run_rank_only_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    stats: Dict[str, Any],
) -> None:
    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        decision = _select_rank_only(
            env, req, agent_c, agent_r, rank_model, rank_mean, rank_std, args
        )
        decision_ms = (time.perf_counter() - started) * 1000.0
        info = _simulate_one(env, req, decision.split_id, decision.server_id, decision.r_action)
        _record_outcome(
            metrics, info, decision.raw_empty, decision.obs_c,
            decision.split_id, decision.server_id, decision_ms, True,
        )
        metrics.active_connections[-1] = len(env.active_connections)
        stats["windows"] += 1
        stats["branches"].append(1)
        stats["wait_ms"].append(0.0)


def _run_deep_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    deep_rmsa,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    stats: Dict[str, Any],
) -> None:
    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_idx = deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
        r_action = (0, 0, 0) if r_idx is None else decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        decision_ms = (time.perf_counter() - started) * 1000.0
        info = _simulate_one(env, req, split_id, server_id, r_action)
        _record_outcome(
            metrics, info, bool(int(np.asarray(raw_c_mask, dtype=bool).sum()) == 0),
            obs_c, split_id, server_id, decision_ms, True,
        )
        metrics.active_connections[-1] = len(env.active_connections)
        stats["windows"] += 1
        stats["branches"].append(1)
        stats["wait_ms"].append(0.0)


def _run_window_joint_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_model,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    stats: Dict[str, Any],
) -> None:
    index = 0
    while index < len(requests):
        window = requests[index : index + args.window_size]
        batch_time = max(float(req.arrival_time) for req in window)
        env.advance_time(batch_time)
        started = time.perf_counter()
        _, decisions, _, branches = _search_window(
            copy.deepcopy(env), window, 0,
            agent_c, agent_r, rank_model, rank_mean, rank_std, args,
            [], [],
        )
        decision_ms = (time.perf_counter() - started) * 1000.0
        stats["windows"] += 1
        stats["branches"].append(branches)
        stats["window_sizes"].append(len(window))

        for req, decision in zip(window, decisions):
            wait_ms = max(0.0, (batch_time - float(req.arrival_time)) * 1000.0)
            info = _simulate_one(env, req, decision.split_id, decision.server_id, decision.r_action)
            if info.get("success", False):
                info["delay_ms"] = float(info.get("delay_ms", 0.0)) + wait_ms
            _record_outcome(
                metrics, info, decision.raw_empty, decision.obs_c,
                decision.split_id, decision.server_id, decision_ms / max(len(window), 1),
                True,
            )
            metrics.active_connections[-1] = len(env.active_connections)
            stats["wait_ms"].append(wait_ms)
        index += args.window_size


def _aggregate_extra(metrics: PerMethodMetrics, stats: Dict[str, Any]) -> Dict[str, Any]:
    agg = _aggregate_metrics(metrics)
    waits = stats["wait_ms"]
    branches = stats["branches"]
    agg.update({
        "windows": float(stats["windows"]),
        "mean_window_size": float(np.mean(stats["window_sizes"])) if stats["window_sizes"] else 1.0,
        "mean_branches": float(np.mean(branches)) if branches else 1.0,
        "p95_branches": float(np.percentile(branches, 95)) if branches else 1.0,
        "mean_wait_ms": float(np.mean(waits)) if waits else 0.0,
        "p95_wait_ms": float(np.percentile(waits, 95)) if waits else 0.0,
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
    for method in methods:
        spec = {"per_seed": {}}
        for seed in seeds:
            metrics = PerMethodMetrics()
            stats = {"windows": 0, "branches": [], "wait_ms": [], "window_sizes": []}
            for requests in episodes_by_seed[seed]:
                env = make_env(
                    args.topology, args.num_slots, args.num_servers, seed,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks,
                    block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                )
                env.reset(requests)
                if method == "rank_only":
                    _run_rank_only_episode(
                        env, requests, agent_c, agent_r,
                        rank_model, rank_mean, rank_std, args, metrics, stats,
                    )
                elif method == "window_joint":
                    _run_window_joint_episode(
                        env, requests, agent_c, agent_r,
                        rank_model, rank_mean, rank_std, args, metrics, stats,
                    )
                elif method == "deep_rmsa":
                    _run_deep_episode(env, requests, agent_c, agent_r, deep_rmsa, args, metrics, stats)
                else:
                    raise ValueError(f"Unknown method: {method}")
            spec["per_seed"][str(seed)] = _aggregate_extra(metrics, stats)
            print(f"[{method}] completed seed {seed}", flush=True)
        first = next(iter(spec["per_seed"].values()))
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key, value in first.items()
            if isinstance(value, (int, float))
        }
        report["methods"][method] = spec
    return report


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Sliding-Window Joint RMSA Oracle Diagnostic",
        "",
        "| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Wait mean/P95 | Decision mean/P95 | Branches mean/P95 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for name, spec in report["methods"].items():
        a = spec["aggregate"]
        lines.append(
            f"| {name} | {a['blocking_rate']:.2%} | {a['raw_mask_empty_rate']:.2%} | "
            f"{a['no_suitable_block_rate']:.2%} | {a['server_overload_rate']:.2%} | "
            f"{a['mean_delay_ms']:.3f}/{a['p95_delay_ms']:.3f} | "
            f"{a['mean_wait_ms']:.3f}/{a['p95_wait_ms']:.3f} | "
            f"{a['mean_decision_time_ms']:.3f}/{a['p95_decision_time_ms']:.3f} | "
            f"{a['mean_branches']:.1f}/{a['p95_branches']:.1f} |"
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_s80_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--methods", default="rank_only,window_joint,deep_rmsa")
    parser.add_argument("--window_size", type=int, default=2)
    parser.add_argument("--top_k", type=int, default=2)
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_sliding_window_joint_s80.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_sliding_window_joint_s80.md")
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
