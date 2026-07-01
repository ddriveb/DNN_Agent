"""Lightweight inference-time resource-penalty sweep for the v1 R-ranker.

For each (λ_path, λ_fs) configuration we run PPO-C + penalized v1 R-ranker on
the standard closed-loop protocol and measure blocking, delay, NSB, overload,
and the resulting path length / FS usage. No dataset regeneration and no
retraining is performed.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch

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
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


# Feature indices inside the base R-feature vector.
_PATH_KM_IDX = FEATURE_NAMES.index("path_length_km")
_REQUIRED_FS_IDX = FEATURE_NAMES.index("required_fs")


def _select_rank_only_r_action_with_penalty(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    model: torch.nn.Module,
    mean: np.ndarray,
    std: np.ndarray,
    split_id: int,
    server_id: int,
    device: str,
    lambda_path: float,
    lambda_fs: float,
) -> int:
    """Score legal R actions with the v1 ranker and subtract resource penalties."""
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
        scores = model(
            torch.as_tensor(normalized, dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()

    # Soft resource penalty: penalize values above the legal-action mean.
    path_km = np.array([float(r_features[int(a)][_PATH_KM_IDX]) for a in legal])
    required_fs = np.array([float(r_features[int(a)][_REQUIRED_FS_IDX]) for a in legal])
    path_std = path_km.std()
    fs_std = required_fs.std()
    path_km_norm = (path_km - path_km.mean()) / (path_std + 1e-6)
    fs_norm = (required_fs - required_fs.mean()) / (fs_std + 1e-6)

    adjusted = scores - lambda_path * path_km_norm - lambda_fs * fs_norm
    return int(legal[int(np.argmax(adjusted))])


def _run_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    rank_model: torch.nn.Module,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    args: argparse.Namespace,
    metrics: PerMethodMetrics,
    lambda_path: float,
    lambda_fs: float,
) -> None:
    """Run one episode with the penalized ranker."""
    for req in requests:
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)

        r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if r_mask.any():
            r_idx = _select_rank_only_r_action_with_penalty(
                env, req, obs_c, obs_r, agent_r, rank_model, rank_mean, rank_std,
                split_id, server_id, args.device, lambda_path, lambda_fs,
            )
            r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        else:
            r_idx = -1
            r_action = (0, 0, 0)

        _, _, _, info = env.step((split_id, server_id), r_action)
        _record_outcome(
            metrics, info, int(raw_c_mask.sum()) == 0, obs_c,
            split_id, server_id, 0.0, True,
        )
        metrics.active_connections[-1] = len(env.active_connections)


def _eval_config(
    args: argparse.Namespace,
    episodes_by_seed: Dict[int, List[List[Any]]],
    agent_c,
    agent_r,
    rank_model: torch.nn.Module,
    rank_mean: np.ndarray,
    rank_std: np.ndarray,
    lambda_path: float,
    lambda_fs: float,
) -> Dict[str, Any]:
    """Evaluate one penalty configuration across all seeds/episodes."""
    per_seed = {}
    metrics_all = PerMethodMetrics()
    for seed, episodes in episodes_by_seed.items():
        metrics_seed = PerMethodMetrics()
        for requests in episodes:
            env = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            _run_episode(
                env, requests, agent_c, agent_r, rank_model, rank_mean, rank_std,
                args, metrics_seed, lambda_path, lambda_fs,
            )
        per_seed[str(seed)] = _aggregate_metrics(metrics_seed)
        # Accumulate into global metrics for aggregate stats.
        for field_name in ("total", "blocked", "raw_empty", "no_suitable_block",
                           "server_overload", "deadline_failure"):
            setattr(
                metrics_all,
                field_name,
                getattr(metrics_all, field_name) + getattr(metrics_seed, field_name),
            )
        metrics_all.delays.extend(metrics_seed.delays)
        metrics_all.fses.extend(metrics_seed.fses)
        metrics_all.wastes.extend(metrics_seed.wastes)
        metrics_all.path_kms.extend(metrics_seed.path_kms)
        metrics_all.active_connections.extend(metrics_seed.active_connections)
        metrics_all.valid_c_actions.extend(metrics_seed.valid_c_actions)
        metrics_all.total_valid_r_actions.extend(metrics_seed.total_valid_r_actions)
        metrics_all.selected_valid_r_actions.extend(metrics_seed.selected_valid_r_actions)
        metrics_all.decision_times_ms.extend(metrics_seed.decision_times_ms)
        metrics_all.actions_same.extend(metrics_seed.actions_same)
        metrics_all.changed_blocked_count += metrics_seed.changed_blocked_count

    aggregate = _aggregate_metrics(metrics_all)
    return {
        "lambda_path": lambda_path,
        "lambda_fs": lambda_fs,
        "aggregate": aggregate,
        "per_seed": per_seed,
    }


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

    # Generate shared request traces once.
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

    configs = []
    for cfg in args.configs.split(";"):
        cfg = cfg.strip()
        if not cfg:
            continue
        name, lam = cfg.split(":", 1)
        lp, lf = lam.split(",")
        configs.append((name.strip(), float(lp), float(lf)))

    results = []
    for name, lp, lf in configs:
        t0 = time.time()
        result = _eval_config(
            args, episodes_by_seed, agent_c, agent_r,
            rank_model, rank_mean, rank_std, lp, lf,
        )
        result["name"] = name
        result["elapsed_seconds"] = time.time() - t0
        results.append(result)
        print(f"[{name}] λ_path={lp} λ_fs={lf} blocking={result['aggregate']['blocking_rate']:.4f}", flush=True)

    # Determine verdict.
    baseline = next((r for r in results if r["lambda_path"] == 0 and r["lambda_fs"] == 0), None)
    baseline_blocking = baseline["aggregate"]["blocking_rate"] if baseline else 1.0

    best = min(results, key=lambda r: r["aggregate"]["blocking_rate"])
    best_blocking = best["aggregate"]["blocking_rate"]

    verdict = "FAIL"
    verdict_text = "No penalty configuration improved over baseline."
    if best_blocking <= 0.328:
        verdict = "PASS"
        verdict_text = (
            f"PASS: config '{best['name']}' reached blocking {best_blocking:.2%} "
            f"(≤ 32.8%). Proceed to training v1.2 with this penalty baked into the objective."
        )
    elif best_blocking <= 0.332:
        verdict = "MARGINAL"
        verdict_text = (
            f"MARGINAL: best config '{best['name']}' blocking {best_blocking:.2%} "
            f"with reduced path/FS usage. Consider a small v1.2 experiment."
        )
    elif best_blocking < baseline_blocking - 0.001:
        verdict = "MARGINAL"
        verdict_text = (
            f"MARGINAL: best config '{best['name']}' improved blocking to {best_blocking:.2%} "
            f"vs baseline {baseline_blocking:.2%}, but not below 33.2%."
        )

    output = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "baseline_blocking": baseline_blocking,
        "best_config": {"name": best["name"], "lambda_path": best["lambda_path"], "lambda_fs": best["lambda_fs"]},
        "best_blocking": best_blocking,
        "results": results,
        "verdict": verdict,
        "verdict_text": verdict_text,
        "elapsed_seconds": time.time() - args._start_time,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(output, indent=2), encoding="utf-8")
    _write_markdown(args.output_md, output)
    return output


def _write_markdown(path: str, report: Dict[str, Any]) -> None:
    lines = [
        "# R-Ranker Resource-Penalty Inference Sweep",
        "",
        "## Setup",
        "",
        f"- Seeds: `{report['config']['seeds']}`",
        f"- Episodes/seed: {report['config']['episodes']}",
        f"- Requests/episode: {report['config']['requests_per_episode']}",
        f"- Topology: `{report['config']['topology']}`",
        "",
        "## Results",
        "",
        "| Config | λ_path | λ_fs | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Avg path km | Avg FS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in report["results"]:
        agg = r["aggregate"]
        lines.append(
            f"| {r['name']} | {r['lambda_path']:.2f} | {r['lambda_fs']:.2f} | "
            f"{agg['blocking_rate']:.2%} | {agg['raw_mask_empty_rate']:.2%} | "
            f"{agg['no_suitable_block_rate']:.2%} | {agg['server_overload_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.2f}/{agg['p95_delay_ms']:.2f} ms | "
            f"{agg['avg_path_km']:.1f} | {agg['avg_fs']:.2f} |"
        )

    lines += [
        "",
        "## Verdict",
        "",
        report["verdict_text"],
        "",
    ]
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_k5m10_h5/ranking_model.pt")
    parser.add_argument("--configs", default=(
        "baseline:0,0;"
        "path_only_low:0.05,0;"
        "path_only_mid:0.10,0;"
        "fs_only_low:0,0.05;"
        "fs_only_mid:0,0.10;"
        "mixed_low:0.05,0.05;"
        "mixed_mid:0.10,0.05"
    ))
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_ranker_resource_penalty_sweep.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_ranker_resource_penalty_sweep.md")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    report = evaluate(args)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")
    print(report["verdict_text"])
