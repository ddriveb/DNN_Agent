"""Standalone evaluation of the resource-regularized v1 R-ranker (best config).

This is the current strongest SA-HMARL closed-loop result. It runs the v1
counterfactual R-ranker with λ_path=0.10, λ_fs=0.05 and reports blocking,
delay, NSB, overload, path length, and FS usage.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
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
from sa_hmarl.evaluation.run_r_ranker_resource_penalty_sweep import (
    _run_episode,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def evaluate(args):
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

    metrics = PerMethodMetrics()
    per_seed = {}

    for seed in seeds:
        rng = __import__("numpy").random.RandomState(seed)
        seed_metrics = PerMethodMetrics()
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            requests = generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                args.arrival_interval, args.holding_min, args.holding_max,
                args.deadline_min, args.deadline_max, args.size_min_mb,
                args.size_max_mb, args.edge_cost_min, args.edge_cost_max,
                args.num_splits, args.split_profile,
            )
            env = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            _run_episode(
                env, requests, agent_c, agent_r, rank_model, rank_mean, rank_std,
                args, seed_metrics, args.lambda_path, args.lambda_fs,
            )
        per_seed[str(seed)] = _aggregate_metrics(seed_metrics)
        # Accumulate into global metrics.
        for field_name in ("total", "blocked", "raw_empty", "no_suitable_block",
                           "server_overload", "deadline_failure"):
            setattr(metrics, field_name, getattr(metrics, field_name) + getattr(seed_metrics, field_name))
        metrics.delays.extend(seed_metrics.delays)
        metrics.fses.extend(seed_metrics.fses)
        metrics.wastes.extend(seed_metrics.wastes)
        metrics.path_kms.extend(seed_metrics.path_kms)
        metrics.active_connections.extend(seed_metrics.active_connections)
        metrics.valid_c_actions.extend(seed_metrics.valid_c_actions)
        metrics.total_valid_r_actions.extend(seed_metrics.total_valid_r_actions)
        metrics.selected_valid_r_actions.extend(seed_metrics.selected_valid_r_actions)
        metrics.decision_times_ms.extend(seed_metrics.decision_times_ms)
        metrics.actions_same.extend(seed_metrics.actions_same)
        metrics.changed_blocked_count += seed_metrics.changed_blocked_count

    aggregate = _aggregate_metrics(metrics)
    report = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "lambda_path": args.lambda_path,
        "lambda_fs": args.lambda_fs,
        "aggregate": aggregate,
        "per_seed": per_seed,
        "elapsed_seconds": time.time() - args._start_time,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, indent=2), encoding="utf-8")

    lines = [
        "# Resource-Regularized v1 R-Ranker Evaluation",
        "",
        f"λ_path = {args.lambda_path}, λ_fs = {args.lambda_fs}",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Blocking | {aggregate['blocking_rate']:.2%} |",
        f"| Raw empty | {aggregate['raw_mask_empty_rate']:.2%} |",
        f"| NSB | {aggregate['no_suitable_block_rate']:.2%} |",
        f"| Overload | {aggregate['server_overload_rate']:.2%} |",
        f"| Delay mean/P95 | {aggregate['mean_delay_ms']:.2f}/{aggregate['p95_delay_ms']:.2f} ms |",
        f"| Avg path km | {aggregate['avg_path_km']:.1f} |",
        f"| Avg FS | {aggregate['avg_fs']:.2f} |",
        "",
        "## Per seed",
        "",
        "| Seed | Blocking | Avg path km | Avg FS |",
        "|---|---:|---:|---:|",
    ]
    for seed, agg in per_seed.items():
        lines.append(
            f"| {seed} | {agg['blocking_rate']:.2%} | {agg['avg_path_km']:.1f} | {agg['avg_fs']:.2f} |"
        )
    Path(args.output_md).write_text("\n".join(lines), encoding="utf-8")
    return report


def build_parser():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_k5m10_h5/ranking_model.pt")
    parser.add_argument("--lambda_path", type=float, default=0.10)
    parser.add_argument("--lambda_fs", type=float, default=0.05)
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/r_ranker_resource_regularized_eval.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/r_ranker_resource_regularized_eval.md")
    return parser


if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    args._start_time = time.time()
    report = evaluate(args)
    print(f"Saved {args.output_json}")
    print(f"Saved {args.output_md}")
    print(f"Blocking: {report['aggregate']['blocking_rate']:.2%}")
