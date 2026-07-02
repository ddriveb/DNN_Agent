"""Main S100 system-level comparison for the final paper/PPT.

This script moves the main comparison from the very high-blocking S24 stress
setting to a more realistic S100 setting.  It evaluates both:

1. R-side backends under the same learned PPO-C policy.
2. C-side heuristic baselines paired with the final v1.2 planner-distilled
   R-ranker.

Default methods:
    ppo_c+v12, ppo_c+deep_rmsa, ppo_c+ppo_r, ppo_c+ksp_bf, ppo_c+ksp_ff,
    ppo_c+ksp_ff_k50_hops, greedy_c+v12, df_c+v12, rf_c+v12, wo_c+v12,
    iwd_c+v12
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.baselines.rmsa_baselines import (
    ksp_bf_action,
    ksp_ff_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    PerMethodMetrics,
    _aggregate_metrics,
    _select_r_action,
)
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import _record_outcome
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _r_backend_env_config(r_mode: str, args: argparse.Namespace) -> tuple[int, str, str]:
    """Return the R-side path breadth/order for a backend.

    PPO-C is kept on the main experiment path configuration.  Tuned R-only
    heuristic baselines can override the R observation and execution path set.
    """
    if r_mode in ("ksp_ff_k50_hops", "v12_k50_hops"):
        return args.ksp_ff_k50_hops_k_paths, "hops", "start_asc"
    return args.k_paths, args.path_sort_strategy, args.block_sort_strategy


def _parse_methods(spec: str) -> List[tuple[str, str, str]]:
    """Parse method specs like ``ppo_c+v12`` into (name, c_mode, r_mode)."""
    methods = []
    for item in spec.split(","):
        name = item.strip()
        if not name:
            continue
        if "+" not in name:
            raise ValueError(f"Method must look like c_mode+r_mode: {name}")
        c_mode, r_mode = name.split("+", 1)
        methods.append((name, c_mode.strip(), r_mode.strip()))
    return methods


def _select_c_action(
    c_mode: str,
    agent_c,
    env,
    req,
    obs_c: Dict[str, Any],
    args: argparse.Namespace,
    rng: np.random.RandomState,
    server_selected_count: np.ndarray,
) -> tuple[int, np.ndarray]:
    """Select a flat C action id, returning raw mask for metric accounting."""
    raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    if c_mode == "ppo_c":
        c_idx, raw_c_mask, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        return int(c_idx if c_idx is not None else 0), np.asarray(raw_c_mask, dtype=bool)

    c_idx = select_offloading_action(
        c_mode.replace("_c", ""),
        env,
        req,
        obs_c,
        raw_mask,
        rng=rng,
        server_selected_count=server_selected_count,
    )
    return int(c_idx if c_idx is not None else 0), raw_mask


def _select_r_action_idx(
    r_mode: str,
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    rank_policy: CounterfactualRRankerPolicy,
    split_id: int,
    server_id: int,
    deep_rmsa,
) -> Optional[int]:
    if r_mode in ("v12", "v12_k50_hops"):
        return rank_policy.select_action(env, req, obs_c, obs_r, agent_r, split_id, server_id)
    if r_mode == "deep_rmsa":
        return deep_rmsa.select_action(obs_r) if deep_rmsa is not None else None
    if r_mode == "ppo_r":
        return agent_r.select_action(obs_r, deterministic=True)
    if r_mode == "ksp_bf":
        return ksp_bf_action(obs_r)
    if r_mode == "ksp_ff":
        return ksp_ff_action(obs_r)
    if r_mode == "ksp_ff_k50_hops":
        return ksp_ff_highest_mod_action(obs_r)
    raise ValueError(f"Unknown R mode: {r_mode}")


def _run_episode(
    env,
    requests,
    agent_c,
    agent_r,
    rank_policy,
    deep_rmsa,
    method_name: str,
    c_mode: str,
    r_mode: str,
    args: argparse.Namespace,
    seed: int,
    metrics: PerMethodMetrics,
) -> None:
    rng = np.random.RandomState(seed + 100003)
    server_selected_count = np.zeros(args.num_servers, dtype=int)

    for req in requests:
        started = time.perf_counter()
        env.advance_time(req.arrival_time)

        # Keep the C-side policy fixed on the main experiment configuration.
        env.k = args.k_paths
        env.path_sort_strategy = args.path_sort_strategy
        env.block_sort_strategy = args.block_sort_strategy
        obs_c = build_agent_c_observation(env, req)
        c_idx, raw_c_mask = _select_c_action(
            c_mode, agent_c, env, req, obs_c, args, rng, server_selected_count
        )
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        server_selected_count[server_id] += 1

        # R-side tuned baselines may use a wider/differently ordered path set.
        env.k, env.path_sort_strategy, env.block_sort_strategy = _r_backend_env_config(
            r_mode, args
        )
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_idx = _select_r_action_idx(
            r_mode, env, req, obs_c, obs_r, agent_r, rank_policy,
            split_id, server_id, deep_rmsa,
        )
        r_action = (
            (0, 0, 0)
            if r_idx is None
            else decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
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
            int(raw_c_mask.sum()) == 0,
            obs_c,
            split_id,
            server_id,
            decision_ms,
            same_as_ppo,
        )
        metrics.active_connections[-1] = len(env.active_connections)


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = _parse_methods(args.methods)
    requested_r = {r for _, _, r in methods}

    env_proto = make_env(
        args.topology,
        args.num_slots,
        args.num_servers,
        42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
        path_sort_strategy=args.path_sort_strategy,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, rank_ckpt = load_ranking_checkpoint(
        args.ranking_checkpoint, args.device
    )
    rank_policy = CounterfactualRRankerPolicy(
        rank_model,
        rank_mean,
        rank_std,
        device=args.device,
        feature_names=rank_ckpt.get("feature_names"),
        candidate_mode=rank_ckpt.get("candidate_mode", "all_legal"),
        max_candidates=rank_ckpt.get("max_candidates", 48),
        ppo_top_k=rank_ckpt.get("ppo_top_k", 8),
        num_random_candidates=rank_ckpt.get("num_random_candidates", 5),
        min_candidates=rank_ckpt.get("min_candidates", 15),
        candidate_seed=rank_ckpt.get("candidate_seed", 12345),
    )

    deep_rmsa = None
    if "deep_rmsa" in requested_r:
        deep_rmsa = _load_deep_rmsa(
            args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device
        )

    episodes_by_seed = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            episodes.append(generate_requests(
                env_proto,
                rng,
                src,
                args.requests_per_episode,
                args.arrival_interval,
                args.holding_min,
                args.holding_max,
                args.deadline_min,
                args.deadline_max,
                args.size_min_mb,
                args.size_max_mb,
                args.edge_cost_min,
                args.edge_cost_max,
                args.num_splits,
                args.split_profile,
            ))
        episodes_by_seed[seed] = episodes

    report_methods: Dict[str, Any] = {}
    for method_name, c_mode, r_mode in methods:
        spec: Dict[str, Any] = {
            "c_mode": c_mode,
            "r_mode": r_mode,
            "per_seed": {},
        }
        for seed in seeds:
            metrics = PerMethodMetrics()
            for ep_idx, requests in enumerate(episodes_by_seed[seed]):
                env = make_env(
                    args.topology,
                    args.num_slots,
                    args.num_servers,
                    seed + ep_idx,
                    modulation_profile=args.modulation_profile,
                    max_blocks=args.max_blocks,
                    block_sort_strategy=args.block_sort_strategy,
                    k=args.k_paths,
                    path_sort_strategy=args.path_sort_strategy,
                )
                env.reset(requests)
                _run_episode(
                    env,
                    requests,
                    agent_c,
                    agent_r,
                    rank_policy,
                    deep_rmsa,
                    method_name,
                    c_mode,
                    r_mode,
                    args,
                    seed + ep_idx,
                    metrics,
                )
            spec["per_seed"][str(seed)] = _aggregate_metrics(metrics)
            print(f"[{method_name}] completed seed {seed}", flush=True)

        sample = next(iter(spec["per_seed"].values()))
        spec["aggregate"] = {
            key: float(np.mean([row[key] for row in spec["per_seed"].values()]))
            for key in sample
            if isinstance(sample[key], (int, float))
        }
        report_methods[method_name] = spec

    return {
        "config": vars(args),
        "methods": report_methods,
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    lines = [
        "# Main S100 System-Level Comparison",
        "",
        "This is the recommended main comparison table for paper/PPT use.",
        "",
        "## Configuration",
        "",
    ]
    cfg = report["config"]
    for key in [
        "topology", "num_slots", "split_profile", "arrival_interval",
        "holding_min", "holding_max", "size_min_mb", "size_max_mb",
        "seeds", "episodes", "requests_per_episode", "k_paths",
        "path_sort_strategy", "ksp_ff_k50_hops_k_paths",
    ]:
        lines.append(f"- `{key}`: `{cfg[key]}`")

    lines.extend([
        "",
        "## Results",
        "",
        "| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ])
    for name, spec in report["methods"].items():
        agg = spec["aggregate"]
        lines.append(
            f"| {name} | {spec['c_mode']} | {spec['r_mode']} | "
            f"{agg['blocking_rate']:.2%} | {agg['raw_mask_empty_rate']:.2%} | "
            f"{agg['no_suitable_block_rate']:.2%} | {agg['server_overload_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{agg['mean_decision_time_ms']:.3f}/{agg['p95_decision_time_ms']:.3f} ms |"
        )

    if "ppo_c+v12" in report["methods"]:
        base = report["methods"]["ppo_c+v12"]["aggregate"]
        lines.extend(["", "## Delta vs PPO-C + v1.2", ""])
        lines.append("| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |")
        lines.append("|---|---:|---:|---:|---:|")
        for name, spec in report["methods"].items():
            if name == "ppo_c+v12":
                continue
            agg = spec["aggregate"]
            lines.append(
                f"| {name} | "
                f"{(agg['blocking_rate'] - base['blocking_rate']) * 100:+.2f} pp | "
                f"{(agg['no_suitable_block_rate'] - base['no_suitable_block_rate']) * 100:+.2f} pp | "
                f"{(agg['server_overload_rate'] - base['server_overload_rate']) * 100:+.2f} pp | "
                f"{agg['mean_delay_ms'] - base['mean_delay_ms']:+.3f} ms |"
            )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_s100_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument(
        "--methods",
        default=(
            "ppo_c+v12,ppo_c+deep_rmsa,ppo_c+ppo_r,ppo_c+ksp_bf,ppo_c+ksp_ff,"
            "ppo_c+ksp_ff_k50_hops,greedy_c+v12,df_c+v12,rf_c+v12,"
            "wo_c+v12,iwd_c+v12"
        ),
    )
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=100)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--path_sort_strategy", default="km", choices=["km", "hops"])
    parser.add_argument("--ksp_ff_k50_hops_k_paths", type=int, default=50)
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/main_s100_system_comparison.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/main_s100_system_comparison.md")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate(args)
    out_json = Path(args.output_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(Path(args.output_md), report)
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
