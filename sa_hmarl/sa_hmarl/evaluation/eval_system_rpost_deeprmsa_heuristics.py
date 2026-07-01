"""System-level comparison for C agents and R backends.

Compares learned C/R combinations, R-side post-decision value, DeepRMSA, and
simple heuristic baselines on the same request traces.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action, ksp_ff_action
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
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import (
    _load_deep_rmsa,
    _select_post_r_action,
    load_post_checkpoint,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _select_c(method: str, agent_c, env, req, obs_c, rng, server_selected_count):
    if method == "ppo_delay":
        return agent_c.select_action(obs_c, deterministic=True)
    return select_offloading_action(
        method, env, req, obs_c, obs_c["agent_c_mask"],
        rng=rng, server_selected_count=server_selected_count,
    )


def _select_r(
    backend: str,
    env,
    req,
    obs_c,
    obs_r,
    split_id: int,
    server_id: int,
    agent_r,
    deep_rmsa,
    post_pack,
    args,
):
    if backend == "ppo_r":
        return _select_r_action(agent_r, obs_r, env.max_blocks)
    if backend == "deep_rmsa":
        idx = deep_rmsa.select_action(obs_r)
        return (0, 0, 0) if idx is None else decode_agent_r_action(
            idx, len(obs_r["mod_names"]), env.max_blocks
        )
    if backend == "r_post":
        model, mean, std = post_pack
        idx = _select_post_r_action(
            env, req, obs_c, obs_r, agent_r, model, mean, std,
            split_id, server_id, "post_only", 1.0, args.device,
        )
        return decode_agent_r_action(idx, len(obs_r["mod_names"]), env.max_blocks)
    if backend == "ksp_ff":
        idx = ksp_ff_action(obs_r)
        return (0, 0, 0) if idx is None else decode_agent_r_action(
            idx, len(obs_r["mod_names"]), env.max_blocks
        )
    if backend == "ksp_bf":
        idx = ksp_bf_action(obs_r)
        return (0, 0, 0) if idx is None else decode_agent_r_action(
            idx, len(obs_r["mod_names"]), env.max_blocks
        )
    raise ValueError(f"Unknown R backend: {backend}")


def _run_combo(
    c_method: str,
    r_backend: str,
    episodes_by_seed: Dict[int, List[List[Any]]],
    agent_c,
    agent_r,
    deep_rmsa,
    post_pack,
    args,
) -> Dict[str, Any]:
    per_seed = {}
    for seed, episodes in episodes_by_seed.items():
        metrics = PerMethodMetrics()
        rng = np.random.RandomState(seed + 999)
        for requests in episodes:
            env = make_env(
                args.topology, args.num_slots, args.num_servers, seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)
            server_selected_count = np.zeros(args.num_servers, dtype=int)
            for req in requests:
                started = time.perf_counter()
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                c_idx = _select_c(
                    c_method, agent_c, env, req, obs_c, rng, server_selected_count
                )
                if c_idx is None:
                    c_idx = 0
                split_id, server_id = decode_agent_c_action(int(c_idx), args.num_servers)
                server_selected_count[server_id] += 1
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                r_action = _select_r(
                    r_backend, env, req, obs_c, obs_r, split_id, server_id,
                    agent_r, deep_rmsa, post_pack, args,
                )
                decision_ms = (time.perf_counter() - started) * 1000.0
                _, _, _, info = env.step((split_id, server_id), r_action)
                _record_outcome(
                    metrics, info,
                    int(np.asarray(obs_c["agent_c_mask"], dtype=bool).sum()) == 0,
                    obs_c, split_id, server_id, decision_ms, True,
                )
                metrics.active_connections[-1] = len(env.active_connections)
        per_seed[str(seed)] = _aggregate_metrics(metrics)
        print(f"[{c_method}+{r_backend}] completed seed {seed}", flush=True)

    aggregate = {
        key: float(np.mean([row[key] for row in per_seed.values()]))
        for key in next(iter(per_seed.values()))
        if isinstance(next(iter(per_seed.values()))[key], (int, float))
    }
    return {"per_seed": per_seed, "aggregate": aggregate}


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
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
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)
    post_model, post_mean, post_std, post_ckpt = load_post_checkpoint(args.post_checkpoint, args.device)
    post_pack = (post_model, post_mean, post_std)

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

    combo_specs = [
        ("ppo_delay", "ppo_r"),
        ("ppo_delay", "r_post"),
        ("ppo_delay", "deep_rmsa"),
        ("ppo_delay", "ksp_ff"),
        ("ppo_delay", "ksp_bf"),
        ("greedy", "ppo_r"),
        ("greedy", "r_post"),
        ("greedy", "deep_rmsa"),
        ("wo", "ksp_bf"),
        ("df", "ksp_bf"),
        ("rf", "ksp_bf"),
        ("iwd", "ksp_bf"),
    ]
    requested = [item.strip() for item in args.combos.split(",") if item.strip()]
    if requested:
        parsed = []
        for item in requested:
            c, r = item.split("+", 1)
            parsed.append((c, r))
        combo_specs = parsed

    results = {}
    for c_method, r_backend in combo_specs:
        name = f"{c_method}+{r_backend}"
        results[name] = _run_combo(
            c_method, r_backend, episodes_by_seed,
            agent_c, agent_r, deep_rmsa, post_pack, args,
        )
    return {
        "config": vars(args),
        "post_checkpoint_mode": post_ckpt.get("mode"),
        "results": results,
    }


def _write_markdown(path: Path, report: Dict[str, Any]) -> None:
    rows = sorted(
        report["results"].items(),
        key=lambda item: item[1]["aggregate"]["blocking_rate"],
    )
    lines = [
        "# System-Level C/R Combination Comparison", "",
        "| Method | Blocking | Delay mean/P95 | NSB | Overload | Decision mean/P95 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in rows:
        agg = result["aggregate"]
        lines.append(
            f"| {name} | {agg['blocking_rate']:.2%} | "
            f"{agg['mean_delay_ms']:.3f}/{agg['p95_delay_ms']:.3f} ms | "
            f"{agg['no_suitable_block_rate']:.2%} | {agg['server_overload_rate']:.2%} | "
            f"{agg['mean_decision_time_ms']:.3f}/{agg['p95_decision_time_ms']:.3f} |"
        )
    lines += [
        "",
        "## Notes",
        "",
        "- `ppo_delay` is the old delay-aware PPO-C checkpoint.",
        "- `r_post` is the H-step R-side post-decision value backend.",
        "- DeepRMSA is evaluated in its native matched action space when using S24/K3/M1.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--deep_rmsa_checkpoint", default="sa_hmarl/checkpoints/deep_rmsa_snap24_reach_mixed.pt")
    parser.add_argument("--post_checkpoint", default="sa_hmarl/checkpoints/r_post_decision_h3_k3m1_compare/c_post_decision_joint.pt")
    parser.add_argument("--combos", default="", help="Optional comma list like ppo_delay+r_post,greedy+ksp_bf")
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=3)
    parser.add_argument("--max_blocks", type=int, default=1)
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
    parser.add_argument("--output_json", default="sa_hmarl/experiments/system_rpost_deeprmsa_heuristics.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/system_rpost_deeprmsa_heuristics.md")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    report = evaluate(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_markdown(out_md, report)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
