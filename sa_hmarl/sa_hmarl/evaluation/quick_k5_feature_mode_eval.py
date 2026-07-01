"""Quick K=5 eval for comparing Agent-C feature modes.

Evaluates a list of Agent-C checkpoints with identical request episodes and prints
a compact comparison table.
"""
from __future__ import annotations

import argparse
import time
from argparse import Namespace
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sa_hmarl.evaluation.k5_system_comparison import (
    _aggregate,
    _eval_episode,
    _load_ppo_c,
    _load_ppo_r,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


def _eval_checkpoint(
    ckpt_path: str,
    agent_r,
    env_proto,
    episodes: Dict[int, List[List[Any]]],
    args: Namespace,
) -> Dict[str, Any]:
    agent_c = _load_ppo_c(ckpt_path, args.device)
    ep_results = []
    for seed, eps in episodes.items():
        for ep_idx, requests in enumerate(eps):
            ep_results.append(
                _eval_episode("agent", agent_c, agent_r, requests, env_proto, args, ep_idx)
            )
    agg = _aggregate(ep_results)
    agg["total"] = int(sum(r["total"] for r in ep_results))
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints", type=str, required=True,
                        help="Comma-separated Agent-C checkpoint paths with labels: label=path,label=path")
    parser.add_argument("--agent_r_checkpoint", type=str, default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    # Parse labeled checkpoints
    ckpt_map: Dict[str, str] = {}
    for item in args.checkpoints.split(","):
        label, path = item.split("=", 1)
        ckpt_map[label.strip()] = path.strip()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    # Pre-generate identical episodes for all checkpoints
    episodes: Dict[int, List[List[Any]]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        episodes[seed] = eps

    print("=" * 100)
    print("Quick K=5 Feature-Mode Comparison")
    print(f"Topology: {args.topology}  Slots: {args.num_slots}  Servers: {args.num_servers}  k: {args.k_paths}")
    print(f"Seeds: {seeds}  Eps/seed: {args.episodes}  Reqs/ep: {args.requests_per_episode}")
    print("=" * 100)

    results: Dict[str, Dict[str, Any]] = {}
    t_start = time.time()
    for label, ckpt_path in ckpt_map.items():
        print(f"\nEvaluating {label}: {ckpt_path}")
        t0 = time.time()
        agg = _eval_checkpoint(ckpt_path, agent_r, env_proto, episodes, args)
        results[label] = agg
        print(f"  blocking={agg['blocking_rate']:.4f}  success={agg['success_rate']:.4f}  "
              f"server_overload={agg.get('server_overload_ratio', 0):.4f}  "
              f"no_valid_c={agg.get('no_valid_c_rate', 0):.4f}  "
              f"avg_valid_r={agg.get('avg_valid_r_actions', 0):.3f}  "
              f"delay={agg.get('avg_delay_ms', 0):.2f}ms  ({time.time()-t0:.1f}s)")

    print("\n" + "=" * 100)
    print("Summary Table")
    print("=" * 100)
    print(f"{'Mode':<35} {'Total':>8} {'Block':>8} {'Succ':>8} {'SrvOvld':>8} {'NoValidC':>8} {'AvgValR':>8} {'Delay':>8}")
    print("-" * 100)
    for label, agg in results.items():
        print(
            f"{label:<35} {agg['total']:>8} {agg['blocking_rate']:>8.4f} "
            f"{agg['success_rate']:>8.4f} {agg.get('server_overload_ratio', 0):>8.4f} "
            f"{agg.get('no_valid_c_rate', 0):>8.4f} {agg.get('avg_valid_r_actions', 0):>8.3f} "
            f"{agg.get('avg_delay_ms', 0):>8.2f}"
        )
    print(f"\nTotal eval time: {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
