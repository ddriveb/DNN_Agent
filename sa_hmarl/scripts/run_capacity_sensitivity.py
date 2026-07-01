"""MEC capacity sensitivity analysis for SA-HMARL.

Runs multi-seed joint evaluation across multiple capacity values.

Usage (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/scripts/run_capacity_sensitivity.py
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import json
import numpy as np
import torch
from collections import Counter
from scipy import stats

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import make_env
from sa_hmarl.evaluation.eval_joint import (
    evaluate_method, METHODS, _aggregate_across_seeds,
)
from sa_hmarl.utils.checkpoint import load_checkpoint


def run_sweep(capacity, seeds, episodes, requests_per_episode, args_common):
    """Run evaluation for one capacity across multiple seeds."""
    cap_list = [float(capacity), float(capacity)]
    env = make_env(
        args_common.topology, args_common.num_slots, args_common.num_servers,
        seeds[0], slot_bw_hz=args_common.slot_bw_hz,
        guard_band_fs=args_common.guard_band_fs, capacities=cap_list,
        modulation_profile=args_common.modulation_profile,
    )
    device = 'cpu'
    mod_reg = ModulationRegistry.from_profile(args_common.modulation_profile)

    agent_c = AgentC(input_dim=17, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device=device)
    ckpt_c = load_checkpoint(args_common.checkpoint_c, map_location=device)
    agent_c.q_net.load_state_dict(ckpt_c["model_state"])
    agent_c.target_net.load_state_dict(ckpt_c["target_state"])

    agent_r = AgentR(input_dim=11, mod_registry=mod_reg, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device=device)
    ckpt_r = load_checkpoint(args_common.checkpoint_r, map_location=device)
    agent_r.q_net.load_state_dict(ckpt_r["model_state"])
    agent_r.target_net.load_state_dict(ckpt_r["target_state"])

    from sa_hmarl.evaluation.eval_joint import _run_single_seed
    seed_results = []
    for s in seeds:
        args_common.seed = s
        seed_results.append(_run_single_seed(env, agent_c, agent_r, args_common))

    per_seed_means, per_seed_episode_blocks, overall, win_count, aggregated = _aggregate_across_seeds(seed_results)
    return overall, aggregated, win_count


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=20)
    parser.add_argument("--arrival_interval", type=float, default=0.1)
    parser.add_argument("--holding_min", type=float, default=6.0)
    parser.add_argument("--holding_max", type=float, default=15.0)
    parser.add_argument("--deadline_min", type=float, default=20.0)
    parser.add_argument("--deadline_max", type=float, default=60.0)
    parser.add_argument("--size_min_mb", type=float, default=20.0)
    parser.add_argument("--size_max_mb", type=float, default=80.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--topology", type=str, default="net1")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--checkpoint_c", type=str, default="sa_hmarl/checkpoints/agent_c_best.pt")
    parser.add_argument("--checkpoint_r", type=str, default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    capacities = [30, 50, 70, 100]
    target = "Agent-C-DQN + Agent-R-DQN"
    ksp_baseline = "Agent-C-DQN + KSP-BF"

    print("=" * 110)
    print("MEC Capacity Sensitivity Analysis")
    print(f"Seeds: {seeds} | Episodes/seed: {args.episodes} | Requests/episode: {args.requests_per_episode}")
    print("=" * 110)

    results = []
    for cap in capacities:
        print(f"\n[Running capacity={cap} GFLOPS]", file=sys.stderr)
        overall, aggregated, win_count = run_sweep(cap, seeds, args.episodes, args.requests_per_episode, args)
        row = {"capacity": cap, "overall": overall, "aggregated": aggregated, "win_count": win_count}
        results.append(row)

    # Print table
    header = (f"{'Capacity':>8} {'Blocking':>12} {'Success':>12} {'Reward':>12} "
              f"{'Delay(ms)':>12} {'Waste':>10} {'AvgFS':>8} {'Path(km)':>10} "
              f"{'SrvOvl%':>10} {'WinCnt':>8}")
    print("\n" + header)
    print("-" * 110)
    for row in results:
        cap = row["capacity"]
        o = row["overall"][target]
        reasons = row["aggregated"][target]["reasons"]
        total_fail = sum(reasons.values())
        srv_ovl = reasons.get("server_overload", 0) / total_fail * 100 if total_fail else 0
        print(f"{cap:>8} {o['blocking_rate'][0]:>12.3f} {o['success_rate'][0]:>12.3f} "
              f"{o['avg_reward'][0]:>+12.3f} {o['avg_delay_ms'][0]:>12.2f} "
              f"{o['avg_block_waste'][0]:>10.3f} {o['avg_fs'][0]:>8.2f} {o['avg_path_len_km'][0]:>10.1f} "
              f"{srv_ovl:>9.1f}% {row['win_count']:>8}/{len(seeds)}")

    print("-" * 110)

    # Print relative to KSP-BF baseline
    print("\nBlocking vs Agent-C + KSP-BF baseline")
    print("-" * 110)
    print(f"{'Capacity':>8} {'Joint':>12} {'KSP-BF':>12} {'Delta':>10} {'Delta%':>10}")
    print("-" * 110)
    for row in results:
        cap = row["capacity"]
        joint_blk = row["overall"][target]["blocking_rate"][0]
        ksp_blk = row["overall"][ksp_baseline]["blocking_rate"][0]
        delta = joint_blk - ksp_blk
        delta_pct = delta / ksp_blk * 100 if ksp_blk > 0 else 0
        print(f"{cap:>8} {joint_blk:>12.3f} {ksp_blk:>12.3f} {delta:>10.3f} {delta_pct:>9.1f}%")
    print("=" * 110)


if __name__ == "__main__":
    main()
