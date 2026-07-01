"""Pressure scan for metro24_bottleneck.

Find a traffic setting where blocking rate is in the 10%-30% sweet spot
for all methods, using existing checkpoints without retraining.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import numpy as np
from itertools import product

from sa_hmarl.evaluation.eval_joint_multitopo import (
    evaluate_topology, _aggregate, METHODS
)
from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_agent_c(ckpt_path: str, device: str = "cpu") -> AgentC:
    agent = AgentC(
        input_dim=17, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device=device
    )
    ckpt = load_checkpoint(Path(ckpt_path), map_location=device)
    agent.q_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_state"])
    return agent


def _load_agent_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> AgentR:
    agent = AgentR(
        input_dim=11, mod_registry=mod_reg, hidden_dims=(128, 64),
        gamma=0.95, epsilon=0.0, device=device,
    )
    ckpt = load_checkpoint(Path(ckpt_path), map_location=device)
    agent.q_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_state"])
    return agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="metro24_bottleneck")
    parser.add_argument("--joint_c", type=str,
                        default="sa_hmarl/checkpoints/joint_metro24_c_best.pt")
    parser.add_argument("--joint_r", type=str,
                        default="sa_hmarl/checkpoints/joint_metro24_r_best.pt")
    parser.add_argument("--sep_c", type=str,
                        default="sa_hmarl/checkpoints/agent_c_multitopo_best.pt")
    parser.add_argument("--sep_r", type=str,
                        default="sa_hmarl/checkpoints/agent_r_multitopo.pt")
    parser.add_argument("--num_slots_list", type=str, default="16,24,32")
    parser.add_argument("--rpe_list", type=str, default="20,40,60")
    parser.add_argument("--interval_list", type=str, default="0.25,0.15,0.1")
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    args = parser.parse_args()

    device = "cpu"
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    joint_c = _load_agent_c(args.joint_c, device)
    joint_r = _load_agent_r(args.joint_r, mod_reg, device)
    sep_c = _load_agent_c(args.sep_c, device)
    sep_r = _load_agent_r(args.sep_r, mod_reg, device)
    # dummy no_team agents (not used in METHODS but required by evaluate_topology signature)
    no_team_c = sep_c
    no_team_r = sep_r

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    num_slots_vals = [int(s.strip()) for s in args.num_slots_list.split(",")]
    rpe_vals = [int(s.strip()) for s in args.rpe_list.split(",")]
    interval_vals = [float(s.strip()) for s in args.interval_list.split(",")]

    print("=" * 120)
    print("Pressure Scan for metro24_bottleneck")
    print("=" * 120)
    print(f"{'Slots':>5} {'RPE':>4} {'Arr':>5} | "
          f"{'Joint':>6} {'Sep':>6} {'A+C+BF':>6} {'G+R':>6} {'G+BF':>6} | "
          f"{'Range':>8} {'Status':>10}")
    print("-" * 120)

    candidates = []
    for num_slots, rpe, interval in product(num_slots_vals, rpe_vals, interval_vals):
        traffic_args = {
            "arrival_interval": interval,
            "deadline_min": args.deadline_min,
            "deadline_max": args.deadline_max,
            "size_min_mb": args.size_min_mb,
            "size_max_mb": args.size_max_mb,
            "num_slots": num_slots,
            "num_servers": args.num_servers,
            "slot_bw_hz": args.slot_bw_hz,
            "guard_band_fs": args.guard_band_fs,
            "modulation_profile": args.modulation_profile,
        }
        all_results = evaluate_topology(
            args.topology,
            joint_c, joint_r, no_team_c, no_team_r, sep_c, sep_r,
            seeds, args.episodes, rpe, traffic_args,
        )
        blocks = {}
        for method in METHODS:
            agg = _aggregate(all_results[method])
            blocks[method] = agg["blocking"]

        vals = list(blocks.values())
        min_b, max_b = min(vals), max(vals)
        range_b = max_b - min_b

        label = "TOO_EASY" if max_b < 0.10 else "TOO_HARD" if min_b > 0.30 else "SWEET_SPOT"
        if label == "SWEET_SPOT":
            candidates.append((num_slots, rpe, interval, blocks, range_b))

        print(
            f"{num_slots:>5} {rpe:>4} {interval:>5.2f} | "
            f"{blocks[METHODS[0]]:>6.3f} {blocks[METHODS[1]]:>6.3f} "
            f"{blocks[METHODS[2]]:>6.3f} {blocks[METHODS[3]]:>6.3f} {blocks[METHODS[4]]:>6.3f} | "
            f"{range_b:>8.3f} {label:>10}"
        )

    print("=" * 120)
    if candidates:
        print(f"\nFound {len(candidates)} SWEET_SPOT candidate(s):")
        # sort by smallest range first (most consistent comparison), then by avg blocking
        candidates.sort(key=lambda x: (x[4], -np.mean(list(x[3].values()))))
        for num_slots, rpe, interval, blocks, range_b in candidates:
            avg_b = np.mean(list(blocks.values()))
            print(
                f"  slots={num_slots} rpe={rpe} interval={interval} => "
                f"avg_blk={avg_b:.3f} range={range_b:.3f}"
            )
    else:
        print("\nNo SWEET_SPOT found. Consider adjusting scan ranges.")
    print("=" * 120)


if __name__ == "__main__":
    main()
