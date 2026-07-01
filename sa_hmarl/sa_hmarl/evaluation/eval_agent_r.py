"""Formal evaluation script for Agent-R DQN vs KSP-FF / KSP-BF baselines.

Run after training (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.train_agent_r \
        --episodes 1000 --requests_per_episode 30 --arrival_interval 0.25 \
        --holding_min 4 --holding_max 10 --size_min_mb 5.0 --size_max_mb 30.0 \
        --deadline_min 30 --deadline_max 100 --slot_bw_hz 1.25e9 --guard_band_fs 1 \
        --waste_coef 0.8

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_agent_r \
        --episodes 20 --requests_per_episode 20

Stress evaluation example:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_agent_r \
        --episodes 20 --requests_per_episode 20 \
        --arrival_interval 0.1 --holding_min 6 --holding_max 15 \
        --size_min_mb 20.0 --size_max_mb 80.0 --deadline_min 20 --deadline_max 60 \
        --slot_bw_hz 1.25e9 --guard_band_fs 1
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
from collections import Counter

from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action, ksp_bf_action
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def evaluate_method(env: SMDPEnv,
                    agent,
                    requests: list,
                    server_id: int,
                    method_name: str,
                    waste_coef: float = 0.8) -> dict:
    """Evaluate one RMSA method on a fixed request sequence.

    Returns dict with standard metrics plus diagnostic fields:
        - fs_list: list of num_slots for successful requests
        - reason_counter: Counter of failure reasons
        - no_valid_action_count: number of times select_action returned None
    """
    env.reset(requests)
    total_reward = 0.0
    blocked = 0
    successes = 0
    total_delay = 0.0
    total_waste = 0.0
    total_fs = 0.0
    total_path_len = 0.0
    fs_list = []
    reason_counter = Counter()
    no_valid_action_count = 0

    for req in requests:
        obs = build_agent_r_observation(env, req, split_id=0, server_id=server_id)

        if method_name == "Agent-R-DQN":
            action_idx = agent.select_action(obs, epsilon=0.0)
        elif method_name == "DeepRMSA-A3C":
            action_idx = agent.select_action(obs)
        elif method_name == "KSP-FF":
            action_idx = ksp_ff_action(obs)
        elif method_name == "KSP-BF":
            action_idx = ksp_bf_action(obs)
        else:
            raise ValueError(f"Unknown method: {method_name}")

        if action_idx is None:
            no_valid_action_count += 1
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(
                action_idx, len(obs["mod_names"]), env.max_blocks
            )

        _, _, done, info = env.step((0, server_id), action_r)
        reward = compute_reward(info, waste_coef)
        total_reward += reward

        if info.get("success", False):
            successes += 1
            fs = info.get("num_slots", 0)
            total_delay += info.get("delay_ms", 0.0)
            total_waste += info.get("block_waste", 0.0)
            total_fs += fs
            total_path_len += info.get("path_dist_km", 0.0)
            fs_list.append(fs)
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reason_counter[reason] += 1

    n = len(requests)
    result = {
        "blocking_rate": blocked / n,
        "success_rate": successes / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / successes if successes > 0 else 0.0,
        "avg_block_waste": total_waste / successes if successes > 0 else 0.0,
        "avg_fs": total_fs / successes if successes > 0 else 0.0,
        "avg_path_len_km": total_path_len / successes if successes > 0 else 0.0,
        "fs_list": fs_list,
        "reason_counter": reason_counter,
        "no_valid_action_rate": no_valid_action_count / n,
    }
    return result


def _fs_histogram(fs_list):
    """Build a simple FS histogram string."""
    if not fs_list:
        return "N/A"
    bins = {
        "1": 0, "2": 0, "3-4": 0, "5-8": 0, "9-16": 0, ">16": 0
    }
    for fs in fs_list:
        if fs == 1:
            bins["1"] += 1
        elif fs == 2:
            bins["2"] += 1
        elif fs <= 4:
            bins["3-4"] += 1
        elif fs <= 8:
            bins["5-8"] += 1
        elif fs <= 16:
            bins["9-16"] += 1
        else:
            bins[">16"] += 1
    total = len(fs_list)
    return ", ".join(f"{k}:{v/total:.1%}" for k, v in bins.items() if v > 0)


def main():
    parser = argparse.ArgumentParser(description="Evaluate Agent-R vs RMSA baselines")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=20)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--target_update_freq", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--learning_starts", type=int, default=100)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--topology", type=str, default="net1")
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to Agent-R checkpoint. If omitted, uses "
                             "<package_root>/checkpoints/agent_r_smoke.pt")
    parser.add_argument("--m_blocks", type=int, default=1,
                        help="DeepRMSA M parameter: FS blocks per path")
    parser.add_argument("--deep_rmsa_checkpoint", type=str, default=None,
                        help="Path to DeepRMSA checkpoint. If omitted, tries "
                             "deep_rmsa_mixed.pt then deep_rmsa_smoke.pt")
    args = parser.parse_args()

    rng = np.random.RandomState(args.seed)
    env = make_env(
        args.topology, args.num_slots, args.num_servers, args.seed,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    device = 'cpu'

    # Load Agent-R DQN
    agent_r = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        gamma=0.95,
        epsilon=0.0,
        device=device,
    )
    package_root = Path(__file__).resolve().parents[2]
    if args.checkpoint is None:
        ckpt_path_r = package_root / "checkpoints" / "agent_r_smoke.pt"
    else:
        ckpt_path_r = Path(args.checkpoint)

    if ckpt_path_r.exists():
        ckpt = load_checkpoint(ckpt_path_r, map_location=device)
        agent_r.q_net.load_state_dict(ckpt["model_state"])
        agent_r.target_net.load_state_dict(ckpt["target_state"])
        print(f"Loaded Agent-R checkpoint from {ckpt_path_r}")
    else:
        print("WARNING: No Agent-R checkpoint found, using random-init Agent-R")

    # Load DeepRMSA A3C
    agent_deep = DeepRMSAAgent(
        num_nodes=env.net.NUM_NODES,
        num_slots=args.num_slots,
        k_path=env.k,
        m_blocks=args.m_blocks,
        mod_registry=mod_reg,
        gamma=0.95,
        device=device,
    )
    if args.deep_rmsa_checkpoint is None:
        ckpt_path_deep = package_root / "checkpoints" / "deep_rmsa_mixed.pt"
        if not ckpt_path_deep.exists():
            ckpt_path_deep = package_root / "checkpoints" / "deep_rmsa_smoke.pt"
    else:
        ckpt_path_deep = Path(args.deep_rmsa_checkpoint)

    if ckpt_path_deep.exists():
        ckpt_deep = torch.load(str(ckpt_path_deep), map_location=device)
        ckpt_nodes = ckpt_deep.get("num_nodes")
        if ckpt_nodes is not None and ckpt_nodes != env.net.NUM_NODES:
            raise ValueError(
                f"DeepRMSA checkpoint topology mismatch: "
                f"ckpt has num_nodes={ckpt_nodes}, "
                f"but env has NUM_NODES={env.net.NUM_NODES}. "
                f"Please use --topology that matches the checkpoint."
            )
        agent_deep.load_state_dict(ckpt_deep)
        print(f"Loaded DeepRMSA checkpoint from {ckpt_path_deep}")
    else:
        print("WARNING: No DeepRMSA checkpoint found, using random-init DeepRMSA")

    # Set eval mode for DeepRMSA (argmax action selection)
    agent_deep.eval()

    # Agent selector for evaluation loop
    agents = {
        "Agent-R-DQN": agent_r,
        "DeepRMSA-A3C": agent_deep,
    }

    # Evaluate all methods on identical request sequences
    all_results = {"Agent-R-DQN": [], "DeepRMSA-A3C": [], "KSP-FF": [], "KSP-BF": []}

    for ep in range(args.episodes):
        src = rng.randint(0, env.net.NUM_NODES)
        server_id = rng.randint(0, len(env.mec.servers))
        requests = generate_requests(env, rng, src, args.requests_per_episode,
                                     arrival_interval=args.arrival_interval,
                                     holding_min=args.holding_min,
                                     holding_max=args.holding_max,
                                     deadline_min=args.deadline_min,
                                     deadline_max=args.deadline_max,
                                     size_min_mb=args.size_min_mb,
                                     size_max_mb=args.size_max_mb,
                                     edge_cost_min=args.edge_cost_min,
                                     edge_cost_max=args.edge_cost_max)

        for method in all_results.keys():
            agent = agents.get(method)
            result = evaluate_method(env, agent, requests, server_id, method, args.waste_coef)
            all_results[method].append(result)

    # Aggregate and display
    print("\n" + "=" * 90)
    print("Agent-R / DeepRMSA Evaluation Results")
    print(f"Episodes: {args.episodes}, Requests/episode: {args.requests_per_episode}, "
          f"Topology: {args.topology}, Slots: {args.num_slots}")
    print(f"Slot BW: {args.slot_bw_hz/1e9:.2f} GHz, Guard: {args.guard_band_fs}")
    print("=" * 90)
    header = (f"{'Method':<15} {'BlkRate':>8} {'SuccRate':>9} {'AvgRwd':>8} "
              f"{'Delay(ms)':>10} {'Waste':>7} {'AvgFS':>7} {'Path(km)':>9}")
    print(header)
    print("-" * 90)

    for method, results in all_results.items():
        avg = {k: np.mean([r[k] for r in results]) for k in results[0].keys()
               if k not in ("fs_list", "reason_counter")}
        print(f"{method:<15} {avg['blocking_rate']:>8.3f} {avg['success_rate']:>9.3f} "
              f"{avg['avg_reward']:>8.3f} {avg['avg_delay_ms']:>10.2f} "
              f"{avg['avg_block_waste']:>7.3f} {avg['avg_fs']:>7.2f} "
              f"{avg['avg_path_len_km']:>9.2f}")

    print("-" * 90)

    # Diagnostic section
    print("\nDiagnostics")
    print("-" * 90)
    for method, results in all_results.items():
        all_fs = []
        all_reasons = Counter()
        no_valid_total = 0.0
        for r in results:
            all_fs.extend(r["fs_list"])
            all_reasons.update(r["reason_counter"])
            no_valid_total += r["no_valid_action_rate"]

        avg_no_valid = no_valid_total / len(results)
        print(f"\n{method}:")
        print(f"  AvgFS (success only): {np.mean(all_fs):.2f}" if all_fs else "  AvgFS: N/A")
        print(f"  FS histogram: {_fs_histogram(all_fs)}")
        print(f"  No-valid-action rate: {avg_no_valid:.3f}")
        if all_reasons:
            total_failures = sum(all_reasons.values())
            print(f"  Failure reasons (out of {total_failures} failures):")
            for reason, count in all_reasons.most_common(6):
                print(f"    {reason}: {count} ({count/total_failures:.1%})")

    print("=" * 90)


if __name__ == "__main__":
    main()
