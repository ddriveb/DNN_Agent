"""Multi-topology evaluation for trained Agent-C / Agent-R.

Evaluates on net1, net2, net3 (seen) and nsfnet (unseen).
For each topology, uses seeds 42,123,456,789,2024 with 20 episodes x 20 requests.

Methods evaluated:
  1. Separate Agent-C + Agent-R (multitopo checkpoints)
  2. Agent-C-DQN + KSP-BF
  3. Compute-Greedy + KSP-BF
  4. Spectrum-Greedy + KSP-BF
  5. Random-valid-C + KSP-BF

Usage (from project root):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_multitopo
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import torch
from collections import Counter
from typing import Dict, List

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_agent_c(ckpt_path: str, device: str = "cpu") -> AgentC:
    agent = AgentC(
        input_dim=17, hidden_dims=(128, 64), gamma=0.95, epsilon=0.0, device=device
    )
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent.q_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_state"])
    return agent


def _load_agent_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> AgentR:
    agent = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        gamma=0.95,
        epsilon=0.0,
        device=device,
    )
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent.q_net.load_state_dict(ckpt["model_state"])
    agent.target_net.load_state_dict(ckpt["target_state"])
    return agent


def _compute_greedy_action(obs_c, valid, key):
    if len(valid) == 0:
        return None
    best = valid[0]
    best_val = obs_c["candidate_features"][best][key]
    if best_val == float("inf"):
        best_val = 1e9
    for idx in valid[1:]:
        val = obs_c["candidate_features"][idx][key]
        if val == float("inf"):
            val = 1e9
        if val < best_val:
            best_val = val
            best = idx
    return int(best)


def _spectrum_greedy_action(obs_c, valid):
    if len(valid) == 0:
        return None
    best = valid[0]
    best_fs = obs_c["candidate_features"][best]["best_fs_estimate"]
    best_count = obs_c["candidate_features"][best]["feasible_count"]
    for idx in valid[1:]:
        fs = obs_c["candidate_features"][idx]["best_fs_estimate"]
        count = obs_c["candidate_features"][idx]["feasible_count"]
        if fs is not None and (best_fs is None or fs < best_fs):
            best = idx
            best_fs = fs
            best_count = count
        elif fs == best_fs and count > best_count:
            best = idx
            best_count = count
    return int(best)


def evaluate_method(
    env: SMDPEnv,
    agent_c: AgentC,
    agent_r: AgentR,
    requests: list,
    method_name: str,
    waste_coef: float = 0.8,
    rng: np.random.RandomState = None,
) -> dict:
    """Evaluate one joint method on a fixed request sequence."""
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
    split_counter = Counter()
    server_counter = Counter()
    mod_counter = Counter()
    no_valid_action_count = 0

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        mask = obs_c["agent_c_mask"]
        valid = np.where(mask)[0]

        if method_name.startswith("Agent-C-DQN"):
            action_idx_c = agent_c.select_action(obs_c, epsilon=0.0)
        elif method_name.startswith("Compute-Greedy"):
            action_idx_c = _compute_greedy_action(obs_c, valid, "edge_compute_ms")
        elif method_name.startswith("Spectrum-Greedy"):
            action_idx_c = _spectrum_greedy_action(obs_c, valid)
        elif method_name.startswith("Random-valid-C"):
            _rng = rng if rng is not None else np.random
            action_idx_c = int(_rng.choice(valid)) if len(valid) > 0 else None
        else:
            raise ValueError(f"Unknown method: {method_name}")

        if action_idx_c is None:
            no_valid_action_count += 1
            action_c = (0, 0)
        else:
            action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

        split_id, server_id = action_c
        split_counter[split_id] += 1
        server_counter[server_id] += 1

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        if "Agent-R" in method_name and "KSP-BF" not in method_name:
            action_idx_r = agent_r.select_action(obs_r, epsilon=0.0)
        else:
            action_idx_r = ksp_bf_action(obs_r)

        if action_idx_r is None:
            action_r = (0, 0, 0)
        else:
            action_r = decode_agent_r_action(
                action_idx_r, len(obs_r["mod_names"]), env.max_blocks
            )

        _, _, done, info = env.step(action_c, action_r)
        server = env.mec.servers[server_id]
        reward = compute_agent_c_reward(info, req.deadline_ms, waste_coef, server.utilization)
        total_reward += reward

        if info.get("success", False):
            successes += 1
            fs = info.get("num_slots", 0)
            total_delay += info.get("delay_ms", 0.0)
            total_waste += info.get("block_waste", 0.0)
            total_fs += fs
            total_path_len += info.get("path_dist_km", 0.0)
            fs_list.append(fs)
            mod_counter[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reason_counter[reason] += 1

    n = len(requests)
    return {
        "blocking_rate": blocked / n,
        "success_rate": successes / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / successes if successes > 0 else 0.0,
        "avg_block_waste": total_waste / successes if successes > 0 else 0.0,
        "avg_fs": total_fs / successes if successes > 0 else 0.0,
        "avg_path_len_km": total_path_len / successes if successes > 0 else 0.0,
        "fs_list": fs_list,
        "reason_counter": reason_counter,
        "split_counter": split_counter,
        "server_counter": server_counter,
        "mod_counter": mod_counter,
        "no_valid_action_rate": no_valid_action_count / n,
    }


METHODS = [
    "Agent-C-DQN + Agent-R-DQN",
    "Agent-C-DQN + KSP-BF",
    "Compute-Greedy + Agent-R-DQN",
    "Compute-Greedy + KSP-BF",
    "Spectrum-Greedy + KSP-BF",
    "Random-valid-C + KSP-BF",
]


def evaluate_topology(
    topology: str,
    agent_c: AgentC,
    agent_r: AgentR,
    seeds: List[int],
    eval_episodes: int,
    requests_per_episode: int,
    traffic_args: dict,
) -> Dict[str, List[Dict]]:
    """Evaluate all methods on one topology across multiple seeds."""
    mod_reg = ModulationRegistry.from_profile(
        traffic_args.get("modulation_profile", "default")
    )
    env = make_env(
        topology=topology,
        num_servers=traffic_args.get("num_servers", 2),
        seed=42,
        num_slots=traffic_args.get("num_slots", 32),
        slot_bw_hz=traffic_args.get("slot_bw_hz", 1.25e9),
        guard_band_fs=traffic_args.get("guard_band_fs", 1),
        modulation_profile=traffic_args.get("modulation_profile", "default"),
    )

    all_results: Dict[str, List[Dict]] = {m: [] for m in METHODS}

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for ep in range(eval_episodes):
            src = rng.randint(0, env.net.NUM_NODES)
            requests = generate_requests(
                env,
                rng,
                src,
                num_requests=requests_per_episode,
                arrival_interval=traffic_args.get("arrival_interval", 0.25),
                holding_min=traffic_args.get("holding_min", 4.0),
                holding_max=traffic_args.get("holding_max", 10.0),
                deadline_min=traffic_args.get("deadline_min", 30.0),
                deadline_max=traffic_args.get("deadline_max", 100.0),
                size_min_mb=traffic_args.get("size_min_mb", 5.0),
                size_max_mb=traffic_args.get("size_max_mb", 30.0),
                edge_cost_min=traffic_args.get("edge_cost_min", 0.5),
                edge_cost_max=traffic_args.get("edge_cost_max", 15.0),
            )
            for method in METHODS:
                result = evaluate_method(env, agent_c, agent_r, requests, method, 0.8, rng=rng)
                all_results[method].append(result)

    return all_results


def _aggregate(results: List[Dict]) -> Dict:
    blocking = float(np.mean([r["blocking_rate"] for r in results]))
    success = float(np.mean([r["success_rate"] for r in results]))
    avg_reward = float(np.mean([r["avg_reward"] for r in results]))
    avg_fs = float(np.mean([r["avg_fs"] for r in results]))
    avg_path = float(np.mean([r["avg_path_len_km"] for r in results]))

    mod_counter: Counter = Counter()
    reason_counter: Counter = Counter()
    for r in results:
        mod_counter.update(r["mod_counter"])
        reason_counter.update(r["reason_counter"])

    return {
        "blocking": blocking,
        "success": success,
        "avg_reward": avg_reward,
        "avg_fs": avg_fs,
        "avg_path": avg_path,
        "mods": dict(mod_counter),
        "reasons": dict(reason_counter),
    }


def _fmt_mods(mod_dict: Dict) -> str:
    if not mod_dict:
        return "N/A"
    total = sum(mod_dict.values())
    parts = [f"{k}={v/total:.1%}" for k, v in sorted(mod_dict.items())]
    return ", ".join(parts)


def _fmt_reasons(reason_dict: Dict) -> str:
    if not reason_dict:
        return "none"
    total = sum(reason_dict.values())
    # Print all reasons sorted by count desc, with count + percentage
    items = sorted(reason_dict.items(), key=lambda x: x[1], reverse=True)
    parts = [f"{k}={v} ({v/total:.1%})" for k, v in items]
    return ", ".join(parts)


def main():
    parser = argparse.ArgumentParser(description="Multi-topology evaluation")
    parser.add_argument("--checkpoint_c", type=str, default=None)
    parser.add_argument("--checkpoint_r", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
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
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    args = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    device = "cpu"
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    if args.checkpoint_c:
        ckpt_c_path = args.checkpoint_c
    else:
        ckpt_c_path = str(ckpt_dir / "agent_c_multitopo_best.pt")
    if args.checkpoint_r:
        ckpt_r_path = args.checkpoint_r
    else:
        ckpt_r_path = str(ckpt_dir / "agent_r_multitopo.pt")

    print(f"Loading Agent-C from {ckpt_c_path}")
    print(f"Loading Agent-R from {ckpt_r_path}")
    agent_c = _load_agent_c(ckpt_c_path, device)
    agent_r = _load_agent_r(ckpt_r_path, mod_reg, device)

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    topologies = ["net1", "net2", "net3", "nsfnet"]

    traffic_args = {
        "arrival_interval": args.arrival_interval,
        "holding_min": args.holding_min,
        "holding_max": args.holding_max,
        "deadline_min": args.deadline_min,
        "deadline_max": args.deadline_max,
        "size_min_mb": args.size_min_mb,
        "size_max_mb": args.size_max_mb,
        "edge_cost_min": args.edge_cost_min,
        "edge_cost_max": args.edge_cost_max,
        "num_slots": args.num_slots,
        "num_servers": args.num_servers,
        "slot_bw_hz": args.slot_bw_hz,
        "guard_band_fs": args.guard_band_fs,
        "modulation_profile": args.modulation_profile,
    }

    method_labels = {
        "Agent-C-DQN + Agent-R-DQN": "1.Separate C+R",
        "Agent-C-DQN + KSP-BF": "2.Agent-C + KSP-BF",
        "Compute-Greedy + Agent-R-DQN": "3.Compute-Greedy + Agent-R-DQN",
        "Compute-Greedy + KSP-BF": "4.Compute-Greedy + KSP-BF",
        "Spectrum-Greedy + KSP-BF": "5.Spectrum-Greedy + KSP-BF",
        "Random-valid-C + KSP-BF": "6.Random-valid-C + KSP-BF",
    }

    print("\n" + "=" * 120)
    print("Multi-Topology Evaluation")
    print(f"Seeds: {seeds} | Episodes/seed: {args.episodes} | Requests/episode: {args.requests_per_episode}")
    print("=" * 120)

    for topo in topologies:
        print(f"\n{'='*120}")
        print(f"Topology: {topo}")
        print(f"{'='*120}")

        all_results = evaluate_topology(
            topo, agent_c, agent_r, seeds, args.episodes, args.requests_per_episode, traffic_args
        )

        header = (
            f"{'Method':<30} {'BlkRate':>8} {'SuccRate':>9} {'AvgRwd':>8} "
            f"{'AvgFS':>7} {'Path(km)':>9}"
        )
        print(header)
        print("-" * 120)

        for method in METHODS:
            label = method_labels[method]
            agg = _aggregate(all_results[method])
            print(
                f"{label:<30} {agg['blocking']:>8.3f} {agg['success']:>9.3f} "
                f"{agg['avg_reward']:>8.3f} {agg['avg_fs']:>7.2f} {agg['avg_path']:>9.2f}"
            )
        print("-" * 120)

        print("\nModulation Distribution")
        print("-" * 120)
        for method in METHODS:
            label = method_labels[method]
            agg = _aggregate(all_results[method])
            mod_str = _fmt_mods(agg["mods"])
            print(f"{label:<30} {mod_str}")

        print("\nFailure Reasons")
        print("-" * 120)
        for method in METHODS:
            label = method_labels[method]
            agg = _aggregate(all_results[method])
            reason_str = _fmt_reasons(agg["reasons"])
            print(f"{label:<30} {reason_str}")

    print("\n" + "=" * 120)
    print("Evaluation complete.")
    print("=" * 120)


if __name__ == "__main__":
    main()
