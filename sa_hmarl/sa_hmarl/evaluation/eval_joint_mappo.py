"""Evaluate MAPPO-style Agent-C/Agent-R checkpoints.

Execution uses only decentralized PPO actors; the centralized value critic is
not loaded for evaluation.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from collections import Counter
from typing import Dict, List

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.eval_joint_multitopo import _load_agent_c, _load_agent_r
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    return agent


def _compute_greedy_action(obs_c, valid, key):
    if len(valid) == 0:
        return None
    best = int(valid[0])
    best_val = obs_c["candidate_features"][best][key]
    if best_val == float("inf"):
        best_val = 1e9
    for idx in valid[1:]:
        val = obs_c["candidate_features"][int(idx)][key]
        if val == float("inf"):
            val = 1e9
        if val < best_val:
            best = int(idx)
            best_val = val
    return best


def _select_c(agent, obs_c, method_name, valid):
    if method_name.startswith("MAPPO"):
        return agent.select_action(obs_c, deterministic=True)
    if method_name.startswith("Separate"):
        return agent.select_action(obs_c, epsilon=0.0)
    if method_name.startswith("Compute-Greedy"):
        return _compute_greedy_action(obs_c, valid, "edge_compute_ms")
    raise ValueError(f"Unknown method: {method_name}")


def _select_r(agent, obs_r, method_name):
    if "PPO-R" in method_name:
        return agent.select_action(obs_r, deterministic=True)
    if "Agent-R-DQN" in method_name or method_name.startswith("Separate"):
        return agent.select_action(obs_r, epsilon=0.0)
    return ksp_bf_action(obs_r)


def evaluate_method(env, agent_c, agent_r, requests, method_name, waste_coef=0.8):
    env.reset(requests)
    blocked = 0
    successes = 0
    total_reward = 0.0
    total_fs = 0.0
    total_path = 0.0
    mod_counter = Counter()
    reason_counter = Counter()

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        mask = obs_c["agent_c_mask"]
        valid = np.where(mask)[0]
        action_idx_c = _select_c(agent_c, obs_c, method_name, valid)
        action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
            action_idx_c, len(env.mec.servers)
        )
        split_id, server_id = action_c

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        action_idx_r = _select_r(agent_r, obs_r, method_name)
        action_r = (0, 0, 0) if action_idx_r is None else decode_agent_r_action(
            action_idx_r, len(obs_r["mod_names"]), env.max_blocks
        )

        _, _, _, info = env.step(action_c, action_r)
        server = env.mec.servers[server_id]
        total_reward += compute_agent_c_reward(
            info, req.deadline_ms, waste_coef, server.utilization
        )
        if info.get("success", False):
            successes += 1
            total_fs += info.get("num_slots", 0)
            total_path += info.get("path_dist_km", 0.0)
            mod_counter[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1
            reason_counter[info.get("reason", "unknown")] += 1

    n = len(requests)
    return {
        "blocking_rate": blocked / n,
        "success_rate": successes / n,
        "avg_reward": total_reward / n,
        "avg_fs": total_fs / successes if successes else 0.0,
        "avg_path_len_km": total_path / successes if successes else 0.0,
        "mod_counter": mod_counter,
        "reason_counter": reason_counter,
    }


def _aggregate(results):
    keys = ("blocking_rate", "success_rate", "avg_reward", "avg_fs", "avg_path_len_km")
    out = {k: float(np.mean([r[k] for r in results])) for k in keys}
    mods = Counter()
    reasons = Counter()
    for r in results:
        mods.update(r["mod_counter"])
        reasons.update(r["reason_counter"])
    out["mod_counter"] = mods
    out["reason_counter"] = reasons
    return out


def _fmt_counter(counter):
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in counter.most_common())


def main():
    parser = argparse.ArgumentParser(description="Evaluate MAPPO-style joint PPO")
    parser.add_argument("--topologies", type=str, default="net1")
    parser.add_argument("--checkpoint_mappo_c", type=str, default=None)
    parser.add_argument("--checkpoint_mappo_r", type=str, default=None)
    parser.add_argument("--checkpoint_sep_c", type=str, default=None)
    parser.add_argument("--checkpoint_sep_r", type=str, default=None)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--episodes", type=int, default=5)
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
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=32)
    parser.add_argument("--num_servers", type=int, default=2)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    mappo_c_path = args.checkpoint_mappo_c or str(ckpt_dir / "joint_mappo_c_best.pt")
    mappo_r_path = args.checkpoint_mappo_r or str(ckpt_dir / "joint_mappo_r_best.pt")
    sep_c_path = args.checkpoint_sep_c or str(ckpt_dir / "agent_c_multitopo_best.pt")
    sep_r_path = args.checkpoint_sep_r or str(ckpt_dir / "agent_r_multitopo.pt")

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    mappo_c = _load_ppo_c(mappo_c_path, args.device)
    mappo_r = _load_ppo_r(mappo_r_path, mod_reg, args.device)
    sep_c = _load_agent_c(sep_c_path, args.device)
    sep_r = _load_agent_r(sep_r_path, mod_reg, args.device)

    methods = [
        ("MAPPO-C + PPO-R", mappo_c, mappo_r),
        ("Compute-Greedy + PPO-R", mappo_c, mappo_r),
        ("Separate-C + Separate-R", sep_c, sep_r),
        ("Compute-Greedy + Agent-R-DQN", sep_c, sep_r),
        ("Compute-Greedy + KSP-BF", sep_c, sep_r),
    ]
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    topologies = [t.strip() for t in args.topologies.split(",") if t.strip()]

    for topo in topologies:
        env = make_env(
            topology=topo,
            num_servers=args.num_servers,
            seed=42,
            num_slots=args.num_slots,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
        )
        all_results: Dict[str, List[Dict]] = {name: [] for name, _, _ in methods}
        for seed in seeds:
            rng = np.random.RandomState(seed)
            for _ in range(args.episodes):
                src = rng.randint(0, env.net.NUM_NODES)
                requests = generate_requests(
                    env,
                    rng,
                    src,
                    args.requests_per_episode,
                    arrival_interval=args.arrival_interval,
                    holding_min=args.holding_min,
                    holding_max=args.holding_max,
                    deadline_min=args.deadline_min,
                    deadline_max=args.deadline_max,
                    size_min_mb=args.size_min_mb,
                    size_max_mb=args.size_max_mb,
                    edge_cost_min=args.edge_cost_min,
                    edge_cost_max=args.edge_cost_max,
                )
                for name, ac, ar in methods:
                    all_results[name].append(
                        evaluate_method(env, ac, ar, requests, name, args.waste_coef)
                    )

        print("\n" + "=" * 104)
        print(f"MAPPO Evaluation | Topology: {topo}")
        print("=" * 104)
        print(f"{'Method':32s} {'BlkRate':>8s} {'SuccRate':>9s} {'AvgRwd':>8s} {'AvgFS':>7s} {'Path(km)':>9s}")
        print("-" * 104)
        aggregates = {}
        for name in all_results:
            agg = _aggregate(all_results[name])
            aggregates[name] = agg
            print(
                f"{name:32s} {agg['blocking_rate']:8.3f} {agg['success_rate']:9.3f} "
                f"{agg['avg_reward']:8.3f} {agg['avg_fs']:7.2f} {agg['avg_path_len_km']:9.2f}"
            )
        print("\nModulation Distribution")
        print("-" * 104)
        for name, agg in aggregates.items():
            print(f"{name:32s} {_fmt_counter(agg['mod_counter'])}")
        print("\nFailure Reasons")
        print("-" * 104)
        for name, agg in aggregates.items():
            print(f"{name:32s} {_fmt_counter(agg['reason_counter'])}")


if __name__ == "__main__":
    main()
