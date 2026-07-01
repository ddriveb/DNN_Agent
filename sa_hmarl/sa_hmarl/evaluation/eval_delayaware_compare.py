"""Compare default/delay-aware/state-aware Agent-C + baseline C methods.

All with BC-PPO-R backend, using identical evaluation protocol as
`eval_offloading_baselines.py` to ensure consistent metrics.

Usage:
    PYTHONPATH=sa_hmarl python -m sa_hmarl.evaluation.eval_delayaware_compare
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import (
    compute_agent_c_reward,
    compute_agent_c_reward_delay_aware,
    generate_requests,
    make_env,
)
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
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
    agent.policy_net.eval()
    return agent


def evaluate_method(
    method_name: str,
    agent_c: Optional[PPOAgentC],
    agent_r: PPOAgentR,
    episodes_data: List[List],
    env_prototype,
    args,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, Any]:
    """Evaluate any C method using the same protocol as eval_offloading_baselines.py."""
    num_servers = len(env_prototype.mec.servers)
    topology = env_prototype.net.topology
    num_slots = env_prototype.net.num_slots
    slot_bw_hz = env_prototype.fs_calc.slot_bw_hz
    guard_band_fs = env_prototype.fs_calc.guard_band_fs
    server_nodes = [s.node_id for s in env_prototype.mec.servers]
    capacities = [s.compute_capacity for s in env_prototype.mec.servers]

    total = 0
    blocked = 0
    success = 0
    total_reward = 0.0
    total_delay = 0.0
    total_fs = 0.0
    total_waste = 0.0
    total_path = 0.0
    deadline_met = 0

    mod_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    server_success_counter = Counter()
    server_overload_counter = Counter()
    reason_counter = Counter()

    server_selected_count = np.zeros(num_servers, dtype=int)

    for requests in episodes_data:
        env = make_env(
            topology=topology,
            num_slots=num_slots,
            num_servers=num_servers,
            seed=42,
            slot_bw_hz=slot_bw_hz,
            guard_band_fs=guard_band_fs,
            modulation_profile=args.modulation_profile,
            server_nodes=server_nodes,
            capacities=capacities,
        )
        env.reset(requests)

        # Per-episode WO state reset
        server_selected_count_ep = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]

            if method_name == "ppo":
                action_idx_c = agent_c.select_action(obs_c, deterministic=True)
            else:
                action_idx_c = select_offloading_action(
                    method_name, env, req, obs_c, mask_c,
                    rng=rng,
                    server_selected_count=server_selected_count_ep,
                )

            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, num_servers
            )
            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count_ep[server_id] += 1
            server_selected_count[server_id] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = agent_r.select_action(obs_r, deterministic=True)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), env.max_blocks)

            _, _, _, info = env.step(action_c, action_r)
            total += 1

            server = env.mec.servers[server_id]
            # Use default reward for consistent comparison across all methods
            reward = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            total_reward += reward

            if info.get("success", False):
                success += 1
                server_success_counter[f"s{server_id}"] += 1
                delay = info.get("delay_ms", 0.0)
                total_delay += delay
                if delay <= req.deadline_ms:
                    deadline_met += 1
                total_fs += info.get("num_slots", 0)
                total_waste += info.get("block_waste", 0.0)
                total_path += info.get("path_dist_km", 0.0)
                mod_counter[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1
                if reason in ("server_overload", "server_saturated"):
                    server_overload_counter[f"s{server_id}"] += 1

    n = max(total, 1)
    s = max(success, 1)
    return {
        "label": method_name,
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "deadline_sat_rate": deadline_met / n,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "server_overload_ratio": reason_counter.get("server_overload", 0) / max(blocked, 1),
        "no_block_ratio": reason_counter.get("no_suitable_block", 0) / max(blocked, 1),
        "mod_counter": dict(mod_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "server_success_counter": dict(server_success_counter),
        "server_overload_counter": dict(server_overload_counter),
        "reason_counter": dict(reason_counter),
    }


def _fmt_pct(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in counter.most_common())


def _agg(results):
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
        "server_overload_ratio", "no_block_ratio",
    ]
    out = {}
    for k in scalar_keys:
        vals = [r[k] for r in results]
        out[k] = float(np.mean(vals))
        out[f"{k}_std"] = float(np.std(vals))
    for ck in ["mod_counter", "split_counter", "server_counter",
               "server_success_counter", "server_overload_counter", "reason_counter"]:
        c = Counter()
        for r in results:
            c.update(r[ck])
        out[ck] = dict(c)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=60)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", type=str, default="default3",
                        choices=["default3", "complex5", "complex5_v2"])
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--default_c_ckpt", type=str,
                        default="sa_hmarl/checkpoints/agent_c_frozen_r_snap24_reach_best.pt")
    parser.add_argument("--delayaware_v1_ckpt", type=str,
                        default="sa_hmarl/checkpoints/agent_c_delayaware_snap24_best.pt")
    parser.add_argument("--delayaware_v2_ckpt", type=str,
                        default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--stateaware_ckpt", type=str, default=None)
    parser.add_argument("--bc_r_ckpt", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--output_json", type=str,
                        default="sa_hmarl/experiments/delayaware_agent_c_snap24_consistent.json")
    parser.add_argument("--output_md", type=str,
                        default="sa_hmarl/experiments/delayaware_agent_c_snap24_consistent.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    # Create env prototype with EXACT same parameters as eval_offloading_baselines.py
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )
    print(f"Environment: topology={env_proto.net.topology} slots={env_proto.net.num_slots} "
          f"servers={len(env_proto.mec.servers)}")
    print(f"Server nodes: {[s.node_id for s in env_proto.mec.servers]}")
    print(f"Capacities: {[s.compute_capacity for s in env_proto.mec.servers]}")

    print(f"\nLoading BC-PPO-R from {args.bc_r_ckpt}")
    agent_r = _load_ppo_r(args.bc_r_ckpt, mod_reg, args.device)

    # Load Agent-C models
    agent_c_models = {}
    if args.default_c_ckpt:
        print(f"Loading default Agent-C from {args.default_c_ckpt}")
        agent_c_models["Original Agent-C"] = ("ppo", _load_ppo_c(args.default_c_ckpt, args.device))
    if args.delayaware_v1_ckpt and Path(args.delayaware_v1_ckpt).exists():
        print(f"Loading Delay-Aware v1 from {args.delayaware_v1_ckpt}")
        agent_c_models["Delay-Aware v1"] = ("ppo", _load_ppo_c(args.delayaware_v1_ckpt, args.device))
    if args.delayaware_v2_ckpt and Path(args.delayaware_v2_ckpt).exists():
        print(f"Loading Delay-Aware v2 from {args.delayaware_v2_ckpt}")
        agent_c_models["Delay-Aware v2"] = ("ppo", _load_ppo_c(args.delayaware_v2_ckpt, args.device))
    if args.stateaware_ckpt and Path(args.stateaware_ckpt).exists():
        print(f"Loading State-Aware from {args.stateaware_ckpt}")
        agent_c_models["State-Aware"] = ("ppo", _load_ppo_c(args.stateaware_ckpt, args.device))

    baseline_methods = ["greedy", "df", "rf"]

    print("\n" + "=" * 100)
    print(f"Consistent Delay-Aware Evaluation | Topology: {args.topology}")
    print(f"Seeds: {seeds}  Episodes/seed: {args.episodes}  Requests/episode: {args.requests_per_episode}")
    print(f"Slots: {args.num_slots}  Servers: {args.num_servers}  SlotBW: {args.slot_bw_hz/1e9:.2f}GHz  Guard: {args.guard_band_fs}")
    print("=" * 100)

    # Pre-generate episodes (same as eval_offloading_baselines.py)
    all_episodes = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            requests = generate_requests(
                env_proto, rng, src,
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
                num_splits=args.num_splits,
                split_profile=args.split_profile,
            )
            eps.append(requests)
        all_episodes[seed] = eps

    # Evaluate all methods
    all_results = {}

    for label, (method, agent_c) in agent_c_models.items():
        print(f"\n--- {label} ---")
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            res = evaluate_method(
                method, agent_c, agent_r,
                all_episodes[seed], env_proto, args,
                rng=rng,
            )
            seed_results.append(res)
            print(f"  Seed {seed}: blk={res['blocking_rate']:.3f} delay={res['avg_delay_ms']:.1f}ms "
                  f"deadline={res['deadline_sat_rate']:.3f} rwd={res['avg_reward']:+.3f} fs={res['avg_fs']:.2f}")
        all_results[label] = _agg(seed_results)

    for method in baseline_methods:
        label = {"greedy": "Greedy-C", "df": "DF-C", "rf": "RF-C"}[method]
        print(f"\n--- {label} ---")
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            res = evaluate_method(
                method, None, agent_r,
                all_episodes[seed], env_proto, args,
                rng=rng,
            )
            seed_results.append(res)
            print(f"  Seed {seed}: blk={res['blocking_rate']:.3f} delay={res['avg_delay_ms']:.1f}ms "
                  f"deadline={res['deadline_sat_rate']:.3f} rwd={res['avg_reward']:+.3f} fs={res['avg_fs']:.2f}")
        all_results[label] = _agg(seed_results)

    # Compute ObjectiveScore = BlockingRate + 0.5 * (AvgDelay_ms / 65.0)
    for name, agg in all_results.items():
        agg["objective_score"] = agg["blocking_rate"] + 0.5 * (agg["avg_delay_ms"] / 65.0)

    # Print aggregate table
    print("\n" + "=" * 100)
    print("AGGREGATE RESULTS (Consistent Evaluation)")
    print("=" * 100)
    header = (f"{'Method':20s} {'BlkRate':>8s} {'SuccRate':>9s} {'AvgRwd':>8s} "
              f"{'AvgDelay':>9s} {'DeadSat':>8s} {'AvgFS':>7s} {'Waste':>7s} "
              f"{'PathKm':>7s} {'SrvOver':>8s} {'NoBlock':>8s} {'ObjScore':>9s}")
    print(header)
    print("-" * 100)

    for name, agg in all_results.items():
        print(f"{name:20s} {agg['blocking_rate']:8.3f} {agg['success_rate']:9.3f} "
              f"{agg['avg_reward']:8.3f} {agg['avg_delay_ms']:9.1f}ms {agg['deadline_sat_rate']:8.3f} "
              f"{agg['avg_fs']:7.2f} {agg['avg_waste']:7.3f} {agg['avg_path_len_km']:7.1f} "
              f"{agg['server_overload_ratio']:8.3f} {agg['no_block_ratio']:8.3f} {agg['objective_score']:9.4f}")

    print("\n" + "-" * 100)
    print("OBJECTIVE SCORE RANKING (lower is better)")
    print("-" * 100)
    ranked = sorted(all_results.items(), key=lambda x: x[1]["objective_score"])
    for i, (name, agg) in enumerate(ranked, 1):
        print(f"  {i}. {name:20s} ObjScore={agg['objective_score']:.4f}  "
              f"(blk={agg['blocking_rate']:.3f}, delay={agg['avg_delay_ms']:.1f}ms)")

    print("\n" + "-" * 100)
    print("SPLIT DISTRIBUTION")
    print("-" * 100)
    for name, agg in all_results.items():
        print(f"  {name:20s} {_fmt_pct(Counter(agg['split_counter']))}")

    print("\n" + "-" * 100)
    print("MODULATION DISTRIBUTION")
    print("-" * 100)
    for name, agg in all_results.items():
        print(f"  {name:20s} {_fmt_pct(Counter(agg['mod_counter']))}")

    print("\n" + "-" * 100)
    print("FAILURE REASONS")
    print("-" * 100)
    for name, agg in all_results.items():
        print(f"  {name:20s} {_fmt_pct(Counter(agg['reason_counter']))}")

    # Save JSON
    if args.output_json:
        out_path = Path(args.output_json)
        clean = {}
        for name, agg in all_results.items():
            clean[name] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in agg.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(clean, open(str(out_path), "w"), indent=2)
        print(f"\nJSON saved to {out_path}")

    # Save Markdown
    if args.output_md:
        md_path = Path(args.output_md)
        lines = ["# Consistent Delay-Aware Agent-C Evaluation\n\n"]
        lines.append(f"**Topology:** {args.topology}  \n")
        lines.append(f"**Slots:** {args.num_slots}  **Servers:** {args.num_servers}  \n")
        lines.append(f"**Seeds:** {args.seeds}  **Episodes/seed:** {args.episodes}  **Requests/episode:** {args.requests_per_episode}  \n\n")
        lines.append("| Method | BlkRate | SuccRate | AvgRwd | AvgDelay | DeadSat | AvgFS | Waste | PathKm | SrvOver | ObjScore |\n")
        lines.append("|--------|---------|----------|--------|----------|---------|-------|-------|--------|---------|----------|\n")
        for name, agg in all_results.items():
            lines.append(
                f"| {name} | {agg['blocking_rate']:.3f} | {agg['success_rate']:.3f} | "
                f"{agg['avg_reward']:+.3f} | {agg['avg_delay_ms']:.1f}ms | {agg['deadline_sat_rate']:.3f} | "
                f"{agg['avg_fs']:.2f} | {agg['avg_waste']:.3f} | {agg['avg_path_len_km']:.1f} | {agg['server_overload_ratio']:.3f} | {agg['objective_score']:.4f} |\n"
            )
        lines.append("\n**Objective Score Ranking (lower is better):**\n\n")
        ranked = sorted(all_results.items(), key=lambda x: x[1]["objective_score"])
        for i, (name, agg) in enumerate(ranked, 1):
            lines.append(f"{i}. **{name}**: {agg['objective_score']:.4f} "
                         f"(blk={agg['blocking_rate']:.3f}, delay={agg['avg_delay_ms']:.1f}ms)\n")
        lines.append("\n")
        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("".join(lines))
        print(f"Markdown saved to {md_path}")


if __name__ == "__main__":
    main()
