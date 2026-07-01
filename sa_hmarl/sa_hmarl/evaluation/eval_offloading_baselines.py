"""Unified evaluation of offloading baselines + R backends.

Compares 6 C baselines × 2 R backends = 12 methods.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_offloading_baselines
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

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


def _load_deep_rmsa(ckpt_path: str, env, mod_reg: ModulationRegistry,
                    device: str = "cpu") -> DeepRMSAAgent:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_nodes = ckpt.get("num_nodes")
    if ckpt_nodes is not None and ckpt_nodes != env.net.NUM_NODES:
        raise ValueError("DeepRMSA checkpoint topology mismatch")
    ckpt_slots = ckpt.get("num_slots")
    if ckpt_slots is not None and ckpt_slots != env.net.num_slots:
        raise ValueError("DeepRMSA checkpoint slot mismatch")
    agent = DeepRMSAAgent(
        num_nodes=env.net.NUM_NODES,
        num_slots=env.net.num_slots,
        k_path=ckpt.get("k_path", env.k),
        m_blocks=ckpt.get("m_blocks", env.max_blocks),
        mod_registry=mod_reg,
        gamma=ckpt.get("gamma", 0.95),
        device=device,
    )
    agent.load_state_dict(ckpt)
    agent.eval()
    return agent


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in counter.most_common())


def _fmt_cnt(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in counter.most_common())


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_offloading_method(
    agent_c: Optional[PPOAgentC],
    agent_r,
    c_method: str,
    r_type: str,
    episodes_data: List[List],
    env_prototype,
    waste_coef: float = 0.8,
    modulation_profile: str = "default",
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, Any]:
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

    for requests in episodes_data:
        env = make_env(
            topology=topology,
            num_slots=num_slots,
            num_servers=num_servers,
            seed=42,
            slot_bw_hz=slot_bw_hz,
            guard_band_fs=guard_band_fs,
            modulation_profile=modulation_profile,
            server_nodes=server_nodes,
            capacities=capacities,
        )
        env.reset(requests)

        # WO state
        server_selected_count = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]

            if c_method == "ppo":
                action_idx_c = agent_c.select_action(obs_c, deterministic=True)
            else:
                action_idx_c = select_offloading_action(
                    c_method, env, req, obs_c, mask_c,
                    rng=rng,
                    server_selected_count=server_selected_count,
                )

            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, num_servers
            )
            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count[server_id] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            num_mods = len(obs_r["mod_names"])

            if r_type == "deep_rmsa":
                action_idx_r = agent_r.select_action(obs_r)
            elif r_type == "ppo":
                action_idx_r = agent_r.select_action(obs_r, deterministic=True)
            else:
                raise ValueError(f"Unknown r_type: {r_type}")

            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(action_idx_r, num_mods, env.max_blocks)

            _, _, _, info = env.step(action_c, action_r)
            total += 1

            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, waste_coef, server.utilization
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
        "no_suitable_block_ratio": reason_counter.get("no_suitable_block", 0) / max(blocked, 1),
        "mod_counter": dict(mod_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "server_success_counter": dict(server_success_counter),
        "server_overload_counter": dict(server_overload_counter),
        "reason_counter": dict(reason_counter),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

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
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", type=str, default="default3",
                        choices=["default3", "complex5", "complex5_v2"])
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="checkpoints/agent_c_frozen_r_snap24_reach_best.pt")
    parser.add_argument("--deep_rmsa_checkpoint", type=str,
                        default="checkpoints/deep_rmsa_snap24_reach_mixed.pt")
    parser.add_argument("--bc_r_checkpoint", type=str,
                        default="checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--output_json", type=str,
                        default="experiments/offloading_baselines_snap24.json")
    parser.add_argument("--output_md", type=str,
                        default="experiments/offloading_baselines_snap24.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    print(f"Loading Agent-C from {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)

    print(f"Loading DeepRMSA from {args.deep_rmsa_checkpoint}")
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )
    agent_r_deep = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

    print(f"Loading BC-PPO-R from {args.bc_r_checkpoint}")
    agent_r_bc = _load_ppo_r(args.bc_r_checkpoint, mod_reg, args.device)

    c_methods = ["ppo", "greedy", "wo", "df", "rf", "iwd"]
    r_backends = [
        ("deep_rmsa", agent_r_deep),
        ("ppo", agent_r_bc),
    ]

    print("=" * 100)
    print(f"Offloading Baselines Evaluation | Topology: {args.topology}")
    print(f"Seeds: {seeds}  Episodes/seed: {args.episodes}  Requests/episode: {args.requests_per_episode}")
    print("=" * 100)

    # Pre-generate episodes
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

    # Evaluate all combinations
    results_deep = {}
    results_bc = {}

    for c_method in c_methods:
        c_label = {"ppo": "Agent-C", "greedy": "Greedy-C", "wo": "WO-C",
                   "df": "DF-C", "rf": "RF-C", "iwd": "IWD-C"}[c_method]

        for r_type, agent_r in r_backends:
            r_label = "DeepRMSA" if r_type == "deep_rmsa" else "BC-PPO-R"
            name = f"{c_label} + {r_label}"
            print(f"\n--- {name} ---")

            seed_results = []
            for seed in seeds:
                rng = np.random.RandomState(seed)
                res = evaluate_offloading_method(
                    agent_c, agent_r,
                    c_method, r_type,
                    all_episodes[seed], env_proto, args.waste_coef,
                    args.modulation_profile,
                    rng=rng,
                )
                seed_results.append(res)
                print(f"  Seed {seed}: blk={res['blocking_rate']:.3f} rwd={res['avg_reward']:+.3f} "
                      f"delay={res['avg_delay_ms']:.1f}ms fs={res['avg_fs']:.2f}")

            # Aggregate
            agg = {}
            scalar_keys = [
                "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
                "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
                "server_overload_ratio", "no_suitable_block_ratio",
            ]
            for k in scalar_keys:
                vals = [r[k] for r in seed_results]
                agg[k] = float(np.mean(vals))
                agg[f"{k}_std"] = float(np.std(vals))

            for counter_key in ["mod_counter", "split_counter", "server_counter",
                                "server_success_counter", "server_overload_counter", "reason_counter"]:
                c = Counter()
                for r in seed_results:
                    c.update(r[counter_key])
                agg[counter_key] = dict(c)

            if r_type == "deep_rmsa":
                results_deep[c_label] = agg
            else:
                results_bc[c_label] = agg

    # Print report
    report_lines = []
    def _print(line=""):
        print(line)
        report_lines.append(line)

    def _print_table(results: Dict, title: str):
        _print("=" * 100)
        _print(title)
        _print("=" * 100)
        header = (f"{'Method':15s} {'BlkRate':>8s} {'SuccRate':>9s} {'AvgRwd':>8s} "
                  f"{'AvgDelay':>9s} {'DeadSat':>8s} {'AvgFS':>7s} {'Waste':>7s} "
                  f"{'PathKm':>7s} {'SrvOver':>8s} {'NoBlock':>8s}")
        _print(header)
        _print("-" * 100)
        for name, agg in results.items():
            _print(
                f"{name:15s} "
                f"{agg['blocking_rate']:8.3f} "
                f"{agg['success_rate']:9.3f} "
                f"{agg['avg_reward']:8.3f} "
                f"{agg['avg_delay_ms']:9.1f} "
                f"{agg['deadline_sat_rate']:8.3f} "
                f"{agg['avg_fs']:7.2f} "
                f"{agg['avg_waste']:7.3f} "
                f"{agg['avg_path_len_km']:7.1f} "
                f"{agg['server_overload_ratio']:8.3f} "
                f"{agg['no_suitable_block_ratio']:8.3f}"
            )

    def _print_dist(results: Dict, label: str):
        _print()
        _print("-" * 100)
        _print(label)
        _print("-" * 100)
        for name, agg in results.items():
            _print(f"  {name:15s}  {_fmt_pct(Counter(agg['mod_counter']))}")

    _print_table(results_deep, "TABLE 1: DeepRMSA Backend")
    _print_dist(results_deep, "MODULATION DISTRIBUTION (DeepRMSA)")

    _print_table(results_bc, "TABLE 2: BC-PPO-R Backend")
    _print_dist(results_bc, "MODULATION DISTRIBUTION (BC-PPO-R)")

    # Table 3: Agent-C improvement over best traditional baseline
    _print()
    _print("=" * 100)
    _print("TABLE 3: Agent-C vs Best Traditional Baseline")
    _print("=" * 100)
    traditional = ["Greedy-C", "WO-C", "DF-C", "RF-C", "IWD-C"]

    for r_label, results in [("DeepRMSA", results_deep), ("BC-PPO-R", results_bc)]:
        _print(f"\n--- {r_label} ---")
        best_baseline = min(traditional, key=lambda m: results[m]["blocking_rate"])
        best_blk = results[best_baseline]["blocking_rate"]
        best_reward = results[best_baseline]["avg_reward"]
        best_delay = results[best_baseline]["avg_delay_ms"]
        best_fs = results[best_baseline]["avg_fs"]

        agent_blk = results["Agent-C"]["blocking_rate"]
        agent_reward = results["Agent-C"]["avg_reward"]
        agent_delay = results["Agent-C"]["avg_delay_ms"]
        agent_fs = results["Agent-C"]["avg_fs"]

        blk_pp = best_blk - agent_blk
        blk_pct = (blk_pp / best_blk * 100) if best_blk > 0 else 0.0

        _print(f"Best baseline: {best_baseline} (blk={best_blk:.3f})")
        _print(f"Agent-C:       blk={agent_blk:.3f}  reward={agent_reward:.3f}  delay={agent_delay:.1f}ms  fs={agent_fs:.2f}")
        _print(f"Improvement:   blocking -{blk_pp:.4f} ({blk_pct:+.1f}%)  "
              f"reward {agent_reward - best_reward:+.3f}  "
              f"delay {agent_delay - best_delay:+.1f}ms  "
              f"fs {agent_fs - best_fs:+.2f}")

    # Server/split/failure details
    for r_label, results in [("DeepRMSA", results_deep), ("BC-PPO-R", results_bc)]:
        _print()
        _print("-" * 100)
        _print(f"SPLIT DISTRIBUTION ({r_label})")
        _print("-" * 100)
        for name, agg in results.items():
            _print(f"  {name:15s}  {_fmt_pct(Counter(agg['split_counter']))}")

        _print()
        _print("-" * 100)
        _print(f"FAILURE REASONS ({r_label})")
        _print("-" * 100)
        for name, agg in results.items():
            _print(f"  {name:15s}  {_fmt_cnt(Counter(agg['reason_counter']))}")

    # JSON export
    if args.output_json:
        out_path = Path(args.output_json)
        clean = {}
        for r_label, results in [("DeepRMSA", results_deep), ("BC-PPO-R", results_bc)]:
            clean[r_label] = {}
            for name, agg in results.items():
                clean[r_label][name] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                                        for k, v in agg.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(clean, open(str(out_path), "w"), indent=2)
        _print(f"\nResults written to {out_path}")

    # Markdown export
    if args.output_md:
        md_path = Path(args.output_md)
        md_lines = ["# Offloading Baselines Evaluation\n",
                    f"**Topology:** {args.topology}  \n",
                    f"**Seeds:** {args.seeds}  \n",
                    f"**Episodes/seed:** {args.episodes}  \n\n"]

        for r_label, results in [("DeepRMSA", results_deep), ("BC-PPO-R", results_bc)]:
            md_lines.append(f"## {r_label} Backend\n\n")
            md_lines.append("| Method | BlkRate | SuccRate | AvgRwd | AvgDelay | AvgFS | Waste | PathKm | SrvOver |\n")
            md_lines.append("|--------|---------|----------|--------|----------|-------|-------|--------|---------|\n")
            for name, agg in results.items():
                md_lines.append(
                    f"| {name} | {agg['blocking_rate']:.3f} | {agg['success_rate']:.3f} | "
                    f"{agg['avg_reward']:+.3f} | {agg['avg_delay_ms']:.1f}ms | "
                    f"{agg['avg_fs']:.2f} | {agg['avg_waste']:.3f} | {agg['avg_path_len_km']:.1f} | "
                    f"{agg['server_overload_ratio']:.3f} |\n"
                )
            md_lines.append("\n")

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("".join(md_lines))
        _print(f"Markdown report written to {md_path}")


if __name__ == "__main__":
    main()
