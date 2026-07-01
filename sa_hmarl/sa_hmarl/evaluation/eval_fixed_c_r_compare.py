"""Fixed-Agent-C R-backend comparison.

Evaluates multiple R backends (PPO-R, Dual-Critic R, DeepRMSA, KSP-BF) under a
single fixed Agent-C policy. For each request, Agent-C selects split/server once
(on a reference env), and the same action_c is applied to all independent env
copies. Each R backend then selects its own action_r on its own copy, ensuring
fair comparison without spectrum-state contamination.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_fixed_c_r_compare
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
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Model loaders (copied from eval_unified)
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    # Read activation from checkpoint args (default to tanh for old checkpoints)
    ckpt_args = ckpt.get("args", {})
    activation = ckpt_args.get("agent_c_activation", "tanh") if isinstance(ckpt_args, dict) else "tanh"
    feature_mode = ckpt_args.get("agent_c_feature_mode", "default") if isinstance(ckpt_args, dict) else "default"
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        activation=activation,
        feature_mode=feature_mode,
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
        raise ValueError(
            "DeepRMSA checkpoint topology mismatch: "
            f"ckpt has num_nodes={ckpt_nodes}, env has NUM_NODES={env.net.NUM_NODES}."
        )
    ckpt_slots = ckpt.get("num_slots")
    if ckpt_slots is not None and ckpt_slots != env.net.num_slots:
        raise ValueError(
            "DeepRMSA checkpoint slot mismatch: "
            f"ckpt has num_slots={ckpt_slots}, env has num_slots={env.net.num_slots}."
        )
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
# Action selection helpers
# ---------------------------------------------------------------------------

def _select_c_ppo(agent, obs_c) -> Optional[int]:
    return agent.select_action(obs_c, deterministic=True)


def _select_r_ppo(agent, obs_r) -> Optional[int]:
    return agent.select_action(obs_r, deterministic=True)


def _select_r_deep(agent, obs_r) -> Optional[int]:
    return agent.select_action(obs_r)


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

def evaluate_fixed_c(
    agent_c: PPOAgentC,
    r_backends: List[Dict[str, Any]],
    episodes_data: List[List],
    env_prototype,
    waste_coef: float = 0.8,
    modulation_profile: str = "default",
) -> Dict[str, Dict]:
    """Run fixed-C evaluation over all episodes.

    Args:
        agent_c: Fixed Agent-C (PPO).
        r_backends: List of dicts with keys 'name', 'agent', 'type'.
        episodes_data: List of request lists (one per episode).
        env_prototype: A prototype env used for topology constants.
        waste_coef: Waste penalty coefficient.

    Returns:
        Dict mapping backend name -> aggregated metrics dict.
    """
    num_backends = len(r_backends)
    num_servers = len(env_prototype.mec.servers)
    topology = env_prototype.net.topology
    num_slots = env_prototype.net.num_slots
    slot_bw_hz = env_prototype.fs_calc.slot_bw_hz
    guard_band_fs = env_prototype.fs_calc.guard_band_fs
    server_nodes = [s.node_id for s in env_prototype.mec.servers]
    capacities = [s.compute_capacity for s in env_prototype.mec.servers]

    # Per-backend accumulators
    totals = [0] * num_backends
    blockeds = [0] * num_backends
    successes = [0] * num_backends
    total_rewards = [0.0] * num_backends
    total_delays = [0.0] * num_backends
    total_fs = [0.0] * num_backends
    total_waste = [0.0] * num_backends
    total_path = [0.0] * num_backends
    deadline_mets = [0] * num_backends

    mod_counters = [Counter() for _ in range(num_backends)]
    split_counters = [Counter() for _ in range(num_backends)]
    server_counters = [Counter() for _ in range(num_backends)]
    server_success_counters = [Counter() for _ in range(num_backends)]
    server_overload_counters = [Counter() for _ in range(num_backends)]
    reason_counters = [Counter() for _ in range(num_backends)]

    for requests in episodes_data:
        # Create independent env copies (one per R backend)
        envs = []
        for _ in range(num_backends):
            env = make_env(
                topology=topology,
                num_slots=num_slots,
                num_servers=num_servers,
                seed=42,  # deterministic init; spectrum diverges per copy
                slot_bw_hz=slot_bw_hz,
                guard_band_fs=guard_band_fs,
                modulation_profile=modulation_profile,
                server_nodes=server_nodes,
                capacities=capacities,
            )
            env.reset(requests)
            envs.append(env)

        # Reference env for Agent-C decisions (index 0)
        ref_env = envs[0]

        for req in requests:
            # --- Agent-C decision on reference env ---
            obs_c = build_agent_c_observation(ref_env, req)
            action_idx_c = _select_c_ppo(agent_c, obs_c)
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, num_servers
            )
            split_id, server_id = action_c

            # --- Each R backend acts on its own env copy ---
            for i, (env, backend) in enumerate(zip(envs, r_backends)):
                split_counters[i][f"split{split_id}"] += 1
                server_counters[i][f"s{server_id}"] += 1

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                r_type = backend["type"]

                if r_type == "ppo":
                    action_idx_r = _select_r_ppo(backend["agent"], obs_r)
                elif r_type == "deep_rmsa":
                    action_idx_r = _select_r_deep(backend["agent"], obs_r)
                elif r_type == "ksp_bf":
                    action_idx_r = ksp_bf_action(obs_r)
                else:
                    raise ValueError(f"Unknown r_type: {r_type}")

                if action_idx_r is None:
                    action_r = (0, 0, 0)
                else:
                    action_r = decode_agent_r_action(
                        action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                    )

                _, _, _, info = env.step(action_c, action_r)
                totals[i] += 1

                server = env.mec.servers[server_id]
                reward = compute_agent_c_reward(
                    info, req.deadline_ms, waste_coef, server.utilization
                )
                total_rewards[i] += reward

                if info.get("success", False):
                    successes[i] += 1
                    server_success_counters[i][f"s{server_id}"] += 1
                    delay = info.get("delay_ms", 0.0)
                    total_delays[i] += delay
                    if delay <= req.deadline_ms:
                        deadline_mets[i] += 1
                    total_fs[i] += info.get("num_slots", 0)
                    total_waste[i] += info.get("block_waste", 0.0)
                    total_path[i] += info.get("path_dist_km", 0.0)
                    mod_counters[i][info.get("modulation", "unknown")] += 1
                else:
                    blockeds[i] += 1
                    reason = info.get("reason", "unknown")
                    reason_counters[i][reason] += 1
                    if reason in ("server_overload", "server_saturated"):
                        server_overload_counters[i][f"s{server_id}"] += 1

    # Assemble results
    results = {}
    for i, backend in enumerate(r_backends):
        name = backend["name"]
        n = max(totals[i], 1)
        s = max(successes[i], 1)
        results[name] = {
            "total": totals[i],
            "blocked": blockeds[i],
            "success": successes[i],
            "blocking_rate": blockeds[i] / n,
            "success_rate": successes[i] / n,
            "avg_reward": total_rewards[i] / n,
            "avg_delay_ms": total_delays[i] / s,
            "deadline_sat_rate": deadline_mets[i] / n,
            "avg_fs": total_fs[i] / s,
            "avg_waste": total_waste[i] / s,
            "avg_path_len_km": total_path[i] / s,
            "server_overload_ratio": reason_counters[i].get("server_overload", 0) / max(blockeds[i], 1),
            "no_suitable_block_ratio": reason_counters[i].get("no_suitable_block", 0) / max(blockeds[i], 1),
            "mod_counter": dict(mod_counters[i]),
            "split_counter": dict(split_counters[i]),
            "server_counter": dict(server_counters[i]),
            "server_success_counter": dict(server_success_counters[i]),
            "server_overload_counter": dict(server_overload_counters[i]),
            "reason_counter": dict(reason_counters[i]),
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Fixed-Agent-C comparison of R backends"
    )
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
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_frozen_r_snap24_reach_best.pt")
    parser.add_argument("--ppo_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/joint_mappo_v2_cshape_metro24_fs002_r_best.pt")
    parser.add_argument("--dual_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/dualcritic_snap24_reach_se_A_r_best.pt")
    parser.add_argument("--deep_rmsa_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/deep_rmsa_snap24_reach_mixed.pt")
    parser.add_argument("--output_json", type=str,
                        default="sa_hmarl/experiments/fixed_c_r_compare_snap24.json")
    parser.add_argument("--output_md", type=str,
                        default="sa_hmarl/experiments/fixed_c_r_compare_snap24.md")
    args = parser.parse_args()

    ckpt_dir = Path(__file__).resolve().parents[2] / "checkpoints"
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    # Load fixed Agent-C
    print(f"Loading fixed Agent-C from {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)

    # Load R backends
    # Need a prototype env for DeepRMSA topology validation
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )

    r_backends = []

    print(f"Loading PPO-R from {args.ppo_r_checkpoint}")
    r_backends.append({
        "name": "PPO-R",
        "type": "ppo",
        "agent": _load_ppo_r(args.ppo_r_checkpoint, mod_reg, args.device),
    })

    print(f"Loading Dual-Critic v2 A R from {args.dual_r_checkpoint}")
    r_backends.append({
        "name": "Dual-Critic v2 A R",
        "type": "ppo",
        "agent": _load_ppo_r(args.dual_r_checkpoint, mod_reg, args.device),
    })

    print(f"Loading DeepRMSA from {args.deep_rmsa_checkpoint}")
    r_backends.append({
        "name": "DeepRMSA",
        "type": "deep_rmsa",
        "agent": _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device),
    })

    r_backends.append({
        "name": "KSP-BF",
        "type": "ksp_bf",
        "agent": None,
    })

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    print("=" * 100)
    print(f"Fixed-Agent-C R-Backend Comparison | Topology: {args.topology}")
    print(f"Fixed C: {args.agent_c_checkpoint}")
    print(f"Seeds: {seeds}  Episodes/seed: {args.episodes}  Requests/episode: {args.requests_per_episode}")
    print("=" * 100)

    # Pre-generate all episodes for all seeds
    all_episodes: Dict[int, List[List]] = {}
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
            )
            eps.append(requests)
        all_episodes[seed] = eps

    # Evaluate per seed and aggregate
    seed_results: Dict[int, Dict[str, Dict]] = {}
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        res = evaluate_fixed_c(
            agent_c, r_backends, all_episodes[seed], env_proto, args.waste_coef,
            args.modulation_profile,
        )
        seed_results[seed] = res
        for name, metrics in res.items():
            print(f"  {name:25s} blk={metrics['blocking_rate']:.3f} rwd={metrics['avg_reward']:+.3f} "
                  f"delay={metrics['avg_delay_ms']:.1f}ms fs={metrics['avg_fs']:.2f}")

    # Aggregate across seeds
    backend_names = [b["name"] for b in r_backends]
    agg_results: Dict[str, Dict] = {}
    for name in backend_names:
        scalar_keys = [
            "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
            "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
            "server_overload_ratio", "no_suitable_block_ratio",
        ]
        agg = {}
        for k in scalar_keys:
            vals = [seed_results[s][name][k] for s in seeds]
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

        # Aggregate counters
        for counter_key in ["mod_counter", "split_counter", "server_counter",
                            "server_success_counter", "server_overload_counter", "reason_counter"]:
            c = Counter()
            for s in seeds:
                c.update(seed_results[s][name][counter_key])
            agg[counter_key] = dict(c)

        agg_results[name] = agg

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    report_lines = []
    def _print(line=""):
        print(line)
        report_lines.append(line)

    _print("=" * 100)
    _print("AGGREGATED RESULTS (mean across seeds)")
    _print("=" * 100)

    header = (f"{'Method':25s} {'BlkRate':>8s} {'SuccRate':>9s} {'AvgRwd':>8s} "
              f"{'AvgDelay':>9s} {'DeadSat':>8s} {'AvgFS':>7s} {'Waste':>7s} "
              f"{'PathKm':>7s} {'SrvOver':>8s} {'NoBlock':>8s}")
    _print(header)
    _print("-" * 100)

    for name, agg in agg_results.items():
        _print(
            f"{name:25s} "
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

    _print()
    _print("-" * 100)
    _print("MODULATION DISTRIBUTION")
    _print("-" * 100)
    for name, agg in agg_results.items():
        _print(f"  {name:25s}  {_fmt_pct(Counter(agg['mod_counter']))}")

    _print()
    _print("-" * 100)
    _print("SPLIT DISTRIBUTION")
    _print("-" * 100)
    for name, agg in agg_results.items():
        _print(f"  {name:25s}  {_fmt_pct(Counter(agg['split_counter']))}")

    _print()
    _print("-" * 100)
    _print("SERVER DISTRIBUTION")
    _print("-" * 100)
    for name, agg in agg_results.items():
        _print(f"  {name:25s}  {_fmt_pct(Counter(agg['server_counter']))}")

    _print()
    _print("-" * 100)
    _print("SERVER SUCCESS DISTRIBUTION")
    _print("-" * 100)
    for name, agg in agg_results.items():
        _print(f"  {name:25s}  {_fmt_pct(Counter(agg['server_success_counter']))}")

    _print()
    _print("-" * 100)
    _print("SERVER OVERLOAD DISTRIBUTION")
    _print("-" * 100)
    for name, agg in agg_results.items():
        _print(f"  {name:25s}  {_fmt_cnt(Counter(agg['server_overload_counter']))}")

    _print()
    _print("-" * 100)
    _print("FAILURE REASONS")
    _print("-" * 100)
    for name, agg in agg_results.items():
        _print(f"  {name:25s}  {_fmt_cnt(Counter(agg['reason_counter']))}")

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------
    if args.output_json:
        out_path = Path(args.output_json)
        clean = {}
        for name, agg in agg_results.items():
            clean[name] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                           for k, v in agg.items()}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(clean, open(str(out_path), "w"), indent=2)
        _print(f"\nResults written to {out_path}")

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------
    if args.output_md:
        md_path = Path(args.output_md)
        md_lines = ["# Fixed-Agent-C R-Backend Comparison\n",
                    f"**Topology:** {args.topology}  \n",
                    f"**Fixed Agent-C:** `{args.agent_c_checkpoint}`  \n",
                    f"**Seeds:** {args.seeds}  \n",
                    f"**Episodes/seed:** {args.episodes}  \n",
                    f"**Requests/episode:** {args.requests_per_episode}  \n\n"]

        md_lines.append("## Aggregated Results\n\n")
        md_lines.append("| Method | BlkRate | SuccRate | AvgRwd | AvgDelay | AvgFS | Waste | PathKm |\n")
        md_lines.append("|--------|---------|----------|--------|----------|-------|-------|--------|\n")
        for name, agg in agg_results.items():
            md_lines.append(
                f"| {name} | {agg['blocking_rate']:.3f} | {agg['success_rate']:.3f} | "
                f"{agg['avg_reward']:+.3f} | {agg['avg_delay_ms']:.1f}ms | "
                f"{agg['avg_fs']:.2f} | {agg['avg_waste']:.3f} | {agg['avg_path_len_km']:.1f} |\n"
            )

        md_lines.append("\n## Modulation Distribution\n\n")
        for name, agg in agg_results.items():
            md_lines.append(f"- **{name}**: {_fmt_pct(Counter(agg['mod_counter']))}\n")

        md_lines.append("\n## Split Distribution\n\n")
        for name, agg in agg_results.items():
            md_lines.append(f"- **{name}**: {_fmt_pct(Counter(agg['split_counter']))}\n")

        md_lines.append("\n## Server Distribution\n\n")
        for name, agg in agg_results.items():
            md_lines.append(f"- **{name}**: {_fmt_pct(Counter(agg['server_counter']))}\n")

        md_lines.append("\n## Failure Reasons\n\n")
        for name, agg in agg_results.items():
            md_lines.append(f"- **{name}**: {_fmt_cnt(Counter(agg['reason_counter']))}\n")

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("".join(md_lines))
        _print(f"Markdown report written to {md_path}")


if __name__ == "__main__":
    main()
