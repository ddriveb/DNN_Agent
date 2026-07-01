"""Full eval: DelayAware-Agent-C vs Original Agent-C vs baselines on best C-gap scenario."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import csv
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
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _compute_objective(blocking_rate: float, avg_delay_ms: float) -> float:
    return blocking_rate + 0.5 * (avg_delay_ms / 65.0)


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode",
        ckpt.get("agent_c_feature_mode", "default"),
    )
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
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
    agent_c: Optional[PPOAgentC],
    agent_r: PPOAgentR,
    c_method: str,
    episodes_data: List[List],
    env_prototype,
    waste_coef: float = 0.8,
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
    no_valid_c = 0

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
            modulation_profile="default",
            server_nodes=server_nodes,
            capacities=capacities,
        )
        env.reset(requests)
        server_selected_count = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]

            if c_method == "ppo":
                c_features, mask_c = agent_c.build_action_features(obs_c)
                ckpt_args = getattr(agent_c, "checkpoint_args", {})
                risk_kwargs = risk_kwargs_from_args(ckpt_args)
                mask_c = apply_agent_c_risk_mask(
                    obs_c,
                    mask_c,
                    num_slots_total=env.net.num_slots,
                    **risk_kwargs,
                )
                action_idx_c, _, _ = agent_c.select_from_features(
                    c_features, mask_c, deterministic=True
                )
            else:
                action_idx_c = select_offloading_action(
                    c_method, env, req, obs_c, mask_c,
                    rng=rng,
                    server_selected_count=server_selected_count,
                )

            if action_idx_c is None:
                no_valid_c += 1
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, num_servers)
            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count[server_id] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            num_mods = len(obs_r["mod_names"])
            action_idx_r = agent_r.select_action(obs_r, deterministic=True)

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
    blocking_rate = blocked / n
    avg_delay = total_delay / s
    return {
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocking_rate,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": avg_delay,
        "deadline_sat_rate": deadline_met / n,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "server_overload_ratio": reason_counter.get("server_overload", 0) / max(blocked, 1),
        "no_suitable_block_ratio": reason_counter.get("no_suitable_block", 0) / max(blocked, 1),
        "no_valid_c_ratio": no_valid_c / n,
        "mod_counter": dict(mod_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "server_success_counter": dict(server_success_counter),
        "server_overload_counter": dict(server_overload_counter),
        "reason_counter": dict(reason_counter),
        "objective": _compute_objective(blocking_rate, avg_delay),
    }


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
        "server_overload_ratio", "no_suitable_block_ratio", "no_valid_c_ratio",
        "objective",
    ]
    agg = {}
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
        total = sum(c.values())
        agg[f"{counter_key}_distribution"] = {k: v / total for k, v in c.items()} if total else {}
    return agg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--enhanced_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_enhanced_delayaware_cgap_best.pt")
    parser.add_argument("--out_json", type=str, default="sa_hmarl/experiments/cgap_delayaware_eval.json")
    parser.add_argument("--out_md", type=str, default="sa_hmarl/experiments/cgap_delayaware_eval.md")
    args = parser.parse_args()

    topology = "snap24_gnutella_reach"
    num_slots = 16
    num_servers = 4
    slot_bw_hz = 1.25e9
    guard_band_fs = 1
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    mod_reg = ModulationRegistry.from_profile("default")
    agent_r = _load_ppo_r("sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt", mod_reg, args.device)
    original_c = _load_ppo_c("sa_hmarl/checkpoints/agent_c_frozen_r_snap24_reach_best.pt", args.device)
    delayaware_c = _load_ppo_c("sa_hmarl/checkpoints/agent_c_delayaware_cgap_best.pt", args.device)
    enhanced_c = None
    if Path(args.enhanced_c_checkpoint).exists():
        enhanced_c = _load_ppo_c(args.enhanced_c_checkpoint, args.device)

    env_proto = make_env(
        topology=topology,
        num_slots=num_slots,
        num_servers=num_servers,
        seed=42,
        slot_bw_hz=slot_bw_hz,
        guard_band_fs=guard_band_fs,
        modulation_profile="default",
    )

    methods = {}
    if enhanced_c is not None:
        methods["Enhanced-DelayAware-Agent-C"] = (enhanced_c, "ppo")
    methods.update({
        "DelayAware-Agent-C": (delayaware_c, "ppo"),
        "Original-Agent-C": (original_c, "ppo"),
        "IWD-C": (None, "iwd"),
        "DF-C": (None, "df"),
        "RF-C": (None, "rf"),
        "Greedy-C": (None, "greedy"),
        "WO-C": (None, "wo"),
    })

    all_results = {}
    for method_name, (agent_c, c_method) in methods.items():
        print(f"\nEvaluating {method_name}...")
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            episodes_data = []
            for _ in range(args.episodes):
                src = rng.randint(0, env_proto.net.NUM_NODES)
                requests = generate_requests(
                    env_proto, rng, src, args.requests_per_episode,
                    arrival_interval=0.20,
                    holding_min=4.0, holding_max=10.0,
                    deadline_min=30.0, deadline_max=100.0,
                    size_min_mb=5.0, size_max_mb=30.0,
                    edge_cost_min=0.5, edge_cost_max=15.0,
                    num_splits=5, split_profile="complex5_v2_lite",
                )
                episodes_data.append(requests)
            res = evaluate_method(
                agent_c, agent_r, c_method, episodes_data, env_proto,
                waste_coef=0.8, rng=np.random.RandomState(seed),
            )
            seed_results.append(res)
            print(f"  seed {seed}: blk={res['blocking_rate']:.4f} delay={res['avg_delay_ms']:.2f}ms obj={res['objective']:.4f}")
        agg = aggregate_seed_results(seed_results)
        all_results[method_name] = agg
        print(f"  => {method_name}: blk={agg['blocking_rate']:.4f} delay={agg['avg_delay_ms']:.2f}ms obj={agg['objective']:.4f}")

    out_dir = Path(args.out_json).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON: {args.out_json}")

    # Generate markdown report
    with open(args.out_md, "w") as f:
        f.write("# C-Gap Delay-Aware Agent-C Full Evaluation\n\n")
        f.write(f"**Scenario:** {topology}, s{num_slots}, r{args.requests_per_episode}, complex5_v2_lite, edge_cost_max=15\n\n")
        f.write(f"**Eval:** seeds={seeds}, episodes={args.episodes}, requests_per_episode={args.requests_per_episode}\n\n")

        # Main table
        f.write("## 1. Main Results\n\n")
        f.write("| Method | Blocking | Success | AvgDelay | Objective | AvgReward | AvgFS | PathKm | SrvOver | NoBlock |\n")
        f.write("|--------|----------|---------|----------|-----------|-----------|-------|--------|---------|---------|\n")
        for method_name in methods:
            r = all_results[method_name]
            f.write(f"| {method_name} | {r['blocking_rate']:.4f} | {r['success_rate']:.4f} | "
                    f"{r['avg_delay_ms']:.2f} | {r['objective']:.4f} | {r['avg_reward']:.4f} | "
                    f"{r['avg_fs']:.2f} | {r['avg_path_len_km']:.2f} | {r['server_overload_ratio']:.4f} | "
                    f"{r['no_suitable_block_ratio']:.4f} |\n")
        f.write("\n")

        # Agent-C comparison
        f.write("## 2. Agent-C Comparison\n\n")
        primary_name = "Enhanced-DelayAware-Agent-C" if "Enhanced-DelayAware-Agent-C" in all_results else "DelayAware-Agent-C"
        da = all_results[primary_name]
        baseline_da = all_results["DelayAware-Agent-C"]
        orig = all_results["Original-Agent-C"]
        iwd = all_results["IWD-C"]
        if primary_name != "DelayAware-Agent-C":
            f.write(f"**Enhanced vs DelayAware:**\n")
            f.write(f"- Blocking: {da['blocking_rate']:.4f} vs {baseline_da['blocking_rate']:.4f} ({da['blocking_rate'] - baseline_da['blocking_rate']:+.4f})\n")
            f.write(f"- Delay: {da['avg_delay_ms']:.2f}ms vs {baseline_da['avg_delay_ms']:.2f}ms ({da['avg_delay_ms'] - baseline_da['avg_delay_ms']:+.2f}ms)\n")
            f.write(f"- Objective: {da['objective']:.4f} vs {baseline_da['objective']:.4f} ({da['objective'] - baseline_da['objective']:+.4f})\n\n")
        f.write(f"**DelayAware vs Original:**\n")
        f.write(f"- Blocking: {baseline_da['blocking_rate']:.4f} vs {orig['blocking_rate']:.4f} ({baseline_da['blocking_rate'] - orig['blocking_rate']:+.4f})\n")
        f.write(f"- Delay: {baseline_da['avg_delay_ms']:.2f}ms vs {orig['avg_delay_ms']:.2f}ms ({baseline_da['avg_delay_ms'] - orig['avg_delay_ms']:+.2f}ms)\n")
        f.write(f"- Objective: {baseline_da['objective']:.4f} vs {orig['objective']:.4f} ({baseline_da['objective'] - orig['objective']:+.4f})\n\n")
        f.write(f"**{primary_name} vs IWD-C (best baseline):**\n")
        f.write(f"- Blocking: {da['blocking_rate']:.4f} vs {iwd['blocking_rate']:.4f} ({da['blocking_rate'] - iwd['blocking_rate']:+.4f})\n")
        f.write(f"- Delay: {da['avg_delay_ms']:.2f}ms vs {iwd['avg_delay_ms']:.2f}ms ({da['avg_delay_ms'] - iwd['avg_delay_ms']:+.2f}ms)\n")
        f.write(f"- Objective: {da['objective']:.4f} vs {iwd['objective']:.4f} ({da['objective'] - iwd['objective']:+.4f})\n\n")

        # Split distribution
        f.write("## 3. Split Distribution Comparison\n\n")
        f.write("| Method | Split Distribution |\n")
        f.write("|--------|-------------------|\n")
        for method_name in [m for m in methods if m in ("Enhanced-DelayAware-Agent-C", "DelayAware-Agent-C", "Original-Agent-C", "IWD-C")]:
            dist = all_results[method_name].get("split_counter_distribution", {})
            dist_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(dist.items()))
            f.write(f"| {method_name} | {dist_str} |\n")
        f.write("\n")

        # Server distribution
        f.write("## 4. Server Distribution Comparison\n\n")
        f.write("| Method | Server Distribution |\n")
        f.write("|--------|--------------------|\n")
        for method_name in [m for m in methods if m in ("Enhanced-DelayAware-Agent-C", "DelayAware-Agent-C", "Original-Agent-C", "IWD-C")]:
            dist = all_results[method_name].get("server_counter_distribution", {})
            dist_str = ", ".join(f"{k}={v:.1%}" for k, v in sorted(dist.items()))
            f.write(f"| {method_name} | {dist_str} |\n")
        f.write("\n")

        # Failure reasons
        f.write("## 5. Failure Reason Comparison\n\n")
        f.write("| Method | no_suitable_block | server_overload | server_saturated |\n")
        f.write("|--------|-------------------|-----------------|------------------|\n")
        for method_name in [m for m in methods if m in ("Enhanced-DelayAware-Agent-C", "DelayAware-Agent-C", "Original-Agent-C", "IWD-C")]:
            rc = all_results[method_name].get("reason_counter", {})
            total = sum(rc.values())
            nsb = rc.get("no_suitable_block", 0) / max(total, 1)
            so = rc.get("server_overload", 0) / max(total, 1)
            ss = rc.get("server_saturated", 0) / max(total, 1)
            f.write(f"| {method_name} | {nsb:.1%} | {so:.1%} | {ss:.1%} |\n")
        f.write("\n")

        # Conclusion
        da_blk = da['blocking_rate']
        da_delay = da['avg_delay_ms']
        da_obj = da['objective']
        iwd_obj = iwd['objective']
        f.write("## 6. Conclusion\n\n")
        if da_blk <= 0.22 and da_delay <= 9.5 and da_obj < iwd_obj:
            f.write(f"✅ **SUCCESS**: {primary_name} meets all targets.\n\n")
        elif da_blk < iwd['blocking_rate'] - 0.03 and da_delay < orig['avg_delay_ms']:
            f.write("✅ **PARTIAL SUCCESS**: Blocking significantly lower than IWD-C and delay reduced vs Original Agent-C.\n\n")
        else:
            f.write(f"❌ **NOT SUCCESSFUL**: {primary_name} did not meet the targets.\n\n")
        f.write(f"- Target: blocking ≤ 0.22, delay ≤ 9.5ms, objective < IWD-C ({iwd_obj:.4f})\n")
        f.write(f"- Actual: blocking = {da_blk:.4f}, delay = {da_delay:.2f}ms, objective = {da_obj:.4f}\n")

    print(f"Saved MD: {args.out_md}")


if __name__ == "__main__":
    main()
