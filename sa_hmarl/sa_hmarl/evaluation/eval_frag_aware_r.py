"""Evaluate preventive fragmentation-aware reranking for Agent-R.

This script keeps the trained R policy unchanged and adds an inference-time
reranking term that penalizes actions predicted to damage local spectrum
continuity.  It is a lightweight way to test whether SD-style information is
useful without adding a third defragmentation agent.

Example:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_frag_aware_r
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.agents.r_frag_selector import select_frag_aware_action
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


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
    feature_mode = ckpt.get("agent_r_feature_mode", "default")
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _select_c_greedy(obs_c: Dict[str, Any]) -> Optional[int]:
    mask = obs_c["agent_c_mask"]
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    best_idx = int(valid[0])
    best_val = float(obs_c["candidate_features"][best_idx].get("edge_compute_ms", 1e9))
    if not np.isfinite(best_val):
        best_val = 1e9
    for idx in valid[1:]:
        idx = int(idx)
        val = float(obs_c["candidate_features"][idx].get("edge_compute_ms", 1e9))
        if not np.isfinite(val):
            val = 1e9
        if val < best_val:
            best_idx = idx
            best_val = val
    return best_idx


def _select_c_agent(agent_c: PPOAgentC, obs_c: Dict[str, Any]) -> Optional[int]:
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c,
            mask,
            num_slots_total=int(ckpt_args.get("num_slots", 24)),
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


def _select_r(agent_r: PPOAgentR, obs_r: Dict[str, Any], mode: str, args) -> Optional[int]:
    if mode == "ppo":
        return agent_r.select_action(obs_r, deterministic=True)
    if mode == "frag":
        return select_frag_aware_action(
            agent_r,
            obs_r,
            waste_coef=args.frag_waste_coef,
            frag_coef=args.frag_delta_coef,
            large_block_coef=args.frag_large_block_coef,
            lfb_drop_coef=args.frag_lfb_drop_coef,
            exact_fit_bonus=args.frag_exact_fit_bonus,
        )
    raise ValueError(f"Unknown R mode: {mode}")


def evaluate_config(
    agent_c: Optional[PPOAgentC],
    agent_r: PPOAgentR,
    c_mode: str,
    r_mode: str,
    episodes_data: List[List],
    env_proto,
    args,
) -> Dict[str, Any]:
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    total = blocked = success = deadline_met = no_valid_c = no_valid_r = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    mod_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    reason_counter = Counter()

    for requests in episodes_data:
        env = make_env(
            topology=env_proto.net.topology,
            num_slots=env_proto.net.num_slots,
            num_servers=num_servers,
            seed=42,
            slot_bw_hz=env_proto.fs_calc.slot_bw_hz,
            guard_band_fs=env_proto.fs_calc.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
            server_nodes=server_nodes,
            capacities=capacities,
        )
        env.reset(requests)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            if c_mode == "agent":
                action_idx_c = _select_c_agent(agent_c, obs_c)
            elif c_mode == "greedy":
                action_idx_c = _select_c_greedy(obs_c)
            else:
                raise ValueError(f"Unknown C mode: {c_mode}")

            if action_idx_c is None:
                no_valid_c += 1
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, num_servers)

            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = _select_r(agent_r, obs_r, r_mode, args)
            if action_idx_r is None:
                no_valid_r += 1
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                )

            _, _, _, info = env.step(action_c, action_r)
            total += 1
            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, args.waste_coef, server.utilization
            )
            total_reward += reward

            if info.get("success", False):
                success += 1
                delay = float(info.get("delay_ms", 0.0))
                total_delay += delay
                total_fs += float(info.get("num_slots", 0))
                total_waste += float(info.get("block_waste", 0.0))
                total_path += float(info.get("path_dist_km", 0.0))
                if delay <= req.deadline_ms:
                    deadline_met += 1
                mod_counter[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1

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
        "no_valid_c": no_valid_c,
        "no_valid_r": no_valid_r,
        "mod_counter": dict(mod_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "reason_counter": dict(reason_counter),
    }


def _aggregate(seed_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
        "no_valid_c", "no_valid_r",
    ]
    agg = {}
    for key in scalar_keys:
        vals = [float(r[key]) for r in seed_results]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    for key in ["mod_counter", "split_counter", "server_counter", "reason_counter"]:
        c = Counter()
        for r in seed_results:
            c.update(r[key])
        agg[key] = dict(c)
    return agg


def _fmt_counter(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in c.most_common())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.20)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=5)
    parser.add_argument("--block_sort_strategy", type=str, default="size_desc",
                        choices=["size_desc", "waste_asc", "start_asc", "center_asc", "mixed"])
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--frag_waste_coef", type=float, default=0.2)
    parser.add_argument("--frag_delta_coef", type=float, default=0.4)
    parser.add_argument("--frag_large_block_coef", type=float, default=0.2)
    parser.add_argument("--frag_lfb_drop_coef", type=float, default=0.2)
    parser.add_argument("--frag_exact_fit_bonus", type=float, default=0.05)
    parser.add_argument("--output_json", type=str,
                        default="sa_hmarl/experiments/frag_aware_r_eval.json")
    parser.add_argument("--output_md", type=str,
                        default="sa_hmarl/experiments/frag_aware_r_eval.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
    )
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    all_episodes = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            episodes.append(
                generate_requests(
                    env_proto,
                    rng,
                    src,
                    num_requests=args.requests_per_episode,
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
            )
        all_episodes[seed] = episodes

    configs = [
        ("Greedy-C + BC-PPO-R", "greedy", "ppo"),
        ("Greedy-C + BC-PPO-R + FragRerank", "greedy", "frag"),
        ("Agent-C + BC-PPO-R", "agent", "ppo"),
        ("Agent-C + BC-PPO-R + FragRerank", "agent", "frag"),
    ]

    results = {}
    for name, c_mode, r_mode in configs:
        print(f"\n--- {name} ---")
        per_seed = []
        for seed in seeds:
            res = evaluate_config(
                agent_c, agent_r, c_mode, r_mode, all_episodes[seed], env_proto, args
            )
            per_seed.append(res)
            print(
                f"seed={seed} blk={res['blocking_rate']:.3f} "
                f"rwd={res['avg_reward']:+.3f} delay={res['avg_delay_ms']:.2f} "
                f"fs={res['avg_fs']:.2f} waste={res['avg_waste']:.3f}"
            )
        results[name] = _aggregate(per_seed)

    report = []
    report.append("# Frag-Aware Agent-R Rerank Evaluation")
    report.append("")
    report.append(
        f"Scenario: `{args.topology}`, slots={args.num_slots}, "
        f"requests/episode={args.requests_per_episode}, split_profile={args.split_profile}, "
        f"max_blocks={args.max_blocks}, block_sort={args.block_sort_strategy}"
    )
    report.append("")
    report.append("| Method | Blocking | AvgReward | Delay(ms) | AvgFS | Waste | PathKm |")
    report.append("|---|---:|---:|---:|---:|---:|---:|")
    for name, _, _ in configs:
        r = results[name]
        report.append(
            f"| {name} | {r['blocking_rate']:.4f} | {r['avg_reward']:+.4f} | "
            f"{r['avg_delay_ms']:.2f} | {r['avg_fs']:.2f} | "
            f"{r['avg_waste']:.3f} | {r['avg_path_len_km']:.1f} |"
        )
    report.append("")
    report.append("## Modulation Distribution")
    report.append("")
    for name, _, _ in configs:
        report.append(f"- {name}: {_fmt_counter(results[name]['mod_counter'])}")
    report.append("")
    report.append("## Failure Reasons")
    report.append("")
    for name, _, _ in configs:
        report.append(f"- {name}: {_fmt_counter(results[name]['reason_counter'])}")

    payload = {
        "args": vars(args),
        "results": results,
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
    Path(args.output_md).write_text("\n".join(report) + "\n", encoding="utf-8")

    print("\n" + "\n".join(report))
    print(f"\nSaved JSON: {args.output_json}")
    print(f"Saved report: {args.output_md}")


if __name__ == "__main__":
    main()
