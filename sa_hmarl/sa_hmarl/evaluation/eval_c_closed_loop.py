"""Closed-loop K-path evaluation for PPO Agent-C checkpoints.

Reports:
    - blocking_rate
    - raw_mask_empty_rate
    - no_valid_c_rate
    - avg_valid_r_actions
    - server_overload_rate

Usage:
    PYTHONPATH=sa_hmarl python sa_hmarl/sa_hmarl/evaluation/eval_c_closed_loop.py \
        --checkpoint sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt \
        --topology snap24_gnutella_reach --num_slots 20 --num_servers 4 --k_paths 5 \
        --max_blocks 10 --block_sort_strategy mixed --split_profile default3 \
        --seeds 42,123 --episodes 20 --requests_per_episode 80
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"),
    )
    num_servers = ckpt_args.get("num_servers")
    if num_servers is None:
        num_servers = ckpt.get("num_servers")
    agent_c_kwargs = dict(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    if feature_mode in (
        "mean_field", "typed_mean_field", "gated_typed_mean_field",
        "fixed_blend_typed_mean_field", "candidate_mean_field",
        "candidate_mean_field_count_only",
    ):
        agent_c_kwargs["num_servers"] = num_servers
    if feature_mode == "fixed_blend_typed_mean_field":
        agent_c_kwargs["fixed_blend_alpha"] = ckpt_args.get("fixed_blend_alpha", 0.5)
    agent = PPOAgentC(**agent_c_kwargs)
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
        feature_mode=ckpt.get("agent_r_feature_mode", "default"),
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _select_c_action(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[int, np.ndarray]:
    features, mask = agent_c.build_action_features(obs_c)
    raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    if action is None:
        action = 0
    return action, raw_mask


def _select_r_action(agent_r: PPOAgentR, obs_r: Dict[str, Any], max_blocks: int) -> Tuple[int, int, int]:
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return 0, 0, 0
    return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), max_blocks)


def evaluate_checkpoint(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

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
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg

    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    agent_c = _load_ppo_c(args.checkpoint, args.device)

    per_seed_results = []
    all_requests_total = 0
    all_blocked = 0
    all_raw_empty = 0
    all_no_valid_c = 0
    all_server_overload = 0
    all_total_valid_r_actions = []
    all_selected_valid_r_actions = []
    all_delays = []
    all_fs = []

    t_start = time.time()

    for seed in seeds:
        rng = np.random.RandomState(seed)
        episodes_data = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            episodes_data.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
                traffic_mode=args.traffic_mode,
                regime_stay_prob=args.regime_stay_prob,
            ))

        seed_total = 0
        seed_blocked = 0
        seed_raw_empty = 0
        seed_no_valid_c = 0
        seed_server_overload = 0
        seed_total_valid_r_actions = []
        seed_selected_valid_r_actions = []
        seed_delays = []
        seed_fs = []

        for requests in episodes_data:
            env = make_env(
                topology=args.topology,
                num_slots=args.num_slots,
                num_servers=args.num_servers,
                seed=42,
                slot_bw_hz=args.slot_bw_hz,
                guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)

            for req in requests:
                obs_c = build_agent_c_observation(env, req)
                action_idx_c, raw_mask = _select_c_action(agent_c, obs_c, env.net.num_slots)
                action_c = decode_agent_c_action(action_idx_c, args.num_servers)
                split_id, server_id = action_c

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
                _, _, _, info = env.step(action_c, action_r)

                success = bool(info.get("success", False))
                reason = info.get("reason", "unknown")
                raw_empty = int(raw_mask.sum()) == 0
                no_valid_c = raw_empty

                total_valid_r_count = 0
                for slist in obs_c["feasible_counts"]:
                    total_valid_r_count += sum(slist)
                selected_valid_r_count = int(obs_c["feasible_counts"][split_id][server_id])

                seed_total += 1
                if not success:
                    seed_blocked += 1
                if raw_empty:
                    seed_raw_empty += 1
                if no_valid_c:
                    seed_no_valid_c += 1
                if reason == "server_overload":
                    seed_server_overload += 1
                seed_total_valid_r_actions.append(total_valid_r_count)
                seed_selected_valid_r_actions.append(selected_valid_r_count)
                if success:
                    seed_delays.append(float(info.get("delay_ms", 0.0)))
                    seed_fs.append(float(info.get("num_slots", 0.0)))

        per_seed_results.append({
            "seed": seed,
            "requests": seed_total,
            "blocking_rate": seed_blocked / max(seed_total, 1),
            "raw_mask_empty_rate": seed_raw_empty / max(seed_total, 1),
            "no_valid_c_rate": seed_no_valid_c / max(seed_total, 1),
            "server_overload_rate": seed_server_overload / max(seed_total, 1),
            "avg_total_valid_r_actions": float(np.mean(seed_total_valid_r_actions)) if seed_total_valid_r_actions else 0.0,
            "avg_selected_valid_r_actions": float(np.mean(seed_selected_valid_r_actions)) if seed_selected_valid_r_actions else 0.0,
            "mean_delay_ms": float(np.mean(seed_delays)) if seed_delays else 0.0,
            "avg_fs": float(np.mean(seed_fs)) if seed_fs else 0.0,
        })

        all_requests_total += seed_total
        all_blocked += seed_blocked
        all_raw_empty += seed_raw_empty
        all_no_valid_c += seed_no_valid_c
        all_server_overload += seed_server_overload
        all_total_valid_r_actions.extend(seed_total_valid_r_actions)
        all_selected_valid_r_actions.extend(seed_selected_valid_r_actions)
        all_delays.extend(seed_delays)
        all_fs.extend(seed_fs)

    return {
        "config": {
            "checkpoint": args.checkpoint,
            "agent_r_checkpoint": args.agent_r_checkpoint,
            "topology": args.topology,
            "num_slots": args.num_slots,
            "num_servers": args.num_servers,
            "k_paths": args.k_paths,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "split_profile": args.split_profile,
            "traffic_mode": args.traffic_mode,
            "regime_stay_prob": args.regime_stay_prob,
            "seeds": seeds,
            "episodes": args.episodes,
            "requests_per_episode": args.requests_per_episode,
        },
        "per_seed": per_seed_results,
        "aggregate": {
            "requests": all_requests_total,
            "blocking_rate": all_blocked / max(all_requests_total, 1),
            "raw_mask_empty_rate": all_raw_empty / max(all_requests_total, 1),
            "no_valid_c_rate": all_no_valid_c / max(all_requests_total, 1),
            "server_overload_rate": all_server_overload / max(all_requests_total, 1),
            "avg_total_valid_r_actions": float(np.mean(all_total_valid_r_actions)) if all_total_valid_r_actions else 0.0,
            "avg_selected_valid_r_actions": float(np.mean(all_selected_valid_r_actions)) if all_selected_valid_r_actions else 0.0,
            "mean_delay_ms": float(np.mean(all_delays)) if all_delays else 0.0,
            "avg_fs": float(np.mean(all_fs)) if all_fs else 0.0,
        },
        "elapsed_seconds": time.time() - t_start,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--agent_r_checkpoint", type=str, default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--traffic_mode", type=str, default="iid",
                        choices=["iid", "markov_regime"])
    parser.add_argument("--regime_stay_prob", type=float, default=0.9)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    result = evaluate_checkpoint(args)

    print(json.dumps(result, indent=2, default=str))

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2, default=str)
        print(f"\nSaved: {args.output}")


if __name__ == "__main__":
    main()
