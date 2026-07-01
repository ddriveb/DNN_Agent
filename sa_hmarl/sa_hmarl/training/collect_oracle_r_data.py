"""Collect Oracle-R teacher data for BC pretraining.

Runs the Oracle-R (future-aware exhaustive simulation) on many episodes and
collects (features, mask, teacher_action) samples.  The resulting .npz file
can be used directly with ``pretrain_agent_r_from_deeprmsa.py`` for BC
pretraining, followed by PPO fine-tuning.

The Oracle-R is VERY expensive (deep-copies the environment per candidate).
Use small episodes and expect long runtimes.  A typical collection of 20
episodes × 60 requests may take ~30-60 minutes depending on topology size.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.collect_oracle_r_data \
        --episodes 20 --requests_per_episode 60 --future_feas_coef 1.0
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.agents.r_future_aware_selector import oracle_r_select
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


def _select_c_agent(agent_c: PPOAgentC, obs_c: Dict[str, Any]) -> Optional[int]:
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        from sa_hmarl.env.c_action_risk import (
            apply_agent_c_risk_mask,
            risk_kwargs_from_args,
        )
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask,
            num_slots_total=int(ckpt_args.get("num_slots", 24)),
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


def main():
    parser = argparse.ArgumentParser(
        description="Collect Oracle-R teacher data for BC pretraining"
    )
    # Environment
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    # Request generation
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    # Data collection
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    # Oracle parameters
    parser.add_argument("--future_feas_coef", type=float, default=1.0)
    parser.add_argument("--future_feas_bonus_coef", type=float, default=0.1)
    parser.add_argument("--future_feas_horizon", type=int, default=5)
    parser.add_argument("--future_feas_norm", type=float, default=60.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    # Agent checkpoints
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_feature_mode", type=str, default="frag_aware",
                        choices=["default", "frag_aware"])
    # Output
    parser.add_argument("--output_npz", type=str,
                        default="experiments/oracle_r_data/oracle_r_teacher.npz")
    parser.add_argument("--output_stats", type=str,
                        default="experiments/oracle_r_data/oracle_r_teacher_stats.json")
    parser.add_argument("--device", type=str, default="cpu")
    args = parser.parse_args()

    print("=" * 90)
    print("Oracle-R Teacher Data Collection")
    print(f"Topology: {args.topology}  Slots: {args.num_slots}  Servers: {args.num_servers}")
    print(f"Episodes: {args.episodes}  Requests/episode: {args.requests_per_episode}")
    print(f"Future Feas: coef={args.future_feas_coef} bonus={args.future_feas_bonus_coef} "
          f"horizon={args.future_feas_horizon} norm={args.future_feas_norm}")
    print(f"Feature mode: {args.agent_r_feature_mode}")
    print("=" * 90)

    # Load fixed Agent-C
    print(f"Loading fixed Agent-C from {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)

    # Create a temporary Agent-R for feature extraction only (no training)
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    feature_agent = PPOAgentR(
        input_dim=17 if args.agent_r_feature_mode == "frag_aware" else 11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        device=args.device,
        feature_mode=args.agent_r_feature_mode,
    )
    feature_agent.policy_net.eval()

    # Prototype env
    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=args.seed,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
    )

    rng = np.random.RandomState(args.seed)

    all_features: List[np.ndarray] = []
    all_masks: List[np.ndarray] = []
    all_actions: List[int] = []

    total_requests = 0
    total_oracle_success = 0
    total_blocked = 0
    total_oracle_time = 0.0
    max_num_actions = 0

    for ep in range(args.episodes):
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

        env = make_env(
            topology=args.topology, num_slots=args.num_slots,
            num_servers=args.num_servers, seed=args.seed + ep,
            slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )
        env.reset(requests)

        ep_start = time.time()
        ep_oracle_success = 0
        ep_blocked = 0

        for req_idx, req in enumerate(requests):
            # Agent-C decision
            obs_c = build_agent_c_observation(env, req)
            action_idx_c = _select_c_agent(agent_c, obs_c)
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, len(env.mec.servers)
            )

            # Oracle-R selection
            t0 = time.time()
            oracle_action, oracle_info = oracle_r_select(
                env, req, action_c, agent_c,
                future_feas_coef=args.future_feas_coef,
                future_feas_bonus_coef=args.future_feas_bonus_coef,
                future_feas_horizon=args.future_feas_horizon,
                future_feas_norm=args.future_feas_norm,
                waste_coef=args.waste_coef,
                c_policy="agent",
                verbose=(req_idx == 0 and ep == 0),
            )
            total_oracle_time += time.time() - t0

            split_id, server_id = action_c

            if oracle_action is not None:
                total_oracle_success += 1
                ep_oracle_success += 1

                # Build R observation and extract features
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                features, mask = feature_agent.build_action_features(obs_r)

                # Pad features to consistent num_actions
                num_actions = features.shape[0]
                if num_actions > max_num_actions:
                    max_num_actions = num_actions

                all_features.append(features)
                all_masks.append(mask)
                all_actions.append(oracle_action)

            # Execute in real environment
            if oracle_action is not None:
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                action_r = decode_agent_r_action(
                    oracle_action, len(obs_r["mod_names"]), args.max_blocks,
                )
            else:
                action_r = (0, 0, 0)

            _, _, _, info = env.step(action_c, action_r)
            total_requests += 1

            if not info.get("success", False):
                ep_blocked += 1
                total_blocked += 1

        ep_elapsed = time.time() - ep_start
        print(
            f"  Ep {ep:3d} | {ep_oracle_success}/{len(requests)} oracle-selected "
            f"blocked={ep_blocked} ({ep_blocked / max(len(requests), 1):.1%}) "
            f"| {ep_elapsed:.1f}s"
        )

    # ------------------------------------------------------------------
    # Pad all features to consistent max_num_actions
    # ------------------------------------------------------------------
    print(f"\nPadding features to max_num_actions={max_num_actions}")
    N = len(all_features)
    input_dim = all_features[0].shape[1] if N > 0 else 17

    features_padded = np.zeros((N, max_num_actions, input_dim), dtype=np.float32)
    masks_padded = np.zeros((N, max_num_actions), dtype=bool)
    actions_arr = np.zeros(N, dtype=np.int64)

    for i in range(N):
        na = all_features[i].shape[0]
        features_padded[i, :na, :] = all_features[i]
        masks_padded[i, :na] = all_masks[i]
        actions_arr[i] = all_actions[i]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    output_path = Path(args.output_npz)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(output_path),
        features=features_padded,
        masks=masks_padded,
        actions=actions_arr,
        input_dim=np.array(input_dim, dtype=np.int32),
        feature_mode=np.array(args.agent_r_feature_mode, dtype=object),
        modulation_profile=np.array(args.modulation_profile, dtype=object),
        max_num_actions=np.array(max_num_actions, dtype=np.int32),
    )
    print(f"Saved {N} samples to {output_path}")

    # Stats
    stats = {
        "num_samples": N,
        "total_requests": total_requests,
        "oracle_action_found": total_oracle_success,
        "oracle_action_rate": total_oracle_success / max(total_requests, 1),
        "blocking_rate": total_blocked / max(total_requests, 1),
        "max_num_actions": max_num_actions,
        "input_dim": input_dim,
        "feature_mode": args.agent_r_feature_mode,
        "total_oracle_time_s": total_oracle_time,
        "avg_oracle_time_per_request_s": total_oracle_time / max(total_requests, 1),
        "args": vars(args),
    }
    stats_path = Path(args.output_stats)
    json.dump(stats, open(str(stats_path), "w"), indent=2)
    print(f"Saved stats to {stats_path}")
    print(f"Oracle time: {total_oracle_time:.1f}s total, "
          f"{total_oracle_time / max(total_requests, 1):.2f}s/request")


if __name__ == "__main__":
    main()
