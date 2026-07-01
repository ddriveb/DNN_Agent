"""Collect teacher trajectories from DeepRMSA for Agent-R behavior cloning.

Runs DeepRMSA (eval/greedy mode) with compute-greedy Agent-C on the target
topology, records (obs_r_features, mask_r, action_r, reward, info) for every
valid step, and saves a compressed NPZ dataset.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.collect_deeprmsa_teacher_data
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from typing import Dict, List, Optional

import numpy as np

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _select_c_greedy(obs_c: Dict) -> Optional[int]:
    """Compute-greedy: pick the server with minimum edge_compute_ms."""
    mask = obs_c["agent_c_mask"]
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    best = int(valid[0])
    best_val = obs_c["candidate_features"][best]["edge_compute_ms"]
    if best_val == float("inf"):
        best_val = 1e9
    for idx in valid[1:]:
        val = obs_c["candidate_features"][int(idx)]["edge_compute_ms"]
        if val == float("inf"):
            val = 1e9
        if val < best_val:
            best = int(idx)
            best_val = val
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=5)
    parser.add_argument("--block_sort_strategy", type=str, default="size_desc",
                        choices=["size_desc", "waste_asc", "start_asc", "center_asc", "mixed"])
    parser.add_argument("--episodes", type=int, default=50)
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
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_r_feature_mode", type=str, default="default",
                        choices=["default", "frag_aware"])
    parser.add_argument("--agent_r_input_dim", type=int, default=None)
    parser.add_argument("--deep_rmsa_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/deep_rmsa_snap24_reach_mixed.pt")
    parser.add_argument("--output", type=str,
                        default="sa_hmarl/experiments/deeprmsa_teacher_snap24_reach.npz")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    # Create prototype env for topology constants
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

    # Dummy AgentR for feature building
    from sa_hmarl.agents.r_agent import AgentR
    input_dim = args.agent_r_input_dim
    if input_dim is None:
        input_dim = 17 if args.agent_r_feature_mode == "frag_aware" else 11
    dummy_agent_r = AgentR(
        input_dim=input_dim,
        mod_registry=mod_reg,
        device=args.device,
        feature_mode=args.agent_r_feature_mode,
    )

    # Load DeepRMSA teacher
    print(f"Loading DeepRMSA from {args.deep_rmsa_checkpoint}")
    ckpt = load_checkpoint(args.deep_rmsa_checkpoint, map_location=args.device)
    agent_r = DeepRMSAAgent(
        num_nodes=env_proto.net.NUM_NODES,
        num_slots=env_proto.net.num_slots,
        k_path=ckpt.get("k_path", env_proto.k),
        m_blocks=ckpt.get("m_blocks", env_proto.max_blocks),
        mod_registry=mod_reg,
        gamma=ckpt.get("gamma", 0.95),
        device=args.device,
    )
    agent_r.load_state_dict(ckpt)
    agent_r.eval()  # deterministic/greedy mode

    # Storage
    feat_list: List[np.ndarray] = []
    mask_list: List[np.ndarray] = []
    action_list: List[int] = []
    reward_list: List[float] = []
    success_list: List[bool] = []
    mod_list: List[str] = []
    delay_list: List[float] = []
    fs_list: List[int] = []
    valid_count = 0
    skip_count = 0

    max_actions = env_proto.k * len(mod_reg.names) * env_proto.max_blocks

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for ep in range(args.episodes):
            env = make_env(
                topology=args.topology,
                num_slots=args.num_slots,
                num_servers=args.num_servers,
                seed=seed + ep,
                slot_bw_hz=args.slot_bw_hz,
                guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
            )
            src = rng.randint(0, env.net.NUM_NODES)
            requests = generate_requests(
                env, rng, src,
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
            env.reset(requests)

            for req in requests:
                obs_c = build_agent_c_observation(env, req)
                action_idx_c = _select_c_greedy(obs_c)
                action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                    action_idx_c, len(env.mec.servers)
                )
                split_id, server_id = action_c

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                mask_r = obs_r["agent_r_mask"]

                # Build PPO-R style features (path-major, mod-major, block-major)
                # PPOAgentR uses AgentR.build_action_features which returns
                # full action features in path-major, mod-major, block-major order.
                features, _ = dummy_agent_r.build_action_features(obs_r)
                # Reconstruct/pad full feature matrix for stable tensor storage.
                num_paths = len(obs_r["candidate_paths"])
                num_mods = len(obs_r["mod_names"])
                num_blocks = len(mask_r) // (num_paths * num_mods) if num_paths > 0 else 0
                full_features = np.zeros((max_actions, input_dim), dtype=np.float32)
                full_mask = np.zeros((max_actions,), dtype=bool)
                if features.shape[0] == len(mask_r):
                    n = len(mask_r)
                    full_features[:n] = features
                    full_mask[:n] = mask_r
                else:
                    # Fallback: features may already be padded
                    n = min(features.shape[0], max_actions)
                    full_features[:n] = features[:n]
                    full_mask[:n] = mask_r[:n]

                # DeepRMSA action
                flat_action = agent_r.select_action(obs_r)
                if flat_action is None:
                    skip_count += 1
                    # Still step env with dummy action to keep state consistent
                    _, _, _, info = env.step(action_c, (0, 0, 0))
                    continue

                valid_count += 1
                action_r = flat_action

                # Step env
                action_r_tuple = (0, 0, 0) if action_r is None else decode_agent_r_action(action_r, num_mods, env.max_blocks)
                _, _, _, info = env.step(action_c, action_r_tuple)

                feat_list.append(full_features)
                mask_list.append(full_mask)
                action_list.append(int(action_r))
                reward_list.append(float(info.get("success", False)))
                success_list.append(bool(info.get("success", False)))
                mod_list.append(str(info.get("modulation", "unknown")))
                delay_list.append(float(info.get("delay_ms", 0.0)))
                fs_list.append(int(info.get("num_slots", 0)))

    if valid_count == 0:
        print("ERROR: No valid teacher actions collected!")
        return

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        str(out_path),
        features=np.stack(feat_list),
        masks=np.stack(mask_list),
        actions=np.array(action_list, dtype=np.int64),
        rewards=np.array(reward_list, dtype=np.float32),
        success=np.array(success_list, dtype=bool),
        modulations=np.array(mod_list),
        delays=np.array(delay_list, dtype=np.float32),
        num_slots=np.array(fs_list, dtype=np.int64),
        input_dim=np.array(input_dim, dtype=np.int64),
        feature_mode=np.array(args.agent_r_feature_mode),
        topology=np.array(args.topology),
        env_num_slots=np.array(args.num_slots, dtype=np.int64),
        max_blocks=np.array(args.max_blocks, dtype=np.int64),
        block_sort_strategy=np.array(args.block_sort_strategy),
        num_splits=np.array(args.num_splits, dtype=np.int64),
        split_profile=np.array(args.split_profile),
        modulation_profile=np.array(args.modulation_profile),
    )
    print(f"Saved {valid_count} valid samples ({skip_count} skipped) to {out_path}")
    print(f"  Features: {np.stack(feat_list).shape}")
    print(f"  Actions: {len(action_list)}")
    print(f"  Success rate: {np.mean(success_list):.2%}")


if __name__ == "__main__":
    main()
