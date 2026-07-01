"""Behavior cloning for Agent-C on complex5_v2 using DF baseline as teacher.

Collects (observation, action) pairs from DF baseline and trains PPO Agent-C
via supervised learning (cross-entropy loss) before RL fine-tuning.

Usage:
    PYTHONPATH=sa_hmarl python -m sa_hmarl.training.train_agent_c_bc \
        --teacher_method df \
        --num_episodes 200 \
        --epochs 300
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
from collections import Counter
from typing import Dict, List
import traceback

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def collect_teacher_data(
    teacher_method: str,
    agent_r: PPOAgentR,
    num_episodes: int,
    requests_per_episode: int,
    num_slots: int,
    num_servers: int,
    arrival_interval: float,
    edge_cost_max: float,
    split_profile: str,
    seed: int,
    device: str,
) -> List[dict]:
    """Collect (obs_c, action_idx) pairs from teacher baseline."""
    env_proto = make_env(
        topology="snap24_gnutella_reach",
        num_slots=num_slots,
        num_servers=num_servers,
        seed=seed,
        slot_bw_hz=1.25e9,
        guard_band_fs=1,
        modulation_profile="default",
    )
    rng = np.random.RandomState(seed)
    data = []
    stats = Counter()
    split_counter = Counter()
    server_counter = Counter()
    reason_counter = Counter()
    valid_action_counts = []

    for ep in range(num_episodes):
        src = rng.randint(0, env_proto.net.NUM_NODES)
        requests = generate_requests(
            env_proto, rng, src, requests_per_episode,
            arrival_interval=arrival_interval,
            holding_min=4.0, holding_max=10.0,
            deadline_min=30.0, deadline_max=100.0,
            size_min_mb=5.0, size_max_mb=30.0,
            edge_cost_min=0.5, edge_cost_max=edge_cost_max,
            num_splits=5, split_profile=split_profile,
        )
        env = make_env(
            topology="snap24_gnutella_reach",
            num_slots=num_slots,
            num_servers=num_servers,
            seed=seed + ep,
            slot_bw_hz=1.25e9,
            guard_band_fs=1,
            modulation_profile="default",
        )
        env.reset(requests)
        server_selected_count = np.zeros(num_servers, dtype=int)
        teacher_rng = np.random.RandomState(seed + ep)

        for req in requests:
            stats["total_requests"] += 1
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]
            valid_count = int(np.sum(mask_c))
            valid_action_counts.append(valid_count)
            if valid_count == 0:
                stats["no_valid_c_mask"] += 1
                continue

            action_idx = select_offloading_action(
                teacher_method, env, req, obs_c, mask_c,
                rng=teacher_rng,
                server_selected_count=server_selected_count,
            )
            if action_idx is None:
                stats["teacher_none"] += 1
                continue
            if action_idx < 0 or action_idx >= len(mask_c) or not mask_c[action_idx]:
                stats["teacher_invalid"] += 1
                continue

            # Convert candidate feature dicts to vectors
            feature_vecs = []
            for feat in obs_c["candidate_features"]:
                edge_ms = feat["edge_compute_ms"]
                if edge_ms == float("inf"):
                    edge_ms = 999.0
                local_ms = feat["local_compute_ms"]
                if local_ms == float("inf"):
                    local_ms = 999.0
                vec = [
                    feat["intermediate_size_mb"],
                    local_ms,
                    edge_ms,
                    feat["server_utilization"],
                    feat["best_fs_estimate"] if feat["best_fs_estimate"] is not None else 0.0,
                    feat["safe_fs_estimate"] if feat["safe_fs_estimate"] is not None else 0.0,
                    feat["feasible_count"],
                ] + list(feat["spectrum_summary"])
                feature_vecs.append(vec)
            features = np.array(feature_vecs, dtype=np.float32)
            mask = mask_c.astype(np.float32)
            data.append({
                "features": features,
                "mask": mask,
                "action": action_idx,
            })
            stats["samples_collected"] += 1

            # Execute action to advance env state
            split_id, server_id = decode_agent_c_action(action_idx, num_servers)
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count[server_id] += 1
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = agent_r.select_action(obs_r, deterministic=True)
            if action_idx_r is None:
                stats["r_action_none"] += 1
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), env.max_blocks)
            _, _, _, info = env.step((split_id, server_id), action_r)
            if info.get("success") is False:
                stats["env_block"] += 1
                reason_counter[info.get("reason", "unknown")] += 1
            else:
                stats["env_success"] += 1

        if (ep + 1) % 20 == 0:
            mean_valid = float(np.mean(valid_action_counts)) if valid_action_counts else 0.0
            print(
                f"  Collected {len(data)} samples from {ep+1} episodes "
                f"(requests={stats['total_requests']}, no_valid={stats['no_valid_c_mask']}, "
                f"teacher_none={stats['teacher_none']}, mean_valid={mean_valid:.2f})",
                flush=True,
            )

    # Attach diagnostics for callers without changing the public return type.
    collect_teacher_data.last_stats = {
        "stats": dict(stats),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "reason_counter": dict(reason_counter),
        "valid_action_mean": float(np.mean(valid_action_counts)) if valid_action_counts else 0.0,
        "valid_action_min": int(np.min(valid_action_counts)) if valid_action_counts else 0,
        "valid_action_max": int(np.max(valid_action_counts)) if valid_action_counts else 0,
    }

    return data


def train_bc(
    agent_c: PPOAgentC,
    data: List[dict],
    epochs: int,
    batch_size: int,
    lr: float,
    device: str,
):
    """Train Agent-C via behavior cloning (cross-entropy)."""
    if len(data) == 0:
        raise RuntimeError("No behavior-cloning samples were collected.")

    optimizer = torch.optim.Adam(agent_c.policy_net.parameters(), lr=lr)
    num_samples = len(data)

    for epoch in range(epochs):
        # Shuffle data
        indices = np.random.permutation(num_samples)
        total_loss = 0.0
        total_correct = 0
        total_masked = 0
        num_batches = 0

        for start in range(0, num_samples, batch_size):
            end = min(start + batch_size, num_samples)
            batch_idx = indices[start:end]

            features_list = [data[i]["features"] for i in batch_idx]
            masks_list = [data[i]["mask"] for i in batch_idx]
            actions = [data[i]["action"] for i in batch_idx]

            # Pad to max actions in batch
            max_actions = max(len(m) for m in masks_list)
            padded_features = []
            padded_masks = []
            for features, mask in zip(features_list, masks_list):
                pad_len = max_actions - len(mask)
                if pad_len > 0:
                    features = np.concatenate(
                        [features, np.zeros((pad_len, features.shape[1]), dtype=np.float32)], axis=0
                    )
                    mask = np.concatenate([mask, np.zeros(pad_len, dtype=np.float32)], axis=0)
                padded_features.append(features)
                padded_masks.append(mask)

            features_t = torch.tensor(np.stack(padded_features), dtype=torch.float32, device=device)
            masks_t = torch.tensor(np.stack(padded_masks), dtype=torch.bool, device=device)
            actions_t = torch.tensor(actions, dtype=torch.long, device=device)

            logits = agent_c.policy_net(features_t).masked_fill(~masks_t, -1e9)
            loss = F.cross_entropy(logits, actions_t)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(agent_c.policy_net.parameters(), 0.5)
            optimizer.step()

            # Metrics
            pred = torch.argmax(logits, dim=1)
            total_correct += (pred == actions_t).sum().item()
            total_masked += (~masks_t[torch.arange(len(actions_t)), actions_t]).sum().item()
            total_loss += loss.item()
            num_batches += 1

        acc = total_correct / num_samples if num_samples > 0 else 0.0
        masked_actions = total_masked
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        print(
            f"Epoch {epoch:3d}: loss={avg_loss:.4f} "
            f"acc={acc:.3f} masked_teacher_actions={masked_actions}",
            flush=True,
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_method", type=str, default="df",
                        choices=["greedy", "df", "rf", "wo", "iwd"])
    parser.add_argument("--num_episodes", type=int, default=200)
    parser.add_argument("--requests_per_episode", type=int, default=120)
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--arrival_interval", type=float, default=0.20)
    parser.add_argument("--edge_cost_max", type=float, default=60.0)
    parser.add_argument("--split_profile", type=str, default="complex5_v2")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--bc_r_ckpt", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--output_ckpt", type=str,
                        default="sa_hmarl/checkpoints/agent_c_complex5v2_bc.pt")
    parser.add_argument("--warm_start_c", type=str, default=None)
    parser.add_argument("--agent_c_activation", type=str, default="tanh",
                        choices=["tanh", "relu", "silu", "gelu", "leaky_relu"])
    args = parser.parse_args()

    print("=" * 60)
    print("Behavior Cloning for Agent-C")
    print(f"Teacher: {args.teacher_method.upper()}")
    print(f"Episodes: {args.num_episodes}, Requests/episode: {args.requests_per_episode}")
    print(f"Slots: {args.num_slots}, Edge max: {args.edge_cost_max}")
    print(f"BC epochs: {args.epochs}, batch_size: {args.batch_size}")
    print("=" * 60)

    # Load frozen R
    mod_reg = ModulationRegistry()
    ckpt = load_checkpoint(args.bc_r_ckpt, map_location=args.device)
    agent_r = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=args.device,
    )
    agent_r.policy_net.load_state_dict(ckpt["model_state"])
    agent_r.policy_net.eval()

    # Create Agent-C
    input_dim = 17  # 7 base + 10 spectrum
    agent_c = PPOAgentC(input_dim=input_dim, hidden_dims=(128, 64), device=args.device,
                        activation=args.agent_c_activation)

    # Optional warm-start from existing checkpoint
    if args.warm_start_c:
        ws_ckpt = load_checkpoint(args.warm_start_c, map_location=args.device)
        agent_c.policy_net.load_state_dict(ws_ckpt["model_state"])
        print(f"Warm-started from {args.warm_start_c}")

    # Collect teacher data
    print("\n--- Collecting teacher data ---")
    data = collect_teacher_data(
        args.teacher_method, agent_r,
        args.num_episodes, args.requests_per_episode,
        args.num_slots, args.num_servers,
        args.arrival_interval, args.edge_cost_max,
        args.split_profile, args.seed, args.device,
    )
    print(f"Total samples: {len(data)}")
    print("Teacher data diagnostics:")
    diag: Dict = getattr(collect_teacher_data, "last_stats", {})
    for key, value in diag.items():
        print(f"  {key}: {value}", flush=True)

    # Train BC
    print("\n--- Training behavior clone ---")
    train_bc(agent_c, data, args.epochs, args.batch_size, args.lr, args.device)

    # Save checkpoint
    Path(args.output_ckpt).parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state": agent_c.policy_net.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": (128, 64),
        "objective": "bc",
    }, args.output_ckpt)
    print(f"\nSaved BC checkpoint: {args.output_ckpt}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
