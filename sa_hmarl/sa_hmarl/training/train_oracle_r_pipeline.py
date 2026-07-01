"""Oracle-R Teacher Pipeline: BC Pretrain + PPO Fine-Tune.

Two-stage pipeline:
  1. BC Pretrain: masked cross-entropy on Oracle-R teacher actions.
  2. PPO Fine-Tune: frag-aware PPO from the BC-pretrained checkpoint,
     with optional future-feasibility reward shaping.

The Oracle-R teacher dataset must be collected first via
``collect_oracle_r_data.py``.  This script assumes the .npz file exists.

Usage:
    # Step 1 only (BC pretrain):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.train_oracle_r_pipeline \
        --dataset experiments/oracle_r_data/oracle_r_teacher.npz \
        --stage bc_only

    # Full pipeline (BC + PPO):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.train_oracle_r_pipeline \
        --dataset experiments/oracle_r_data/oracle_r_teacher.npz \
        --stage full
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from typing import Any, Dict

import numpy as np
import torch
import torch.nn.functional as F

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_agent_r_fragaware_ppo import train as ppo_finetune
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# BC Pretraining (inlined from pretrain_agent_r_from_deeprmsa.py)
# ---------------------------------------------------------------------------

def masked_cross_entropy_loss(agent, features, masks, actions, entropy_coef=0.0):
    logits = agent.policy_net(features)
    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)

    valid_mask = masks.gather(1, actions.unsqueeze(1)).squeeze(1)
    if not valid_mask.any():
        return None, valid_mask, torch.tensor(0.0, device=features.device)

    valid_actions = actions[valid_mask]
    valid_log_probs = log_probs[valid_mask]
    nll = -valid_log_probs.gather(1, valid_actions.unsqueeze(1)).squeeze(1).mean()

    probs = torch.exp(log_probs[valid_mask])
    entropy = -(probs * valid_log_probs[valid_mask]).sum(dim=-1).mean()

    loss = nll - entropy_coef * entropy
    return loss, valid_mask, entropy


def bc_pretrain(args):
    """Behavior-clone from Oracle-R teacher dataset."""
    print("=" * 70)
    print("STAGE 1: BC Pretrain from Oracle-R Teacher")
    print("=" * 70)

    data = np.load(args.dataset, allow_pickle=True)
    features = data["features"]
    masks = data["masks"]
    actions = data["actions"]
    input_dim = int(data["input_dim"]) if "input_dim" in data else int(features.shape[-1])
    feature_mode = (
        str(data["feature_mode"])
        if "feature_mode" in data else ("frag_aware" if input_dim > 11 else "default")
    )
    modulation_profile = (
        str(data["modulation_profile"])
        if "modulation_profile" in data else "default"
    )

    N = len(features)
    n_val = max(1, int(N * args.bc_val_ratio))
    if n_val >= N:
        n_val = max(0, N // 5)
    indices = np.random.permutation(N)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    print(f"Dataset: {N} samples, input_dim={input_dim}, feature_mode={feature_mode}")
    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}")

    mod_reg = ModulationRegistry.from_profile(modulation_profile)
    agent = PPOAgentR(
        input_dim=input_dim,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        lr=args.bc_lr,
        entropy_coef=args.bc_entropy_coef,
        device=args.device,
        feature_mode=feature_mode,
    )

    best_val_acc = -1.0
    best_state: Dict[str, Any] = {}
    ckpt_dir = Path(args.bc_checkpoint).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.bc_epochs + 1):
        # Training
        perm = np.random.permutation(len(train_idx))
        total_loss = total_entropy = 0.0
        total_valid = total_skipped = 0
        num_batches = 0

        for i in range(0, len(train_idx), args.bc_batch_size):
            batch_idx = perm[i:i + args.bc_batch_size]
            bf = torch.tensor(features[train_idx][batch_idx], dtype=torch.float32, device=agent.device)
            bm = torch.tensor(masks[train_idx][batch_idx], dtype=torch.bool, device=agent.device)
            ba = torch.tensor(actions[train_idx][batch_idx], dtype=torch.long, device=agent.device)

            result = masked_cross_entropy_loss(agent, bf, bm, ba, args.bc_entropy_coef)
            if result[0] is None:
                total_skipped += len(batch_idx)
                continue

            loss, valid_mask, entropy = result
            agent.optimizer.zero_grad()
            loss.backward()
            agent.optimizer.step()

            total_loss += loss.item()
            total_entropy += entropy.item()
            total_valid += valid_mask.sum().item()
            total_skipped += (len(batch_idx) - valid_mask.sum().item())
            num_batches += 1

        # Validation
        agent.policy_net.eval()
        with torch.no_grad():
            vf = torch.tensor(features[val_idx], dtype=torch.float32, device=agent.device)
            vm = torch.tensor(masks[val_idx], dtype=torch.bool, device=agent.device)
            va = torch.tensor(actions[val_idx], dtype=torch.long, device=agent.device)

            logits = agent.policy_net(vf)
            masked_logits = logits.masked_fill(~vm, -1e9)
            pred = torch.argmax(masked_logits, dim=-1)
            valid_mask_v = vm.gather(1, va.unsqueeze(1)).squeeze(1)
            acc = (pred == va).float().mean().item()
            valid_acc = ((pred == va) & valid_mask_v).float().sum() / valid_mask_v.sum().clamp_min(1)
        agent.policy_net.train()

        if valid_acc > best_val_acc:
            best_val_acc = valid_acc
            best_state = {k: v.cpu().clone() for k, v in agent.policy_net.state_dict().items()}

        if epoch % 5 == 0 or epoch == 1 or epoch == args.bc_epochs:
            avg_loss = total_loss / max(num_batches, 1)
            print(
                f"  BC Epoch {epoch:3d}/{args.bc_epochs} | "
                f"loss={avg_loss:.4f} ent={total_entropy / max(num_batches, 1):.4f} | "
                f"val_acc={acc:.3f} val_valid_acc={valid_acc:.3f} "
                f"valid_ratio={total_valid / max(total_valid + total_skipped, 1):.3f}"
            )

    # Save best BC checkpoint
    agent.policy_net.load_state_dict(best_state)
    bc_path = Path(args.bc_checkpoint)
    bc_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": best_state,
            "input_dim": input_dim,
            "hidden_dims": (128, 64),
            "agent_r_feature_mode": feature_mode,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "modulation_profile": modulation_profile,
            "val_acc": float(best_val_acc),
        },
        bc_path,
    )
    print(f"BC checkpoint saved to {bc_path} (val_acc={best_val_acc:.3f})")
    return str(bc_path)


# ---------------------------------------------------------------------------
# Pipeline entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oracle-R Teacher Pipeline: BC Pretrain + PPO Fine-Tune"
    )
    # Pipeline control
    parser.add_argument("--stage", type=str, default="full",
                        choices=["bc_only", "full"])
    parser.add_argument("--dataset", type=str,
                        default="experiments/oracle_r_data/oracle_r_teacher.npz")
    parser.add_argument("--bc_checkpoint", type=str,
                        default="checkpoints/ppo_r_oracle_bc_best.pt")
    parser.add_argument("--ppo_checkpoint_prefix", type=str,
                        default="checkpoints/ppo_r_oracle_ft")
    # BC parameters
    parser.add_argument("--bc_epochs", type=int, default=50)
    parser.add_argument("--bc_batch_size", type=int, default=256)
    parser.add_argument("--bc_lr", type=float, default=1e-4)
    parser.add_argument("--bc_entropy_coef", type=float, default=0.001)
    parser.add_argument("--bc_val_ratio", type=float, default=0.1)
    # PPO parameters (mirrors train_agent_r_fragaware_ppo)
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--ppo_episodes", type=int, default=200)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_feature_mode", type=str, default="frag_aware")
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--future_feas_coef", type=float, default=0.0)
    parser.add_argument("--future_feas_bonus_coef", type=float, default=0.0)
    parser.add_argument("--future_feas_horizon", type=int, default=5)
    parser.add_argument("--future_feas_norm", type=float, default=60.0)
    parser.add_argument("--ppo_lr", type=float, default=1e-5)
    parser.add_argument("--entropy_coef", type=float, default=0.005)
    parser.add_argument("--ppo_epochs", type=int, default=2)
    parser.add_argument("--eval_freq", type=int, default=25)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Stage 1: BC Pretrain
    bc_path = args.bc_checkpoint
    if args.stage in ("bc_only", "full"):
        bc_path = bc_pretrain(args)

    if args.stage == "bc_only":
        print("\nBC pretraining complete. Skipping PPO fine-tune (stage=bc_only).")
        return

    # Stage 2: PPO Fine-Tune
    # We construct a PPO args object compatible with train_agent_r_fragaware_ppo.train()
    print("\n" + "=" * 70)
    print("STAGE 2: PPO Fine-Tune from BC Checkpoint")
    print(f"Warm-start from: {bc_path}")
    print("=" * 70)

    # Create a namespace with PPO training args
    ppo_args = argparse.Namespace()
    ppo_args.topology = args.topology
    ppo_args.num_slots = args.num_slots
    ppo_args.num_servers = args.num_servers
    ppo_args.max_blocks = args.max_blocks
    ppo_args.block_sort_strategy = args.block_sort_strategy
    ppo_args.num_splits = args.num_splits
    ppo_args.split_profile = args.split_profile
    ppo_args.modulation_profile = args.modulation_profile
    ppo_args.slot_bw_hz = args.slot_bw_hz
    ppo_args.guard_band_fs = args.guard_band_fs
    ppo_args.episodes = args.ppo_episodes
    ppo_args.requests_per_episode = args.requests_per_episode
    ppo_args.arrival_interval = 0.20
    ppo_args.holding_min = 4.0
    ppo_args.holding_max = 10.0
    ppo_args.deadline_min = 30.0
    ppo_args.deadline_max = 100.0
    ppo_args.size_min_mb = 5.0
    ppo_args.size_max_mb = 30.0
    ppo_args.edge_cost_min = 0.5
    ppo_args.edge_cost_max = 15.0
    ppo_args.c_policy = "agent"
    ppo_args.agent_c_checkpoint = args.agent_c_checkpoint
    ppo_args.warm_start_r = bc_path
    ppo_args.agent_r_feature_mode = args.agent_r_feature_mode
    ppo_args.agent_r_input_dim = 17 if args.agent_r_feature_mode == "frag_aware" else 11
    ppo_args.waste_coef = args.waste_coef
    ppo_args.frag_waste_coef = 0.1
    ppo_args.frag_delta_coef = 0.3
    ppo_args.frag_large_block_coef = 0.2
    ppo_args.frag_lfb_drop_coef = 0.2
    ppo_args.frag_exact_fit_bonus = 0.05
    ppo_args.fs_penalty_coef = 0.02
    ppo_args.future_feas_coef = args.future_feas_coef
    ppo_args.future_feas_bonus_coef = args.future_feas_bonus_coef
    ppo_args.future_feas_horizon = args.future_feas_horizon
    ppo_args.future_feas_metric = "mean"
    ppo_args.future_feas_norm = args.future_feas_norm
    ppo_args.future_feas_eval_reward = False
    ppo_args.gamma = 0.95
    ppo_args.lr = args.ppo_lr
    ppo_args.entropy_coef = args.entropy_coef
    ppo_args.max_grad_norm = 0.5
    ppo_args.clip_coef = 0.2
    ppo_args.ppo_epochs = args.ppo_epochs
    ppo_args.target_kl = 0.03
    ppo_args.normalize_advantage = True
    ppo_args.eval_freq = args.eval_freq
    ppo_args.eval_episodes = 5
    ppo_args.eval_seeds = "42,123"
    ppo_args.best_metric = "blocking"
    ppo_args.delay_coef = 0.5
    ppo_args.seed = args.seed
    ppo_args.device = args.device
    ppo_args.checkpoint_prefix = args.ppo_checkpoint_prefix

    ppo_finetune(ppo_args)


if __name__ == "__main__":
    main()
