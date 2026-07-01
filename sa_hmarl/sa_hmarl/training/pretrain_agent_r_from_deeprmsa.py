"""Behavior-clone PPO-R actor from DeepRMSA teacher trajectories.

Trains the PPO-R policy network with masked cross-entropy on teacher actions.
Invalid teacher actions (mask=False) are skipped and counted.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.training.pretrain_agent_r_from_deeprmsa
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse

import numpy as np
import torch
import torch.nn.functional as F

from sa_hmarl.agents.ppo_agents import PPOAgentR
from sa_hmarl.network.modulation import ModulationRegistry


def masked_cross_entropy_loss(agent: PPOAgentR, features, masks, actions, entropy_coef: float = 0.0):
    """Compute masked cross-entropy loss for a batch.

    Args:
        agent: PPOAgentR instance.
        features: (batch, num_actions, input_dim) float tensor.
        masks: (batch, num_actions) bool tensor.
        actions: (batch,) long tensor.
        entropy_coef: Entropy regularization coefficient.

    Returns:
        loss (scalar tensor), valid_mask (bool tensor), entropy (scalar tensor).
    """
    logits = agent.policy_net(features)  # (batch, num_actions)
    masked_logits = logits.masked_fill(~masks, -1e9)
    log_probs = F.log_softmax(masked_logits, dim=-1)

    # Check which teacher actions are valid under the mask
    valid_mask = masks.gather(1, actions.unsqueeze(1)).squeeze(1)
    if not valid_mask.any():
        return None, valid_mask, torch.tensor(0.0, device=features.device)

    valid_actions = actions[valid_mask]
    valid_log_probs = log_probs[valid_mask]
    nll = -valid_log_probs.gather(1, valid_actions.unsqueeze(1)).squeeze(1).mean()

    # Entropy over valid actions only
    probs = torch.exp(log_probs[valid_mask])
    entropy = -(probs * valid_log_probs[valid_mask]).sum(dim=-1).mean()

    loss = nll - entropy_coef * entropy
    return loss, valid_mask, entropy


def evaluate(agent: PPOAgentR, features, masks, actions):
    """Evaluate accuracy and loss on a dataset (no grad)."""
    agent.policy_net.eval()
    with torch.no_grad():
        features_t = torch.tensor(features, dtype=torch.float32, device=agent.device)
        masks_t = torch.tensor(masks, dtype=torch.bool, device=agent.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=agent.device)

        logits = agent.policy_net(features_t)
        masked_logits = logits.masked_fill(~masks_t, -1e9)
        log_probs = F.log_softmax(masked_logits, dim=-1)

        valid_mask = masks_t.gather(1, actions_t.unsqueeze(1)).squeeze(1)
        pred = torch.argmax(masked_logits, dim=-1)
        acc = (pred == actions_t).float().mean().item()
        valid_acc = ((pred == actions_t) & valid_mask).float().sum() / valid_mask.sum().clamp_min(1)

        if valid_mask.any():
            nll = -log_probs[valid_mask].gather(1, actions_t[valid_mask].unsqueeze(1)).squeeze(1).mean()
        else:
            nll = torch.tensor(0.0, device=agent.device)

    agent.policy_net.train()
    return float(nll.item()), acc, float(valid_acc.item()), float(valid_mask.float().mean().item())



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str,
                        default="sa_hmarl/experiments/deeprmsa_teacher_snap24_reach.npz")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--entropy_coef", type=float, default=0.001)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--checkpoint_prefix", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24")
    parser.add_argument("--modulation_profile", type=str, default=None,
                        choices=[None, "default", "extended"])
    parser.add_argument("--agent_r_feature_mode", type=str, default=None,
                        choices=[None, "default", "frag_aware"])
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    print(f"Loading dataset from {args.dataset}")
    data = np.load(args.dataset, allow_pickle=True)
    features = data["features"]      # (N, num_actions, input_dim)
    masks = data["masks"]            # (N, num_actions)
    actions = data["actions"]        # (N,)
    input_dim = int(data["input_dim"]) if "input_dim" in data else int(features.shape[-1])
    feature_mode = (
        str(data["feature_mode"])
        if "feature_mode" in data else ("frag_aware" if input_dim > 11 else "default")
    )
    if args.agent_r_feature_mode is not None:
        feature_mode = args.agent_r_feature_mode
    modulation_profile = (
        str(data["modulation_profile"])
        if "modulation_profile" in data else "default"
    )
    if args.modulation_profile is not None:
        modulation_profile = args.modulation_profile

    N = len(features)
    n_val = int(N * args.val_ratio)
    if N > 1 and n_val == 0:
        n_val = 1
    if n_val >= N:
        n_val = max(0, N - 1)
    indices = np.random.permutation(N)
    train_idx = indices[n_val:]
    val_idx = indices[:n_val]

    train_features = features[train_idx]
    train_masks = masks[train_idx]
    train_actions = actions[train_idx]

    val_features = features[val_idx]
    val_masks = masks[val_idx]
    val_actions = actions[val_idx]

    print(f"Train: {len(train_idx)}  Val: {len(val_idx)}  Total: {N}")
    print(
        f"Dataset: actions={features.shape[1]} input_dim={input_dim} "
        f"feature_mode={feature_mode} modulation_profile={modulation_profile}"
    )

    mod_reg = ModulationRegistry.from_profile(modulation_profile)
    agent = PPOAgentR(
        input_dim=input_dim,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        lr=args.lr,
        entropy_coef=args.entropy_coef,
        device=args.device,
        feature_mode=feature_mode,
    )

    best_val_acc = -1.0
    ckpt_dir = Path(args.checkpoint_prefix).parent
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        # Training
        perm = np.random.permutation(len(train_idx))
        total_loss = 0.0
        total_entropy = 0.0
        total_valid = 0
        total_skipped = 0
        num_batches = 0

        for i in range(0, len(train_idx), args.batch_size):
            batch_idx = perm[i:i + args.batch_size]
            bf = torch.tensor(train_features[batch_idx], dtype=torch.float32, device=agent.device)
            bm = torch.tensor(train_masks[batch_idx], dtype=torch.bool, device=agent.device)
            ba = torch.tensor(train_actions[batch_idx], dtype=torch.long, device=agent.device)

            result = masked_cross_entropy_loss(agent, bf, bm, ba, args.entropy_coef)
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
        val_nll, val_acc, val_valid_acc, val_valid_ratio = evaluate(
            agent, val_features, val_masks, val_actions
        )

        if num_batches > 0:
            avg_loss = total_loss / num_batches
            avg_ent = total_entropy / num_batches
        else:
            avg_loss = 0.0
            avg_ent = 0.0

        skip_ratio = total_skipped / max(len(train_idx), 1)

        print(
            f"Ep {epoch:3d} | train_loss={avg_loss:.4f} entropy={avg_ent:.4f} "
            f"skip={skip_ratio:.3%} | val_nll={val_nll:.4f} val_acc={val_acc:.4f} "
            f"val_valid_acc={val_valid_acc:.4f} valid_ratio={val_valid_ratio:.3%}"
        )

        # Save best
        if val_valid_acc > best_val_acc:
            best_val_acc = val_valid_acc
            torch.save({
                "model_state": agent.policy_net.state_dict(),
                "input_dim": agent.input_dim,
                "hidden_dims": agent.hidden_dims,
                "agent_r_feature_mode": feature_mode,
                "max_blocks": int(data["max_blocks"]) if "max_blocks" in data else None,
                "block_sort_strategy": str(data["block_sort_strategy"]) if "block_sort_strategy" in data else None,
                "modulation_profile": modulation_profile,
                "dataset": args.dataset,
                "val_valid_acc": val_valid_acc,
                "epoch": epoch,
            }, f"{args.checkpoint_prefix}_best.pt")

    # Save last
    torch.save({
        "model_state": agent.policy_net.state_dict(),
        "input_dim": agent.input_dim,
        "hidden_dims": agent.hidden_dims,
        "agent_r_feature_mode": feature_mode,
        "max_blocks": int(data["max_blocks"]) if "max_blocks" in data else None,
        "block_sort_strategy": str(data["block_sort_strategy"]) if "block_sort_strategy" in data else None,
        "modulation_profile": modulation_profile,
        "dataset": args.dataset,
        "val_valid_acc": val_valid_acc,
        "epoch": args.epochs,
    }, f"{args.checkpoint_prefix}_last.pt")

    print(f"\nBest val_valid_acc: {best_val_acc:.4f}")
    print(f"Checkpoints saved to {args.checkpoint_prefix}_best.pt / _last.pt")


if __name__ == "__main__":
    main()
