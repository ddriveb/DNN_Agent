"""Train an R-side TD post-decision value network from replay transitions.

Fitted-Q iteration style:
- Maintain online network and target network.
- Target = r_t + gamma * max_a' V_target(z_{t+1}^{(a')})
- If next candidates empty or episode ends, target = r_t.
- Loss = HuberLoss(V_online(z_t), target).
- Hard-update target network every target_update_interval epochs.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sa_hmarl.agents.post_decision_value import PostDecisionValueNetwork


def _load_split(root: Path, split: str) -> Dict[str, np.ndarray]:
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _compute_starts(lengths: np.ndarray) -> np.ndarray:
    """Cumulative start indices for ragged next-candidate arrays."""
    if len(lengths) == 0:
        return np.empty(0, dtype=np.int64)
    starts = np.empty_like(lengths)
    starts[0] = 0
    starts[1:] = np.cumsum(lengths[:-1])
    return starts


def _compute_td_target(
    next_candidates: np.ndarray,
    next_lengths: np.ndarray,
    rewards: np.ndarray,
    dones: np.ndarray,
    target_net: PostDecisionValueNetwork,
    mean: np.ndarray,
    std: np.ndarray,
    gamma: float,
    device: str,
) -> np.ndarray:
    """Compute Bellman target for a batch of transitions."""
    starts = _compute_starts(next_lengths)
    targets = rewards.copy().astype(np.float32)
    has_next = (next_lengths > 0) & (~dones)
    if not np.any(has_next):
        return targets

    # Evaluate all next candidates in one forward pass.
    all_indices = np.concatenate([
        np.arange(starts[i], starts[i] + next_lengths[i])
        for i in np.flatnonzero(has_next)
    ])
    if len(all_indices) == 0:
        return targets

    all_cands = next_candidates[all_indices]
    normalized = (all_cands - mean) / std
    with torch.no_grad():
        values = target_net(
            torch.as_tensor(normalized, dtype=torch.float32, device=device)
        ).cpu().numpy()

    # Scatter max back to each transition.
    offset = 0
    for i in np.flatnonzero(has_next):
        L = int(next_lengths[i])
        vals = values[offset:offset + L]
        targets[i] += gamma * float(vals.max())
        offset += L
    return targets


def _evaluate_split(
    data: Dict[str, np.ndarray],
    online_net: PostDecisionValueNetwork,
    target_net: PostDecisionValueNetwork,
    mean: np.ndarray,
    std: np.ndarray,
    gamma: float,
    device: str,
    batch_size: int = 2048,
) -> Dict[str, float]:
    """Compute loss, value stats, and target stats on a split."""
    z = data["z_t"]
    normalized = (z - mean) / std
    rewards = data["reward"].astype(np.float32)
    dones = data["done"].astype(bool)
    next_lengths = data["next_candidate_lengths"].astype(np.int64)
    next_candidates = data["next_candidates"]

    # Compute targets in batches to avoid huge memory use.
    all_targets = []
    for start in range(0, len(z), batch_size):
        end = min(start + batch_size, len(z))
        target_batch = _compute_td_target(
            next_candidates, next_lengths[start:end],
            rewards[start:end], dones[start:end],
            target_net, mean, std, gamma, device,
        )
        all_targets.append(target_batch)
    targets = np.concatenate(all_targets)

    # Compute predictions in batches.
    all_preds = []
    for start in range(0, len(z), batch_size):
        end = min(start + batch_size, len(z))
        x = torch.as_tensor(normalized[start:end], dtype=torch.float32, device=device)
        with torch.no_grad():
            all_preds.append(online_net(x).cpu().numpy())
    preds = np.concatenate(all_preds)

    finite = np.isfinite(preds) & np.isfinite(targets)
    loss = float(np.mean(np.where(
        np.abs(preds - targets) <= 1.0,
        0.5 * (preds - targets) ** 2,
        np.abs(preds - targets) - 0.5,
    )))
    return {
        "loss": loss,
        "pred_mean": float(preds.mean()),
        "pred_std": float(preds.std()),
        "pred_min": float(preds.min()),
        "pred_max": float(preds.max()),
        "target_mean": float(targets.mean()),
        "target_std": float(targets.std()),
        "target_min": float(targets.min()),
        "target_max": float(targets.max()),
        "finite_rate": float(finite.mean()),
        "value_collapsed": bool(preds.std() < 1e-4),
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    root = Path(args.dataset_dir)
    metadata = json.loads((root / "replay_report.json").read_text(encoding="utf-8"))
    train_data = _load_split(root, "train")
    val_data = _load_split(root, "val")
    test_data = _load_split(root, "test")

    mean = np.asarray(metadata["train_feature_mean"], dtype=np.float32)
    std = np.asarray(metadata["train_feature_std"], dtype=np.float32)
    input_dim = train_data["z_t"].shape[1]

    online_net = PostDecisionValueNetwork(
        input_dim, args.hidden_dims, args.dropout,
    ).to(args.device)
    target_net = copy.deepcopy(online_net)
    target_net.eval()

    optimizer = torch.optim.AdamW(
        online_net.parameters(), lr=args.lr, weight_decay=args.weight_decay,
    )

    n = len(train_data["z_t"])
    normalized_train = ((train_data["z_t"] - mean) / std).astype(np.float32)
    rewards = train_data["reward"].astype(np.float32)
    dones = train_data["done"].astype(bool)
    next_lengths = train_data["next_candidate_lengths"].astype(np.int64)
    next_candidates = train_data["next_candidates"]

    history = []
    best_val_loss = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        online_net.train()
        order = np.random.permutation(n)
        train_losses = []
        for start in range(0, n, args.batch_size):
            indices = order[start:start + args.batch_size]
            x = torch.as_tensor(normalized_train[indices], dtype=torch.float32, device=args.device)
            target = _compute_td_target(
                next_candidates, next_lengths[indices],
                rewards[indices], dones[indices],
                target_net, mean, std, args.gamma, args.device,
            )
            y = torch.as_tensor(target, dtype=torch.float32, device=args.device)
            pred = online_net(x)
            loss = F.smooth_l1_loss(pred, y)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(online_net.parameters(), args.max_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Hard update target network.
        if epoch % args.target_update_interval == 0:
            target_net.load_state_dict(online_net.state_dict())
            target_net.eval()

        # Evaluate.
        online_net.eval()
        train_metrics = _evaluate_split(train_data, online_net, target_net, mean, std, args.gamma, args.device)
        val_metrics = _evaluate_split(val_data, online_net, target_net, mean, std, args.gamma, args.device)

        history.append({
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "val_loss": val_metrics["loss"],
            "val_pred_std": val_metrics["pred_std"],
            "val_target_std": val_metrics["target_std"],
        })

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[epoch={epoch}] train_loss={train_metrics['loss']:.5f} "
                f"val_loss={val_metrics['loss']:.5f} "
                f"pred_std={val_metrics['pred_std']:.4f} "
                f"target_std={val_metrics['target_std']:.4f}",
                flush=True,
            )

        if val_metrics["loss"] < best_val_loss - args.min_delta:
            best_val_loss = val_metrics["loss"]
            best_state = {k: v.detach().cpu().clone() for k, v in online_net.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in online_net.state_dict().items()}
    online_net.load_state_dict(best_state)
    target_net.load_state_dict(best_state)

    test_metrics = _evaluate_split(test_data, online_net, target_net, mean, std, args.gamma, args.device)

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "td_post_decision.pt"
    torch.save({
        "model_state_dict": online_net.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "feature_names": metadata["feature_names"],
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "gamma": args.gamma,
        "reward_block_coef": metadata["reward_block_coef"],
        "reward_delay_coef": metadata["reward_delay_coef"],
        "train_result": {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
            "history": history,
        },
        "test_metrics": test_metrics,
    }, checkpoint_path)

    report = {
        "dataset": str(root),
        "checkpoint": str(checkpoint_path),
        "train_result": {
            "best_val_loss": best_val_loss,
            "epochs_trained": len(history),
            "history": history,
        },
        "test_metrics": test_metrics,
        "config": vars(args),
        "elapsed_seconds": time.time() - getattr(args, "_start_time", time.time()),
    }
    _atomic_write_json(output / "training_report.json", report)
    _write_markdown(output / "training_report.md", report, metadata)
    return report


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _write_markdown(path: Path, report: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    test = report["test_metrics"]
    lines = [
        "# TD R-Side Post-Decision Value Training", "",
        f"- Dataset: `{report['dataset']}`",
        f"- Reward: `-{metadata['reward_block_coef']} * I[blocked] - {metadata['reward_delay_coef']} * delay_norm`",
        f"- Gamma: `{metadata['gamma']}`",
        f"- Checkpoint: `{report['checkpoint']}`", "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Best val loss | {report['train_result']['best_val_loss']:.5f} |",
        f"| Epochs trained | {report['train_result']['epochs_trained']} |",
        f"| Test loss | {test['loss']:.5f} |",
        f"| Pred mean ± std | {test['pred_mean']:.4f} ± {test['pred_std']:.4f} |",
        f"| Target mean ± std | {test['target_mean']:.4f} ± {test['target_std']:.4f} |",
        f"| Finite rate | {test['finite_rate']:.4f} |",
        f"| Value collapsed | {test['value_collapsed']} |",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="sa_hmarl/datasets/r_td_post_decision_k5m10")
    parser.add_argument("--output_dir", default="sa_hmarl/checkpoints/r_td_post_decision_k5m10")
    parser.add_argument("--hidden_dims", type=lambda value: tuple(int(x) for x in value.split(",")), default=(128, 64))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--gamma", type=float, default=0.95)
    parser.add_argument("--target_update_interval", type=int, default=5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=5)
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args._start_time = time.time()
    report = train(args)
    print(f"Training complete. Test loss={report['test_metrics']['loss']:.5f}")
