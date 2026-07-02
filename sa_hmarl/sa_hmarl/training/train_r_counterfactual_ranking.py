"""Train a planner-distilled listwise ranking network from R-side groups.

Each training example is a group of candidate R actions at a single state.
The offline teacher is an H-step counterfactual planner: every candidate action
is evaluated by branching the simulator, executing that action, and rolling out
future requests.  The target is a softmax over those planner returns (higher =
better).  The network learns a cheap score f_theta(phi(s, a_i)) and is trained
with KL(q || p) plus a small regression loss on normalized returns.

This is planner imitation / planner distillation, not TD bootstrapping and not
behavioral cloning of PPO-R.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import stats

from sa_hmarl.agents.counterfactual_r_ranker import (
    CounterfactualActionValueRanker,
    build_counterfactual_r_ranker,
)


def _atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _feasibility_mask(features: np.ndarray, feature_names: List[str]) -> np.ndarray:
    """Return a mask that is True only for path-mod-feasible candidates.

    Must be computed on *raw* (unnormalized) features because normalization
    collapses the binary feasibility indicator to near-zero.
    """
    try:
        idx = feature_names.index("path_mod_feasible")
    except ValueError:
        return np.ones(features.shape[:2], dtype=bool)
    return features[..., idx] > 0.5


def _masked_softmax(values: torch.Tensor, mask: torch.Tensor, temperature: float) -> torch.Tensor:
    """Softmax over valid entries with temperature.  Invalid entries get zero weight."""
    if not mask.any(dim=-1).all():
        # Keep behavior defined for defensive use; valid datasets should have
        # at least two feasible candidates per group.
        mask = mask.clone()
        empty = ~mask.any(dim=-1)
        if empty.any():
            mask[empty, 0] = True
    logits = values.masked_fill(~mask, float("-inf")) / temperature
    probs = F.softmax(logits, dim=-1)
    probs = probs.masked_fill(~mask, 0.0)
    return probs


def _ranking_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    tau_label: float,
    tau_model: float,
) -> torch.Tensor:
    q = _masked_softmax(returns, mask, tau_label)
    p = _masked_softmax(scores, mask, tau_model)
    eps = 1e-10
    return (q * (torch.log(q + eps) - torch.log(p + eps))).sum(dim=-1).mean()


def _regression_loss(
    scores: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    return_mean: float,
    return_std: float,
) -> torch.Tensor:
    target = (returns - return_mean) / (return_std + 1e-8)
    diff = F.smooth_l1_loss(scores, target, reduction="none")
    return (diff * mask.float()).sum() / mask.sum().clamp_min(1.0)


def _compute_group_metrics(
    scores: np.ndarray,
    returns: np.ndarray,
    mask: np.ndarray,
    ppo_action_index: np.ndarray,
) -> Dict[str, Any]:
    top1_correct = []
    top3_correct = []
    spearmans = []
    model_regrets = []
    ppo_regrets = []
    ppo_agreements = []

    for i in range(len(mask)):
        valid = mask[i]
        if valid.sum() < 2:
            continue
        s = scores[i, valid]
        r = returns[i, valid]
        best_idx = int(r.argmax())

        order = np.argsort(-s, kind="stable")
        top1_correct.append(int(order[0] == best_idx))
        top3_correct.append(int(best_idx in order[: min(3, len(order))]))

        # Spearman correlation.
        rank_score = stats.rankdata(-s)
        rank_return = stats.rankdata(-r)
        if len(s) >= 2:
            rho, _ = stats.spearmanr(rank_score, rank_return)
            if not np.isnan(rho):
                spearmans.append(float(rho))

        model_regrets.append(float(r[best_idx] - r[order[0]]))

        ppo_idx = int(ppo_action_index[i])
        if 0 <= ppo_idx < len(r):
            ppo_regrets.append(float(r[best_idx] - r[ppo_idx]))
            ppo_agreements.append(int(order[0] == ppo_idx))

    def _mean(values):
        return float(np.mean(values)) if values else 0.0

    return {
        "top1_accuracy": _mean(top1_correct),
        "top3_accuracy": _mean(top3_correct),
        "spearman_mean": _mean(spearmans),
        "model_regret_mean": _mean(model_regrets),
        "ppo_regret_mean": _mean(ppo_regrets),
        "ppo_agreement": _mean(ppo_agreements),
        "score_std": float(np.std(scores[mask])) if mask.any() else 0.0,
    }


def _selection_score(metrics: Dict[str, Any], metric_name: str) -> float:
    """Return a higher-is-better score for checkpoint selection."""
    if metric_name == "top1":
        return float(metrics["top1_accuracy"])
    if metric_name == "top3":
        return float(metrics["top3_accuracy"])
    if metric_name == "spearman":
        return float(metrics["spearman_mean"])
    if metric_name == "kl":
        return -float(metrics["kl"])
    if metric_name == "regret":
        return -float(metrics["model_regret_mean"])
    raise ValueError(f"Unknown selection metric: {metric_name}")


def _evaluate(
    features: torch.Tensor,
    returns: torch.Tensor,
    mask: torch.Tensor,
    ppo_action_index: torch.Tensor,
    feas_mask: torch.Tensor,
    model: CounterfactualActionValueRanker,
    return_mean: float,
    return_std: float,
    tau_label: float,
    tau_model: float,
    batch_size: int,
    device: str,
) -> Dict[str, Any]:
    model.eval()
    eff_mask = mask & feas_mask
    all_scores = []
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            end = min(start + batch_size, features.size(0))
            x = features[start:end].to(device)
            scores = model(x, eff_mask[start:end].to(device))  # [B, C]
            all_scores.append(scores.cpu().numpy())
    scores = np.concatenate(all_scores, axis=0)
    metrics = _compute_group_metrics(scores, returns.numpy(), eff_mask.numpy(), ppo_action_index.numpy())

    # KL on the split.
    total_kl = 0.0
    total_groups = 0
    with torch.no_grad():
        for start in range(0, features.size(0), batch_size):
            end = min(start + batch_size, features.size(0))
            x = features[start:end].to(device)
            mask_t = eff_mask[start:end].to(device)
            scores_t = model(x, mask_t)
            returns_t = returns[start:end].to(device)
            kl = _ranking_loss(scores_t, returns_t, mask_t, tau_label, tau_model)
            total_kl += float(kl.item()) * (end - start)
            total_groups += (end - start)
    metrics["kl"] = total_kl / max(total_groups, 1)
    return metrics


def _compute_group_weights(
    train_data: Dict[str, np.ndarray],
    metadata: Dict[str, Any],
    args: argparse.Namespace,
) -> np.ndarray:
    """Return normalized sampling weights for each training group."""
    returns = train_data["returns"]
    mask = train_data["mask"]
    n = returns.shape[0]
    if n == 0:
        return np.ones(1, dtype=float) / 1.0

    # Return contrast per group.
    ranges = np.zeros(n, dtype=float)
    for i in range(n):
        valid = returns[i, mask[i]]
        if valid.size > 0:
            ranges[i] = float(valid.max() - valid.min())
    global_std = float(np.std(ranges)) + 1e-8
    norm_ranges = ranges / global_std

    # Pressure proxy: 1 - raw_r_valid_ratio (higher = fewer legal R actions).
    feature_names = list(metadata.get("feature_names", []))
    try:
        rvr_idx = feature_names.index("raw_r_valid_ratio")
        pressures = np.zeros(n, dtype=float)
        for i in range(n):
            valid_vals = train_data["features"][i, mask[i], rvr_idx]
            if valid_vals.size > 0:
                pressures[i] = 1.0 - float(np.mean(valid_vals))
    except ValueError:
        pressures = np.zeros(n, dtype=float)

    # Future NSB positive indicator.
    nsb = train_data.get("future_nsb_counts", np.zeros_like(mask, dtype=np.int64))
    nsb_positive = np.array([
        int(np.any(nsb[i, mask[i]] > 0)) for i in range(n)
    ], dtype=float)

    weights = (
        1.0
        + float(args.balance_alpha) * norm_ranges
        + float(args.balance_beta) * pressures
        + float(args.balance_gamma) * nsb_positive
    )
    weights = np.maximum(weights, 1e-6)
    weights /= weights.sum()
    return weights


def train(args: argparse.Namespace) -> Dict[str, Any]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    root = Path(args.dataset_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))

    train_data = dict(np.load(root / "train.npz", allow_pickle=False))
    val_data = dict(np.load(root / "val.npz", allow_pickle=False))
    test_data = dict(np.load(root / "test.npz", allow_pickle=False))

    mean = np.asarray(metadata["train_feature_mean"], dtype=np.float32)
    std = np.maximum(
        np.asarray(metadata["train_feature_std"], dtype=np.float32),
        float(args.feature_std_floor),
    )
    input_dim = train_data["features"].shape[2]

    # Compute feasibility masks on raw features before normalization.
    train_feas_np = _feasibility_mask(train_data["features"], metadata["feature_names"])
    val_feas_np = _feasibility_mask(val_data["features"], metadata["feature_names"])
    test_feas_np = _feasibility_mask(test_data["features"], metadata["feature_names"])

    # Normalize features (valid entries only).
    def _norm(data, feas_np):
        f = (data["features"].astype(np.float32) - mean) / std
        return {
            "features": torch.as_tensor(f, dtype=torch.float32),
            "returns": torch.as_tensor(data["returns"], dtype=torch.float32),
            "mask": torch.as_tensor(data["mask"], dtype=torch.bool),
            "ppo_action_index": torch.as_tensor(data["ppo_action_index"], dtype=torch.long),
            "feas": torch.as_tensor(feas_np, dtype=torch.bool),
        }

    train_t = _norm(train_data, train_feas_np)
    val_t = _norm(val_data, val_feas_np)
    test_t = _norm(test_data, test_feas_np)

    # Global return statistics for regression target (feasible candidates only).
    train_eff = train_t["mask"] & train_t["feas"]
    valid_returns = train_t["returns"][train_eff].numpy()
    return_mean = float(np.mean(valid_returns))
    return_std = float(np.std(valid_returns))

    model = build_counterfactual_r_ranker(
        args.model_type, input_dim, args.hidden_dims, args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    n = train_t["features"].size(0)
    best_val_top1 = -1.0
    best_selection_score = float("-inf")
    best_state = None
    patience_left = args.patience
    history: List[Dict[str, Any]] = []

    # Optional group-level balanced sampling weights.
    group_weights = _compute_group_weights(train_data, metadata, args) if args.balanced_sampler else None

    for epoch in range(1, args.epochs + 1):
        model.train()
        if group_weights is not None:
            order = np.random.choice(n, size=n, replace=True, p=group_weights)
        else:
            order = np.random.permutation(n)
        train_losses = []
        for start in range(0, n, args.batch_size):
            indices = order[start:start + args.batch_size]
            x = train_t["features"][indices].to(args.device)
            r = train_t["returns"][indices].to(args.device)
            m = train_t["mask"][indices].to(args.device)
            feas_m = train_t["feas"][indices].to(args.device)
            eff_m = m & feas_m

            scores = model(x, eff_m)  # [B, C]
            loss_rank = _ranking_loss(scores, r, eff_m, args.tau_label, args.tau_model)
            loss_reg = _regression_loss(scores, r, eff_m, return_mean, return_std)
            loss = loss_rank + args.reg_weight * loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            train_losses.append(float(loss.item()))

        # Evaluate.
        train_metrics = _evaluate(
            train_t["features"], train_t["returns"], train_t["mask"], train_t["ppo_action_index"],
            train_t["feas"], model, return_mean, return_std, args.tau_label, args.tau_model,
            args.eval_batch_size, args.device,
        )
        val_metrics = _evaluate(
            val_t["features"], val_t["returns"], val_t["mask"], val_t["ppo_action_index"],
            val_t["feas"], model, return_mean, return_std, args.tau_label, args.tau_model,
            args.eval_batch_size, args.device,
        )
        train_metrics["loss"] = float(np.mean(train_losses))

        history.append({
            "epoch": epoch,
            "train": train_metrics,
            "val": val_metrics,
        })

        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[epoch={epoch}] train_loss={train_metrics['loss']:.4f} "
                f"val_top1={val_metrics['top1_accuracy']:.2%} "
                f"val_top3={val_metrics['top3_accuracy']:.2%} "
                f"val_spearman={val_metrics['spearman_mean']:.3f} "
                f"val_ppo_agreement={val_metrics['ppo_agreement']:.2%} "
                f"score_std={val_metrics['score_std']:.4f}",
                flush=True,
            )

        current_selection_score = _selection_score(val_metrics, args.selection_metric)
        if current_selection_score > best_selection_score + args.min_delta:
            best_val_top1 = val_metrics["top1_accuracy"]
            best_selection_score = current_selection_score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if patience_left <= 0:
            print(f"Early stopping at epoch {epoch}", flush=True)
            break

    if best_state is None:
        best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)

    test_metrics = _evaluate(
        test_t["features"], test_t["returns"], test_t["mask"], test_t["ppo_action_index"],
        test_t["feas"], model, return_mean, return_std, args.tau_label, args.tau_model,
        args.eval_batch_size, args.device,
    )

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_path = output / "ranking_model.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": input_dim,
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "model_type": args.model_type,
        "feature_names": metadata["feature_names"],
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "return_mean": return_mean,
        "return_std": return_std,
        "tau_label": args.tau_label,
        "tau_model": args.tau_model,
        "best_val_top1": best_val_top1,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_selection_score,
        "test_metrics": test_metrics,
        "candidate_mode": metadata.get("candidate_mode", "v1"),
        "max_candidates": metadata.get("max_candidates", args.batch_size),
        "ppo_top_k": metadata.get("ppo_top_k", 8),
        "num_random_candidates": metadata.get("num_random_candidates", 5),
        "min_candidates": metadata.get("min_candidates", 15),
        "candidate_seed": metadata.get("candidate_seed", 12345),
    }, checkpoint_path)

    report = {
        "dataset": str(root),
        "checkpoint": str(checkpoint_path),
        "best_val_top1": best_val_top1,
        "selection_metric": args.selection_metric,
        "best_selection_score": best_selection_score,
        "test_metrics": test_metrics,
        "history": history,
        "config": vars(args),
        "elapsed_seconds": time.time() - getattr(args, "_start_time", time.time()),
    }
    _atomic_write_json(output / "training_report.json", report)
    _write_markdown(output / "training_report.md", report, metadata)
    return report


def _write_markdown(path: Path, report: Dict[str, Any], metadata: Dict[str, Any]) -> None:
    test = report["test_metrics"]
    lines = [
        "# Counterfactual R-Side Ranking Training", "",
        f"- Dataset: `{report['dataset']}`",
        f"- Return coefs: current_block={metadata['return_coefs']['current_block']}, "
        f"future_block={metadata['return_coefs']['future_block']}, "
        f"future_nsb={metadata['return_coefs']['future_nsb']}, "
        f"delay={metadata['return_coefs']['delay']}, fs={metadata['return_coefs']['fs']}"
        + (
            f", path_penalty={metadata['return_coefs'].get('path_penalty', 0)}, "
            f"fs_penalty={metadata['return_coefs'].get('fs_penalty', 0)}"
            if metadata['return_coefs'].get('path_penalty') or metadata['return_coefs'].get('fs_penalty')
            else ""
        ),
        f"- Horizon: {metadata['horizon']}",
        f"- Checkpoint: `{report['checkpoint']}`", "",
        f"- Model type: `{report['config'].get('model_type', 'mlp')}`", "",
        f"- Selection metric: `{report.get('selection_metric', 'top1')}`", "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Best checkpoint val top-1 | {report['best_val_top1']:.2%} |",
        f"| Best `{report.get('selection_metric', 'top1')}` selection score | "
        f"{report.get('best_selection_score', 0):.4f} |",
        f"| Test top-1 | {test['top1_accuracy']:.2%} |",
        f"| Test top-3 | {test['top3_accuracy']:.2%} |",
        f"| Test Spearman mean | {test['spearman_mean']:.3f} |",
        f"| Test model regret | {test['model_regret_mean']:.4f} |",
        f"| Test PPO regret | {test['ppo_regret_mean']:.4f} |",
        f"| Test PPO agreement | {test['ppo_agreement']:.2%} |",
        f"| Test score std | {test['score_std']:.4f} |",
        f"| Test KL | {test['kl']:.4f} |",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="sa_hmarl/datasets/r_counterfactual_ranking_v1_2_mixed_low")
    parser.add_argument("--output_dir", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low")
    parser.add_argument(
        "--model_type",
        default="mlp",
        choices=["mlp", "deepset", "set_transformer"],
    )
    parser.add_argument("--selection_metric", default="top1",
                        choices=["top1", "top3", "spearman", "kl", "regret"])
    parser.add_argument("--hidden_dims", type=lambda value: tuple(int(x) for x in value.split(",")), default=(128, 64))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--eval_batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--tau_label", type=float, default=0.5)
    parser.add_argument("--tau_model", type=float, default=1.0)
    parser.add_argument("--reg_weight", type=float, default=0.1)
    parser.add_argument("--feature_std_floor", type=float, default=1e-6)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=5)
    parser.add_argument("--balanced_sampler", action="store_true",
                        help="Use group-level weighted sampling: upweight high return-contrast / high-pressure / NSB-positive groups.")
    parser.add_argument("--balance_alpha", type=float, default=1.0,
                        help="Weight on normalized return range.")
    parser.add_argument("--balance_beta", type=float, default=1.0,
                        help="Weight on pressure (1 - raw_r_valid_ratio).")
    parser.add_argument("--balance_gamma", type=float, default=1.0,
                        help="Weight on future-NSB-positive indicator.")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    args._start_time = time.time()
    report = train(args)
    print(
        f"Training complete. Best checkpoint val top-1={report['best_val_top1']:.2%} "
        f"selection_metric={report.get('selection_metric', 'top1')} "
        f"selection_score={report.get('best_selection_score', 0):.4f} "
        f"Test top-1={report['test_metrics']['top1_accuracy']:.2%}"
    )
