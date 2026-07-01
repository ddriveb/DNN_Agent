"""Train a lightweight C-side downstream-RMSA-aware ranker."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats


class CDownstreamRanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dims=(64, 64), dropout: float = 0.05):
        super().__init__()
        layers: List[nn.Module] = []
        prev = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(prev, hidden), nn.ReLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            prev = hidden
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


def _load_split(dataset_dir: Path, name: str) -> Dict[str, np.ndarray]:
    data = np.load(dataset_dir / f"{name}.npz")
    return {key: data[key] for key in data.files}


def _group_indices(group_ids: np.ndarray) -> List[np.ndarray]:
    groups = []
    for gid in np.unique(group_ids):
        idx = np.flatnonzero(group_ids == gid)
        if len(idx) >= 2:
            groups.append(idx)
    return groups


def _soft_targets(values: torch.Tensor, tau: float) -> torch.Tensor:
    centered = values - values.max()
    return F.softmax(centered / max(tau, 1e-6), dim=0)


def _evaluate(model, split, mean, std, device: str) -> Dict[str, float]:
    model.eval()
    groups = _group_indices(split["group_ids"])
    top1 = []
    ranks = []
    regrets = []
    spearmans = []
    with torch.no_grad():
        for idx in groups:
            x = (split["features"][idx] - mean) / std
            y = split["returns"][idx]
            scores = model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
            pred_order = np.argsort(-scores)
            true_order = np.argsort(-y)
            best_true = int(true_order[0])
            best_pred = int(pred_order[0])
            top1.append(best_pred == best_true)
            ranks.append(int(np.where(pred_order == best_true)[0][0]) + 1)
            regrets.append(float(y[best_true] - y[best_pred]))
            if len(y) >= 3 and np.std(scores) > 0 and np.std(y) > 0:
                spearmans.append(float(stats.spearmanr(scores, y).statistic))
    return {
        "groups": int(len(groups)),
        "top1_accuracy": float(np.mean(top1)) if top1 else 0.0,
        "mean_rank": float(np.mean(ranks)) if ranks else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "spearman_mean": float(np.mean(spearmans)) if spearmans else 0.0,
    }


def train(args: argparse.Namespace) -> Dict[str, Any]:
    dataset_dir = Path(args.dataset_dir)
    metadata = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    train_split = _load_split(dataset_dir, "train")
    val_split = _load_split(dataset_dir, "val")
    test_split = _load_split(dataset_dir, "test")
    mean = np.load(dataset_dir / "feature_mean.npy").astype(np.float32)
    std = np.load(dataset_dir / "feature_std.npy").astype(np.float32)
    std[std < 1e-6] = 1.0

    device = args.device
    model = CDownstreamRanker(
        input_dim=train_split["features"].shape[1],
        hidden_dims=tuple(args.hidden_dims),
        dropout=args.dropout,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    groups = _group_indices(train_split["group_ids"])
    rng = np.random.RandomState(args.seed)

    history = []
    best_val = -1.0
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        rng.shuffle(groups)
        losses = []
        model.train()
        for idx in groups:
            x = (train_split["features"][idx] - mean) / std
            y = train_split["returns"][idx]
            x_t = torch.as_tensor(x, dtype=torch.float32, device=device)
            y_t = torch.as_tensor(y, dtype=torch.float32, device=device)
            scores = model(x_t)
            target = _soft_targets(y_t, args.tau)
            loss = -(target * F.log_softmax(scores, dim=0)).sum()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            opt.step()
            losses.append(float(loss.item()))

        train_m = _evaluate(model, train_split, mean, std, device)
        val_m = _evaluate(model, val_split, mean, std, device)
        train_m["loss"] = float(np.mean(losses)) if losses else 0.0
        history.append({"epoch": epoch, "train": train_m, "val": val_m})
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[epoch={epoch}] loss={train_m['loss']:.4f} "
                f"val_top1={val_m['top1_accuracy']:.2%} "
                f"val_regret={val_m['mean_regret']:.4f} "
                f"val_spearman={val_m['spearman_mean']:.3f}",
                flush=True,
            )
        score = val_m["top1_accuracy"] - 0.1 * val_m["mean_regret"]
        if score > best_val + args.min_delta:
            best_val = score
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print(f"Early stopping at epoch {epoch}", flush=True)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    test_m = _evaluate(model, test_split, mean, std, device)
    report = {
        "dataset": args.dataset_dir,
        "feature_names": metadata["feature_names"],
        "history": history,
        "best_val_score": best_val,
        "test_metrics": test_m,
        "config": vars(args),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = output_dir / "c_downstream_ranker.pt"
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": train_split["features"].shape[1],
        "hidden_dims": tuple(args.hidden_dims),
        "dropout": args.dropout,
        "feature_names": metadata["feature_names"],
        "feature_mean": mean,
        "feature_std": std,
        "metadata": metadata,
    }, ckpt_path)
    report["checkpoint"] = str(ckpt_path)
    (output_dir / "training_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    _write_md(output_dir / "training_report.md", report)
    return report


def _write_md(path: Path, report: Dict[str, Any]) -> None:
    test = report["test_metrics"]
    lines = [
        "# C-side Downstream-RMSA Ranker Training",
        "",
        f"- Dataset: `{report['dataset']}`",
        f"- Checkpoint: `{report['checkpoint']}`",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Test groups | {test['groups']} |",
        f"| Test top-1 | {test['top1_accuracy']:.2%} |",
        f"| Test mean rank | {test['mean_rank']:.2f} |",
        f"| Test regret | {test['mean_regret']:.4f} |",
        f"| Test Spearman | {test['spearman_mean']:.3f} |",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="sa_hmarl/datasets/c_downstream_ranker")
    parser.add_argument("--output_dir", default="sa_hmarl/checkpoints/c_downstream_ranker")
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--min_delta", type=float, default=1e-4)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.05)
    parser.add_argument("--hidden_dims", type=int, nargs="+", default=[64, 64])
    parser.add_argument("--tau", type=float, default=0.25)
    parser.add_argument("--max_grad_norm", type=float, default=5.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=5)
    return parser


if __name__ == "__main__":
    parsed = build_parser().parse_args()
    result = train(parsed)
    print(
        f"Training complete. Test top1={result['test_metrics']['top1_accuracy']:.2%} "
        f"regret={result['test_metrics']['mean_regret']:.4f}"
    )
