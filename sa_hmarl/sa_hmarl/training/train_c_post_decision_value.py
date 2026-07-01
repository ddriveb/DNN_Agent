"""Train and evaluate the C-side post-decision value network."""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import numpy as np
import torch
import torch.nn.functional as F

from sa_hmarl.agents.post_decision_value import (
    PostDecisionValueNetwork,
    pairwise_ranking_loss,
)


def _load_split(root: Path, split: str) -> Dict[str, np.ndarray]:
    with np.load(root / f"{split}.npz", allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _group_index_map(group_ids: np.ndarray) -> List[np.ndarray]:
    order = np.argsort(group_ids, kind="stable")
    sorted_groups = group_ids[order]
    boundaries = np.flatnonzero(np.diff(sorted_groups)) + 1
    return [part for part in np.split(order, boundaries) if len(part)]


def _group_batches(
    groups: List[np.ndarray], rng: np.random.RandomState,
    groups_per_batch: int, shuffle: bool,
) -> Iterable[np.ndarray]:
    order = np.arange(len(groups))
    if shuffle:
        rng.shuffle(order)
    for start in range(0, len(order), groups_per_batch):
        yield np.concatenate([groups[index] for index in order[start:start + groups_per_batch]])


def _rankdata(values: np.ndarray) -> np.ndarray:
    """Average ranks for ties, sufficient for Spearman diagnostics."""
    order = np.argsort(values, kind="stable")
    ranks = np.empty(len(values), dtype=float)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and values[order[end]] == values[order[start]]:
            end += 1
        ranks[order[start:end]] = 0.5 * (start + end - 1)
        start = end
    return ranks


def _spearman(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 2 or np.std(a) == 0 or np.std(b) == 0:
        return 0.0
    return float(np.corrcoef(_rankdata(a), _rankdata(b))[0, 1])


def evaluate_predictions(
    labels: np.ndarray, predictions: np.ndarray, group_ids: np.ndarray,
) -> Dict[str, float]:
    groups = _group_index_map(group_ids)
    top1, exact_top1, nonconstant_top1 = [], [], []
    regrets, normalized_regrets, group_spearman, pair_correct, pair_total = [], [], [], 0, 0
    for indices in groups:
        target = labels[indices]
        pred = predictions[indices]
        best_target = int(np.argmax(target))
        best_pred = int(np.argmax(pred))
        exact_top1.append(best_target == best_pred)
        value_optimal = bool(np.isclose(target[best_pred], target.max(), atol=1e-5, rtol=0.0))
        top1.append(value_optimal)
        value_range = float(target.max() - target.min())
        regret = float(target.max() - target[best_pred])
        regrets.append(regret)
        if value_range > 1e-8:
            nonconstant_top1.append(value_optimal)
            normalized_regrets.append(regret / value_range)
            group_spearman.append(_spearman(target, pred))
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                target_diff = float(target[i] - target[j])
                if abs(target_diff) <= 1e-5:
                    continue
                pair_total += 1
                pair_correct += int(np.sign(target_diff) == np.sign(float(pred[i] - pred[j])))
    return {
        "mae": float(np.mean(np.abs(labels - predictions))),
        "rmse": float(np.sqrt(np.mean((labels - predictions) ** 2))),
        "global_spearman": _spearman(labels, predictions),
        "mean_group_spearman": float(np.mean(group_spearman)) if group_spearman else 0.0,
        "top1_accuracy": float(np.mean(top1)) if top1 else 0.0,
        "exact_index_top1_accuracy": float(np.mean(exact_top1)) if exact_top1 else 0.0,
        "nonconstant_top1_accuracy": float(np.mean(nonconstant_top1)) if nonconstant_top1 else 0.0,
        "mean_regret": float(np.mean(regrets)) if regrets else 0.0,
        "mean_normalized_regret": float(np.mean(normalized_regrets)) if normalized_regrets else 0.0,
        "pairwise_accuracy": pair_correct / max(pair_total, 1),
        "evaluated_groups": len(groups),
        "nonconstant_groups": len(group_spearman),
    }


def _predict(
    model: PostDecisionValueNetwork, features: np.ndarray,
    mean: np.ndarray, std: np.ndarray, device: str, batch_size: int = 4096,
) -> np.ndarray:
    model.eval()
    outputs = []
    normalized = (features - mean) / std
    with torch.no_grad():
        for start in range(0, len(normalized), batch_size):
            tensor = torch.as_tensor(normalized[start:start + batch_size], dtype=torch.float32, device=device)
            outputs.append(model(tensor).cpu().numpy())
    return np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)


def benchmark_group_inference(
    model: PostDecisionValueNetwork, data: Dict[str, np.ndarray],
    mean: np.ndarray, std: np.ndarray, device: str,
) -> Dict[str, float]:
    model.eval()
    times = []
    groups = _group_index_map(data["group_ids"])
    with torch.no_grad():
        for indices in groups:
            tensor = torch.as_tensor(
                (data["features"][indices] - mean) / std,
                dtype=torch.float32, device=device,
            )
            started = time.perf_counter()
            model(tensor)
            if device.startswith("cuda"):
                torch.cuda.synchronize()
            times.append((time.perf_counter() - started) * 1000.0)
    return {
        "mean_group_inference_ms": float(np.mean(times)),
        "p95_group_inference_ms": float(np.percentile(times, 95)),
    }


def train_mode(
    mode: str, train: Dict[str, np.ndarray], val: Dict[str, np.ndarray],
    mean: np.ndarray, std: np.ndarray, args: argparse.Namespace,
) -> Tuple[PostDecisionValueNetwork, Dict[str, Any]]:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    model = PostDecisionValueNetwork(
        train["features"].shape[1], args.hidden_dims, args.dropout
    ).to(args.device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_groups = _group_index_map(train["group_ids"])
    rng = np.random.RandomState(args.seed)
    best_state, best_monitor, best_val_loss, patience_left = None, float("inf"), float("inf"), args.patience
    history = []
    normalized_train = ((train["features"] - mean) / std).astype(np.float32)

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses, regression_losses, ranking_losses = [], [], []
        for indices in _group_batches(train_groups, rng, args.groups_per_batch, True):
            x = torch.as_tensor(normalized_train[indices], dtype=torch.float32, device=args.device)
            y = torch.as_tensor(train["labels"][indices], dtype=torch.float32, device=args.device)
            group = torch.as_tensor(train["group_ids"][indices], dtype=torch.long, device=args.device)
            prediction = model(x)
            regression = F.huber_loss(prediction, y, delta=args.huber_delta)
            ranking = pairwise_ranking_loss(prediction, y, group)
            beta = args.rank_coef if mode == "joint" else 0.0
            loss = regression + beta * ranking
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            losses.append(float(loss.item()))
            regression_losses.append(float(regression.item()))
            ranking_losses.append(float(ranking.item()))

        val_prediction = _predict(model, val["features"], mean, std, args.device)
        val_loss = float(np.mean(np.where(
            np.abs(val_prediction - val["labels"]) <= args.huber_delta,
            0.5 * (val_prediction - val["labels"]) ** 2,
            args.huber_delta * (np.abs(val_prediction - val["labels"]) - 0.5 * args.huber_delta),
        )))
        val_metrics = evaluate_predictions(val["labels"], val_prediction, val["group_ids"])
        history.append({
            "epoch": epoch, "train_loss": float(np.mean(losses)),
            "train_regression_loss": float(np.mean(regression_losses)),
            "train_ranking_loss": float(np.mean(ranking_losses)),
            "val_loss": val_loss, **{f"val_{key}": value for key, value in val_metrics.items()},
        })
        monitor = val_metrics["mean_regret"] if mode == "joint" else val_loss
        if monitor < best_monitor - args.min_delta:
            best_monitor = monitor
            best_val_loss = val_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            patience_left = args.patience
        else:
            patience_left -= 1
        if epoch == 1 or epoch % args.log_every == 0:
            print(
                f"[{mode}][epoch={epoch}] train={np.mean(losses):.5f} "
                f"val={val_loss:.5f} top1={val_metrics['top1_accuracy']:.3f} "
                f"regret={val_metrics['mean_regret']:.4f}", flush=True,
            )
        if patience_left <= 0:
            break
    if best_state is None:
        raise RuntimeError("No post-decision checkpoint was selected")
    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": best_val_loss,
        "best_validation_monitor": best_monitor,
        "validation_monitor_name": "mean_regret" if mode == "joint" else "huber_loss",
        "epochs_trained": len(history),
        "history": history,
    }


def _save_checkpoint(
    path: Path, model: PostDecisionValueNetwork, mode: str,
    mean: np.ndarray, std: np.ndarray, metadata: Dict[str, Any],
    train_result: Dict[str, Any], val_metrics: Dict[str, Any], test_metrics: Dict[str, Any],
    args: argparse.Namespace,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "input_dim": len(metadata["feature_names"]),
        "hidden_dims": list(args.hidden_dims),
        "dropout": args.dropout,
        "feature_names": metadata["feature_names"],
        "feature_mean": mean.astype(np.float32),
        "feature_std": std.astype(np.float32),
        "label_name": metadata["label_name"],
        "mode": mode,
        "rank_coef": args.rank_coef if mode == "joint" else 0.0,
        "train_result": train_result,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
    }, path)


def run_training(args: argparse.Namespace) -> Dict[str, Any]:
    root = Path(args.dataset_dir)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    train, val, test = (_load_split(root, split) for split in ("train", "val", "test"))
    train_seeds, val_seeds, test_seeds = (set(data["seeds"].tolist()) for data in (train, val, test))
    if train_seeds & val_seeds or train_seeds & test_seeds or val_seeds & test_seeds:
        raise ValueError("Dataset seed leakage detected")
    mean = np.asarray(metadata["train_feature_mean"], dtype=np.float32)
    std = np.asarray(metadata["train_feature_std"], dtype=np.float32)
    modes = [mode.strip() for mode in args.modes.split(",") if mode.strip()]
    if any(mode not in ("regression", "joint") for mode in modes):
        raise ValueError(f"Unknown modes: {modes}")

    report = {"dataset": str(root), "modes": {}, "test_used_for_selection": False}
    for mode in modes:
        started = time.time()
        model, train_result = train_mode(mode, train, val, mean, std, args)
        val_prediction = _predict(model, val["features"], mean, std, args.device)
        test_prediction = _predict(model, test["features"], mean, std, args.device)
        val_metrics = evaluate_predictions(val["labels"], val_prediction, val["group_ids"])
        test_metrics = evaluate_predictions(test["labels"], test_prediction, test["group_ids"])
        test_metrics.update(benchmark_group_inference(model, test, mean, std, args.device))
        checkpoint = Path(args.output_dir) / f"c_post_decision_{mode}.pt"
        _save_checkpoint(
            checkpoint, model, mode, mean, std, metadata,
            train_result, val_metrics, test_metrics, args,
        )
        report["modes"][mode] = {
            "checkpoint": str(checkpoint), "train": train_result,
            "validation": val_metrics, "test": test_metrics,
            "elapsed_seconds": time.time() - started,
        }
    # Model choice is validation-only. Test metrics are reported after selection.
    selected_mode = min(
        modes,
        key=lambda mode: report["modes"][mode]["validation"]["mean_regret"],
    )
    report["selected_mode_by_validation"] = selected_mode
    selected = report["modes"][selected_mode]["test"]
    report["verdict"] = (
        "PROCEED_TO_CLOSED_LOOP_INTEGRATION"
        if selected["mean_group_spearman"] >= args.min_group_spearman
        and selected["top1_accuracy"] >= args.min_top1
        and selected["mean_normalized_regret"] <= args.max_normalized_regret
        and selected["mean_group_inference_ms"] < args.max_mean_inference_ms
        else "STOP_AND_AUDIT_POST_DECISION_MODEL"
    )
    output = Path(args.output_dir) / "training_report.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    lines = [
        "# C-Side Post-Decision Value Training", "",
        "| Mode | Val MAE | Val Top-1 | Test MAE | Test Spearman | Test Top-1 | Test Regret | Mean/P95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        val_m, test_m = report["modes"][mode]["validation"], report["modes"][mode]["test"]
        lines.append(
            f"| {mode} | {val_m['mae']:.4f} | {val_m['top1_accuracy']:.2%} | "
            f"{test_m['mae']:.4f} | {test_m['global_spearman']:.4f} | "
            f"{test_m['top1_accuracy']:.2%} | {test_m['mean_regret']:.4f} | "
            f"{test_m['mean_group_inference_ms']:.3f}/{test_m['p95_group_inference_ms']:.3f} |"
        )
    lines += ["", f"Selected by validation: **{selected_mode}**", "",
              f"**Verdict: {report['verdict']}**"]
    (Path(args.output_dir) / "training_report.md").write_text("\n".join(lines), encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default="sa_hmarl/datasets/c_post_decision")
    parser.add_argument("--output_dir", default="sa_hmarl/checkpoints/post_decision")
    parser.add_argument("--modes", default="regression,joint")
    parser.add_argument("--hidden_dims", type=lambda value: tuple(int(x) for x in value.split(",")), default=(128, 64))
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--min_delta", type=float, default=1e-5)
    parser.add_argument("--groups_per_batch", type=int, default=32)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-5)
    parser.add_argument("--rank_coef", type=float, default=0.2)
    parser.add_argument("--huber_delta", type=float, default=1.0)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_every", type=int, default=10)
    parser.add_argument("--min_group_spearman", type=float, default=0.4)
    parser.add_argument("--min_top1", type=float, default=0.5)
    parser.add_argument("--max_normalized_regret", type=float, default=0.2)
    parser.add_argument("--max_mean_inference_ms", type=float, default=10.0)
    return parser


if __name__ == "__main__":
    result = run_training(build_parser().parse_args())
    print(result["verdict"])
