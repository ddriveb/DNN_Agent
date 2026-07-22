"""Train the native pure-RMSA candidate Ranker on common-future groups."""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class NativeRMSARanker(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: Sequence[int] = (128, 64)):
        super().__init__()
        layers: List[nn.Module] = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, int(hidden)), nn.SiLU()])
            previous = int(hidden)
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.net(features).squeeze(-1)


def _atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    os.replace(tmp, path)


def _utility(metric: Dict[str, Any], horizon: int) -> float:
    count = float(metric["block_count"])
    discounted = float(metric["discounted_blocks"])
    first = horizon + 1 if metric["first_block_time"] is None else int(metric["first_block_time"])
    # One fewer block dominates every timing difference within the horizon.
    return float(-count - 0.01 * discounted + 0.0001 * first)


def _load_groups(root: Path, horizon: int) -> Tuple[List[Dict[str, Any]], List[str]]:
    groups: List[Dict[str, Any]] = []
    feature_names: Optional[List[str]] = None
    for path in sorted(root.glob("seed_*/COMMON_FUTURE_RESULTS.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        names = list(payload["feature_names"])
        if feature_names is None:
            feature_names = names
        elif names != feature_names:
            raise RuntimeError(f"Feature schema mismatch in {path}")
        seed = int(payload["seed"])
        for group in payload["groups"]:
            rows = [row for row in group["rows"] if row["continuation"] == "native_proposer_top1"]
            if len(rows) != len(group["candidate_features"]):
                raise RuntimeError(f"Candidate/feature mismatch in {path} group {group['request_index']}")
            features = np.asarray(group["candidate_features"], dtype=np.float32)
            metrics = [row["metrics"][str(horizon)] for row in rows]
            utilities = np.asarray([_utility(metric, horizon) for metric in metrics], dtype=np.float32)
            counts = np.asarray([metric["block_count"] for metric in metrics], dtype=np.int64)
            deployed = next(i for i, row in enumerate(rows) if row["is_deployed"])
            groups.append({
                "seed": seed, "request_index": int(group["request_index"]),
                "stratum": group["stratum"], "features": features,
                "utilities": utilities, "counts": counts, "deployed_index": deployed,
                "sensitive": bool(np.ptp(utilities) > 1e-12),
            })
    if feature_names is None:
        raise RuntimeError(f"No completed dataset shards found under {root}")
    return groups, feature_names


def _metrics(
    model: NativeRMSARanker,
    groups: Sequence[Dict[str, Any]],
    mean: np.ndarray,
    std: np.ndarray,
    device: str,
) -> Dict[str, float]:
    model.eval()
    count_regrets = []
    utility_regrets = []
    deployed_count_regrets = []
    best_hits = []
    sensitive = 0
    with torch.no_grad():
        for group in groups:
            x = (group["features"] - mean) / std
            scores = model(torch.as_tensor(x, dtype=torch.float32, device=device)).cpu().numpy()
            selected = int(np.argmax(scores))
            counts = group["counts"]
            utilities = group["utilities"]
            count_regrets.append(float(counts[selected] - counts.min()))
            utility_regrets.append(float(utilities.max() - utilities[selected]))
            deployed_count_regrets.append(float(counts[group["deployed_index"]] - counts.min()))
            best_hits.append(float(counts[selected] == counts.min()))
            sensitive += int(group["sensitive"])
    model.train()
    return {
        "groups": len(groups), "sensitive_groups": sensitive,
        "mean_count_regret": float(np.mean(count_regrets)) if count_regrets else 0.0,
        "mean_utility_regret": float(np.mean(utility_regrets)) if utility_regrets else 0.0,
        "deployed_mean_count_regret": float(np.mean(deployed_count_regrets)) if deployed_count_regrets else 0.0,
        "best_count_hit_rate": float(np.mean(best_hits)) if best_hits else 0.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--horizon", type=int, default=100)
    parser.add_argument("--val-seeds", default="8307")
    parser.add_argument("--test-seeds", default="8308")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--reg-weight", type=float, default=1.0)
    parser.add_argument("--groups-per-step", type=int, default=16)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    status: Dict[str, Any] = {"stage": "load", "started": time.time(), "args": vars(args)}
    _atomic_json(output / "RUN_STATUS.json", status)
    groups, feature_names = _load_groups(Path(args.dataset_root), args.horizon)
    val_seeds = {int(x) for x in args.val_seeds.split(",")}
    test_seeds = {int(x) for x in args.test_seeds.split(",")}
    if val_seeds & test_seeds:
        raise ValueError(f"Validation/test seed overlap: {sorted(val_seeds & test_seeds)}")
    train = [g for g in groups if g["seed"] not in val_seeds | test_seeds]
    val = [g for g in groups if g["seed"] in val_seeds]
    test = [g for g in groups if g["seed"] in test_seeds]
    if not train or not val or not test:
        raise RuntimeError(f"Empty seed split: train={len(train)} val={len(val)} test={len(test)}")
    train_candidates = np.concatenate([g["features"] for g in train], axis=0)
    mean = train_candidates.mean(axis=0).astype(np.float32)
    std = train_candidates.std(axis=0).astype(np.float32)
    std[std < 1e-6] = 1.0
    model = NativeRMSARanker(len(feature_names)).to(args.device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    checkpoint = output / "native_ranker_best.pt"
    best_key = (float("inf"), float("inf"), float("inf"))
    history = []
    status.update({
        "stage": "train", "train_groups": len(train), "val_groups": len(val),
        "test_groups": len(test), "feature_dim": len(feature_names),
    })
    _atomic_json(output / "RUN_STATUS.json", status)

    for epoch in range(1, args.epochs + 1):
        order = np.random.permutation(len(train))
        optimizer.zero_grad()
        losses = []
        pending = 0
        for position, group_index in enumerate(order, start=1):
            group = train[int(group_index)]
            x = (group["features"] - mean) / std
            utilities = group["utilities"]
            target_centered = utilities - utilities.mean()
            scores = model(torch.as_tensor(x, dtype=torch.float32, device=args.device))
            target = torch.as_tensor(utilities / args.temperature, dtype=torch.float32, device=args.device)
            target_prob = torch.softmax(target, dim=0)
            listwise = F.kl_div(torch.log_softmax(scores, dim=0), target_prob, reduction="batchmean")
            centered_scores = scores - scores.mean()
            regression = F.smooth_l1_loss(
                centered_scores,
                torch.as_tensor(target_centered, dtype=torch.float32, device=args.device),
            )
            loss = listwise + args.reg_weight * regression
            (loss / args.groups_per_step).backward()
            losses.append(float(loss.item()))
            pending += 1
            if pending == args.groups_per_step or position == len(order):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                pending = 0
        val_metrics = _metrics(model, val, mean, std, args.device)
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val": val_metrics}
        history.append(row)
        key = (val_metrics["mean_count_regret"], val_metrics["mean_utility_regret"], row["train_loss"])
        if key < best_key:
            best_key = key
            torch.save({
                "model_state": model.state_dict(), "input_dim": len(feature_names),
                "hidden_dims": (128, 64), "feature_names": feature_names,
                "feature_mean": mean, "feature_std": std,
                "horizon": args.horizon, "continuation": "native_proposer_top1",
                "label": "-count - 0.01*discounted + 0.0001*first_block_time",
                "epoch": epoch, "val_metrics": val_metrics,
            }, checkpoint)
        if epoch == 1 or epoch % 10 == 0 or epoch == args.epochs:
            print(json.dumps(row), flush=True)

    saved = torch.load(checkpoint, map_location=args.device, weights_only=False)
    model.load_state_dict(saved["model_state"])
    train_metrics = _metrics(model, train, mean, std, args.device)
    val_metrics = _metrics(model, val, mean, std, args.device)
    test_metrics = _metrics(model, test, mean, std, args.device)
    result = {
        "protocol": "pure_rmsa_native_ranker_r2", "checkpoint": str(checkpoint),
        "selected_epoch": int(saved["epoch"]), "feature_names": feature_names,
        "split": {
            "train_seeds": sorted({g['seed'] for g in train}),
            "val_seeds": sorted(val_seeds), "test_seeds": sorted(test_seeds),
        },
        "train": train_metrics, "validation": val_metrics, "test": test_metrics,
        "history": history,
    }
    _atomic_json(output / "TRAINING_RESULTS.json", result)
    lines = [
        "# Pure-RMSA Native Ranker Training", "",
        f"Selected epoch: {saved['epoch']}", "",
        "| Split | Groups | Sensitive | Model count regret | Deployed count regret | Best hit |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in (("train", train_metrics), ("validation", val_metrics), ("test", test_metrics)):
        lines.append(
            f"| {name} | {metrics['groups']} | {metrics['sensitive_groups']} | "
            f"{metrics['mean_count_regret']:.4f} | {metrics['deployed_mean_count_regret']:.4f} | "
            f"{100*metrics['best_count_hit_rate']:.2f}% |"
        )
    (output / "TRAINING_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    status.update({"stage": "complete", "completed": time.time(), "selected_epoch": int(saved["epoch"])})
    _atomic_json(output / "RUN_STATUS.json", status)


if __name__ == "__main__":
    main()
