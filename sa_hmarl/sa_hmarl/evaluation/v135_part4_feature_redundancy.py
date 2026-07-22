"""Part 4: feature redundancy and incremental predictive value."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import stats


DATASET_DIR = Path("sa_hmarl/datasets/v135_paired_feature_ablation_medium")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_split(split: str):
    d = np.load(str(DATASET_DIR / f"{split}.npz"), allow_pickle=True)
    mask = d["mask"].astype(bool)
    X = d["features_poststate_v1"].astype(np.float32)
    y = d["returns"].astype(np.float32)
    return X, y, mask


def _normalize(X_train: np.ndarray, X_test: np.ndarray, mask_train: np.ndarray, mask_test: np.ndarray):
    valid = X_train[mask_train]
    mean = valid.mean(axis=0)
    std = valid.std(axis=0) + 1e-8
    return (X_train - mean) / std, (X_test - mean) / std, mean, std


def _group_metrics(scores: np.ndarray, returns: np.ndarray, mask: np.ndarray) -> Dict[str, float]:
    top1 = 0
    top1_tie_aware = 0
    spearmans = []
    pairwise_correct = 0
    pairwise_total = 0
    regret_sum = 0.0
    n = 0
    for i in range(scores.shape[0]):
        m = mask[i]
        if m.sum() < 2:
            continue
        s = scores[i, m]
        r = returns[i, m]
        best_return = r.max()
        best_set = set(np.where(r == best_return)[0])
        pred_idx = int(np.argmax(s))
        if pred_idx in best_set:
            top1 += 1
            top1_tie_aware += 1
        elif len(best_set) > 1:
            top1_tie_aware += 1
        # Spearman
        if np.std(s) > 1e-8 and np.std(r) > 1e-8:
            rho, _ = stats.spearmanr(s, r)
            if np.isfinite(rho):
                spearmans.append(float(rho))
        # Pairwise accuracy
        for a in range(len(s)):
            for b in range(a + 1, len(s)):
                pairwise_total += 1
                if (s[a] > s[b]) == (r[a] > r[b]):
                    pairwise_correct += 1
        regret_sum += best_return - r[pred_idx]
        n += 1
    return {
        "groups": n,
        "top1_accuracy": float(top1 / n) if n else 0.0,
        "top1_tie_aware": float(top1_tie_aware / n) if n else 0.0,
        "spearman_mean": float(np.mean(spearmans)) if spearmans else 0.0,
        "spearman_std": float(np.std(spearmans)) if spearmans else 0.0,
        "pairwise_accuracy": float(pairwise_correct / pairwise_total) if pairwise_total else 0.0,
        "model_regret_mean": float(regret_sum / n) if n else 0.0,
    }


def _fit_ridge(X_train: np.ndarray, y_train: np.ndarray, mask_train: np.ndarray, cols: List[int], alpha: float = 1.0) -> np.ndarray:
    X = X_train[:, :, cols][mask_train]
    y = y_train[mask_train]
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    Xty = X.T @ y
    w = np.linalg.solve(XtX, Xty)
    return w


class _TinyMLP(nn.Module):
    def __init__(self, input_dim: int, hidden: Tuple[int, ...] = (64, 32)):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def _fit_mlp(X_train: np.ndarray, y_train: np.ndarray, mask_train: np.ndarray,
             X_test: np.ndarray, y_test: np.ndarray, mask_test: np.ndarray,
             cols: List[int], seed: int = 0) -> np.ndarray:
    torch.manual_seed(seed)
    np.random.seed(seed)
    Xtr = torch.tensor(X_train[:, :, cols][mask_train], dtype=torch.float32)
    ytr = torch.tensor(y_train[mask_train], dtype=torch.float32)
    Xte = torch.tensor(X_test[:, :, cols][mask_test], dtype=torch.float32)
    model = _TinyMLP(Xtr.shape[1], (64, 32))
    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    bs = 256
    n = Xtr.shape[0]
    for epoch in range(80):
        perm = torch.randperm(n)
        model.train()
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            pred = model(Xtr[idx])
            loss = nn.functional.mse_loss(pred, ytr[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    with torch.no_grad():
        # Reconstruct full 2D score arrays
        scores_test = np.full(X_test.shape[:2], -1e9, dtype=np.float32)
        scores_test[mask_test] = model(Xte).cpu().numpy()
    return scores_test


def _subset_indices(names: List[str], include: List[str]) -> List[int]:
    return [names.index(n) for n in include]


def _feature_subsets(names: List[str]) -> Dict[str, List[int]]:
    old = list(range(25))
    new = list(range(25, 41))
    delta_names = ["delta_frag", "free_block_count_delta", "occupied_slot_hops", "path_conflict_after"]
    delta = _subset_indices(names, delta_names)
    # low-correlation new features will be determined dynamically
    return {
        "M0_old25": old,
        "M1_new16": new,
        "M2_full41": list(range(41)),
        "M3_old25_plus_delta": sorted(set(old) | set(delta)),
    }


def _incremental_r2(X_train: np.ndarray, y_train: np.ndarray, mask_train: np.ndarray,
                    X_test: np.ndarray, y_test: np.ndarray, mask_test: np.ndarray,
                    names: List[str]) -> Dict[str, Any]:
    """For each new feature, fit old25 + feature and report R^2 and ranking improvement on test."""
    old = list(range(25))
    Xv = X_train[mask_train]
    yv = y_train[mask_train]
    # baseline old25 metrics
    w0 = _fit_ridge(X_train, y_train, mask_train, cols=old, alpha=1e-3)
    scores0 = np.full(X_test.shape[:2], -1e9, dtype=np.float32)
    scores0[mask_test] = (X_test[:, :, old][mask_test] @ w0).astype(np.float32)
    base_metrics = _group_metrics(scores0, y_test, mask_test)
    out = {"old25_test_top1": base_metrics["top1_accuracy"], "old25_test_spearman": base_metrics["spearman_mean"]}
    increments = []
    for j_new in range(25, 41):
        cols = old + [j_new]
        w = _fit_ridge(X_train, y_train, mask_train, cols=cols, alpha=1e-3)
        scores = np.full(X_test.shape[:2], -1e9, dtype=np.float32)
        scores[mask_test] = (X_test[:, :, cols][mask_test] @ w).astype(np.float32)
        pred_metrics = _group_metrics(scores, y_test, mask_test)
        pred_train = Xv[:, cols] @ w
        ss_res = ((yv - pred_train) ** 2).sum()
        ss_tot = ((yv - yv.mean()) ** 2).sum()
        r2 = 1 - ss_res / ss_tot
        increments.append({
            "feature": names[j_new],
            "train_r2": float(r2),
            "delta_test_top1": float(pred_metrics["top1_accuracy"] - base_metrics["top1_accuracy"]),
            "delta_test_spearman": float(pred_metrics["spearman_mean"] - base_metrics["spearman_mean"]),
        })
    out["new_feature_increments"] = increments
    return out


def _low_corr_new_features(corr: np.ndarray, names: List[str], threshold: float = 0.8) -> List[str]:
    low_corr = []
    for j_new in range(25, 41):
        max_corr = max(abs(corr[j_new, j_old]) for j_old in range(25))
        if max_corr < threshold:
            low_corr.append(names[j_new])
    return low_corr


def main():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    X_train, y_train, mask_train = _load_split("train")
    X_test, y_test, mask_test = _load_split("test")
    meta = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    names = meta["feature_names_poststate_v1"]

    # Normalize full matrix once (used for correlation and subset selection)
    X_train_n, X_test_n, mean, std = _normalize(X_train, X_test, mask_train, mask_test)

    # Correlation matrix and low-correlation new features
    corr = np.corrcoef(X_train_n[mask_train], rowvar=False)
    low_corr_names = _low_corr_new_features(corr, names, threshold=0.8)

    subsets = _feature_subsets(names)
    subsets["M4_old25_plus_nonredundant"] = sorted(set(range(25)) | set(_subset_indices(names, low_corr_names)))

    results = {}
    for mname, idx in subsets.items():
        # Ridge
        w = _fit_ridge(X_train_n, y_train, mask_train, cols=idx, alpha=1e-3)
        scores_ridge = np.full(X_test.shape[:2], -1e9, dtype=np.float32)
        scores_ridge[mask_test] = (X_test_n[:, :, idx][mask_test] @ w).astype(np.float32)
        ridge_metrics = _group_metrics(scores_ridge, y_test, mask_test)
        # MLP
        scores_mlp = _fit_mlp(X_train_n, y_train, mask_train, X_test_n, y_test, mask_test, cols=idx, seed=42)
        mlp_metrics = _group_metrics(scores_mlp, y_test, mask_test)
        results[mname] = {
            "features": [names[i] for i in idx],
            "n_features": len(idx),
            "ridge": ridge_metrics,
            "mlp": mlp_metrics,
        }

    incremental = _incremental_r2(X_train_n, y_train, mask_train, X_test_n, y_test, mask_test, names)

    out = {
        "low_correlation_new_features": low_corr_names,
        "subset_results": results,
        "incremental_r2": incremental,
        "condition_number": float(np.linalg.cond(X_train_n[mask_train])),
    }
    (OUTPUT_DIR / "FEATURE_REDUNDANCY_ANALYSIS.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Feature Redundancy Analysis",
        "",
        f"Condition number (standardized train): **{out['condition_number']:.2e}**",
        "",
        f"New features with max |r| to old25 < 0.8: {', '.join(low_corr_names) if low_corr_names else 'None'}",
        "",
        "## Model comparison (test set)",
        "",
        "| Model | N feats | Ridge top1 | Ridge tie-aware top1 | Ridge Spearman | Ridge pairwise | MLP top1 | MLP tie-aware top1 | MLP Spearman | MLP pairwise |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mname, r in results.items():
        lines.append(
            f"| {mname} | {r['n_features']} | "
            f"{r['ridge']['top1_accuracy']:.2%} | {r['ridge']['top1_tie_aware']:.2%} | {r['ridge']['spearman_mean']:.3f} | {r['ridge']['pairwise_accuracy']:.2%} | "
            f"{r['mlp']['top1_accuracy']:.2%} | {r['mlp']['top1_tie_aware']:.2%} | {r['mlp']['spearman_mean']:.3f} | {r['mlp']['pairwise_accuracy']:.2%} |"
        )
    lines += ["", "## Incremental value of each new feature when added to old25 (ridge on test)", "", f"Baseline old25: top1={incremental['old25_test_top1']:.2%}, Spearman={incremental['old25_test_spearman']:.3f}", "", "| Feature | Train R² | Δ test top1 | Δ test Spearman |", "|---|---:|---:|---:|"]
    for inc in incremental["new_feature_increments"]:
        lines.append(f"| {inc['feature']} | {inc['train_r2']:.4f} | {inc['delta_test_top1']:+.2%} | {inc['delta_test_spearman']:+.3f} |")
    lines.append("")

    (OUTPUT_DIR / "FEATURE_REDUNDANCY_ANALYSIS.md").write_text("\n".join(lines), encoding="utf-8")
    print("Part 4 complete. Report written to", OUTPUT_DIR / "FEATURE_REDUNDANCY_ANALYSIS.md")


if __name__ == "__main__":
    main()
