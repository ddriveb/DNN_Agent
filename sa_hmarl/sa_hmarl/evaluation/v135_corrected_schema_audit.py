"""Compact schema audit for v1.35 corrected dataset.

Reads the corrected paired dataset and reports:
  - per-feature statistics
  - correlations and condition number
  - duplicate / near-duplicate feature pairs
  - max correlation of each compact-v2 feature with the old 25-d base
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")
DATASET_DIR = Path("sa_hmarl/datasets/v135_corrected_paired_feature_ablation")


def _load_split(split: str):
    d = np.load(str(DATASET_DIR / f"{split}.npz"), allow_pickle=True)
    meta = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    return d, meta


def _split_stats(arr: np.ndarray, mask: np.ndarray, names: List[str]) -> List[Dict[str, Any]]:
    out = []
    for j, name in enumerate(names):
        col = arr[mask, j]
        if col.size == 0:
            out.append({"name": name, "constant": True})
            continue
        std = float(col.std())
        group_vars = []
        for i in range(arr.shape[0]):
            m = mask[i]
            vals = arr[i, m, j]
            if vals.size > 1:
                group_vars.append(float(vals.var()))
        out.append({
            "name": name,
            "mean": float(col.mean()),
            "std": std,
            "min": float(col.min()),
            "max": float(col.max()),
            "constant": std < 1e-8,
            "group_variance_mean": float(np.mean(group_vars)) if group_vars else 0.0,
            "unique_values": int(len(np.unique(col))),
        })
    return out


def _condition_number(X: np.ndarray) -> float:
    Xs = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
    return float(np.linalg.cond(Xs))


def _duplicate_pairs(corr: np.ndarray, names: List[str], threshold: float = 0.99) -> List[Dict[str, Any]]:
    pairs = []
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            if abs(corr[i, j]) > threshold:
                pairs.append({"f1": names[i], "f2": names[j], "corr": float(corr[i, j])})
    return pairs


def _new_vs_old_max_corr(corr: np.ndarray, names: List[str], n_old: int) -> Dict[str, Dict[str, Any]]:
    out = {}
    for j_new in range(n_old, len(names)):
        max_corr = 0.0
        max_old = ""
        for j_old in range(n_old):
            c = abs(corr[j_new, j_old])
            if c > max_corr:
                max_corr = c
                max_old = names[j_old]
        out[names[j_new]] = {"max_old_feature": max_old, "abs_corr": float(max_corr)}
    return out


def _label_correlations(X: np.ndarray, y: np.ndarray, names: List[str]) -> Dict[str, float]:
    out = {}
    y_std = float(y.std())
    for j, name in enumerate(names):
        col = X[:, j]
        if col.std() < 1e-12 or y_std < 1e-12:
            out[name] = 0.0
        else:
            out[name] = float(np.corrcoef(col, y)[0, 1])
    return out


def _recommend_pruned_compact(
    names_compact: List[str],
    stats_compact: List[Dict[str, Any]],
    corr_compact: np.ndarray,
    label_corr: Dict[str, float],
    names_v1: List[str],
) -> Tuple[List[str], Dict[str, str]]:
    """Return (pruned_names, reason_for_drop)."""
    drops: Dict[str, str] = {}
    n = len(names_compact)
    # Constant or near-zero group variance.
    for s in stats_compact:
        if s.get("constant") or s.get("group_variance_mean", 1.0) < 1e-12:
            drops[s["name"]] = "constant or zero group variance"
    # Exact / near duplicates.
    for i in range(n):
        for j in range(i + 1, n):
            c = abs(corr_compact[i, j])
            if c > 0.99:
                fi, fj = names_compact[i], names_compact[j]
                # Prefer keeping the feature that is already in v1 base (pre-action, online-stable).
                fi_in_v1 = fi in names_v1
                fj_in_v1 = fj in names_v1
                if fi_in_v1 and not fj_in_v1:
                    drop, keep = fj, fi
                elif fj_in_v1 and not fi_in_v1:
                    drop, keep = fi, fj
                else:
                    # Keep the one with stronger absolute label correlation.
                    ci = abs(label_corr.get(fi, 0.0))
                    cj = abs(label_corr.get(fj, 0.0))
                    drop, keep = (fj, fi) if ci >= cj else (fi, fj)
                if drop not in drops:
                    drops[drop] = f"near-duplicate of {keep} (|r|={c:.4f})"
    pruned = [n for n in names_compact if n not in drops]
    return pruned, drops


def _audit_feature_set(name: str, arr: np.ndarray, mask: np.ndarray, names: List[str]) -> Dict[str, Any]:
    stats = _split_stats(arr, mask, names)
    X = arr[mask]
    corr = np.corrcoef(X, rowvar=False)
    cond = _condition_number(X)
    dupes = _duplicate_pairs(corr, names)
    return {
        "name": name,
        "dim": len(names),
        "condition_number": cond,
        "duplicate_pairs": dupes,
        "constant_features": [s["name"] for s in stats if s["constant"]],
        "stats": stats,
    }


def main():
    global DATASET_DIR, OUTPUT_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=str(DATASET_DIR),
                        help="Path to the corrected paired dataset directory containing metadata.json and .npz splits.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR),
                        help="Directory to write COMPACT_SCHEMA_AUDIT.json/md.")
    args = parser.parse_args()

    DATASET_DIR = Path(args.dataset_dir)
    OUTPUT_DIR = Path(args.output_dir)

    if not (DATASET_DIR / "metadata.json").exists():
        raise FileNotFoundError(f"Dataset metadata not found at {DATASET_DIR / 'metadata.json'}")

    d, meta = _load_split("train")
    mask = d["mask"].astype(bool)

    names_v1 = meta["feature_names_v1"]
    names_compact = meta["feature_names_poststate_compact_v2"]
    names_full = meta.get("feature_names_poststate_full_corrected", [])

    audit_v1 = _audit_feature_set("v1_25", d["features_v1"], mask, names_v1)
    audit_compact = _audit_feature_set("poststate_compact_v2", d["features_poststate_compact_v2"], mask, names_compact)
    audit_full = _audit_feature_set("poststate_full_corrected", d["features_poststate_full_corrected"], mask, names_full) if names_full else None

    # Label correlations and pruning recommendation for compact v2.
    X_compact = d["features_poststate_compact_v2"][mask]
    y = d["returns_label_a"][mask]
    label_corr_compact = _label_correlations(X_compact, y, names_compact)
    corr_compact = np.corrcoef(X_compact, rowvar=False)
    pruned_names, drop_reasons = _recommend_pruned_compact(
        names_compact, audit_compact["stats"], corr_compact, label_corr_compact, names_v1
    )

    # New compact features vs old v1 base.
    n_old = len(names_v1)
    X_v1 = d["features_v1"][mask]
    joint = np.concatenate([X_v1, X_compact], axis=1)
    corr_joint = np.corrcoef(joint, rowvar=False)
    compact_vs_old = _new_vs_old_max_corr(corr_joint, names_v1 + names_compact, n_old)

    result = {
        "dataset_dir": str(DATASET_DIR),
        "train_groups": int(d["group_id"].shape[0]),
        "v1_25": audit_v1,
        "poststate_compact_v2": audit_compact,
        "poststate_compact_v2_label_corr": label_corr_compact,
        "compact_v2_pruned": {
            "feature_names": pruned_names,
            "dim": len(pruned_names),
            "dropped": drop_reasons,
        },
        "compact_v2_pruned_afterstate": {
            "feature_names": [n for n in pruned_names if n not in names_v1],
            "dim": len([n for n in pruned_names if n not in names_v1]),
        },
        "compact_vs_old_max_corr": compact_vs_old,
    }
    if audit_full:
        result["poststate_full_corrected"] = audit_full

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "COMPACT_SCHEMA_AUDIT.json").write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")

    lines = ["# Compact Schema Audit", ""]
    lines.append(f"Dataset: `{DATASET_DIR}`")
    lines.append(f"Train groups: {result['train_groups']}")
    lines.append("")

    for key in ["v1_25", "poststate_compact_v2"]:
        a = result[key]
        lines.append(f"## {a['name']} (dim={a['dim']})")
        lines.append(f"- Condition number: **{a['condition_number']:.2e}**")
        lines.append(f"- Constant features: {a['constant_features']}")
        if a["duplicate_pairs"]:
            lines.append("")
            lines.append("| Feature 1 | Feature 2 | Correlation |")
            lines.append("|---|---|---:|")
            for p in a["duplicate_pairs"]:
                lines.append(f"| {p['f1']} | {p['f2']} | {p['corr']:.4f} |")
        else:
            lines.append("- No duplicate pairs (|r| > 0.99).")
        lines.append("")
        lines.append("| Feature | Mean | Std | Min | Max | Constant | Group var |")
        lines.append("|---|---:|---:|---:|---:|:---:|:---:|")
        for s in a["stats"]:
            if s.get("constant"):
                lines.append(f"| {s['name']} | - | - | - | - | yes | - |")
            else:
                lines.append(
                    f"| {s['name']} | {s['mean']:.4f} | {s['std']:.4f} | "
                    f"{s['min']:.4f} | {s['max']:.4f} | no | {s['group_variance_mean']:.6f} |"
                )
        lines.append("")

    lines.append("## Compact v2 label correlations")
    lines.append("")
    lines.append("| Feature | Label corr |")
    lines.append("|---|---:|")
    for name in names_compact:
        lines.append(f"| {name} | {label_corr_compact.get(name, 0):.4f} |")
    lines.append("")

    lines.append("## Compact v2 pruning recommendation")
    lines.append("")
    lines.append(f"- Pruned dim: {len(pruned_names)} (from {len(names_compact)})")
    lines.append(f"- Dropped: {list(drop_reasons.keys())}")
    lines.append("")
    lines.append("| Dropped feature | Reason |")
    lines.append("|---|---|")
    for feat, reason in drop_reasons.items():
        lines.append(f"| {feat} | {reason} |")
    lines.append("")
    lines.append(f"```text\npruned_compact_v2 = {pruned_names}\n```")
    lines.append("")

    lines.append("## Compact v2 features vs old v1 base")
    lines.append("")
    lines.append("| New feature | Most correlated old feature | |r| |")
    lines.append("|---|---|---:|")
    for feat, info in compact_vs_old.items():
        lines.append(f"| {feat} | {info['max_old_feature']} | {info['abs_corr']:.3f} |")
    lines.append("")

    (OUTPUT_DIR / "COMPACT_SCHEMA_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Schema audit written to", OUTPUT_DIR / "COMPACT_SCHEMA_AUDIT.md")


if __name__ == "__main__":
    main()
