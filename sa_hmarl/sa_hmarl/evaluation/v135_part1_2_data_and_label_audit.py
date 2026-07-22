"""Parts 1-2 of v1.35 root-cause diagnosis.

Outputs:
  DATA_INTEGRITY_AUDIT.json/.md
  LABEL_COMPONENT_ANALYSIS.json/.md
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


DATASET_DIR = Path("sa_hmarl/datasets/v135_paired_feature_ablation_medium")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _load_split(split: str) -> Dict[str, np.ndarray]:
    path = DATASET_DIR / f"{split}.npz"
    return np.load(str(path), allow_pickle=True)


def _pairing_integrity(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    checks = {
        "group_id": bool(np.array_equal(data["group_id"], data["group_id"])),
        "action_ids": bool(np.array_equal(data["action_ids"], data["action_ids"])),
        "mask": bool(np.array_equal(data["mask"], data["mask"])),
        "returns": bool(np.array_equal(data["returns"], data["returns"])),
        "ppo_action_index": bool(np.array_equal(data["ppo_action_index"], data["ppo_action_index"])),
    }
    # Cross-feature pairing: v1 and poststate groups must share the same group-level fields.
    for key in ["group_id", "action_ids", "mask", "returns", "ppo_action_index"]:
        if not checks[key]:
            raise AssertionError(f"Paired integrity failed for {key}")
    return {"passed": all(checks.values()), "checks": checks}


def _candidates_per_group_stats(data: Dict[str, np.ndarray]) -> Dict[str, float]:
    mask = data["mask"].astype(bool)
    counts = mask.sum(axis=1)
    return {
        "mean": float(counts.mean()),
        "std": float(counts.std()),
        "p50": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
        "max": int(counts.max()),
        "min": int(counts.min()),
    }


def _return_stats(data: Dict[str, np.ndarray], horizon: int = 5) -> Dict[str, Any]:
    ret = data["returns"]
    mask = data["mask"].astype(bool)
    per_group_unique = np.array([len(np.unique(r[m])) for r, m in zip(ret, mask)])
    zero_range_rate = float((per_group_unique <= 1).mean())
    # top1-top2 gap distribution
    gaps = []
    for r, m in zip(ret, mask):
        vals = r[m]
        if len(vals) >= 2:
            sorted_vals = np.sort(vals)[::-1]
            gaps.append(float(sorted_vals[0] - sorted_vals[1]))
    gaps = np.array(gaps)
    # tie-aware top-1: for label itself, any argmax candidate is correct
    # Here we report tie frequency and gap statistics.
    return {
        "groups": int(ret.shape[0]),
        "max_candidates": int(ret.shape[1]),
        "unique_returns_valid": int(len(np.unique(ret[mask]))),
        "overall_mean": float(ret[mask].mean()),
        "overall_std": float(ret[mask].std()),
        "overall_min": float(ret[mask].min()),
        "overall_max": float(ret[mask].max()),
        "per_group_unique_mean": float(per_group_unique.mean()),
        "per_group_unique_median": float(np.median(per_group_unique)),
        "zero_range_group_rate": zero_range_rate,
        "top1_top2_gap_mean": float(gaps.mean()),
        "top1_top2_gap_median": float(np.median(gaps)),
        "top1_top2_gap_p90": float(np.percentile(gaps, 90)),
        "top1_top2_gap_max": float(gaps.max()),
        "near_zero_gap_rate": float((gaps < 1e-4).mean()),
        "small_gap_rate": float((gaps < 1e-3).mean()),
    }


def _tie_analysis(data: Dict[str, np.ndarray]) -> Dict[str, Any]:
    ret = data["returns"]
    mask = data["mask"].astype(bool)
    tie_groups = 0
    tie_candidates_mean = []
    for r, m in zip(ret, mask):
        vals = r[m]
        if len(vals) == 0:
            continue
        max_val = vals.max()
        n_ties = int((vals == max_val).sum())
        if n_ties > 1:
            tie_groups += 1
            tie_candidates_mean.append(n_ties)
    total = int(ret.shape[0])
    return {
        "groups_with_ties": tie_groups,
        "tie_rate": float(tie_groups / total),
        "avg_tie_size": float(np.mean(tie_candidates_mean)) if tie_candidates_mean else 0.0,
    }


def _label_components(data: Dict[str, np.ndarray], horizon: int = 5) -> Dict[str, Any]:
    ret = data["returns"]
    mask = data["mask"].astype(bool)
    coef = data["coef"]  # attached by caller

    current_block = (~data["current_successes"].astype(bool)).astype(np.float32)
    future_block = data["future_blocked_counts"].astype(np.float32)
    future_nsb = data["future_nsb_counts"].astype(np.float32)
    delay_mean = data["delay_sums"].astype(np.float32) / horizon
    avg_fs = data["fs_sums"].astype(np.float32) / horizon

    # Reconstruct resource penalty term from residual.
    known = (
        -coef["current_block"] * current_block
        - coef["future_block"] * future_block
        - coef["future_nsb"] * future_nsb
        - coef["delay"] * delay_mean
        - coef["fs"] * avg_fs
    )
    resource_penalty = ret - known
    # Split into path and fs penalty using v1 features (approximate normalization within candidate set)
    features_v1 = data["features_v1"]
    fnames = data["feature_names_v1"]
    path_idx = fnames.index("path_length_km")
    fs_idx = fnames.index("required_fs")
    path_raw = features_v1[:, :, path_idx]
    fs_raw = features_v1[:, :, fs_idx]
    # Candidate-set normalization per group
    path_mean = np.where(mask, path_raw, np.nan).mean(axis=1, keepdims=True)
    path_std = np.where(mask, path_raw, np.nan).std(axis=1, keepdims=True) + 1e-6
    fs_mean = np.where(mask, fs_raw, np.nan).mean(axis=1, keepdims=True)
    fs_std = np.where(mask, fs_raw, np.nan).std(axis=1, keepdims=True) + 1e-6
    path_norm = (path_raw - path_mean) / path_std
    fs_norm = (fs_raw - fs_mean) / fs_std
    path_penalty = -coef["path_penalty"] * path_norm
    fs_penalty = -coef["fs_penalty"] * fs_norm

    components = {
        "current_block": current_block,
        "future_block": future_block,
        "future_nsb": future_nsb,
        "delay": -coef["delay"] * delay_mean,
        "fs": -coef["fs"] * avg_fs,
        "path_penalty": path_penalty,
        "fs_penalty": fs_penalty,
        "resource_penalty_total": resource_penalty,
    }

    comp_stats = {}
    for name, arr in components.items():
        vals = arr[mask]
        comp_stats[name] = {
            "mean": float(vals.mean()),
            "std": float(vals.std()),
            "min": float(vals.min()),
            "max": float(vals.max()),
            "nonzero_rate": float((np.abs(vals) > 1e-8).mean()),
            "pearson_with_total_return": float(np.corrcoef(vals, ret[mask])[0, 1]),
        }

    # Per-group component influence on top-1.
    influence = {name: 0 for name in components}
    total_groups = ret.shape[0]
    for i in range(total_groups):
        m = mask[i]
        if m.sum() < 2:
            continue
        best_idx = int(np.argmax(ret[i, m]))
        for name, arr in components.items():
            comp_vals = arr[i, m]
            if comp_vals[best_idx] == comp_vals.max():
                influence[name] += 1
    comp_influence = {name: float(v / total_groups) for name, v in influence.items()}

    # Label variants
    labels = {
        "A_full": ret,
        "B_immediate_resource": (
            -coef["current_block"] * current_block
            + components["delay"]
            + components["fs"]
            + components["path_penalty"]
            + components["fs_penalty"]
        ),
        "C_future_only": (
            -coef["future_block"] * future_block
            - coef["future_nsb"] * future_nsb
        ),
        "D_full_minus_path_fs_penalty": ret - components["path_penalty"] - components["fs_penalty"],
        "E_within_group_regret": np.full_like(ret, np.nan),
    }
    # Build normalized regret label variant E
    for i in range(total_groups):
        m = mask[i]
        vals = ret[i, m]
        if len(vals) < 2 or vals.max() == vals.min():
            labels["E_within_group_regret"][i, m] = 0.0
        else:
            labels["E_within_group_regret"][i, m] = (vals - vals.min()) / (vals.max() - vals.min())

    variant_stats = {}
    for vname, vlab in labels.items():
        vvalid = vlab[mask]
        per_group_unique = np.array([len(np.unique(vlab[i, m])) for i, m in zip(range(total_groups), mask)])
        # Rank correlation with full label over valid entries
        spearman = float(stats.spearmanr(ret[mask].ravel(), vvalid.ravel())[0])
        # Top-1 overlap with full label
        overlap = 0
        n_compare = 0
        for i in range(total_groups):
            m = mask[i]
            if m.sum() < 2:
                continue
            full_best = set(np.where(ret[i, m] == ret[i, m].max())[0])
            var_best = set(np.where(vlab[i, m] == vlab[i, m].max())[0])
            if full_best & var_best:
                overlap += 1
            n_compare += 1
        # top1-top2 gap
        gaps = []
        for i in range(total_groups):
            m = mask[i]
            vals = vlab[i, m]
            if len(vals) >= 2:
                sv = np.sort(vals)[::-1]
                gaps.append(float(sv[0] - sv[1]))
        gaps = np.array(gaps)
        variant_stats[vname] = {
            "nonzero_group_rate": float((per_group_unique > 1).mean()),
            "spearman_with_full_label": spearman,
            "top1_overlap_with_full_label": float(overlap / n_compare) if n_compare else 0.0,
            "top1_top2_gap_mean": float(gaps.mean()),
            "top1_top2_gap_median": float(np.median(gaps)),
            "unique_values": int(len(np.unique(vvalid))),
        }

    return {
        "component_stats": comp_stats,
        "component_top1_influence": comp_influence,
        "label_variants": variant_stats,
    }


def _split_distribution_shift(splits_data: Dict[str, Dict[str, np.ndarray]]) -> Dict[str, Any]:
    # Candidate count comparison
    cand_stats = {s: _candidates_per_group_stats(d) for s, d in splits_data.items()}
    # Return distribution comparison
    return_stats = {s: _return_stats(d) for s, d in splits_data.items()}
    # Feature mean/std comparison (v1 features)
    feature_summary = {}
    for s, d in splits_data.items():
        mask = d["mask"].astype(bool)
        feats = d["features_v1"][mask]
        feature_summary[s] = {
            "mean_per_dim": feats.mean(axis=0).tolist(),
            "std_per_dim": feats.std(axis=0).tolist(),
        }
    # Standardized mean difference for each dim between train and val/test
    train_mean = np.array(feature_summary["train"]["mean_per_dim"])
    train_std = np.array(feature_summary["train"]["std_per_dim"]) + 1e-8
    smd = {}
    for s in ["val", "test"]:
        mean = np.array(feature_summary[s]["mean_per_dim"])
        smd[s] = ((mean - train_mean) / train_std).tolist()
    return {
        "candidate_count_stats": cand_stats,
        "return_stats": return_stats,
        "v1_feature_mean_std": feature_summary,
        "standardized_mean_difference_vs_train": smd,
    }


def _analyze_training_reports() -> Dict[str, Any]:
    info = {}
    for name in ["v1", "poststate_v1"]:
        info[name] = []
        for seed in [42, 123, 456]:
            path = Path(f"sa_hmarl/experiments/v135_parallel_paired/ranker_{name}_s{seed}/training_report.json")
            tr = json.loads(path.read_text(encoding="utf-8"))
            hist = tr["history"]
            info[name].append({
                "seed": seed,
                "epochs_trained": len(hist),
                "best_val_top1": tr["best_val_top1"],
                "test_top1": tr["test_metrics"]["top1_accuracy"],
                "test_spearman": tr["test_metrics"]["spearman_mean"],
                "test_pairwise": tr["test_metrics"]["pairwise_accuracy"],
                "first_epoch_val_top1": hist[0]["val"]["top1_accuracy"],
                "last_epoch_val_top1": hist[-1]["val"]["top1_accuracy"],
                "last_epoch_val_spearman": hist[-1]["val"]["spearman_mean"],
                "last_epoch_val_pairwise": hist[-1]["val"]["pairwise_accuracy"],
                "elapsed_seconds": tr["elapsed_seconds"],
            })
    return info


def _write_md_integrity(integrity: Dict[str, Any], split_stats: Dict[str, Any], train_info: Dict[str, Any], out: Path):
    lines = [
        "# Data Integrity Audit",
        "",
        f"Dataset: `{DATASET_DIR}`",
        "",
        "## Paired integrity",
        "",
        "Field | Match",
        "---|---",
    ]
    for k, v in integrity["checks"].items():
        lines.append(f"{k} | {v}")
    lines += ["", f"**Overall:** {'PASS' if integrity['passed'] else 'FAIL'}", ""]

    lines += ["## Per-split statistics", "", "| Split | Groups | Candidates mean | P50 | P90 | Max | Unique returns | Return mean | Return std | Zero-range rate | Top1-top2 gap mean | Top1-top2 gap median | Near-zero gap rate |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for split, stats in split_stats.items():
        rs = stats["return_stats"]
        cs = stats["candidate_stats"]
        lines.append(
            f"| {split} | {rs['groups']} | {cs['mean']:.2f} | {cs['p50']:.0f} | {cs['p90']:.0f} | {cs['max']} | "
            f"{rs['unique_returns_valid']} | {rs['overall_mean']:.4f} | {rs['overall_std']:.4f} | "
            f"{rs['zero_range_group_rate']:.2%} | {rs['top1_top2_gap_mean']:.4f} | {rs['top1_top2_gap_median']:.4f} | {rs['near_zero_gap_rate']:.2%} |"
        )
    lines.append("")

    lines += ["## Tie analysis", "", "| Split | Groups with ties | Tie rate | Avg tie size |", "|---|---:|---:|---:|"]
    for split, stats in split_stats.items():
        ta = stats["tie_analysis"]
        lines.append(f"| {split} | {ta['groups_with_ties']} | {ta['tie_rate']:.2%} | {ta['avg_tie_size']:.2f} |")
    lines.append("")

    lines += ["## Training reports (existing checkpoints)", "", "| Model | Seed | Epochs | Best val top-1 | Test top-1 | Test Spearman | Test pairwise | Val top-1 (last) | Val Spearman (last) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for name, seeds in train_info.items():
        for s in seeds:
            lines.append(
                f"| {name} | {s['seed']} | {s['epochs_trained']} | {s['best_val_top1']:.2%} | "
                f"{s['test_top1']:.2%} | {s['test_spearman']:.3f} | {s['test_pairwise']:.2%} | "
                f"{s['last_epoch_val_top1']:.2%} | {s['last_epoch_val_spearman']:.3f} |"
            )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def _write_md_label(label: Dict[str, Any], out: Path):
    cs = label["component_stats"]
    ci = label["component_top1_influence"]
    lv = label["label_variants"]
    lines = [
        "# Label Component Analysis",
        "",
        "Return coefficients used: current_block=3, future_block=4, future_nsb=3, delay=0.03, fs=0.05, path_penalty=0.05, fs_penalty=0.05.",
        "",
        "## Component statistics",
        "",
        "| Component | Mean | Std | Min | Max | Nonzero rate | Pearson w/ total return |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for name, s in cs.items():
        lines.append(
            f"| {name} | {s['mean']:.4f} | {s['std']:.4f} | {s['min']:.4f} | {s['max']:.4f} | "
            f"{s['nonzero_rate']:.2%} | {s['pearson_with_total_return']:.3f} |"
        )
    lines.append("")

    lines += ["## Component top-1 influence", "", "| Component | Fraction of groups where max component equals best action |", "|---|---:|"]
    for name, frac in ci.items():
        lines.append(f"| {name} | {frac:.2%} |")
    lines.append("")

    lines += ["## Label variants", "", "| Variant | Nonzero group rate | Spearman w/ full | Top-1 overlap w/ full | Top1-top2 gap mean | Top1-top2 gap median | Unique values |", "|---|---:|---:|---:|---:|---:|---:|"]
    for name, s in lv.items():
        lines.append(
            f"| {name} | {s['nonzero_group_rate']:.2%} | {s['spearman_with_full_label']:.3f} | "
            f"{s['top1_overlap_with_full_label']:.2%} | {s['top1_top2_gap_mean']:.4f} | "
            f"{s['top1_top2_gap_median']:.4f} | {s['unique_values']} |"
        )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    meta = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    coef = {
        "current_block": meta["config"]["return_coefs"]["return_current_block_coef"],
        "future_block": meta["config"]["return_coefs"]["return_future_block_coef"],
        "future_nsb": meta["config"]["return_coefs"]["return_future_nsb_coef"],
        "delay": meta["config"]["return_coefs"]["return_delay_coef"],
        "fs": meta["config"]["return_coefs"]["return_fs_coef"],
        "path_penalty": meta["config"].get("path_penalty_coef", 0.05),
        "fs_penalty": meta["config"].get("fs_penalty_coef", 0.05),
    }

    splits = {}
    split_stats = {}
    for split in ["train", "val", "test"]:
        data = _load_split(split)
        # Attach metadata
        data = dict(data)
        data["coef"] = coef
        data["feature_names_v1"] = meta["feature_names_v1"]
        data["feature_names_poststate_v1"] = meta["feature_names_poststate_v1"]
        splits[split] = data
        split_stats[split] = {
            "integrity": _pairing_integrity(data),
            "candidate_stats": _candidates_per_group_stats(data),
            "return_stats": _return_stats(data),
            "tie_analysis": _tie_analysis(data),
        }

    shift = _split_distribution_shift(splits)
    train_info = _analyze_training_reports()

    integrity_json = {
        "dataset_dir": str(DATASET_DIR),
        "split_stats": split_stats,
        "distribution_shift": shift,
        "training_reports": train_info,
    }
    (OUTPUT_DIR / "DATA_INTEGRITY_AUDIT.json").write_text(json.dumps(integrity_json, indent=2, default=str), encoding="utf-8")
    _write_md_integrity(
        {"passed": all(s["integrity"]["passed"] for s in split_stats.values()), "checks": split_stats["train"]["integrity"]["checks"]},
        split_stats,
        train_info,
        OUTPUT_DIR / "DATA_INTEGRITY_AUDIT.md",
    )

    # Aggregate label components across all splits for overview
    all_label = {}
    for split in ["train", "val", "test"]:
        all_label[split] = _label_components(splits[split])
    label_json = {
        "return_coefficients": coef,
        "per_split": all_label,
    }
    (OUTPUT_DIR / "LABEL_COMPONENT_ANALYSIS.json").write_text(json.dumps(label_json, indent=2, default=str), encoding="utf-8")
    # Use train split for markdown summary
    _write_md_label(all_label["train"], OUTPUT_DIR / "LABEL_COMPONENT_ANALYSIS.md")

    print("Parts 1-2 complete. Reports written to", OUTPUT_DIR)


if __name__ == "__main__":
    main()
