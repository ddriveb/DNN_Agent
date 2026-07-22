"""Full-dataset validation for the corrected paired-feature dataset.

Checks shard completeness, strict pairing, candidate validity, and label quality.
Outputs JSON and Markdown reports.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")
DATASET_DIR = Path("sa_hmarl/datasets/v135_corrected_paired_feature_ablation/full")


def _load_split(dataset_dir: Path, split: str) -> Tuple[Dict[str, np.ndarray], Dict[str, Any]]:
    d = np.load(str(dataset_dir / f"{split}.npz"), allow_pickle=True)
    meta = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
    return d, meta


def _check_pairing(d: Dict[str, np.ndarray], feature_keys: List[str]) -> List[str]:
    errors: List[str] = []
    n = d["group_id"].shape[0]
    for key in ["group_id", "action_ids", "mask", "returns_label_a"]:
        for fk in feature_keys:
            if d[fk].shape[0] != n:
                errors.append(f"{fk} has {d[fk].shape[0]} rows, expected {n}")
    # group_id, action_ids, mask, returns must match across feature matrices.
    for i in range(n):
        mask = d["mask"][i].astype(bool)
        legal_count = int(mask.sum())
        for fk in feature_keys:
            if d[fk][i].shape[0] < legal_count:
                errors.append(f"{fk} row {i} length {d[fk][i].shape[0]} < legal_count {legal_count}")
        aids = d["action_ids"][i][:legal_count]
        if aids.min() < 0:
            errors.append(f"row {i} negative action_id")
        # action_ids should be unique within group.
        if len(np.unique(aids)) != legal_count:
            errors.append(f"row {i} duplicate action_ids")
    return errors


def _split_quality(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
    returns = d["returns_label_a"]
    mask = d["mask"].astype(bool)
    n_groups = returns.shape[0]
    zero_range = 0
    tie = 0
    top1_top2_gap_sum = 0.0
    return_ranges = []
    valid_group_count = 0
    candidate_counts = []
    for i in range(n_groups):
        vals = returns[i][mask[i]]
        if vals.size == 0:
            continue
        valid_group_count += 1
        candidate_counts.append(int(vals.size))
        rng = float(vals.max() - vals.min())
        return_ranges.append(rng)
        if rng < 1e-8:
            zero_range += 1
        sorted_vals = np.sort(vals)
        if vals.size >= 2:
            gap = float(sorted_vals[-1] - sorted_vals[-2])
        else:
            gap = 0.0
        top1_top2_gap_sum += gap
        # tie if top value shared by more than one candidate.
        top_count = int((vals == sorted_vals[-1]).sum())
        if top_count > 1:
            tie += 1
    cc = np.array(candidate_counts)
    return {
        "groups": n_groups,
        "valid_groups": valid_group_count,
        "candidate_count_mean": float(cc.mean()) if cc.size else 0.0,
        "candidate_count_p50": float(np.median(cc)) if cc.size else 0.0,
        "candidate_count_p90": float(np.percentile(cc, 90)) if cc.size else 0.0,
        "candidate_count_min": int(cc.min()) if cc.size else 0,
        "candidate_count_max": int(cc.max()) if cc.size else 0,
        "zero_range_rate": zero_range / max(valid_group_count, 1),
        "tie_rate": tie / max(valid_group_count, 1),
        "return_range_mean": float(np.mean(return_ranges)) if return_ranges else 0.0,
        "top1_top2_gap_mean": top1_top2_gap_sum / max(valid_group_count, 1),
    }


def _load_distribution(d: Dict[str, np.ndarray]) -> Dict[str, Any]:
    buckets, counts = np.unique(d["load_bucket"], return_counts=True)
    return {
        "buckets": {int(b): int(c) for b, c in zip(buckets, counts)},
        "server_util_mean": float(d["server_utilization"].mean()),
        "server_util_std": float(d["server_utilization"].std()),
        "spectrum_util_mean": float(d["spectrum_utilization"].mean()),
        "spectrum_util_std": float(d["spectrum_utilization"].std()),
    }


def _check_shards(dataset_dir: Path, meta: Dict[str, Any]) -> Dict[str, Any]:
    shard_dir = dataset_dir / "shards"
    expected: List[str] = []
    for shard_id, info in meta.get("shards", {}).items():
        expected.append(shard_id)
    missing = []
    failed = []
    ok = []
    for sid in expected:
        npz = shard_dir / f"{sid}.npz"
        smeta = shard_dir / f"{sid}.metadata.json"
        if not npz.exists() or not smeta.exists():
            missing.append(sid)
            continue
        try:
            m = json.loads(smeta.read_text(encoding="utf-8"))
            if m.get("status") == "ok":
                ok.append(sid)
            else:
                failed.append((sid, m.get("error", "unknown")[:200]))
        except Exception as e:
            failed.append((sid, str(e)))
    return {
        "expected": expected,
        "ok": ok,
        "missing": missing,
        "failed": failed,
        "config_hash": meta.get("config_hash"),
    }


def main():
    global OUTPUT_DIR, DATASET_DIR
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset_dir", default=str(DATASET_DIR),
                        help="Path to the corrected paired dataset directory containing metadata.json and .npz splits.")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR),
                        help="Directory to write FULL_DATASET_VALIDATION.json/md.")
    args = parser.parse_args()
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    d_train, meta = _load_split(dataset_dir, "train")
    d_val, _ = _load_split(dataset_dir, "val")
    d_test, _ = _load_split(dataset_dir, "test")

    feature_keys = ["features_v1", "features_poststate_compact_v2", "features_poststate_full_corrected"]
    shard_report = _check_shards(dataset_dir, meta)
    pairing_errors = {
        "train": _check_pairing(d_train, feature_keys),
        "val": _check_pairing(d_val, feature_keys),
        "test": _check_pairing(d_test, feature_keys),
    }

    quality = {
        "train": _split_quality(d_train),
        "val": _split_quality(d_val),
        "test": _split_quality(d_test),
    }
    load_dist = {
        "train": _load_distribution(d_train),
        "val": _load_distribution(d_val),
        "test": _load_distribution(d_test),
    }

    result = {
        "dataset_dir": str(dataset_dir),
        "metadata": {
            "config_hash": meta.get("config_hash"),
            "return_coefs": meta.get("return_coefs"),
            "horizon": meta["config"].get("horizon"),
        },
        "shards": shard_report,
        "pairing_errors": pairing_errors,
        "quality": quality,
        "load_distribution": load_dist,
        "passed": (
            not shard_report["missing"]
            and not shard_report["failed"]
            and all(not v for v in pairing_errors.values())
        ),
    }

    (output_dir / "FULL_DATASET_VALIDATION.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# Full Dataset Validation",
        "",
        f"Dataset: `{dataset_dir}`",
        f"Config hash: `{result['metadata']['config_hash']}`",
        f"Overall passed: **{result['passed']}**",
        "",
        "## Shards",
        f"- Expected: {len(shard_report['expected'])}",
        f"- OK: {len(shard_report['ok'])}",
        f"- Missing: {len(shard_report['missing'])}",
        f"- Failed: {len(shard_report['failed'])}",
    ]
    if shard_report["missing"]:
        lines.append(f"- Missing shards: {shard_report['missing']}")
    if shard_report["failed"]:
        lines.append("- Failed shards:")
        for sid, err in shard_report["failed"]:
            lines.append(f"  - {sid}: {err}")
    lines.append("")

    lines.append("## Pairing errors")
    for split, errs in pairing_errors.items():
        lines.append(f"- {split}: {len(errs)} errors")
        for e in errs[:5]:
            lines.append(f"  - {e}")
    lines.append("")

    lines.append("## Quality")
    lines.append("| Split | Groups | cand mean | cand P50 | cand P90 | zero-range | tie rate | return range | top1-top2 gap |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for split, q in quality.items():
        lines.append(
            f"| {split} | {q['groups']} | {q['candidate_count_mean']:.1f} | "
            f"{q['candidate_count_p50']:.1f} | {q['candidate_count_p90']:.1f} | "
            f"{q['zero_range_rate']:.3f} | {q['tie_rate']:.3f} | "
            f"{q['return_range_mean']:.4f} | {q['top1_top2_gap_mean']:.4f} |"
        )
    lines.append("")

    lines.append("## Load distribution")
    lines.append("| Split | Server util mean/std | Spectrum util mean/std | Buckets |")
    lines.append("|---|---|---|---|")
    for split, ld in load_dist.items():
        lines.append(
            f"| {split} | {ld['server_util_mean']:.3f} ± {ld['server_util_std']:.3f} | "
            f"{ld['spectrum_util_mean']:.3f} ± {ld['spectrum_util_std']:.3f} | {ld['buckets']} |"
        )
    lines.append("")

    (output_dir / "FULL_DATASET_VALIDATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("Validation report:", output_dir / "FULL_DATASET_VALIDATION.md")
    print("Passed:", result["passed"])


if __name__ == "__main__":
    main()
