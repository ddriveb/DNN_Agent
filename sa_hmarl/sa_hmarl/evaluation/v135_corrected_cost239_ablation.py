"""Phase 1 COST239 corrected paired ablation.

Trains five ranker variants with identical states, labels, splits, and hyperparameters:
  M0: v1_25
  M1: v1_plus_compact_afterstate (full compact v2)
  M2: compact_afterstate_only
  M3: v1_plus_delta_only
  M4: v1_plus_compact_regularized

Checkpoint selection uses model regret by default.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


DATASET_DIR = Path("sa_hmarl/datasets/v135_corrected_paired_feature_ablation")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")


def _prepare_subset(
    paired_dir: Path,
    output_dir: Path,
    feature_source: str,
    feature_indices: List[int],
    feature_names: List[str],
    label_key: str = "returns_label_a",
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = np.load(str(paired_dir / f"{split}.npz"), allow_pickle=True)
        features = src[feature_source][:, :, feature_indices]
        np.savez_compressed(
            output_dir / f"{split}.npz",
            features=features,
            returns=src[label_key],
            mask=src["mask"],
            ppo_action_index=src["ppo_action_index"],
            future_nsb_counts=src.get("future_nsb_counts", np.zeros_like(src["mask"], dtype=np.int64)),
        )
    train = np.load(str(output_dir / "train.npz"), allow_pickle=True)
    features = train["features"].astype(np.float32)
    mask = train["mask"].astype(bool)
    valid = features[mask]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)

    # Read metadata to get return coefs / horizon.
    meta = json.loads((paired_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    return_coefs = meta.get("return_coefs", cfg.get("return_coefs", {}))

    metadata = {
        "feature_names": list(feature_names),
        "feature_dim": int(features.shape[2]),
        "train_feature_mean": mean.tolist(),
        "train_feature_std": std.tolist(),
        "dataset_dir": str(paired_dir),
        "feature_key": feature_source,
        "candidate_mode": cfg.get("candidate_mode", "ppo_r_topk_only"),
        "max_candidates": int(features.shape[1]),
        "horizon": cfg.get("horizon", 5),
        "return_coefs": return_coefs,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def _train_one(dataset_dir: Path, output_dir: Path, seed: int, extra_args: List[str]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "-m", "sa_hmarl.training.train_r_counterfactual_ranking",
        "--dataset_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--seed", str(seed),
        "--selection_metric", "regret",
        *extra_args,
    ]
    env = {**dict(os.environ), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    proc = subprocess.run(cmd, check=False, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    if proc.returncode != 0:
        print(f"[ablation] training failed: {' '.join(cmd)}")
        print(proc.stdout.decode("utf-8", errors="replace"))
        proc.check_returncode()
    report_path = output_dir / "training_report.json"
    return json.loads(report_path.read_text(encoding="utf-8"))


def _aggregate(metrics_list: List[Dict[str, Any]]) -> Dict[str, Any]:
    out = {}
    keys = list(metrics_list[0].keys())
    for k in keys:
        vals = [m[k] for m in metrics_list]
        if isinstance(vals[0], (int, float)):
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        elif isinstance(vals[0], dict):
            out[k] = _aggregate(vals)
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--paired_dataset_dir", default=str(DATASET_DIR))
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--max_workers", type=int, default=3)
    parser.add_argument("--label_key", default="returns_label_a")
    args = parser.parse_args()

    paired_dir = Path(args.paired_dataset_dir)
    meta = json.loads((paired_dir / "metadata.json").read_text(encoding="utf-8"))
    names_v1 = meta["feature_names_v1"]
    names_compact = meta["feature_names_poststate_compact_v2"]

    old25_indices = list(range(len(names_v1)))
    compact_indices = list(range(len(names_compact)))

    afterstate_names = [n for n in names_compact if n not in names_v1]
    afterstate_indices = [names_compact.index(n) for n in afterstate_names]

    delta_names = ["delta_frag", "free_block_count_delta"]
    delta_indices = [names_compact.index(n) for n in delta_names]

    # Pruned compact v2: remove exact/near duplicates identified by the schema audit.
    # Pruning follows the schema audit recommendation: remove exact/near duplicates
    # and group-constant context copies from the compact v2 afterstate dimensions.
    pruned_drop = {
        "global_frag_delta",           # exact duplicate of delta_frag
        "path_free_ratio_after",       # near-duplicate of free_ratio (|r|>0.99)
        "global_free_ratio_mean_after", # group-constant after global aggregation
    }
    pruned_afterstate_names = [n for n in names_compact if n not in names_v1 and n not in pruned_drop]
    pruned_afterstate_indices = [names_compact.index(n) for n in pruned_afterstate_names]
    pruned_compact_names = names_v1 + pruned_afterstate_names
    pruned_compact_indices = list(range(len(names_v1))) + [len(names_v1) + i for i in range(len(pruned_afterstate_names))]
    # Note: the pruned compact matrix is just the full compact matrix with dropped columns;
    # the indices above are for slicing that matrix.

    # M0: v1_25 from features_v1.
    # M1: full compact v2.
    # M2: compact afterstate only.
    # M3: v1_25 + delta_frag + free_block_count_delta (concatenate v1 + compact dims).
    # M4: full compact v2 regularized.
    # M5: v1 + pruned compact v2.
    # M6: pruned afterstate only.
    variants: List[Tuple[str, str, List[int], List[str], List[str]]] = [
        ("M0_v1_25", "features_v1", old25_indices, [names_v1[i] for i in old25_indices], []),
        ("M1_v1_plus_compact", "features_poststate_compact_v2", compact_indices, names_compact, []),
        ("M2_compact_afterstate_only", "features_poststate_compact_v2", afterstate_indices, afterstate_names, []),
        ("M3_v1_plus_delta_only", "features_v1_and_compact", old25_indices + delta_indices,
         [names_v1[i] for i in old25_indices] + [names_compact[i] for i in delta_indices], []),
        ("M4_v1_plus_compact_reg", "features_poststate_compact_v2", compact_indices, names_compact,
         ["--dropout", "0.2", "--weight_decay", "1e-4"]),
        ("M5_v1_plus_pruned", "features_v1_and_compact", list(range(25)) + pruned_afterstate_indices, names_v1 + pruned_afterstate_names, []),
        ("M6_pruned_afterstate_only", "features_poststate_compact_v2", pruned_afterstate_indices, pruned_afterstate_names, []),
    ]

    tmp_root = Path(tempfile.mkdtemp(prefix="v135_corrected_ablation_"))
    all_results = {}
    try:
        tasks = []
        for short_name, source, idx, feat_names, extra in variants:
            subset_dir = tmp_root / short_name
            if source == "features_v1_and_compact":
                # Need to concatenate v1 base with selected compact dims.
                _prepare_concat_subset(paired_dir, subset_dir, idx, feat_names, args.label_key)
            else:
                _prepare_subset(paired_dir, subset_dir, source, idx, feat_names, args.label_key)
            for seed in args.seeds:
                ckpt_dir = Path(args.output_dir) / f"ranker_{short_name}_s{seed}"
                tasks.append((short_name, seed, subset_dir, ckpt_dir, extra))

        results_by_variant: Dict[str, List[Dict[str, Any]]] = {v[0]: [] for v in variants}

        def _run(t):
            short_name, seed, subset_dir, ckpt_dir, extra = t
            report = _train_one(subset_dir, ckpt_dir, seed, extra)
            return short_name, seed, report

        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(_run, t): t for t in tasks}
            for fut in as_completed(futures):
                short_name, seed, report = fut.result()
                results_by_variant[short_name].append(report)
                tm = report["test_metrics"]
                print(
                    f"[ablation] {short_name} seed={seed} "
                    f"test_top1={tm['top1_accuracy']:.2%} "
                    f"test_tie1={tm.get('tie_aware_top1', 0):.2%} "
                    f"ndcg3={tm.get('ndcg_at_3', 0):.3f} "
                    f"regret={tm['model_regret_mean']:.4f} "
                    f"spearman={tm['spearman_mean']:.3f}",
                    flush=True,
                )

        for short_name, reports in results_by_variant.items():
            all_results[short_name] = {
                "seed_reports": [
                    {"seed": args.seeds[i], **r["test_metrics"]} for i, r in enumerate(reports)
                ],
                "aggregate": _aggregate([r["test_metrics"] for r in reports]),
                "epochs": [len(r["history"]) for r in reports],
            }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "COST239_PAIRED_ABLATION.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# COST239 Corrected Paired Ablation",
        "",
        f"Dataset: `{args.paired_dataset_dir}`",
        f"Label: `{args.label_key}`",
        f"Seeds: {args.seeds}",
        "",
        "| Model | N feats | Test top-1 | Tie-aware top-1 | NDCG@3 | Spearman | Pairwise acc | Model regret |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    key_order = ["M0_v1_25", "M1_v1_plus_compact", "M2_compact_afterstate_only", "M3_v1_plus_delta_only", "M4_v1_plus_compact_reg", "M5_v1_plus_pruned", "M6_pruned_afterstate_only"]
    n_feats = {
        "M0_v1_25": 25,
        "M1_v1_plus_compact": len(names_compact),
        "M2_compact_afterstate_only": len(afterstate_indices),
        "M3_v1_plus_delta_only": 25 + len(delta_indices),
        "M4_v1_plus_compact_reg": len(names_compact),
        "M5_v1_plus_pruned": 25 + len(pruned_afterstate_indices),
        "M6_pruned_afterstate_only": len(pruned_afterstate_indices),
    }
    for k in key_order:
        agg = all_results[k]["aggregate"]
        lines.append(
            f"| {k} | {n_feats[k]} | "
            f"{agg['top1_accuracy']['mean']:.2%} ± {agg['top1_accuracy']['std']:.2%} | "
            f"{agg.get('tie_aware_top1', {}).get('mean', 0):.2%} ± {agg.get('tie_aware_top1', {}).get('std', 0):.2%} | "
            f"{agg.get('ndcg_at_3', {}).get('mean', 0):.3f} ± {agg.get('ndcg_at_3', {}).get('std', 0):.3f} | "
            f"{agg['spearman_mean']['mean']:.3f} ± {agg['spearman_mean']['std']:.3f} | "
            f"{agg['pairwise_accuracy']['mean']:.2%} ± {agg['pairwise_accuracy']['std']:.2%} | "
            f"{agg['model_regret_mean']['mean']:.4f} ± {agg['model_regret_mean']['std']:.4f} |"
        )
    (out_dir / "COST239_PAIRED_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("COST239 ablation complete. Report:", out_dir / "COST239_PAIRED_ABLATION.md")


def _prepare_concat_subset(
    paired_dir: Path,
    output_dir: Path,
    feature_indices: List[int],
    feature_names: List[str],
    label_key: str,
):
    """Prepare a subset that concatenates v1 base with selected compact dims.

    feature_indices is assumed to be [0..24, compact_idx1, compact_idx2, ...].
    The first 25 indices are taken from features_v1; the rest from features_poststate_compact_v2.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    n_v1 = 25
    for split in ("train", "val", "test"):
        src = np.load(str(paired_dir / f"{split}.npz"), allow_pickle=True)
        v1_part = src["features_v1"][:, :, :n_v1]
        compact_part = src["features_poststate_compact_v2"][:, :, feature_indices[n_v1:]]
        features = np.concatenate([v1_part, compact_part], axis=2)
        np.savez_compressed(
            output_dir / f"{split}.npz",
            features=features,
            returns=src[label_key],
            mask=src["mask"],
            ppo_action_index=src["ppo_action_index"],
            future_nsb_counts=src.get("future_nsb_counts", np.zeros_like(src["mask"], dtype=np.int64)),
        )
    train = np.load(str(output_dir / "train.npz"), allow_pickle=True)
    features = train["features"].astype(np.float32)
    mask = train["mask"].astype(bool)
    valid = features[mask]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)

    meta = json.loads((paired_dir / "metadata.json").read_text(encoding="utf-8"))
    cfg = meta["config"]
    return_coefs = meta.get("return_coefs", cfg.get("return_coefs", {}))
    metadata = {
        "feature_names": list(feature_names),
        "feature_dim": int(features.shape[2]),
        "train_feature_mean": mean.tolist(),
        "train_feature_std": std.tolist(),
        "dataset_dir": str(paired_dir),
        "feature_key": "features_v1_and_compact",
        "candidate_mode": cfg.get("candidate_mode", "ppo_r_topk_only"),
        "max_candidates": int(features.shape[1]),
        "horizon": cfg.get("horizon", 5),
        "return_coefs": return_coefs,
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
