"""Part 5: controlled neural-network feature ablations.

Trains several feature-subset models with identical groups/actions/labels/splits
and the same training hyperparameters (3 seeds each).
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


DATASET_DIR = Path("sa_hmarl/datasets/v135_paired_feature_ablation_medium")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _prepare_subset(paired_dir: Path, output_dir: Path, feature_indices: List[int], feature_names: List[str]) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = np.load(str(paired_dir / f"{split}.npz"), allow_pickle=True)
        features = src["features_poststate_v1"][:, :, feature_indices]
        np.savez_compressed(
            output_dir / f"{split}.npz",
            features=features,
            returns=src["returns"],
            mask=src["mask"],
            ppo_action_index=src["ppo_action_index"],
        )
    train = np.load(str(output_dir / "train.npz"), allow_pickle=True)
    features = train["features"].astype(np.float32)
    mask = train["mask"].astype(bool)
    valid = features[mask]
    mean = valid.mean(axis=0).astype(np.float32)
    std = valid.std(axis=0).astype(np.float32)
    metadata = {
        "feature_names": list(feature_names),
        "feature_dim": int(features.shape[2]),
        "train_feature_mean": mean.tolist(),
        "train_feature_std": std.tolist(),
        "dataset_dir": str(paired_dir),
        "feature_key": "features_poststate_v1_subset",
        "candidate_mode": "v1",
        "max_candidates": int(features.shape[1]),
        "horizon": 5,
        "return_coefs": {
            "current_block": 3.0,
            "future_block": 4.0,
            "future_nsb": 3.0,
            "future_server_overload": 0.0,
            "delay": 0.03,
            "fs": 0.05,
        },
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return output_dir


def _train_one(dataset_dir: Path, output_dir: Path, seed: int, extra_args: List[str]) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py",
        "--dataset_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--seed", str(seed),
        *extra_args,
    ]
    env = {**dict(os.environ), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"}
    subprocess.run(cmd, check=True, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
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
    args = parser.parse_args()

    paired_dir = Path(args.paired_dataset_dir)
    meta = json.loads((paired_dir / "metadata.json").read_text(encoding="utf-8"))
    names = meta["feature_names_poststate_v1"]

    # Feature subsets
    old25 = list(range(25))
    new16 = list(range(25, 41))
    delta_names = ["delta_frag", "free_block_count_delta", "occupied_slot_hops", "path_conflict_after"]
    delta_indices = [names.index(n) for n in delta_names]
    nonredundant_names = ["frag_after", "delta_frag", "free_block_count_delta"]
    nonredundant_indices = [names.index(n) for n in nonredundant_names]

    variants: List[Tuple[str, List[int], List[str], List[str]]] = [
        ("A_v1_25", old25, [names[i] for i in old25], []),
        ("B_afterstate_only_16", new16, [names[i] for i in new16], []),
        ("C_full_41", list(range(41)), names, []),
        ("D_v1_plus_delta", sorted(set(old25) | set(delta_indices)), [names[i] for i in sorted(set(old25) | set(delta_indices))], []),
        ("E_v1_plus_nonredundant", sorted(set(old25) | set(nonredundant_indices)), [names[i] for i in sorted(set(old25) | set(nonredundant_indices))], []),
        ("F_full_41_regularized", list(range(41)), names, ["--dropout", "0.2", "--weight_decay", "1e-4"]),
        ("G_full_41_wider", list(range(41)), names, ["--hidden_dims", "256,128"]),
    ]

    tmp_root = Path(tempfile.mkdtemp(prefix="v135_ablation_"))
    all_results = {}
    try:
        tasks = []
        for short_name, idx, feat_names, extra in variants:
            subset_dir = tmp_root / short_name
            _prepare_subset(paired_dir, subset_dir, idx, feat_names)
            for seed in args.seeds:
                ckpt_dir = Path(args.output_dir) / f"ranker_{short_name}_s{seed}"
                tasks.append((short_name, seed, subset_dir, ckpt_dir, extra))

        def _run(t):
            short_name, seed, subset_dir, ckpt_dir, extra = t
            report = _train_one(subset_dir, ckpt_dir, seed, extra)
            return short_name, seed, report

        results_by_variant: Dict[str, List[Dict[str, Any]]] = {v[0]: [] for v in variants}
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            futures = {ex.submit(_run, t): t for t in tasks}
            for fut in as_completed(futures):
                short_name, seed, report = fut.result()
                results_by_variant[short_name].append(report)
                print(f"[ablation] {short_name} seed={seed} test_top1={report['test_metrics']['top1_accuracy']:.2%}", flush=True)

        for short_name, reports in results_by_variant.items():
            all_results[short_name] = {
                "seed_reports": [{"seed": args.seeds[i], **r["test_metrics"]} for i, r in enumerate(reports)],
                "aggregate": _aggregate([r["test_metrics"] for r in reports]),
                "epochs": [len(r["history"]) for r in reports],
            }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "CONTROLLED_FEATURE_ABLATION.json").write_text(json.dumps(all_results, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Controlled Neural-Network Feature Ablation",
        "",
        f"Dataset: `{args.paired_dataset_dir}`",
        f"Seeds: {args.seeds}",
        "",
        "| Model | N feats | Test top-1 | Tie-aware top-1 | Spearman | Pairwise acc | Model regret |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    key_order = ["A_v1_25", "B_afterstate_only_16", "C_full_41", "D_v1_plus_delta", "E_v1_plus_nonredundant", "F_full_41_regularized", "G_full_41_wider"]
    for k in key_order:
        agg = all_results[k]["aggregate"]
        n_feats = {
            "A_v1_25": 25, "B_afterstate_only_16": 16, "C_full_41": 41, "D_v1_plus_delta": 29,
            "E_v1_plus_nonredundant": 28, "F_full_41_regularized": 41, "G_full_41_wider": 41,
        }[k]
        lines.append(
            f"| {k} | {n_feats} | "
            f"{agg['top1_accuracy']['mean']:.2%} ± {agg['top1_accuracy']['std']:.2%} | "
            f"{agg.get('top1_tie_aware', agg['top1_accuracy'])['mean']:.2%} ± {agg.get('top1_tie_aware', agg['top1_accuracy'])['std']:.2%} | "
            f"{agg['spearman_mean']['mean']:.3f} ± {agg['spearman_mean']['std']:.3f} | "
            f"{agg['pairwise_accuracy']['mean']:.2%} ± {agg['pairwise_accuracy']['std']:.2%} | "
            f"{agg['model_regret_mean']['mean']:.4f} ± {agg['model_regret_mean']['std']:.4f} |"
        )
    (out_dir / "CONTROLLED_FEATURE_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print("Part 5 complete. Report written to", out_dir / "CONTROLLED_FEATURE_ABLATION.md")


if __name__ == "__main__":
    main()
