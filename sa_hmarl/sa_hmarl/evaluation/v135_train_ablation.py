"""Train paired v1 vs poststate_v1 rankers from a v1.35 canonical dataset.

For each feature key (`features_v1`, `features_poststate_v1`) the script:
  - prepares a temporary dataset dir with the standard `features` array,
  - trains 3 seeds with identical hyperparameters,
  - reports mean/std of test metrics.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


def _prepare_subset(paired_dir: Path, output_dir: Path, feature_key: str, feature_names: List[str]) -> Path:
    """Create a temporary dataset dir compatible with train_r_counterfactual_ranking.py."""
    output_dir.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        src = np.load(str(paired_dir / f"{split}.npz"), allow_pickle=True)
        features = src[feature_key]
        np.savez_compressed(
            output_dir / f"{split}.npz",
            features=features,
            returns=src["returns"],
            mask=src["mask"],
            ppo_action_index=src["ppo_action_index"],
        )
    # Compute mean/std over train features (valid entries only).
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
        "feature_key": feature_key,
        "candidate_mode": "ppo_r_topk_only",
        "max_candidates": int(features.shape[1]),
        "horizon": 5,
        "return_coefs": {
            "current_block": 1.0,
            "future_block": 1.0,
            "future_nsb": 1.0,
            "future_server_overload": 0.0,
            "delay": 0.0,
            "fs": 0.0,
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
    env = {**dict(os.environ), "OMP_NUM_THREADS": "1", "OPENBLAS_NUM_THREADS": "1"}
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
    parser.add_argument("--paired_dataset_dir", default="sa_hmarl/datasets/v135_paired_feature_ablation_medium")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v135_parallel_paired")
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--train_extra_args", default="", help="Extra args passed to train script, e.g. '--epochs 80 --lr 3e-4'")
    args = parser.parse_args()

    paired_dir = Path(args.paired_dataset_dir)
    meta = json.loads((paired_dir / "metadata.json").read_text(encoding="utf-8"))
    feature_specs = [
        ("features_v1", meta["feature_names_v1"], "v1"),
        ("features_poststate_v1", meta["feature_names_poststate_v1"], "poststate_v1"),
    ]

    tmp_root = Path(tempfile.mkdtemp(prefix="v135_train_"))
    all_results = {}
    try:
        for feature_key, feature_names, short_name in feature_specs:
            print(f"[train_ablation] Training {short_name} ({len(feature_names)} dims)...", flush=True)
            subset_dir = tmp_root / short_name
            _prepare_subset(paired_dir, subset_dir, feature_key, feature_names)
            seed_reports = []
            for seed in args.seeds:
                ckpt_dir = Path(args.output_dir) / f"ranker_{short_name}_s{seed}"
                extra = args.train_extra_args.split() if args.train_extra_args else []
                report = _train_one(subset_dir, ckpt_dir, seed, extra)
                seed_reports.append(report["test_metrics"])
            all_results[short_name] = {
                "seed_test_metrics": seed_reports,
                "aggregate": _aggregate(seed_reports),
            }
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "FEATURE_ONLY_ABLATION.json"
    out_json.write_text(json.dumps(all_results, indent=2), encoding="utf-8")

    lines = [
        "# Feature-Only Ablation Training Results",
        "",
        f"Dataset: `{args.paired_dataset_dir}`",
        f"Training seeds: {args.seeds}",
        "",
        "| Model | Test top-1 | Test Spearman | Pairwise acc | PPO regret |",
        "|---|---:|---:|---:|---:|",
    ]
    for short_name in ("v1", "poststate_v1"):
        agg = all_results[short_name]["aggregate"]
        lines.append(
            f"| {short_name} | "
            f"{agg['top1_accuracy']['mean']:.2%} ± {agg['top1_accuracy']['std']:.2%} | "
            f"{agg['spearman_mean']['mean']:.3f} ± {agg['spearman_mean']['std']:.3f} | "
            f"{agg['pairwise_accuracy']['mean']:.2%} ± {agg['pairwise_accuracy']['std']:.2%} | "
            f"{agg['ppo_regret_mean']['mean']:.4f} ± {agg['ppo_regret_mean']['std']:.4f} |"
        )
    (out_dir / "FEATURE_ONLY_ABLATION.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[train_ablation] Saved report to {out_json}")


if __name__ == "__main__":
    main()
