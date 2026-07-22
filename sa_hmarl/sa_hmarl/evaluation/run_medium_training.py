"""Train SA-HMARL v1.3 rankers on the medium dataset and its subsets.

Runs 3 seeds for each train-group size (1.3k, 3k, 5k, full) using the simple loss:
  reg_weight=1.0, lambda_pair=0, lambda_hard=0.

Outputs:
  - per-checkpoint training_report.json / ranking_model.pt
  - sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_TRAINING_REPORT.md
  - sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_TRAINING_REPORT.json
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


DATASET_NAME = "r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium"
SUBSET_SIZES = [(1300, "1k3"), (3000, "3k"), (5000, "5k")]
SEEDS = [42, 43, 44]
MAX_TRAIN_WORKERS = 3


def _make_subset(source_dir: Path, target_dir: Path, n_groups: int) -> None:
    target_dir.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        src = dict(np.load(str(source_dir / f"{split}.npz"), allow_pickle=False))
        if split == "train":
            n = min(n_groups, src["features"].shape[0])
            subset = {k: v[:n] for k, v in src.items() if v.shape[0] == src["features"].shape[0]}
            # Keep any scalar / object arrays unchanged (they do not have group axis).
            for k, v in src.items():
                if k not in subset:
                    subset[k] = v
        else:
            subset = src
        np.savez_compressed(str(target_dir / f"{split}.npz"), **subset)
    shutil.copy(source_dir / "metadata.json", target_dir / "metadata.json")


def _run_training(
    dataset_dir: Path,
    output_dir: Path,
    seed: int,
    root: Path,
) -> Dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.training.train_r_counterfactual_ranking",
        "--dataset_dir", str(dataset_dir),
        "--output_dir", str(output_dir),
        "--model_type", "mlp",
        "--hidden_dims", "128,64",
        "--reg_weight", "1.0",
        "--lambda_pair", "0.0",
        "--lambda_hard", "0.0",
        "--seed", str(seed),
        "--epochs", "80",
        "--patience", "15",
        "--batch_size", "64",
        "--lr", "0.0003",
        "--selection_metric", "top1",
        "--device", "cpu",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    log_path = output_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(
            cmd,
            cwd=root,
            env=env,
            stdout=logf,
            stderr=subprocess.STDOUT,
            timeout=3600,
        )
    report_path = output_dir / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"error": "no report", "returncode": proc.returncode}
    report["_output_dir"] = str(output_dir)
    report["_seed"] = seed
    report["_returncode"] = proc.returncode
    return report


def _summarize_reports(reports: List[Dict[str, Any]]) -> Dict[str, Any]:
    by_size: Dict[str, List[Dict[str, Any]]] = {}
    for r in reports:
        key = r.get("_size_key", "full")
        by_size.setdefault(key, []).append(r)

    summary: Dict[str, Any] = {}
    for key, reps in by_size.items():
        metrics = {}
        for m in ["top1_accuracy", "tie_aware_top1", "top3_accuracy",
                  "spearman_mean", "kendall_tau_b", "ndcg_at_3",
                  "model_regret_mean", "ppo_regret_mean", "relative_regret_reduction",
                  "ppo_agreement", "pairwise_accuracy"]:
            vals = [rep.get("test_metrics", {}).get(m) for rep in reps]
            vals = [v for v in vals if v is not None]
            if vals:
                metrics[m] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "min": float(np.min(vals)),
                    "max": float(np.max(vals)),
                    "values": vals,
                }
        summary[key] = {
            "seeds": [rep.get("_seed") for rep in reps],
            "n_seeds": len(reps),
            "metrics": metrics,
        }
    return summary


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    source_dir = root / "sa_hmarl" / "datasets" / DATASET_NAME
    if not (source_dir / "metadata.json").exists():
        print(f"[train] Medium dataset not found: {source_dir}")
        return 1

    exp_dir = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking"
    exp_dir.mkdir(parents=True, exist_ok=True)

    base_out = root / "sa_hmarl" / "checkpoints" / f"{DATASET_NAME}_training"
    base_out.mkdir(parents=True, exist_ok=True)

    # Build subset dirs.
    work_dir = Path(tempfile.mkdtemp(prefix="medium_training_"))
    dataset_dirs: List[Tuple[Path, str]] = []
    for n, label in SUBSET_SIZES:
        sub = work_dir / f"{DATASET_NAME}_{label}"
        _make_subset(source_dir, sub, n)
        dataset_dirs.append((sub, label))
    dataset_dirs.append((source_dir, "full"))

    tasks: List[Tuple[Path, str, int]] = []
    for ds_dir, label in dataset_dirs:
        for seed in SEEDS:
            out = base_out / f"{label}_seed{seed}"
            tasks.append((ds_dir, label, seed, out, root))

    reports: List[Dict[str, Any]] = []
    print(f"[train] Launching {len(tasks)} training runs (max_workers={MAX_TRAIN_WORKERS})")
    with ProcessPoolExecutor(max_workers=MAX_TRAIN_WORKERS) as pool:
        futures = {
            pool.submit(_run_training, ds_dir, out, seed, root): (size_label, seed)
            for ds_dir, size_label, seed, out, root in tasks
        }
        for fut in as_completed(futures):
            size_label, seed = futures[fut]
            try:
                report = fut.result()
            except Exception as exc:
                report = {"error": str(exc), "_size_key": size_label, "_seed": seed}
            report["_size_key"] = size_label
            reports.append(report)
            tm = report.get("test_metrics", {})
            print(
                f"[train] {size_label} seed={seed} "
                f"top1={tm.get('top1_accuracy', -1):.2%} "
                f"spearman={tm.get('spearman_mean', -99):.3f} "
                f"regret_red={tm.get('relative_regret_reduction', -99):.3f}",
                flush=True,
            )

    summary = _summarize_reports(reports)
    payload = {
        "dataset": str(source_dir),
        "subsets": [label for _, label in dataset_dirs],
        "seeds": SEEDS,
        "summary": summary,
        "reports": reports,
    }
    json_path = exp_dir / "MEDIUM_TRAINING_REPORT.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Markdown report.
    lines = [
        "# SA-HMARL v1.3 Medium Dataset Training Report",
        "",
        f"- Dataset: `{source_dir}`",
        "- Model: MLP [128,64]",
        "- Loss: listwise KL + SmoothL1 MSE (reg_weight=1.0, lambda_pair=0, lambda_hard=0)",
        f"- Seeds: {SEEDS}",
        "",
        "## Learning curve",
        "",
        "| Train groups | Top-1 | Tie-aware Top-1 | Top-3 | Spearman | Kendall tau-b | NDCG@3 | Model regret | PPO regret | Rel. regret red. | PPO agree |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    ordered_keys = [label for _, label in dataset_dirs]
    for key in ordered_keys:
        if key not in summary:
            continue
        m = summary[key]["metrics"]
        def fmt(name):
            if name not in m:
                return "—"
            return f"{m[name]['mean']:.3f}±{m[name]['std']:.3f}"
        lines.append(
            f"| {key} | {fmt('top1_accuracy')} | {fmt('tie_aware_top1')} | "
            f"{fmt('top3_accuracy')} | {fmt('spearman_mean')} | {fmt('kendall_tau_b')} | "
            f"{fmt('ndcg_at_3')} | {fmt('model_regret_mean')} | {fmt('ppo_regret_mean')} | "
            f"{fmt('relative_regret_reduction')} | {fmt('ppo_agreement')} |"
        )

    lines.extend(["", "## Per-seed details", "", "| Subset | Seed | Top-1 | Spearman | Rel. regret red. | Output dir |", "|---|---|---:|---:|---|---|"])
    for r in reports:
        tm = r.get("test_metrics", {})
        lines.append(
            f"| {r.get('_size_key', '?')} | {r.get('_seed', '?')} | "
            f"{tm.get('top1_accuracy', -1):.2%} | {tm.get('spearman_mean', -99):.3f} | "
            f"{tm.get('relative_regret_reduction', -99):.3f} | {r.get('_output_dir', '')} |"
        )

    (exp_dir / "MEDIUM_TRAINING_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[train] Wrote {exp_dir / 'MEDIUM_TRAINING_REPORT.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
