"""Retrain SA-HMARL v1.3 rankers on the full medium dataset with regret-based checkpoint selection.

Runs 3 seeds in parallel.  Checkpoint is selected by lowest validation model regret,
not by test set performance.

Outputs:
    sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_*/ranking_model.pt
    sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_TRAINING_REPORT_CORRECTED.md
    sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_TRAINING_REPORT_CORRECTED.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

import numpy as np


DATASET_DIR = "sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium"
OUTPUT_BASE = "sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected"
SEEDS = [42, 123, 456]
MAX_WORKERS = 3


def _run_training(seed: int, root: Path) -> Dict[str, Any]:
    out_dir = root / OUTPUT_BASE / f"seed_{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.training.train_r_counterfactual_ranking",
        "--dataset_dir", str(root / DATASET_DIR),
        "--output_dir", str(out_dir),
        "--model_type", "mlp",
        "--hidden_dims", "128,64",
        "--reg_weight", "1.0",
        "--lambda_pair", "0.0",
        "--lambda_hard", "0.0",
        "--selection_metric", "regret",
        "--seed", str(seed),
        "--epochs", "80",
        "--patience", "15",
        "--batch_size", "64",
        "--lr", "0.0003",
        "--device", "cpu",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    log_path = out_dir / "train.log"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=logf, stderr=subprocess.STDOUT, timeout=3600)
    report_path = out_dir / "training_report.json"
    if report_path.exists():
        report = json.loads(report_path.read_text(encoding="utf-8"))
    else:
        report = {"error": "no report", "returncode": proc.returncode}
    report["_output_dir"] = str(out_dir)
    report["_seed"] = seed
    report["_returncode"] = proc.returncode
    return report


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    dataset_dir = root / DATASET_DIR
    if not (dataset_dir / "metadata.json").exists():
        print(f"[retrain] Dataset not found: {dataset_dir}")
        return 1

    reports: List[Dict[str, Any]] = []
    print(f"[retrain] Retraining {len(SEEDS)} seeds with selection_metric=regret (max_workers={MAX_WORKERS})")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_training, seed, root): seed for seed in SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                report = fut.result()
            except Exception as exc:
                report = {"error": str(exc), "_seed": seed, "_returncode": -1}
            reports.append(report)
            tm = report.get("test_metrics", {})
            print(
                f"[retrain] seed={seed} "
                f"val_model_regret={report.get('best_selection_score', 0):.4f} "
                f"test_model_regret={tm.get('model_regret_mean', -1):.4f} "
                f"test_ppo_regret={tm.get('ppo_regret_mean', -1):.4f} "
                f"aggregate_reduction={tm.get('aggregate_regret_reduction', -99):.2%}",
                flush=True,
            )

    # Select checkpoint by lowest validation model regret (selection_score = -val regret).
    valid_reports = [r for r in reports if r.get("_returncode") == 0 and "best_selection_score" in r]
    if not valid_reports:
        print("[retrain] No successful training runs")
        return 1
    selected = min(valid_reports, key=lambda r: r["best_selection_score"])
    selected_path = Path(selected["_output_dir"]) / "ranking_model.pt"

    exp_dir = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking"
    exp_dir.mkdir(parents=True, exist_ok=True)

    summary = {}
    for r in reports:
        tm = r.get("test_metrics", {})
        summary[f"seed_{r.get('_seed', '?')}"] = {
            "val_selection_score": r.get("best_selection_score"),
            "test_model_regret": tm.get("model_regret_mean"),
            "test_ppo_regret": tm.get("ppo_regret_mean"),
            "absolute_regret_improvement": tm.get("absolute_regret_improvement"),
            "aggregate_regret_reduction": tm.get("aggregate_regret_reduction"),
            "model_better_rate": tm.get("model_better_than_ppo_rate"),
            "oracle_tie_hit_rate": tm.get("oracle_tie_hit_rate"),
            "ppo_oracle_tie_hit_rate": tm.get("ppo_oracle_tie_hit_rate"),
        }

    payload = {
        "dataset": str(dataset_dir),
        "seeds": SEEDS,
        "selection_metric": "regret",
        "selected_seed": selected["_seed"],
        "selected_checkpoint": str(selected_path),
        "summary": summary,
        "reports": reports,
    }
    json_path = exp_dir / "MEDIUM_TRAINING_REPORT_CORRECTED.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    lines = [
        "# SA-HMARL v1.3 Medium Dataset Training Report (Corrected)",
        "",
        f"- Dataset: `{dataset_dir}`",
        "- Model: MLP [128,64]",
        "- Loss: listwise KL + SmoothL1 MSE (reg_weight=1.0, lambda_pair=0, lambda_hard=0)",
        f"- Seeds: {SEEDS}",
        "- Checkpoint selection: **lowest validation model regret** (selection_metric=regret)",
        "",
        "## Per-seed results",
        "",
        "| Seed | Val selection score | Test model regret | Test PPO regret | Abs. improvement | Aggregate reduction | Model better rate | Oracle tie hit |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in SEEDS:
        s = summary[f"seed_{seed}"]
        lines.append(
            f"| {seed} | {s['val_selection_score']:.4f} | {s['test_model_regret']:.4f} | "
            f"{s['test_ppo_regret']:.4f} | {s['absolute_regret_improvement']:.4f} | "
            f"{s['aggregate_regret_reduction']:.2%} | {s['model_better_rate']:.2%} | {s['oracle_tie_hit_rate']:.2%} |"
        )
    lines.extend([
        "",
        f"## Selected checkpoint",
        "",
        f"- Seed: {selected['_seed']}",
        f"- Path: `{selected_path}`",
        f"- Selection was based on validation regret only; test metrics are reported for diagnostics.",
    ])
    (exp_dir / "MEDIUM_TRAINING_REPORT_CORRECTED.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[retrain] Selected seed {selected['_seed']} checkpoint: {selected_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
