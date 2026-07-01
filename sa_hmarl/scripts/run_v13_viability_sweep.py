#!/usr/bin/env python3
"""End-to-end sweep for v1.3 viability-labelled R-ranker.

For each eta_phi, generate a dataset, train a ranker, and evaluate it in closed
loop. After screening, run a formal eval for the best 1-2 variants and the v1.2
baseline, then compile a summary table.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import (
    _candidate_feature_stats,
    _dataset_diagnostics,
)


ROOT = Path(__file__).resolve().parents[2]
DATASET_BASE = ROOT / "sa_hmarl" / "datasets" / "r_counterfactual_ranking_v1_3_sweep"
CHECKPOINT_BASE = ROOT / "sa_hmarl" / "checkpoints" / "r_counterfactual_ranking_v1_3_sweep"
EXPERIMENTS_DIR = ROOT / "sa_hmarl" / "experiments"

V1_2_CHECKPOINT = ROOT / "sa_hmarl" / "checkpoints" / "r_counterfactual_ranking_v1_2_mixed_low" / "ranking_model.pt"
AGENT_C = ROOT / "sa_hmarl" / "checkpoints" / "agent_c_delayaware_v2_snap24_best.pt"
AGENT_R = ROOT / "sa_hmarl" / "checkpoints" / "agent_r_mixed.pt"
DEEP_RMSA = ROOT / "sa_hmarl" / "checkpoints" / "deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt"
SUPERVISED = ROOT / "sa_hmarl" / "checkpoints" / "r_post_decision_h3_formal" / "c_post_decision_joint.pt"

SCENARIO_A = {
    "topology": "snap24_gnutella_reach",
    "num_slots": 24,
    "num_servers": 4,
    "k_paths": 5,
    "max_blocks": 10,
    "block_sort_strategy": "mixed",
    "split_profile": "default3",
    "num_splits": 3,
    "arrival_interval": 0.09,
    "holding_min": 4.0,
    "holding_max": 14.0,
    "deadline_min": 30.0,
    "deadline_max": 100.0,
    "size_min_mb": 5.0,
    "size_max_mb": 30.0,
    "edge_cost_min": 0.5,
    "edge_cost_max": 15.0,
    "requests_per_episode": 80,
}

RETURN_COEFS = {
    "return_current_block_coef": 3.0,
    "return_future_block_coef": 4.0,
    "return_future_nsb_coef": 3.0,
    "return_delay_coef": 0.03,
    "return_fs_coef": 0.05,
    "path_penalty_coef": 0.05,
    "fs_penalty_coef": 0.05,
    "horizon": 5,
    "candidate_mode": "v1",
}

TRAIN_ARGS = {
    "epochs": 80,
    "batch_size": 64,
    "lr": 3e-4,
    "hidden_dims": "128,64",
    "patience": 15,
}


def _run(cmd: List[str], desc: str) -> None:
    print(f"\n[{desc}] {' '.join(cmd)}", flush=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(ROOT / "sa_hmarl")
    subprocess.run(cmd, cwd=ROOT, env=env, check=True)


def _generate_base_dataset(args: argparse.Namespace) -> Path:
    """Generate the eta=0 dataset; Phi_after is stored but returns are pure v1.2."""
    out = DATASET_BASE / "eta_0"
    if args.skip_generation and (out / "metadata.json").exists():
        print("  Skip base dataset generation (already exists)")
        return out
    cmd = [
        sys.executable,
        "sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py",
        "--output_dir", str(out),
        "--agent_c_checkpoint", str(AGENT_C),
        "--agent_r_checkpoint", str(AGENT_R),
        "--splits", "train,val,test",
        "--train_episodes", str(args.train_episodes),
        "--val_episodes", str(args.val_episodes),
        "--test_episodes", str(args.test_episodes),
        "--viability_phi_coef", "0.0",
        "--viability_alpha_kc", str(args.viability_alpha_kc),
        "--viability_beta_kr", str(args.viability_beta_kr),
        "--viability_mode", args.viability_mode,
        "--device", args.device,
    ]
    for k, v in SCENARIO_A.items():
        cmd += [f"--{k}", str(v)]
    for k, v in RETURN_COEFS.items():
        cmd += [f"--{k}", str(v)]
    _run(cmd, "generate base (eta=0)")
    return out


def _derive_eta_dataset(base_dir: Path, eta: float, args: argparse.Namespace) -> Path:
    """Create a dataset for a specific eta by scaling the stored Phi_after."""
    out = DATASET_BASE / f"eta_{eta:g}"
    if args.skip_generation and (out / "metadata.json").exists():
        print(f"  Skip derive dataset for eta={eta} (already exists)")
        return out
    out.mkdir(parents=True, exist_ok=True)

    base_meta = json.loads((base_dir / "metadata.json").read_text(encoding="utf-8"))
    splits = base_meta["splits"].keys()

    diagnostics = {}
    for split in splits:
        data = dict(np.load(base_dir / f"{split}.npz", allow_pickle=False))
        if "phi_after" not in data:
            raise ValueError("Base dataset does not contain phi_after; regenerate with viability_phi_coef=0")
        # Adjust returns; keep everything else identical.
        data["returns"] = data["returns"] + float(eta) * data["phi_after"].astype(data["returns"].dtype)
        np.savez_compressed(out / f"{split}.npz", **data)
        diag = _dataset_diagnostics(data, base_meta.get("horizon", 5))
        diag.update(_candidate_feature_stats(data))
        diagnostics[split] = diag

    metadata = {
        **base_meta,
        "viability": {
            "viability_phi_coef": eta,
            "viability_alpha_kc": args.viability_alpha_kc,
            "viability_beta_kr": args.viability_beta_kr,
            "viability_mode": args.viability_mode,
            "derived_from": str(base_dir),
        },
        "diagnostics": diagnostics,
        "verdict": "DERIVED_FROM_ETA0",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"  Derived eta={eta} dataset at {out}")
    return out


def _train_ranker(dataset_dir: Path, eta: float, args: argparse.Namespace) -> Path:
    out = CHECKPOINT_BASE / f"eta_{eta:g}"
    if args.skip_training and (out / "ranking_model.pt").exists():
        print(f"  Skip training for eta={eta} (already exists)")
        return out
    cmd = [
        sys.executable,
        "sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py",
        "--dataset_dir", str(dataset_dir),
        "--output_dir", str(out),
        "--device", args.device,
    ]
    for k, v in TRAIN_ARGS.items():
        cmd += [f"--{k}", str(v)]
    _run(cmd, f"train eta={eta}")
    return out


def _eval_checkpoint(
    checkpoint: Path,
    output_json: Path,
    output_md: Path,
    seeds: List[int],
    episodes: int,
    methods: str,
    args: argparse.Namespace,
) -> Dict[str, Any]:
    cmd = [
        sys.executable,
        "sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_closed_loop.py",
        "--agent_c_checkpoint", str(AGENT_C),
        "--agent_r_checkpoint", str(AGENT_R),
        "--ranking_checkpoint", str(checkpoint),
        "--supervised_checkpoint", str(SUPERVISED),
        "--deep_rmsa_checkpoint", str(DEEP_RMSA),
        "--methods", methods,
        "--seeds", ",".join(str(s) for s in seeds),
        "--episodes", str(episodes),
        "--output_json", str(output_json),
        "--output_md", str(output_md),
        "--device", args.device,
    ]
    for k, v in SCENARIO_A.items():
        cmd += [f"--{k}", str(v)]
    _run(cmd, f"eval {checkpoint.name} seeds={seeds} eps={episodes}")
    return json.loads(output_json.read_text(encoding="utf-8"))


def _method_blocking(data: Dict[str, Any], method: str) -> float:
    per_seed = data["methods"][method]["per_seed"]
    return float(np.mean([v["blocking_rate"] for v in per_seed.values()]))


def _method_metric(data: Dict[str, Any], method: str, key: str) -> float:
    per_seed = data["methods"][method]["per_seed"]
    vals = [v[key] for v in per_seed.values() if key in v]
    return float(np.mean(vals)) if vals else 0.0


def _compile_summary(
    screening: Dict[float, Dict[str, Any]],
    formal: Dict[str, Dict[str, Any]],
    baseline_v12: Dict[str, Any],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    rows: List[Dict[str, Any]] = []

    # Baseline methods from the v1.2 formal eval (or dedicated baseline eval).
    for method in ["ppo_r", "deep_rmsa", "ksp_bf", "counterfactual_rank_only"]:
        if method not in baseline_v12["methods"]:
            continue
        rows.append({
            "method": method,
            "eta_phi": 0.0 if method == "counterfactual_rank_only" else None,
            "blocking_rate": _method_blocking(baseline_v12, method),
            "raw_mask_empty_rate": _method_metric(baseline_v12, method, "raw_mask_empty_rate"),
            "no_suitable_block_rate": _method_metric(baseline_v12, method, "no_suitable_block_rate"),
            "server_overload_rate": _method_metric(baseline_v12, method, "server_overload_rate"),
            "mean_delay_ms": _method_metric(baseline_v12, method, "mean_delay_ms"),
            "p95_delay_ms": _method_metric(baseline_v12, method, "p95_delay_ms"),
            "avg_fs": _method_metric(baseline_v12, method, "avg_fs"),
            "avg_path_km": _method_metric(baseline_v12, method, "avg_path_km"),
        })

    v12_blocking = _method_blocking(baseline_v12, "counterfactual_rank_only")
    deep_blocking = _method_blocking(baseline_v12, "deep_rmsa")

    for eta, data in formal.items():
        method = "counterfactual_rank_only"
        blocking = _method_blocking(data, method)
        row = {
            "method": f"v1.3_eta_{eta:g}",
            "eta_phi": eta,
            "blocking_rate": blocking,
            "raw_mask_empty_rate": _method_metric(data, method, "raw_mask_empty_rate"),
            "no_suitable_block_rate": _method_metric(data, method, "no_suitable_block_rate"),
            "server_overload_rate": _method_metric(data, method, "server_overload_rate"),
            "mean_delay_ms": _method_metric(data, method, "mean_delay_ms"),
            "p95_delay_ms": _method_metric(data, method, "p95_delay_ms"),
            "avg_fs": _method_metric(data, method, "avg_fs"),
            "avg_path_km": _method_metric(data, method, "avg_path_km"),
            "delta_blocking_vs_v12": blocking - v12_blocking,
            "delta_blocking_vs_deeprmsa": blocking - deep_blocking,
            "delta_delay_vs_v12": _method_metric(data, method, "mean_delay_ms") - _method_metric(baseline_v12, "counterfactual_rank_only", "mean_delay_ms"),
            "delta_overload_vs_v12": _method_metric(data, method, "server_overload_rate") - _method_metric(baseline_v12, "counterfactual_rank_only", "server_overload_rate"),
        }
        # Dataset / training diagnostics.
        dataset_dir = DATASET_BASE / f"eta_{eta:g}"
        checkpoint_dir = CHECKPOINT_BASE / f"eta_{eta:g}"
        try:
            meta = json.loads((dataset_dir / "metadata.json").read_text(encoding="utf-8"))
            train_diag = meta["diagnostics"].get("train", {})
            row["dataset_phi_after_mean"] = train_diag.get("phi_after_mean", 0.0)
            row["dataset_phi_after_std"] = train_diag.get("phi_after_std", 0.0)
            row["dataset_phi_after_range_nonzero_rate"] = train_diag.get("phi_after_range_nonzero_rate", 0.0)
        except Exception:
            row["dataset_phi_after_mean"] = 0.0
            row["dataset_phi_after_std"] = 0.0
            row["dataset_phi_after_range_nonzero_rate"] = 0.0
        try:
            tr = json.loads((checkpoint_dir / "training_report.json").read_text(encoding="utf-8"))
            row["training_top1"] = tr["test_metrics"]["top1_accuracy"]
            row["training_spearman"] = tr["test_metrics"]["spearman_mean"]
        except Exception:
            row["training_top1"] = 0.0
            row["training_spearman"] = 0.0
        rows.append(row)

    return {"rows": rows, "config": vars(args)}


def _write_summary(summary: Dict[str, Any], path_json: Path, path_md: Path) -> None:
    path_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = summary["rows"]
    headers = [
        "method", "eta_phi", "blocking_rate", "raw_empty", "nsb", "overload",
        "delay_mean", "delay_p95", "avg_fs", "avg_path_km",
        "Δblk_v12", "Δblk_deep", "Δdelay_v12", "Δoverload_v12",
        "phi_mean", "phi_std", "phi_nonzero", "train_top1", "train_spearman",
    ]
    lines = ["# v1.3 Viability Sweep Summary", ""]
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "|".join("---" for _ in headers) + "|")
    for r in rows:
        vals = [
            str(r.get("method", "")),
            f"{r.get('eta_phi') if r.get('eta_phi') is not None else ''}",
            f"{r.get('blocking_rate', 0)*100:.2f}%",
            f"{r.get('raw_mask_empty_rate', 0)*100:.2f}%",
            f"{r.get('no_suitable_block_rate', 0)*100:.2f}%",
            f"{r.get('server_overload_rate', 0)*100:.2f}%",
            f"{r.get('mean_delay_ms', 0):.3f}",
            f"{r.get('p95_delay_ms', 0):.3f}",
            f"{r.get('avg_fs', 0):.3f}",
            f"{r.get('avg_path_km', 0):.1f}",
            f"{r.get('delta_blocking_vs_v12', 0)*100:+.2f}pp" if "delta_blocking_vs_v12" in r else "",
            f"{r.get('delta_blocking_vs_deeprmsa', 0)*100:+.2f}pp" if "delta_blocking_vs_deeprmsa" in r else "",
            f"{r.get('delta_delay_vs_v12', 0):+.3f}" if "delta_delay_vs_v12" in r else "",
            f"{r.get('delta_overload_vs_v12', 0)*100:+.2f}pp" if "delta_overload_vs_v12" in r else "",
            f"{r.get('dataset_phi_after_mean', 0):.4f}",
            f"{r.get('dataset_phi_after_std', 0):.4f}",
            f"{r.get('dataset_phi_after_range_nonzero_rate', 0):.2%}",
            f"{r.get('training_top1', 0):.2%}",
            f"{r.get('training_spearman', 0):.3f}",
        ]
        lines.append("| " + " | ".join(vals) + " |")
    path_md.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--etas", type=lambda s: [float(x) for x in s.split(",")], default=[0.1, 0.3, 0.5, 1.0])
    parser.add_argument("--train_episodes", type=int, default=10)
    parser.add_argument("--val_episodes", type=int, default=10)
    parser.add_argument("--test_episodes", type=int, default=10)
    parser.add_argument("--screening_seeds", type=lambda s: [int(x) for x in s.split(",")], default=[3030, 4040, 5050])
    parser.add_argument("--screening_episodes", type=int, default=5)
    parser.add_argument("--formal_seeds", type=lambda s: [int(x) for x in s.split(",")], default=[3030, 4040, 5050, 6060, 7070])
    parser.add_argument("--formal_episodes", type=int, default=20)
    parser.add_argument("--top_k_formal", type=int, default=2)
    parser.add_argument("--viability_alpha_kc", type=float, default=1.0)
    parser.add_argument("--viability_beta_kr", type=float, default=0.3)
    parser.add_argument("--viability_mode", default="obs_c_after")
    parser.add_argument("--skip_generation", action="store_true")
    parser.add_argument("--skip_training", action="store_true")
    parser.add_argument("--skip_screening", action="store_true")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    start = time.time()

    screening_results: Dict[float, Dict[str, Any]] = {}
    base_dir = _generate_base_dataset(args)
    for eta in args.etas:
        print(f"\n{'='*60}\nEta_phi = {eta}\n{'='*60}", flush=True)
        if eta == 0.0:
            dataset_dir = base_dir
        else:
            dataset_dir = _derive_eta_dataset(base_dir, eta, args)
        checkpoint_dir = _train_ranker(dataset_dir, eta, args)

        out_json = EXPERIMENTS_DIR / f"r_ranker_v1_3_phi_eta_{eta:g}_screening.json"
        out_md = EXPERIMENTS_DIR / f"r_ranker_v1_3_phi_eta_{eta:g}_screening.md"
        data = _eval_checkpoint(
            checkpoint_dir / "ranking_model.pt",
            out_json, out_md,
            args.screening_seeds, args.screening_episodes,
            "counterfactual_rank_only,ppo_r,deep_rmsa,ksp_bf",
            args,
        )
        screening_results[eta] = data
        blocking = _method_blocking(data, "counterfactual_rank_only")
        print(f"  Screening blocking eta={eta}: {blocking*100:.2f}%", flush=True)

    # Pick top-k by screening blocking.
    ranked = sorted(args.etas, key=lambda e: _method_blocking(screening_results[e], "counterfactual_rank_only"))
    top_etas = ranked[: args.top_k_formal]
    print(f"\nTop {args.top_k_formal} etas for formal eval: {top_etas}", flush=True)

    # Formal eval: v1.2 baseline (all methods) and top v1.3 checkpoints.
    baseline_out_json = EXPERIMENTS_DIR / "r_ranker_v1_3_baseline_v12_formal.json"
    baseline_out_md = EXPERIMENTS_DIR / "r_ranker_v1_3_baseline_v12_formal.md"
    baseline_v12 = _eval_checkpoint(
        V1_2_CHECKPOINT, baseline_out_json, baseline_out_md,
        args.formal_seeds, args.formal_episodes,
        "counterfactual_rank_only,ppo_r,deep_rmsa,ksp_bf",
        args,
    )

    formal_results: Dict[float, Dict[str, Any]] = {}
    for eta in top_etas:
        out_json = EXPERIMENTS_DIR / f"r_ranker_v1_3_phi_eta_{eta:g}_formal.json"
        out_md = EXPERIMENTS_DIR / f"r_ranker_v1_3_phi_eta_{eta:g}_formal.md"
        data = _eval_checkpoint(
            CHECKPOINT_BASE / f"eta_{eta:g}" / "ranking_model.pt",
            out_json, out_md,
            args.formal_seeds, args.formal_episodes,
            "counterfactual_rank_only",
            args,
        )
        formal_results[eta] = data

    summary = _compile_summary(screening_results, formal_results, baseline_v12, args)
    summary["elapsed_seconds"] = time.time() - start
    summary["top_etas_formal"] = top_etas
    _write_summary(
        summary,
        EXPERIMENTS_DIR / "r_ranker_v1_3_viability_sweep_summary.json",
        EXPERIMENTS_DIR / "r_ranker_v1_3_viability_sweep_summary.md",
    )
    print("\nSweep complete. Summary written to:")
    print(f"  {EXPERIMENTS_DIR / 'r_ranker_v1_3_viability_sweep_summary.md'}")


if __name__ == "__main__":
    main()
