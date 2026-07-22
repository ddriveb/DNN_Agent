"""System causal ceiling analysis for corrected SA-HMARL v1.35.

Uses either:
  1. a closed-loop evaluation JSON (preferred), which gives the true baseline
     blocking-rate decomposition, or
  2. the paired dataset's per-group future counts (fallback).

It also reports candidate-dependent future optical headroom from Label-B counters
when they are present.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")


def _headroom_from_dataset(dataset_dir: Path) -> Dict[str, Any]:
    """Compute candidate-dependent future optical range from Label-B counters."""
    out: Dict[str, Any] = {"available": False, "splits": {}}
    for split in ("train", "val", "test"):
        npz_path = dataset_dir / f"{split}.npz"
        if not npz_path.exists():
            continue
        d = dict(np.load(str(npz_path), allow_pickle=True))
        if "future_optical_blocked_counts_h5" not in d:
            continue
        mask = d["mask"].astype(bool)
        n_groups = mask.shape[0]
        horizon_stats: Dict[str, Any] = {}
        for h in [5, 12, 20]:
            key = f"future_optical_blocked_counts_h{h}"
            counts = d[key]
            ranges = []
            nonzero = 0
            positive = 0
            for i in range(n_groups):
                vals = counts[i][mask[i]].astype(float)
                if vals.size == 0:
                    continue
                rng = float(vals.max() - vals.min())
                ranges.append(rng)
                if rng > 1e-8:
                    nonzero += 1
                    if vals.max() > vals.min():
                        positive += 1
            arr = np.array(ranges)
            n = max(len(ranges), 1)
            horizon_stats[f"H{h}"] = {
                "nonzero_range_rate": nonzero / n,
                "positive_headroom_rate": positive / n,
                "mean_range": float(arr.mean()) if arr.size else 0.0,
                "max_range": float(arr.max()) if arr.size else 0.0,
            }
        out["splits"][split] = horizon_stats
        out["available"] = True
    return out


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--closed_loop_json",
                        help="Path to CLOSED_LOOP_EVAL.json from v135_corrected_closed_loop.py.")
    parser.add_argument("--dataset_dir", default="sa_hmarl/datasets/v135_corrected_paired_feature_ablation/full")
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Baseline breakdown from closed-loop if available.
    baseline: Dict[str, Any] = {}
    source = "dataset"
    if args.closed_loop_json:
        cl = json.loads(Path(args.closed_loop_json).read_text(encoding="utf-8"))
        ppo = cl["methods"].get("ppo_r", {}).get("aggregate", {})
        total_blocking = ppo.get("blocking_rate", 0.0)
        overload = ppo.get("server_overload_rate", 0.0)
        optical = ppo.get("optical_block_rate", 0.0)
        baseline = {
            "blocking_rate": total_blocking,
            "server_overload_rate": overload,
            "optical_block_rate": optical,
            "overload_share_of_blocking": overload / max(total_blocking, 1e-9),
            "optical_share_of_blocking": optical / max(total_blocking, 1e-9),
        }
        source = f"closed_loop:{args.closed_loop_json}"
    else:
        # Fallback: use PPO action's future blocked counts from the dataset.
        dataset_dir = Path(args.dataset_dir)
        d = dict(np.load(str(dataset_dir / "train.npz"), allow_pickle=True))
        mask = d["mask"].astype(bool)
        ppo_idx = d["ppo_action_index"]
        future_blocked = d["future_blocked_counts"]
        future_overload = d["future_server_overload_counts"]
        total = 0
        blocked = 0
        overload_blocked = 0
        for i in range(mask.shape[0]):
            legal = mask[i]
            n = int(legal.sum())
            if n == 0:
                continue
            idx = int(ppo_idx[i])
            if idx < 0 or idx >= n:
                continue
            total += 1
            if int(future_blocked[i][idx]) > 0:
                blocked += 1
                if int(future_overload[i][idx]) > 0:
                    overload_blocked += 1
        baseline = {
            "blocking_rate": blocked / max(total, 1),
            "server_overload_rate": overload_blocked / max(total, 1),
            "optical_block_rate": (blocked - overload_blocked) / max(total, 1),
            "overload_share_of_blocking": overload_blocked / max(blocked, 1),
            "optical_share_of_blocking": (blocked - overload_blocked) / max(blocked, 1),
        }
        source = f"dataset:{args.dataset_dir}"

    headroom = _headroom_from_dataset(Path(args.dataset_dir))

    total_blocking = baseline["blocking_rate"]
    overload = baseline["server_overload_rate"]
    optical = baseline["optical_block_rate"]
    ceiling = {
        "source": source,
        "ppo_r_blocking_rate": total_blocking,
        "overload_component": overload,
        "optical_component": optical,
        "r_side_best_case_blocking_rate": max(total_blocking - optical, 0.0),
        "r_side_theoretical_max_reduction_pp": optical,
        "r_side_theoretical_max_reduction_relative": optical / max(total_blocking, 1e-9),
    }

    result = {
        "baseline_breakdown": baseline,
        "causal_ceiling": ceiling,
        "optical_headroom": headroom,
        "regime": "compute-overload-dominated" if overload > optical else "optical-dominated",
    }

    (output_dir / "SYSTEM_CAUSAL_CEILING.json").write_text(
        json.dumps(result, indent=2, default=str), encoding="utf-8"
    )

    lines = [
        "# System Causal Ceiling Analysis (corrected)",
        "",
        f"Source: `{source}`",
        "",
        "## Baseline PPO-R blocking decomposition",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Blocking rate | {baseline['blocking_rate']:.2%} |",
        f"| Server overload rate | {baseline['server_overload_rate']:.2%} |",
        f"| Optical-block rate | {baseline['optical_block_rate']:.2%} |",
        f"| Overload share of blocking | {baseline['overload_share_of_blocking']:.2%} |",
        f"| Optical share of blocking | {baseline['optical_share_of_blocking']:.2%} |",
        "",
        "## R-side theoretical ceiling",
        "",
        f"- Best-case blocking if optical failures eliminated: **{ceiling['r_side_best_case_blocking_rate']:.2%}**",
        f"- Maximum absolute reduction: **{ceiling['r_side_theoretical_max_reduction_pp']:.2%}**",
        f"- Maximum relative reduction: **{ceiling['r_side_theoretical_max_reduction_relative']:.1%}**",
        "",
        f"**Regime:** {result['regime']}",
        "",
    ]

    if headroom["available"]:
        lines += [
            "## Candidate-dependent future optical headroom (Label B)",
            "",
            "| Split / Horizon | Nonzero range rate | Positive headroom rate | Mean range | Max range |",
            "|---|---:|---:|---:|---:|",
        ]
        for split, stats in headroom["splits"].items():
            for h, s in stats.items():
                lines.append(
                    f"| {split} {h} | {s['nonzero_range_rate']:.2%} | "
                    f"{s['positive_headroom_rate']:.2%} | "
                    f"{s['mean_range']:.4f} | {s['max_range']:.1f} |"
                )
        lines.append("")
    else:
        lines.append(
            "Label-B counters are not present in the dataset; optical headroom could not be measured."
        )
        lines.append("")

    (output_dir / "SYSTEM_CAUSAL_CEILING.md").write_text("\n".join(lines), encoding="utf-8")
    print("Causal ceiling report:", output_dir / "SYSTEM_CAUSAL_CEILING.md")


if __name__ == "__main__":
    main()
