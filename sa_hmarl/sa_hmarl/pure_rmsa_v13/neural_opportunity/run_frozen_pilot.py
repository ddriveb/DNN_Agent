"""Frozen five-seed closed-loop diagnostic; no tuning on pilot seeds."""
from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path

import numpy as np

from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.run_phaseA import (
    _bootstrap_ci,
    _wtl,
)

from .protocol import PILOT_SEEDS, PROTOCOL_ID, TOPOLOGY_CONFIG
from .run_closed_loop_smoke import _run

ARMS = (
    "ksp_ff_k50",
    "full_direct",
    "opportunity_teacher",
    "neural_opportunity",
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--eval-requests", type=int, default=10000)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise RuntimeError("native Direct Sketch kernel must be unset")
    name_by_topology = {
        "xlron_nsfnet_deeprmsa": "nsfnet",
        "xlron_usnet_gcnrmsa": "usnet",
        "xlron_jpn48": "jpn48",
    }
    rows = []
    for topology, config in TOPOLOGY_CONFIG.items():
        weights = (
            args.weights_root
            / name_by_topology[topology]
            / "deployment_weights.npz"
        )
        for seed in PILOT_SEEDS:
            for arm in ARMS:
                row = _run(
                    topology,
                    config,
                    arm,
                    weights,
                    args.warmup,
                    args.eval_requests,
                    seed=seed,
                )
                row.update(topology=topology, seed=seed)
                rows.append(row)
    summaries = {}
    for topology_index, topology in enumerate(TOPOLOGY_CONFIG):
        selected = [row for row in rows if row["topology"] == topology]
        by_arm = {
            arm: {row["seed"]: row for row in selected if row["arm"] == arm}
            for arm in ARMS
        }
        means = {
            arm: {
                "blocking": float(
                    np.mean([by_arm[arm][seed]["blocking"] for seed in PILOT_SEEDS])
                ),
                "mean_ms": float(
                    np.mean([by_arm[arm][seed]["mean_ms"] for seed in PILOT_SEEDS])
                ),
            }
            for arm in ARMS
        }
        comparisons = {}
        for comparison_index, baseline in enumerate(
            ("ksp_ff_k50", "full_direct", "opportunity_teacher")
        ):
            deltas = [
                100.0
                * (
                    by_arm["neural_opportunity"][seed]["blocking"]
                    - by_arm[baseline][seed]["blocking"]
                )
                for seed in PILOT_SEEDS
            ]
            count_deltas = [
                by_arm["neural_opportunity"][seed]["blocked"]
                - by_arm[baseline][seed]["blocked"]
                for seed in PILOT_SEEDS
            ]
            comparisons[f"neural_vs_{baseline}"] = {
                "delta_pp": float(np.mean(deltas)),
                "ci95": _bootstrap_ci(
                    deltas, 20260820 + topology_index * 10 + comparison_index
                ),
                "wins_ties_losses": _wtl(count_deltas),
            }
        summaries[topology] = {"means": means, "comparisons": comparisons}
    result = {
        "protocol_id": PROTOCOL_ID,
        "frozen_model": True,
        "diagnostic_because_offline_gate_failed": True,
        "seeds": PILOT_SEEDS,
        "warmup": args.warmup,
        "eval_requests": args.eval_requests,
        "summaries": summaries,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "FROZEN_PILOT_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.output_dir / "per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
