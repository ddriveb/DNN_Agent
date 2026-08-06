"""Frozen 10-seed confirmation against the locked Phase-A fact table."""
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

from .protocol import CONFIRMATORY_SEEDS, PROTOCOL_ID, TOPOLOGY_CONFIG
from .run_closed_loop_smoke import _run


def _load_fact_table(path: Path):
    result = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            if row["arm"] not in ("ksp_ff_k50", "full_direct", "opp_only"):
                continue
            result[(row["topology"], int(row["seed"]), row["arm"])] = {
                "blocked": int(row["blocked"]),
                "blocking": float(row["blocking"]),
                "mean_ms": float(row["selector_mean_ms"]),
            }
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights-root", type=Path, required=True)
    parser.add_argument("--fact-table", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"):
        raise RuntimeError("native Direct Sketch kernel must be unset")
    facts = _load_fact_table(args.fact_table)
    short_names = {
        "xlron_nsfnet_deeprmsa": "nsfnet",
        "xlron_usnet_gcnrmsa": "usnet",
        "xlron_jpn48": "jpn48",
    }
    neural_rows = []
    for topology, config in TOPOLOGY_CONFIG.items():
        weights = (
            args.weights_root / short_names[topology] / "deployment_weights.npz"
        )
        for seed in CONFIRMATORY_SEEDS:
            ksp = _run(
                topology, config, "ksp_ff_k50", weights, 1000, 10000, seed=seed
            )
            expected = facts[(topology, seed, "ksp_ff_k50")]
            if ksp["blocked"] != expected["blocked"]:
                raise AssertionError(
                    f"KSP parity failed for {topology} seed {seed}: "
                    f"{ksp['blocked']} != {expected['blocked']}"
                )
            neural = _run(
                topology,
                config,
                "neural_opportunity",
                weights,
                1000,
                10000,
                seed=seed,
            )
            neural_rows.append({"topology": topology, "seed": seed, **neural})
    summaries = {}
    for topology_index, topology in enumerate(TOPOLOGY_CONFIG):
        by_seed = {
            row["seed"]: row
            for row in neural_rows
            if row["topology"] == topology
        }
        means = {
            "ksp_ff_k50": float(
                np.mean(
                    [facts[(topology, seed, "ksp_ff_k50")]["blocking"] for seed in CONFIRMATORY_SEEDS]
                )
            ),
            "full_direct": float(
                np.mean(
                    [facts[(topology, seed, "full_direct")]["blocking"] for seed in CONFIRMATORY_SEEDS]
                )
            ),
            "opportunity_teacher": float(
                np.mean(
                    [facts[(topology, seed, "opp_only")]["blocking"] for seed in CONFIRMATORY_SEEDS]
                )
            ),
            "neural_opportunity": float(
                np.mean([by_seed[seed]["blocking"] for seed in CONFIRMATORY_SEEDS])
            ),
        }
        comparisons = {}
        for comparison_index, (name, fact_arm) in enumerate(
            (
                ("ksp_ff_k50", "ksp_ff_k50"),
                ("full_direct", "full_direct"),
                ("opportunity_teacher", "opp_only"),
            )
        ):
            deltas = [
                100.0
                * (
                    by_seed[seed]["blocking"]
                    - facts[(topology, seed, fact_arm)]["blocking"]
                )
                for seed in CONFIRMATORY_SEEDS
            ]
            count_deltas = [
                by_seed[seed]["blocked"]
                - facts[(topology, seed, fact_arm)]["blocked"]
                for seed in CONFIRMATORY_SEEDS
            ]
            comparisons[f"neural_vs_{name}"] = {
                "delta_pp": float(np.mean(deltas)),
                "ci95": _bootstrap_ci(
                    deltas, 20260830 + topology_index * 10 + comparison_index
                ),
                "wins_ties_losses": _wtl(count_deltas),
            }
        full_gain = means["ksp_ff_k50"] - means["full_direct"]
        neural_gain = means["ksp_ff_k50"] - means["neural_opportunity"]
        retention = neural_gain / full_gain
        latency = float(
            np.mean([by_seed[seed]["mean_ms"] for seed in CONFIRMATORY_SEEDS])
        )
        gate = {
            "beats_ksp": comparisons["neural_vs_ksp_ff_k50"]["ci95"][1] < 0.0,
            "retention_ge_85pct": retention >= 0.85,
            "full_direct_noninferior_0p20pp": comparisons[
                "neural_vs_full_direct"
            ]["ci95"][1]
            <= 0.20,
            "latency_le_0p060ms": latency <= 0.060,
        }
        summaries[topology] = {
            "mean_blocking": means,
            "comparisons": comparisons,
            "full_direct_gain_pp": 100.0 * full_gain,
            "neural_gain_pp": 100.0 * neural_gain,
            "gain_retention": retention,
            "neural_mean_ms": latency,
            "gate": gate,
            "passes": all(gate.values()),
        }
    verdict = (
        "GO_NEURAL_OPPORTUNITY_DEPLOYMENT_VARIANT"
        if all(summary["passes"] for summary in summaries.values())
        else "HOLD_NEURAL_OPPORTUNITY"
    )
    result = {
        "protocol_id": PROTOCOL_ID,
        "model_frozen_before_confirmatory_seeds": True,
        "seeds": CONFIRMATORY_SEEDS,
        "fact_table": str(args.fact_table),
        "summaries": summaries,
        "verdict": verdict,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "CONFIRMATORY_RESULTS.json").write_text(
        json.dumps(result, indent=2, sort_keys=True), encoding="utf-8"
    )
    with (args.output_dir / "neural_per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=neural_rows[0].keys())
        writer.writeheader()
        writer.writerows(neural_rows)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
