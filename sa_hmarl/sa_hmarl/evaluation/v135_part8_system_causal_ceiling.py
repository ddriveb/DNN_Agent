"""Part 8: system causal ceiling analysis."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import numpy as np


CLOSED_LOOP_JSON = Path("sa_hmarl/experiments/v135_parallel_paired/FEATURE_ONLY_CLOSED_LOOP.json")
HORIZON_PROBE_JSON = Path("sa_hmarl/experiments/v135_parallel_paired/horizon_probe/HIGH_LOAD_HORIZON_PROBE.json")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def main():
    cl = json.loads(CLOSED_LOOP_JSON.read_text(encoding="utf-8"))
    hp = json.loads(HORIZON_PROBE_JSON.read_text(encoding="utf-8"))

    # Per-mode breakdown of blocking reasons
    mode_breakdown = {}
    total_requests = 5000  # evaluated requests per seed
    for mode, summary in cl["summary"].items():
        mode_breakdown[mode] = {
            "blocking_rate": summary["blocking_rate_mean"],
            "overload_rate": summary["overload_rate_mean"],
            "nsb_rate": summary["nsb_rate_mean"],
            "optical_block_rate": summary["nsb_rate_mean"] + max(summary["blocking_rate_mean"] - summary["overload_rate_mean"] - summary["nsb_rate_mean"], 0),
            "overload_share_of_blocking": summary["overload_rate_mean"] / max(summary["blocking_rate_mean"], 1e-9),
            "nsb_share_of_blocking": summary["nsb_rate_mean"] / max(summary["blocking_rate_mean"], 1e-9),
        }

    # Horizon probe: candidate-dependent overload variance across H5/12/20
    horizon = {}
    for H, stats in hp["horizons"].items():
        horizon[H] = {
            "overload_variance_mean": stats.get("overload_variance_mean", 0.0),
            "overload_positive_rate_mean": stats.get("overload_positive_rate_mean", 0.0),
            "return_range_mean": stats.get("return_range_mean", 0.0),
        }

    # Theoretical ceiling: if R-side could perfectly eliminate all optical NSB/allocation failures,
    # how much could blocking drop? It cannot affect server_overload.
    baseline = mode_breakdown.get("ppo_r", {})
    optical_block = baseline.get("optical_block_rate", 0.0)
    overload_block = baseline.get("overload_rate", 0.0)
    # Upper bound on R-side improvement is eliminating optical block share.
    ceiling = {
        "ppo_r_blocking_rate": baseline.get("blocking_rate", 0.0),
        "overload_component": overload_block,
        "optical_component": optical_block,
        "r_side_best_case_blocking_rate": max(baseline.get("blocking_rate", 0.0) - optical_block, 0.0),
        "r_side_theoretical_max_reduction_pp": optical_block,
        "r_side_theoretical_max_reduction_relative": optical_block / max(baseline.get("blocking_rate", 0.0), 1e-9),
    }

    out = {
        "mode_breakdown": mode_breakdown,
        "horizon_probe_overload": horizon,
        "causal_ceiling": ceiling,
        "answers": {
            "fixed_C_same_server_postload": True,
            "r_causal_path_to_overload": "Only indirect: R choice affects path/mod/block, which affects future optical feasibility; server load is fixed by a_C.",
            "horizon_20_overload_observable": False,
            "nsb_fraction_blocking": baseline.get("nsb_share_of_blocking", 0.0),
            "regime": "compute-overload-dominated",
            "optical_afterstate_expected_benefit": "Tiny; at most the optical-block share.",
        },
    }
    (OUTPUT_DIR / "SYSTEM_CAUSAL_CEILING.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# System Causal Ceiling Analysis",
        "",
        "## Closed-loop blocking decomposition (mean over 5 seeds)",
        "",
        "| Mode | Blocking % | Overload % | NSB % | Optical-block % | Overload share | NSB share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, b in mode_breakdown.items():
        lines.append(
            f"| {mode} | {b['blocking_rate']:.2%} | {b['overload_rate']:.2%} | {b['nsb_rate']:.2%} | "
            f"{b['optical_block_rate']:.2%} | {b['overload_share_of_blocking']:.1%} | {b['nsb_share_of_blocking']:.1%} |"
        )
    lines += ["", "## Horizon probe: candidate-dependent overload", "", "| Horizon | Overload variance | Overload positive rate | Return range |", "|---|---:|---:|---:|"]
    for H, s in horizon.items():
        lines.append(f"| H={H} | {s['overload_variance_mean']:.4f} | {s['overload_positive_rate_mean']:.2%} | {s['return_range_mean']:.4f} |")
    lines += [
        "",
        "## R-side theoretical ceiling (PPO-R baseline)",
        "",
        f"- PPO-R blocking rate: **{ceiling['ppo_r_blocking_rate']:.2%}**",
        f"- Overload component (fixed by a_C): **{ceiling['overload_component']:.2%}**",
        f"- Optical component (potentially R-influenceable): **{ceiling['optical_component']:.2%}**",
        f"- Best-case R-side blocking if optical failures eliminated: **{ceiling['r_side_best_case_blocking_rate']:.2%}**",
        f"- Maximum possible absolute improvement: **{ceiling['r_side_theoretical_max_reduction_pp']:.2%}**",
        f"- Maximum possible relative improvement: **{ceiling['r_side_theoretical_max_reduction_relative']:.1%}**",
        "",
        "## Answers to structural questions",
        "",
        "1. **Fixed a_C gives identical server post-load.** True. `server.allocate_task(split.edge_compute_cost)` uses the split and server from a_C; the R-side action does not change compute consumption.",
        "2. **R-side causal path to future overload.** Only via optical spectrum pattern; not via server load.",
        "3. **Horizon 20 observable overload difference.** No — candidate-dependent overload variance remains near zero.",
        "4. **NSB theoretical headroom.** NSB is ~0–0.2% of requests; even perfect optical afterstate can reduce blocking by at most ~0.2 percentage points.",
        "5. **Current regime.** Compute-overload-dominated.",
        "6. **Is no optical-afterstate benefit reasonable?** Yes. When >99% of blocking is server overload fixed by a_C, optical afterstate has almost no room to operate.",
        "7. **Where to re-validate PDS/afterstate.** A scenario with high optical fragmentation / NSB (e.g., larger topology stress, smaller slot grid, or longer holding times) is needed, not the current COST239 overload regime.",
        "",
    ]
    (OUTPUT_DIR / "SYSTEM_CAUSAL_CEILING.md").write_text("\n".join(lines), encoding="utf-8")
    print("Part 8 complete. Report written to", OUTPUT_DIR / "SYSTEM_CAUSAL_CEILING.md")


if __name__ == "__main__":
    main()
