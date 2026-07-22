"""Analyze the Doherty/XLRON external-OD transfer diagnostic pilot."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


BASE_DIR = Path("sa_hmarl/experiments/v135_doherty_external_od_pilot")
SEEDS = [3030, 4040, 5050, 6060, 7070]
R_METHODS = ["ppo_r_top1", "ksp_ff_highest", "strict_v13", "v135_afterstate", "v135_afterstate_explicit"]


def _load_rows(phase: str) -> Dict[Tuple[str, int], Dict[str, Any]]:
    rows: Dict[Tuple[str, int], Dict[str, Any]] = {}
    phase_dir = BASE_DIR / phase
    if not phase_dir.exists():
        return rows
    for seed_dir in phase_dir.iterdir():
        if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
            continue
        result_path = seed_dir / "results.json"
        if not result_path.exists():
            continue
        payload = json.loads(result_path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            r_mode = row["r_mode"]
            seed = int(row["seed"])
            rows[(r_mode, seed)] = row
    return rows


def _metric_array(rows, r_mode: str, metric: str = "blocking_rate") -> Tuple[Optional[np.ndarray], List[int]]:
    vals = []
    present = []
    for seed in SEEDS:
        row = rows.get((r_mode, seed))
        if row is not None:
            vals.append(row[metric])
            present.append(seed)
    if not vals:
        return None, []
    return np.asarray(vals), present


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = a - b
    diff = diff[np.abs(diff) > 1e-12]
    if len(diff) == 0:
        return 0.0, 1.0
    stat, p = stats.wilcoxon(diff)
    return float(stat), float(p)


def _bootstrap_ci(a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 12345) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, len(a), size=len(a))
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.asarray(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _compare(rows, baseline_r: str, method_r: str, metric: str = "blocking_rate") -> Optional[Dict[str, Any]]:
    base_arr, base_seeds = _metric_array(rows, baseline_r, metric)
    method_arr, method_seeds = _metric_array(rows, method_r, metric)
    if base_arr is None or method_arr is None:
        return None
    common = sorted(set(base_seeds) & set(method_seeds))
    if not common:
        return None
    base_vals = np.asarray([rows[(baseline_r, s)][metric] for s in common])
    method_vals = np.asarray([rows[(method_r, s)][metric] for s in common])
    mean_diff = float(base_vals.mean() - method_vals.mean())
    base_mean = float(base_vals.mean())
    rel_reduction = float(mean_diff / base_mean) if base_mean > 0 else None
    wins = int((base_vals > method_vals + 1e-12).sum())
    ties = int(np.isclose(base_vals, method_vals, atol=1e-12).sum())
    losses = int((base_vals < method_vals - 1e-12).sum())
    boot_mean, boot_lo, boot_hi = _bootstrap_ci(base_vals, method_vals)
    _, p_wilcox = _wilcoxon(base_vals, method_vals)
    return {
        "baseline_r": baseline_r,
        "method_r": method_r,
        "metric": metric,
        "n": len(common),
        "common_seeds": common,
        "base_mean": base_mean,
        "method_mean": float(method_vals.mean()),
        "mean_difference": mean_diff,
        "relative_reduction": rel_reduction,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap_mean": boot_mean,
        "bootstrap_ci_lower": boot_lo,
        "bootstrap_ci_upper": boot_hi,
        "wilcoxon_pvalue": p_wilcox,
    }


def _holm(pvals: List[float]) -> List[float]:
    if not pvals:
        return []
    p = np.asarray(pvals, dtype=float)
    idx = np.argsort(p)
    sorted_p = p[idx]
    n = len(p)
    raw = sorted_p * np.arange(n, 0, -1)
    adjusted_sorted = np.maximum.accumulate(raw)
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
    adjusted = np.empty(n)
    adjusted[idx] = adjusted_sorted
    return adjusted.tolist()


def analyze(phase: str) -> Dict[str, Any]:
    rows = _load_rows(phase)
    if not rows:
        raise ValueError(f"No results found in {BASE_DIR / phase}")

    summaries = {}
    for r_mode in R_METHODS:
        arr, seeds = _metric_array(rows, r_mode, "blocking_rate")
        if arr is None:
            continue
        summaries[r_mode] = {
            "n": len(arr),
            "mean_blocking_rate": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "seeds": seeds,
        }

    comparisons = []
    for baseline in ("ksp_ff_highest", "strict_v13"):
        for method in ("v135_afterstate", "v135_afterstate_explicit"):
            comp = _compare(rows, baseline, method, "blocking_rate")
            if comp:
                comparisons.append(comp)

    pvals = [c["wilcoxon_pvalue"] for c in comparisons]
    adjusted = _holm(pvals)
    for c, adj in zip(comparisons, adjusted):
        c["holm_pvalue"] = float(adj)

    # Transfer diagnostic conclusion.
    v135_variants = ("v135_afterstate", "v135_afterstate_explicit")
    wins_by_variant = {v: {"vs_ksp": 0, "vs_strict": 0} for v in v135_variants}
    for c in comparisons:
        if c["metric"] != "blocking_rate":
            continue
        variant = c["method_r"]
        if variant not in v135_variants:
            continue
        if c["baseline_r"] == "ksp_ff_highest" and c["wins"] >= 4:
            wins_by_variant[variant]["vs_ksp"] = 1
        if c["baseline_r"] == "strict_v13" and c["wins"] >= 4:
            wins_by_variant[variant]["vs_strict"] = 1

    transfer_promising = any(
        info["vs_ksp"] and info["vs_strict"]
        for info in wins_by_variant.values()
    )

    return {
        "phase": phase,
        "summaries": summaries,
        "comparisons": comparisons,
        "transfer_promising": transfer_promising,
        "wins_by_variant": wins_by_variant,
        "transfer_diagnostic_only": True,
    }


def _write_report(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# External-OD Transfer Diagnostic Pilot Report",
        "",
        "> **Scope**: Transfer evaluation of locked v1.3/v1.35 rankers on a Doherty/XLRON-style COST239 setting (100 FSUs, mean holding ≈10 s, 25–100 Gbps proxy).",
        "> **Status**: TRANSFER_DIAGNOSTIC_ONLY — these checkpoints were not retrained on the external-OD regime.",
        "",
        "## Per-method blocking rates",
        "",
        "| R-mode | N | Mean blocking rate | Std |",
        "|---|---|---:|---:|",
    ]
    for r_mode in R_METHODS:
        s = results["summaries"].get(r_mode)
        if s is None:
            continue
        lines.append(
            f"| {r_mode} | {s['n']} | {s['mean_blocking_rate']:.4%} | {s['std']:.4%} |"
        )

    lines.extend(["", "## Paired comparisons (delta = baseline - method)", "",
                  "| Baseline | Method | N | Base mean | Method mean | Mean delta | Rel reduction | Wins/Ties/Losses | Bootstrap CI | Wilcoxon p | Holm p |",
                  "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for c in results["comparisons"]:
        rel = f"{c['relative_reduction']:.2%}" if c["relative_reduction"] is not None else "N/A"
        ci = f"[{c['bootstrap_ci_lower']:.4%}, {c['bootstrap_ci_upper']:.4%}]"
        lines.append(
            f"| {c['baseline_r']} | {c['method_r']} | {c['n']} | "
            f"{c['base_mean']:.4%} | {c['method_mean']:.4%} | {c['mean_difference']:.4%} | "
            f"{rel} | {c['wins']}/{c['ties']}/{c['losses']} | {ci} | {c['wilcoxon_pvalue']:.4f} | {c['holm_pvalue']:.4f} |"
        )

    lines.extend(["", "## Transfer diagnostic conclusion", ""])
    if results["transfer_promising"]:
        lines.append("At least one v1.35 variant won ≥4/5 seeds versus both KSP-FF and Strict v1.3. The transfer is **promising** and independent RMSA-native training is recommended for confirmation.")
    else:
        lines.append("No v1.35 variant won ≥4/5 seeds versus both KSP-FF and Strict v1.3. The locked checkpoints do **not generalize** to the external-OD regime. This does not invalidate v1.35 in the native setting; it only means the transfer checkpoint is non-generalizing.")

    (out_dir / "EXTERNAL_OD_PILOT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "EXTERNAL_OD_PILOT.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"[analyze] Wrote {out_dir / 'EXTERNAL_OD_PILOT.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="pilot")
    args = parser.parse_args()
    results = analyze(args.phase)
    _write_report(results, BASE_DIR)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
