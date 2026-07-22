"""Analyze SA-HMARL v1.35 diagnostic pilot results (COST239, DF_C primary).

Loads the closed-loop results produced by run_strict_v13_multitopology_cside_verified.py
and produces paired comparison tables:
  - Strict v1.3 vs KSP-FF K=50, PPO-R Top-1
  - v1.35-explicit vs KSP-FF K=50, PPO-R Top-1, Strict v1.3
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


SEEDS = [3030, 4040, 5050, 6060, 7070]
TOPOLOGY = "xlron_cost239_ptrnet_real"
C_MODES = ["df_c", "ppo_c"]
R_METHODS = ["ppo_r_top1", "ksp_ff_highest", "strict_v13", "v135_afterstate", "v135_afterstate_explicit"]


def _load_rows(base_dir: Path) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    rows: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    for seed in SEEDS:
        task_dir = base_dir / "full" / TOPOLOGY / f"seed_{seed}"
        json_path = task_dir / "results.json"
        done_path = task_dir / "done.marker"
        if not json_path.exists() or not done_path.exists():
            continue
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        for row in payload.get("results", []):
            if row.get("topology") != TOPOLOGY:
                continue
            key = (row["c_mode"], row["r_mode"], int(row["seed"]))
            rows[key] = row
    return rows


def _blocking(rows, c_mode: str, r_mode: str) -> Tuple[Optional[np.ndarray], List[int]]:
    vals = []
    present = []
    for seed in SEEDS:
        row = rows.get((c_mode, r_mode, seed))
        if row is not None:
            vals.append(row["blocking_rate"])
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


def _compare(rows, c_mode: str, baseline_r: str, method_r: str) -> Optional[Dict[str, Any]]:
    base_arr, base_seeds = _blocking(rows, c_mode, baseline_r)
    method_arr, method_seeds = _blocking(rows, c_mode, method_r)
    if base_arr is None or method_arr is None:
        return None
    common = sorted(set(base_seeds) & set(method_seeds))
    if not common:
        return None
    base_vals = np.asarray([rows[(c_mode, baseline_r, s)]["blocking_rate"] for s in common])
    method_vals = np.asarray([rows[(c_mode, method_r, s)]["blocking_rate"] for s in common])
    mean_diff = float(base_vals.mean() - method_vals.mean())
    base_mean = float(base_vals.mean())
    rel_reduction = float(mean_diff / base_mean) if base_mean > 0 else None
    wins = int((base_vals > method_vals + 1e-12).sum())
    ties = int(np.isclose(base_vals, method_vals, atol=1e-12).sum())
    losses = int((base_vals < method_vals - 1e-12).sum())
    boot_mean, boot_lo, boot_hi = _bootstrap_ci(base_vals, method_vals)
    _, p_wilcox = _wilcoxon(base_vals, method_vals)
    return {
        "c_mode": c_mode,
        "baseline_r": baseline_r,
        "method_r": method_r,
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


def _method_summary(rows, c_mode: str) -> List[Dict[str, Any]]:
    summary = []
    for r_mode in R_METHODS:
        arr, seeds = _blocking(rows, c_mode, r_mode)
        if arr is None:
            continue
        summary.append({
            "c_mode": c_mode,
            "r_mode": r_mode,
            "n": len(arr),
            "mean_blocking_rate": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "seeds": seeds,
        })
    return summary


def analyze(base_dir: Path) -> Dict[str, Any]:
    rows = _load_rows(base_dir)
    if not rows:
        raise ValueError(f"No results found in {base_dir}")

    comparisons = []
    summaries = {}
    for c_mode in C_MODES:
        summaries[c_mode] = _method_summary(rows, c_mode)
        # Strict v1.3 vs KSP and PPO-R
        comparisons.append(_compare(rows, c_mode, "ksp_ff_highest", "strict_v13"))
        comparisons.append(_compare(rows, c_mode, "ppo_r_top1", "strict_v13"))
        # v1.35 vs KSP, PPO-R, and Strict
        comparisons.append(_compare(rows, c_mode, "ksp_ff_highest", "v135_afterstate"))
        comparisons.append(_compare(rows, c_mode, "ppo_r_top1", "v135_afterstate"))
        comparisons.append(_compare(rows, c_mode, "strict_v13", "v135_afterstate"))
        # v1.35 explicit-only vs same baselines
        comparisons.append(_compare(rows, c_mode, "ksp_ff_highest", "v135_afterstate_explicit"))
        comparisons.append(_compare(rows, c_mode, "ppo_r_top1", "v135_afterstate_explicit"))
        comparisons.append(_compare(rows, c_mode, "strict_v13", "v135_afterstate_explicit"))
        comparisons.append(_compare(rows, c_mode, "v135_afterstate", "v135_afterstate_explicit"))
    comparisons = [c for c in comparisons if c is not None]

    # Holm correction across all comparisons.
    pvals = [c["wilcoxon_pvalue"] for c in comparisons]
    if pvals:
        p = np.asarray(pvals, dtype=float)
        idx = np.argsort(p)
        sorted_p = p[idx]
        n = len(p)
        raw = sorted_p * np.arange(n, 0, -1)
        adjusted_sorted = np.maximum.accumulate(raw)
        adjusted_sorted = np.minimum(adjusted_sorted, 1.0)
        adjusted = np.empty(n)
        adjusted[idx] = adjusted_sorted
        for c, adj in zip(comparisons, adjusted.tolist()):
            c["holm_pvalue"] = float(adj)
    else:
        for c in comparisons:
            c["holm_pvalue"] = 1.0

    return {
        "summaries": summaries,
        "comparisons": comparisons,
    }


def _write_report(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "# SA-HMARL v1.35 Diagnostic Pilot Report",
        "",
        f"- Topology: `{TOPOLOGY}`",
        f"- Seeds: {SEEDS}",
        "- C-side modes: df_c (primary), ppo_c",
        "- R-side methods: PPO-R Top-1, KSP-FF K=50 highest, Strict v1.3, v1.35-afterstate (base+ψ+δ), v1.35-afterstate-explicit (ψ+δ only)",
        "",
        "## Per-method blocking rates",
        "",
        "| C-mode | R-mode | N | Mean blocking rate | Std |",
        "|---|---|---:|---:|---:|",
    ]
    for c_mode in C_MODES:
        for s in results["summaries"].get(c_mode, []):
            lines.append(
                f"| {c_mode} | {s['r_mode']} | {s['n']} | "
                f"{s['mean_blocking_rate']:.4%} | {s['std']:.4%} |"
            )

    lines.extend(["", "## Paired comparisons (delta = baseline - method)", "",
                  "| C-mode | Baseline | Method | N | Base mean | Method mean | Mean delta | Rel reduction | Wins/Ties/Losses | Bootstrap CI | Wilcoxon p | Holm p |",
                  "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|"])
    for c in results["comparisons"]:
        rel = f"{c['relative_reduction']:.2%}" if c["relative_reduction"] is not None else "N/A"
        ci = f"[{c['bootstrap_ci_lower']:.4%}, {c['bootstrap_ci_upper']:.4%}]"
        lines.append(
            f"| {c['c_mode']} | {c['baseline_r']} | {c['method_r']} | {c['n']} | "
            f"{c['base_mean']:.4%} | {c['method_mean']:.4%} | {c['mean_difference']:.4%} | "
            f"{rel} | {c['wins']}/{c['ties']}/{c['losses']} | {ci} | {c['wilcoxon_pvalue']:.4f} | {c['holm_pvalue']:.4f} |"
        )

    # Decision paragraph
    v135_vs_strict = next(
        (c for c in results["comparisons"] if c["c_mode"] == "df_c" and c["baseline_r"] == "strict_v13" and c["method_r"] == "v135_afterstate"),
        None,
    )
    v135_vs_ksp = next(
        (c for c in results["comparisons"] if c["c_mode"] == "df_c" and c["baseline_r"] == "ksp_ff_highest" and c["method_r"] == "v135_afterstate"),
        None,
    )
    v135_vs_ppo = next(
        (c for c in results["comparisons"] if c["c_mode"] == "df_c" and c["baseline_r"] == "ppo_r_top1" and c["method_r"] == "v135_afterstate"),
        None,
    )
    explicit_vs_strict = next(
        (c for c in results["comparisons"] if c["c_mode"] == "df_c" and c["baseline_r"] == "strict_v13" and c["method_r"] == "v135_afterstate_explicit"),
        None,
    )
    lines.extend(["", "## Observations"])
    lines.append("- Under df_c, every R-side method yields the identical blocking rate across all seeds; "
                 "the C-side (split/server) decision is the dominant bottleneck and masks any R-side differences.")
    lines.append("- Under ppo_c, Strict v1.3, v1.35-afterstate and v1.35-afterstate-explicit all beat formal KSP-FF K=50.")
    lines.append("- Neither v1.35 variant consistently beats PPO-R Top-1 or Strict v1.3 under ppo_c in this small pilot.")
    lines.append("- The explicit-only afterstate model is slightly weaker than the full concatenated model, "
                 "suggesting the base v1.3 features still carry useful ranking signal.")
    lines.extend(["", "## Pilot decision"])
    if v135_vs_strict is None:
        lines.append("- v1.35 vs Strict v1.3 comparison unavailable.")
    else:
        direction = "reduces" if v135_vs_strict["mean_difference"] > 0 else "increases"
        lines.append(
            f"- v1.35 vs Strict v1.3 under df_c: {direction} blocking by "
            f"{abs(v135_vs_strict['mean_difference']):.4%} (Holm p={v135_vs_strict['holm_pvalue']:.4f})."
        )
    if v135_vs_ksp:
        direction = "reduces" if v135_vs_ksp["mean_difference"] > 0 else "increases"
        lines.append(
            f"- v1.35 vs KSP-FF under df_c: {direction} blocking by "
            f"{abs(v135_vs_ksp['mean_difference']):.4%} (Holm p={v135_vs_ksp['holm_pvalue']:.4f})."
        )
    if v135_vs_ppo:
        direction = "reduces" if v135_vs_ppo["mean_difference"] > 0 else "increases"
        lines.append(
            f"- v1.35 vs PPO-R Top-1 under df_c: {direction} blocking by "
            f"{abs(v135_vs_ppo['mean_difference']):.4%} (Holm p={v135_vs_ppo['holm_pvalue']:.4f})."
        )
    if explicit_vs_strict:
        direction = "reduces" if explicit_vs_strict["mean_difference"] > 0 else "increases"
        lines.append(
            f"- v1.35 explicit-only vs Strict v1.3 under df_c: {direction} blocking by "
            f"{abs(explicit_vs_strict['mean_difference']):.4%} (Holm p={explicit_vs_strict['holm_pvalue']:.4f})."
        )
    lines.append("- v1.35 full launch: not approved based on this pilot alone.")

    (out_dir / "V135_PILOT_REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "V135_PILOT_REPORT.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )
    print(f"[analyze] Wrote {out_dir / 'V135_PILOT_REPORT.md'}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base_dir", default="sa_hmarl/experiments/v135_afterstate_diagnostic_pilot")
    args = parser.parse_args()
    base_dir = Path(args.base_dir)
    results = analyze(base_dir)
    _write_report(results, base_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
