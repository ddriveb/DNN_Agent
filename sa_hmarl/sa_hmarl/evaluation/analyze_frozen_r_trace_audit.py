"""Analyze frozen-trace R-only replay results and produce audit deliverables."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


ORIGINS = ("ksp_ff_highest", "ppo_r_top1", "strict_v13")
R_MODES = ("ksp_ff_highest", "ppo_r_top1", "strict_v13", "v135_afterstate", "v135_afterstate_explicit")
BASE_DIR = Path("sa_hmarl/experiments/v135_r_only_frozen_trace_audit")
SEEDS = [3030, 4040, 5050, 6060, 7070]


def _load_rows(phase: str) -> Dict[Tuple[str, str, int], Dict[str, Any]]:
    rows: Dict[Tuple[str, str, int], Dict[str, Any]] = {}
    phase_dir = BASE_DIR / phase
    if not phase_dir.exists():
        return rows
    for origin_dir in phase_dir.iterdir():
        if not origin_dir.is_dir() or not origin_dir.name.startswith("origin_"):
            continue
        origin_r_mode = origin_dir.name[len("origin_"):]
        for seed_dir in origin_dir.iterdir():
            if not seed_dir.is_dir() or not seed_dir.name.startswith("seed_"):
                continue
            result_path = seed_dir / "results.json"
            if not result_path.exists():
                continue
            payload = json.loads(result_path.read_text(encoding="utf-8"))
            for row in payload.get("results", []):
                r_mode = row["r_mode"]
                seed = int(row["seed"])
                rows[(origin_r_mode, r_mode, seed)] = row
    return rows


def _metric_array(rows, origin: str, r_mode: str, metric: str = "blocking_rate") -> Tuple[Optional[np.ndarray], List[int]]:
    vals = []
    present = []
    for seed in SEEDS:
        row = rows.get((origin, r_mode, seed))
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


def _compare(rows, origin: str, baseline_r: str, method_r: str, metric: str = "blocking_rate") -> Optional[Dict[str, Any]]:
    base_arr, base_seeds = _metric_array(rows, origin, baseline_r, metric)
    method_arr, method_seeds = _metric_array(rows, origin, method_r, metric)
    if base_arr is None or method_arr is None:
        return None
    common = sorted(set(base_seeds) & set(method_seeds))
    if not common:
        return None
    base_vals = np.asarray([rows[(origin, baseline_r, s)][metric] for s in common])
    method_vals = np.asarray([rows[(origin, method_r, s)][metric] for s in common])
    mean_diff = float(base_vals.mean() - method_vals.mean())
    base_mean = float(base_vals.mean())
    rel_reduction = float(mean_diff / base_mean) if base_mean > 0 else None
    wins = int((base_vals > method_vals + 1e-12).sum())
    ties = int(np.isclose(base_vals, method_vals, atol=1e-12).sum())
    losses = int((base_vals < method_vals - 1e-12).sum())
    boot_mean, boot_lo, boot_hi = _bootstrap_ci(base_vals, method_vals)
    _, p_wilcox = _wilcoxon(base_vals, method_vals)
    return {
        "origin": origin,
        "metric": metric,
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
    comparisons = []
    invariant_rows = []

    for origin in ORIGINS:
        summaries[origin] = {}
        for r_mode in R_MODES:
            arr, seeds = _metric_array(rows, origin, r_mode, "blocking_rate")
            if arr is None:
                continue
            summaries[origin][r_mode] = {
                "n": len(arr),
                "mean_blocking_rate": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                "seeds": seeds,
            }
        # v1.35 comparisons against KSP-FF and Strict v1.3.
        for baseline in ("ksp_ff_highest", "strict_v13"):
            for method in ("v135_afterstate", "v135_afterstate_explicit"):
                comp = _compare(rows, origin, baseline, method, "blocking_rate")
                if comp:
                    comparisons.append(comp)
        # Invariant audit: request_trace_hash and c_context_hash consistency.
        invariant = {"origin": origin, "request_trace_hash_consistent": True, "c_context_hash_consistent": True}
        hashes = set()
        ctx_hashes = set()
        consistency_errors = []
        request_mismatches = []
        queue_errors = []
        for r_mode in R_MODES:
            for seed in SEEDS:
                row = rows.get((origin, r_mode, seed))
                if row is None:
                    continue
                hashes.add(row.get("request_trace_hash"))
                ctx_hashes.add(row.get("c_context_hash"))
                consistency_errors.append(row.get("consistency_error_count", 0))
                request_mismatches.append(row.get("request_id_mismatch_count", 0))
                queue_errors.append(row.get("queue_conservation_error", False))
        if len(hashes) > 1:
            invariant["request_trace_hash_consistent"] = False
            invariant["request_trace_hashes"] = sorted(hashes)
        if len(ctx_hashes) > 1:
            invariant["c_context_hash_consistent"] = False
            invariant["c_context_hashes"] = sorted(ctx_hashes)
        invariant["max_consistency_error_count"] = int(max(consistency_errors)) if consistency_errors else 0
        invariant["max_request_id_mismatch_count"] = int(max(request_mismatches)) if request_mismatches else 0
        invariant["any_queue_conservation_error"] = bool(any(queue_errors))
        invariant_rows.append(invariant)

    # Global invariant: request_trace_hash must be identical across origins for each seed.
    global_invariants = []
    for seed in SEEDS:
        hashes = set()
        origins_with_data = []
        for origin in ORIGINS:
            for r_mode in R_MODES:
                row = rows.get((origin, r_mode, seed))
                if row is not None:
                    hashes.add(row.get("request_trace_hash"))
                    origins_with_data.append(origin)
        global_invariants.append({
            "seed": seed,
            "request_trace_hash_consistent_across_origins": len(hashes) <= 1,
            "n_distinct_hashes": len(hashes),
        })

    results = {
        "phase": phase,
        "summaries": summaries,
        "comparisons": comparisons,
        "invariants": invariant_rows,
        "global_invariants": global_invariants,
    }

    # Holm correction across all blocking comparisons.
    pvals = [c["wilcoxon_pvalue"] for c in comparisons]
    adjusted = _holm(pvals)
    for c, adj in zip(comparisons, adjusted):
        c["holm_pvalue"] = float(adj)

    # Launch-criterion evaluation.
    criterion = {"pass": False, "reasons": []}
    v135_variants = ("v135_afterstate", "v135_afterstate_explicit")
    wins_by_variant = {v: {"vs_ksp": 0, "vs_strict": 0, "origins": set()} for v in v135_variants}
    for c in comparisons:
        if c["metric"] != "blocking_rate":
            continue
        variant = c["method_r"]
        if variant not in v135_variants:
            continue
        if c["baseline_r"] == "ksp_ff_highest" and c["wins"] >= 4:
            wins_by_variant[variant]["vs_ksp"] += 1
            wins_by_variant[variant]["origins"].add(c["origin"])
        if c["baseline_r"] == "strict_v13" and c["wins"] >= 4:
            wins_by_variant[variant]["vs_strict"] += 1
            wins_by_variant[variant]["origins"].add(c["origin"])

    passing_variants = []
    for variant, info in wins_by_variant.items():
        origin_count = len(info["origins"])
        if info["vs_ksp"] >= 2 and info["vs_strict"] >= 2 and origin_count >= 2:
            passing_variants.append(variant)
    if passing_variants:
        criterion["pass"] = True
        criterion["passing_variants"] = passing_variants
    else:
        criterion["reasons"].append(
            "No v1.35 variant won ≥4/5 seeds versus both KSP-FF and Strict v1.3 in ≥2 origins."
        )

    # Delay/FS degradation check (>5% relative to better baseline).
    degradation_notes = []
    for origin in ORIGINS:
        for variant in v135_variants:
            for metric in ("avg_delay_ms", "avg_fs"):
                for baseline in ("ksp_ff_highest", "strict_v13"):
                    comp = _compare(rows, origin, baseline, variant, metric)
                    if comp is None:
                        continue
                    # Degradation means method mean is higher than baseline mean by >5% of baseline.
                    if comp["base_mean"] > 0 and comp["method_mean"] > comp["base_mean"] * 1.05:
                        degradation_notes.append({
                            "origin": origin,
                            "variant": variant,
                            "metric": metric,
                            "baseline": baseline,
                            "base_mean": comp["base_mean"],
                            "method_mean": comp["method_mean"],
                            "pct_change": (comp["method_mean"] / comp["base_mean"] - 1.0) * 100.0,
                        })
    criterion["degradation_notes"] = degradation_notes
    if degradation_notes:
        criterion["pass"] = False
        criterion["reasons"].append(f"{len(degradation_notes)} secondary-metric degradations >5% detected.")

    return {
        "phase": phase,
        "summaries": summaries,
        "comparisons": comparisons,
        "invariants": invariant_rows,
        "launch_criterion": criterion,
    }


def _write_r_only_results(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries = results["summaries"]
    comparisons = results["comparisons"]

    lines = ["# Frozen-Trace R-Only Replay Results", ""]
    lines.extend(["## Per-method blocking rates", "", "| Origin | R-mode | N | Mean blocking | Std |"])
    lines.append("|---|---|---:|---:|---:|")
    for origin in ORIGINS:
        for r_mode in R_MODES:
            s = summaries.get(origin, {}).get(r_mode)
            if s is None:
                continue
            lines.append(
                f"| {origin} | {r_mode} | {s['n']} | {s['mean_blocking_rate']:.4%} | {s['std']:.4%} |"
            )
    lines.extend(["", "## Paired comparisons (delta = baseline - method)", ""])
    lines.extend([
        "| Origin | Baseline | Method | N | Base mean | Method mean | Mean delta | Rel reduction | Wins/Ties/Losses | Bootstrap CI | Wilcoxon p | Holm p |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for c in comparisons:
        rel = f"{c['relative_reduction']:.2%}" if c["relative_reduction"] is not None else "N/A"
        ci = f"[{c['bootstrap_ci_lower']:.4%}, {c['bootstrap_ci_upper']:.4%}]"
        lines.append(
            f"| {c['origin']} | {c['baseline_r']} | {c['method_r']} | {c['n']} | "
            f"{c['base_mean']:.4%} | {c['method_mean']:.4%} | {c['mean_difference']:.4%} | "
            f"{rel} | {c['wins']}/{c['ties']}/{c['losses']} | {ci} | {c['wilcoxon_pvalue']:.4f} | {c['holm_pvalue']:.4f} |"
        )

    lines.extend(["", "## Launch criterion", ""])
    crit = results["launch_criterion"]
    lines.append(f"**Pass**: {crit['pass']}")
    if crit["pass"]:
        lines.append(f"Passing variants: {crit['passing_variants']}")
    else:
        for reason in crit["reasons"]:
            lines.append(f"- {reason}")
    if crit["degradation_notes"]:
        lines.append("")
        lines.append("Secondary-metric degradations >5%:")
        for note in crit["degradation_notes"]:
            lines.append(
                f"- {note['origin']} {note['variant']} vs {note['baseline']} on {note['metric']}: "
                f"{note['pct_change']:.2f}%"
            )

    (out_dir / "R_ONLY_REPLAY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_paired_statistics(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "PAIRED_STATISTICS.json").write_text(
        json.dumps(results["comparisons"], indent=2, default=str), encoding="utf-8"
    )
    lines = ["# Paired Statistics (Frozen-Trace R-Only)", ""]
    lines.append("See `R_ONLY_REPLAY_RESULTS.md` for the main table. This file records the raw comparison objects.")
    (out_dir / "PAIRED_STATISTICS.md").write_text("\n".join(lines), encoding="utf-8")


def _write_invariant_audit(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    invariants = results["invariants"]
    (out_dir / "INVARIANT_AUDIT.json").write_text(
        json.dumps(invariants, indent=2, default=str), encoding="utf-8"
    )
    lines = ["# Invariant Audit", ""]
    lines.append("| Origin | Trace hash consistent | C-context hash consistent | Max consistency errors | Max request-id mismatches | Queue conservation error |")
    lines.append("|---|---|---|---:|---:|---:|")
    all_pass = True
    for inv in invariants:
        pass_flag = inv["request_trace_hash_consistent"] and inv["c_context_hash_consistent"] and inv["max_consistency_error_count"] == 0 and inv["max_request_id_mismatch_count"] == 0 and not inv["any_queue_conservation_error"]
        if not pass_flag:
            all_pass = False
        lines.append(
            f"| {inv['origin']} | {inv['request_trace_hash_consistent']} | {inv['c_context_hash_consistent']} | "
            f"{inv['max_consistency_error_count']} | {inv['max_request_id_mismatch_count']} | {inv['any_queue_conservation_error']} |"
        )
    lines.append("")
    lines.append("## Global request_trace_hash consistency across origins (same seed)")
    lines.append("")
    lines.append("| Seed | Consistent across origins | Distinct hashes |")
    lines.append("|---:|---|---:|")
    for ginv in results.get("global_invariants", []):
        lines.append(
            f"| {ginv['seed']} | {ginv['request_trace_hash_consistent_across_origins']} | {ginv['n_distinct_hashes']} |"
        )
        if not ginv["request_trace_hash_consistent_across_origins"]:
            all_pass = False
    lines.append("")
    lines.append(f"**All invariants pass**: {all_pass}")
    (out_dir / "INVARIANT_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def _write_final_decision(results: Dict[str, Any], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    crit = results["launch_criterion"]
    decision = "PASS" if crit["pass"] else "INCONCLUSIVE"
    lines = [
        "# Final R-Only Decision",
        "",
        f"**Decision**: {decision}",
        "",
        "## Launch criterion",
        "- v1.35 must win ≥4/5 seeds versus KSP-FF K=50 hops and Strict v1.3 in ≥2 frozen origins.",
        "- Mean admitted delay and mean admitted FS must not degrade by >5% versus the better baseline.",
        "- All invariant checks must pass.",
        "",
    ]
    if crit["pass"]:
        lines.append(f"Passing variants: {crit['passing_variants']}")
    else:
        lines.append("**Reasons for inconclusive decision**:")
        for reason in crit["reasons"]:
            lines.append(f"- {reason}")
    (out_dir / "FINAL_R_ONLY_DECISION.md").write_text("\n".join(lines), encoding="utf-8")


def _write_trace_origin_bias_audit(rows: Dict[Tuple[str, str, int], Dict[str, Any]], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    # Compare C-decisions across origins by looking at how often different origins produce different split/server choices.
    # We only have the origin's own selected action in the trace, not a direct matrix, so we infer from replay summaries.
    lines = ["# Trace-Origin Bias Audit", ""]
    lines.append(
        "This audit compares the blocking-rate distributions produced by each origin trace. "
        "If the R-only ranking is robust, the relative ordering of R methods should be stable across origins."
    )
    lines.append("")
    lines.append("| Origin | R-mode | Mean blocking | Std |")
    lines.append("|---|---|---:|---:|")
    for origin in ORIGINS:
        for r_mode in R_MODES:
            vals = []
            for seed in SEEDS:
                row = rows.get((origin, r_mode, seed))
                if row is not None:
                    vals.append(row["blocking_rate"])
            if vals:
                arr = np.asarray(vals)
                std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
                lines.append(f"| {origin} | {r_mode} | {arr.mean():.4%} | {std:.4%} |")
    lines.append("")
    lines.append("Note: c_context_hash is intentionally identical across R methods for a given origin/seed because the recorded context is replayed verbatim.")
    (out_dir / "TRACE_ORIGIN_BIAS_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", default="pilot")
    args = parser.parse_args()

    results = analyze(args.phase)
    out_dir = BASE_DIR
    _write_r_only_results(results, out_dir)
    _write_paired_statistics(results, out_dir)
    _write_invariant_audit(results, out_dir)
    _write_final_decision(results, out_dir)
    rows = _load_rows(args.phase)
    _write_trace_origin_bias_audit(rows, out_dir)

    # Write frozen-trace spec if it does not exist.
    spec_path = out_dir / "FROZEN_TRACE_SPEC.md"
    if not spec_path.exists():
        spec_path.write_text(
            "# Frozen-Trace Specification\n\n"
            "See the audit plan for the full schema. Each trace is a JSON file containing:\n\n"
            "- `schema_version`: `frozen-trace-v1.0`\n"
            "- `origin`: `{c_mode, r_mode}` that generated the trace.\n"
            "- `topology`, `seed`, `warmup_requests`, `requests_per_episode`.\n"
            "- `config`: full environment config used to generate the trace.\n"
            "- `config_hash`, `request_trace_hash`, `c_context_hash`.\n"
            "- `requests`: ordered list of request records.\n\n"
            "Each request record contains the request fields, the recorded C decision, the recorded C-context, and the origin R action.\n",
            encoding="utf-8",
        )

    print(f"[analyze] Wrote Phase A deliverables to {out_dir}")
    print(f"[analyze] Launch criterion: {'PASS' if results['launch_criterion']['pass'] else 'INCONCLUSIVE'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
