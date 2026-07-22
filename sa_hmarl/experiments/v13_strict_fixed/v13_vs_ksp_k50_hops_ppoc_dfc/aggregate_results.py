"""Aggregate per-seed formal results and compute paired statistics.

Reads:
  formal/seed_*/seed_*.json

Writes:
  RAW_RESULTS.json
  PER_SEED_RESULTS.csv
  PPOC_COMPARISON.json / .md
  DFC_COMPARISON.json / .md
  C_SIDE_INTERACTION.json / .md
  PAIRED_STATISTICS.json / .md
  ACTION_DISTRIBUTION_AUDIT.json / .md
  FINAL_DECISION.json / .md
  STATUS.md
  MANIFEST.json
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


try:
    from scipy import stats
except Exception as exc:  # pragma: no cover
    raise ImportError("scipy is required for Wilcoxon and permutation tests") from exc


METHOD_RENAME = {
    "ppo_c+ppo_r_top1": "PPO-C + PPO-R Top-1",
    "ppo_c+ksp_ff_highest": "PPO-C + KSP-FF K=50 hops",
    "ppo_c+strict_v13": "PPO-C + Strict v1.3",
    "ppo_c+old_v13": "PPO-C + Legacy E1-trained gated baseline",
    "df_c+ppo_r_top1": "DF_C + PPO-R Top-1",
    "df_c+ksp_ff_highest": "DF_C + KSP-FF K=50 hops",
    "df_c+strict_v13": "DF_C + Strict v1.3",
    "df_c+old_v13": "DF_C + Legacy E1-trained gated baseline",
}

MAIN_METHODS = [
    "ppo_c+ppo_r_top1",
    "ppo_c+ksp_ff_highest",
    "ppo_c+strict_v13",
    "df_c+ppo_r_top1",
    "df_c+ksp_ff_highest",
    "df_c+strict_v13",
]


def _percent(x: float) -> str:
    return f"{x*100:.4f}%"


def _pp(x: float) -> str:
    return f"{x*100:.2f}pp"


def _bootstrap_paired_ci(diff: np.ndarray, n_resamples: int = 100000, ci: float = 0.95) -> Tuple[float, float]:
    rng = np.random.RandomState(12345)
    n = len(diff)
    boot = rng.choice(diff, size=(n_resamples, n), replace=True)
    means = boot.mean(axis=1)
    alpha = 1 - ci
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def _sign_flip_permutation_p(diff: np.ndarray, n_iters: int = 1_000_000, seed: int = 54321) -> float:
    """Monte Carlo sign-flip permutation p-value for H0: median=0 (two-sided)."""
    rng = np.random.RandomState(seed)
    n = len(diff)
    obs_stat = float(np.mean(diff))
    count = 0
    for _ in range(n_iters):
        signs = rng.choice([-1.0, 1.0], size=n)
        perm_stat = float(np.mean(diff * signs))
        if abs(perm_stat) >= abs(obs_stat):
            count += 1
    return count / n_iters


def _paired_stats(a: np.ndarray, b: np.ndarray) -> Dict[str, Any]:
    """a and b must be aligned by seed; returns stats for a - b."""
    diff = a - b
    n = len(diff)
    mean_diff = float(np.mean(diff))
    median_diff = float(np.median(diff))
    ci_lo, ci_hi = _bootstrap_paired_ci(diff)
    try:
        wilcoxon = stats.wilcoxon(diff, alternative="two-sided")
        wilcoxon_p = float(wilcoxon.pvalue)
    except Exception:
        wilcoxon_p = float("nan")
    sign_flip_p = _sign_flip_permutation_p(diff)
    wins = int(np.sum(diff < 0))
    losses = int(np.sum(diff > 0))
    ties = int(np.sum(diff == 0))
    return {
        "n": n,
        "mean_diff": mean_diff,
        "median_diff": median_diff,
        "mean_diff_pp": _pp(mean_diff),
        "bootstrap_95_ci": [float(ci_lo), float(ci_hi)],
        "bootstrap_95_ci_pp": [f"{ci_lo*100:.4f}%", f"{ci_hi*100:.4f}%"],
        "wilcoxon_p": wilcoxon_p,
        "sign_flip_permutation_p": sign_flip_p,
        "wins_a_lower": wins,
        "losses_a_lower": losses,
        "ties": ties,
        "relative_reduction": float(mean_diff / np.mean(b)) if np.mean(b) != 0 else float("nan"),
    }


def _holm_correction(pvalues: List[float], alpha: float = 0.05) -> List[float]:
    """Return Holm-adjusted p-values."""
    n = len(pvalues)
    order = np.argsort(pvalues)
    sorted_p = np.array(pvalues)[order]
    adjusted = np.minimum.accumulate(sorted_p * (n - np.arange(n)))  # simple step-up
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty(n)
    out[order] = adjusted
    return [float(v) for v in out]


def load_results(formal_dir: Path) -> Tuple[Dict[str, List[Dict[str, Any]]], List[int]]:
    """Load all per-seed JSON files."""
    by_method: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    seeds: List[int] = []
    for seed_dir in sorted(formal_dir.glob("seed_*")):
        json_file = seed_dir / f"seed_{seed_dir.name.split('_')[1]}.json"
        if not json_file.exists():
            continue
        data = json.loads(json_file.read_text(encoding="utf-8"))
        seed = data["config"]["seed"]
        seeds.append(seed)
        for row in data.get("results", []):
            by_method[row["method_name"]].append(row)
    return dict(by_method), seeds


def build_raw_results(by_method: Dict[str, List[Dict[str, Any]]], seeds: List[int]) -> Dict[str, Any]:
    raw: Dict[str, Any] = {"seeds": seeds, "methods": {}}
    for method, rows in by_method.items():
        rows_sorted = sorted(rows, key=lambda r: r["seed"])
        raw["methods"][method] = {
            "display_name": METHOD_RENAME.get(method, method),
            "per_seed": rows_sorted,
        }
    return raw


def build_per_seed_csv(by_method: Dict[str, List[Dict[str, Any]]], seeds: List[int], out_path: Path) -> None:
    fieldnames = [
        "seed", "c_mode", "r_mode", "method_name", "display_name",
        "total", "admitted", "blocked", "blocking_rate",
        "server_overload", "overload_rate", "no_suitable_block", "nsb_rate",
        "other_failure", "other_rate", "avg_delay_ms", "delay_p95_ms",
        "avg_fs", "avg_decision_ms", "decision_p95_ms",
        "avg_path_length_km", "avg_hop_count", "ppo_r_agreement",
        "fallback_rate", "ranker_top1_in_candidates_rate",
    ]
    rows = []
    for method in MAIN_METHODS + [m for m in by_method if m not in MAIN_METHODS]:
        for r in sorted(by_method.get(method, []), key=lambda x: x["seed"]):
            rows.append({
                "seed": r["seed"],
                "c_mode": r["c_mode"],
                "r_mode": r["r_mode"],
                "method_name": method,
                "display_name": METHOD_RENAME.get(method, method),
                "total": r["total"],
                "admitted": r["admitted"],
                "blocked": r["blocked"],
                "blocking_rate": r["blocking_rate"],
                "server_overload": r.get("server_overload", 0),
                "overload_rate": r.get("overload_rate", 0.0),
                "no_suitable_block": r.get("no_suitable_block", 0),
                "nsb_rate": r.get("nsb_rate", 0.0),
                "other_failure": r.get("other_failure", 0),
                "other_rate": r.get("other_rate", 0.0),
                "avg_delay_ms": r.get("avg_delay_ms", 0.0),
                "delay_p95_ms": r.get("delay_p95_ms", 0.0),
                "avg_fs": r.get("avg_fs", 0.0),
                "avg_decision_ms": r.get("avg_decision_ms", 0.0),
                "decision_p95_ms": r.get("decision_p95_ms", 0.0),
                "avg_path_length_km": r.get("avg_path_length_km", 0.0),
                "avg_hop_count": r.get("avg_hop_count", 0.0),
                "ppo_r_agreement": r.get("ppo_agreement_rate", 0.0),
                "fallback_rate": r.get("fallback_rate", 0.0),
                "ranker_top1_in_candidates_rate": r.get("ranker_top1_in_candidates_rate", 0.0),
            })
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _blocking_array(by_method: Dict[str, List[Dict[str, Any]]], method: str, seeds: List[int]) -> np.ndarray:
    rows = {r["seed"]: r for r in by_method.get(method, [])}
    return np.array([rows[s]["blocking_rate"] for s in seeds])


def _metric_array(by_method: Dict[str, List[Dict[str, Any]]], method: str, seeds: List[int], key: str) -> np.ndarray:
    rows = {r["seed"]: r for r in by_method.get(method, [])}
    return np.array([rows[s].get(key, 0.0) for s in seeds])


def summarize_method(by_method: Dict[str, List[Dict[str, Any]]], method: str, seeds: List[int]) -> Dict[str, Any]:
    arr = _blocking_array(by_method, method, seeds)
    return {
        "display_name": METHOD_RENAME.get(method, method),
        "n": len(arr),
        "blocking_rate_mean": float(np.mean(arr)),
        "blocking_rate_std": float(np.std(arr, ddof=1)),
        "blocking_rate_median": float(np.median(arr)),
        "blocking_rate_p25": float(np.percentile(arr, 25)),
        "blocking_rate_p75": float(np.percentile(arr, 75)),
        "blocking_rate_min": float(np.min(arr)),
        "blocking_rate_max": float(np.max(arr)),
        "overload_rate_mean": float(np.mean(_metric_array(by_method, method, seeds, "overload_rate"))),
        "nsb_rate_mean": float(np.mean(_metric_array(by_method, method, seeds, "nsb_rate"))),
        "avg_delay_ms_mean": float(np.mean(_metric_array(by_method, method, seeds, "avg_delay_ms"))),
        "avg_fs_mean": float(np.mean(_metric_array(by_method, method, seeds, "avg_fs"))),
        "avg_decision_ms_mean": float(np.mean(_metric_array(by_method, method, seeds, "avg_decision_ms"))),
        "ppo_r_agreement_mean": float(np.mean(_metric_array(by_method, method, seeds, "ppo_agreement_rate"))),
    }


def build_c_comparison(
    by_method: Dict[str, List[Dict[str, Any]]],
    seeds: List[int],
    c_prefix: str,
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    strict = f"{c_prefix}+strict_v13"
    ksp = f"{c_prefix}+ksp_ff_highest"
    ppor = f"{c_prefix}+ppo_r_top1"
    legacy = f"{c_prefix}+old_v13"

    summaries = {
        "strict_v1.3": summarize_method(by_method, strict, seeds),
        "ksp_ff_k50_hops": summarize_method(by_method, ksp, seeds),
        "ppo_r_top1": summarize_method(by_method, ppor, seeds),
    }
    if legacy in by_method:
        summaries["legacy_e1"] = summarize_method(by_method, legacy, seeds)

    strict_arr = _blocking_array(by_method, strict, seeds)
    ksp_arr = _blocking_array(by_method, ksp, seeds)
    ppor_arr = _blocking_array(by_method, ppor, seeds)

    cmp = {
        "c_side": c_prefix.upper(),
        "summaries": summaries,
        "strict_vs_ksp": _paired_stats(strict_arr, ksp_arr),
        "strict_vs_ppo_r": _paired_stats(strict_arr, ppor_arr),
        "ksp_vs_ppo_r": _paired_stats(ksp_arr, ppor_arr),
    }

    out_json.write_text(json.dumps(cmp, indent=2, default=str), encoding="utf-8")

    lines = [
        f"# {c_prefix.upper()} Comparison: Strict v1.3 vs KSP-FF K=50 hops",
        "",
        f"Seeds: {len(seeds)}",
        f"Requests per seed: {summaries['strict_v1.3']['n'] * 6000 if summaries['strict_v1.3']['n'] else 0} total evaluated",
        "",
        "## Summary statistics",
        "",
        "| Method | Mean blocking | Std | Median | P25 | P75 | Overload | NSB | Avg delay ms | Avg FS |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for key, s in summaries.items():
        lines.append(
            f"| {s['display_name']} | {s['blocking_rate_mean']:.4%} | {s['blocking_rate_std']:.4%} | "
            f"{s['blocking_rate_median']:.4%} | {s['blocking_rate_p25']:.4%} | {s['blocking_rate_p75']:.4%} | "
            f"{s['overload_rate_mean']:.4%} | {s['nsb_rate_mean']:.4%} | {s['avg_delay_ms_mean']:.2f} | {s['avg_fs_mean']:.2f} |"
        )
    lines.extend([
        "",
        "## Strict v1.3 vs KSP-FF K=50 hops",
        "",
        f"Mean difference: {cmp['strict_vs_ksp']['mean_diff_pp']}",
        f"Median difference: {_pp(cmp['strict_vs_ksp']['median_diff'])}",
        f"Bootstrap 95% CI: [{cmp['strict_vs_ksp']['bootstrap_95_ci_pp'][0]}, {cmp['strict_vs_ksp']['bootstrap_95_ci_pp'][1]}]",
        f"Wilcoxon p: {cmp['strict_vs_ksp']['wilcoxon_p']:.6f}",
        f"Sign-flip permutation p: {cmp['strict_vs_ksp']['sign_flip_permutation_p']:.6f}",
        f"Strict better seeds: {cmp['strict_vs_ksp']['wins_a_lower']}/{len(seeds)}",
        f"Relative reduction: {cmp['strict_vs_ksp']['relative_reduction']*100:.2f}%",
        "",
        "## Strict v1.3 vs PPO-R Top-1",
        "",
        f"Mean difference: {cmp['strict_vs_ppo_r']['mean_diff_pp']}",
        f"Bootstrap 95% CI: [{cmp['strict_vs_ppo_r']['bootstrap_95_ci_pp'][0]}, {cmp['strict_vs_ppo_r']['bootstrap_95_ci_pp'][1]}]",
        f"Wilcoxon p: {cmp['strict_vs_ppo_r']['wilcoxon_p']:.6f}",
        f"Sign-flip permutation p: {cmp['strict_vs_ppo_r']['sign_flip_permutation_p']:.6f}",
        f"Strict better seeds: {cmp['strict_vs_ppo_r']['wins_a_lower']}/{len(seeds)}",
        "",
        "## KSP-FF K=50 hops vs PPO-R Top-1",
        "",
        f"Mean difference: {cmp['ksp_vs_ppo_r']['mean_diff_pp']}",
        f"Bootstrap 95% CI: [{cmp['ksp_vs_ppo_r']['bootstrap_95_ci_pp'][0]}, {cmp['ksp_vs_ppo_r']['bootstrap_95_ci_pp'][1]}]",
        f"Wilcoxon p: {cmp['ksp_vs_ppo_r']['wilcoxon_p']:.6f}",
        f"Sign-flip permutation p: {cmp['ksp_vs_ppo_r']['sign_flip_permutation_p']:.6f}",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return cmp


def build_interaction(
    ppoc_cmp: Dict[str, Any],
    dfc_cmp: Dict[str, Any],
    by_method: Dict[str, List[Dict[str, Any]]],
    seeds: List[int],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    strict_ppo = _blocking_array(by_method, "ppo_c+strict_v13", seeds)
    ksp_ppo = _blocking_array(by_method, "ppo_c+ksp_ff_highest", seeds)
    strict_df = _blocking_array(by_method, "df_c+strict_v13", seeds)
    ksp_df = _blocking_array(by_method, "df_c+ksp_ff_highest", seeds)

    delta_ppo = strict_ppo - ksp_ppo
    delta_df = strict_df - ksp_df
    interaction = delta_df - delta_ppo

    interaction_stats = _paired_stats(delta_df, delta_ppo)  # (df delta) - (ppo delta)

    out = {
        "delta_ppo_c": {
            "mean": float(np.mean(delta_ppo)),
            "median": float(np.median(delta_ppo)),
            "std": float(np.std(delta_ppo, ddof=1)),
            "bootstrap_95_ci": list(_bootstrap_paired_ci(delta_ppo)),
        },
        "delta_df_c": {
            "mean": float(np.mean(delta_df)),
            "median": float(np.median(delta_df)),
            "std": float(np.std(delta_df, ddof=1)),
            "bootstrap_95_ci": list(_bootstrap_paired_ci(delta_df)),
        },
        "interaction": {
            "mean": float(np.mean(interaction)),
            "median": float(np.median(interaction)),
            "std": float(np.std(interaction, ddof=1)),
            "bootstrap_95_ci": list(_bootstrap_paired_ci(interaction)),
            "wilcoxon_p": float(stats.wilcoxon(interaction, alternative="two-sided").pvalue),
        },
        "interpretation": "",
    }

    mean_int = out["interaction"]["mean"]
    ci_lo, ci_hi = out["interaction"]["bootstrap_95_ci"]
    if ci_hi < 0:
        interp = "Strict v1.3's advantage over KSP-FF is larger under DF_C than under PPO-C (interaction significant)."
    elif ci_lo > 0:
        interp = "Strict v1.3's advantage over KSP-FF is smaller under DF_C than under PPO-C (interaction significant)."
    else:
        interp = "No significant C-side interaction; Strict v1.3's advantage over KSP-FF is similar under PPO-C and DF_C."
    out["interpretation"] = interp

    out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# C-Side Interaction: Strict v1.3 vs KSP-FF K=50 hops",
        "",
        f"Δ_PPO-C = blocking(Strict, PPO-C) - blocking(KSP-FF, PPO-C)",
        f"Δ_DF_C  = blocking(Strict, DF_C) - blocking(KSP-FF, DF_C)",
        f"Interaction = Δ_DF_C - Δ_PPO-C",
        "",
        "| Quantity | Mean | Median | Std | Bootstrap 95% CI |",
        "|---|---:|---:|---:|---:|",
        f"| Δ_PPO-C | {out['delta_ppo_c']['mean']*100:.4f}pp | {out['delta_ppo_c']['median']*100:.4f}pp | {out['delta_ppo_c']['std']*100:.4f}pp | [{out['delta_ppo_c']['bootstrap_95_ci'][0]*100:.4f}%, {out['delta_ppo_c']['bootstrap_95_ci'][1]*100:.4f}%] |",
        f"| Δ_DF_C  | {out['delta_df_c']['mean']*100:.4f}pp | {out['delta_df_c']['median']*100:.4f}pp | {out['delta_df_c']['std']*100:.4f}pp | [{out['delta_df_c']['bootstrap_95_ci'][0]*100:.4f}%, {out['delta_df_c']['bootstrap_95_ci'][1]*100:.4f}%] |",
        f"| Interaction | {out['interaction']['mean']*100:.4f}pp | {out['interaction']['median']*100:.4f}pp | {out['interaction']['std']*100:.4f}pp | [{out['interaction']['bootstrap_95_ci'][0]*100:.4f}%, {out['interaction']['bootstrap_95_ci'][1]*100:.4f}%] |",
        "",
        f"Wilcoxon p for interaction: {out['interaction']['wilcoxon_p']:.6f}",
        "",
        f"**Interpretation:** {interp}",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_paired_statistics(
    ppoc_cmp: Dict[str, Any],
    dfc_cmp: Dict[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    # Holm correction across the two primary comparisons (Strict vs KSP under PPO-C and DF_C).
    raw_ps = [
        ppoc_cmp["strict_vs_ksp"]["wilcoxon_p"],
        dfc_cmp["strict_vs_ksp"]["wilcoxon_p"],
    ]
    adjusted = _holm_correction(raw_ps)
    ppoc_cmp["strict_vs_ksp"]["holm_adjusted_p"] = adjusted[0]
    dfc_cmp["strict_vs_ksp"]["holm_adjusted_p"] = adjusted[1]

    out = {
        "primary_comparisons": {
            "ppo_c_strict_vs_ksp": ppoc_cmp["strict_vs_ksp"],
            "df_c_strict_vs_ksp": dfc_cmp["strict_vs_ksp"],
        },
        "secondary_comparisons": {
            "ppo_c_strict_vs_ppo_r": ppoc_cmp["strict_vs_ppo_r"],
            "df_c_strict_vs_ppo_r": dfc_cmp["strict_vs_ppo_r"],
            "ppo_c_ksp_vs_ppo_r": ppoc_cmp["ksp_vs_ppo_r"],
            "df_c_ksp_vs_ppo_r": dfc_cmp["ksp_vs_ppo_r"],
        },
        "holm_correction": {
            "raw_pvalues": raw_ps,
            "adjusted_pvalues": adjusted,
            "alpha": 0.05,
        },
    }
    out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Paired Statistics",
        "",
        "All comparisons are paired by traffic seed. n = number of seeds.",
        "",
        "## Primary comparisons (Holm-corrected)",
        "",
        "| Comparison | Mean Δ | Median Δ | 95% CI | Wilcoxon p | Holm-adj p | Strict better / total | Relative reduction |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| PPO-C Strict vs KSP-FF | {out['primary_comparisons']['ppo_c_strict_vs_ksp']['mean_diff_pp']} | {_pp(out['primary_comparisons']['ppo_c_strict_vs_ksp']['median_diff'])} | [{out['primary_comparisons']['ppo_c_strict_vs_ksp']['bootstrap_95_ci_pp'][0]}, {out['primary_comparisons']['ppo_c_strict_vs_ksp']['bootstrap_95_ci_pp'][1]}] | {out['primary_comparisons']['ppo_c_strict_vs_ksp']['wilcoxon_p']:.6f} | {out['primary_comparisons']['ppo_c_strict_vs_ksp']['holm_adjusted_p']:.6f} | {out['primary_comparisons']['ppo_c_strict_vs_ksp']['wins_a_lower']}/{out['primary_comparisons']['ppo_c_strict_vs_ksp']['n']} | {out['primary_comparisons']['ppo_c_strict_vs_ksp']['relative_reduction']*100:.2f}% |",
        f"| DF_C Strict vs KSP-FF | {out['primary_comparisons']['df_c_strict_vs_ksp']['mean_diff_pp']} | {_pp(out['primary_comparisons']['df_c_strict_vs_ksp']['median_diff'])} | [{out['primary_comparisons']['df_c_strict_vs_ksp']['bootstrap_95_ci_pp'][0]}, {out['primary_comparisons']['df_c_strict_vs_ksp']['bootstrap_95_ci_pp'][1]}] | {out['primary_comparisons']['df_c_strict_vs_ksp']['wilcoxon_p']:.6f} | {out['primary_comparisons']['df_c_strict_vs_ksp']['holm_adjusted_p']:.6f} | {out['primary_comparisons']['df_c_strict_vs_ksp']['wins_a_lower']}/{out['primary_comparisons']['df_c_strict_vs_ksp']['n']} | {out['primary_comparisons']['df_c_strict_vs_ksp']['relative_reduction']*100:.2f}% |",
        "",
        "## Secondary comparisons",
        "",
        "| Comparison | Mean Δ | 95% CI | Wilcoxon p | Sign-flip p | Better / total |",
        "|---|---:|---:|---:|---:|---:|",
        f"| PPO-C Strict vs PPO-R | {out['secondary_comparisons']['ppo_c_strict_vs_ppo_r']['mean_diff_pp']} | [{out['secondary_comparisons']['ppo_c_strict_vs_ppo_r']['bootstrap_95_ci_pp'][0]}, {out['secondary_comparisons']['ppo_c_strict_vs_ppo_r']['bootstrap_95_ci_pp'][1]}] | {out['secondary_comparisons']['ppo_c_strict_vs_ppo_r']['wilcoxon_p']:.6f} | {out['secondary_comparisons']['ppo_c_strict_vs_ppo_r']['sign_flip_permutation_p']:.6f} | {out['secondary_comparisons']['ppo_c_strict_vs_ppo_r']['wins_a_lower']}/{out['secondary_comparisons']['ppo_c_strict_vs_ppo_r']['n']} |",
        f"| DF_C Strict vs PPO-R | {out['secondary_comparisons']['df_c_strict_vs_ppo_r']['mean_diff_pp']} | [{out['secondary_comparisons']['df_c_strict_vs_ppo_r']['bootstrap_95_ci_pp'][0]}, {out['secondary_comparisons']['df_c_strict_vs_ppo_r']['bootstrap_95_ci_pp'][1]}] | {out['secondary_comparisons']['df_c_strict_vs_ppo_r']['wilcoxon_p']:.6f} | {out['secondary_comparisons']['df_c_strict_vs_ppo_r']['sign_flip_permutation_p']:.6f} | {out['secondary_comparisons']['df_c_strict_vs_ppo_r']['wins_a_lower']}/{out['secondary_comparisons']['df_c_strict_vs_ppo_r']['n']} |",
        f"| PPO-C KSP-FF vs PPO-R | {out['secondary_comparisons']['ppo_c_ksp_vs_ppo_r']['mean_diff_pp']} | [{out['secondary_comparisons']['ppo_c_ksp_vs_ppo_r']['bootstrap_95_ci_pp'][0]}, {out['secondary_comparisons']['ppo_c_ksp_vs_ppo_r']['bootstrap_95_ci_pp'][1]}] | {out['secondary_comparisons']['ppo_c_ksp_vs_ppo_r']['wilcoxon_p']:.6f} | {out['secondary_comparisons']['ppo_c_ksp_vs_ppo_r']['sign_flip_permutation_p']:.6f} | {out['secondary_comparisons']['ppo_c_ksp_vs_ppo_r']['wins_a_lower']}/{out['secondary_comparisons']['ppo_c_ksp_vs_ppo_r']['n']} |",
        f"| DF_C KSP-FF vs PPO-R | {out['secondary_comparisons']['df_c_ksp_vs_ppo_r']['mean_diff_pp']} | [{out['secondary_comparisons']['df_c_ksp_vs_ppo_r']['bootstrap_95_ci_pp'][0]}, {out['secondary_comparisons']['df_c_ksp_vs_ppo_r']['bootstrap_95_ci_pp'][1]}] | {out['secondary_comparisons']['df_c_ksp_vs_ppo_r']['wilcoxon_p']:.6f} | {out['secondary_comparisons']['df_c_ksp_vs_ppo_r']['sign_flip_permutation_p']:.6f} | {out['secondary_comparisons']['df_c_ksp_vs_ppo_r']['wins_a_lower']}/{out['secondary_comparisons']['df_c_ksp_vs_ppo_r']['n']} |",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out


def build_action_distribution_audit(
    by_method: Dict[str, List[Dict[str, Any]]],
    seeds: List[int],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {"methods": {}}
    for method in MAIN_METHODS:
        rows = sorted(by_method.get(method, []), key=lambda r: r["seed"])
        path_lens = []
        hop_counts = []
        mod_names = []
        path_idx_counts: Counter = Counter()
        block_start_counts: Counter = Counter()
        required_fs_counts: Counter = Counter()
        for r in rows:
            path_lens.extend(r.get("path_length_list", []))
            hop_counts.extend(r.get("hop_count_list", []))
            mod_names.extend(r.get("mod_name_list", []))
            path_idx_counts.update({int(k): int(round(v * r["admitted"])) for k, v in r.get("path_idx_distribution", {}).items()})
            block_start_counts.update({int(k): int(round(v * r["admitted"])) for k, v in r.get("block_start_distribution", {}).items()})
            required_fs_counts.update({int(k): int(round(v * r["admitted"])) for k, v in r.get("required_fs_distribution", {}).items()})
        out["methods"][method] = {
            "n_admitted_observations": len(path_lens),
            "path_length_km": {
                "mean": float(np.mean(path_lens)) if path_lens else 0.0,
                "std": float(np.std(path_lens, ddof=1)) if path_lens else 0.0,
                "median": float(np.median(path_lens)) if path_lens else 0.0,
            },
            "hop_count": {
                "mean": float(np.mean(hop_counts)) if hop_counts else 0.0,
                "std": float(np.std(hop_counts, ddof=1)) if hop_counts else 0.0,
                "median": float(np.median(hop_counts)) if hop_counts else 0.0,
            },
            "modulation_counts": dict(Counter(mod_names).most_common()),
            "path_idx_distribution": dict(path_idx_counts.most_common(10)),
            "block_start_distribution_top10": dict(block_start_counts.most_common(10)),
            "required_fs_distribution": dict(required_fs_counts.most_common(10)),
        }
    out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Action Distribution Audit",
        "",
        "Aggregated over admitted requests across all formal seeds.",
        "",
        "| Method | N admitted | Avg path km | Median path km | Avg hops | Median hops | Modulation (top) |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for method in MAIN_METHODS:
        d = out["methods"][method]
        mod_str = ", ".join(f"{k}={v}" for k, v in list(d["modulation_counts"].items())[:4])
        lines.append(
            f"| {METHOD_RENAME.get(method, method)} | {d['n_admitted_observations']} | "
            f"{d['path_length_km']['mean']:.2f} | {d['path_length_km']['median']:.2f} | "
            f"{d['hop_count']['mean']:.2f} | {d['hop_count']['median']:.2f} | {mod_str} |"
        )
    lines.extend([
        "",
        "## Top path indices (by count)",
        "",
    ])
    for method in MAIN_METHODS:
        d = out["methods"][method]
        lines.append(f"**{METHOD_RENAME.get(method, method)}:** {d['path_idx_distribution']}")
    lines.extend([
        "",
        "## Top required FS counts",
        "",
    ])
    for method in MAIN_METHODS:
        d = out["methods"][method]
        lines.append(f"**{METHOD_RENAME.get(method, method)}:** {d['required_fs_distribution']}")
    lines.extend([
        "",
        "## Top block start slots",
        "",
    ])
    for method in MAIN_METHODS:
        d = out["methods"][method]
        lines.append(f"**{METHOD_RENAME.get(method, method)}:** {d['block_start_distribution_top10']}")
    lines.extend([
        "",
        "**Note:** Causal claims about why these distributions differ require trajectory-divergence or action-level counterfactual diagnostics; the table itself is descriptive only.",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out


def classify(strict_vs_ksp: Dict[str, Any]) -> str:
    mean_diff = strict_vs_ksp["mean_diff"]
    ci_lo, ci_hi = strict_vs_ksp["bootstrap_95_ci"]
    adj_p = strict_vs_ksp.get("holm_adjusted_p", strict_vs_ksp["wilcoxon_p"])
    wins = strict_vs_ksp["wins_a_lower"]
    n = strict_vs_ksp["n"]
    if mean_diff < 0 and ci_hi < 0 and adj_p < 0.05 and wins / n >= 0.70:
        return "Significant win"
    if mean_diff < 0:
        return "Directional win"
    if abs(mean_diff) < 1e-6:
        return "No advantage"
    return "Loss"


def build_final_decision(
    ppoc_cmp: Dict[str, Any],
    dfc_cmp: Dict[str, Any],
    interaction: Dict[str, Any],
    out_json: Path,
    out_md: Path,
) -> Dict[str, Any]:
    ppoc_class = classify(ppoc_cmp["strict_vs_ksp"])
    dfc_class = classify(dfc_cmp["strict_vs_ksp"])

    lowest_method = min(
        [
            ("PPO-C + PPO-R Top-1", ppoc_cmp["summaries"]["ppo_r_top1"]["blocking_rate_mean"]),
            ("PPO-C + KSP-FF K=50 hops", ppoc_cmp["summaries"]["ksp_ff_k50_hops"]["blocking_rate_mean"]),
            ("PPO-C + Strict v1.3", ppoc_cmp["summaries"]["strict_v1.3"]["blocking_rate_mean"]),
            ("DF_C + PPO-R Top-1", dfc_cmp["summaries"]["ppo_r_top1"]["blocking_rate_mean"]),
            ("DF_C + KSP-FF K=50 hops", dfc_cmp["summaries"]["ksp_ff_k50_hops"]["blocking_rate_mean"]),
            ("DF_C + Strict v1.3", dfc_cmp["summaries"]["strict_v1.3"]["blocking_rate_mean"]),
        ],
        key=lambda x: x[1],
    )

    out = {
        "ppo_c_strict_vs_ksp_classification": ppoc_class,
        "df_c_strict_vs_ksp_classification": dfc_class,
        "interaction_mean_pp": interaction["interaction"]["mean"] * 100,
        "lowest_blocking_combination": lowest_method[0],
        "lowest_blocking_rate": lowest_method[1],
        "answers": {},
    }

    # Overall recommendation.
    if ppoc_class == "Significant win" and dfc_class == "Significant win":
        rec = "Strict v1.3 shows robust cross-C-strategy advantage over KSP-FF K=50 hops."
        v135_allowed = True
    elif (ppoc_class in ("Significant win", "Directional win")) and (dfc_class in ("Significant win", "Directional win")):
        rec = "Strict v1.3 is directionally better than KSP-FF under both C strategies, but significance is conditional on C strategy."
        v135_allowed = True
    elif ppoc_class in ("Significant win", "Directional win") and dfc_class in ("No advantage", "Loss"):
        rec = "Strict v1.3 advantage is PPO-C-specific; it does not generalize to DF_C."
        v135_allowed = False
    elif dfc_class == "Loss":
        rec = "Strict v1.3 degrades under DF_C; do not expand to v1.35 until the C-policy distribution shift is diagnosed."
        v135_allowed = False
    else:
        rec = "Inconclusive; do not claim general superiority over KSP-FF."
        v135_allowed = False

    out["overall_recommendation"] = rec
    out["v1_35_diagnostic_pilot_allowed"] = v135_allowed

    out["answers"] = {
        "1_ppo_c_significant": ppoc_class == "Significant win",
        "2_df_c_significant": dfc_class == "Significant win",
        "3_cross_c_stable": ppoc_class in ("Significant win", "Directional win") and dfc_class in ("Significant win", "Directional win"),
        "4_df_c_distribution_shift": dfc_class in ("No advantage", "Loss") or abs(interaction["interaction"]["mean"]) > 0.001,
        "5_strict_vs_ppo_r_stable": ppoc_cmp["strict_vs_ppo_r"]["mean_diff"] < 0 and dfc_cmp["strict_vs_ppo_r"]["mean_diff"] < 0,
        "6_lowest_blocking": lowest_method[0],
        "7_v1_35_allowed": v135_allowed,
        "8_paper_conclusion": rec,
        "9_next_recommended_experiment": "If cross-C advantage holds: run v1.35 diagnostic pilot with ≤1000 groups under Strict protocol and explicit gate safety baseline. If advantage is PPO-C-specific: first diagnose C-policy distribution shift and train a COST239-native PPO-R checkpoint before any broader claim.",
    }

    out_json.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# FINAL DECISION: Strict v1.3 vs KSP-FF K=50 hops under PPO-C and DF_C",
        "",
        "## Primary results",
        "",
        f"- PPO-C + Strict v1.3 vs PPO-C + KSP-FF K=50 hops: **{ppoc_class}**",
        f"- DF_C + Strict v1.3 vs DF_C + KSP-FF K=50 hops: **{dfc_class}**",
        f"- C-side interaction mean: {interaction['interaction']['mean']*100:.4f}pp",
        f"- Lowest-blocking C×R combination: **{lowest_method[0]}** ({lowest_method[1]:.4%})",
        "",
        "## Answers to the required questions",
        "",
        f"1. **PPO-C significantly better?** {'Yes' if out['answers']['1_ppo_c_significant'] else 'No' if ppoc_class=='Loss' else 'Directional only'}.",
        f"2. **DF_C significantly better?** {'Yes' if out['answers']['2_df_c_significant'] else 'No' if dfc_class=='Loss' else 'Directional only'}.",
        f"3. **Advantage cross-C stable?** {'Yes' if out['answers']['3_cross_c_stable'] else 'No'}.",
        f"4. **DF_C causes distribution shift?** {'Yes' if out['answers']['4_df_c_distribution_shift'] else 'No clear evidence'}.",
        f"5. **Strict stable vs PPO-R?** {'Yes' if out['answers']['5_strict_vs_ppo_r_stable'] else 'No'}.",
        f"6. **Lowest blocking combination:** {out['answers']['6_lowest_blocking']}.",
        f"7. **v1.35 diagnostic pilot allowed?** {'Yes' if out['answers']['7_v1_35_allowed'] else 'No'}.",
        f"8. **Strongest paper conclusion:** {out['answers']['8_paper_conclusion']}",
        f"9. **Next recommended experiment:** {out['answers']['9_next_recommended_experiment']}",
        "",
        "## Caveats",
        "",
        "- PPO-R checkpoint `agent_r_mixed.pt` was trained on NSFNET/extended/32 slots; PPO-C checkpoint was trained on COST239/default/100 slots. Both are deployed here on COST239/default/320 slots with longer episodes. Results are therefore existing-system evaluations, not fully native comparisons.",
        "- DF_C + Strict v1.3 is a cross-C-policy transfer test because the ranker was trained on data generated under PPO-C.",
        "- Causal explanations for any blocking differences (fragmentation, load balancing, future-resource protection) remain hypotheses without trajectory-divergence or action-level counterfactual diagnostics.",
    ]
    out_md.write_text("\n".join(lines), encoding="utf-8")
    return out


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--formal_dir", default=str(ROOT / "sa_hmarl/experiments/v13_strict_fixed/v13_vs_ksp_k50_hops_ppoc_dfc/formal"))
    parser.add_argument("--out_dir", default=str(ROOT / "sa_hmarl/experiments/v13_strict_fixed/v13_vs_ksp_k50_hops_ppoc_dfc"))
    args = parser.parse_args()

    formal_dir = Path(args.formal_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_method, seeds = load_results(formal_dir)
    if not seeds:
        print("[aggregate] No results found. Exiting.")
        return 1

    print(f"[aggregate] Loaded {len(seeds)} seeds: {seeds}")

    raw = build_raw_results(by_method, seeds)
    (out_dir / "RAW_RESULTS.json").write_text(json.dumps(raw, indent=2, default=str), encoding="utf-8")
    build_per_seed_csv(by_method, seeds, out_dir / "PER_SEED_RESULTS.csv")

    ppoc_cmp = build_c_comparison(by_method, seeds, "ppo_c", out_dir / "PPOC_COMPARISON.json", out_dir / "PPOC_COMPARISON.md")
    dfc_cmp = build_c_comparison(by_method, seeds, "df_c", out_dir / "DFC_COMPARISON.json", out_dir / "DFC_COMPARISON.md")
    interaction = build_interaction(ppoc_cmp, dfc_cmp, by_method, seeds, out_dir / "C_SIDE_INTERACTION.json", out_dir / "C_SIDE_INTERACTION.md")
    paired = build_paired_statistics(ppoc_cmp, dfc_cmp, out_dir / "PAIRED_STATISTICS.json", out_dir / "PAIRED_STATISTICS.md")
    action_dist = build_action_distribution_audit(by_method, seeds, out_dir / "ACTION_DISTRIBUTION_AUDIT.json", out_dir / "ACTION_DISTRIBUTION_AUDIT.md")
    final = build_final_decision(ppoc_cmp, dfc_cmp, interaction, out_dir / "FINAL_DECISION.json", out_dir / "FINAL_DECISION.md")

    status = {
        "completed": True,
        "n_seeds": len(seeds),
        "seeds": seeds,
        "ppo_c_classification": final["ppo_c_strict_vs_ksp_classification"],
        "df_c_classification": final["df_c_strict_vs_ksp_classification"],
        "lowest_blocking_combination": final["lowest_blocking_combination"],
        "v1_35_allowed": final["v1_35_diagnostic_pilot_allowed"],
    }
    (out_dir / "STATUS.md").write_text(
        f"# Status\n\n- Completed: {status['completed']}\n- Seeds: {status['n_seeds']}\n"
        f"- PPO-C classification: {status['ppo_c_classification']}\n"
        f"- DF_C classification: {status['df_c_classification']}\n"
        f"- Lowest blocking: {status['lowest_blocking_combination']}\n"
        f"- v1.35 allowed: {status['v1_35_allowed']}\n",
        encoding="utf-8",
    )

    manifest = {
        "deliverables": [
            "PROTOCOL_LOCK.md",
            "PREREGISTRATION.json",
            "C_SIDE_IMPLEMENTATION_AUDIT.md",
            "CHECKPOINT_AUDIT.md",
            "SMOKE_TEST.md",
            "RAW_RESULTS.json",
            "PER_SEED_RESULTS.csv",
            "PPOC_COMPARISON.json", "PPOC_COMPARISON.md",
            "DFC_COMPARISON.json", "DFC_COMPARISON.md",
            "C_SIDE_INTERACTION.json", "C_SIDE_INTERACTION.md",
            "PAIRED_STATISTICS.json", "PAIRED_STATISTICS.md",
            "ACTION_DISTRIBUTION_AUDIT.json", "ACTION_DISTRIBUTION_AUDIT.md",
            "FINAL_DECISION.json", "FINAL_DECISION.md",
            "STATUS.md",
            "MANIFEST.json",
        ],
        "seeds": seeds,
    }
    (out_dir / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print("[aggregate] All deliverables written to", out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
