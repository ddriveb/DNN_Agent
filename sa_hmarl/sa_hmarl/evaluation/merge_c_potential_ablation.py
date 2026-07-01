"""Merge per-seed partial JSON files into a final ablation report.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.merge_c_potential_ablation \
        --inputs s42.json s123.json s456.json \
        --output_json final.json --output_md final.md
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
def _validate_configs(all_reports: List[Dict[str, Any]]) -> Optional[str]:
    """Return error message if configs are inconsistent, else None."""
    if not all_reports:
        return "No input files"
    ref = all_reports[0].get("config", {})
    ref_cp_c = ref.get("agent_c_checkpoint", "")
    ref_cp_r = ref.get("agent_r_checkpoint", "")
    ref_methods = set(ref.get("methods", []))
    ref_eps = ref.get("episodes", 0)
    ref_rpe = ref.get("requests_per_episode", 0)

    for i, r in enumerate(all_reports[1:], 1):
        cfg = r.get("config", {})
        if cfg.get("agent_c_checkpoint", "") != ref_cp_c:
            return f"File {i}: agent_c_checkpoint mismatch"
        if cfg.get("agent_r_checkpoint", "") != ref_cp_r:
            return f"File {i}: agent_r_checkpoint mismatch"
        if set(cfg.get("methods", [])) != ref_methods:
            return f"File {i}: methods mismatch: {set(cfg.get('methods',[]))} vs {ref_methods}"
        if cfg.get("episodes", 0) != ref_eps:
            return f"File {i}: episodes mismatch"
        if cfg.get("requests_per_episode", 0) != ref_rpe:
            return f"File {i}: requests_per_episode mismatch"
    return None


# ---------------------------------------------------------------------------
# Merge episode metrics
# ---------------------------------------------------------------------------
def _merge_episodes(all_reports: List[Dict[str, Any]]) -> Dict[str, Dict[int, List[Dict[str, Any]]]]:
    """Merge per-method, per-seed metrics from all report files.

    Supports both final JSON (methods key) and partial JSON (per_method_seed key).
    Returns: {method: {seed: [per_seed_metric_dict, ...]}}
             where each metric_dict contains SEED-level aggregates (not per-episode).
    """
    merged: Dict[str, Dict[int, List[Dict]]] = {}

    for report in all_reports:
        cfg = report.get("config", {})
        seeds = cfg.get("seeds", [])
        methods = cfg.get("methods", [])

        # Try final JSON format first (methods.METHOD.per_seed)
        methods_data = report.get("methods", {})
        if methods_data and any(isinstance(v, dict) and "per_seed" in v for v in methods_data.values()):
            for method in methods:
                if method not in merged:
                    merged[method] = {}
                per_seed_list = methods_data.get(method, {}).get("per_seed", [])
                for ps in per_seed_list:
                    seed = ps.get("seed")
                    if seed is None:
                        continue
                    if seed not in merged[method]:
                        merged[method][seed] = []
                    merged[method][seed].append(ps)
        else:
            # Try partial JSON format (per_method_seed)
            per_ms = report.get("per_method_seed", {})
            for method in methods:
                if method not in merged:
                    merged[method] = {}
                seed_dict = per_ms.get(method, {})
                for seed_str, eps_list in seed_dict.items():
                    seed = int(seed_str)
                    if seed not in seeds:
                        continue
                    if seed not in merged[method]:
                        merged[method][seed] = []
                    merged[method][seed].extend(eps_list)

    return merged


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------
def _bootstrap_paired_ci(
    rates_a: List[float],
    rates_b: List[float],
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> Tuple[float, float, float]:
    """Episode-paired bootstrap 95% CI for mean(rates_a - rates_b).

    Each rate list should have one entry per episode, aligned by (seed, episode).
    """
    n = min(len(rates_a), len(rates_b))
    diffs = np.array(rates_a[:n]) - np.array(rates_b[:n])
    mean_delta = float(np.mean(diffs))
    if n < 2:
        return mean_delta, mean_delta, mean_delta
    rng = np.random.RandomState(seed)
    boot_means = [float(np.mean(diffs[rng.randint(0, n, size=n)])) for _ in range(n_bootstrap)]
    ci_lo = float(np.percentile(boot_means, 2.5))
    ci_hi = float(np.percentile(boot_means, 97.5))
    return mean_delta, ci_lo, ci_hi


def _episode_blocking_rate_map(
    reports: List[Dict[str, Any]], method: str
) -> Dict[Tuple[int, int], float]:
    """Return episode blocking rates keyed by ``(seed, episode_index)``."""
    rates: Dict[Tuple[int, int], float] = {}
    for report in reports:
        for seed_str, episodes in report.get("per_method_seed", {}).get(method, {}).items():
            seed = int(seed_str)
            for episode_index, metrics in enumerate(episodes):
                total = int(metrics.get("total", 0))
                if total <= 0:
                    continue
                rates[(seed, episode_index)] = float(metrics.get("blocked", 0)) / total
    return rates


# ---------------------------------------------------------------------------
# Combine per-seed aggregate dicts into a multi-seed aggregate
# ---------------------------------------------------------------------------
def _combine_per_seed_aggs(seed_dicts: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Weighted average of per-seed aggregate dicts.

    Each seed_dict should have keys like blocking_rate, total, mean_delay_ms, etc.
    """
    if not seed_dicts:
        return {}
    totals = [d.get("total", 0) for d in seed_dicts]
    total_n = sum(totals)
    if total_n == 0:
        return seed_dicts[0]

    # Weighted average for rates
    result = {}
    rate_keys = ["blocking_rate", "raw_mask_empty_rate", "no_suitable_block_rate",
                 "server_overload_rate", "deadline_failure_rate",
                 "agreement_with_ppo_c"]
    for k in rate_keys:
        vals = [d.get(k, 0) * d.get("total", 0) for d in seed_dicts]
        result[k] = sum(vals) / max(total_n, 1)

    # Simple mean for continuous metrics (weighted by counts)
    result["total"] = total_n
    result["mean_delay_ms"] = float(np.mean([d.get("mean_delay_ms", 0) for d in seed_dicts]))
    result["avg_fs"] = float(np.mean([d.get("avg_fs", 0) for d in seed_dicts]))
    result["mean_active_connections"] = float(np.mean([d.get("mean_active_connections", 0) for d in seed_dicts]))
    result["mean_decision_time_ms"] = float(np.mean([d.get("mean_decision_time_ms", 0) for d in seed_dicts]))
    result["changed_action_count"] = sum(d.get("changed_action_count", 0) for d in seed_dicts)
    result["changed_action_blocking_rate"] = None  # Not meaningful in aggregate

    # Copy rest from first dict
    for k in ["avg_waste", "avg_path_km", "p50_delay_ms", "p95_delay_ms",
              "p95_active_connections", "max_active_connections",
              "mean_valid_c_actions", "mean_total_valid_r_actions",
              "mean_selected_valid_r_actions", "p95_decision_time_ms"]:
        result[k] = float(np.mean([d.get(k, 0) for d in seed_dicts]))

    return result


# ---------------------------------------------------------------------------
# Verdict
# ---------------------------------------------------------------------------
def _compute_verdict(ppo_agg: Dict, name: str, method_agg: Dict,
                     ppo_per_seed: List[Dict], method_per_seed: List[Dict]) -> Dict:
    blk_imp = ppo_agg["blocking_rate"] - method_agg["blocking_rate"]
    empty_imp = ppo_agg["raw_mask_empty_rate"] - method_agg["raw_mask_empty_rate"]
    overload_chg = method_agg["server_overload_rate"] - ppo_agg["server_overload_rate"]
    seeds_imp = sum(1 for p, s in zip(ppo_per_seed, method_per_seed) if p["blocking_rate"] > s["blocking_rate"])
    ts = len(ppo_per_seed)

    if blk_imp >= 0.01 and empty_imp > -0.005 and seeds_imp >= max(2, ts - 1) and overload_chg <= 0.005:
        status = "PASS"
    elif blk_imp >= 0.003 or (blk_imp > 0 and seeds_imp >= 2):
        status = "MARGINAL"
    else:
        status = "FAIL"
    return {"method": name, "status": status, "blocking_improvement_pp": blk_imp * 100,
            "raw_mask_empty_improvement_pp": empty_imp * 100, "seeds_improved": f"{seeds_imp}/{ts}",
            "server_overload_change_pp": overload_chg * 100}


# ---------------------------------------------------------------------------
# Markdown report
# ---------------------------------------------------------------------------
def _build_markdown(report: Dict) -> str:
    cfg = report["config"]
    methods_list = cfg["methods"]

    def _f(v): return "N/A" if v is None else f"{v:.4f}"
    def _pct(v): return "N/A" if v is None else f"{v*100:.2f}%"
    def _dpp(a, b): return (a - b) * 100.0

    ppo_agg = report["methods"]["ppo_c"]["aggregate"]

    lines = [
        "# C-Side Demand-Aware Potential — Resource-Component Ablation",
        "",
        f"- Agent-C: `{cfg['agent_c_checkpoint']}`  |  PPO-R: `{cfg['agent_r_checkpoint']}`",
        f"- Seeds: {cfg['seeds']}  |  Eps/seed: {cfg['episodes']}  |  Req/ep: {cfg['requests_per_episode']}",
        f"- Arrival: {cfg.get('arrival_interval','?')}s  |  Methods: {', '.join(methods_list)}",
        f"- Frozen R: {report['frozen_params_unchanged']['agent_r']}",
        "",
        "## Aggregate Comparison",
        "",
        "| Metric | " + " | ".join(m.replace("_"," ").title() for m in methods_list) + " |",
        "|---|" + "|".join("---" for _ in methods_list) + "|",
    ]
    for label, key, pct in [
        ("Total requests", "total", False),
        ("Blocking rate", "blocking_rate", True),
        ("Raw-mask-empty rate", "raw_mask_empty_rate", True),
        ("No-suitable-block rate", "no_suitable_block_rate", True),
        ("Server-overload rate", "server_overload_rate", True),
        ("Deadline-infeasible rate", "deadline_failure_rate", True),
        ("Mean delay ms", "mean_delay_ms", False),
        ("Avg FS", "avg_fs", False),
    ]:
        vals = [_pct(report["methods"][m]["aggregate"][key]) if pct else _f(report["methods"][m]["aggregate"][key]) for m in methods_list]
        lines.append(f"| {label} | " + " | ".join(vals) + " |")
    lines.append("")

    # Per-seed
    lines.extend(["## Per-Seed Blocking", "",
        "| Seed | " + " | ".join(m.replace("_"," ").title() for m in methods_list) + " |",
        "|---|" + "|".join("---" for _ in methods_list) + "|"])
    for i, seed in enumerate(cfg["seeds"]):
        vals = [_pct(report["methods"][m]["per_seed"][i]["blocking_rate"]) for m in methods_list]
        lines.append(f"| {seed} | " + " | ".join(vals) + " |")
    lines.append("")

    # Verdicts
    lines.extend(["## Verdicts", ""])
    for m in methods_list:
        if m == "ppo_c":
            continue
        v = report["methods"][m].get("verdict", {})
        ci = report["methods"][m].get("bootstrap_ci", {})
        lines.append(f"### {m.replace('_',' ').title()}: **{v.get('status','N/A')}**")
        lines.append(f"- Blocking Δ: {v.get('blocking_improvement_pp',0):.2f} pp  |  Seeds: {v.get('seeds_improved','?')}")
        if ci:
            lines.append(f"- Bootstrap 95% CI: [{ci.get('ci_95_lo',0)*100:+.2f}, {ci.get('ci_95_hi',0)*100:+.2f}] pp")
        lines.append("")

    # Resource-Component Decomposition
    lines.extend(["## Resource-Component Decomposition", "",
        "| Component | Blocking | vs PPO-C | Bootstrap 95% CI |",
        "|---|---:|---:|:---:|",
        f"| PPO-C baseline | {_pct(ppo_agg['blocking_rate'])} | — | — |"])
    for m_key, m_label in [("success_filter_only","Success-filter only"),
                            ("spectrum_only","Spectrum only"),
                            ("compute_only","Compute only"),
                            ("potential_only","Full potential")]:
        if m_key not in report["methods"]:
            continue
        m_agg = report["methods"][m_key]["aggregate"]
        ci = report["methods"][m_key].get("bootstrap_ci", {})
        ci_str = f"[{ci.get('ci_95_lo',0)*100:+.2f}, {ci.get('ci_95_hi',0)*100:+.2f}] pp" if ci else "—"
        lines.append(f"| {m_label} | {_pct(m_agg['blocking_rate'])} | {_dpp(ppo_agg['blocking_rate'], m_agg['blocking_rate']):+.2f}pp | {ci_str} |")
    lines.append("")

    # Resource verdict
    rv = report.get("resource_verdict", "INCONCLUSIVE")
    lines.extend([f"## Resource-Component Verdict", "", f"**{rv}**", ""])

    # Changed-action behavior
    lines.extend(["## Changed-Action Behavior (descriptive only)", "",
        "| Method | Changed | Chg-Blk | Mean/P95 Decision ms |",
        "|---|---:|---:|---:|"])
    for m in methods_list:
        if m == "ppo_c":
            continue
        agg = report["methods"][m]["aggregate"]
        chg = agg.get("changed_action_count", 0)
        chg_blk = _pct(agg.get("changed_action_blocking_rate"))
        dec = f"{_f(agg.get('mean_decision_time_ms'))}/{_f(agg.get('p95_decision_time_ms'))}"
        lines.append(f"| {m.replace('_',' ').title()} | {chg} | {chg_blk} | {dec} |")
    lines.append("")

    lines.extend(["---", f"*Elapsed: {report['elapsed_seconds']:.1f}s*"])
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Merge per-seed ablation JSONs")
    parser.add_argument("--inputs", type=str, nargs="+", required=True)
    parser.add_argument("--output_json", type=str, required=True)
    parser.add_argument("--output_md", type=str, required=True)
    args = parser.parse_args()

    # Load all inputs
    all_reports = []
    for path in args.inputs:
        with open(path, "r", encoding="utf-8") as f:
            all_reports.append(json.load(f))

    # Final per-seed reports contain aggregates only. Their companion partial
    # files retain episode-level metrics required for a genuinely paired CI.
    episode_reports = []
    for path in args.inputs:
        input_path = Path(path)
        partial_path = Path(f"{path}.partial.json")
        source_path = input_path if "per_method_seed" in all_reports[len(episode_reports)] else partial_path
        if not source_path.exists():
            raise SystemExit(
                f"ERROR: episode-level data missing for {path}; expected {partial_path}"
            )
        with source_path.open("r", encoding="utf-8") as f:
            episode_reports.append(json.load(f))

    # Validate
    err = _validate_configs(all_reports)
    if err:
        print(f"ERROR: {err}")
        raise SystemExit(1)

    cfg = all_reports[0]["config"]
    methods = cfg.get("methods", [])
    seeds_all = []
    for r in all_reports:
        seeds_all.extend(r["config"].get("seeds", []))
    seeds_all = sorted(set(seeds_all))

    # Collect per-seed data from all reports (methods.METHOD.per_seed format)
    per_seed_data: Dict[str, List[Dict]] = {m: [] for m in methods}
    for report in all_reports:
        methods_data = report.get("methods", {})
        for method in methods:
            per_seed_list = methods_data.get(method, {}).get("per_seed", [])
            for ps in per_seed_list:
                per_seed_data[method].append(ps)

    # Aggregate across all seeds
    aggregate_data: Dict[str, Dict] = {}
    for method in methods:
        aggregate_data[method] = _combine_per_seed_aggs(per_seed_data[method])

    ppo_agg = aggregate_data["ppo_c"]
    ppo_ps = per_seed_data["ppo_c"]

    ppo_episode_rates = _episode_blocking_rate_map(episode_reports, "ppo_c")
    methods_output: Dict[str, Dict] = {"ppo_c": {"aggregate": ppo_agg, "per_seed": ppo_ps}}
    for method in methods:
        if method == "ppo_c":
            continue
        agg = aggregate_data[method]
        ps = per_seed_data[method]
        verdict = _compute_verdict(ppo_agg, method, agg, ppo_ps, ps)
        entry = {"aggregate": agg, "per_seed": ps, "verdict": verdict}
        if method in ("spectrum_only", "compute_only", "potential_only"):
            method_episode_rates = _episode_blocking_rate_map(episode_reports, method)
            paired_keys = sorted(set(ppo_episode_rates) & set(method_episode_rates))
            if len(paired_keys) != len(ppo_episode_rates):
                raise SystemExit(
                    f"ERROR: incomplete episode pairing for {method}: "
                    f"{len(paired_keys)}/{len(ppo_episode_rates)} episodes"
                )
            ppo_rates = [ppo_episode_rates[key] for key in paired_keys]
            method_rates = [method_episode_rates[key] for key in paired_keys]
            delta, ci_lo, ci_hi = _bootstrap_paired_ci(
                method_rates, ppo_rates, n_bootstrap=10000
            )
            entry["bootstrap_ci"] = {"delta_vs_ppo_c": delta, "ci_95_lo": ci_lo, "ci_95_hi": ci_hi}
        methods_output[method] = entry

    # Resource-component verdict
    spec_ci = methods_output.get("spectrum_only", {}).get("bootstrap_ci", {})
    comp_ci = methods_output.get("compute_only", {}).get("bootstrap_ci", {})
    full_ci = methods_output.get("potential_only", {}).get("bootstrap_ci", {})
    has_spec = "spectrum_only" in methods_output
    has_comp = "compute_only" in methods_output
    has_pot = "potential_only" in methods_output

    spec_sig = has_spec and spec_ci.get("ci_95_hi", 1) < 0
    comp_sig = has_comp and comp_ci.get("ci_95_hi", 1) < 0
    full_sig = has_pot and full_ci.get("ci_95_hi", 1) < 0

    if has_spec and has_comp:
        spec_delta = spec_ci.get("delta_vs_ppo_c", 0)
        comp_delta = comp_ci.get("delta_vs_ppo_c", 0)
        full_delta_v = full_ci.get("delta_vs_ppo_c", 0)
        best_single = min(spec_delta, comp_delta)
        full_beats = full_delta_v < best_single if has_pot else False

        # Primary determination: which component dominates blocking improvement
        spec_dominates = spec_sig and (not comp_sig or abs(spec_delta) > abs(comp_delta) * 1.5)
        comp_dominates = comp_sig and (not spec_sig or abs(comp_delta) > abs(spec_delta) * 1.5)

        if full_sig and full_beats and spec_sig and comp_sig:
            resource_verdict = "MULTI_RESOURCE_SYNERGY_SUPPORTED"
        elif spec_dominates:
            resource_verdict = "SPECTRUM_SUPPORTED"
        elif comp_dominates:
            resource_verdict = "COMPUTE_SUPPORTED"
        elif full_sig and not (spec_sig or comp_sig):
            resource_verdict = "NOT_SUPPORTED"
        else:
            resource_verdict = "INCONCLUSIVE"
    else:
        resource_verdict = "INCONCLUSIVE"

    # Frozen check
    r_frozen = all(r.get("frozen_params_unchanged", {}).get("agent_r", False) for r in all_reports)
    total_elapsed = sum(r.get("elapsed_seconds", 0) for r in all_reports)

    report = {
        "config": {**cfg, "seeds": seeds_all},
        "methods": methods_output,
        "resource_verdict": resource_verdict,
        "frozen_params_unchanged": {"agent_r": r_frozen},
        "elapsed_seconds": total_elapsed,
    }

    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    Path(args.output_md).write_text(_build_markdown(report), encoding="utf-8")

    print(f"Merged {len(all_reports)} files → {args.output_json}")
    print(f"Methods: {methods}")
    print(f"Seeds: {seeds_all}")
    print(f"Resource verdict: {resource_verdict}")
    print(f"Total elapsed: {total_elapsed:.0f}s")


if __name__ == "__main__":
    main()
