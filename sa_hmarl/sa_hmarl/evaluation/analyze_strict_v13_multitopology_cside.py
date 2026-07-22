"""Analyze and report Strict v1.3 multi-topology C-side fair evaluation results.

Implements the Strict v1.3 multi-topology fair closed-loop evaluation analyzer
with integrity gating, exact sign-flip paired tests, Holm correction, and the
required deliverable set.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import stats


SCHEMA_VERSION = "v1.3-corrected-2026-07-14"
TOPOLOGIES = [
    "xlron_cost239_ptrnet_real",
    "xlron_german17",
    "xlron_nsfnet_deeprmsa",
    "xlron_jpn48",
]
SEEDS = [3030, 4040, 5050, 6060, 7070]
C_MODES_MAIN = ["ppo_c", "df_c"]
R_MAIN = ["ppo_r_top1", "ksp_ff_highest", "strict_v13"]
EXPECTED_MAIN_MATRIX_CELLS = len(TOPOLOGIES) * len(SEEDS) * len(C_MODES_MAIN) * len(R_MAIN)

# Verified input directories.  Priority order: verified, then seed43/seed44 variants.
_VERIFIED_ROOTS = [
    Path("sa_hmarl/experiments/v13_strict_multitopology_cside_verified"),
    Path("sa_hmarl/experiments/v13_strict_multitopology_cside_verified_seed43"),
    Path("sa_hmarl/experiments/v13_strict_multitopology_cside_verified_seed44"),
]
# Fallback for backward compatibility when no verified dir exists.
_FALLBACK_ROOT = Path("sa_hmarl/experiments/v13_strict_multitopology_cside_corrected")

CHECKPOINTS = {
    "agent_c_cost239_safe": "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt",
    "agent_c_delayaware_snap24": "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt",
    "agent_r_mixed": "sa_hmarl/checkpoints/agent_r_mixed.pt",
    "strict_v13": "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt",
    "old_v13": "sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _sha256(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "N/A"


def _resolve_base_dirs() -> List[Path]:
    roots: List[Path] = []
    for p in _VERIFIED_ROOTS:
        if p.exists():
            roots.append(p)
    if not roots and _FALLBACK_ROOT.exists():
        roots.append(_FALLBACK_ROOT)
    return roots


def _base_dir() -> Path:
    roots = _resolve_base_dirs()
    return roots[0] if roots else _VERIFIED_ROOTS[0]


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Loading and integrity
# ---------------------------------------------------------------------------

def _load_all_results(smoke: bool = False) -> Tuple[Dict[Tuple[str, str, str, int], Dict[str, Any]], Dict[str, Any]]:
    """Load all per-task results and run integrity checks.

    Returns (data, integrity_report).
    ``data`` is keyed by (topology, c_mode, r_mode, seed).
    """
    sub = "smoke" if smoke else "per_task"
    roots = _resolve_base_dirs()

    integrity: Dict[str, Any] = {
        "roots_searched": [str(r) for r in roots],
        "schema_version": SCHEMA_VERSION,
        "schema_violations": [],
        "config_hash_mismatches": [],
        "trace_hash_mismatches": [],
        "checkpoint_sha256_mismatches": [],
        "source_manifest_issues": [],
        "duplicate_cells": [],
        "superseded_cells": [],
        "rows_loaded": 0,
        "files_loaded": [],
        "files_failed": [],
    }

    # cell -> list of (path, mtime, row)
    raw_cells: Dict[Tuple[str, str, str, int], List[Tuple[Path, float, Dict[str, Any]]]] = {}

    for root in roots:
        for topology in TOPOLOGIES:
            for seed in SEEDS:
                task_dir = root / sub / topology / f"seed_{seed}"
                json_path = task_dir / "results.json"
                if not json_path.exists():
                    continue
                payload = _load_json(json_path)
                if payload is None:
                    integrity["files_failed"].append(str(json_path))
                    continue
                integrity["files_loaded"].append(str(json_path))
                mtime = os.path.getmtime(json_path)
                for row in payload.get("results", []):
                    key = (row.get("topology"), row.get("c_mode"), row.get("r_mode"), int(row.get("seed", -1)))
                    raw_cells.setdefault(key, []).append((json_path, mtime, row))
                    integrity["rows_loaded"] += 1

    data: Dict[Tuple[str, str, str, int], Dict[str, Any]] = {}
    cell_sources: Dict[Tuple[str, str, str, int], List[str]] = {}

    # Resolve duplicates / superseded outputs by keeping the latest mtime per cell.
    for key, entries in raw_cells.items():
        if len(entries) > 1:
            integrity["duplicate_cells"].append({
                "cell": key,
                "occurrences": len(entries),
                "sources": [str(e[0]) for e in entries],
            })
            entries = sorted(entries, key=lambda e: e[1], reverse=True)
            # All but the latest are superseded.
            for e in entries[1:]:
                integrity["superseded_cells"].append({
                    "cell": key,
                    "source": str(e[0]),
                    "mtime": e[1],
                })
        chosen = entries[0]
        data[key] = chosen[2]
        cell_sources.setdefault(key, []).append(str(chosen[0]))

    # Integrity checks across chosen rows.
    _check_integrity(data, integrity)

    return data, integrity


def _check_integrity(data: Dict[Tuple[str, str, str, int], Dict[str, Any]], integrity: Dict[str, Any]) -> None:
    """Populate integrity report with hash / schema consistency violations."""
    if not data:
        return

    # Global consistency: checkpoint_sha256s and source_manifest (if present).
    global_ckpt_ref: Optional[Dict[str, str]] = None
    global_manifest_ref: Optional[Any] = None

    # Per-topology-seed consistency: config_hash, request_trace_hash.
    per_ts_ref: Dict[Tuple[str, int], Dict[str, str]] = {}

    for key, row in data.items():
        topology, c_mode, r_mode, seed = key

        # Schema version.
        if row.get("schema_version") != SCHEMA_VERSION:
            integrity["schema_violations"].append({
                "cell": key,
                "got": row.get("schema_version"),
                "expected": SCHEMA_VERSION,
            })

        # Global checkpoint hashes.
        ckpts = row.get("checkpoint_sha256s")
        if isinstance(ckpts, dict):
            if global_ckpt_ref is None:
                global_ckpt_ref = dict(ckpts)
            elif ckpts != global_ckpt_ref:
                integrity["checkpoint_sha256_mismatches"].append({
                    "cell": key,
                    "got": dict(ckpts),
                    "expected": dict(global_ckpt_ref),
                })

        # Source manifest (optional; validated only when present).
        manifest = row.get("source_manifest")
        if manifest is not None:
            if global_manifest_ref is None:
                global_manifest_ref = manifest
            elif manifest != global_manifest_ref:
                integrity["source_manifest_issues"].append({
                    "cell": key,
                    "detail": "source_manifest differs across rows",
                })

        # Per topology-seed hashes.
        ts_key = (topology, seed)
        cfg_hash = row.get("config_hash")
        trace_hash = row.get("request_trace_hash")
        ts_record = {"config_hash": cfg_hash, "request_trace_hash": trace_hash}
        if ts_key not in per_ts_ref:
            per_ts_ref[ts_key] = ts_record
        else:
            ref = per_ts_ref[ts_key]
            if cfg_hash != ref["config_hash"]:
                integrity["config_hash_mismatches"].append({
                    "cell": key,
                    "got": cfg_hash,
                    "expected": ref["config_hash"],
                })
            if trace_hash != ref["request_trace_hash"]:
                integrity["trace_hash_mismatches"].append({
                    "cell": key,
                    "got": trace_hash,
                    "expected": ref["request_trace_hash"],
                })


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def _blocking_by_seed_aligned(
    data: Dict[Tuple[str, str, str, int], Dict[str, Any]],
    topology: str,
    c_mode: str,
    r_mode: str,
) -> Tuple[np.ndarray, List[int], List[int]]:
    """Return (values, present_seed_ids, missing_seed_ids) in fixed SEEDS order."""
    vals: List[float] = []
    present: List[int] = []
    missing: List[int] = []
    for seed in SEEDS:
        row = data.get((topology, c_mode, r_mode, seed))
        if row is None:
            missing.append(seed)
        else:
            present.append(seed)
            vals.append(float(row["blocking_rate"]))
    return np.asarray(vals, dtype=float), present, missing


def _paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 5000, seed: int = 12345
) -> Tuple[float, float, float]:
    """Bootstrap CI of mean(a-b) using paired resampling."""
    rng = np.random.RandomState(seed)
    n = len(a)
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs.append(float(a[idx].mean() - b[idx].mean()))
    diffs_arr = np.asarray(diffs)
    return float(diffs_arr.mean()), float(np.percentile(diffs_arr, 2.5)), float(np.percentile(diffs_arr, 97.5))


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = np.asarray(a) - np.asarray(b)
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0, 1.0
    stat, p = stats.wilcoxon(diff)
    return float(stat), float(p)


def _exact_sign_flip_test(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    """Exact sign-flip test for paired differences d = a - b.

    Enumerates all 2^n sign assignments.  Returns (one_sided_p, two_sided_p).
    One-sided H0: mean(d)=0 vs H1: mean(d)>0.
    Two-sided H0: mean(d)=0 vs H1: mean(d)!=0.
    """
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    n = len(d)
    if n == 0:
        return 1.0, 1.0
    obs_mean = float(d.mean())
    obs_abs = abs(obs_mean)
    total = 1 << n
    count_one = 0
    count_two = 0
    # Enumerate sign patterns via bit mask (+1 for bit 0, -1 for bit 1).
    for mask in range(total):
        signs = np.where(np.bitwise_and(np.right_shift(mask, np.arange(n)), 1), -1.0, 1.0)
        flipped_mean = float((d * signs).mean())
        if flipped_mean >= obs_mean - 1e-15:
            count_one += 1
        if abs(flipped_mean) >= obs_abs - 1e-15:
            count_two += 1
    return count_one / total, count_two / total


def _permutation_test(a: np.ndarray, b: np.ndarray, n_perm: int = 20000, seed: int = 12345) -> float:
    """Backward-compatible wrapper returning the one-sided exact sign-flip p-value."""
    return _exact_sign_flip_test(a, b)[0]


def _holm(pvals: List[float]) -> List[float]:
    """Holm step-down correction.

    Sorted p ascending: raw_i = p_i * (m - i + 1).
    adjusted_sorted = maximum.accumulate(raw_i), clipped to [0, 1], mapped back.
    """
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    if m == 0:
        return []
    order = np.argsort(p)
    sorted_p = p[order]
    raw = sorted_p * np.arange(m, 0, -1)
    adj_sorted = np.maximum.accumulate(raw)
    adj_sorted = np.clip(adj_sorted, 0.0, 1.0)
    adj = np.empty(m)
    adj[order] = adj_sorted
    return adj.tolist()


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

def _test_permutation_and_holm() -> None:
    """Known-input unit tests for permutation direction and Holm step-down."""
    # Permutation: baseline blocking strictly larger than strict blocking ->
    # observed mean(a-b) > 0, so one-sided p should be small.
    base = np.array([0.10, 0.12, 0.11, 0.13, 0.10])
    strict = np.array([0.05, 0.04, 0.06, 0.05, 0.05])
    p_one, p_two = _exact_sign_flip_test(base, strict)
    assert 0.0 <= p_one <= 0.05, f"expected small one-sided p, got {p_one}"
    assert 0.0 <= p_two <= 0.10, f"expected small two-sided p, got {p_two}"
    p_small = _permutation_test(base, strict, n_perm=5000, seed=42)
    assert 0.0 <= p_small <= 0.05, f"expected small p, got {p_small}"

    # Reverse: strict worse -> observed mean(a-b) < 0, p should be large.
    strict_worse = np.array([0.20, 0.22, 0.21, 0.23, 0.20])
    p_large = _permutation_test(base, strict_worse, n_perm=5000, seed=42)
    assert p_large >= 0.95, f"expected large p, got {p_large}"

    # Holm: classic example with 3 ordered p-values.
    pvals = [0.01, 0.03, 0.06]
    adj = _holm(pvals)
    expected = [0.03, 0.06, 0.06]
    assert np.allclose(adj, expected), f"Holm mismatch: {adj} vs {expected}"

    # Required example: p=[0.01,0.04,0.05] -> [0.03,0.08,0.08].
    pvals2 = [0.01, 0.04, 0.05]
    adj2 = _holm(pvals2)
    expected2 = [0.03, 0.08, 0.08]
    assert np.allclose(adj2, expected2), f"Holm required example mismatch: {adj2} vs {expected2}"

    # Monotonicity and clipping.
    pvals3 = [0.4, 0.4, 0.4]
    adj3 = _holm(pvals3)
    assert all(a <= 1.0 for a in adj3), "Holm adjusted p must be <= 1"
    assert adj3[0] >= adj3[1] >= adj3[2]
    print("[analyze tests] exact sign-flip and Holm unit tests passed")


# ---------------------------------------------------------------------------
# Matrix completeness
# ---------------------------------------------------------------------------

def _matrix_completeness(
    data: Dict[Tuple[str, str, str, int], Dict[str, Any]]
) -> Dict[str, Any]:
    """Compute main-matrix completeness and per-method seed presence."""
    expected_cells: List[Tuple[str, str, str, int]] = [
        (topology, c_mode, r_mode, seed)
        for topology in TOPOLOGIES
        for c_mode in C_MODES_MAIN
        for r_mode in R_MAIN
        for seed in SEEDS
    ]
    present_cells = set(data.keys())
    missing_cells = [c for c in expected_cells if c not in present_cells]

    # Per (topology, c_mode, r_mode): present/missing seeds.
    per_method: Dict[str, Dict[str, Any]] = {}
    missing_seed_ids_by_method: Dict[str, Dict[str, List[int]]] = {}
    for topology in TOPOLOGIES:
        for c_mode in C_MODES_MAIN:
            for r_mode in R_MAIN:
                method_key = f"{c_mode}+{r_mode}"
                present: List[int] = []
                missing: List[int] = []
                for seed in SEEDS:
                    if (topology, c_mode, r_mode, seed) in data:
                        present.append(seed)
                    else:
                        missing.append(seed)
                per_method[f"{topology}|{method_key}"] = {
                    "present_seed_ids": present,
                    "missing_seed_ids": missing,
                    "complete": len(missing) == 0,
                }
                missing_seed_ids_by_method.setdefault(method_key, {})[topology] = missing

    return {
        "expected_cells": EXPECTED_MAIN_MATRIX_CELLS,
        "present_cells": len(present_cells),
        "missing_cells": len(missing_cells),
        "missing_cell_list": missing_cells,
        "per_method": per_method,
        "missing_seed_ids_by_method": missing_seed_ids_by_method,
        "main_matrix_complete": len(missing_cells) == 0,
    }


# ---------------------------------------------------------------------------
# Paired comparison
# ---------------------------------------------------------------------------

def _compare(
    data: Dict[Tuple[str, str, str, int], Dict[str, Any]],
    topology: str,
    c_mode: str,
    baseline_r: str,
) -> Optional[Dict[str, Any]]:
    strict_vals, strict_present, strict_missing = _blocking_by_seed_aligned(
        data, topology, c_mode, "strict_v13"
    )
    base_vals, base_present, base_missing = _blocking_by_seed_aligned(
        data, topology, c_mode, baseline_r
    )

    common = [s for s in SEEDS if s in strict_present and s in base_present]
    missing = [s for s in SEEDS if s not in common]
    incomplete = len(common) != len(SEEDS)

    if not common:
        return None

    # Align values to common seeds in fixed order.
    strict_aligned = np.asarray(
        [strict_vals[strict_present.index(s)] for s in common], dtype=float
    )
    base_aligned = np.asarray(
        [base_vals[base_present.index(s)] for s in common], dtype=float
    )

    d = base_aligned - strict_aligned
    mean_diff = float(d.mean())
    std_diff = float(d.std(ddof=1)) if len(d) > 1 else 0.0

    if base_aligned.mean() > 0:
        rel_reduction = float(d.mean() / base_aligned.mean())
    else:
        rel_reduction = None

    wins = int((strict_aligned < base_aligned - 1e-12).sum())
    ties = int(np.isclose(strict_aligned, base_aligned, atol=1e-12).sum())
    losses = int((strict_aligned > base_aligned + 1e-12).sum())

    mean_boot, lo, hi = _paired_bootstrap_ci(base_aligned, strict_aligned)
    _, p_wilcox = _wilcoxon(base_aligned, strict_aligned)

    if not incomplete:
        p_one, p_two = _exact_sign_flip_test(base_aligned, strict_aligned)
    else:
        p_one, p_two = float("nan"), float("nan")

    return {
        "topology": topology,
        "c_mode": c_mode,
        "baseline_r": baseline_r,
        "n_seeds": len(common),
        "common_seed_ids": common,
        "missing_seed_ids": missing,
        "strict_present_seed_ids": strict_present,
        "baseline_present_seed_ids": base_present,
        "incomplete": incomplete,
        "mean_difference": mean_diff,
        "std_difference": std_diff,
        "relative_reduction": rel_reduction,
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "bootstrap_mean_base_minus_strict": mean_boot,
        "bootstrap_ci_lower": lo,
        "bootstrap_ci_upper": hi,
        "wilcoxon_pvalue": p_wilcox,
        "exact_sign_flip_one_sided_pvalue": p_one,
        "exact_sign_flip_two_sided_pvalue": p_two,
        "raw_pvalue_for_holm": p_one,
    }


# ---------------------------------------------------------------------------
# Aggregate statistics
# ---------------------------------------------------------------------------

def _aggregate(
    data: Dict[Tuple[str, str, str, int], Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    aggregate: Dict[str, Dict[str, Any]] = {}
    for topology in TOPOLOGIES:
        c_modes = list(C_MODES_MAIN)
        if topology == "xlron_cost239_ptrnet_real":
            c_modes.append("ppo_c_snap24")
        for c_mode in c_modes:
            for r_mode in R_MAIN + ["old_v13"]:
                arr, present, _ = _blocking_by_seed_aligned(data, topology, c_mode, r_mode)
                if len(arr) == 0:
                    continue
                rows = [data[(topology, c_mode, r_mode, s)] for s in present]
                aggregate[f"{topology}|{c_mode}|{r_mode}"] = {
                    "blocking_mean": float(arr.mean()),
                    "blocking_std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
                    "per_seed": {int(r["seed"]): float(r["blocking_rate"]) for r in rows},
                    "present_seed_ids": present,
                    "overload_mean": float(np.mean([r["overload_rate"] for r in rows])),
                    "nsb_mean": float(np.mean([r["nsb_rate"] for r in rows])),
                    "other_mean": float(np.mean([r["other_rate"] for r in rows])),
                    "avg_delay_ms": float(np.mean([r["avg_delay_ms"] for r in rows])),
                    "avg_fs": float(np.mean([r["avg_fs"] for r in rows])),
                    "avg_path_length_km": float(np.mean([r["avg_path_length_km"] for r in rows])),
                    "avg_hop_count": float(np.mean([r["avg_hop_count"] for r in rows])),
                    "avg_decision_ms": float(np.mean([r["avg_decision_ms"] for r in rows])),
                    "decision_p95_ms": float(np.mean([r["decision_p95_ms"] for r in rows])),
                    "avg_policy_decision_ms": float(np.mean([r.get("avg_policy_decision_ms", 0.0) for r in rows])),
                    "avg_audit_decision_ms": float(np.mean([r.get("avg_audit_decision_ms", 0.0) for r in rows])),
                    "failure_reason_distribution": _merge_failure_reasons(rows),
                }
    return aggregate


def _merge_failure_reasons(rows: List[Dict[str, Any]]) -> Dict[str, float]:
    totals: Dict[str, int] = {}
    total_requests = sum(int(r.get("total", 0)) for r in rows)
    for r in rows:
        for reason, count in r.get("failure_reason_distribution", {}).items():
            totals[reason] = totals.get(reason, 0) + int(count)
    if total_requests == 0:
        return {k: 0.0 for k in totals}
    return {k: float(v) / total_requests for k, v in totals.items()}


# ---------------------------------------------------------------------------
# Top-level analysis
# ---------------------------------------------------------------------------

def analyze(
    data: Dict[Tuple[str, str, str, int], Dict[str, Any]],
    integrity: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Run full analysis."""
    if integrity is None:
        integrity = {}

    matrix = _matrix_completeness(data)

    comparisons: List[Dict[str, Any]] = []
    for topology in TOPOLOGIES:
        for c_mode in C_MODES_MAIN:
            for baseline_r in ("ppo_r_top1", "ksp_ff_highest"):
                comp = _compare(data, topology, c_mode, baseline_r)
                if comp is not None:
                    comparisons.append(comp)

    # Holm correction by family: (c_mode, baseline_r) across topologies.
    families: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
    for comp in comparisons:
        key = (comp["c_mode"], comp["baseline_r"])
        families.setdefault(key, []).append(comp)
    for key, comps in families.items():
        pvals = [c["raw_pvalue_for_holm"] for c in comps]
        adj = _holm(pvals)
        for c, a in zip(comps, adj):
            c["holm_adjusted_pvalue"] = a

    aggregate = _aggregate(data)

    return {
        "schema_version": SCHEMA_VERSION,
        "integrity": integrity,
        "matrix_completeness": matrix,
        "comparisons": comparisons,
        "aggregate": aggregate,
    }


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------

def _fmt_pct(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x) * 100:.2f}%"


def _fmt_pp(x: Optional[float]) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x) * 100:.2f} pp"


def _fmt_f(x: Optional[float], decimals: int = 4) -> str:
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return "N/A"
    return f"{float(x):.{decimals}f}"


def _normalized_dist(dist: Any) -> Dict[str, float]:
    """Extract a normalized probability distribution from either a flat dict
    of probabilities or a verified-format ``_distribution`` output.
    """
    if isinstance(dist, dict) and "normalized" in dist:
        return {str(k): float(v) for k, v in dist["normalized"].items()}
    if isinstance(dist, dict):
        return {str(k): float(v) for k, v in dist.items()}
    return {}


# ---------------------------------------------------------------------------
# Report writers
# ---------------------------------------------------------------------------

def _write_main_report(results: Dict[str, Any], out_dir: Path) -> None:
    agg = results["aggregate"]
    comps = results["comparisons"]
    integrity = results["integrity"]
    matrix = results["matrix_completeness"]

    lines = ["# Strict v1.3 Multi-Topology C-side Fair Evaluation Results", ""]

    lines.append("## Integrity summary")
    lines.append(f"- Roots searched: {integrity.get('roots_searched', [])}")
    lines.append(f"- Rows loaded: {integrity.get('rows_loaded', 0)}")
    lines.append(f"- Schema violations: {len(integrity.get('schema_violations', []))}")
    lines.append(f"- Config hash mismatches: {len(integrity.get('config_hash_mismatches', []))}")
    lines.append(f"- Trace hash mismatches: {len(integrity.get('trace_hash_mismatches', []))}")
    lines.append(f"- Checkpoint SHA-256 mismatches: {len(integrity.get('checkpoint_sha256_mismatches', []))}")
    lines.append(f"- Source manifest issues: {len(integrity.get('source_manifest_issues', []))}")
    lines.append(f"- Duplicate cells: {len(integrity.get('duplicate_cells', []))}")
    lines.append(f"- Superseded cells: {len(integrity.get('superseded_cells', []))}")
    lines.append("")

    lines.append("## Main matrix completeness")
    lines.append(f"- Expected cells: {matrix['expected_cells']}")
    lines.append(f"- Present cells: {matrix['present_cells']}")
    lines.append(f"- Missing cells: {matrix['missing_cells']}")
    lines.append(f"- Main matrix complete: {matrix['main_matrix_complete']}")
    lines.append("")

    lines.append("## Per-topology aggregate blocking")
    lines.append("| Topology | C-side | R-side | Blocking mean ± std | Overload | NSB | Other |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for key, v in sorted(agg.items()):
        topology, c_mode, r_mode = key.split("|")
        lines.append(
            f"| {topology} | {c_mode} | {r_mode} | {_fmt_pct(v['blocking_mean'])} ± {_fmt_pct(v['blocking_std'])} | "
            f"{_fmt_pct(v['overload_mean'])} | {_fmt_pct(v['nsb_mean'])} | {_fmt_pct(v['other_mean'])} |"
        )
    lines.append("")

    lines.append("## Per-topology paired comparisons")
    for comp in comps:
        rel = _fmt_pct(comp.get("relative_reduction")) if comp.get("relative_reduction") is not None else "N/A"
        marker = " (INCOMPLETE)" if comp.get("incomplete") else ""
        lines.append(
            f"- **{comp['topology']} / {comp['c_mode']} / strict_v13 vs {comp['baseline_r']}**{marker}: "
            f"n={comp['n_seeds']}, common={comp['common_seed_ids']}, missing={comp.get('missing_seed_ids', [])}, "
            f"mean diff={_fmt_pp(comp['mean_difference'])}, rel red={rel}, "
            f"wins={comp['wins']}/{comp['ties']}/{comp['losses']}, "
            f"bootstrap CI=[{_fmt_f(comp['bootstrap_ci_lower'])}, {_fmt_f(comp['bootstrap_ci_upper'])}], "
            f"Wilcoxon p={_fmt_f(comp['wilcoxon_pvalue'])}, exact one-sided p={_fmt_f(comp.get('exact_sign_flip_one_sided_pvalue'))}, "
            f"Holm adj. p={_fmt_f(comp.get('holm_adjusted_pvalue'))}"
        )

    (out_dir / "MULTITOPOLOGY_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "MULTITOPOLOGY_RESULTS.json").write_text(
        json.dumps(results, indent=2, default=str), encoding="utf-8"
    )


def _write_paired_statistics(results: Dict[str, Any], out_dir: Path) -> None:
    comps = results["comparisons"]

    lines = ["# Paired Statistics: Strict v1.3 vs R-side Baselines", ""]
    lines.append(
        "| Topology | C-side | Baseline | n | Common seeds | Missing seeds | Mean diff (pp) | "
        "Rel. reduction | Wins/Ties/Losses | Boot. CI | Wilcoxon p | Exact one-sided p | Holm adj. p |"
    )
    lines.append("|---|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|")

    payload: List[Dict[str, Any]] = []
    for comp in comps:
        rel = comp.get("relative_reduction")
        rel_str = _fmt_pct(rel) if rel is not None else "N/A"
        ci = f"[{_fmt_f(comp['bootstrap_ci_lower'])}, {_fmt_f(comp['bootstrap_ci_upper'])}]"
        lines.append(
            f"| {comp['topology']} | {comp['c_mode']} | {comp['baseline_r']} | {comp['n_seeds']} | "
            f"{comp['common_seed_ids']} | {comp.get('missing_seed_ids', [])} | "
            f"{_fmt_pp(comp['mean_difference'])} | {rel_str} | "
            f"{comp['wins']}/{comp['ties']}/{comp['losses']} | {ci} | "
            f"{_fmt_f(comp['wilcoxon_pvalue'])} | {_fmt_f(comp.get('exact_sign_flip_one_sided_pvalue'))} | "
            f"{_fmt_f(comp.get('holm_adjusted_pvalue'))} |"
        )
        payload.append(comp)

    (out_dir / "PAIRED_STATISTICS.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "PAIRED_STATISTICS.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_failure_decomposition(results: Dict[str, Any], out_dir: Path) -> None:
    agg = results["aggregate"]

    lines = ["# Failure Decomposition", ""]
    lines.append("| Topology | C-side | R-side | Blocking | Overload | NSB | Other | c_no_valid | r_no_valid |")
    lines.append("|---|---|---|---:|---:|---:|---:|---:|---:|")

    payload: Dict[str, Dict[str, Any]] = {}
    for key, v in sorted(agg.items()):
        topology, c_mode, r_mode = key.split("|")
        fr = v.get("failure_reason_distribution", {})
        payload[key] = {
            "topology": topology,
            "c_mode": c_mode,
            "r_mode": r_mode,
            "blocking_mean": v["blocking_mean"],
            "overload_mean": v["overload_mean"],
            "nsb_mean": v["nsb_mean"],
            "other_mean": v["other_mean"],
            "c_no_valid_action_rate": fr.get("c_no_valid_action", 0.0),
            "r_no_valid_action_rate": fr.get("r_no_valid_action", 0.0),
            "failure_reason_distribution": fr,
        }
        lines.append(
            f"| {topology} | {c_mode} | {r_mode} | {_fmt_pct(v['blocking_mean'])} | "
            f"{_fmt_pct(v['overload_mean'])} | {_fmt_pct(v['nsb_mean'])} | {_fmt_pct(v['other_mean'])} | "
            f"{_fmt_pct(fr.get('c_no_valid_action', 0.0))} | {_fmt_pct(fr.get('r_no_valid_action', 0.0))} |"
        )

    (out_dir / "FAILURE_DECOMPOSITION.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "FAILURE_DECOMPOSITION.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_action_distribution_audit(results: Dict[str, Any], out_dir: Path) -> None:
    agg = results["aggregate"]

    lines = ["# Action Distribution Audit", ""]
    lines.append("| Topology | C-side | R-side | Avg path idx | Avg hops | Avg path km | Avg FS | Top mod |")
    lines.append("|---|---|---|---:|---:|---:|---:|---|")

    payload: Dict[str, Dict[str, Any]] = {}
    for key, v in sorted(agg.items()):
        topology, c_mode, r_mode = key.split("|")
        # Representative row: use first present seed.
        representative_key = None
        for seed in SEEDS:
            candidate = (topology, c_mode, r_mode, seed)
            if candidate in []:  # placeholder; we don't store raw data per seed here
                break
        # Aggregate-level audit from full result rows.
        rows = [
            results.get("_data", {}).get((topology, c_mode, r_mode, s))
            for s in v.get("present_seed_ids", [])
        ]
        rows = [r for r in rows if r is not None]

        def _top_key(dist: Dict[str, float]) -> str:
            if not dist:
                return "N/A"
            return max(dist.items(), key=lambda kv: kv[1])[0]

        mod_dists = [_normalized_dist(r.get("mod_distribution", {})) for r in rows]
        merged_mod: Dict[str, float] = {}
        for d in mod_dists:
            for k, v_ in d.items():
                merged_mod[k] = merged_mod.get(k, 0.0) + float(v_)
        total = sum(merged_mod.values())
        if total > 0:
            merged_mod = {k: v_ / total for k, v_ in merged_mod.items()}

        top_mod = _top_key(merged_mod)
        payload[key] = {
            "topology": topology,
            "c_mode": c_mode,
            "r_mode": r_mode,
            "avg_hop_count": v["avg_hop_count"],
            "avg_path_length_km": v["avg_path_length_km"],
            "avg_fs": v["avg_fs"],
            "top_modulation": top_mod,
            "mod_distribution": merged_mod,
        }
        lines.append(
            f"| {topology} | {c_mode} | {r_mode} | {_fmt_f(v.get('avg_selected_path_idx'), 2) if 'avg_selected_path_idx' in v else 'N/A'} | "
            f"{_fmt_f(v['avg_hop_count'], 2)} | {_fmt_f(v['avg_path_length_km'], 2)} | {_fmt_f(v['avg_fs'], 2)} | {top_mod} |"
        )

    (out_dir / "ACTION_DISTRIBUTION_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "ACTION_DISTRIBUTION_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_latency_audit(results: Dict[str, Any], out_dir: Path) -> None:
    agg = results["aggregate"]

    lines = ["# Latency Audit", ""]
    lines.append("| Topology | C-side | R-side | Avg decision ms | P95 decision ms | Policy ms | Audit ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|")

    payload: Dict[str, Dict[str, Any]] = {}
    for key, v in sorted(agg.items()):
        topology, c_mode, r_mode = key.split("|")
        payload[key] = {
            "topology": topology,
            "c_mode": c_mode,
            "r_mode": r_mode,
            "avg_decision_ms": v["avg_decision_ms"],
            "decision_p95_ms": v["decision_p95_ms"],
            "avg_policy_decision_ms": v["avg_policy_decision_ms"],
            "avg_audit_decision_ms": v["avg_audit_decision_ms"],
        }
        lines.append(
            f"| {topology} | {c_mode} | {r_mode} | {_fmt_f(v['avg_decision_ms'], 3)} | "
            f"{_fmt_f(v['decision_p95_ms'], 3)} | {_fmt_f(v['avg_policy_decision_ms'], 3)} | {_fmt_f(v['avg_audit_decision_ms'], 3)} |"
        )

    (out_dir / "LATENCY_AUDIT.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "LATENCY_AUDIT.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _write_matrix_completeness(results: Dict[str, Any], out_dir: Path) -> None:
    matrix = results["matrix_completeness"]

    lines = ["# Full Matrix Completeness", ""]
    lines.append(f"- Expected cells: {matrix['expected_cells']}")
    lines.append(f"- Present cells: {matrix['present_cells']}")
    lines.append(f"- Missing cells: {matrix['missing_cells']}")
    lines.append(f"- Main matrix complete: {matrix['main_matrix_complete']}")
    lines.append("")

    if matrix["missing_cell_list"]:
        lines.append("## Missing cells")
        for cell in matrix["missing_cell_list"]:
            lines.append(f"- {cell}")
        lines.append("")

    lines.append("## Per-method seed presence")
    lines.append("| Topology | Method | Present seeds | Missing seeds | Complete |")
    lines.append("|---|---|---|---|---|")
    for key, v in sorted(matrix["per_method"].items()):
        topology, method = key.split("|", 1)
        lines.append(
            f"| {topology} | {method} | {v['present_seed_ids']} | {v['missing_seed_ids']} | {v['complete']} |"
        )

    (out_dir / "FULL_MATRIX_COMPLETENESS.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "FULL_MATRIX_COMPLETENESS.json").write_text(json.dumps(matrix, indent=2, default=str), encoding="utf-8")


def _write_tables(results: Dict[str, Any], out_dir: Path) -> None:
    agg = results["aggregate"]
    comps = {f"{c['topology']}|{c['c_mode']}|{c['baseline_r']}": c for c in results["comparisons"]}

    def get_agg(topology, c_mode, r_mode):
        return agg.get(f"{topology}|{c_mode}|{r_mode}", {})

    def get_comp(topology, c_mode, baseline):
        return comps.get(f"{topology}|{c_mode}|{baseline}", {})

    # PPO-C table
    lines = ["# Strict v1.3 vs R-side baselines under PPO-C", ""]
    lines.append("| Topology | PPO-R Top-1 | KSP-FF highest K50 | strict v1.3 | v1.3 vs KSP abs pp | v1.3 vs KSP relative | wins vs KSP | Holm adj. p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for topology in TOPOLOGIES:
        c_mode = "ppo_c"
        ppo = get_agg(topology, c_mode, "ppo_r_top1")
        ksp = get_agg(topology, c_mode, "ksp_ff_highest")
        strict = get_agg(topology, c_mode, "strict_v13")
        comp = get_comp(topology, c_mode, "ksp_ff_highest")
        if not strict:
            continue
        rel = comp.get("relative_reduction")
        rel_str = _fmt_pct(rel) if rel is not None else "N/A"
        lines.append(
            f"| {topology} | {_fmt_pct(ppo.get('blocking_mean', 0))} ± {_fmt_pct(ppo.get('blocking_std', 0))} | "
            f"{_fmt_pct(ksp.get('blocking_mean', 0))} ± {_fmt_pct(ksp.get('blocking_std', 0))} | "
            f"{_fmt_pct(strict.get('blocking_mean', 0))} ± {_fmt_pct(strict.get('blocking_std', 0))} | "
            f"{_fmt_pp(comp.get('mean_difference', 0))} | "
            f"{rel_str} | "
            f"{comp.get('wins', 0)}/{comp.get('ties', 0)}/{comp.get('losses', 0)} | "
            f"{_fmt_f(comp.get('holm_adjusted_pvalue'))} |"
        )
    (out_dir / "PPO_C_TABLE.md").write_text("\n".join(lines), encoding="utf-8")

    # df_c table
    lines = ["# Strict v1.3 vs R-side baselines under df_c", ""]
    lines.append("| Topology | PPO-R Top-1 | KSP-FF highest K50 | strict v1.3 | v1.3 vs KSP abs pp | v1.3 vs KSP relative | wins vs KSP | Holm adj. p |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for topology in TOPOLOGIES:
        c_mode = "df_c"
        ppo = get_agg(topology, c_mode, "ppo_r_top1")
        ksp = get_agg(topology, c_mode, "ksp_ff_highest")
        strict = get_agg(topology, c_mode, "strict_v13")
        comp = get_comp(topology, c_mode, "ksp_ff_highest")
        if not strict:
            continue
        rel = comp.get("relative_reduction")
        rel_str = _fmt_pct(rel) if rel is not None else "N/A"
        lines.append(
            f"| {topology} | {_fmt_pct(ppo.get('blocking_mean', 0))} ± {_fmt_pct(ppo.get('blocking_std', 0))} | "
            f"{_fmt_pct(ksp.get('blocking_mean', 0))} ± {_fmt_pct(ksp.get('blocking_std', 0))} | "
            f"{_fmt_pct(strict.get('blocking_mean', 0))} ± {_fmt_pct(strict.get('blocking_std', 0))} | "
            f"{_fmt_pp(comp.get('mean_difference', 0))} | "
            f"{rel_str} | "
            f"{comp.get('wins', 0)}/{comp.get('ties', 0)}/{comp.get('losses', 0)} | "
            f"{_fmt_f(comp.get('holm_adjusted_pvalue'))} |"
        )
    (out_dir / "DF_C_TABLE.md").write_text("\n".join(lines), encoding="utf-8")

    # C-side sensitivity
    lines = ["# C-side sensitivity: strict v1.3 and KSP under PPO-C vs df_c", ""]
    lines.append("| Topology | PPO-C+v1.3 | df_c+v1.3 | PPO-C+KSP | df_c+KSP |")
    lines.append("|---|---:|---:|---:|---:|")
    for topology in TOPOLOGIES:
        ppoc_v13 = get_agg(topology, "ppo_c", "strict_v13")
        dfc_v13 = get_agg(topology, "df_c", "strict_v13")
        ppoc_ksp = get_agg(topology, "ppo_c", "ksp_ff_highest")
        dfc_ksp = get_agg(topology, "df_c", "ksp_ff_highest")
        lines.append(
            f"| {topology} | {_fmt_pct(ppoc_v13.get('blocking_mean', 0))} ± {_fmt_pct(ppoc_v13.get('blocking_std', 0))} | "
            f"{_fmt_pct(dfc_v13.get('blocking_mean', 0))} ± {_fmt_pct(dfc_v13.get('blocking_std', 0))} | "
            f"{_fmt_pct(ppoc_ksp.get('blocking_mean', 0))} ± {_fmt_pct(ppoc_ksp.get('blocking_std', 0))} | "
            f"{_fmt_pct(dfc_ksp.get('blocking_mean', 0))} ± {_fmt_pct(dfc_ksp.get('blocking_std', 0))} |"
        )
    (out_dir / "C_SIDE_SENSITIVITY.md").write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Final decision
# ---------------------------------------------------------------------------

QUESTIONS = [
    "Q1. Is the main matrix (4 topologies × 5 seeds × 2 C modes × 3 R modes) complete?",
    "Q2. Did all integrity checks (schema, hashes, checkpoints, trace) pass without violations?",
    "Q3. Under ppo_c, does strict_v13 show lower mean blocking than ksp_ff_highest in at least 3 of 4 topologies?",
    "Q4. Under df_c, does strict_v13 show lower mean blocking than ksp_ff_highest in at least 3 of 4 topologies?",
    "Q5. Under ppo_c, does strict_v13 show lower mean blocking than ppo_r_top1 in at least 3 of 4 topologies?",
    "Q6. Under df_c, does strict_v13 show lower mean blocking than ppo_r_top1 in at least 3 of 4 topologies?",
    "Q7. Are the v1.3 vs KSP-FF highest reductions statistically significant after Holm correction at alpha=0.05?",
    "Q8. Is there a C-side sensitivity: does the ranking of ppo_c+v1.3 vs df_c+v1.3 differ across topologies?",
    "Q9. Is the benefit source-topology specific (only COST239) or cross-topology?",
    "Q10. Final recommendation: launch full v1.35, run diagnostic pilot, or retrain topology-diverse?",
]


def _classify(results: Dict[str, Any]) -> Tuple[str, str]:
    comps = results["comparisons"]
    ksp_comps = [c for c in comps if c["baseline_r"] == "ksp_ff_highest" and not c.get("incomplete", False)]
    ppo_comps = [c for c in comps if c["baseline_r"] == "ppo_r_top1" and not c.get("incomplete", False)]

    def count_positive(comp_list):
        return sum(1 for c in comp_list if c["mean_difference"] < -1e-12)

    def count_4wins(comp_list):
        return sum(1 for c in comp_list if c["wins"] >= 4)

    ksp_positive = count_positive(ksp_comps)
    ksp_4wins = count_4wins(ksp_comps)
    ppo_positive = count_positive(ppo_comps)

    ppoc_v13 = {c["topology"]: c for c in ksp_comps if c["c_mode"] == "ppo_c"}
    dfc_v13 = {c["topology"]: c for c in ksp_comps if c["c_mode"] == "df_c"}
    ppoc_better = sum(
        1 for t in TOPOLOGIES
        if ppoc_v13.get(t, {}).get("mean_difference", 0) < dfc_v13.get(t, {}).get("mean_difference", 0)
    )

    if ksp_positive >= 3 and ppo_positive >= 3 and ksp_4wins >= 2 and ppoc_better >= 2:
        return "A", "strict v1.3 shows cross-topology generalization evidence"
    if ksp_positive >= 3 and ppoc_better >= 2:
        return "B", "benefit under PPO-C but df_c compresses or changes direction in some topologies"
    if ksp_positive == 1 and TOPOLOGIES[0] in [c["topology"] for c in ksp_comps if c["mean_difference"] < -1e-12]:
        return "C", "benefit is source-topology specific (COST239)"
    if ksp_positive >= 3 and ppoc_better < 2:
        return "D", "df_c is effective but PPO-C zero-shot transfer is the bottleneck"
    if ksp_positive <= 1:
        return "E", "no consistent cross-topology advantage over KSP-FF highest"
    return "B", "mixed directional gains"


def _answer_questions(results: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    comps = results["comparisons"]
    matrix = results["matrix_completeness"]
    integrity = results["integrity"]

    def comps_for(c_mode: str, baseline_r: str):
        return [c for c in comps if c["c_mode"] == c_mode and c["baseline_r"] == baseline_r]

    def positive_count(c_mode: str, baseline_r: str) -> int:
        return sum(1 for c in comps_for(c_mode, baseline_r) if c.get("mean_difference", 0) < -1e-12 and not c.get("incomplete", False))

    def sig_after_holm(c_mode: str, baseline_r: str, alpha: float = 0.05) -> bool:
        family = [c for c in comps_for(c_mode, baseline_r) if not c.get("incomplete", False)]
        return all(c.get("holm_adjusted_pvalue", 1.0) < alpha for c in family if c.get("mean_difference", 0) < -1e-12)

    q1 = matrix["main_matrix_complete"]
    q2 = (
        len(integrity.get("schema_violations", [])) == 0
        and len(integrity.get("config_hash_mismatches", [])) == 0
        and len(integrity.get("trace_hash_mismatches", [])) == 0
        and len(integrity.get("checkpoint_sha256_mismatches", [])) == 0
    )
    q3 = positive_count("ppo_c", "ksp_ff_highest") >= 3
    q4 = positive_count("df_c", "ksp_ff_highest") >= 3
    q5 = positive_count("ppo_c", "ppo_r_top1") >= 3
    q6 = positive_count("df_c", "ppo_r_top1") >= 3
    q7 = sig_after_holm("ppo_c", "ksp_ff_highest") and sig_after_holm("df_c", "ksp_ff_highest")

    ksp_comps = [c for c in comps if c["baseline_r"] == "ksp_ff_highest" and not c.get("incomplete", False)]
    ppoc_v13 = {c["topology"]: c for c in ksp_comps if c["c_mode"] == "ppo_c"}
    dfc_v13 = {c["topology"]: c for c in ksp_comps if c["c_mode"] == "df_c"}
    q8 = any(
        (ppoc_v13.get(t, {}).get("mean_difference", 0) - dfc_v13.get(t, {}).get("mean_difference", 0)) > 1e-12
        for t in TOPOLOGIES
    )

    cost239_better = any(
        c["topology"] == TOPOLOGIES[0] and c["mean_difference"] < -1e-12
        for c in ksp_comps
    )
    other_better = any(
        c["topology"] != TOPOLOGIES[0] and c["mean_difference"] < -1e-12
        for c in ksp_comps
    )
    q9 = "cross-topology" if cost239_better and other_better else ("source-topology specific" if cost239_better and not other_better else "no consistent benefit")

    grade, reason = _classify(results)
    if grade == "A":
        q10 = "launch full v1.35 (conditional on pilot confirmation)"
    elif grade in ("B", "C", "D"):
        q10 = "run diagnostic pilot; consider topology-diverse retraining if benefit is narrow"
    else:
        q10 = "retrain topology-diverse v1.3; do not launch v1.35"

    answers = [q1, q2, q3, q4, q5, q6, q7, q8, q9, q10]
    return {
        "grade": grade,
        "reason": reason,
        "questions": [
            {"question": QUESTIONS[i], "answer": answers[i]}
            for i in range(len(QUESTIONS))
        ],
    }


def _write_final_decision(results: Dict[str, Any], out_dir: Path) -> None:
    decision = _answer_questions(results)
    grade = decision["grade"]
    reason = decision["reason"]

    lines = ["# Final Multi-Topology Decision", ""]
    lines.append(f"**Classification:** {grade}")
    lines.append(f"**Reason:** {reason}")
    lines.append("")
    lines.append("## Ten evaluation questions")
    for q in decision["questions"]:
        lines.append(f"- **{q['question']}** {q['answer']}")
    lines.append("")
    lines.append("## Decision")
    lines.append(f"- Launch full v1.35: **{'only if pilot reaches A' if grade == 'A' else 'no'}**")
    lines.append(f"- Launch v1.35 diagnostic pilot: **{'yes' if grade in ('A', 'B', 'C', 'D') else 'no'}**")
    lines.append("- Topology-diverse v1.3 retraining: recommended if grade is C/D/E and at least some topologies show promise.")

    (out_dir / "FINAL_DECISION.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "FINAL_DECISION.json").write_text(json.dumps(decision, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Status
# ---------------------------------------------------------------------------

def _write_status(out_dir: Path, phase_results: Optional[List[Dict[str, Any]]] = None) -> None:
    lines = ["# Status: v13 Strict Multi-Topology C-side Evaluation", ""]
    lines.append("## Completed deliverables")
    deliverables = [
        "MULTITOPOLOGY_RESULTS.md/.json",
        "PAIRED_STATISTICS.md/.json",
        "FAILURE_DECOMPOSITION.md/.json",
        "ACTION_DISTRIBUTION_AUDIT.md/.json",
        "LATENCY_AUDIT.md/.json",
        "FULL_MATRIX_COMPLETENESS.md/.json",
        "PPO_C_TABLE.md",
        "DF_C_TABLE.md",
        "C_SIDE_SENSITIVITY.md",
        "FINAL_DECISION.md/.json",
        "STATUS.md/.json",
    ]
    for name in deliverables:
        lines.append(f"- {name}")
    if phase_results:
        done = sum(1 for r in phase_results if r.get("status") == "done")
        failed = sum(1 for r in phase_results if r.get("status") == "failed")
        lines.append(f"- Tasks: {done} done, {failed} failed")

    status_payload = {
        "phase": "analysis",
        "deliverables": deliverables,
        "phase_results": phase_results,
    }

    (out_dir / "STATUS.md").write_text("\n".join(lines), encoding="utf-8")
    (out_dir / "STATUS.json").write_text(json.dumps(status_payload, indent=2, default=str), encoding="utf-8")


# ---------------------------------------------------------------------------
# Synthetic / self-test helpers
# ---------------------------------------------------------------------------

def _make_synthetic_row(
    topology: str,
    seed: int,
    c_mode: str,
    r_mode: str,
    blocking_rate: float,
    trace_hash: str,
    config_hash: str,
) -> Dict[str, Any]:
    """Create a minimal valid result row for unit testing the analyzer."""
    return {
        "schema_version": SCHEMA_VERSION,
        "topology": topology,
        "seed": seed,
        "c_mode": c_mode,
        "r_mode": r_mode,
        "method_name": f"{c_mode}+{r_mode}",
        "total": 1000,
        "admitted": 900,
        "blocked": 100,
        "r_decisions": 900,
        "c_no_valid_action": 10,
        "r_no_valid_action": 10,
        "blocking_rate": blocking_rate,
        "server_overload": 30,
        "overload_rate": 0.03,
        "no_suitable_block": 50,
        "nsb_rate": 0.05,
        "other_failure": 10,
        "other_rate": 0.01,
        "failure_reason_distribution": {
            "c_no_valid_action": 10,
            "r_no_valid_action": 10,
            "no_suitable_block": 50,
            "server_overload": 30,
        },
        "avg_delay_ms": 10.0,
        "delay_p95_ms": 20.0,
        "avg_fs": 3.0,
        "avg_decision_ms": 1.0,
        "decision_p95_ms": 2.0,
        "avg_policy_decision_ms": 0.5,
        "avg_audit_decision_ms": 0.5,
        "avg_path_length_km": 100.0,
        "avg_hop_count": 3.0,
        "ppo_agreement_rate": 0.9,
        "ppo_agreement_numerator": 810,
        "ppo_agreement_denominator": 900,
        "mod_distribution": {"QPSK": 0.5, "16QAM": 0.5},
        "path_idx_distribution": {"0": 0.5, "1": 0.5},
        "block_start_distribution": {"0": 1.0},
        "required_fs_distribution": {"2": 1.0},
        "hop_count_distribution": {"3": 1.0},
        "path_length_km_distribution": {"100": 1.0},
        "split_distribution": {"0": 1.0},
        "server_distribution": {"0": 1.0},
        "split_server_joint_distribution": {"(0, 0)": 1.0},
        "request_trace_hash": trace_hash,
        "config_hash": config_hash,
        "code_hash": "codehash",
        "checkpoint_sha256s": {
            "agent_r": "sha1",
            "strict_v13": "sha2",
            "old_v13": "sha3",
            "ppo_c": "sha4",
        },
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(phase_results: Optional[List[Dict[str, Any]]] = None) -> int:
    _test_permutation_and_holm()
    out_dir = _base_dir()
    out_dir.mkdir(parents=True, exist_ok=True)

    data, integrity = _load_all_results(smoke=False)
    if not data:
        print("[analyze] No full results found.")
        return 1

    results = analyze(data, integrity)

    _write_main_report(results, out_dir)
    _write_paired_statistics(results, out_dir)
    _write_failure_decomposition(results, out_dir)
    _write_action_distribution_audit(results, out_dir)
    _write_latency_audit(results, out_dir)
    _write_matrix_completeness(results, out_dir)
    _write_tables(results, out_dir)
    _write_final_decision(results, out_dir)
    _write_status(out_dir, phase_results)

    print(f"[analyze] Wrote reports to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
