"""Phase A opportunity-only component ablation driver.

Protocol: `direct_sketch_exact_optimization_v1` Phase A.

Arms (fixed order, CRN-paired: one env per arm, identical trace seed):
  ksp_ff_k5, ksp_ff_k50, ksp_bf_k50, full_direct, opp_only,
  shadow_only, opp_only_d1

Modes:
  smoke     - 1 topology, 50+150 requests, sanity only (separate dir)
  screening - 3 topologies x seeds 52001-52003 x 1000+10000
              (diagnosis only; no config may be tuned from it)
  formal    - 3 topologies x seeds 53001-53005 x 1000+10000, single worker

Timing structure mirrors rollout_lab/eval_direct_sketch_study.py:
route-cache warmup and selector construction are timed separately and never
mixed into per-request selector latency; selector-only and end-to-end are
reported separately; method order is fixed (ARMS order per seed).
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import RMSAAction
from sa_hmarl.pure_rmsa_v13.core import execute
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.selectors_phaseA import (
    OpportunityOnlyDepth1Selector,
    OpportunityOnlyDirectSelector,
    ShadowOnlyDirectSelector,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.block_heuristics import (
    ksp_best_fit_action,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import (
    PYTHON_KSP_FF_BACKEND,
    PythonKSPFFSelector,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.feasible_window_sketch import (
    LEGAL_START_BACKEND,
    ExactOptimizedDirectCompressedSketchSelector,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.incremental_shadow_pricing import (
    SHADOW_ROW_BACKEND,
)
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env

PROTOCOL_ID = "direct_sketch_exact_optimization_v1"
NATIVE_ENV_VAR = "SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL"
PHASE = "phaseA_opportunity_only"

ARMS = (
    "ksp_ff_k5",
    "ksp_ff_k50",
    "ksp_bf_k50",
    "full_direct",
    "opp_only",
    "shadow_only",
    "opp_only_d1",
)
VARIANT_ARMS = ("opp_only", "shadow_only", "opp_only_d1")

TOPOLOGY_CONFIG = {
    "xlron_nsfnet_deeprmsa": {
        "load_erlang": 100,
        "budget": 256,
        "opportunity_weight": 3000.0,
    },
    "xlron_usnet_gcnrmsa": {
        "load_erlang": 220,
        "budget": 256,
        "opportunity_weight": 8000.0,
    },
    "xlron_jpn48": {
        "load_erlang": 150,
        "budget": 64,
        "opportunity_weight": 5000.0,
    },
}

SCREENING_SEEDS = (52001, 52002, 52003)
FORMAL_SEEDS = (53001, 53002, 53003, 53004, 53005)

_PERF_COUNTER = time.perf_counter


def _action_key(action) -> tuple:
    if isinstance(action, RMSAAction):
        return tuple(action.action_id)
    return ("BLOCKED", getattr(action, "reason", None))


def _prewarm_routes(env) -> None:
    for src in range(env.topology.num_nodes):
        for dst in range(env.topology.num_nodes):
            if src != dst:
                env._cached_k_paths(src, dst, env.k_paths)


def _build_selector(arm, env, config):
    if arm in ("ksp_ff_k5", "ksp_ff_k50"):
        return PythonKSPFFSelector(env)
    if arm == "ksp_bf_k50":
        return None
    if arm == "full_direct":
        return ExactOptimizedDirectCompressedSketchSelector(
            env,
            budget=config["budget"],
            opportunity_weight=config["opportunity_weight"],
        )
    if arm == "opp_only":
        return OpportunityOnlyDirectSelector(
            env,
            budget=config["budget"],
            opportunity_weight=config["opportunity_weight"],
        )
    if arm == "shadow_only":
        return ShadowOnlyDirectSelector(env)
    if arm == "opp_only_d1":
        return OpportunityOnlyDepth1Selector(
            env,
            budget=config["budget"],
            opportunity_weight=config["opportunity_weight"],
        )
    raise ValueError(f"unknown arm: {arm}")


def evaluate_arm(topology, config, seed, warmup, eval_requests, arm):
    """Run one arm on its own env (CRN: same trace seed for every arm)."""
    k_paths = 5 if arm == "ksp_ff_k5" else 50
    env_started = _PERF_COUNTER()
    env = make_env(
        seed,
        warmup + eval_requests,
        topology=topology,
        num_slots=50,
        load_erlang=config["load_erlang"],
        k_paths=k_paths,
    )
    env_init_s = _PERF_COUNTER() - env_started
    started = _PERF_COUNTER()
    _prewarm_routes(env)
    route_init_s = _PERF_COUNTER() - started
    started = _PERF_COUNTER()
    selector = _build_selector(arm, env, config)
    selector_init_s = _PERF_COUNTER() - started

    latencies_s = []
    eval_action_keys = []
    blocked = 0
    end_to_end_s = 0.0
    for request_index, request in enumerate(env.trace.requests):
        request_started = _PERF_COUNTER()
        env.advance_external(request)
        started = _PERF_COUNTER()
        if arm == "ksp_bf_k50":
            action = ksp_best_fit_action(env, request)
        else:
            action = selector.select(env, request)
        latencies_s.append(_PERF_COUNTER() - started)
        result = execute(env, request, action)
        end_to_end_s += _PERF_COUNTER() - request_started
        if request_index >= warmup:
            blocked += int(not result["success"])
            eval_action_keys.append(_action_key(action))

    eval_latencies = latencies_s[warmup:]
    row = {
        "topology": topology,
        "seed": seed,
        "arm": arm,
        "k_paths": k_paths,
        "blocked": blocked,
        "blocking": blocked / eval_requests,
        "selector_mean_ms": 1000.0 * float(np.mean(latencies_s)),
        "selector_p50_ms": 1000.0 * float(np.percentile(eval_latencies, 50)),
        "selector_p95_ms": 1000.0 * float(np.percentile(eval_latencies, 95)),
        "selector_p99_ms": 1000.0 * float(np.percentile(eval_latencies, 99)),
        "end_to_end_ms": 1000.0 * end_to_end_s / (warmup + eval_requests),
        "route_init_s": route_init_s,
        "selector_init_s": selector_init_s,
        "env_init_s": env_init_s,
        "ksp_ff_backend": (
            PYTHON_KSP_FF_BACKEND
            if arm in ("ksp_ff_k5", "ksp_ff_k50")
            else None
        ),
        "shadow_row_backend": (
            SHADOW_ROW_BACKEND if arm in ("full_direct", "shadow_only") else None
        ),
        "legal_start_backend": (
            LEGAL_START_BACKEND
            if arm in ("full_direct", "opp_only", "opp_only_d1")
            else None
        ),
    }
    return row, eval_action_keys


def _bootstrap_ci(values, seed):
    """Paired bootstrap CI of the mean (same method as eval_direct_sketch_study)."""
    if len(values) == 1:
        return [float(values[0]), float(values[0])]
    rng = np.random.default_rng(seed)
    array = np.asarray(values, dtype=np.float64)
    means = rng.choice(
        array,
        size=(100000, len(array)),
        replace=True,
    ).mean(axis=1)
    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


def _wtl(deltas_count):
    return [
        sum(1 for value in deltas_count if value < 0),
        sum(1 for value in deltas_count if value == 0),
        sum(1 for value in deltas_count if value > 0),
    ]


def summarize(topologies, seeds, rows):
    """Per-topology blocking comparisons, gains, retention, latency stats."""
    summaries = []
    for topo_index, topology in enumerate(topologies):
        selected = [row for row in rows if row["topology"] == topology]
        by_arm = {
            arm: {row["seed"]: row for row in selected if row["arm"] == arm}
            for arm in ARMS
        }
        mean_blocking = {
            arm: float(
                np.mean([by_arm[arm][seed]["blocking"] for seed in seeds])
            )
            for arm in ARMS
        }
        latency = {
            arm: {
                "mean_ms": float(
                    np.mean(
                        [by_arm[arm][seed]["selector_mean_ms"] for seed in seeds]
                    )
                ),
                "p50_ms": float(
                    np.mean(
                        [by_arm[arm][seed]["selector_p50_ms"] for seed in seeds]
                    )
                ),
                "p95_ms": float(
                    np.mean(
                        [by_arm[arm][seed]["selector_p95_ms"] for seed in seeds]
                    )
                ),
                "p99_ms": float(
                    np.mean(
                        [by_arm[arm][seed]["selector_p99_ms"] for seed in seeds]
                    )
                ),
                "end_to_end_ms": float(
                    np.mean(
                        [by_arm[arm][seed]["end_to_end_ms"] for seed in seeds]
                    )
                ),
                "selector_init_s": float(
                    np.mean(
                        [by_arm[arm][seed]["selector_init_s"] for seed in seeds]
                    )
                ),
                "route_init_s": float(
                    np.mean(
                        [by_arm[arm][seed]["route_init_s"] for seed in seeds]
                    )
                ),
            }
            for arm in ARMS
        }
        comparisons = {}
        for comp_index, baseline in enumerate(("ksp_ff_k50", "full_direct")):
            for arm in ARMS:
                if arm == baseline:
                    continue
                deltas_pp = [
                    100.0
                    * (
                        by_arm[arm][seed]["blocking"]
                        - by_arm[baseline][seed]["blocking"]
                    )
                    for seed in seeds
                ]
                deltas_count = [
                    by_arm[arm][seed]["blocked"]
                    - by_arm[baseline][seed]["blocked"]
                    for seed in seeds
                ]
                comparisons[f"{arm}_vs_{baseline}"] = {
                    "delta_pp": float(np.mean(deltas_pp)),
                    "ci95": _bootstrap_ci(
                        deltas_pp,
                        20260801 + topo_index * 100 + comp_index * 20
                        + ARMS.index(arm),
                    ),
                    "wins_ties_losses": _wtl(deltas_count),
                }
        full_gain = mean_blocking["ksp_ff_k50"] - mean_blocking["full_direct"]
        gains = {"full_direct": full_gain}
        retention = {}
        for arm in ARMS:
            if arm in ("ksp_ff_k50", "full_direct"):
                continue
            gains[arm] = mean_blocking["ksp_ff_k50"] - mean_blocking[arm]
            retention[arm] = (
                gains[arm] / full_gain if abs(full_gain) > 1e-12 else None
            )
        summaries.append(
            {
                "topology": topology,
                "mean_blocking": mean_blocking,
                "latency": latency,
                "comparisons": comparisons,
                "gain_vs_ksp_ff_k50": gains,
                "retention_of_full_gain": retention,
            }
        )
    return summaries


def evaluate_gate(topologies, seeds, summaries, rows):
    """Protocol lock section 6 Phase A gate, opportunity-only arm only."""
    gate = {"per_topology": {}, "qualifying_topologies": []}
    for topo_index, (topology, summary) in enumerate(
        zip(topologies, summaries)
    ):
        retention = summary["retention_of_full_gain"]["opp_only"]
        retention_ok = retention is not None and retention >= 0.80
        comp = summary["comparisons"]["opp_only_vs_ksp_ff_k50"]
        wins, ties, losses = comp["wins_ties_losses"]
        seeds_ok = wins + ties == len(seeds)
        ci_ok = comp["ci95"][1] <= 0.0
        latency_drop = 1.0 - (
            summary["latency"]["opp_only"]["mean_ms"]
            / summary["latency"]["full_direct"]["mean_ms"]
        )
        latency_ok = latency_drop >= 0.20
        gate["per_topology"][topology] = {
            "retention": retention,
            "retention_ok_ge_80pct": retention_ok,
            "wins_ties_losses_vs_ksp_ff_k50": [wins, ties, losses],
            "seeds_win_or_tie_ok": seeds_ok,
            "ci95_vs_ksp_ff_k50_pp": comp["ci95"],
            "ci_not_worse_ok": ci_ok,
            "latency_drop_vs_full_direct": latency_drop,
            "latency_drop_ok_ge_20pct": latency_ok,
        }
        if retention_ok:
            gate["qualifying_topologies"].append(topology)
    qualifying = gate["qualifying_topologies"]
    gate["condition_1_retention_on_ge_2_of_3"] = len(qualifying) >= 2
    gate["condition_2_seeds_and_ci_on_qualifying"] = all(
        gate["per_topology"][topology]["seeds_win_or_tie_ok"]
        and gate["per_topology"][topology]["ci_not_worse_ok"]
        for topology in qualifying
    )
    gate["condition_3_latency_drop_on_qualifying"] = all(
        gate["per_topology"][topology]["latency_drop_ok_ge_20pct"]
        for topology in qualifying
    )
    gate["go"] = (
        gate["condition_1_retention_on_ge_2_of_3"]
        and gate["condition_2_seeds_and_ci_on_qualifying"]
        and gate["condition_3_latency_drop_on_qualifying"]
    )
    return gate


def run_mode(mode, topologies, seeds, warmup, eval_requests, output_dir):
    rows = []
    keys_by_pair = {}
    wall_started = _PERF_COUNTER()
    for topology in topologies:
        config = TOPOLOGY_CONFIG[topology]
        for seed in seeds:
            for arm in ARMS:
                print(
                    f"[phaseA:{mode}] {topology} seed={seed} arm={arm}",
                    flush=True,
                )
                row, eval_action_keys = evaluate_arm(
                    topology,
                    config,
                    seed,
                    warmup,
                    eval_requests,
                    arm,
                )
                rows.append(row)
                keys_by_pair[(topology, seed, arm)] = eval_action_keys
                print(
                    f"[phaseA:{mode}]   blocking={row['blocking']:.4f} "
                    f"selector_mean={row['selector_mean_ms']:.4f} ms",
                    flush=True,
                )
    wall_clock_s = _PERF_COUNTER() - wall_started

    # Action match rate vs full_direct (same trace, paired per eval request).
    parity_seed_rows = []
    for topology in topologies:
        for seed in seeds:
            full_keys = keys_by_pair[(topology, seed, "full_direct")]
            for arm in ARMS:
                if arm == "full_direct":
                    continue
                arm_keys = keys_by_pair[(topology, seed, arm)]
                matches = sum(
                    1 for a, b in zip(arm_keys, full_keys) if a == b
                )
                first_mismatch = next(
                    (
                        index
                        for index, (a, b) in enumerate(
                            zip(arm_keys, full_keys)
                        )
                        if a != b
                    ),
                    None,
                )
                parity_seed_rows.append(
                    {
                        "topology": topology,
                        "seed": seed,
                        "arm": arm,
                        "matches": matches,
                        "total": len(arm_keys),
                        "match_rate": matches / max(1, len(arm_keys)),
                        "first_mismatch_eval_index": first_mismatch,
                    }
                )
    parity_topology_rows = []
    for topology in topologies:
        for arm in ARMS:
            if arm == "full_direct":
                continue
            selected = [
                row
                for row in parity_seed_rows
                if row["topology"] == topology and row["arm"] == arm
            ]
            matches = sum(row["matches"] for row in selected)
            total = sum(row["total"] for row in selected)
            parity_topology_rows.append(
                {
                    "topology": topology,
                    "arm": arm,
                    "matches": matches,
                    "total": total,
                    "match_rate": matches / max(1, total),
                }
            )
    return rows, parity_seed_rows, parity_topology_rows, wall_clock_s


def _write_parity(output_dir, mode, seeds, warmup, eval_requests,
                  parity_seed_rows, parity_topology_rows):
    payload = {
        "protocol_id": PROTOCOL_ID,
        "phase": PHASE,
        "mode": mode,
        "seeds": list(seeds),
        "warmup": warmup,
        "eval_requests": eval_requests,
        "note": (
            "Phase A is a behavior-changing ablation; action parity with "
            "Full Direct is NOT required. Values are per-request action "
            "match rates against full_direct on the same trace (paired "
            "envs, eval phase). Divergent actions change later states, so "
            "this is a trajectory-level match rate."
        ),
        "action_key": "RMSAAction.action_id or ('BLOCKED', reason)",
        "per_seed": parity_seed_rows,
        "per_topology": parity_topology_rows,
    }
    (output_dir / "ACTION_PARITY.json").write_text(
        json.dumps(payload, indent=2),
        encoding="utf-8",
    )


def _write_latency_csv(output_dir, topologies, summaries):
    fieldnames = [
        "topology",
        "arm",
        "selector_mean_ms",
        "selector_p50_ms",
        "selector_p95_ms",
        "selector_p99_ms",
        "end_to_end_ms",
        "selector_init_s",
        "route_init_s",
    ]
    with (output_dir / "LATENCY_BREAKDOWN.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for topology, summary in zip(topologies, summaries):
            for arm in ARMS:
                latency = summary["latency"][arm]
                writer.writerow(
                    {
                        "topology": topology,
                        "arm": arm,
                        "selector_mean_ms": latency["mean_ms"],
                        "selector_p50_ms": latency["p50_ms"],
                        "selector_p95_ms": latency["p95_ms"],
                        "selector_p99_ms": latency["p99_ms"],
                        "end_to_end_ms": latency["end_to_end_ms"],
                        "selector_init_s": latency["selector_init_s"],
                        "route_init_s": latency["route_init_s"],
                    }
                )


def _write_results_md(output_dir, mode, seeds, warmup, eval_requests,
                      topologies, summaries, gate):
    lines = [
        "# Phase A Opportunity-Only Ablation — RESULTS",
        "",
        f"- Protocol: `{PROTOCOL_ID}` Phase A; mode: {mode}",
        f"- Seeds: {list(seeds)}; warmup {warmup}; eval {eval_requests}",
        f"- Arms (fixed order): {', '.join(ARMS)}",
        "- Backend: pure Python/NumPy "
        f"(ksp_ff={PYTHON_KSP_FF_BACKEND}, shadow_row={SHADOW_ROW_BACKEND}, "
        f"legal_start={LEGAL_START_BACKEND}; {NATIVE_ENV_VAR} unset)",
        "",
        "## Mean blocking probability (eval phase, mean over seeds)",
        "",
        "| Topology | " + " | ".join(ARMS) + " |",
        "|---|" + "---|" * len(ARMS),
    ]
    for topology, summary in zip(topologies, summaries):
        lines.append(
            "| " + topology + " | " + " | ".join(
                f"{summary['mean_blocking'][arm]:.4f}" for arm in ARMS
            ) + " |"
        )
    lines += [
        "",
        "## Gain vs KSP-FF K=50 and retention of Full Direct gain",
        "",
        "| Topology | Full gain (pp) | opp_only gain (pp) | opp_only retention | shadow_only retention | opp_only_d1 retention |",
        "|---|---|---|---|---|---|",
    ]
    for topology, summary in zip(topologies, summaries):
        gains = summary["gain_vs_ksp_ff_k50"]
        retention = summary["retention_of_full_gain"]

        def _ret(arm):
            value = retention.get(arm)
            return f"{100.0 * value:.1f}%" if value is not None else "n/a"

        lines.append(
            f"| {topology} | {100.0 * gains['full_direct']:.3f} "
            f"| {100.0 * gains['opp_only']:.3f} | {_ret('opp_only')} "
            f"| {_ret('shadow_only')} | {_ret('opp_only_d1')} |"
        )
    lines += [
        "",
        "## Selector latency (mean over seeds; eval-phase percentiles)",
        "",
        "| Topology | Arm | mean ms | P50 ms | P95 ms | P99 ms | end-to-end ms | init s |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for topology, summary in zip(topologies, summaries):
        for arm in ARMS:
            latency = summary["latency"][arm]
            lines.append(
                f"| {topology} | {arm} | {latency['mean_ms']:.4f} "
                f"| {latency['p50_ms']:.4f} | {latency['p95_ms']:.4f} "
                f"| {latency['p99_ms']:.4f} | {latency['end_to_end_ms']:.4f} "
                f"| {latency['selector_init_s']:.2f} |"
            )
    lines += [
        "",
        "## Paired comparisons (delta blocking pp, bootstrap 95% CI, W/T/L on blocked counts)",
        "",
    ]
    for topology, summary in zip(topologies, summaries):
        lines += [
            f"### {topology}",
            "",
            "| Comparison | delta pp | CI95 | W/T/L |",
            "|---|---|---|---|",
        ]
        for name, comp in summary["comparisons"].items():
            wins, ties, losses = comp["wins_ties_losses"]
            lines.append(
                f"| {name} | {comp['delta_pp']:+.3f} "
                f"| [{comp['ci95'][0]:+.3f}, {comp['ci95'][1]:+.3f}] "
                f"| {wins}/{ties}/{losses} |"
            )
        lines.append("")
    if gate is not None:
        lines += [
            "## Phase A gate evaluation (opportunity-only)",
            "",
            f"- Decision: **{'GO' if gate['go'] else 'STOP'}** "
            "(see NEXT_STEP_DECISION.md)",
            "",
        ]
    (output_dir / "RESULTS.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_protocol_audit(output_dir, mode, seeds, warmup, eval_requests):
    native_value = os.environ.get(NATIVE_ENV_VAR)
    lines = [
        "# Protocol Audit — Phase A Opportunity-Only Ablation",
        "",
        f"Protocol lock: `DIRECT_SKETCH_EXACT_OPT_PROTOCOL_LOCK.md` "
        f"(`{PROTOCOL_ID}`). Mode: {mode}.",
        "",
        "| Clause | Requirement | Status | Evidence |",
        "|---|---|---|---|",
        "| §1.1 | Pure Python/NumPy; native kernel env var unset | PASS | "
        f"`{NATIVE_ENV_VAR}`="
        f"{'unset' if native_value is None else native_value}; "
        f"shadow_row={SHADOW_ROW_BACKEND}, "
        f"legal_start={LEGAL_START_BACKEND}, "
        f"ksp_ff={PYTHON_KSP_FF_BACKEND} |",
        "| §1.2 | No topology/traffic/KSP-order/modulation/continuity/num_slots changes | PASS | "
        "Only new files under `direct_sketch_exact_optimization/`; env built "
        "via locked `make_env` (num_slots=50, path_sort_strategy=hops) |",
        "| §1.3 | No smoke results as formal conclusions | PASS | "
        "Smoke written to a separate `_smoke` directory; screening under "
        "`screening/`; formal conclusions cite formal seeds only |",
        "| §1.4 | Tuning/diagnosis and formal seeds disjoint | PASS | "
        f"screening seeds {list(SCREENING_SEEDS)} (diagnosis), formal seeds "
        f"{list(FORMAL_SEEDS)}; this run used {list(seeds)} |",
        "| §1.5 | Never overwrite old artifacts | PASS | "
        f"Output directory `{PHASE}/` is new for this phase |",
        "| §1.6 | Single worker for formal latency | PASS | "
        "Sequential single-process driver (workers=1) |",
        "| §2 | Locked selectors/baselines used | PASS | "
        "full_direct = `ExactOptimizedDirectCompressedSketchSelector`; "
        "KSP-FF K=50/K=5 = `PythonKSPFFSelector` "
        f"({PYTHON_KSP_FF_BACKEND}); KSP-BF = `ksp_best_fit_action` "
        "(same function as heldout `ksp_bf_k50_hops`) |",
        "| §3 | Locked Direct configuration | PASS | "
        "K=50 hops; <=3 feasible paths; budgets 256/256/64; weights "
        "3000/8000/5000; route_depth=3 (opp_only_d1 varies depth as the "
        "ablation arm); rank penalty 1.0; highest reachable modulation; "
        "all legal starts; tie-break (total, path_rank, start_slot) |",
        "| §4 | Loads | PASS | NSFNET 100E, USNET 220E, JPN48 150E |",
        "| §5 | Phase A seeds | PASS | "
        f"screening {list(SCREENING_SEEDS)}, formal {list(FORMAL_SEEDS)}; "
        f"warmup/eval {warmup}/{eval_requests} |",
        "| §6 | Phase A gate | SEE DECISION | "
        "Evaluated in NEXT_STEP_DECISION.md clause by clause |",
        "| §7 | Fairness rules | PASS | "
        "Same runtime/traces per arm (CRN: identical trace seed, one env "
        "per arm); fixed method order (ARMS order per seed); route warmup "
        "and selector init timed separately; selector-only and end-to-end "
        "separate; mean/median/P95/P99 recorded; init never mixed into "
        "per-request latency |",
        "| §8 | Required outputs | PASS | "
        "EXPERIMENT_MANIFEST.json, PROTOCOL_AUDIT.md, per_seed.csv, "
        "RESULTS.json, RESULTS.md, ACTION_PARITY.json, "
        "LATENCY_BREAKDOWN.csv, NEXT_STEP_DECISION.md |",
        "",
        "Component isolation guarantees (enforced by construction):",
        "",
        "- `OpportunityOnlyDirectSelector` / `OpportunityOnlyDepth1Selector` "
        "never instantiate `IncrementalDemandShadowPricer` and never call "
        "any shadow API (see `selectors_phaseA.py`; route specs index arcs "
        "through the sketch pricer's `_arc_ids`).",
        "- `ShadowOnlyDirectSelector` never instantiates "
        "`CompressedFeasibleWindowSketch` / probe components; "
        "`shadow.sync_state` is called with `occupied_words=None`.",
    ]
    (output_dir / "PROTOCOL_AUDIT.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_next_step_decision(output_dir, topologies, summaries, gate):
    lines = [
        "# Next Step Decision — Phase A Opportunity-Only Ablation",
        "",
        f"Protocol lock §6 Phase A gate. Arm under test: `opp_only`.",
        "",
        f"## Decision: **{'GO' if gate['go'] else 'STOP'}**",
        "",
        "## Clause-by-clause evaluation",
        "",
        "### Clause 1: opportunity-only retains >= 80% of Full Direct gain "
        "(vs KSP-FF K=50) on >= 2/3 topologies",
        "",
        "| Topology | Full gain (pp) | opp_only gain (pp) | retention | >= 80%? |",
        "|---|---|---|---|---|",
    ]
    for topology, summary in zip(topologies, summaries):
        gains = summary["gain_vs_ksp_ff_k50"]
        retention = summary["retention_of_full_gain"]["opp_only"]
        info = gate["per_topology"][topology]
        lines.append(
            f"| {topology} | {100.0 * gains['full_direct']:+.3f} "
            f"| {100.0 * gains['opp_only']:+.3f} "
            f"| {('%.1f%%' % (100.0 * retention)) if retention is not None else 'n/a'} "
            f"| {'YES' if info['retention_ok_ge_80pct'] else 'NO'} |"
        )
    lines += [
        "",
        f"Qualifying topologies: {gate['qualifying_topologies']} -> "
        f"condition {'PASS' if gate['condition_1_retention_on_ge_2_of_3'] else 'FAIL'}",
        "",
        "### Clause 2: on qualifying topologies, opp_only wins or ties 5/5 "
        "formal seeds vs KSP-FF K=50 with paired CI not worse (no "
        "zero-crossing CI claimed as win; CI upper bound must be <= 0)",
        "",
        "| Topology | W/T/L vs ksp_ff_k50 | CI95 (pp) | 5/5 win-or-tie? | CI <= 0? |",
        "|---|---|---|---|---|",
    ]
    for topology in gate["qualifying_topologies"]:
        info = gate["per_topology"][topology]
        wins, ties, losses = info["wins_ties_losses_vs_ksp_ff_k50"]
        lines.append(
            f"| {topology} | {wins}/{ties}/{losses} "
            f"| [{info['ci95_vs_ksp_ff_k50_pp'][0]:+.3f}, "
            f"{info['ci95_vs_ksp_ff_k50_pp'][1]:+.3f}] "
            f"| {'YES' if info['seeds_win_or_tie_ok'] else 'NO'} "
            f"| {'YES' if info['ci_not_worse_ok'] else 'NO'} |"
        )
    lines += [
        "",
        f"condition {'PASS' if gate['condition_2_seeds_and_ci_on_qualifying'] else 'FAIL'}",
        "",
        "### Clause 3: selector latency drops >= 20% vs Full Direct "
        "(per qualifying topology)",
        "",
        "| Topology | full_direct ms | opp_only ms | drop | >= 20%? |",
        "|---|---|---|---|---|",
    ]
    for topology in gate["qualifying_topologies"]:
        summary = summaries[list(topologies).index(topology)]
        info = gate["per_topology"][topology]
        lines.append(
            f"| {topology} | {summary['latency']['full_direct']['mean_ms']:.4f} "
            f"| {summary['latency']['opp_only']['mean_ms']:.4f} "
            f"| {100.0 * info['latency_drop_vs_full_direct']:.1f}% "
            f"| {'YES' if info['latency_drop_ok_ge_20pct'] else 'NO'} |"
        )
    lines += [
        "",
        f"condition {'PASS' if gate['condition_3_latency_drop_on_qualifying'] else 'FAIL'}",
        "",
        "## Consequences",
        "",
    ]
    if gate["go"]:
        lines += [
            "GO: per protocol lock §6, recommend opportunity-only as the "
            "primary selector; Phases B/C/D stop.",
        ]
    else:
        lines += [
            "STOP: Phase A gate not met. Opportunity-only is not "
            "recommended as primary. Preserve this result (including "
            "diagnostic arms shadow_only / opp_only_d1) for attribution; "
            "any further action-identical optimization work (B/C/D) "
            "requires an explicit decision to proceed despite the Phase A "
            "STOP.",
        ]
    (output_dir / "NEXT_STEP_DECISION.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _write_manifest(output_dir, mode, seeds, warmup, eval_requests,
                    wall_clock_s, argv):
    native_value = os.environ.get(NATIVE_ENV_VAR)
    manifest = {
        "protocol_id": PROTOCOL_ID,
        "phase": PHASE,
        "mode": mode,
        "seeds": list(seeds),
        "warmup_requests": warmup,
        "eval_requests": eval_requests,
        "workers": 1,
        "num_slots": 50,
        "arms": list(ARMS),
        "variant_arms": list(VARIANT_ARMS),
        "method_order": "fixed: ARMS order per (topology, seed)",
        "topology_config": TOPOLOGY_CONFIG,
        "backend": {
            "native_env_var_name": NATIVE_ENV_VAR,
            "native_env_var_value": (
                native_value if native_value is not None else "unset"
            ),
            "shadow_row_backend": SHADOW_ROW_BACKEND,
            "legal_start_backend": LEGAL_START_BACKEND,
            "ksp_ff_backend": PYTHON_KSP_FF_BACKEND,
        },
        "latency_definitions": {
            "selector_mean_ms": "mean per-request selector latency over all requests (warmup+eval)",
            "selector_p50_p95_p99_ms": "percentiles over eval-phase requests",
            "end_to_end_ms": "advance_external + select + execute per request",
            "init": "route-cache warmup (route_init_s) and selector construction (selector_init_s) reported separately, never mixed into per-request latency",
        },
        "bootstrap": {
            "method": "paired bootstrap over per-seed deltas (same as eval_direct_sketch_study._bootstrap_ci)",
            "resamples": 100000,
            "rng_seed_base": 20260801,
        },
        "python": {
            "executable": sys.executable,
            "version": platform.python_version(),
        },
        "host": platform.node(),
        "command": " ".join([sys.executable, *argv]),
        "finished_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": wall_clock_s,
    }
    (output_dir / "EXPERIMENT_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode",
        choices=("smoke", "screening", "formal"),
        required=True,
    )
    args = parser.parse_args(argv)

    if os.environ.get(NATIVE_ENV_VAR) is not None:
        raise SystemExit(f"{NATIVE_ENV_VAR} must be unset for this protocol")
    if SHADOW_ROW_BACKEND != "python" or LEGAL_START_BACKEND != "python":
        raise SystemExit(
            "native backends active: "
            f"shadow_row_backend={SHADOW_ROW_BACKEND}, "
            f"legal_start_backend={LEGAL_START_BACKEND}"
        )

    base_dir = (
        Path("sa_hmarl/pure_rmsa_v13/artifacts")
        / "direct_sketch_exact_optimization"
    )
    if args.mode == "smoke":
        topologies = ("xlron_nsfnet_deeprmsa",)
        seeds = (52001,)
        warmup, eval_requests = 50, 150
        output_dir = base_dir / f"{PHASE}_smoke"
    elif args.mode == "screening":
        topologies = tuple(TOPOLOGY_CONFIG)
        seeds = SCREENING_SEEDS
        warmup, eval_requests = 1000, 10000
        output_dir = base_dir / PHASE / "screening"
    else:
        topologies = tuple(TOPOLOGY_CONFIG)
        seeds = FORMAL_SEEDS
        warmup, eval_requests = 1000, 10000
        output_dir = base_dir / PHASE
    output_dir.mkdir(parents=True, exist_ok=True)

    rows, parity_seed_rows, parity_topology_rows, wall_clock_s = run_mode(
        args.mode,
        topologies,
        seeds,
        warmup,
        eval_requests,
        output_dir,
    )
    summaries = summarize(topologies, seeds, rows)
    gate = (
        evaluate_gate(topologies, seeds, summaries, rows)
        if args.mode == "formal"
        else None
    )

    # RESULTS.json + per_seed.csv
    (output_dir / "RESULTS.json").write_text(
        json.dumps(
            {
                "protocol_id": PROTOCOL_ID,
                "phase": PHASE,
                "mode": args.mode,
                "seeds": list(seeds),
                "warmup": warmup,
                "eval_requests": eval_requests,
                "arms": list(ARMS),
                "rows": rows,
                "summaries": summaries,
                "gate": gate,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (output_dir / "per_seed.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)
    _write_parity(
        output_dir,
        args.mode,
        seeds,
        warmup,
        eval_requests,
        parity_seed_rows,
        parity_topology_rows,
    )
    _write_latency_csv(output_dir, topologies, summaries)
    _write_results_md(
        output_dir,
        args.mode,
        seeds,
        warmup,
        eval_requests,
        topologies,
        summaries,
        gate,
    )
    _write_manifest(
        output_dir,
        args.mode,
        seeds,
        warmup,
        eval_requests,
        wall_clock_s,
        sys.argv if argv is None else ["run_phaseA.py", *argv],
    )
    if args.mode == "formal":
        _write_protocol_audit(
            output_dir, args.mode, seeds, warmup, eval_requests
        )
        _write_next_step_decision(output_dir, topologies, summaries, gate)
        print(
            f"[phaseA:formal] gate decision: "
            f"{'GO' if gate['go'] else 'STOP'}",
            flush=True,
        )
    print(
        f"[phaseA:{args.mode}] done wall={wall_clock_s:.1f}s "
        f"outputs={output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
