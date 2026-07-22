#!/usr/bin/env python3
"""Stage A — unit tests for the paper-standard RMSA core.

Verifies the protocol properties pinned in PROTOCOL_LOCK.md against the
upstream DeepRMSA behavioral oracle (deep_rmsa_parity_adapter.py, previously
audited in v135_r_only_standard_and_blocking_diagnosis):

- FS boundaries (inclusive reach, upstream cal_FS formula)
- Direction-independent spectrum (dual-fiber, independent arcs)
- Continuity/contiguity of spectrum blocks
- First-Fit block selection (fuzzed against upstream judge_availability)
- KSP-FF end-to-end parity vs upstream oracle (fuzzed)
- Release order (release-before-arrival, FIFO ties)
- Holding-time truncation (resample ttl == 0 or >= 2*mean)
- Uniform all-OD traffic (all ordered pairs, src != dst)
- KSP path orderings (km vs hops)
- No excluded components (snap24 / masked K=3/M=1 / C-side machinery)

Writes UNIT_TEST_REPORT.md and results/unit_tests.json.
Exit code 0 iff all tests pass.
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np

EXP_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(EXP_DIR))
sys.path.insert(0, str(EXP_DIR.parents[1]))  # sa_hmarl package container
sys.path.insert(0, str(EXP_DIR.parents[0] / "v135_r_only_standard_and_blocking_diagnosis"))

import deep_rmsa_parity_adapter as oracle  # upstream behavioral oracle

from paper_rmsa_core import (
    PaperOpticalNetwork,
    PaperRMSAEnv,
    _topology_edges,
    best_modulation_index,
    contiguous_free_blocks,
    eligible_blocks,
    first_fit_start,
    generate_paper_requests,
    get_candidate_paths,
    ksp_ff_decide,
    load_run_config,
    paper_modulation_registry,
    required_fs,
)

CFG = load_run_config()
NSF = CFG["topologies"]["nsfnet"]["topology_key"]
COST = CFG["topologies"]["cost239"]["topology_key"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _build_oracle_linkmap(topology: str) -> Tuple[Dict[int, Dict[int, Tuple[int, float]]], Dict[Tuple[int, int], int]]:
    """Directed link IDs for the upstream oracle: 2i / 2i+1 per edge."""
    linkmap: Dict[int, Dict[int, Tuple[int, float]]] = {}
    arc_to_link: Dict[Tuple[int, int], int] = {}
    for i, (u, v, length) in enumerate(_topology_edges(topology)):
        u, v = int(u), int(v)
        linkmap.setdefault(u, {})[v] = (2 * i, float(length))
        linkmap.setdefault(v, {})[u] = (2 * i + 1, float(length))
        arc_to_link[(u, v)] = 2 * i
        arc_to_link[(v, u)] = 2 * i + 1
    return linkmap, arc_to_link


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def test_fs_boundaries() -> Dict[str, Any]:
    """Inclusive reach boundaries + upstream cal_FS formula (oracle-checked)."""
    mod_reg = paper_modulation_registry()
    cases = [
        # (bandwidth, path_len, expected_mod, expected_fs per oracle)
        (25, 624.0, "16QAM", None), (25, 625.0, "16QAM", None), (25, 626.0, "8QAM", None),
        (100, 624.0, "16QAM", None), (100, 625.0, "16QAM", None),
        (100, 1249.0, "8QAM", None), (100, 1250.0, "8QAM", None), (100, 1251.0, "QPSK", None),
        (75, 2499.0, "QPSK", None), (75, 2500.0, "QPSK", None), (75, 2501.0, "BPSK", None),
        (100, 9999.0, "BPSK", None), (100, 10000.0, "BPSK", None),
    ]
    details = []
    ok_all = True
    for bw, plen, expected_mod, _ in cases:
        mod_idx = best_modulation_index(plen, mod_reg)
        got_mod = mod_reg.names[mod_idx] if mod_idx is not None else None
        got_fs = required_fs(bw, mod_reg[mod_idx].spectral_efficiency) if mod_idx is not None else None
        oracle_fs = oracle.deeprmsa_fs(bw, plen)
        ok = (got_mod == expected_mod) and (got_fs == oracle_fs)
        ok_all &= ok
        details.append({"bw": bw, "len": plen, "expected_mod": expected_mod,
                        "got_mod": got_mod, "fs": got_fs, "oracle_fs": oracle_fs, "pass": ok})
    # Spot-check exact values from the prior parity report.
    spot = {(25, 624.0): 2, (25, 625.0): 2, (25, 626.0): 2,
            (100, 1249.0): 4, (100, 1250.0): 4, (100, 1251.0): 5,
            (75, 2499.0): 4, (75, 2500.0): 4, (75, 2501.0): 7}
    for (bw, plen), fs_expected in spot.items():
        mod_idx = best_modulation_index(plen, mod_reg)
        got = required_fs(bw, mod_reg[mod_idx].spectral_efficiency)
        ok_all &= (got == fs_expected)
    return {"name": "fs_boundaries", "pass": bool(ok_all), "details": details}


def test_direction_independence() -> Dict[str, Any]:
    """Dual-fiber: counter-directional arcs have independent spectrum."""
    net = PaperOpticalNetwork(NSF, 100)
    assert net.num_directed_arcs == 44, net.num_directed_arcs
    ok = net.allocate([0, 1], 10, 5)
    assert ok
    fwd_occupied = bool(np.all(net.arc_states[(0, 1)][10:15]))
    rev_free = not np.any(net.arc_states[(1, 0)])  # reverse direction untouched
    # A reverse-direction request on the same slots must succeed.
    ok_rev = net.allocate([1, 0], 10, 5)
    # Same-direction overlapping allocation must fail.
    ok_conflict = not net.allocate([0, 1], 12, 4)
    # Same-direction non-overlapping allocation must succeed.
    ok_adjacent = net.allocate([0, 1], 15, 4)
    cost_net = PaperOpticalNetwork(COST, 100)
    details = {
        "nsfnet_directed_arcs": net.num_directed_arcs,
        "cost239_directed_arcs": cost_net.num_directed_arcs,
        "forward_occupied": fwd_occupied,
        "reverse_untouched": rev_free,
        "reverse_allocation_ok": ok_rev,
        "conflict_rejected": ok_conflict,
        "adjacent_ok": ok_adjacent,
    }
    passed = (fwd_occupied and rev_free and ok_rev and ok_conflict and ok_adjacent
              and net.num_directed_arcs == 44 and cost_net.num_directed_arcs == 52)
    return {"name": "direction_independence", "pass": bool(passed), "details": details}


def test_continuity_contiguity() -> Dict[str, Any]:
    """Same slots on every arc of a path; allocations are one contiguous block."""
    net = PaperOpticalNetwork(NSF, 100)
    path = get_candidate_paths(net, 0, 3, 1, "km")[0]
    assert len(path) >= 3
    ok = net.allocate(path, 40, 6)
    assert ok
    arcs = list(zip(path[:-1], path[1:]))
    same_on_all_arcs = all(
        np.array_equal(net.arc_states[a][40:46], np.ones(6, dtype=bool)) for a in arcs
    )
    # Availability on the path must show one occupied hole [40,46).
    avail = net.get_availability(path)
    blocks = contiguous_free_blocks(avail)
    expected_blocks = [(0, 40), (46, 54)]
    contiguity_ok = blocks == expected_blocks
    # first_fit must never straddle the hole: 41 slots fit only in [46,100).
    ff = first_fit_start(avail, 41)
    ff_ok = ff == 46
    # A 7-slot request First-Fits at the lowest block start.
    ff7_ok = first_fit_start(avail, 7) == 0
    elig = eligible_blocks(avail, 41)
    elig_ok = elig == [(46, 54)]
    passed = same_on_all_arcs and contiguity_ok and ff_ok and ff7_ok and elig_ok
    return {"name": "continuity_contiguity", "pass": bool(passed),
            "details": {"path": path, "blocks_after_alloc": blocks,
                        "first_fit_41": ff, "eligible_41": elig}}


def test_first_fit() -> Dict[str, Any]:
    """First-Fit = lowest-start eligible block; fuzzed vs upstream oracle."""
    avail = np.ones(100, dtype=bool)
    avail[0:5] = False
    avail[8:12] = False
    # free blocks: [5,8) size 3, [12,100) size 88
    checks = [
        (first_fit_start(avail, 3) == 5),
        (first_fit_start(avail, 4) == 12),
        (first_fit_start(avail, 88) == 12),
        (first_fit_start(avail, 89) is None),
        (first_fit_start(np.zeros(100, dtype=bool), 1) is None),
        (first_fit_start(np.ones(100, dtype=bool), 100) == 0),
    ]
    # Fuzz against upstream judge_availability (fs_id=0).
    rng = np.random.RandomState(7)
    fuzz_ok = 0
    fuzz_total = 0
    for p in (0.1, 0.3, 0.5, 0.7, 0.9):
        for _ in range(40):
            occ = rng.rand(100) < p
            slot_temp = [0 if o else 1 for o in occ]  # oracle: 1 = free
            req = int(rng.randint(2, 10))
            flag, fs, fe = oracle.deeprmsa_judge_availability(slot_temp, req, 0)
            mine = first_fit_start(~occ, req)
            fuzz_total += 1
            if flag == 1:
                if mine == fs and fe == fs + req - 1:
                    fuzz_ok += 1
            else:
                if mine is None:
                    fuzz_ok += 1
    passed = all(checks) and fuzz_ok == fuzz_total
    return {"name": "first_fit", "pass": bool(passed),
            "details": {"static_checks": checks, "oracle_fuzz": f"{fuzz_ok}/{fuzz_total}"}}


def test_ksp_ff_oracle_fuzz() -> Dict[str, Any]:
    """End-to-end KSP-FF parity vs upstream K-SP-FF benchmark loop (fuzzed)."""
    mod_reg = paper_modulation_registry()
    linkmap, arc_to_link = _build_oracle_linkmap(NSF)
    num_links = 44
    env = PaperRMSAEnv(NSF, 100, mod_reg=mod_reg)
    rng = np.random.RandomState(13)
    stats = {"total": 0, "agree": 0, "admitted": 0}
    mismatches: List[Dict[str, Any]] = []
    for p in (0.1, 0.3, 0.5, 0.7):
        for _ in range(30):
            # Random directed slot map shared by both implementations.
            slot_map = [[1 if rng.rand() > p else 0 for _ in range(100)] for _ in range(num_links)]
            for arc, lid in arc_to_link.items():
                env.net.arc_states[arc] = np.array([s == 0 for s in slot_map[lid]], dtype=bool)
            src = int(rng.randint(0, 14))
            dst = int(rng.randint(0, 14))
            if src == dst:
                continue
            bw = int(rng.randint(25, 101))
            paths = get_candidate_paths(env.net, src, dst, 5, "km")
            # Oracle: same 0-based node labels as our linkmap.
            admitted_o, path_id_o, fs_o, fe_o, num_fs_o = oracle.deeprmsa_ksp_ff_action(
                paths, linkmap, slot_map, bw, k_path=5, m=1)
            # Ours.
            req = type("Req", (), {"req_id": 0, "src_node": src, "dst_node": dst,
                                   "bitrate_gbps": bw, "arrival_time": 0.0,
                                   "holding_time": 1.0})
            view = env.build_view(req, 5, "km")
            decision = ksp_ff_decide(view)
            admitted_m = decision is not None
            stats["total"] += 1
            agree = admitted_o == admitted_m
            if admitted_o and admitted_m:
                path_id_m, start_m = decision
                num_fs_m = view["paths"][path_id_m]["required_fs"]
                agree = (path_id_o == path_id_m and fs_o == start_m
                         and num_fs_o == num_fs_m and fe_o == fs_o + num_fs_o - 1)
            stats["agree"] += int(agree)
            stats["admitted"] += int(admitted_m)
            if not agree and len(mismatches) < 5:
                mismatches.append({"p": p, "src": src, "dst": dst, "bw": bw,
                                   "oracle": [admitted_o, path_id_o, fs_o, num_fs_o],
                                   "mine": [admitted_m] + (list(decision) if decision else [])})
    passed = stats["agree"] == stats["total"] and stats["total"] > 0
    return {"name": "ksp_ff_oracle_fuzz", "pass": bool(passed),
            "details": {**stats, "mismatches": mismatches}}


def test_release_fifo() -> Dict[str, Any]:
    """Release-before-arrival; releases in (release_time, insertion) order."""
    env = PaperRMSAEnv(NSF, 100)
    net = env.net
    path = get_candidate_paths(net, 0, 1, 1, "km")[0]

    class _Req:
        def __init__(self, i, hold):
            self.req_id = i; self.src_node = path[0]; self.dst_node = path[-1]
            self.bitrate_gbps = 100; self.arrival_time = 0.0; self.holding_time = hold

    env.reset()
    env.advance_time(0.0)
    # Three connections at t=0 with holding 5 (A), 3 (B), 3 (C) — B before C.
    info_a = env.commit(_Req(0, 5.0), env.build_view(_Req(0, 5.0), 1, "km")["paths"][0], 0)
    info_b = env.commit(_Req(1, 3.0), env.build_view(_Req(1, 3.0), 1, "km")["paths"][0], 10)
    info_c = env.commit(_Req(2, 3.0), env.build_view(_Req(2, 3.0), 1, "km")["paths"][0], 20)
    assert info_a["success"] and info_b["success"] and info_c["success"]
    order: List[int] = []
    orig_release = net.release

    def spy_release(p, s, n):
        order.append(s)
        orig_release(p, s, n)

    net.release = spy_release  # type: ignore[assignment]
    env.advance_time(3.0)  # releases B (start 10) then C (start 20), FIFO tie
    tie_order_ok = order == [10, 20]
    fs_a = info_a["required_fs"]
    a_still = bool(np.all(net.arc_states[(path[0], path[1])][0:fs_a]))
    env.advance_time(5.0)  # releases A (start 0) before any request at t=5
    release_before_ok = order == [10, 20, 0]
    all_free = not np.any(net.arc_states[(path[0], path[1])])
    passed = tie_order_ok and a_still and release_before_ok and all_free
    return {"name": "release_fifo", "pass": bool(passed),
            "details": {"release_order": order, "tie_fifo_ok": tie_order_ok,
                        "a_held_until_5": a_still, "released_before_arrival": release_before_ok}}


def test_holding_truncation() -> Dict[str, Any]:
    """TTL: exponential(mean=10), resampled while == 0 or >= 2*mean."""
    reqs = generate_paper_requests(14, seed=555, num_requests=40000,
                                   arrival_interval=0.04)
    holds = np.array([r.holding_time for r in reqs])
    positive = bool(np.all(holds > 0))
    below_cap = bool(np.all(holds < 20.0))
    # Theoretical mean of Exp(10) truncated to (0, 20): 10*(1-3e^-2)/(1-e^-2).
    theo = 10.0 * (1.0 - 3.0 * math.exp(-2.0)) / (1.0 - math.exp(-2.0))
    mean_ok = abs(float(holds.mean()) - theo) < 0.06
    passed = positive and below_cap and mean_ok
    return {"name": "holding_truncation", "pass": bool(passed),
            "details": {"n": len(holds), "min": float(holds.min()), "max": float(holds.max()),
                        "mean": float(holds.mean()), "theoretical_mean": theo}}


def test_uniform_all_od() -> Dict[str, Any]:
    """Uniform over all N*(N-1) ordered OD pairs; src != dst always."""
    n = 14
    reqs = generate_paper_requests(n, seed=777, num_requests=60000,
                                   arrival_interval=0.04)
    src_neq_dst = all(r.src_node != r.dst_node for r in reqs)
    counts = np.zeros((n, n), dtype=int)
    bitrates = set()
    for r in reqs:
        counts[r.src_node, r.dst_node] += 1
        bitrates.add(r.bitrate_gbps)
    off_diag = counts[~np.eye(n, dtype=bool)]
    chi2 = float(np.sum((off_diag - off_diag.mean()) ** 2) / off_diag.mean())
    dof = len(off_diag) - 1
    # Normal approximation for the p-value lower bound (large dof).
    z = (chi2 - dof) / math.sqrt(2 * dof)
    p_approx = 0.5 * math.erfc(z / math.sqrt(2))
    all_pairs_seen = bool(np.all(off_diag > 0))
    bitrate_ok = min(bitrates) >= 25 and max(bitrates) <= 100
    arrivals = np.array([r.arrival_time for r in reqs])
    inter = np.diff(arrivals)
    arrival_mean_ok = abs(float(inter.mean()) - 0.04) < 0.002
    passed = (src_neq_dst and all_pairs_seen and bitrate_ok
              and p_approx > 1e-4 and arrival_mean_ok)
    return {"name": "uniform_all_od", "pass": bool(passed),
            "details": {"src_neq_dst": src_neq_dst, "pairs_seen": int(np.sum(off_diag > 0)),
                        "pairs_total": n * (n - 1), "chi2": chi2, "dof": dof,
                        "p_approx": p_approx, "bitrate_range": [min(bitrates), max(bitrates)],
                        "mean_inter_arrival": float(inter.mean())}}


def test_ksp_path_order() -> Dict[str, Any]:
    """km order = non-decreasing km; hops order = non-decreasing hops then km."""
    details: Dict[str, Any] = {}
    passed = True
    for topo, nn in ((NSF, 14), (COST, 11)):
        net = PaperOpticalNetwork(topo, 100)
        differ = 0
        min_paths_k50 = 999
        for s in range(nn):
            for d in range(nn):
                if s == d:
                    continue
                km_paths = get_candidate_paths(net, s, d, 5, "km")
                hop_paths = get_candidate_paths(net, s, d, 5, "hops")
                km_lens = [net.path_length_km(p) for p in km_paths]
                hop_keys = [(net.path_hops(p), net.path_length_km(p)) for p in hop_paths]
                passed &= all(km_lens[i] <= km_lens[i + 1] + 1e-9 for i in range(len(km_lens) - 1))
                passed &= all(hop_keys[i] <= hop_keys[i + 1] for i in range(len(hop_keys) - 1))
                passed &= len(km_paths) == 5 and len(hop_paths) == 5
                if km_paths != hop_paths:
                    differ += 1
                min_paths_k50 = min(min_paths_k50, len(get_candidate_paths(net, s, d, 50, "hops")))
        details[topo] = {"od_pairs_with_different_ordering": differ,
                         "min_paths_found_k50": min_paths_k50}
        passed &= differ > 0  # the two orderings must be genuinely distinct
    return {"name": "ksp_path_order", "pass": bool(passed), "details": details}


def test_no_excluded_components() -> Dict[str, Any]:
    """Pipeline sources must not reference excluded modules/artifacts."""
    forbidden = [
        "snap24_gnutella_reach", "deep_rmsa_snap24_reach_mixed",
        "optical_only_rmsa_env", "optical_only_rmsa_evaluator",
        "ppo_agents", "counterfactual_r_ranker", "AgentC", "df_c",
    ]
    offenders = []
    for path in sorted(EXP_DIR.glob("*.py")):
        if path.name == Path(__file__).name:
            continue  # the guard file lists the tokens itself
        text = path.read_text(encoding="utf-8")
        for token in forbidden:
            if token in text:
                offenders.append({"file": path.name, "token": token})
    return {"name": "no_excluded_components", "pass": not offenders,
            "details": {"offenders": offenders}}


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
ALL_TESTS: List[Callable[[], Dict[str, Any]]] = [
    test_fs_boundaries,
    test_direction_independence,
    test_continuity_contiguity,
    test_first_fit,
    test_ksp_ff_oracle_fuzz,
    test_release_fifo,
    test_holding_truncation,
    test_uniform_all_od,
    test_ksp_path_order,
    test_no_excluded_components,
]


def main() -> int:
    t0 = time.perf_counter()
    results = []
    for test in ALL_TESTS:
        try:
            res = test()
        except Exception as exc:  # noqa: BLE001 - report and fail the stage
            res = {"name": test.__name__, "pass": False,
                   "details": {"exception": str(exc), "trace": traceback.format_exc()}}
        results.append(res)
        print(f"[{'PASS' if res['pass'] else 'FAIL'}] {res['name']}", flush=True)

    all_pass = all(r["pass"] for r in results)
    elapsed = time.perf_counter() - t0

    (EXP_DIR / "results").mkdir(exist_ok=True)
    (EXP_DIR / "results" / "unit_tests.json").write_text(
        json.dumps({"all_pass": all_pass, "elapsed_seconds": elapsed,
                    "results": results}, indent=2, default=str),
        encoding="utf-8",
    )

    lines = [
        "# UNIT TEST REPORT — Stage A",
        "",
        f"* Protocol: `{CFG['protocol_id']}`",
        f"* Date: 2026-07-17",
        f"* Overall: **{'PASS' if all_pass else 'FAIL'}** ({sum(r['pass'] for r in results)}/{len(results)} tests)",
        f"* Elapsed: {elapsed:.1f} s",
        "",
        "Oracle: `deep_rmsa_parity_adapter.py` (behavioral extraction of upstream",
        "DeepRMSA, audited in v135_r_only_standard_and_blocking_diagnosis).",
        "",
        "| Test | Result | Key details |",
        "|---|---|---|",
    ]
    for r in results:
        detail = json.dumps(r["details"], default=str)
        if len(detail) > 400:
            detail = detail[:400] + "…"
        lines.append(f"| `{r['name']}` | {'PASS' if r['pass'] else 'FAIL'} | {detail} |")
    lines.append("")
    (EXP_DIR / "UNIT_TEST_REPORT.md").write_text("\n".join(lines), encoding="utf-8")

    print(f"[stage-a] all_pass={all_pass} -> UNIT_TEST_REPORT.md ({elapsed:.1f}s)")
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
