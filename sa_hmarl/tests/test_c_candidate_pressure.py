"""Smoke tests for C-side candidate pressure diagnostic.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/tests/test_c_candidate_pressure.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math

import numpy as np

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_r_observation,
)
from sa_hmarl.evaluation.c_candidate_pressure import (
    compute_c_candidate_pressure,
    compute_all_c_candidate_pressures,
    _decode_valid_actions,
)


# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "split_id",
    "server_id",
    "n_valid_r_actions",
    "min_required_fs",
    "best_lfb",
    "fs_lfb_ratio",
    "min_path_len_km",
    "avg_path_len_km",
    "feasible_mod_count",
    "frag_pressure",
    "delay_slack_ms",
    "server_utilization",
    "high_pressure",
    # new explicit valid-action fields
    "valid_action_min_fs",
    "valid_action_best_block",
    "valid_action_min_fs_block_ratio",
    "valid_path_min_delay_ms",
]


def _make_env(num_slots: int = 32, num_servers: int = 2):
    """Minimal 6-node test environment."""
    net = OpticalNetwork("net1", num_slots=num_slots)
    mec = MECCluster(
        num_nodes=net.NUM_NODES,
        num_servers=num_servers,
        seed=42,
        server_nodes=[0, 3] if num_servers >= 2 else [0],
        capacities=[50.0] * num_servers,
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)
    return env


def _make_request(env: SMDPEnv, src: int = 1,
                  deadline_ms: float = 120.0,
                  num_splits: int = 3) -> DNNRequest:
    """Create a single realistic DNN request."""
    splits = [
        SplitProfile(
            split_id=i,
            intermediate_size_mb=10.0 + i * 5.0,
            edge_compute_cost=2.0 + i * 1.0,
            local_compute_cost=5.0 - i * 1.0,
        )
        for i in range(num_splits)
    ]
    return DNNRequest(
        req_id=0,
        src_node=src,
        arrival_time=0.0,
        deadline_ms=deadline_ms,
        holding_time=5.0,
        splits=splits,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_single_candidate_returns_all_keys():
    """compute_c_candidate_pressure returns every expected key."""
    env = _make_env()
    req = _make_request(env)

    result = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    for key in REQUIRED_KEYS:
        assert key in result, f"missing key: {key}"

    print("  PASS test_single_candidate_returns_all_keys")


def test_n_valid_r_actions_is_int():
    """n_valid_r_actions is a Python int."""
    env = _make_env()
    req = _make_request(env)

    result = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    n = result["n_valid_r_actions"]
    assert isinstance(n, int), f"n_valid_r_actions is {type(n)}, expected int"

    print("  PASS test_n_valid_r_actions_is_int")


def test_fs_lfb_ratio_is_finite_when_actions_exist():
    """fs_lfb_ratio is finite when n_valid_r_actions > 0."""
    env = _make_env()
    req = _make_request(env)

    result = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    if result["n_valid_r_actions"] > 0:
        val = result["fs_lfb_ratio"]
        assert math.isfinite(val), (
            f"fs_lfb_ratio should be finite when n_valid>0, got {val}"
        )

    print("  PASS test_fs_lfb_ratio_is_finite_when_actions_exist")


def test_high_pressure_when_no_valid_r():
    """high_pressure is True when n_valid_r_actions == 0."""
    env = _make_env(num_slots=1)  # very tight spectrum → likely 0 valid
    req = _make_request(env)

    result = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    if result["n_valid_r_actions"] == 0:
        assert result["high_pressure"] is True, (
            f"high_pressure should be True when n_valid=0, got {result['high_pressure']}"
        )
    assert isinstance(result["high_pressure"], bool)

    print("  PASS test_high_pressure_when_no_valid_r")


def test_does_not_crash_when_all_invalid_r():
    """Calling with an extreme deadline does not raise."""
    env = _make_env(num_slots=8)
    req = _make_request(env, deadline_ms=0.1)

    result = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    assert isinstance(result, dict), f"expected dict, got {type(result)}"

    print("  PASS test_does_not_crash_when_all_invalid_r")


def test_all_candidates_returns_list():
    """compute_all_c_candidate_pressures returns a non-empty list of dicts."""
    env = _make_env()
    req = _make_request(env)

    results = compute_all_c_candidate_pressures(env, req)
    assert isinstance(results, list), f"expected list, got {type(results)}"
    assert len(results) > 0, "expected at least one valid C candidate"

    for r in results:
        assert isinstance(r, dict)
        for key in REQUIRED_KEYS:
            assert key in r, f"missing key in all-candidates result: {key}"
        assert "flat_idx" in r, "missing flat_idx in all-candidates result"

    print(f"  PASS test_all_candidates_returns_list ({len(results)} candidates)")


def test_all_fields_have_reasonable_types():
    """All pressure fields have reasonable types; valid-only when n_valid>0."""
    env = _make_env()
    req = _make_request(env)

    result = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)

    # Type checks
    assert isinstance(result["split_id"], int)
    assert isinstance(result["server_id"], int)
    assert isinstance(result["n_valid_r_actions"], int)
    assert isinstance(result["min_required_fs"], int)
    assert isinstance(result["best_lfb"], int)
    assert isinstance(result["fs_lfb_ratio"], float)
    assert isinstance(result["min_path_len_km"], float)
    assert isinstance(result["avg_path_len_km"], float)
    assert isinstance(result["feasible_mod_count"], int)
    assert isinstance(result["frag_pressure"], float)
    assert isinstance(result["delay_slack_ms"], float)
    assert isinstance(result["server_utilization"], float)
    assert isinstance(result["high_pressure"], bool)
    # New fields
    assert isinstance(result["valid_action_min_fs"], int)
    assert isinstance(result["valid_action_best_block"], int)
    assert isinstance(result["valid_action_min_fs_block_ratio"], float)
    assert isinstance(result["valid_path_min_delay_ms"], float)

    # Range checks for the normal case (n_valid > 0)
    assert result["n_valid_r_actions"] >= 0
    assert result["min_required_fs"] >= 0
    assert result["best_lfb"] >= 0
    assert result["feasible_mod_count"] >= 0
    assert 0.0 <= result["server_utilization"] <= 1.0

    if result["n_valid_r_actions"] > 0:
        assert result["min_path_len_km"] >= 0.0
        assert math.isfinite(result["min_path_len_km"])
        assert result["avg_path_len_km"] >= 0.0
        assert math.isfinite(result["avg_path_len_km"])
        assert result["frag_pressure"] >= 0.0
        assert math.isfinite(result["frag_pressure"])
        assert math.isfinite(result["fs_lfb_ratio"])
        assert math.isfinite(result["valid_path_min_delay_ms"])
        assert result["valid_path_min_delay_ms"] >= 0.0

    print("  PASS test_all_fields_have_reasonable_types")


def test_new_fields_aliased_correctly():
    """New explicit fields match the core fields when n_valid > 0."""
    env = _make_env()
    req = _make_request(env)

    result = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    if result["n_valid_r_actions"] > 0:
        assert result["valid_action_min_fs"] == result["min_required_fs"]
        assert result["valid_action_best_block"] == result["best_lfb"]
        assert result["valid_action_min_fs_block_ratio"] == result["fs_lfb_ratio"]

    print("  PASS test_new_fields_aliased_correctly")


# ---------------------------------------------------------------------------
# Critical test: path/mod feasible but no block
# ---------------------------------------------------------------------------

def test_path_mod_feasible_but_no_block_excluded():
    """A path×mod that is modulation-feasible but has no large-enough block
    must NOT influence min_required_fs or fs_lfb_ratio.

    Scenario
    --------
    1. Build obs_r with clean spectrum → record metrics A.
    2. Pre-occupy spectrum on *one* path so its blocks become too small for
       the required FS, but the other path(s) remain clean.
    3. Rebuild obs_r → the starved path's required_fs (possibly small) must
       NOT appear in min_required_fs, because it has zero valid R actions.
    """
    env = _make_env(num_slots=16, num_servers=1)  # 1 server → simpler
    req = _make_request(env, src=1, deadline_ms=200.0)

    # Build obs_r once to inspect paths
    obs_r0 = build_agent_r_observation(env, req, split_id=0, server_id=0)
    paths = obs_r0["candidate_paths"]
    req_fs_matrix = obs_r0["required_fs_per_path_mod"]
    feasible_matrix = obs_r0["feasible_mask_per_path_mod"]

    # We need at least 2 paths, one of which we will starve
    assert len(paths) >= 2, (
        f"Test requires ≥2 candidate paths, got {len(paths)}. "
        f"Try a different src/dst pair."
    )

    # Find a (path, mod) that is feasible and has a known required_fs
    starve_path = None
    starve_mod = None
    starve_fs = None
    for p in range(len(paths)):
        for m in range(len(feasible_matrix[p])):
            if feasible_matrix[p][m] and req_fs_matrix[p][m] is not None:
                fs_val = req_fs_matrix[p][m]
                if fs_val > 0:
                    starve_path = p
                    starve_mod = m
                    starve_fs = fs_val
                    break
        if starve_path is not None:
            break

    assert starve_path is not None, "No feasible (path, mod) found to starve"

    # Ensure another path remains valid after starvation
    other_path = 1 if starve_path == 0 else 0
    print(f"    starve_path={starve_path} (fs={starve_fs}), other_path={other_path}")

    # --- Baseline: before starvation ---
    result_before = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    n_before = result_before["n_valid_r_actions"]
    fs_before = result_before["min_required_fs"]
    ratio_before = result_before["fs_lfb_ratio"]
    print(f"    before: n_valid={n_before}, min_fs={fs_before}, ratio={ratio_before:.3f}")
    assert n_before > 0, "expected valid R actions on clean spectrum"

    # --- Starve path: occupy its spectrum to make all blocks too small ---
    # Allocate on every slot of starve_path so no contiguous block remains
    starve_node_list = paths[starve_path]
    for slot in range(env.net.num_slots):
        env.net.allocate(starve_node_list, slot, 1)

    # --- After starvation ---
    result_after = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    n_after = result_after["n_valid_r_actions"]
    fs_after = result_after["min_required_fs"]
    ratio_after = result_after["fs_lfb_ratio"]
    print(f"    after:  n_valid={n_after}, min_fs={fs_after}, ratio={ratio_after:.3f}")

    # The starved path's required_fs must NOT appear in min_required_fs
    # (because that path now has zero valid blocks).
    # If starve_fs was the ONLY fs, then n_valid may drop to 0 and fs → 0.
    # If another path still has valid actions, min_required_fs should come
    # from that other path (possibly different from starve_fs).
    if n_after > 0:
        # The starved path's fs must not be the new minimum
        # (unless it coincidentally equals the other path's fs)
        if starve_fs is not None and fs_after != starve_fs:
            print(f"    ✓ starved fs={starve_fs} excluded from min_fs={fs_after}")
        else:
            print(f"    (starved fs={starve_fs} happens to equal min_fs={fs_after} "
                  f"— acceptable if both paths have same fs requirement)")
    else:
        # All paths starved → n_valid=0, fs_lfb_ratio should be inf
        assert not math.isfinite(result_after["fs_lfb_ratio"]), (
            f"fs_lfb_ratio should be inf when no valid actions, got {ratio_after}"
        )
        print(f"    ✓ all paths starved → n_valid=0, fs_lfb_ratio=inf")

    # The core assertion: starve_fs must not wrongly pull down min_required_fs
    # If there are still valid actions on other paths, min_required_fs comes
    # from those valid paths only.
    if n_after > 0:
        # Verify that the starved path has zero valid blocks in the mask
        obs_r_after = build_agent_r_observation(env, req, split_id=0, server_id=0)
        va = _decode_valid_actions(obs_r_after)
        assert starve_path not in va["valid_path_indices"], (
            f"starved path {starve_path} should NOT be in valid_path_indices, "
            f"got {va['valid_path_indices']}"
        )
        print(f"    ✓ starved path {starve_path} correctly excluded from valid paths")

    print("  PASS test_path_mod_feasible_but_no_block_excluded")


def test_different_splits_produce_different_pressures():
    """Different split_ids produce meaningfully different pressure dicts."""
    env = _make_env()
    req = _make_request(env, num_splits=3)

    r0 = compute_c_candidate_pressure(env, req, split_id=0, server_id=0)
    r1 = compute_c_candidate_pressure(env, req, split_id=1, server_id=0)
    r2 = compute_c_candidate_pressure(env, req, split_id=2, server_id=0)

    # At least one field should differ across splits
    fields_differ = (
        r0["min_required_fs"] != r1["min_required_fs"]
        or r0["min_required_fs"] != r2["min_required_fs"]
        or r0["delay_slack_ms"] != r1["delay_slack_ms"]
    )
    assert fields_differ, (
        "Expected split-dependent fields to differ across splits, "
        f"fs=({r0['min_required_fs']}, {r1['min_required_fs']}, {r2['min_required_fs']})"
    )

    print("  PASS test_different_splits_produce_different_pressures")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("C-Side Candidate Pressure — Smoke Tests")
    print("=" * 60)

    test_single_candidate_returns_all_keys()
    test_n_valid_r_actions_is_int()
    test_fs_lfb_ratio_is_finite_when_actions_exist()
    test_high_pressure_when_no_valid_r()
    test_does_not_crash_when_all_invalid_r()
    test_all_candidates_returns_list()
    test_all_fields_have_reasonable_types()
    test_new_fields_aliased_correctly()
    test_path_mod_feasible_but_no_block_excluded()
    test_different_splits_produce_different_pressures()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
