"""Deterministic unit audit of KSP-FF plain vs. highest-mod baselines.

Constructs synthetic Agent-R observations and verifies that:
1. ksp_ff_action selects the first legal flat action (path-major, mod-major, block-major).
2. ksp_ff_highest_mod_action selects the highest-feasible-modulation action on the first
   hops-ordered path that has a feasible modulation, then the lowest-start-slot block.
3. Both fall back to None when the mask is empty.
4. The default modulation registry order is BPSK, QPSK, 8QAM, 16QAM.

This test does NOT load any neural checkpoint and runs in milliseconds.
"""
import numpy as np
import pytest

from sa_hmarl.baselines.rmsa_baselines import (
    ksp_ff_action,
    ksp_ff_highest_mod_action,
    _decode_action_idx,
)
from sa_hmarl.env.action_mask import build_agent_r_mask
from sa_hmarl.network.modulation import ModulationRegistry


NUM_PATHS = 2
NUM_MODS = 4
NUM_BLOCKS = 3
MOD_NAMES = ["BPSK", "QPSK", "8QAM", "16QAM"]


def _make_obs(
    feasible_mask_per_path_mod,
    required_fs_per_path_mod,
    candidate_blocks_per_path_mod,
):
    """Build a consistent obs_r dict and the corresponding legal mask."""
    mask = build_agent_r_mask(
        num_paths=NUM_PATHS,
        num_modulations=NUM_MODS,
        num_blocks=NUM_BLOCKS,
        mod_names=MOD_NAMES,
        feasible_mask_per_path_mod=feasible_mask_per_path_mod,
        required_fs_per_path_mod=required_fs_per_path_mod,
        candidate_blocks_per_path_mod=candidate_blocks_per_path_mod,
    )
    # Dummy candidate_paths and mod_names to satisfy highest-mod introspection.
    return {
        "agent_r_mask": mask,
        "candidate_paths": [list(range(p + 2)) for p in range(NUM_PATHS)],
        "mod_names": MOD_NAMES,
        "feasible_mask_per_path_mod": feasible_mask_per_path_mod,
        "required_fs_per_path_mod": required_fs_per_path_mod,
        "candidate_blocks_per_path_mod": candidate_blocks_per_path_mod,
    }


def _decode(obs, action):
    return _decode_action_idx(action, NUM_MODS, NUM_BLOCKS)


# ---------------------------------------------------------------------------
# Registry-order sanity check (modulation.py:24-29)
# ---------------------------------------------------------------------------

def test_modulation_registry_default_order_is_bpsk_first():
    reg = ModulationRegistry.from_profile("default")
    names = [reg[i].name for i in range(reg.num_formats)]
    assert names == ["BPSK", "QPSK", "8QAM", "16QAM"]


# ---------------------------------------------------------------------------
# KSP-FF plain: first legal flat action
# ---------------------------------------------------------------------------

def test_ksp_ff_plain_picks_first_legal_flat_action():
    """Plain = path0/BPSK/block0 when all are legal."""
    feasible = [[True] * NUM_MODS for _ in range(NUM_PATHS)]
    req_fs = [[2, 1, 1, 1], [2, 1, 1, 1]]
    blocks = [
        [[(0, 4), (10, 4), (20, 4)] for _ in range(NUM_MODS)],
        [[(5, 4), (15, 4), (25, 4)] for _ in range(NUM_MODS)],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_action(obs)
    assert action == 0
    p, m, b = _decode(obs, action)
    assert (p, m, b) == (0, 0, 0)


def test_ksp_ff_plain_prefers_bpsk_when_bpsk_is_legal():
    """Plain chooses the lowest-modulation-index legal format."""
    feasible = [[True, True, False, False], [True] * NUM_MODS]
    req_fs = [[2, 1, None, None], [2, 1, 1, 1]]
    blocks = [
        [[(0, 4), (10, 4)], [(0, 4)], [], []],
        [[(5, 4)], [(5, 4)], [(5, 4)], [(5, 4)]],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_action(obs)
    p, m, b = _decode(obs, action)
    assert (p, m) == (0, 0), "plain must prefer BPSK when it is legal"
    assert b == 0


def test_ksp_ff_plain_uses_start_asc_block_order():
    """Even if a later block is listed first, plain respects b_idx=0 = lowest start."""
    feasible = [[True, False, False, False], [True] * NUM_MODS]
    req_fs = [[2, None, None, None], [2, 1, 1, 1]]
    # Candidate blocks are already sorted by start_asc in real obs; here we test that
    # b_idx 0 maps to the lowest start slot.
    blocks = [
        [[(0, 4), (5, 4), (10, 4)], [], [], []],
        [[(20, 4)], [(20, 4)], [(20, 4)], [(20, 4)]],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_action(obs)
    p, m, b = _decode(obs, action)
    assert (p, m, b) == (0, 0, 0)
    assert obs["candidate_blocks_per_path_mod"][p][m][b][0] == 0


def test_ksp_ff_plain_falls_back_to_path1_when_path0_empty():
    feasible = [[False] * NUM_MODS, [True] * NUM_MODS]
    req_fs = [[None] * NUM_MODS, [2, 1, 1, 1]]
    blocks = [
        [[] for _ in range(NUM_MODS)],
        [[(5, 4)], [(5, 4)], [(5, 4)], [(5, 4)]],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_action(obs)
    p, m, b = _decode(obs, action)
    assert p == 1
    assert (m, b) == (0, 0)


def test_ksp_ff_plain_returns_none_for_empty_mask():
    feasible = [[False] * NUM_MODS for _ in range(NUM_PATHS)]
    req_fs = [[None] * NUM_MODS for _ in range(NUM_PATHS)]
    blocks = [[[] for _ in range(NUM_MODS)] for _ in range(NUM_PATHS)]
    obs = _make_obs(feasible, req_fs, blocks)
    assert ksp_ff_action(obs) is None


# ---------------------------------------------------------------------------
# KSP-FF highest-mod: path-ordered, highest feasible modulation, start-asc block
# ---------------------------------------------------------------------------

def test_ksp_ff_highest_prefers_highest_modulation_on_first_path():
    """On path0 BPSK needs 2 FS, QPSK needs 1 FS -> highest picks QPSK."""
    feasible = [[True, True, False, False], [True] * NUM_MODS]
    req_fs = [[2, 1, None, None], [2, 1, 1, 1]]
    blocks = [
        [[(0, 4), (10, 4)], [(0, 4)], [], []],
        [[(5, 4)], [(5, 4)], [(5, 4)], [(5, 4)]],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_highest_mod_action(obs)
    p, m, b = _decode(obs, action)
    assert (p, m) == (0, 1), "highest must pick QPSK (smaller req_fs) on path0"
    assert b == 0


def test_ksp_ff_highest_breaks_equal_fs_tie_by_higher_modulation():
    feasible = [[True, True, True, True], [False] * NUM_MODS]
    req_fs = [[3, 3, 3, 3], [None] * NUM_MODS]
    blocks = [[[(0, 4)] for _ in range(NUM_MODS)], [[] for _ in range(NUM_MODS)]]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_highest_mod_action(obs)
    assert _decode(obs, action) == (0, 3, 0)


def test_ksp_ff_highest_stops_at_first_path_with_any_feasible_mod():
    """If path0 has a feasible (even low) mod, highest does not look at path1."""
    feasible = [[True, False, False, False], [True, True, False, False]]
    req_fs = [[2, None, None, None], [2, 1, None, None]]
    blocks = [
        [[(0, 4)], [], [], []],
        [[(5, 4)], [(5, 4)], [], []],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_highest_mod_action(obs)
    p, m, b = _decode(obs, action)
    assert (p, m, b) == (0, 0, 0)


def test_ksp_ff_highest_falls_back_to_second_path_when_first_is_empty():
    feasible = [[False] * NUM_MODS, [True, True, False, False]]
    req_fs = [[None] * NUM_MODS, [2, 1, None, None]]
    blocks = [
        [[] for _ in range(NUM_MODS)],
        [[(5, 4)], [(5, 4)], [], []],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_highest_mod_action(obs)
    p, m, b = _decode(obs, action)
    assert p == 1
    assert m == 1  # QPSK is highest feasible on path1
    assert b == 0


def test_ksp_ff_highest_uses_lowest_start_slot_block():
    """Multiple feasible blocks on the chosen (path, mod): pick lowest start slot."""
    feasible = [[True, False, False, False], [True] * NUM_MODS]
    req_fs = [[2, None, None, None], [2, 1, 1, 1]]
    blocks = [
        [[(5, 4), (0, 4), (10, 4)], [], [], []],  # listed out of order
        [[(20, 4)], [(20, 4)], [(20, 4)], [(20, 4)]],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    action = ksp_ff_highest_mod_action(obs)
    p, m, b = _decode(obs, action)
    assert (p, m) == (0, 0)
    # highest-mod sorts valid blocks by (start_slot, block_idx)
    start_slot = obs["candidate_blocks_per_path_mod"][p][m][b][0]
    assert start_slot == 0


def test_ksp_ff_highest_returns_none_for_empty_mask():
    feasible = [[False] * NUM_MODS for _ in range(NUM_PATHS)]
    req_fs = [[None] * NUM_MODS for _ in range(NUM_PATHS)]
    blocks = [[[] for _ in range(NUM_MODS)] for _ in range(NUM_PATHS)]
    obs = _make_obs(feasible, req_fs, blocks)
    assert ksp_ff_highest_mod_action(obs) is None


# ---------------------------------------------------------------------------
# Direct comparison on a single synthetic state
# ---------------------------------------------------------------------------

def test_plain_and_highest_diverge_when_high_mod_exists_on_first_path():
    """The canonical case where plain is suboptimal vs. highest."""
    feasible = [[True, True, False, False], [True] * NUM_MODS]
    req_fs = [[2, 1, None, None], [2, 1, 1, 1]]
    blocks = [
        [[(0, 4), (10, 4)], [(0, 4)], [], []],
        [[(5, 4)], [(5, 4)], [(5, 4)], [(5, 4)]],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    plain = ksp_ff_action(obs)
    high = ksp_ff_highest_mod_action(obs)
    assert _decode(obs, plain) == (0, 0, 0)
    assert _decode(obs, high) == (0, 1, 0)


def test_plain_and_highest_agree_when_only_bpsk_is_feasible():
    feasible = [[True, False, False, False], [True] * NUM_MODS]
    req_fs = [[2, None, None, None], [2, 1, 1, 1]]
    blocks = [
        [[(0, 4)], [], [], []],
        [[(5, 4)], [(5, 4)], [(5, 4)], [(5, 4)]],
    ]
    obs = _make_obs(feasible, req_fs, blocks)
    plain = ksp_ff_action(obs)
    high = ksp_ff_highest_mod_action(obs)
    assert plain == high
    assert _decode(obs, plain) == (0, 0, 0)
