"""Unit tests for FF-KSP heuristic action selection.

Tests the spectrum-first KSP-FF variant (ff_ksp_highest_mod_action) against
KSP-FF (ksp_ff_highest_mod_action) and independent brute-force references.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from typing import Any, Dict, List, Optional, Tuple

from sa_hmarl.baselines.rmsa_baselines import (
    ksp_ff_highest_mod_action,
    ff_ksp_highest_mod_action,
)


def _make_obs(num_paths: int,
              num_mods: int,
              num_blocks: int,
              mask_flat: List[bool],
              blocks: List[List[List[Tuple[int, int]]]],
              required_fs: List[List[Optional[int]]],
              path_lengths_km: Optional[List[float]] = None,
             ) -> Dict[str, Any]:
    """Build a minimal observation for Agent-R baselines.

    blocks[path_idx][mod_idx] is a list of (start_slot, size) tuples.
    required_fs[path_idx][mod_idx] is the integer FS demand or None.
    """
    assert len(mask_flat) == num_paths * num_mods * num_blocks
    candidate_paths = [[i] for i in range(num_paths)]
    mod_names = [f"M{i}" for i in range(num_mods)]
    return {
        "agent_r_mask": np.asarray(mask_flat, dtype=bool),
        "candidate_paths": candidate_paths,
        "mod_names": mod_names,
        "required_fs_per_path_mod": required_fs,
        "candidate_blocks_per_path_mod": blocks,
        "path_features": [
            {"path_length_km": (path_lengths_km[i] if path_lengths_km else float(i)),
             "hop_count": 1}
            for i in range(num_paths)
        ],
    }


def _decode(action_idx: int, num_paths: int, num_mods: int, num_blocks: int
           ) -> Tuple[int, int, int]:
    path_idx = action_idx // (num_mods * num_blocks)
    rem = action_idx % (num_mods * num_blocks)
    mod_idx = rem // num_blocks
    block_idx = rem % num_blocks
    return path_idx, mod_idx, block_idx


class TestFFKSPBasics:
    """Mandatory semantic tests from the protocol."""

    def test_ff_ksp_prefers_lower_start_slot_over_path_order(self):
        """Path 0 earliest start=10, path 1 earliest start=2.

        KSP-FF must pick path 0/start 10; FF-KSP must pick path 1/start 2.
        """
        num_paths, num_mods, num_blocks = 2, 1, 2
        # path 0, mod 0: two blocks at start 10 and 20
        # path 1, mod 0: two blocks at start 2 and 30
        blocks = [
            [[(10, 3), (20, 3)]],   # path 0, mod 0
            [[(2, 3), (30, 3)]],    # path 1, mod 0
        ]
        required_fs = [[2], [2]]
        mask = [
            True, True,   # path 0, mod 0, both blocks legal
            True, True,   # path 1, mod 0, both blocks legal
        ]
        obs = _make_obs(num_paths, num_mods, num_blocks, mask, blocks, required_fs)

        ksp = ksp_ff_highest_mod_action(obs)
        ffksp = ff_ksp_highest_mod_action(obs)

        assert ksp is not None
        assert ffksp is not None
        p_ksp, _, b_ksp = _decode(ksp, num_paths, num_mods, num_blocks)
        p_ff, _, b_ff = _decode(ffksp, num_paths, num_mods, num_blocks)
        assert p_ksp == 0 and b_ksp == 0, "KSP-FF should prefer path 0 first"
        assert p_ff == 1 and b_ff == 0, "FF-KSP should prefer lowest start slot"

    def test_ff_ksp_tie_breaks_by_path_order_when_starts_equal(self):
        """When two paths share the same first start slot, FF-KSP picks the one
        earlier in hops/km order (lower path_idx)."""
        num_paths, num_mods, num_blocks = 2, 1, 1
        blocks = [[[(5, 2)]], [[(5, 2)]]]
        required_fs = [[2], [2]]
        mask = [True, True]
        obs = _make_obs(num_paths, num_mods, num_blocks, mask, blocks, required_fs)
        ffksp = ff_ksp_highest_mod_action(obs)
        assert ffksp == 0, "FF-KSP should tie-break by path order -> action 0"

    def test_ff_ksp_chooses_highest_modulation_per_path(self):
        """On a single path, FF-KSP must select the feasible modulation with the
        smallest required_fs (highest spectral efficiency)."""
        num_paths, num_mods, num_blocks = 1, 2, 1
        # mod 0 needs 2 FS (higher modulation), mod 1 needs 4 FS (lower modulation)
        blocks = [[[(0, 4)], [(0, 2)]]]
        required_fs = [[2, 4]]
        mask = [
            True,   # path 0, mod 0, block 0
            True,   # path 0, mod 1, block 0
        ]
        obs = _make_obs(num_paths, num_mods, num_blocks, mask, blocks, required_fs)
        ffksp = ff_ksp_highest_mod_action(obs)
        _, m, _ = _decode(ffksp, num_paths, num_mods, num_blocks)
        assert m == 0, "FF-KSP should choose highest feasible modulation (mod 0)"

    def test_ff_ksp_breaks_equal_fs_tie_by_higher_modulation_index(self):
        obs = _make_obs(
            1, 4, 1, [True] * 4,
            [[[(0, 4)], [(0, 4)], [(0, 4)], [(0, 4)]]],
            [[3, 3, 3, 3]],
        )
        action = ff_ksp_highest_mod_action(obs)
        assert _decode(action, 1, 4, 1) == (0, 3, 0)

    def test_ff_ksp_falls_back_when_highest_modulation_unreachable(self):
        """If the highest modulation is not feasible (mask not set for it), FF-KSP
        falls back to the next feasible modulation with smallest required_fs."""
        num_paths, num_mods, num_blocks = 1, 2, 1
        blocks = [[[(0, 4)], [(0, 2)]]]
        required_fs = [[2, 4]]
        mask = [
            False,  # path 0, mod 0, block 0 -- not legal
            True,   # path 0, mod 1, block 0 -- legal
        ]
        obs = _make_obs(num_paths, num_mods, num_blocks, mask, blocks, required_fs)
        ffksp = ff_ksp_highest_mod_action(obs)
        _, m, _ = _decode(ffksp, num_paths, num_mods, num_blocks)
        assert m == 1, "FF-KSP should fall back to mod 1 when mod 0 is illegal"

    def test_ff_ksp_returns_none_for_empty_mask(self):
        obs = _make_obs(2, 2, 2, [False] * 8, [[[(0, 1)]], [[(0, 1)]]], [[1, 1], [1, 1]])
        assert ff_ksp_highest_mod_action(obs) is None
        assert ksp_ff_highest_mod_action(obs) is None

    def test_ff_ksp_equals_ksp_ff_when_k_is_one(self):
        """With a single path, both heuristics should return the same action."""
        num_paths, num_mods, num_blocks = 1, 2, 2
        blocks = [[[(0, 4), (5, 4)], [(0, 2), (5, 2)]]]
        required_fs = [[2, 4]]
        mask = [True, True, True, True]
        obs = _make_obs(num_paths, num_mods, num_blocks, mask, blocks, required_fs)
        assert ff_ksp_highest_mod_action(obs) == ksp_ff_highest_mod_action(obs)

    def test_ff_ksp_same_start_picks_same_path_as_ksp_ff(self):
        """If all paths share the same first-fit start, FF-KSP and KSP-FF should
        agree on the path (the first path in hops/km order with a legal block)."""
        num_paths, num_mods, num_blocks = 2, 1, 1
        blocks = [[[(5, 2)]], [[(5, 2)]]]
        required_fs = [[2], [2]]
        mask = [True, True]
        obs = _make_obs(num_paths, num_mods, num_blocks, mask, blocks, required_fs)
        assert ff_ksp_highest_mod_action(obs) == ksp_ff_highest_mod_action(obs)

    def test_ff_ksp_action_is_legal_in_mask(self):
        """For random small masks, the returned action must satisfy the raw mask."""
        for _ in range(50):
            num_paths, num_mods, num_blocks = 3, 2, 4
            mask = np.random.choice([True, False], size=num_paths * num_mods * num_blocks)
            if not mask.any():
                continue
            blocks = [[[(b * 3, 5) for b in range(num_blocks)] for _ in range(num_mods)]
                      for _ in range(num_paths)]
            required_fs = [[np.random.choice([2, 3, 4]) for _ in range(num_mods)]
                          for _ in range(num_paths)]
            obs = _make_obs(num_paths, num_mods, num_blocks, mask.tolist(), blocks, required_fs)
            a = ff_ksp_highest_mod_action(obs)
            assert a is not None and mask[a], "FF-KSP action must be legal"

    def test_ff_ksp_matches_brute_force_reference(self):
        """Independent brute-force implementation of FF-KSP semantics."""
        for _ in range(50):
            num_paths, num_mods, num_blocks = 3, 2, 4
            mask = np.random.choice([True, False], size=num_paths * num_mods * num_blocks)
            if not mask.any():
                continue
            blocks = [[[(b * 3 + p, 5) for b in range(num_blocks)] for _ in range(num_mods)]
                      for p in range(num_paths)]
            required_fs = [[np.random.choice([2, 3, 4]) for _ in range(num_mods)]
                          for _ in range(num_paths)]
            obs = _make_obs(num_paths, num_mods, num_blocks, mask.tolist(), blocks, required_fs)

            # Brute force: for each path, best modulation, then global sort by start_slot.
            best_ref = None
            for p in range(num_paths):
                best_mod = None
                best_fs = float('inf')
                for m in range(num_mods):
                    fs = required_fs[p][m]
                    if fs is None or fs <= 0:
                        continue
                    start = p * num_mods * num_blocks + m * num_blocks
                    if not mask[start:start + num_blocks].any():
                        continue
                    if fs < best_fs:
                        best_fs = fs
                        best_mod = m
                if best_mod is None:
                    continue
                start = p * num_mods * num_blocks + best_mod * num_blocks
                for b in range(num_blocks):
                    idx = start + b
                    if not mask[idx]:
                        continue
                    block_start = blocks[p][best_mod][b][0]
                    if best_ref is None or (block_start, p, idx) < (best_ref[0], best_ref[1], best_ref[2]):
                        best_ref = (block_start, p, idx)

            if best_ref is None:
                assert ff_ksp_highest_mod_action(obs) is None
            else:
                assert ff_ksp_highest_mod_action(obs) == best_ref[2]


if __name__ == "__main__":
    import pytest
    sys.exit(pytest.main([__file__, "-v"]))
