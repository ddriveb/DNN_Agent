"""Test spectrum block extraction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sa_hmarl.network.spectrum_blocks import (
    extract_contiguous_blocks, extract_candidate_blocks,
    first_fit, best_fit
)


def test_extract_blocks():
    avail = np.array([1, 1, 0, 1, 1, 1, 0, 1, 0, 0], dtype=bool)
    blocks = extract_contiguous_blocks(avail, min_size=2)
    assert blocks == [(0, 2), (3, 3)]
    print("test_extract_blocks PASSED")


def test_candidate_blocks():
    avail = np.array([1, 1, 0, 1, 1, 1, 1, 0, 1, 1], dtype=bool)
    cands = extract_candidate_blocks(avail, req_fs=2, max_candidates=3)
    assert len(cands) == 3
    # First should be largest block: start=3, size=4
    assert cands[0].start_slot == 3
    assert cands[0].size == 4
    assert cands[0].waste_ratio == (4 - 2) / 4
    print("test_candidate_blocks PASSED")


def test_first_best_fit():
    avail = np.array([1, 1, 0, 1, 1, 1, 0, 1], dtype=bool)
    assert first_fit(avail, 2) == 0
    assert best_fit(avail, 2) == 0  # block (0,2) is tightest fit for 2
    print("test_first_best_fit PASSED")


def test_mixed_candidate_blocks_are_diverse_and_unique():
    avail = np.array(
        [1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 1, 1],
        dtype=bool,
    )
    cands = extract_candidate_blocks(avail, req_fs=2, max_candidates=4, sort_by="mixed")
    keys = [(b.start_slot, b.size) for b in cands]
    assert len(keys) == len(set(keys))
    assert len(cands) == 4
    # The first candidate still preserves the conservative largest-block prior.
    assert keys[0] == (3, 4)
    # Mixed ranking should also expose a tight/early block, not only large ones.
    assert (0, 2) in keys
    print("test_mixed_candidate_blocks_are_diverse_and_unique PASSED")


if __name__ == "__main__":
    test_extract_blocks()
    test_candidate_blocks()
    test_first_best_fit()
    test_mixed_candidate_blocks_are_diverse_and_unique()
