"""Candidate spectrum block extraction for Agent-R.

Given a path's available slot array and required FS count,
extract and rank candidate contiguous free blocks.
"""
import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass


@dataclass
class SpectrumBlock:
    """A candidate contiguous free spectrum block on a path."""
    start_slot: int      # Starting slot index
    size: int            # Total block size (may be larger than needed)
    req_fs: int          # Required FS count (for waste calculation)

    @property
    def waste_ratio(self) -> float:
        """Fraction of the block that would be wasted."""
        if self.size <= 0:
            return 1.0
        return (self.size - self.req_fs) / self.size

    @property
    def left_fit_start(self) -> int:
        """Start slot if using left-fit within this block."""
        return self.start_slot

    @property
    def center_fit_start(self) -> int:
        """Start slot if using center-fit within this block."""
        padding = (self.size - self.req_fs) // 2
        return self.start_slot + padding


def extract_contiguous_blocks(avail: np.ndarray,
                              min_size: int = 1) -> List[Tuple[int, int]]:
    """Extract all contiguous free blocks from availability array.

    Args:
        avail: Boolean array where True = free.
        min_size: Minimum block size to include.

    Returns:
        List of (start_slot, block_size).
    """
    blocks = []
    n = len(avail)
    i = 0
    while i < n:
        if avail[i]:
            start = i
            while i < n and avail[i]:
                i += 1
            size = i - start
            if size >= min_size:
                blocks.append((start, size))
        else:
            i += 1
    return blocks


def extract_candidate_blocks(avail: np.ndarray,
                             req_fs: int,
                             max_candidates: int = 5,
                             sort_by: str = "size_desc") -> List[SpectrumBlock]:
    """Extract candidate contiguous free blocks for Agent-R.

    Args:
        avail: Boolean array where True = free.
        req_fs: Required number of frequency slots.
        max_candidates: Maximum number of candidates to return.
        sort_by: Sorting strategy:
            - "size_desc": Largest block first ( conservative )
            - "waste_asc": Smallest waste first ( efficient )
            - "start_asc": Lowest starting slot first ( First-Fit-like )
            - "center_asc": Closest to center first ( fragmentation-friendly )
            - "mixed": De-duplicated union of size_desc, waste_asc, start_asc,
              and center_asc.  This exposes diverse candidate blocks to Agent-R
              without changing the action abstraction.

    Returns:
        List of SpectrumBlock candidates (size >= req_fs).
    """
    raw_blocks = extract_contiguous_blocks(avail, min_size=req_fs)
    candidates = [SpectrumBlock(start, size, req_fs) for start, size in raw_blocks]

    if sort_by == "mixed":
        return _extract_mixed_candidates(avail, req_fs, max_candidates)

    if sort_by == "size_desc":
        candidates.sort(key=lambda b: b.size, reverse=True)
    elif sort_by == "waste_asc":
        candidates.sort(key=lambda b: b.waste_ratio)
    elif sort_by == "start_asc":
        candidates.sort(key=lambda b: b.start_slot)
    elif sort_by == "center_asc":
        center = len(avail) // 2
        candidates.sort(key=lambda b: abs(b.start_slot - center))
    else:
        raise ValueError(f"Unknown sort_by: {sort_by}")

    return candidates[:max_candidates]


def _extract_mixed_candidates(avail: np.ndarray,
                              req_fs: int,
                              max_candidates: int) -> List[SpectrumBlock]:
    """Return a de-duplicated diverse candidate set.

    A single ordering can hide useful choices.  For example, ``size_desc`` may
    expose only large conservative blocks, while ``waste_asc`` can over-focus on
    tight fits.  The mixed strategy takes the front of several rankings and
    keeps their first occurrence order.
    """
    strategies = ("size_desc", "waste_asc", "start_asc", "center_asc")
    per_strategy = max(max_candidates, 1)
    selected = []
    seen = set()
    for strategy in strategies:
        for block in extract_candidate_blocks(
            avail, req_fs, max_candidates=per_strategy, sort_by=strategy
        ):
            key = (block.start_slot, block.size)
            if key in seen:
                continue
            selected.append(block)
            seen.add(key)
            if len(selected) >= max_candidates:
                return selected
    return selected


def first_fit(avail: np.ndarray, req_fs: int) -> Optional[int]:
    """Classic First-Fit: return start slot of first sufficient block."""
    blocks = extract_contiguous_blocks(avail, min_size=req_fs)
    if not blocks:
        return None
    return blocks[0][0]


def best_fit(avail: np.ndarray, req_fs: int) -> Optional[int]:
    """Best-Fit: return start slot of the tightest fitting block."""
    blocks = extract_contiguous_blocks(avail, min_size=req_fs)
    if not blocks:
        return None
    # Tightest fit = smallest size >= req_fs
    best = min(blocks, key=lambda b: b[1])
    return best[0]


def exact_fit(avail: np.ndarray, req_fs: int) -> Optional[int]:
    """Exact-Fit: return start slot of a block exactly matching req_fs."""
    blocks = extract_contiguous_blocks(avail, min_size=req_fs)
    for start, size in blocks:
        if size == req_fs:
            return start
    return None
