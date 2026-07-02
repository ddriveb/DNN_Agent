"""Rule-based RMSA baselines for Agent-R comparison.

- KSP-FF: First-Fit over candidate (path, mod, block) actions.
- KSP-BF: Best-Fit (minimum block waste) over valid actions.
"""
import numpy as np
from typing import Optional, Dict, Any


def _decode_action_idx(action_idx: int, num_modulations: int, num_blocks: int) -> tuple:
    """Decode flat action index into (path_idx, mod_idx, block_idx)."""
    path_idx = action_idx // (num_modulations * num_blocks)
    rem = action_idx % (num_modulations * num_blocks)
    mod_idx = rem // num_blocks
    block_idx = rem % num_blocks
    return path_idx, mod_idx, block_idx


def ksp_ff_action(obs: Dict[str, Any]) -> Optional[int]:
    """Select the first valid (path, mod, block) action (First-Fit).

    Returns:
        Flat action index, or None if no valid action exists.
    """
    mask = obs["agent_r_mask"]
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    return int(valid[0])


def ksp_ff_highest_mod_action(obs: Dict[str, Any]) -> Optional[int]:
    """Path-ordered KSP-FF with highest feasible modulation.

    This is the stronger EON-style heuristic used for tuned comparisons:
    scan candidate paths in their provided order, choose the feasible modulation
    requiring the fewest FS on that path, then choose the lowest-start-slot
    feasible block (First-Fit).  It assumes the observation's block list is
    ordered by ``start_asc``; it still checks block starts defensively.
    """
    mask = np.asarray(obs["agent_r_mask"], dtype=bool)
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return None

    num_paths = len(obs["candidate_paths"])
    num_mods = len(obs["mod_names"])
    if num_paths == 0 or num_mods == 0:
        return None
    num_blocks = len(mask) // (num_paths * num_mods)

    for path_idx in range(num_paths):
        best_mod = None
        best_req_fs = float("inf")
        for mod_idx in range(num_mods):
            req_fs = obs["required_fs_per_path_mod"][path_idx][mod_idx]
            if req_fs is None or req_fs <= 0:
                continue
            start = path_idx * num_mods * num_blocks + mod_idx * num_blocks
            end = start + num_blocks
            if not mask[start:end].any():
                continue
            # Smaller FS demand corresponds to the highest feasible modulation.
            if float(req_fs) < best_req_fs:
                best_req_fs = float(req_fs)
                best_mod = mod_idx

        if best_mod is None:
            continue

        start = path_idx * num_mods * num_blocks + best_mod * num_blocks
        blocks = obs["candidate_blocks_per_path_mod"][path_idx][best_mod]
        valid_blocks = []
        for block_idx in range(num_blocks):
            action_idx = start + block_idx
            if not mask[action_idx]:
                continue
            block_start = blocks[block_idx][0] if block_idx < len(blocks) else block_idx
            valid_blocks.append((block_start, block_idx, action_idx))
        if valid_blocks:
            valid_blocks.sort(key=lambda item: (item[0], item[1]))
            return int(valid_blocks[0][2])

    return None


def ksp_bf_action(obs: Dict[str, Any]) -> Optional[int]:
    """Select the valid action with minimum block waste (Best-Fit).

    Ties are broken by smaller action index (path-major, mod-major, block-major).

    Returns:
        Flat action index, or None if no valid action exists.
    """
    mask = obs["agent_r_mask"]
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None

    num_paths = len(obs["candidate_paths"])
    num_mods = len(obs["mod_names"])
    num_blocks = len(mask) // (num_paths * num_mods)

    best_action = None
    best_waste = float('inf')

    for action_idx in valid:
        path_idx, mod_idx, block_idx = _decode_action_idx(action_idx, num_mods, num_blocks)

        req_fs = obs["required_fs_per_path_mod"][path_idx][mod_idx]
        blocks = obs["candidate_blocks_per_path_mod"][path_idx][mod_idx]

        if block_idx < len(blocks):
            block_size = blocks[block_idx][1]
            if req_fs is not None and req_fs > 0:
                waste = (block_size - req_fs) / block_size
            else:
                waste = 1.0
        else:
            waste = 1.0

        if waste < best_waste:
            best_waste = waste
            best_action = action_idx

    return int(best_action) if best_action is not None else None
