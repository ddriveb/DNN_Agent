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
