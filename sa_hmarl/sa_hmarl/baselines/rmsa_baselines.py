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
            if (
                float(req_fs) < best_req_fs
                or (float(req_fs) == best_req_fs and (best_mod is None or mod_idx > best_mod))
            ):
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


def ff_ksp_highest_mod_action(obs: Dict[str, Any]) -> Optional[int]:
    """Spectrum-first KSP with highest feasible modulation per path.

    FF-KSP (First-Fit over K-Shortest-Paths) scans the spectrum in ascending
    start-slot order.  For each candidate start slot it considers paths in the
    provided hops/km order and picks the highest feasible modulation on that
    path whose block starts at or before the slot.  The first successful slot
    wins, breaking ties by path order and then by action index.

    Equivalent loop semantics:
        for start_slot in ascending unique start slots across all paths/mods:
            for path_idx in hops/km order:
                best_mod = feasible modulation with smallest required_fs on path
                if best_mod exists and its block starts at start_slot and is legal:
                    return that action
        return None

    This is intentionally *not* the same as KSP-FF: KSP-FF is path-first, while
    FF-KSP is spectrum-start-first.
    """
    mask = np.asarray(obs["agent_r_mask"], dtype=bool)
    if not mask.any():
        return None

    num_paths = len(obs["candidate_paths"])
    num_mods = len(obs["mod_names"])
    if num_paths == 0 or num_mods == 0:
        return None
    num_blocks = len(mask) // (num_paths * num_mods)

    # Collect, for every path, the highest feasible modulation (smallest req_fs).
    # Tie-break: smaller req_fs, then lower mod_idx (higher spectral efficiency
    # typically corresponds to lower mod_idx in the default registry).
    path_best_mod: List[Optional[int]] = []
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
            if (
                float(req_fs) < best_req_fs
                or (float(req_fs) == best_req_fs and (best_mod is None or mod_idx > best_mod))
            ):
                best_req_fs = float(req_fs)
                best_mod = mod_idx
        path_best_mod.append(best_mod)

    # Gather candidate (start_slot, path_idx, block_idx, action_idx) tuples.
    # Only the best modulation per path is considered, consistent with the
    # "highest feasible modulation" step in FF-KSP.
    candidates = []
    for path_idx in range(num_paths):
        mod_idx = path_best_mod[path_idx]
        if mod_idx is None:
            continue
        start = path_idx * num_mods * num_blocks + mod_idx * num_blocks
        blocks = obs["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        for block_idx in range(num_blocks):
            action_idx = start + block_idx
            if not mask[action_idx]:
                continue
            block_start = blocks[block_idx][0] if block_idx < len(blocks) else block_idx
            candidates.append((block_start, path_idx, block_idx, action_idx))

    if not candidates:
        return None

    # Global sort: lowest start_slot first, then path order (hops/km), then
    # lower action index.  This matches the FF-KSP spectrum-first rule.
    candidates.sort(key=lambda item: (item[0], item[1], item[3]))
    return int(candidates[0][3])


def _feasible_modulation_for_path(
    obs: Dict[str, Any],
    path_idx: int,
    mod_registry: "ModulationRegistry",
) -> Optional[int]:
    """Return the best modulation index for a single path in rescue order.

    Best means the feasible modulation requiring the fewest FS; ties are broken
    by higher spectral efficiency, then by smaller mod index.  Returns None if
    no modulation on the path has a legal block.
    """
    num_mods = len(obs["mod_names"])
    num_paths = len(obs["candidate_paths"])
    if num_paths == 0 or num_mods == 0:
        return None
    num_blocks = len(obs["agent_r_mask"]) // (num_paths * num_mods)

    best_mod: Optional[int] = None
    best_key: Optional[Tuple[int, float, int]] = None
    mask = np.asarray(obs["agent_r_mask"], dtype=bool)
    for mod_idx in range(num_mods):
        req_fs = obs["required_fs_per_path_mod"][path_idx][mod_idx]
        if req_fs is None or req_fs <= 0:
            continue
        start = path_idx * num_mods * num_blocks + mod_idx * num_blocks
        end = start + num_blocks
        if not mask[start:end].any():
            continue
        mod = mod_registry[mod_idx] if mod_idx < mod_registry.num_formats else None
        se = mod.spectral_efficiency if mod is not None else float(-mod_idx)
        # Smaller required FS wins; tie-break by higher SE, then smaller mod index.
        key = (int(req_fs), -float(se), mod_idx)
        if best_key is None or key < best_key:
            best_key = key
            best_mod = mod_idx
    return best_mod


def expanded_ksp_ff_rescue_action(
    obs: Dict[str, Any],
    min_path_idx: int = 50,
    max_path_idx: int = 499,
    mod_registry: Optional["ModulationRegistry"] = None,
) -> Optional[int]:
    """Deterministic expanded KSP-FF rescue over paths beyond the K=50 pool.

    Scans ``candidate_paths`` in the provided order (hops/km/lexicographic) and
    returns the first legal action whose ``path_idx`` is in
    ``[min_path_idx, max_path_idx]``.  On each path, the feasible modulation
    requiring the fewest FS is chosen (highest spectral efficiency on ties,
    then smallest mod index), and the lowest start-slot feasible block is used.

    Returns None if no expanded-path action is legal, or if the input
    observation has no candidate paths in the requested range.

    The function does not mutate ``obs``.
    """
    if mod_registry is None:
        from sa_hmarl.network.modulation import ModulationRegistry
        mod_registry = ModulationRegistry.from_profile("default")

    mask = np.asarray(obs["agent_r_mask"], dtype=bool)
    if not mask.any():
        return None

    num_paths = len(obs["candidate_paths"])
    num_mods = len(obs["mod_names"])
    if num_paths == 0 or num_mods == 0:
        return None
    num_blocks = len(mask) // (num_paths * num_mods)

    upper = min(max_path_idx + 1, num_paths)
    for path_idx in range(min_path_idx, upper):
        best_mod = _feasible_modulation_for_path(obs, path_idx, mod_registry)
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
