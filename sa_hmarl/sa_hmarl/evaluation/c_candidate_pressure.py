"""C-side candidate pressure diagnostic.

For each Agent-C candidate action ``(split, server)``, build the corresponding
Agent-R observation and extract spectrum-pressure metrics **without mutating**
environment state.  This module is pure analysis — it does not depend on
training logic and does not modify any existing interface.

.. important::
   All spectrum-pressure metrics are computed **exclusively from valid R
   actions** (``agent_r_mask == True``).  Path×mod combos that are
   modulation-feasible but have no large-enough spectrum block are excluded.

Metrics computed per candidate
------------------------------
- ``n_valid_r_actions``           : count of valid R (path, mod, block) tuples
- ``min_required_fs``             : smallest required FS among **valid** R actions
- ``best_lfb``                    : largest block size among **valid** R actions
- ``fs_lfb_ratio``                : min_required_fs / best_lfb
- ``valid_action_min_fs``         : same as ``min_required_fs`` (explicit name)
- ``valid_action_best_block``     : same as ``best_lfb`` (explicit name)
- ``valid_action_min_fs_block_ratio`` : same as ``fs_lfb_ratio`` (explicit name)
- ``min_path_len_km``             : shortest distance among paths with ≥1 valid R action
- ``avg_path_len_km``             : mean distance among paths with ≥1 valid R action
- ``valid_path_min_delay_ms``     : min transmission delay across valid-R paths
- ``feasible_mod_count``          : number of modulations with ≥1 valid R action
- ``frag_pressure``               : composite fragmentation pressure (higher = worse)
- ``delay_slack_ms``              : deadline − estimated delay (uses valid-R paths)
- ``server_utilization``          : current utilisation of the target edge server
- ``high_pressure``               : bool — True when the candidate looks risky

Functions
---------
- ``compute_c_candidate_pressure(env, req, split_id, server_id) -> dict``
- ``compute_all_c_candidate_pressures(env, req) -> list[dict]``
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Set

import numpy as np

from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.fs_demand import (
    DEFAULT_PROP_SPEED_KM_S,
    DEFAULT_PROC_PER_HOP_S,
    DEFAULT_SETUP_TIME_S,
)
from sa_hmarl.env.observation_builder import build_agent_r_observation
from sa_hmarl.env.request import DNNRequest


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _decode_valid_actions(
    obs_r: Dict[str, Any],
) -> Dict[str, Any]:
    """Extract valid-action metadata from an Agent-R observation.

    Returns a dict with keys:
        valid_path_indices : Set[int] — paths with ≥1 valid (mod, block)
        valid_mod_indices  : Set[int] — modulations with ≥1 valid (path, block)
        valid_fs_values    : List[int] — required_fs for each valid (path, mod)
        valid_block_sizes  : List[int] — block sizes of each valid action
        valid_path_indices_list : List[int] — paths appearing in valid actions
            (may contain duplicates from multi-mod/block — useful for per-path
            stats)
    """
    mask: np.ndarray = obs_r["agent_r_mask"]
    num_mods = len(obs_r["mod_names"])
    num_blocks = _num_blocks_from_obs(obs_r)
    req_fs_matrix: List[List[Optional[int]]] = obs_r["required_fs_per_path_mod"]
    blocks_matrix: List[List[List[Any]]] = obs_r["candidate_blocks_per_path_mod"]

    valid_indices = np.where(mask)[0]

    valid_paths: Set[int] = set()
    valid_mods: Set[int] = set()
    valid_fs_vals: List[int] = []
    valid_block_sizes: List[int] = []
    valid_path_list: List[int] = []  # one entry per valid action

    # Track which (path, mod) pairs we've already counted for fs
    seen_path_mod: Set[tuple] = set()

    for flat in valid_indices:
        flat = int(flat)
        p = flat // (num_mods * num_blocks)
        rem = flat % (num_mods * num_blocks)
        m = rem // num_blocks
        b = rem % num_blocks

        valid_paths.add(p)
        valid_mods.add(m)
        valid_path_list.append(p)

        # required_fs for this (path, mod) — count once per unique pair
        if (p, m) not in seen_path_mod:
            seen_path_mod.add((p, m))
            try:
                fs = req_fs_matrix[p][m]
                if fs is not None and fs > 0:
                    valid_fs_vals.append(int(fs))
            except (IndexError, TypeError):
                pass

        # Block size
        try:
            block_tuple = blocks_matrix[p][m][b]
            if isinstance(block_tuple, (tuple, list)) and len(block_tuple) >= 2:
                valid_block_sizes.append(int(block_tuple[1]))  # (start, size)
        except (IndexError, TypeError):
            pass

    return {
        "valid_path_indices": valid_paths,
        "valid_mod_indices": valid_mods,
        "valid_fs_values": valid_fs_vals,
        "valid_block_sizes": valid_block_sizes,
        "valid_path_indices_list": valid_path_list,
    }


def _num_blocks_from_obs(obs_r: Dict[str, Any]) -> int:
    """Infer max_blocks from the observation shape."""
    mask = obs_r["agent_r_mask"]
    num_mods = len(obs_r["mod_names"])
    num_paths = len(obs_r.get("candidate_paths", []))
    if num_paths > 0 and num_mods > 0:
        return len(mask) // (num_paths * num_mods)
    # Fallback: count blocks from first path×mod that has any
    blocks_matrix = obs_r.get("candidate_blocks_per_path_mod", [])
    for p_blocks in blocks_matrix:
        for m_blocks in p_blocks:
            if m_blocks:
                return len(m_blocks)
    return 1


def _path_transmission_delay_ms(path_features: Dict[str, Any]) -> float:
    """Compute transmission delay for a single path (ms).

    prop + proc + setup, matching _handle_arrival conventions.
    """
    path_km = float(path_features.get("path_length_km", 0.0))
    hops = int(path_features.get("hop_count", 0))
    prop_ms = (path_km / DEFAULT_PROP_SPEED_KM_S) * 1000.0
    proc_ms = hops * DEFAULT_PROC_PER_HOP_S * 1000.0
    setup_ms = DEFAULT_SETUP_TIME_S * 1000.0
    return prop_ms + proc_ms + setup_ms


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_c_candidate_pressure(
    env: SMDPEnv,
    req: DNNRequest,
    split_id: int,
    server_id: int,
    agent_r: Any = None,
) -> Dict[str, Any]:
    """Compute spectrum-pressure metrics for a single (split, server) candidate.

    All core spectrum metrics are derived **exclusively** from valid R actions
    (``agent_r_mask == True``).  Path×mod combos that lack a large-enough
    spectrum block do not contribute to ``min_required_fs``, ``best_lfb``,
    ``fs_lfb_ratio``, path lengths, fragmentation pressure, or delay slack.

    Args:
        env: Current SMDP environment (not mutated).
        req: The DNN request to evaluate.
        split_id: Index into ``req.splits``.
        server_id: Index into ``env.mec.servers``.
        agent_r: Unused; reserved for future R-policy-aware pressure.

    Returns:
        Dict with all pressure metrics.
    """
    # --- Build R observation (read-only) ------------------------------------
    obs_r = build_agent_r_observation(env, req, split_id, server_id)

    mask: np.ndarray = obs_r["agent_r_mask"]
    path_features: List[Dict[str, Any]] = obs_r["path_features"]
    mod_names: List[str] = obs_r["mod_names"]
    paths: List[List[int]] = obs_r["candidate_paths"]

    # --- Decode valid actions ------------------------------------------------
    va = _decode_valid_actions(obs_r)
    valid_paths: Set[int] = va["valid_path_indices"]
    valid_mods: Set[int] = va["valid_mod_indices"]
    valid_fs_vals: List[int] = va["valid_fs_values"]
    valid_block_sizes: List[int] = va["valid_block_sizes"]

    n_valid = int(np.sum(mask))

    # --- Core spectrum metrics (valid-R only) --------------------------------
    if n_valid > 0:
        min_required_fs = int(min(valid_fs_vals))
        best_lfb = int(max(valid_block_sizes))
        fs_lfb_ratio = float(min_required_fs) / max(float(best_lfb), 1.0)

        # Path lengths: only paths with ≥1 valid R action
        valid_path_lens = [
            float(path_features[p]["path_length_km"]) for p in valid_paths
        ]
        min_path_len_km = float(min(valid_path_lens))
        avg_path_len_km = float(np.mean(valid_path_lens))

        # Frag pressure: frag_index of valid paths
        valid_frag_indices = [
            float(path_features[p].get("frag_index", 0.0)) for p in valid_paths
        ]
        mean_frag = float(np.mean(valid_frag_indices))

        # Valid path min delay
        valid_path_delays = [
            _path_transmission_delay_ms(path_features[p]) for p in valid_paths
        ]
        valid_path_min_delay_ms = float(min(valid_path_delays))

    else:
        min_required_fs = 0
        best_lfb = 0
        fs_lfb_ratio = float("inf")
        min_path_len_km = float("inf")
        avg_path_len_km = float("inf")
        mean_frag = 0.0
        valid_path_min_delay_ms = float("inf")

    feasible_mod_count = len(valid_mods)
    frag_pressure = fs_lfb_ratio * (1.0 + mean_frag) if n_valid > 0 else float("inf")

    # --- delay_slack_ms (uses valid-R path min delay) ------------------------
    split = req.splits[split_id]
    server = env.mec.servers[server_id]

    edge_compute_ms = env._estimate_compute_ms(server, split.edge_compute_cost)
    local_compute_ms = env._estimate_local_compute_ms(server, split.local_compute_cost)

    if n_valid > 0:
        total_est_delay_ms = (
            valid_path_min_delay_ms
            + (edge_compute_ms if edge_compute_ms != float("inf") else 0.0)
            + (local_compute_ms if local_compute_ms != float("inf") else 0.0)
        )
        delay_slack_ms = float(req.deadline_ms - total_est_delay_ms)
    else:
        delay_slack_ms = float("-inf")

    # --- server_utilization --------------------------------------------------
    server_utilization = float(server.utilization)

    # --- high_pressure -------------------------------------------------------
    high_pressure = (
        n_valid == 0
        or (n_valid > 0 and fs_lfb_ratio > 0.8)
        or delay_slack_ms < 0.0
    )

    return {
        "split_id": int(split_id),
        "server_id": int(server_id),
        # counts
        "n_valid_r_actions": n_valid,
        "feasible_mod_count": feasible_mod_count,
        # FS / block metrics (valid-R only)
        "min_required_fs": min_required_fs,
        "best_lfb": best_lfb,
        "fs_lfb_ratio": fs_lfb_ratio if n_valid > 0 else float("inf"),
        # Explicit valid-action names
        "valid_action_min_fs": min_required_fs,
        "valid_action_best_block": best_lfb,
        "valid_action_min_fs_block_ratio": (
            fs_lfb_ratio if n_valid > 0 else float("inf")
        ),
        # path metrics (valid-R paths only)
        "min_path_len_km": min_path_len_km,
        "avg_path_len_km": avg_path_len_km,
        "valid_path_min_delay_ms": valid_path_min_delay_ms,
        # pressure
        "frag_pressure": frag_pressure,
        "delay_slack_ms": delay_slack_ms,
        "server_utilization": server_utilization,
        "high_pressure": high_pressure,
    }


def compute_all_c_candidate_pressures(
    env: SMDPEnv,
    req: DNNRequest,
    agent_r: Any = None,
) -> List[Dict[str, Any]]:
    """Compute pressure metrics for every valid Agent-C candidate.

    Iterates over all ``(split, server)`` pairs that are valid under the
    Agent-C action mask and returns a list of pressure dicts.

    Args:
        env: Current SMDP environment.
        req: The DNN request to evaluate.
        agent_r: Unused; reserved for future use.

    Returns:
        List of pressure dicts, one per valid C candidate, in split-major
        order (split 0 → all servers, split 1 → all servers, …).
    """
    from sa_hmarl.env.observation_builder import build_agent_c_observation

    obs_c = build_agent_c_observation(env, req)
    c_mask: np.ndarray = obs_c["agent_c_mask"]
    num_servers = len(env.mec.servers)

    results: List[Dict[str, Any]] = []
    for flat_idx in np.where(c_mask)[0]:
        flat_idx = int(flat_idx)
        split_id = flat_idx // num_servers
        server_id = flat_idx % num_servers
        pressure = compute_c_candidate_pressure(
            env, req, split_id, server_id, agent_r=agent_r,
        )
        pressure["flat_idx"] = flat_idx
        results.append(pressure)

    return results
