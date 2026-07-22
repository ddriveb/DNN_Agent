"""Compact post-decision (afterstate) feature schema v2 for SA-HMARL v1.35 corrected validation.

This module builds on `r_poststate_features.py` but defines a smaller, non-redundant
schema (`poststate_compact_v2`) intended for the corrected dataset.

Key differences from poststate_v1:
- Allocates `required_fs` slots, not the free-block size.
- Drops exact-duplicate features.
- Drops constant features.
- Avoids `max(link_lfb)` global statistics (they stay near num_slots).
- Uses mean / min / percentile global statistics instead.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    _apply_allocation_to_link_states,
    _compute_spectrum_field,
    _global_spectrum_stats,
    _path_availability,
    _path_conflict_count,
    _path_to_edges,
    _per_edge_stats_after,
    _phi_spec,
    _summarize_availability,
    compute_optical_afterstate,
)


# Compact v2 schema.  Dimension target: 28-33.
POSTSTATE_COMPACT_V2_FEATURE_NAMES = [
    # --- pre-action / context features (deduplicated v1 base) ------------------
    "path_length_km",
    "hop_count",
    "lfb",
    "free_ratio",
    "frag_index",
    "spectral_efficiency",
    "reach_km",
    "required_fs",
    "block_size",
    "block_waste",
    "split_norm",
    "server_norm",
    "deadline_norm",
    "holding_norm",
    "intermediate_size_norm",
    "server_utilization",
    "selected_valid_r_ratio",
    "k_r_total_ratio",
    "phi_spec_norm",
    "raw_r_valid_ratio",
    "path_idx_norm",
    "block_idx_norm",
    # --- compact afterstate features -------------------------------------------
    "delta_frag",
    "free_block_count_delta",
    "occupied_slot_hops_norm",
    "path_conflict_after_norm",
    "path_free_ratio_after",
    "min_edge_free_ratio_after",
    "global_frag_delta",
    "global_free_ratio_mean_after",
    "global_free_ratio_p10_after",
    "global_lfb_mean_after",
    "global_lfb_p10_after",
]


def _percentile_float(arr: np.ndarray, q: float) -> float:
    arr = np.asarray(arr)
    if arr.size == 0:
        return 0.0
    return float(np.percentile(arr, q))


def _global_spectrum_stats_v2(
    link_states: Dict[Tuple[int, int], np.ndarray],
    num_slots: int,
) -> Dict[str, float]:
    """Return richer global spectrum stats (mean/min/percentile) per link."""
    frag_vals = []
    free_ratio_vals = []
    lfb_vals = []
    free_block_vals = []
    for slots in link_states.values():
        stats = _summarize_availability(~slots, num_slots)
        frag_vals.append(stats["frag_index"])
        free_ratio_vals.append(stats["free_ratio"])
        lfb_vals.append(stats["max_free_slots"])
        free_block_vals.append(stats["free_block_count"])

    frag_arr = np.asarray(frag_vals)
    free_ratio_arr = np.asarray(free_ratio_vals)
    lfb_arr = np.asarray(lfb_vals)
    free_block_arr = np.asarray(free_block_vals)

    return {
        "avg_frag_index": float(np.mean(frag_arr)) if frag_arr.size else 0.0,
        "min_frag_index": float(np.min(frag_arr)) if frag_arr.size else 0.0,
        "p10_frag_index": _percentile_float(frag_arr, 10.0),
        "avg_free_ratio": float(np.mean(free_ratio_arr)) if free_ratio_arr.size else 0.0,
        "min_free_ratio": float(np.min(free_ratio_arr)) if free_ratio_arr.size else 0.0,
        "p10_free_ratio": _percentile_float(free_ratio_arr, 10.0),
        "avg_lfb": float(np.mean(lfb_arr)) if lfb_arr.size else 0.0,
        "min_lfb": float(np.min(lfb_arr)) if lfb_arr.size else 0.0,
        "p10_lfb": _percentile_float(lfb_arr, 10.0),
        "avg_free_block_count": float(np.mean(free_block_arr)) if free_block_arr.size else 0.0,
    }


def compute_optical_afterstate_compact_v2(
    env,
    path_idx: int,
    mod_idx: int,
    block_idx: int,
    obs_r: Dict[str, Any],
) -> Dict[str, float]:
    """Compute compact optical afterstate features for a candidate R action."""
    path = obs_r["candidate_paths"][path_idx]
    path_edges = _path_to_edges(path)
    path_edge_set = set(path_edges)
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]

    num_total_slots = int(env.net.num_slots)
    max_hops = max(len(path_edges), 1)
    active = getattr(env, "active_connections", [])
    total_active = max(len(active), 1)

    if block_idx >= len(blocks):
        return {
            "delta_frag": 0.0,
            "free_block_count_delta": 0.0,
            "occupied_slot_hops_norm": 0.0,
            "path_conflict_after_norm": 0.0,
            "path_free_ratio_after": 0.0,
            "min_edge_free_ratio_after": 0.0,
            "global_frag_delta": 0.0,
            "global_free_ratio_mean_after": 0.0,
            "global_free_ratio_p10_after": 0.0,
            "global_lfb_mean_after": 0.0,
            "global_lfb_p10_after": 0.0,
        }

    start_slot = int(blocks[block_idx][0])
    fs_req = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    num_slots = int(fs_req) if fs_req is not None else int(blocks[block_idx][1])

    # Pre-state stats.
    avail_before = env.net.get_available_slots(path)
    stats_before_path = _summarize_availability(avail_before, num_total_slots)
    stats_before_global = _global_spectrum_stats_v2(env.net.link_states, num_total_slots)

    # Post-state stats.
    post_link_states = _apply_allocation_to_link_states(
        env.net.link_states, path_edges, start_slot, num_slots
    )
    avail_after = _path_availability(post_link_states, path_edges, num_total_slots)
    stats_after_path = _summarize_availability(avail_after, num_total_slots)
    stats_after_global = _global_spectrum_stats_v2(post_link_states, num_total_slots)
    edge_stats = _per_edge_stats_after(post_link_states, path_edges, num_total_slots)

    conflict_after = _path_conflict_count(env, path_edge_set)

    occupied_slot_hops = max_hops * num_slots
    # Normalize so that a 5-hop path consuming all 320 slots equals ~1.
    occupied_slot_hops_norm = occupied_slot_hops / max(num_total_slots * 5, 1)

    return {
        "delta_frag": float(stats_after_global["avg_frag_index"] - stats_before_global["avg_frag_index"]),
        "free_block_count_delta": float(stats_after_global["avg_free_block_count"] - stats_before_global["avg_free_block_count"]),
        "occupied_slot_hops_norm": float(occupied_slot_hops_norm),
        "path_conflict_after_norm": float(conflict_after / total_active),
        "path_free_ratio_after": float(stats_after_path["free_ratio"]),
        "min_edge_free_ratio_after": float(edge_stats["min_edge_free_ratio"]),
        "global_frag_delta": float(stats_after_global["avg_frag_index"] - stats_before_global["avg_frag_index"]),
        "global_free_ratio_mean_after": float(stats_after_global["avg_free_ratio"]),
        "global_free_ratio_p10_after": float(stats_after_global["p10_free_ratio"]),
        "global_lfb_mean_after": float(stats_after_global["avg_lfb"]),
        "global_lfb_p10_after": float(stats_after_global["p10_lfb"]),
    }


def build_poststate_compact_v2_feature(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    action_idx: int,
    split_id: int,
    server_id: int,
    feature_names: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Build the compact poststate v2 feature vector for one candidate R action."""
    if feature_names is None:
        feature_names = POSTSTATE_COMPACT_V2_FEATURE_NAMES
    feature_names = list(feature_names)

    num_paths = max(len(obs_r["candidate_paths"]), 1)
    num_mods = max(len(obs_r["mod_names"]), 1)
    max_blocks = max(int(env.max_blocks), 1)
    path_idx = action_idx // (num_mods * max_blocks)
    rem = action_idx % (num_mods * max_blocks)
    mod_idx = rem // max_blocks
    block_idx = rem % max_blocks

    base = np.asarray(r_features[action_idx], dtype=np.float32)

    raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    k_c, k_r = _compute_spectrum_field(obs_c, 0.95)
    num_c = max(len(obs_c["candidate_features"]), 1)
    max_r_total = max(num_c * num_paths * num_mods * max_blocks, 1)
    selected_valid = int(obs_c["feasible_counts"][split_id][server_id])
    c_ctx = obs_r.get("c_context", {})

    context = np.asarray([
        float(split_id) / max(len(req.splits) - 1, 1),
        float(server_id) / max(len(env.mec.servers) - 1, 1),
        float(req.deadline_ms) / 100.0,
        float(req.holding_time) / 10.0,
        float(c_ctx.get("intermediate_size_mb", 0.0)) / 100.0,
        float(obs_c["server_utilizations"][server_id]),
        selected_valid / max(num_paths * num_mods * max_blocks, 1),
    ], dtype=np.float32)

    field = np.asarray([
        k_r / max_r_total,
        _phi_spec(k_c, k_r, 0.3) / 8.0,
        int(raw_r_mask.sum()) / max(len(raw_r_mask), 1),
    ], dtype=np.float32)

    action = np.asarray([
        path_idx / max(num_paths - 1, 1),
        block_idx / max(max_blocks - 1, 1),
    ], dtype=np.float32)

    optical_after = compute_optical_afterstate_compact_v2(
        env, path_idx, mod_idx, block_idx, obs_r
    )

    feature_values = {
        "path_length_km": float(base[0]),
        "hop_count": float(base[1]),
        "lfb": float(base[2]),
        "free_ratio": float(base[3]),
        "frag_index": float(base[4]),
        "spectral_efficiency": float(base[5]),
        "reach_km": float(base[6]),
        "required_fs": float(base[7]),
        "block_size": float(base[8]),
        "block_waste": float(base[9]),
        "split_norm": float(context[0]),
        "server_norm": float(context[1]),
        "deadline_norm": float(context[2]),
        "holding_norm": float(context[3]),
        "intermediate_size_norm": float(context[4]),
        "server_utilization": float(context[5]),
        "selected_valid_r_ratio": float(context[6]),
        "k_r_total_ratio": float(field[0]),
        "phi_spec_norm": float(field[1]),
        "raw_r_valid_ratio": float(field[2]),
        "path_idx_norm": float(action[0]),
        "block_idx_norm": float(action[1]),
    }
    feature_values.update(optical_after)

    vector = np.asarray([feature_values.get(name, 0.0) for name in feature_names], dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Non-finite poststate_compact_v2 feature for action {action_idx}")
    return vector


def build_poststate_compact_v2_feature_batch(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    action_indices: Iterable[int],
    split_id: int,
    server_id: int,
    feature_names: Optional[Iterable[str]] = None,
) -> np.ndarray:
    """Vectorized wrapper that builds compact v2 features for a list of actions."""
    return np.stack([
        build_poststate_compact_v2_feature(
            env, req, obs_c, obs_r, r_features, int(idx), split_id, server_id,
            feature_names=feature_names,
        )
        for idx in action_indices
    ]).astype(np.float32)


__all__ = [
    "POSTSTATE_COMPACT_V2_FEATURE_NAMES",
    "compute_optical_afterstate_compact_v2",
    "build_poststate_compact_v2_feature",
    "build_poststate_compact_v2_feature_batch",
]
