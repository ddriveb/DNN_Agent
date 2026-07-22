"""Explicit post-decision (afterstate) feature computation for R-side ranker.

Provides lightweight analytical/trial feature computation that does **not** require
deep-copying the whole environment.  It works by temporarily marking the slots that
a candidate R action would occupy on the relevant path edges, computing spectrum
statistics on the modified link-state view, and then discarding the view.

All functions are written to be usable both offline (dataset generation) and online
(inference), so the training and inference feature distributions stay aligned.
"""
from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# Names of the poststate_v1 feature schema, in fixed order.
POSTSTATE_V1_FEATURE_NAMES = [
    # --- pre-action / context features (same as current v1.3 base) ----------------
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
    "path_mod_feasible",
    "split_norm",
    "server_norm",
    "deadline_norm",
    "holding_norm",
    "intermediate_size_norm",
    "server_utilization",
    "selected_valid_r_ratio",
    "k_c_valid_ratio",
    "k_r_total_ratio",
    "phi_spec_norm",
    "raw_r_valid_ratio",
    "path_idx_norm",
    "mod_idx_norm",
    "block_idx_norm",
    # --- explicit afterstate features -------------------------------------------
    "path_lfb_after",
    "path_free_ratio_after",
    "global_lfb_after",
    "global_lfb_ratio_after",
    "frag_after",
    "delta_frag",
    "free_block_count_after",
    "free_block_count_delta",
    "min_edge_lfb_margin_after",
    "min_edge_free_ratio_after",
    "bottleneck_edge_util_after",
    "occupied_slot_hops",
    "path_conflict_after",
    "server_util_context",
    "server_margin_context",
    "holding_time_norm",
]


def _path_to_edges(path: Sequence[int]) -> List[Tuple[int, int]]:
    """Return normalized edge keys for a path."""
    return [(min(int(u), int(v)), max(int(u), int(v))) for u, v in zip(path[:-1], path[1:])]


def _edge_set(path: Sequence[int]) -> set:
    return set(_path_to_edges(path))


def _max_consecutive(arr: np.ndarray) -> int:
    if not np.any(arr):
        return 0
    max_len = curr = 0
    for v in arr:
        if v:
            curr += 1
            max_len = max(max_len, curr)
        else:
            curr = 0
    return max_len


def _count_blocks(arr: np.ndarray) -> int:
    count = 0
    in_block = False
    for v in arr:
        if v and not in_block:
            count += 1
            in_block = True
        elif not v:
            in_block = False
    return count


def _summarize_availability(free_arr: np.ndarray, num_slots: int) -> Dict[str, float]:
    total_free = int(np.sum(free_arr))
    max_free = _max_consecutive(free_arr)
    free_blocks = _count_blocks(free_arr)
    if total_free == 0:
        frag = 1.0
    elif total_free == num_slots:
        frag = 0.0
    else:
        frag = 1.0 - (max_free / total_free)
    return {
        "total_free_slots": total_free,
        "max_free_slots": max_free,
        "largest_free_block_ratio": max_free / max(num_slots, 1),
        "free_block_count": free_blocks,
        "frag_index": float(frag),
        "free_ratio": total_free / max(num_slots, 1),
    }


def _apply_allocation_to_link_states(
    link_states: Dict[Tuple[int, int], np.ndarray],
    path_edges: Sequence[Tuple[int, int]],
    start_slot: int,
    num_slots: int,
) -> Dict[Tuple[int, int], np.ndarray]:
    """Return a new link-state dict with the candidate allocation applied.

    The caller owns the returned dict; modifying it does not affect ``env``.
    """
    new_states = {}
    for key, slots in link_states.items():
        if key in path_edges:
            arr = slots.copy()
            arr[start_slot:start_slot + num_slots] = True
            new_states[key] = arr
        else:
            # Re-use the original array for edges not touched by the action.
            new_states[key] = slots
    return new_states


def _path_availability(link_states: Dict[Tuple[int, int], np.ndarray],
                       path_edges: Sequence[Tuple[int, int]],
                       num_slots: int) -> np.ndarray:
    avail = np.ones(num_slots, dtype=bool)
    for key in path_edges:
        avail &= ~link_states[key]
    return avail


def _global_spectrum_stats(link_states: Dict[Tuple[int, int], np.ndarray],
                           num_slots: int) -> Dict[str, float]:
    total_slots = len(link_states) * num_slots
    occupied = 0
    frag_vals = []
    max_free_vals = []
    free_block_vals = []
    for slots in link_states.values():
        occupied += int(np.sum(slots))
        stats = _summarize_availability(~slots, num_slots)
        frag_vals.append(stats["frag_index"])
        max_free_vals.append(stats["max_free_slots"])
        free_block_vals.append(stats["free_block_count"])
    return {
        "spectrum_utilization": occupied / max(total_slots, 1),
        "avg_frag_index": float(np.mean(frag_vals)) if frag_vals else 0.0,
        "largest_free_block_ratio": (max(max_free_vals) / max(num_slots, 1)) if max_free_vals else 0.0,
        "avg_free_block_count": float(np.mean(free_block_vals)) if free_block_vals else 0.0,
        "num_links": len(link_states),
    }


def _per_edge_stats_after(
    link_states: Dict[Tuple[int, int], np.ndarray],
    path_edges: Sequence[Tuple[int, int]],
    num_slots: int,
) -> Dict[str, float]:
    """Return bottleneck/min edge statistics over path edges after allocation."""
    min_lfb_margin = num_slots
    min_free_ratio = 1.0
    max_util = 0.0
    for key in path_edges:
        free_arr = ~link_states[key]
        stats = _summarize_availability(free_arr, num_slots)
        lfb_margin = stats["max_free_slots"]
        free_ratio = stats["free_ratio"]
        util = 1.0 - free_ratio
        if lfb_margin < min_lfb_margin:
            min_lfb_margin = lfb_margin
        if free_ratio < min_free_ratio:
            min_free_ratio = free_ratio
        if util > max_util:
            max_util = util
    return {
        "min_edge_lfb_margin": float(min_lfb_margin),
        "min_edge_free_ratio": float(min_free_ratio),
        "bottleneck_edge_util": float(max_util),
    }


def _path_conflict_count(env, path_edges: set) -> int:
    """Count active connections whose path shares at least one edge with path_edges."""
    active = getattr(env, "active_connections", [])
    count = 0
    for _release_time, _counter, conn in active:
        conn_path = conn.get("path")
        if not conn_path:
            continue
        conn_edges = _edge_set(conn_path)
        if conn_edges & path_edges:
            count += 1
    return count


def compute_optical_afterstate(
    env,
    path_idx: int,
    mod_idx: int,
    block_idx: int,
    obs_r: Dict[str, Any],
) -> Dict[str, float]:
    """Compute optical afterstate features for a candidate R action without mutating env."""
    path = obs_r["candidate_paths"][path_idx]
    path_edges = _path_to_edges(path)
    path_edge_set = set(path_edges)
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    if block_idx >= len(blocks):
        # Should not happen for legal actions; return zeros.
        return {
            "path_lfb_after": 0.0,
            "path_free_ratio_after": 0.0,
            "global_lfb_after": 0.0,
            "global_lfb_ratio_after": 0.0,
            "frag_after": 1.0,
            "delta_frag": 0.0,
            "free_block_count_after": 0.0,
            "free_block_count_delta": 0.0,
            "min_edge_lfb_margin_after": 0.0,
            "min_edge_free_ratio_after": 0.0,
            "bottleneck_edge_util_after": 1.0,
            "occupied_slot_hops": 0.0,
            "path_conflict_after": 0.0,
        }
    start_slot = int(blocks[block_idx][0])
    # The environment allocates the required FS, not the whole free-block size.
    fs_req = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    num_slots = int(fs_req) if fs_req is not None else int(blocks[block_idx][1])
    num_total_slots = int(env.net.num_slots)

    # Pre-state stats.
    avail_before = env.net.get_available_slots(path)
    stats_before_path = _summarize_availability(avail_before, num_total_slots)
    stats_before_global = _global_spectrum_stats(env.net.link_states, num_total_slots)

    # Post-state stats.
    post_link_states = _apply_allocation_to_link_states(
        env.net.link_states, path_edges, start_slot, num_slots
    )
    avail_after = _path_availability(post_link_states, path_edges, num_total_slots)
    stats_after_path = _summarize_availability(avail_after, num_total_slots)
    stats_after_global = _global_spectrum_stats(post_link_states, num_total_slots)
    edge_stats = _per_edge_stats_after(post_link_states, path_edges, num_total_slots)

    return {
        "path_lfb_after": float(stats_after_path["max_free_slots"]),
        "path_free_ratio_after": float(stats_after_path["free_ratio"]),
        "global_lfb_after": float(stats_after_global["largest_free_block_ratio"]) * num_total_slots,
        "global_lfb_ratio_after": float(stats_after_global["largest_free_block_ratio"]),
        "frag_after": float(stats_after_global["avg_frag_index"]),
        "delta_frag": float(stats_after_global["avg_frag_index"] - stats_before_global["avg_frag_index"]),
        "free_block_count_after": float(stats_after_global["avg_free_block_count"]),
        "free_block_count_delta": float(stats_after_global["avg_free_block_count"] - stats_before_global["avg_free_block_count"]),
        "min_edge_lfb_margin_after": float(edge_stats["min_edge_lfb_margin"]),
        "min_edge_free_ratio_after": float(edge_stats["min_edge_free_ratio"]),
        "bottleneck_edge_util_after": float(edge_stats["bottleneck_edge_util"]),
        "occupied_slot_hops": float(max(len(path_edges), 1) * num_slots),
        "path_conflict_after": float(_path_conflict_count(env, path_edge_set)),
    }


def _compute_spectrum_field(obs_c: Dict[str, Any], util_threshold: float = 0.95) -> Tuple[int, int]:
    """Compute K_C_valid and K_R_total for one request observation."""
    from sa_hmarl.agents.c_agent import AgentC

    num_servers = len(obs_c["server_utilizations"])
    feasible_counts = obs_c["feasible_counts"]
    k_c_valid = 0
    k_r_total = 0
    for split_id, server_list in enumerate(feasible_counts):
        for server_id, count in enumerate(server_list):
            cand_idx = split_id * num_servers + server_id
            feat_dict = obs_c["candidate_features"][cand_idx]
            available = float(feat_dict.get("server_available_compute", 0.0))
            edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
            edge_ms = feat_dict.get("edge_compute_ms", 0.0)
            compute_ok = (edge_cost <= available) and np.isfinite(edge_ms) and edge_ms != float("inf")
            util = float(obs_c["server_utilizations"][server_id])
            util_ok = util <= util_threshold
            diag = AgentC.compute_r_feasibility_diagnostics(obs_c, feat_dict)
            n_r = int(diag.get("valid_r_actions", 0))
            if compute_ok and util_ok and n_r > 0:
                k_c_valid += 1
            k_r_total += n_r
    return k_c_valid, k_r_total


def _phi_spec(k_c_valid: int, k_r_total: int, alpha: float) -> float:
    return float(np.log1p(k_c_valid) + alpha * np.log1p(k_r_total))


def build_poststate_v1_feature(
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
    """Build the full poststate_v1 feature vector for one candidate R action."""
    if feature_names is None:
        feature_names = POSTSTATE_V1_FEATURE_NAMES
    feature_names = list(feature_names)

    num_paths = max(len(obs_r["candidate_paths"]), 1)
    num_mods = max(len(obs_r["mod_names"]), 1)
    max_blocks = max(int(env.max_blocks), 1)
    path_idx = action_idx // (num_mods * max_blocks)
    rem = action_idx % (num_mods * max_blocks)
    mod_idx = rem // max_blocks
    block_idx = rem % max_blocks

    # Base pre-action features from PPO-R feature builder.
    base = np.asarray(r_features[action_idx], dtype=np.float32)

    # Context features (same as current v1.3).
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
        k_c / num_c,
        k_r / max_r_total,
        _phi_spec(k_c, k_r, 0.3) / 8.0,
        int(raw_r_mask.sum()) / max(len(raw_r_mask), 1),
    ], dtype=np.float32)

    action = np.asarray([
        path_idx / max(num_paths - 1, 1),
        mod_idx / max(num_mods - 1, 1),
        block_idx / max(max_blocks - 1, 1),
    ], dtype=np.float32)

    # Optical afterstate features.
    optical_after = compute_optical_afterstate(env, path_idx, mod_idx, block_idx, obs_r)

    # Server context (same for all R candidates under fixed a_C).
    server_util = float(obs_c["server_utilizations"][server_id])
    server_features = {
        "server_util_context": server_util,
        "server_margin_context": 1.0 - server_util,
        "holding_time_norm": float(req.holding_time) / 100.0,
    }

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
        "path_mod_feasible": float(base[10]),
        "split_norm": float(context[0]),
        "server_norm": float(context[1]),
        "deadline_norm": float(context[2]),
        "holding_norm": float(context[3]),
        "intermediate_size_norm": float(context[4]),
        "server_utilization": float(context[5]),
        "selected_valid_r_ratio": float(context[6]),
        "k_c_valid_ratio": float(field[0]),
        "k_r_total_ratio": float(field[1]),
        "phi_spec_norm": float(field[2]),
        "raw_r_valid_ratio": float(field[3]),
        "path_idx_norm": float(action[0]),
        "mod_idx_norm": float(action[1]),
        "block_idx_norm": float(action[2]),
    }
    feature_values.update(optical_after)
    feature_values.update(server_features)

    vector = np.asarray([feature_values.get(name, 0.0) for name in feature_names], dtype=np.float32)
    if not np.all(np.isfinite(vector)):
        raise ValueError(f"Non-finite poststate_v1 feature for action {action_idx}")
    return vector


def build_poststate_v1_feature_batch(
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
    """Vectorized wrapper that builds poststate_v1 features for a list of actions."""
    return np.stack([
        build_poststate_v1_feature(
            env, req, obs_c, obs_r, r_features, int(idx), split_id, server_id,
            feature_names=feature_names,
        )
        for idx in action_indices
    ]).astype(np.float32)


__all__ = [
    "POSTSTATE_V1_FEATURE_NAMES",
    "build_poststate_v1_feature",
    "build_poststate_v1_feature_batch",
    "compute_optical_afterstate",
]
