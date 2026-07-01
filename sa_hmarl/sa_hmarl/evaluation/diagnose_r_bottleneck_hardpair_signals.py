"""Diagnostic: do bottleneck-edge / release-aware / hard-pair signals matter?

This script freezes the trained Agent-C and evaluates every legal R action at
each decision point in the 24-slot default3 scenario.  For each candidate it
computes:

  - basic action features (path km, required FS, modulation, block size, ...)
  - bottleneck-edge features (min LFB, pressure, post-allocation fragmentation)
  - release-aware features (soon-to-release FS, remaining holding times)
  - counterfactual H-step return (G_v12 and a hard return) via PPO-C + PPO-R
    rollout on the same request trace

It then correlates the new signals with the H-step return, mines hard pairs
(actions that look similar on current metrics but differ in future return), and
compares v1.2-ranker vs DeepRMSA selected actions.

No new model is trained and no checkpoint is modified.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
from scipy import stats

try:
    from sklearn.metrics import roc_auc_score

    _HAS_SKLEARN = True
except Exception:  # pragma: no cover
    _HAS_SKLEARN = False

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _rollout_future,
    _select_c_action_from_obs,
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.eval_r_post_decision_closed_loop import _load_deep_rmsa
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


# -----------------------------------------------------------------------------
# Return coefficients
# -----------------------------------------------------------------------------
V12_RETURN_COEFS = {
    "current_block": 3.0,
    "future_block": 4.0,
    "future_nsb": 3.0,
    "future_server_overload": 2.0,
    "delay": 0.03,
    "fs": 0.05,
}

# A harder return that explicitly penalizes path length / FS consumption.
HARD_RETURN_COEFS = {
    "current_block": 4.0,
    "future_block": 5.0,
    "future_nsb": 4.0,
    "future_server_overload": 2.0,
    "delay": 0.03,
    "fs": 0.05,
    "path_km": 0.5 / 1000.0,  # per km
    "required_fs": 0.5 / 24.0,  # per slot (num_slots=24)
}


# -----------------------------------------------------------------------------
# Data classes
# -----------------------------------------------------------------------------
@dataclass
class CandidateRecord:
    seed: int
    episode: int
    request: int
    req_id: int

    split_id: int
    server_id: int

    r_action_idx: int
    path_idx: int
    mod_idx: int
    block_idx: int

    selected_by_v12: bool = False
    selected_by_deep: bool = False

    # Basic features
    path_km: float = 0.0
    required_fs: int = 0
    required_fs_norm: float = 0.0
    mod_idx_val: int = 0
    spectral_efficiency: float = 0.0
    reach_km: float = 0.0
    block_size: int = 0
    block_waste: float = 0.0
    block_start: int = 0
    path_hop_count: int = 0

    # Bottleneck-edge features
    edge_utilization_mean: float = 0.0
    edge_utilization_max: float = 0.0
    edge_free_slot_ratio_mean: float = 0.0
    edge_free_slot_ratio_min: float = 0.0
    edge_largest_free_block_mean: float = 0.0
    edge_largest_free_block_min: float = 0.0
    min_edge_lfb_margin: float = 0.0
    bottleneck_lfb_pressure: float = 0.0
    path_pressure_sum: float = 0.0
    path_overlap_count: int = 0
    path_overlap_ratio: float = 0.0
    post_allocation_min_lfb: float = 0.0
    post_allocation_frag_delta_on_path: float = 0.0

    # Release-aware features
    soon_release_fs_1step: float = 0.0
    soon_release_fs_3step: float = 0.0
    avg_remaining_holding_on_path: float = 0.0
    max_remaining_holding_on_path: float = 0.0
    long_hold_occupied_ratio: float = 0.0
    post_lfb_after_near_release: float = 0.0

    # Outcomes / returns
    current_success: bool = False
    current_reason: str = ""
    delay_ms: float = 0.0

    future_blocked_count: int = 0
    future_nsb_count: int = 0
    future_server_overload_count: int = 0
    future_delay_mean: float = 0.0
    future_avg_fs: float = 0.0

    G_v12: float = 0.0
    G_hard: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            k: (
                bool(v)
                if isinstance(v, (np.bool_,))
                else (
                    int(v)
                    if isinstance(v, (np.integer,))
                    else (
                        float(v)
                        if isinstance(v, (np.floating,))
                        else v
                    )
                )
            )
            for k, v in self.__dict__.items()
        }


@dataclass
class StateRecord:
    seed: int
    episode: int
    request: int
    req_id: int

    c_mask_empty: bool = False
    r_mask_empty: bool = False
    r_mask_count: int = 0
    selected_split: int = 0
    selected_server: int = 0

    legal_r_indices: List[int] = field(default_factory=list)
    v12_r_idx: Optional[int] = None
    deep_r_idx: Optional[int] = None
    ppo_r_idx: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": int(self.seed),
            "episode": int(self.episode),
            "request": int(self.request),
            "req_id": int(self.req_id),
            "c_mask_empty": bool(self.c_mask_empty),
            "r_mask_empty": bool(self.r_mask_empty),
            "r_mask_count": int(self.r_mask_count),
            "selected_split": int(self.selected_split),
            "selected_server": int(self.selected_server),
            "legal_r_indices": [int(x) for x in self.legal_r_indices],
            "v12_r_idx": None if self.v12_r_idx is None else int(self.v12_r_idx),
            "deep_r_idx": None if self.deep_r_idx is None else int(self.deep_r_idx),
            "ppo_r_idx": None if self.ppo_r_idx is None else int(self.ppo_r_idx),
        }


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _edges_of_path(path: List[int]) -> List[Tuple[int, int]]:
    return [(min(u, v), max(u, v)) for u, v in zip(path[:-1], path[1:])]


def _edge_array_stats(env, path: List[int]) -> Dict[str, Any]:
    """Compute per-edge spectrum statistics along a path (pre-allocation)."""
    edges = _edges_of_path(path)
    if not edges:
        return {
            "edges": [],
            "utils": np.zeros(1, dtype=float),
            "free_ratios": np.ones(1, dtype=float),
            "lfbs": np.full(1, float(env.net.num_slots), dtype=float),
            "frags": np.zeros(1, dtype=float),
        }
    utils = []
    free_ratios = []
    lfbs = []
    frags = []
    for edge in edges:
        occupied = env.net.link_states[edge]
        free_arr = ~occupied
        s = env.net.summarize_availability(free_arr)
        utils.append(float(np.mean(occupied)))
        free_ratios.append(float(np.mean(free_arr)))
        lfbs.append(float(s["max_free_slots"]))
        frags.append(float(s["frag_index"]))
    return {
        "edges": edges,
        "utils": np.asarray(utils, dtype=float),
        "free_ratios": np.asarray(free_ratios, dtype=float),
        "lfbs": np.asarray(lfbs, dtype=float),
        "frags": np.asarray(frags, dtype=float),
    }


def _post_allocation_edge_stats(env, path: List[int]) -> Dict[str, Any]:
    """Edge stats after an allocation has already been applied to env."""
    edges = _edges_of_path(path)
    if not edges:
        return {
            "lfbs": np.full(1, float(env.net.num_slots), dtype=float),
            "frags": np.zeros(1, dtype=float),
        }
    lfbs = []
    frags = []
    for edge in edges:
        occupied = env.net.link_states[edge]
        free_arr = ~occupied
        s = env.net.summarize_availability(free_arr)
        lfbs.append(float(s["max_free_slots"]))
        frags.append(float(s["frag_index"]))
    return {
        "lfbs": np.asarray(lfbs, dtype=float),
        "frags": np.asarray(frags, dtype=float),
    }


def _path_overlap(env, path: List[int]) -> Tuple[int, float]:
    """Count active lightpaths that share at least one edge with path."""
    path_edges = set(_edges_of_path(path))
    count = 0
    for conn in env.active_connections:
        conn_edges = set(_edges_of_path(conn["path"]))
        if conn_edges & path_edges:
            count += 1
    ratio = count / max(len(env.active_connections), 1)
    return count, ratio


def _release_features(env, path: List[int], arrival_interval: float) -> Dict[str, float]:
    """Release-aware statistics for the candidate path."""
    path_edges = set(_edges_of_path(path))
    now = float(env.time)
    overlapping = []
    for conn in env.active_connections:
        conn_edges = set(_edges_of_path(conn["path"]))
        if conn_edges & path_edges:
            overlapping.append(conn)

    if not overlapping:
        return {
            "soon_release_fs_1step": 0.0,
            "soon_release_fs_3step": 0.0,
            "avg_remaining_holding_on_path": 0.0,
            "max_remaining_holding_on_path": 0.0,
            "long_hold_occupied_ratio": 0.0,
            "post_lfb_after_near_release": 0.0,
        }

    remainings = [float(conn["release_time"] - now) for conn in overlapping]
    median_remaining = float(np.median(remainings))

    soon_1 = sum(
        float(conn["num_slots"])
        for conn in overlapping
        if conn["release_time"] <= now + arrival_interval
    )
    soon_3 = sum(
        float(conn["num_slots"])
        for conn in overlapping
        if conn["release_time"] <= now + 3.0 * arrival_interval
    )

    long_count = sum(1 for r in remainings if r > median_remaining)

    # Compute min LFB along path if connections releasing within 1 step are freed.
    edges = _edges_of_path(path)
    min_lfb_after = float(env.net.num_slots)
    for edge in edges:
        occupied = env.net.link_states[edge].copy()
        for conn in overlapping:
            if conn["release_time"] <= now + arrival_interval:
                start = conn["start_slot"]
                n = conn["num_slots"]
                # Only free slots that belong to this edge.
                edge_set = set(_edges_of_path(conn["path"]))
                if edge in edge_set:
                    occupied[start : start + n] = False
        free_arr = ~occupied
        lfb = env.net._max_consecutive(free_arr)
        min_lfb_after = min(min_lfb_after, float(lfb))

    return {
        "soon_release_fs_1step": float(soon_1),
        "soon_release_fs_3step": float(soon_3),
        "avg_remaining_holding_on_path": float(np.mean(remainings)),
        "max_remaining_holding_on_path": float(np.max(remainings)),
        "long_hold_occupied_ratio": float(long_count) / len(remainings),
        "post_lfb_after_near_release": float(min_lfb_after),
    }


# -----------------------------------------------------------------------------
# Return computation
# -----------------------------------------------------------------------------
def _compute_v12_return(info: Dict[str, Any], future: Dict[str, Any]) -> float:
    current_block = 0.0 if info.get("success", False) else 1.0
    return float(
        -V12_RETURN_COEFS["current_block"] * current_block
        - V12_RETURN_COEFS["future_block"] * float(future.get("blocked", 0))
        - V12_RETURN_COEFS["future_nsb"] * float(future.get("no_suitable_block", 0))
        - V12_RETURN_COEFS["future_server_overload"] * float(future.get("server_overload", 0))
        - V12_RETURN_COEFS["delay"] * float(future.get("delay_mean", 0.0))
        - V12_RETURN_COEFS["fs"] * float(future.get("avg_fs", 0.0))
    )


def _compute_hard_return(
    info: Dict[str, Any], future: Dict[str, Any], path_km: float, required_fs: int
) -> float:
    current_block = 0.0 if info.get("success", False) else 1.0
    return float(
        -HARD_RETURN_COEFS["current_block"] * current_block
        - HARD_RETURN_COEFS["future_block"] * float(future.get("blocked", 0))
        - HARD_RETURN_COEFS["future_nsb"] * float(future.get("no_suitable_block", 0))
        - HARD_RETURN_COEFS["future_server_overload"] * float(future.get("server_overload", 0))
        - HARD_RETURN_COEFS["delay"] * float(future.get("delay_mean", 0.0))
        - HARD_RETURN_COEFS["fs"] * float(future.get("avg_fs", 0.0))
        - HARD_RETURN_COEFS["path_km"] * float(path_km)
        - HARD_RETURN_COEFS["required_fs"] * float(required_fs)
    )


# -----------------------------------------------------------------------------
# Per-candidate feature builder
# -----------------------------------------------------------------------------
def _build_candidate_record(
    env_snapshot,
    req,
    obs_r: Dict[str, Any],
    agent_r_for_features,
    agent_c,
    agent_r,
    ranker_model,
    ranker_mean,
    ranker_std,
    num_servers: int,
    arrival_interval: float,
    horizon: int,
    util_threshold: float,
    alpha: float,
    requests: List[Any],
    step_idx: int,
    split_id: int,
    server_id: int,
    r_action_idx: int,
    state: StateRecord,
    device: str,
) -> CandidateRecord:
    """Evaluate one legal R action and return a populated CandidateRecord."""
    num_mods = len(obs_r["mod_names"])
    path_idx, mod_idx, block_idx = decode_agent_r_action(
        int(r_action_idx), num_mods, env_snapshot.max_blocks
    )

    path = obs_r["candidate_paths"][path_idx]
    required_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    block_tuples = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    start_slot, block_size = block_tuples[block_idx] if block_idx < len(block_tuples) else (0, 0)

    # Pre-allocation bottleneck features from the snapshot.
    pre_stats = _edge_array_stats(env_snapshot, path)
    overlap_count, overlap_ratio = _path_overlap(env_snapshot, path)

    # Simulate the candidate action.
    branch = copy.deepcopy(env_snapshot)
    r_action = (path_idx, mod_idx, block_idx)
    _, _, _, info = branch.step((split_id, server_id), r_action)
    success = bool(info.get("success", False))

    # Post-allocation bottleneck features.
    if success:
        post_stats = _post_allocation_edge_stats(branch, path)
        post_min_lfb = float(np.min(post_stats["lfbs"]))
        post_frag_delta = float(np.mean(post_stats["frags"]) - np.mean(pre_stats["frags"]))
    else:
        # Allocation did not happen; fall back to pre-allocation values.
        post_min_lfb = float(np.min(pre_stats["lfbs"]))
        post_frag_delta = 0.0

    # H-step future rollout (PPO-C + PPO-R, matching v1.2 dataset generation).
    future = _rollout_future(
        branch,
        requests,
        step_idx + 1,
        horizon,
        agent_c,
        agent_r,
        num_servers,
        util_threshold,
        alpha,
    )

    mod = branch.mod_reg[mod_idx]
    path_km = float(branch.net.path_length_km(path))
    rec = CandidateRecord(
        seed=state.seed,
        episode=state.episode,
        request=state.request,
        req_id=state.req_id,
        split_id=split_id,
        server_id=server_id,
        r_action_idx=int(r_action_idx),
        path_idx=path_idx,
        mod_idx=mod_idx,
        block_idx=block_idx,
        selected_by_v12=(state.v12_r_idx == int(r_action_idx)),
        selected_by_deep=(state.deep_r_idx == int(r_action_idx)),
        path_km=path_km,
        required_fs=int(required_fs),
        required_fs_norm=float(required_fs) / max(branch.net.num_slots, 1),
        mod_idx_val=mod_idx,
        spectral_efficiency=float(mod.spectral_efficiency),
        reach_km=float(mod.reach_km),
        block_size=int(block_size),
        block_waste=float(block_size - required_fs),
        block_start=int(start_slot),
        path_hop_count=max(0, len(path) - 1),
        edge_utilization_mean=float(np.mean(pre_stats["utils"])),
        edge_utilization_max=float(np.max(pre_stats["utils"])),
        edge_free_slot_ratio_mean=float(np.mean(pre_stats["free_ratios"])),
        edge_free_slot_ratio_min=float(np.min(pre_stats["free_ratios"])),
        edge_largest_free_block_mean=float(np.mean(pre_stats["lfbs"])),
        edge_largest_free_block_min=float(np.min(pre_stats["lfbs"])),
        min_edge_lfb_margin=float(np.min(pre_stats["lfbs"]) - required_fs),
        bottleneck_lfb_pressure=float(required_fs) / max(float(np.min(pre_stats["lfbs"])), 1.0),
        path_pressure_sum=float(np.sum(required_fs / np.maximum(pre_stats["lfbs"], 1.0))),
        path_overlap_count=int(overlap_count),
        path_overlap_ratio=float(overlap_ratio),
        post_allocation_min_lfb=post_min_lfb,
        post_allocation_frag_delta_on_path=post_frag_delta,
        **_release_features(env_snapshot, path, arrival_interval),
        current_success=success,
        current_reason=str(info.get("reason", "")),
        delay_ms=float(info.get("delay_ms", 0.0)) if success else 0.0,
        future_blocked_count=int(future.get("blocked", 0)),
        future_nsb_count=int(future.get("no_suitable_block", 0)),
        future_server_overload_count=int(future.get("server_overload", 0)),
        future_delay_mean=float(future.get("delay_mean", 0.0)),
        future_avg_fs=float(future.get("avg_fs", 0.0)),
    )
    rec.G_v12 = _compute_v12_return(info, future)
    rec.G_hard = _compute_hard_return(info, future, path_km, int(required_fs))
    return rec


# -----------------------------------------------------------------------------
# Policy selection helpers
# -----------------------------------------------------------------------------
def _select_v12_r_action(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    split_id: int,
    server_id: int,
    device: str,
) -> int:
    """Select the action with the highest v1.2 ranker score among legal actions."""
    r_features, r_mask = agent_r_for_features.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - ranker_mean) / ranker_std
    with torch.no_grad():
        scores = ranker_model(
            torch.as_tensor(normalized, dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()
    return int(legal[int(np.argmax(scores))])


def _select_ppo_r_action_idx(agent_r, obs_r: Dict[str, Any], max_blocks: int) -> int:
    raw_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    action = agent_r.select_action(obs_r, deterministic=True)
    if action is None:
        action = 0
    return int(action)


# -----------------------------------------------------------------------------
# Main diagnostic loop
# -----------------------------------------------------------------------------
def _snapshot_before_r_decision(env, expected_req_id: int):
    """Copy a state whose event-queue head is still the current request."""
    if not env.event_queue:
        raise RuntimeError("Cannot snapshot an empty event queue")
    queued_req = env.event_queue[0][2]
    if int(queued_req.req_id) != int(expected_req_id):
        raise RuntimeError(
            "R counterfactual snapshot is not pre-decision: "
            f"expected request {expected_req_id}, queue head is {queued_req.req_id}"
        )
    return copy.deepcopy(env)


def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg

    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading PPO-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    print(f"Loading v1.2 ranker: {args.ranking_checkpoint}")
    ranker_model, ranker_mean, ranker_std, _ = load_ranking_checkpoint(
        args.ranking_checkpoint, args.device
    )
    print(f"Loading DeepRMSA: {args.deep_rmsa_checkpoint}")
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

    # Generate request traces up front so every counterfactual sees the same future.
    all_episodes: Dict[int, List[List[Any]]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
                traffic_mode=args.traffic_mode,
            ))
        all_episodes[seed] = eps

    candidate_records: List[CandidateRecord] = []
    state_records: List[StateRecord] = []
    t_start = time.time()

    for seed in sorted(all_episodes.keys()):
        for ep_idx, requests in enumerate(all_episodes[seed]):
            env = make_env(
                topology=args.topology,
                num_slots=args.num_slots,
                num_servers=args.num_servers,
                seed=seed,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)

            for step_idx, req in enumerate(requests):
                env.advance_time(req.arrival_time)
                obs_c = build_agent_c_observation(env, req)
                raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                c_mask_empty = int(raw_c_mask.sum()) == 0

                c_action_idx, _, _ = _select_c_action_from_obs(
                    agent_c, obs_c, env.net.num_slots
                )
                split_id, server_id = decode_agent_c_action(c_action_idx, args.num_servers)

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
                legal = np.flatnonzero(raw_r_mask).tolist()

                state = StateRecord(
                    seed=seed,
                    episode=ep_idx,
                    request=step_idx,
                    req_id=int(req.req_id),
                    c_mask_empty=c_mask_empty,
                    r_mask_empty=(len(legal) == 0),
                    r_mask_count=len(legal),
                    selected_split=split_id,
                    selected_server=server_id,
                    legal_r_indices=legal,
                )

                # Snapshot before any R decision mutates the env.
                snapshot = _snapshot_before_r_decision(env, req.req_id)

                # Selected actions from the different policies.
                state.v12_r_idx = (
                    _select_v12_r_action(
                        env, req, obs_c, obs_r, agent_r,
                        ranker_model, ranker_mean, ranker_std,
                        split_id, server_id, args.device,
                    )
                    if legal
                    else None
                )
                state.deep_r_idx = (
                    deep_rmsa.select_action(obs_r)
                    if legal
                    else None
                )
                state.ppo_r_idx = (
                    _select_ppo_r_action_idx(agent_r, obs_r, env.max_blocks)
                    if legal
                    else None
                )

                # Evaluate every legal R candidate.
                for r_idx in legal:
                    rec = _build_candidate_record(
                        snapshot, req, obs_r, agent_r, agent_c, agent_r,
                        ranker_model, ranker_mean, ranker_std,
                        args.num_servers, args.arrival_interval, args.horizon,
                        args.util_threshold, args.alpha, requests, step_idx,
                        split_id, server_id, r_idx, state, args.device,
                    )
                    candidate_records.append(rec)

                # Advance the real environment using the v1.2 ranker action so the
                # diagnostic trajectory matches the final deployed method.
                if state.v12_r_idx is not None:
                    v12_action = decode_agent_r_action(
                        state.v12_r_idx, len(obs_r["mod_names"]), env.max_blocks
                    )
                else:
                    v12_action = (0, 0, 0)
                env.step((split_id, server_id), v12_action)

                state_records.append(state)

                if (step_idx + 1) % 20 == 0:
                    print(
                        f"[seed={seed} ep={ep_idx}] processed {step_idx + 1}/{len(requests)} requests, "
                        f"candidates so far={len(candidate_records)}",
                        flush=True,
                    )

    elapsed = time.time() - t_start
    print(f"Collected {len(candidate_records)} candidates from {len(state_records)} states in {elapsed:.1f}s")

    summary = _aggregate_summary(candidate_records, state_records, args)
    return {
        "config": _config_dict(args),
        "summary": summary,
        "state_records": [s.to_dict() for s in state_records],
        "candidate_records": [c.to_dict() for c in candidate_records],
        "elapsed_seconds": elapsed,
    }


# -----------------------------------------------------------------------------
# Aggregation and statistics
# -----------------------------------------------------------------------------
BOTTLENECK_FEATURES = [
    ("edge_utilization_mean", "lower_better"),
    ("edge_utilization_max", "lower_better"),
    ("edge_free_slot_ratio_mean", "higher_better"),
    ("edge_free_slot_ratio_min", "higher_better"),
    ("edge_largest_free_block_mean", "higher_better"),
    ("edge_largest_free_block_min", "higher_better"),
    ("min_edge_lfb_margin", "higher_better"),
    ("bottleneck_lfb_pressure", "lower_better"),
    ("path_pressure_sum", "lower_better"),
    ("path_overlap_count", "lower_better"),
    ("path_overlap_ratio", "lower_better"),
    ("post_allocation_min_lfb", "higher_better"),
    ("post_allocation_frag_delta_on_path", "lower_better"),
]

RELEASE_FEATURES = [
    ("soon_release_fs_1step", "higher_better"),
    ("soon_release_fs_3step", "higher_better"),
    ("avg_remaining_holding_on_path", "lower_better"),
    ("max_remaining_holding_on_path", "lower_better"),
    ("long_hold_occupied_ratio", "lower_better"),
    ("post_lfb_after_near_release", "higher_better"),
]

BASIC_FEATURES = [
    ("path_km", "lower_better"),
    ("required_fs_norm", "lower_better"),
    ("block_waste", "lower_better"),
]


def _corr_one(
    values: np.ndarray, target: np.ndarray, method: str
) -> Optional[float]:
    if len(values) < 3 or np.std(values) == 0 or np.std(target) == 0:
        return None
    if method == "pearson":
        r, _ = stats.pearsonr(values, target)
    else:
        r, _ = stats.spearmanr(values, target)
    return float(r)


def _partial_corr(
    x: np.ndarray, y: np.ndarray, controls: np.ndarray
) -> Optional[float]:
    """Partial correlation of x and y after residualizing on controls."""
    if len(x) < 4:
        return None
    controls = np.asarray(controls)
    if controls.ndim == 1:
        controls = controls.reshape(-1, 1)
    # Residualize y on controls.
    y_res = _residualize(y, controls)
    x_res = _residualize(x, controls)
    if np.std(y_res) == 0 or np.std(x_res) == 0:
        return None
    r, _ = stats.pearsonr(x_res, y_res)
    return float(r)


def _residualize(y: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Return residuals of y after OLS regression on X (with intercept)."""
    y = np.asarray(y, dtype=float)
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    Xb = np.column_stack([np.ones(len(X)), X])
    # Normal equations with small ridge for numerical stability.
    XtX = Xb.T @ Xb + 1e-6 * np.eye(Xb.shape[1])
    beta = np.linalg.solve(XtX, Xb.T @ y)
    return y - Xb @ beta


def _feature_correlation_table(
    records: List[CandidateRecord],
    features: List[Tuple[str, str]],
    target_name: str,
    target_extractor,
    control_names: Optional[List[str]] = None,
) -> Dict[str, Any]:
    rows = []
    target = np.asarray([target_extractor(r) for r in records], dtype=float)
    controls = None
    if control_names:
        controls = np.column_stack([
            np.asarray([getattr(r, n) for r in records], dtype=float)
            for n in control_names
        ])
    for name, direction in features:
        vals = np.asarray([getattr(r, name) for r in records], dtype=float)
        pearson = _corr_one(vals, target, "pearson")
        spearman = _corr_one(vals, target, "spearman")
        partial = None
        if controls is not None and len(vals) >= 4:
            partial = _partial_corr(vals, target, controls)
        rows.append({
            "feature": name,
            "direction": direction,
            "pearson": pearson,
            "spearman": spearman,
            "partial_pearson_vs_pathkm_requiredfs": partial,
            "n": int(len(vals)),
        })
    return {"target": target_name, "control_for_partial": control_names, "rows": rows}


def _hard_pairs(records: List[CandidateRecord], return_gap_threshold: float = 0.1):
    """Mine hard pairs among successful candidates in the same state."""
    successful = [r for r in records if r.current_success]
    pairs = []
    for i in range(len(successful)):
        a = successful[i]
        for j in range(i + 1, len(successful)):
            b = successful[j]
            if a.seed != b.seed or a.episode != b.episode or a.request != b.request:
                continue
            if abs(a.delay_ms - b.delay_ms) >= 1.0:
                continue
            if abs(a.block_waste - b.block_waste) >= 1.0:
                continue
            if (
                a.future_blocked_count != b.future_blocked_count
                or a.future_nsb_count != b.future_nsb_count
                or abs(a.G_v12 - b.G_v12) > return_gap_threshold
            ):
                pairs.append((a, b))
    return pairs


def _pair_auc_and_accuracy(
    pairs: List[Tuple[CandidateRecord, CandidateRecord]],
    feature_name: str,
    direction: str,
) -> Tuple[Optional[float], Optional[float]]:
    """Return (accuracy, auc) for predicting the higher-G_v12 action."""
    if not pairs:
        return None, None
    correct = 0
    labels = []
    scores = []
    for a, b in pairs:
        fa = getattr(a, feature_name)
        fb = getattr(b, feature_name)
        higher_return = a if a.G_v12 > b.G_v12 else b
        if direction == "lower_better":
            pred = a if fa < fb else b
            # Score: lower value -> higher score -> positive label = higher return.
            score_a = -fa
            score_b = -fb
        else:
            pred = a if fa > fb else b
            score_a = fa
            score_b = fb
        if pred is higher_return:
            correct += 1
        # Binary label: 1 if a is higher return, 0 otherwise.
        if a.G_v12 > b.G_v12:
            labels.extend([1, 0])
            scores.extend([score_a, score_b])
        elif b.G_v12 > a.G_v12:
            labels.extend([0, 1])
            scores.extend([score_a, score_b])
        # Ties in return contribute nothing to AUC.
    accuracy = correct / len(pairs)
    auc = None
    if len(labels) >= 2 and len(set(labels)) == 2 and _HAS_SKLEARN:
        try:
            auc = float(roc_auc_score(labels, scores))
        except Exception:
            auc = None
    return float(accuracy), auc


def _aggregate_v12_vs_deep(
    records: List[CandidateRecord],
) -> Dict[str, Any]:
    """Compare v1.2-ranker selected actions vs DeepRMSA selected actions."""
    states: Dict[Tuple[int, int, int], Dict[str, Any]] = {}
    for r in records:
        key = (r.seed, r.episode, r.request)
        states.setdefault(key, {"v12": None, "deep": None})
        if r.selected_by_v12:
            states[key]["v12"] = r
        if r.selected_by_deep:
            states[key]["deep"] = r

    diffs = []
    counts = {"v12_better": 0, "deep_better": 0, "same": 0, "incomparable": 0}
    for recs in states.values():
        v12 = recs["v12"]
        deep = recs["deep"]
        if v12 is None or deep is None:
            counts["incomparable"] += 1
            continue
        if not (v12.current_success and deep.current_success):
            counts["incomparable"] += 1
            continue
        if v12.G_v12 > deep.G_v12:
            counts["v12_better"] += 1
        elif deep.G_v12 > v12.G_v12:
            counts["deep_better"] += 1
        else:
            counts["same"] += 1
        diffs.append({
            "path_km": float(v12.path_km - deep.path_km),
            "required_fs_norm": float(v12.required_fs_norm - deep.required_fs_norm),
            "bottleneck_lfb_pressure": float(v12.bottleneck_lfb_pressure - deep.bottleneck_lfb_pressure),
            "path_pressure_sum": float(v12.path_pressure_sum - deep.path_pressure_sum),
            "min_edge_lfb_margin": float(v12.min_edge_lfb_margin - deep.min_edge_lfb_margin),
            "soon_release_fs_1step": float(v12.soon_release_fs_1step - deep.soon_release_fs_1step),
            "avg_remaining_holding_on_path": float(v12.avg_remaining_holding_on_path - deep.avg_remaining_holding_on_path),
            "G_v12": float(v12.G_v12 - deep.G_v12),
            "future_blocked_count": int(v12.future_blocked_count - deep.future_blocked_count),
            "future_nsb_count": int(v12.future_nsb_count - deep.future_nsb_count),
        })

    def _mean(values: List[float]) -> Optional[float]:
        return float(np.mean(values)) if values else None

    return {
        "comparable_states": len(diffs),
        "counts": counts,
        "mean_diff": {
            "path_km": _mean([d["path_km"] for d in diffs]),
            "required_fs_norm": _mean([d["required_fs_norm"] for d in diffs]),
            "bottleneck_lfb_pressure": _mean([d["bottleneck_lfb_pressure"] for d in diffs]),
            "path_pressure_sum": _mean([d["path_pressure_sum"] for d in diffs]),
            "min_edge_lfb_margin": _mean([d["min_edge_lfb_margin"] for d in diffs]),
            "soon_release_fs_1step": _mean([d["soon_release_fs_1step"] for d in diffs]),
            "avg_remaining_holding_on_path": _mean([d["avg_remaining_holding_on_path"] for d in diffs]),
            "G_v12": _mean([d["G_v12"] for d in diffs]),
            "future_blocked_count": _mean([d["future_blocked_count"] for d in diffs]),
            "future_nsb_count": _mean([d["future_nsb_count"] for d in diffs]),
        },
    }


def _aggregate_summary(
    records: List[CandidateRecord],
    state_records: List[StateRecord],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    total_states = len(state_records)
    multi_action_states = [s for s in state_records if s.r_mask_count >= 2]
    legal_counts = [s.r_mask_count for s in state_records]

    summary = {
        "total_states": total_states,
        "multi_action_states": len(multi_action_states),
        "multi_action_rate": len(multi_action_states) / max(total_states, 1),
        "r_mask_empty_states": sum(1 for s in state_records if s.r_mask_empty),
        "legal_r_count_mean": float(np.mean(legal_counts)) if legal_counts else 0.0,
        "legal_r_count_std": float(np.std(legal_counts)) if legal_counts else 0.0,
        "legal_r_count_p50": float(np.median(legal_counts)) if legal_counts else 0.0,
        "total_candidates": len(records),
        "success_rate": float(np.mean([r.current_success for r in records])) if records else 0.0,
    }

    # Correlations.
    summary["correlations_vs_G_v12"] = {
        "basic": _feature_correlation_table(records, BASIC_FEATURES, "G_v12", lambda r: r.G_v12),
        "bottleneck": _feature_correlation_table(
            records, BOTTLENECK_FEATURES, "G_v12", lambda r: r.G_v12,
            control_names=["path_km", "required_fs_norm"],
        ),
        "release": _feature_correlation_table(
            records, RELEASE_FEATURES, "G_v12", lambda r: r.G_v12,
            control_names=["path_km", "required_fs_norm"],
        ),
    }
    summary["correlations_vs_future_blocked"] = {
        "basic": _feature_correlation_table(records, BASIC_FEATURES, "future_blocked_count", lambda r: r.future_blocked_count),
        "bottleneck": _feature_correlation_table(records, BOTTLENECK_FEATURES, "future_blocked_count", lambda r: r.future_blocked_count),
        "release": _feature_correlation_table(records, RELEASE_FEATURES, "future_blocked_count", lambda r: r.future_blocked_count),
    }
    summary["correlations_vs_future_nsb"] = {
        "basic": _feature_correlation_table(records, BASIC_FEATURES, "future_nsb_count", lambda r: r.future_nsb_count),
        "bottleneck": _feature_correlation_table(records, BOTTLENECK_FEATURES, "future_nsb_count", lambda r: r.future_nsb_count),
        "release": _feature_correlation_table(records, RELEASE_FEATURES, "future_nsb_count", lambda r: r.future_nsb_count),
    }

    # Hard-pair analysis (only on successful candidates).
    pairs = _hard_pairs(records, return_gap_threshold=args.hard_pair_gap_threshold)
    total_pairs = 0
    for s in state_records:
        succ = [r for r in records if r.current_success and r.seed == s.seed and r.episode == s.episode and r.request == s.request]
        n = len(succ)
        total_pairs += n * (n - 1) // 2

    summary["hard_pairs"] = {
        "hard_pair_count": len(pairs),
        "total_successful_pairs": total_pairs,
        "hard_pair_rate": len(pairs) / max(total_pairs, 1),
        "mean_return_gap": float(np.mean([abs(a.G_v12 - b.G_v12) for a, b in pairs])) if pairs else 0.0,
    }

    # Pairwise prediction performance.
    pair_results = []
    all_features = BOTTLENECK_FEATURES + RELEASE_FEATURES + BASIC_FEATURES
    for name, direction in all_features:
        acc, auc = _pair_auc_and_accuracy(pairs, name, direction)
        pair_results.append({"feature": name, "direction": direction, "accuracy": acc, "auc": auc})
    summary["hard_pairs"]["pair_prediction"] = pair_results
    summary["hard_pairs"]["best_accuracy"] = max(
        (r["accuracy"] for r in pair_results if r["accuracy"] is not None),
        default=None,
    )
    summary["hard_pairs"]["best_accuracy_feature"] = next(
        (r["feature"] for r in pair_results if r["accuracy"] == summary["hard_pairs"]["best_accuracy"]),
        None,
    )

    # v1.2 vs DeepRMSA.
    summary["v12_vs_deep"] = _aggregate_v12_vs_deep(records)

    # Verdict.
    summary["verdict"] = _verdict(summary, args)
    return summary


def _verdict(summary: Dict[str, Any], args: argparse.Namespace) -> Dict[str, Any]:
    hp = summary["hard_pairs"]
    hard_rate = hp["hard_pair_rate"]
    best_acc = hp.get("best_accuracy")

    # Best absolute |Spearman| among bottleneck features vs G_v12.
    bottleneck_rows = summary["correlations_vs_G_v12"]["bottleneck"]["rows"]
    best_spearman = max(
        (abs(r["spearman"]) for r in bottleneck_rows if r["spearman"] is not None),
        default=0.0,
    )
    best_partial = max(
        (abs(r["partial_pearson_vs_pathkm_requiredfs"]) for r in bottleneck_rows
         if r["partial_pearson_vs_pathkm_requiredfs"] is not None),
        default=0.0,
    )

    gates = {
        "hard_pair_rate >= 10%": hard_rate >= 0.10,
        "bottleneck |Spearman| >= 0.20": best_spearman >= 0.20,
        "hard-pair accuracy >= 58%": best_acc is not None and best_acc >= 0.58,
        "partial corr >= 0.15 (not path_km/required_fs only)": best_partial >= 0.15,
    }

    # PASS requires all gates.
    if all(gates.values()):
        return {
            "label": "PASS_TO_V13_TRAINING",
            "reasons": [f"PASS: {k}" for k, v in gates.items() if v],
            "gates": {k: bool(v) for k, v in gates.items()},
        }

    # MARGINAL: hard_pair_rate >= 5% and at least one of spearman/accuracy in weak band.
    if hard_rate >= 0.05 and (0.10 <= best_spearman < 0.20 or (best_acc is not None and 0.54 <= best_acc < 0.58)):
        return {
            "label": "MARGINAL",
            "reasons": [
                f"hard_pair_rate={hard_rate:.2%} >= 5%",
                f"spearman={best_spearman:.3f}",
                f"accuracy={best_acc}",
            ],
            "gates": {k: bool(v) for k, v in gates.items()},
        }

    return {
        "label": "FAIL",
        "reasons": [f"{'PASS' if v else 'FAIL'}: {k}" for k, v in gates.items()],
        "gates": {k: bool(v) for k, v in gates.items()},
    }


# -----------------------------------------------------------------------------
# Config and markdown
# -----------------------------------------------------------------------------
def _config_dict(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "agent_c_checkpoint": args.agent_c_checkpoint,
        "agent_r_checkpoint": args.agent_r_checkpoint,
        "ranking_checkpoint": args.ranking_checkpoint,
        "deep_rmsa_checkpoint": args.deep_rmsa_checkpoint,
        "topology": args.topology,
        "num_slots": args.num_slots,
        "num_servers": args.num_servers,
        "k_paths": args.k_paths,
        "max_blocks": args.max_blocks,
        "block_sort_strategy": args.block_sort_strategy,
        "split_profile": args.split_profile,
        "num_splits": args.num_splits,
        "seeds": [int(s.strip()) for s in args.seeds.split(",") if s.strip()],
        "episodes": args.episodes,
        "requests_per_episode": args.requests_per_episode,
        "horizon": args.horizon,
        "arrival_interval": args.arrival_interval,
        "holding_min": args.holding_min,
        "holding_max": args.holding_max,
        "deadline_min": args.deadline_min,
        "deadline_max": args.deadline_max,
        "size_min_mb": args.size_min_mb,
        "size_max_mb": args.size_max_mb,
        "edge_cost_min": args.edge_cost_min,
        "edge_cost_max": args.edge_cost_max,
        "modulation_profile": args.modulation_profile,
        "util_threshold": args.util_threshold,
        "alpha": args.alpha,
    }


def _fmt(x: Optional[float]) -> str:
    if x is None:
        return "N/A"
    return f"{x:.4f}"


def _build_markdown(report: Dict[str, Any]) -> str:
    s = report["summary"]
    cfg = report["config"]
    lines = [
        "# R-Side Bottleneck / Release / Hard-Pair Signal Diagnostic",
        "",
        f"- Scenario: `{cfg['topology']}`, {cfg['num_slots']} slots, {cfg['num_servers']} servers, "
        f"k={cfg['k_paths']}, max_blocks={cfg['max_blocks']}",
        f"- Load: arrival_interval={cfg['arrival_interval']}, holding={cfg['holding_min']}-{cfg['holding_max']}, "
        f"size={cfg['size_min_mb']}-{cfg['size_max_mb']} MB",
        f"- Seeds: {cfg['seeds']}, episodes/seed={cfg['episodes']}, requests/episode={cfg['requests_per_episode']}",
        f"- H-step horizon H={cfg['horizon']}",
        f"- Checkpoints: C=`{cfg['agent_c_checkpoint']}`, ranker=`{cfg['ranking_checkpoint']}`",
        "",
        "## Aggregate counts",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Total states | {s['total_states']} |",
        f"| Multi-action (>=2 legal R) states | {s['multi_action_states']} ({s['multi_action_rate']:.2%}) |",
        f"| R-mask empty states | {s['r_mask_empty_states']} |",
        f"| Legal R count mean/std/median | {s['legal_r_count_mean']:.2f} / {s['legal_r_count_std']:.2f} / {s['legal_r_count_p50']:.0f} |",
        f"| Total candidates evaluated | {s['total_candidates']} |",
        f"| Current success rate | {s['success_rate']:.2%} |",
        "",
        "## Return formulas",
        "",
        "**G_v12** = -3·current_block -4·future_blocked -3·future_nsb -2·future_overload -0.03·future_delay_mean -0.05·future_avg_fs",
        "",
        "**G_hard** = -4·current_block -5·future_blocked -4·future_nsb -2·future_overload -0.03·future_delay_mean -0.05·future_avg_fs -0.0005·path_km -0.0208·required_fs",
        "",
        "## Correlations with G_v12",
        "",
    ]

    def _corr_table(rows: List[Dict[str, Any]]) -> List[str]:
        out = [
            "| Feature | Direction | Pearson | Spearman | Partial (vs path_km+required_fs) |",
            "|---|---|---:|---:|---:|",
        ]
        for r in rows:
            out.append(
                f"| {r['feature']} | {r['direction']} | {_fmt(r['pearson'])} | {_fmt(r['spearman'])} | {_fmt(r.get('partial_pearson_vs_pathkm_requiredfs'))} |"
            )
        return out

    lines.append("### Basic features")
    lines.extend(_corr_table(s["correlations_vs_G_v12"]["basic"]["rows"]))
    lines.append("")
    lines.append("### Bottleneck-edge features")
    lines.extend(_corr_table(s["correlations_vs_G_v12"]["bottleneck"]["rows"]))
    lines.append("")
    lines.append("### Release-aware features")
    lines.extend(_corr_table(s["correlations_vs_G_v12"]["release"]["rows"]))
    lines.append("")

    lines.append("## Correlations with future_blocked_count")
    lines.extend(_corr_table(s["correlations_vs_future_blocked"]["bottleneck"]["rows"]))
    lines.append("")
    lines.append("## Correlations with future_nsb_count")
    lines.extend(_corr_table(s["correlations_vs_future_nsb"]["bottleneck"]["rows"]))
    lines.append("")

    hp = s["hard_pairs"]
    lines.extend([
        "## Hard-pair analysis",
        "",
        "Definition: both actions succeed, current delay diff < 1 ms, block_waste diff < 1, "
        "and future_blocked / future_nsb differ or |G_v12| gap > 0.1.",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Hard-pair count | {hp['hard_pair_count']} |",
        f"| Total successful pairs | {hp['total_successful_pairs']} |",
        f"| Hard-pair rate | {hp['hard_pair_rate']:.2%} |",
        f"| Mean return gap | {hp['mean_return_gap']:.4f} |",
        f"| Best hard-pair accuracy | {hp['best_accuracy']} ({hp['best_accuracy_feature']}) |",
        "",
        "### Pairwise prediction accuracy",
        "",
        "| Feature | Direction | Accuracy | AUC |",
        "|---|---|---:|---:|",
    ])
    for r in hp["pair_prediction"]:
        auc = _fmt(r.get("auc"))
        lines.append(f"| {r['feature']} | {r['direction']} | {_fmt(r['accuracy'])} | {auc} |")
    lines.append("")

    vd = s["v12_vs_deep"]
    lines.extend([
        "## v1.2 ranker vs DeepRMSA selected actions",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Comparable states | {vd['comparable_states']} |",
        f"| v1.2 better | {vd['counts']['v12_better']} |",
        f"| DeepRMSA better | {vd['counts']['deep_better']} |",
        f"| Same return | {vd['counts']['same']} |",
        f"| Incomparable | {vd['counts']['incomparable']} |",
        "",
        "| Mean difference (v1.2 - DeepRMSA) | Value |",
        "|---|---:|",
    ])
    for k, v in vd["mean_diff"].items():
        lines.append(f"| {k} | {_fmt(v)} |")
    lines.append("")

    verdict = s["verdict"]
    lines.extend([
        "## Verdict",
        "",
        f"**{verdict['label']}**",
        "",
        "Reasons:",
    ])
    for reason in verdict["reasons"]:
        lines.append(f"- {reason}")
    lines.append("")
    lines.append(f"Elapsed: {report['elapsed_seconds']:.1f}s")
    return "\n".join(lines)


# -----------------------------------------------------------------------------
# CLI
# -----------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--deep_rmsa_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--arrival_interval", type=float, default=0.09)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=14.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--traffic_mode", type=str, default="iid")
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--hard_pair_gap_threshold", type=float, default=0.1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_json", type=str, default=None)
    parser.add_argument("--output_md", type=str, default=None)
    args = parser.parse_args()

    report = _run_diagnostic(args)

    out_json = Path(args.output_json or "sa_hmarl/experiments/r_bottleneck_hardpair_signal_diagnostic.json")
    out_md = Path(args.output_md or "sa_hmarl/experiments/r_bottleneck_hardpair_signal_diagnostic.md")
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    out_md.write_text(_build_markdown(report), encoding="utf-8")
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
    print(f"Verdict: {report['summary']['verdict']['label']}")


if __name__ == "__main__":
    main()
