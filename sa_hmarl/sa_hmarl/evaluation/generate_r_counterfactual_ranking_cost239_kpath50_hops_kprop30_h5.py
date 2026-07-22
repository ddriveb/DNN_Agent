"""SA-HMARL v1.3 counterfactual R-ranking dataset generator (COST239, K_path=50).

This script implements the strict v1.3 protocol:
  - topology = xlron_cost239_ptrnet_real
  - frozen PPO-C (K_C=5) selects split/server
  - frozen PPO-R proposes top-K_prop legal flat actions under K_path=50
  - each candidate is evaluated by rolling out H=5 future transitions with the
    frozen C+R pair
  - labels use only the v1.3 coefficients (current block, future block, future
    NSB, mean delay, mean FS)
  - features are the 25-d v1 pre-decision features

Output is a directory of per-shard .npz files plus a top-level metadata.json.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    build_poststate_v1_feature_batch,
)
from sa_hmarl.evaluation.v135_parallel_launcher import (
    ShardedParallelLauncher,
    _atomic_write_json,
    _config_hash,
)
from sa_hmarl.training.utils import generate_requests, make_env


# ---------------------------------------------------------------------------
# Protocol constants
# ---------------------------------------------------------------------------
PPO_R_TOPK = 1  # provenance bitmask bit

SUPPORTED_FEATURE_MODES = {
    "v1.3": ("v1.3 base features", FEATURE_NAMES, None),
    "v1.35_poststate_v1": (
        "v1.3 base + explicit optical afterstate (poststate_v1)",
        POSTSTATE_V1_FEATURE_NAMES,
        build_poststate_v1_feature_batch,
    ),
}

def _feature_mode_config(args: argparse.Namespace):
    """Return (description, feature_names, builder_fn) for the chosen feature mode."""
    return SUPPORTED_FEATURE_MODES[args.feature_mode]

DEFAULT_K_PATHS_C = 5
DEFAULT_K_PATHS_R = 50
DEFAULT_PATH_SORT_STRATEGY = "hops"
DEFAULT_BLOCK_SORT_STRATEGY = "start_asc"
DEFAULT_MAX_CANDIDATES = 30
DEFAULT_PPO_TOP_K = 30
DEFAULT_LABEL_HORIZON = 5
DEFAULT_GAMMA = 1.0

LABEL_COEFS = {
    "current_block": 3.0,
    "future_block": 4.0,
    "future_nsb": 3.0,
    "delay": 0.03,
    "fs": 0.05,
}

# Superset expected by the training report writer.
RETURN_COEFS = {
    **LABEL_COEFS,
    "future_server_overload": 0.0,
    "future_optical": 0.0,
    "future_overload": 0.0,
    "future_other": 0.0,
    "path_penalty": 0.0,
    "fs_penalty": 0.0,
    "return_mode": "legacy_total_return",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _group_id(split_name: str, seed: int, episode: int, request_index: int) -> int:
    split_code = {"train": 1, "val": 2, "test": 3}[split_name]
    return split_code * 10**15 + seed * 10**7 + episode * 10**6 + request_index


def _state_hash(env, req, split_id: int, server_id: int) -> str:
    h = hashlib.sha256()
    h.update(
        f"t={env.time:.9f}|req={req.req_id}|split={split_id}|server={server_id}".encode()
    )
    for key in sorted(env.net.link_states.keys()):
        h.update(str(key).encode())
        h.update(np.asarray(env.net.link_states[key], dtype=np.int64).tobytes())
    utils = np.asarray([s.utilization for s in env.mec.servers], dtype=np.float32)
    h.update(utils.tobytes())
    return h.hexdigest()[:16]


def _trace_hash(requests: List[Any], start_idx: int, horizon: int) -> str:
    h = hashlib.sha256()
    h.update(f"start={start_idx}|horizon={horizon}".encode())
    for offset in range(horizon):
        idx = start_idx + offset
        if idx >= len(requests):
            break
        req = requests[idx]
        h.update(
            f"req_id={req.req_id}|src={req.src_node}|arr={req.arrival_time:.9f}|"
            f"hold={req.holding_time:.6f}|deadline={req.deadline_ms:.6f}".encode()
        )
        for sp in req.splits:
            h.update(
                f"sp={sp.split_id}|size={sp.intermediate_size_mb:.6f}|"
                f"edge={sp.edge_compute_cost:.6f}|local={sp.local_compute_cost:.6f}".encode()
            )
    return h.hexdigest()[:16]


def _ppo_r_topk_actions_with_scores(
    agent_r, obs_r: Dict[str, Any], top_k: int
) -> Tuple[List[int], np.ndarray]:
    """Stable top-k legal flat actions ranked by PPO-R logits plus raw logits."""
    features, mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(mask, dtype=bool))
    if legal.size == 0:
        return [], np.array([], dtype=np.float32)
    with torch.no_grad():
        x = torch.as_tensor(
            features, dtype=torch.float32, device=agent_r.device
        ).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    top_k = min(top_k, legal.size)
    top_local = np.argsort(-logits[legal], kind="stable")[:top_k]
    actions = [int(legal[i]) for i in top_local]
    scores = logits[actions].astype(np.float32)
    return actions, scores


def _ppo_r_topk_actions(
    agent_r, obs_r: Dict[str, Any], top_k: int
) -> List[int]:
    """Stable top-k legal flat actions ranked by PPO-R logits."""
    actions, _ = _ppo_r_topk_actions_with_scores(agent_r, obs_r, top_k)
    return actions


def _action_to_path_idx(action_id: int, num_mods: int, max_blocks: int) -> int:
    return action_id // (num_mods * max_blocks)


def _depth_stratum(max_path_idx: int) -> int:
    """Stratify groups by the deepest path index among PPO-R candidates.

    0: max_path_idx < 5
    1: 5 <= max_path_idx < 10
    2: 10 <= max_path_idx < 20
    3: max_path_idx >= 20
    """
    if max_path_idx < 5:
        return 0
    if max_path_idx < 10:
        return 1
    if max_path_idx < 20:
        return 2
    return 3


def _keep_group(max_path_idx: int, group_filter: str) -> bool:
    """Return True iff a group with this max_path_idx should be saved.

    'all' keeps every non-warmup group with at least one legal candidate.
    'deep_path_only' reproduces the legacy E=1-only filter.
    """
    if group_filter == "deep_path_only":
        return max_path_idx >= 5
    if group_filter == "all":
        return True
    raise ValueError(f"Unknown group_filter: {group_filter}")


def _execute_fixed_r(
    env, req, split_id: int, server_id: int, r_action_idx: int, obs_r: Dict[str, Any]
):
    r_action = decode_agent_r_action(
        int(r_action_idx), len(obs_r["mod_names"]), env.max_blocks
    )
    _, _, _, info = env.step((split_id, server_id), r_action)
    return info


def _rollout_future(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c,
    agent_r,
    num_servers: int,
    k_c: int,
    k_r: int,
    path_sort_c: str,
    block_sort_c: str,
    path_sort_r: str,
    block_sort_r: str,
    gamma: float,
) -> Dict[str, Any]:
    """Roll out ``horizon`` future transitions using frozen PPO-C + PPO-R.

    The environment's ``k`` and sort strategies are toggled before every C/R
    observation as required by the v1.3 protocol.

    Returns both aggregated metrics and per-step component arrays so that the
    v1.3 total-return label can be reconstructed exactly from saved components.
    """
    blocked_per_step = np.full(horizon, np.nan, dtype=np.float32)
    nsb_per_step = np.full(horizon, np.nan, dtype=np.float32)
    overload_per_step = np.full(horizon, np.nan, dtype=np.float32)
    delay_per_step = np.full(horizon, np.nan, dtype=np.float32)
    fs_per_step = np.full(horizon, np.nan, dtype=np.float32)

    actual_horizon = 0
    for offset in range(horizon):
        t = start_idx + offset
        if t >= len(requests):
            break
        actual_horizon += 1
        req = requests[t]
        env.advance_time(req.arrival_time)

        # Future C observation under K_C.
        env.k = k_c
        env.path_sort_strategy = path_sort_c
        env.block_sort_strategy = block_sort_c
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if int(raw_c_mask.sum()) == 0:
            blocked_per_step[offset] = 1.0
            nsb_per_step[offset] = 0.0
            overload_per_step[offset] = 0.0
            delay_per_step[offset] = 0.0
            fs_per_step[offset] = 0.0
            env.k = k_r
            env.path_sort_strategy = path_sort_r
            env.block_sort_strategy = block_sort_r
            env.step((0, 0), (0, 0, 0))
            continue

        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split, server = decode_agent_c_action(c_idx, num_servers)

        # Future R observation and step under K_path.
        env.k = k_r
        env.path_sort_strategy = path_sort_r
        env.block_sort_strategy = block_sort_r
        obs_r = build_agent_r_observation(env, req, split, server)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if int(raw_r_mask.sum()) == 0:
            blocked_per_step[offset] = 1.0
            nsb_per_step[offset] = 0.0
            overload_per_step[offset] = 0.0
            delay_per_step[offset] = 0.0
            fs_per_step[offset] = 0.0
            env.step((split, server), (0, 0, 0))
            continue

        r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split, server), r_action)

        if not info.get("success", False):
            blocked_per_step[offset] = 1.0
            delay_per_step[offset] = 0.0
            fs_per_step[offset] = 0.0
            reason = info.get("reason", "")
            if reason == "no_suitable_block":
                nsb_per_step[offset] = 1.0
                overload_per_step[offset] = 0.0
            elif reason == "server_overload":
                nsb_per_step[offset] = 0.0
                overload_per_step[offset] = 1.0
            else:
                nsb_per_step[offset] = 0.0
                overload_per_step[offset] = 0.0
        else:
            blocked_per_step[offset] = 0.0
            nsb_per_step[offset] = 0.0
            overload_per_step[offset] = 0.0
            delay_per_step[offset] = float(info.get("delay_ms", 0.0))
            fs_per_step[offset] = float(info.get("num_slots", 0))

    # Discounted sums (gamma=1.0 collapses to raw counts).
    weights = np.array([gamma**h for h in range(1, horizon + 1)], dtype=np.float32)
    active = ~np.isnan(blocked_per_step)
    discounted_blocked = float(np.nansum(blocked_per_step[:horizon] * weights[:horizon]))
    discounted_nsb = float(np.nansum(nsb_per_step[:horizon] * weights[:horizon]))

    success_count = int(np.nansum((blocked_per_step == 0.0) & active))
    delay_sum = float(np.nansum(delay_per_step))
    fs_sum = float(np.nansum(fs_per_step))
    mean_delay = delay_sum / max(success_count, 1)
    mean_fs = fs_sum / max(success_count, 1)

    blocked = int(np.nansum(blocked_per_step))
    no_suitable_block = int(np.nansum(nsb_per_step))
    server_overload = int(np.nansum(overload_per_step))
    other = blocked - no_suitable_block - server_overload

    return {
        "blocked": blocked,
        "blocked_discounted": discounted_blocked,
        "no_suitable_block": no_suitable_block,
        "no_suitable_block_discounted": discounted_nsb,
        "server_overload": server_overload,
        "other": other,
        "mean_delay_ms": float(mean_delay),
        "mean_fs": float(mean_fs),
        "success_count": success_count,
        "blocked_per_step": blocked_per_step,
        "nsb_per_step": nsb_per_step,
        "overload_per_step": overload_per_step,
        "delay_per_step": delay_per_step,
        "fs_per_step": fs_per_step,
        "actual_horizon": actual_horizon,
    }


def _compute_label(
    info: Dict[str, Any],
    future: Dict[str, Any],
    coefs: Dict[str, float],
    gamma: float,
) -> float:
    """v1.3 label: no path penalty, no FS penalty, no fragmentation proxy."""
    b_t = 0.0 if info.get("success", False) else 1.0
    # With gamma=1.0 the discounted sums equal raw counts; with gamma != 1.0 the
    # protocol still expects discounted future sums.
    if gamma == 1.0:
        fut_block = float(future.get("blocked", 0))
        fut_nsb = float(future.get("no_suitable_block", 0))
    else:
        fut_block = float(future.get("blocked_discounted", future.get("blocked", 0)))
        fut_nsb = float(
            future.get("no_suitable_block_discounted", future.get("no_suitable_block", 0))
        )
    delay = float(future.get("mean_delay_ms", 0.0))
    fs = float(future.get("mean_fs", 0.0))
    return float(
        -coefs["current_block"] * b_t
        - coefs["future_block"] * fut_block
        - coefs["future_nsb"] * fut_nsb
        - coefs["delay"] * delay
        - coefs["fs"] * fs
    )


def _reconstruct_label_from_components(
    current_block: float,
    future: Dict[str, Any],
    coefs: Dict[str, float],
    gamma: float,
) -> float:
    """Reconstruct the v1.3 total return from stored per-step components.

    This must be bit-identical to ``_compute_label`` for gamma=1.0.
    """
    horizon = int(future.get("actual_horizon", 0))
    if horizon == 0:
        fut_block = 0.0
        fut_nsb = 0.0
    else:
        weights = np.array([gamma**h for h in range(1, horizon + 1)], dtype=np.float32)
        bp = future["blocked_per_step"][:horizon]
        npb = future["nsb_per_step"][:horizon]
        if gamma == 1.0:
            fut_block = float(np.nansum(bp))
            fut_nsb = float(np.nansum(npb))
        else:
            fut_block = float(np.nansum(bp * weights))
            fut_nsb = float(np.nansum(npb * weights))
    delay = float(future.get("mean_delay_ms", 0.0))
    fs = float(future.get("mean_fs", 0.0))
    return float(
        -coefs["current_block"] * current_block
        - coefs["future_block"] * fut_block
        - coefs["future_nsb"] * fut_nsb
        - coefs["delay"] * delay
        - coefs["fs"] * fs
    )


def _pad_groups(
    groups: List[Dict[str, Any]], k: int, h: int, feature_dim: int
) -> Dict[str, np.ndarray]:
    """Stack variable-length candidate groups into padded arrays.

    Adds per-candidate label components and PPO metadata while preserving full
    backward compatibility with the original ``features``/``returns``/``mask``
    fields used by the training script.
    """
    empty = {
        "features": np.empty((0, k, feature_dim), dtype=np.float32),
        "returns": np.empty((0, k), dtype=np.float32),
        "action_ids": np.empty((0, k), dtype=np.int32),
        "path_indices": np.empty((0, k), dtype=np.int32),
        "mod_indices": np.empty((0, k), dtype=np.int32),
        "block_indices": np.empty((0, k), dtype=np.int32),
        "mask": np.empty((0, k), dtype=bool),
        "provenance": np.empty((0, k), dtype=np.uint8),
        "group_id": np.empty((0,), dtype=np.int64),
        "req_id": np.empty((0,), dtype=np.int64),
        "state_hash": np.empty((0,), dtype=object),
        "trace_hash": np.empty((0,), dtype=object),
        "c_action": np.empty((0,), dtype=np.int32),
        "max_path_idx": np.empty((0,), dtype=np.int32),
        "max_action_id": np.empty((0,), dtype=np.int32),
        "deep_path_gate": np.empty((0,), dtype=np.int32),
        "depth_stratum": np.empty((0,), dtype=np.int32),
        "ppo_action_index": np.empty((0,), dtype=np.int64),
        "current_block": np.empty((0, k), dtype=np.float32),
        "current_nsb": np.empty((0, k), dtype=np.float32),
        "current_overload": np.empty((0, k), dtype=np.float32),
        "immediate_delay": np.empty((0, k), dtype=np.float32),
        "immediate_fs": np.empty((0, k), dtype=np.float32),
        "future_blocked_per_step": np.empty((0, k, h), dtype=np.float32),
        "future_nsb_per_step": np.empty((0, k, h), dtype=np.float32),
        "future_overload_per_step": np.empty((0, k, h), dtype=np.float32),
        "future_delay_per_step": np.empty((0, k, h), dtype=np.float32),
        "future_fs_per_step": np.empty((0, k, h), dtype=np.float32),
        "actual_horizon": np.empty((0, k), dtype=np.int32),
        "ppo_logits": np.empty((0, k), dtype=np.float32),
        "ppo_rank": np.empty((0, k), dtype=np.int32),
    }
    n = len(groups)
    if n == 0:
        return empty

    features = np.zeros((n, k, feature_dim), dtype=np.float32)
    returns = np.zeros((n, k), dtype=np.float32)
    action_ids = np.zeros((n, k), dtype=np.int32)
    path_indices = np.zeros((n, k), dtype=np.int32)
    mod_indices = np.full((n, k), -1, dtype=np.int32)
    block_indices = np.full((n, k), -1, dtype=np.int32)
    mask = np.zeros((n, k), dtype=bool)
    provenance = np.zeros((n, k), dtype=np.uint8)

    current_block = np.zeros((n, k), dtype=np.float32)
    current_nsb = np.zeros((n, k), dtype=np.float32)
    current_overload = np.zeros((n, k), dtype=np.float32)
    immediate_delay = np.zeros((n, k), dtype=np.float32)
    immediate_fs = np.zeros((n, k), dtype=np.float32)

    future_blocked_per_step = np.full((n, k, h), np.nan, dtype=np.float32)
    future_nsb_per_step = np.full((n, k, h), np.nan, dtype=np.float32)
    future_overload_per_step = np.full((n, k, h), np.nan, dtype=np.float32)
    future_delay_per_step = np.full((n, k, h), np.nan, dtype=np.float32)
    future_fs_per_step = np.full((n, k, h), np.nan, dtype=np.float32)
    actual_horizon = np.zeros((n, k), dtype=np.int32)
    ppo_logits = np.full((n, k), np.nan, dtype=np.float32)
    ppo_rank = np.full((n, k), -1, dtype=np.int32)

    for i, g in enumerate(groups):
        c = len(g["action_ids"])
        features[i, :c] = g["features"]
        returns[i, :c] = g["returns"]
        action_ids[i, :c] = g["action_ids"]
        path_indices[i, :c] = g["path_indices"]
        mod_indices[i, :c] = g["mod_indices"]
        block_indices[i, :c] = g["block_indices"]
        mask[i, :c] = True
        provenance[i, :c] = PPO_R_TOPK

        current_block[i, :c] = g["current_block"]
        current_nsb[i, :c] = g["current_nsb"]
        current_overload[i, :c] = g["current_overload"]
        immediate_delay[i, :c] = g["immediate_delay"]
        immediate_fs[i, :c] = g["immediate_fs"]

        for j in range(c):
            future = g["future_components"][j]
            ah = int(future["actual_horizon"])
            actual_horizon[i, j] = ah
            if ah > 0:
                future_blocked_per_step[i, j, :ah] = future["blocked_per_step"][:ah]
                future_nsb_per_step[i, j, :ah] = future["nsb_per_step"][:ah]
                future_overload_per_step[i, j, :ah] = future["overload_per_step"][:ah]
                future_delay_per_step[i, j, :ah] = future["delay_per_step"][:ah]
                future_fs_per_step[i, j, :ah] = future["fs_per_step"][:ah]
        ppo_logits[i, :c] = g["ppo_logits"]
        ppo_rank[i, :c] = g["ppo_rank"]

    return {
        "features": features,
        "returns": returns,
        "action_ids": action_ids,
        "path_indices": path_indices,
        "mod_indices": mod_indices,
        "block_indices": block_indices,
        "mask": mask,
        "provenance": provenance,
        "group_id": np.asarray([g["group_id"] for g in groups], dtype=np.int64),
        "req_id": np.asarray([g["req_id"] for g in groups], dtype=np.int64),
        "state_hash": np.asarray([g["state_hash"] for g in groups], dtype=object),
        "trace_hash": np.asarray([g["trace_hash"] for g in groups], dtype=object),
        "c_action": np.asarray([g["c_action"] for g in groups], dtype=np.int32),
        "max_path_idx": np.asarray([g["max_path_idx"] for g in groups], dtype=np.int32),
        "max_action_id": np.asarray([g["max_action_id"] for g in groups], dtype=np.int32),
        "deep_path_gate": np.asarray([g["deep_path_gate"] for g in groups], dtype=np.int32),
        "depth_stratum": np.asarray([g["depth_stratum"] for g in groups], dtype=np.int32),
        "ppo_action_index": np.asarray([g["ppo_action_index"] for g in groups], dtype=np.int64),
        "current_block": current_block,
        "current_nsb": current_nsb,
        "current_overload": current_overload,
        "immediate_delay": immediate_delay,
        "immediate_fs": immediate_fs,
        "future_blocked_per_step": future_blocked_per_step,
        "future_nsb_per_step": future_nsb_per_step,
        "future_overload_per_step": future_overload_per_step,
        "future_delay_per_step": future_delay_per_step,
        "future_fs_per_step": future_fs_per_step,
        "actual_horizon": actual_horizon,
        "ppo_logits": ppo_logits,
        "ppo_rank": ppo_rank,
    }


# ---------------------------------------------------------------------------
# Per-shard generation
# ---------------------------------------------------------------------------
def _generate_shard(
    env,
    requests: List[Any],
    agent_c,
    agent_r,
    split_name: str,
    seed: int,
    episode: int,
    args: argparse.Namespace,
) -> Dict[str, np.ndarray]:
    groups: List[Dict[str, Any]] = []
    env.reset(requests)
    num_servers = len(env.mec.servers)
    _, feature_names, feature_builder = _feature_mode_config(args)

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)

        # 1. C-side observation under K_C.
        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        assert env.k == args.k_paths_c, "C observation was not built with K_C"

        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), num_servers)

        # 3. R-side observation under K_path.
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        assert env.k == args.k_paths_r, "R observation was not built with K_path"
        num_paths_r = len(obs_r["candidate_paths"])
        actual_mask_len = len(obs_r["agent_r_mask"])
        expected_mask_len = num_paths_r * len(obs_r["mod_names"]) * env.max_blocks
        if actual_mask_len != expected_mask_len or num_paths_r > args.k_paths_r:
            raise AssertionError(
                f"R mask length mismatch at req {request_index}: "
                f"actual={actual_mask_len}, expected={expected_mask_len}, "
                f"num_paths={num_paths_r}, env.k={env.k}, "
                f"args.k_paths_r={args.k_paths_r}, mod_names={len(obs_r['mod_names'])}, "
                f"max_blocks={env.max_blocks}"
            )

        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()

        # Baseline PPO-R action for advancing the real trajectory.
        ppo_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        is_warmup = request_index < args.warmup
        if not is_warmup and len(legal) > 0:
            # 5. Candidate selection: top-K_prop PPO-R legal flat actions only.
            top_k = min(args.ppo_top_k, args.max_candidates)
            candidate_actions, candidate_logits = _ppo_r_topk_actions_with_scores(
                agent_r, obs_r, top_k
            )
            if len(candidate_actions) > args.max_candidates:
                candidate_actions = candidate_actions[: args.max_candidates]
                candidate_logits = candidate_logits[: args.max_candidates]

            if candidate_actions:
                num_mods = len(obs_r["mod_names"])
                max_blocks = int(env.max_blocks)

                # Gate tracking / optional deep-path-only filter.
                path_indices_sel = [
                    _action_to_path_idx(a, num_mods, max_blocks) for a in candidate_actions
                ]
                max_path_idx = int(max(path_indices_sel))
                deep_path_gate = int(max_path_idx >= 5)
                depth_stratum = _depth_stratum(max_path_idx)

                # Old v1.3 hard filter (path_idx >= 5) is now explicit and off by default.
                if not _keep_group(max_path_idx, args.group_filter):
                    continue

                # K_t = min(K_prop, |A_legal|) is enforced by _ppo_r_topk_actions_with_scores
                # and the clipping above; assert it here as a dataset-level invariant.
                assert len(candidate_actions) <= args.max_candidates, (
                    f"candidate count {len(candidate_actions)} exceeds max_candidates "
                    f"{args.max_candidates}"
                )

                # 6. Snapshot before the R decision.
                snapshot = _snapshot_before_r_decision(env, req.req_id)
                trace_hash = _trace_hash(
                    requests, request_index + 1, args.label_horizon
                )

                # Shared continuation RNG state (cloned per candidate).
                base_rng = np.random.RandomState(
                    seed + episode * 100000 + request_index + 12345
                )
                base_rng_state = base_rng.get_state()

                if feature_builder is not None:
                    cand_features_matrix = feature_builder(
                        env,
                        req,
                        obs_c,
                        obs_r,
                        r_features,
                        candidate_actions,
                        split_id,
                        server_id,
                        feature_names=feature_names,
                    )

                cand_features = []
                cand_returns = []
                cand_action_ids = []
                cand_path_indices = []
                cand_mod_indices = []
                cand_block_indices = []
                cand_current_block = []
                cand_current_nsb = []
                cand_current_overload = []
                cand_immediate_delay = []
                cand_immediate_fs = []
                cand_future_components = []
                cand_ppo_logits = []
                cand_ppo_rank = []

                ppo_action_index = (
                    candidate_actions.index(ppo_r_idx)
                    if int(ppo_r_idx) in candidate_actions
                    else -1
                )

                for rank, r_action_idx in enumerate(candidate_actions):
                    # 7. Branch from the pre-R-decision snapshot.
                    branch = copy.deepcopy(snapshot)
                    # Clone the shared continuation RNG state for this
                    # candidate. The continuation policies are deterministic,
                    # so the RNG is not consumed further, but it is kept
                    # available on the branch for bit-exact reproducibility.
                    branch_rng = np.random.RandomState(0)
                    branch_rng.set_state(base_rng_state)
                    branch._cont_rng_state = branch_rng.get_state()

                    info = _execute_fixed_r(
                        branch, req, split_id, server_id, int(r_action_idx), obs_r
                    )

                    actual_horizon = max(
                        min(args.label_horizon, len(requests) - request_index - 1), 0
                    )
                    future = _rollout_future(
                        branch,
                        requests,
                        request_index + 1,
                        actual_horizon,
                        agent_c,
                        agent_r,
                        num_servers,
                        args.k_paths_c,
                        args.k_paths_r,
                        args.path_sort_strategy_c,
                        args.block_sort_strategy_c,
                        args.path_sort_strategy_r,
                        args.block_sort_strategy_r,
                        args.gamma,
                    )

                    ret = _compute_label(info, future, LABEL_COEFS, args.gamma)
                    ret2 = _reconstruct_label_from_components(
                        0.0 if info.get("success", False) else 1.0,
                        future,
                        LABEL_COEFS,
                        args.gamma,
                    )
                    if abs(ret - ret2) > 1e-6:
                        raise AssertionError(
                            f"Label reconstruction mismatch at req {request_index}: "
                            f"{ret} vs {ret2}"
                        )

                    if feature_builder is None:
                        feat = _r_feature_vector(
                            env,
                            req,
                            obs_c,
                            obs_r,
                            r_features,
                            int(r_action_idx),
                            split_id,
                            server_id,
                            feature_names=feature_names,
                        )
                    else:
                        feat = cand_features_matrix[rank].astype(np.float32)

                    path_idx, mod_idx, block_idx = decode_agent_r_action(
                        int(r_action_idx), num_mods, max_blocks
                    )

                    cand_features.append(feat)
                    cand_returns.append(ret)
                    cand_action_ids.append(int(r_action_idx))
                    cand_path_indices.append(path_idx)
                    cand_mod_indices.append(mod_idx)
                    cand_block_indices.append(block_idx)

                    success = info.get("success", False)
                    reason = info.get("reason", "")
                    cand_current_block.append(0.0 if success else 1.0)
                    cand_current_nsb.append(
                        1.0 if (not success and reason == "no_suitable_block") else 0.0
                    )
                    cand_current_overload.append(
                        1.0 if (not success and reason == "server_overload") else 0.0
                    )
                    cand_immediate_delay.append(
                        float(info.get("delay_ms", 0.0)) if success else 0.0
                    )
                    cand_immediate_fs.append(
                        float(info.get("num_slots", 0)) if success else 0.0
                    )
                    cand_future_components.append(future)
                    cand_ppo_logits.append(float(candidate_logits[rank]))
                    cand_ppo_rank.append(rank)

                # All candidates share the same future trace.
                assert all(
                    trace_hash == _trace_hash(
                        requests, request_index + 1, args.label_horizon
                    )
                    for _ in candidate_actions
                ), "trace hash mismatch within group"

                groups.append(
                    {
                        "group_id": _group_id(
                            split_name, seed, episode, request_index
                        ),
                        "req_id": int(req.req_id),
                        "state_hash": _state_hash(env, req, split_id, server_id),
                        "trace_hash": trace_hash,
                        "c_action": int(c_idx),
                        "features": np.stack(cand_features).astype(np.float32),
                        "returns": np.asarray(cand_returns, dtype=np.float32),
                        "action_ids": np.asarray(cand_action_ids, dtype=np.int32),
                        "path_indices": np.asarray(
                            cand_path_indices, dtype=np.int32
                        ),
                        "mod_indices": np.asarray(
                            cand_mod_indices, dtype=np.int32
                        ),
                        "block_indices": np.asarray(
                            cand_block_indices, dtype=np.int32
                        ),
                        "max_path_idx": int(max(cand_path_indices)),
                        "max_action_id": int(max(cand_action_ids)),
                        "deep_path_gate": int(deep_path_gate),
                        "depth_stratum": int(depth_stratum),
                        "ppo_action_index": int(ppo_action_index),
                        "current_block": np.asarray(
                            cand_current_block, dtype=np.float32
                        ),
                        "current_nsb": np.asarray(
                            cand_current_nsb, dtype=np.float32
                        ),
                        "current_overload": np.asarray(
                            cand_current_overload, dtype=np.float32
                        ),
                        "immediate_delay": np.asarray(
                            cand_immediate_delay, dtype=np.float32
                        ),
                        "immediate_fs": np.asarray(
                            cand_immediate_fs, dtype=np.float32
                        ),
                        "future_components": cand_future_components,
                        "ppo_logits": np.asarray(
                            cand_ppo_logits, dtype=np.float32
                        ),
                        "ppo_rank": np.asarray(cand_ppo_rank, dtype=np.int32),
                    }
                    )

        # Advance real trajectory with frozen PPO-R.
        deployed_action = decode_agent_r_action(
            int(ppo_r_idx), len(obs_r["mod_names"]), env.max_blocks
        )
        env.step((split_id, server_id), deployed_action)

    return _pad_groups(
        groups, args.max_candidates, args.label_horizon, len(feature_names)
    )


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------
def _shard_worker(shard: Dict[str, Any], output_path: Path) -> Dict[str, Any]:
    import resource

    cfg = shard["config"]
    env = make_env(
        cfg["topology"],
        cfg["num_slots"],
        cfg["num_servers"],
        shard["seed"],
        modulation_profile=cfg["modulation_profile"],
        max_blocks=cfg["max_blocks"],
        block_sort_strategy=cfg["block_sort_strategy_r"],
        path_sort_strategy=cfg["path_sort_strategy_r"],
        k=cfg["k_paths_r"],
    )

    agent_c = _load_ppo_c(cfg["agent_c_checkpoint"], cfg["device"])
    agent_r = _load_ppo_r(
        cfg["agent_r_checkpoint"], env.mod_reg, cfg["device"]
    )

    rng = np.random.RandomState(shard["seed"])
    src = int(rng.randint(0, env.net.NUM_NODES))
    requests = generate_requests(
        env,
        rng,
        src,
        cfg["requests_per_episode"],
        cfg["arrival_interval"],
        cfg["holding_min"],
        cfg["holding_max"],
        cfg["deadline_min"],
        cfg["deadline_max"],
        cfg["size_min_mb"],
        cfg["size_max_mb"],
        cfg["edge_cost_min"],
        cfg["edge_cost_max"],
        cfg["num_splits"],
        cfg["split_profile"],
    )

    args = shard["args_namespace"]
    arrays = _generate_shard(
        env,
        requests,
        agent_c,
        agent_r,
        shard["split"],
        shard["seed"],
        shard["episode"],
        args,
    )

    peak_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0
    # Valid groups are rows with at least one valid candidate.
    valid_groups = int(np.asarray(arrays["mask"], dtype=bool).any(axis=1).sum())
    return {
        "arrays": arrays,
        "groups": valid_groups,
        "peak_rss_mb": peak_rss,
    }


# ---------------------------------------------------------------------------
# Config / shards
# ---------------------------------------------------------------------------
def _make_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "agent_c_checkpoint": args.agent_c_checkpoint,
        "agent_r_checkpoint": args.agent_r_checkpoint,
        "topology": args.topology,
        "num_slots": args.num_slots,
        "num_servers": args.num_servers,
        "modulation_profile": args.modulation_profile,
        "max_blocks": args.max_blocks,
        "num_splits": args.num_splits,
        "split_profile": args.split_profile,
        "arrival_interval": args.arrival_interval,
        "holding_min": args.holding_min,
        "holding_max": args.holding_max,
        "deadline_min": args.deadline_min,
        "deadline_max": args.deadline_max,
        "size_min_mb": args.size_min_mb,
        "size_max_mb": args.size_max_mb,
        "edge_cost_min": args.edge_cost_min,
        "edge_cost_max": args.edge_cost_max,
        "k_paths_c": args.k_paths_c,
        "k_paths_r": args.k_paths_r,
        "path_sort_strategy_c": args.path_sort_strategy_c,
        "path_sort_strategy_r": args.path_sort_strategy_r,
        "block_sort_strategy_c": args.block_sort_strategy_c,
        "block_sort_strategy_r": args.block_sort_strategy_r,
        "max_candidates": args.max_candidates,
        "ppo_top_k": args.ppo_top_k,
        "candidate_mode": args.candidate_mode,
        "label_horizon": args.label_horizon,
        "gamma": args.gamma,
        "group_filter": args.group_filter,
        "label_coefs": LABEL_COEFS,
        "return_coefs": RETURN_COEFS,
        "device": args.device,
        "requests_per_episode": args.requests_per_episode,
        "warmup_requests": args.warmup,
        "schema": "r_counterfactual_ranking_v1.3",
        "schema_version": "1.3.1",
        "version": "1.3",
        "feature_mode": args.feature_mode,
        "component_arrays": [
            "current_block",
            "current_nsb",
            "current_overload",
            "immediate_delay",
            "immediate_fs",
            "future_blocked_per_step",
            "future_nsb_per_step",
            "future_overload_per_step",
            "future_delay_per_step",
            "future_fs_per_step",
            "actual_horizon",
            "ppo_logits",
            "ppo_rank",
        ],
    }


def _make_shards(args: argparse.Namespace) -> List[Dict[str, Any]]:
    seeds = args.seeds
    if len(seeds) != 3:
        raise ValueError(
            f"--seeds must contain exactly 3 integers (train,val,test); got {seeds}"
        )
    cfg = _make_config(args)
    shards = []
    split_map = {
        "train": args.n_train_shards,
        "val": args.n_val_shards,
        "test": args.n_test_shards,
    }
    base_seeds = {
        "train": seeds[0],
        "val": seeds[1],
        "test": seeds[2],
    }
    for split_name, n_shards in split_map.items():
        for episode in range(n_shards):
            seed = base_seeds[split_name] + episode
            shards.append(
                {
                    "shard_id": f"{split_name}_s{seed}_e{episode}",
                    "split": split_name,
                    "seed": seed,
                    "episode": episode,
                    "config": cfg,
                    "args_namespace": args,
                }
            )
    return shards


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output_dir",
        default="sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5",
    )
    parser.add_argument(
        "--agent_c_checkpoint",
        default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt",
    )
    parser.add_argument(
        "--agent_r_checkpoint",
        default="sa_hmarl/checkpoints/agent_r_mixed.pt",
    )
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument(
        "--seeds",
        type=lambda s: [int(x.strip()) for x in s.split(",")],
        default="1001,2001,3001",
        help="Comma-separated train/val/test base seeds.",
    )
    parser.add_argument("--requests_per_episode", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=1500)
    parser.add_argument("--n_train_shards", type=int, default=3)
    parser.add_argument("--n_val_shards", type=int, default=1)
    parser.add_argument("--n_test_shards", type=int, default=1)
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument(
        "--log_interval",
        type=float,
        default=300.0,
        help="Seconds between launcher progress logs.",
    )
    parser.add_argument("--k_paths_c", type=int, default=DEFAULT_K_PATHS_C)
    parser.add_argument("--k_paths_r", type=int, default=DEFAULT_K_PATHS_R)
    parser.add_argument(
        "--path_sort_strategy_c", default=DEFAULT_PATH_SORT_STRATEGY
    )
    parser.add_argument(
        "--path_sort_strategy_r", default=DEFAULT_PATH_SORT_STRATEGY
    )
    parser.add_argument(
        "--block_sort_strategy_c", default=DEFAULT_BLOCK_SORT_STRATEGY
    )
    parser.add_argument(
        "--block_sort_strategy_r", default=DEFAULT_BLOCK_SORT_STRATEGY
    )
    parser.add_argument("--max_candidates", type=int, default=DEFAULT_MAX_CANDIDATES)
    parser.add_argument("--ppo_top_k", type=int, default=DEFAULT_PPO_TOP_K)
    parser.add_argument("--label_horizon", type=int, default=DEFAULT_LABEL_HORIZON)
    parser.add_argument("--gamma", type=float, default=DEFAULT_GAMMA)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--candidate_mode",
        default="ppo_r_topk_only",
        help="Candidate selection mode (must be ppo_r_topk_only for this dataset).",
    )
    parser.add_argument(
        "--group_filter",
        choices=["all", "deep_path_only"],
        default="all",
        help=(
            "Group retention policy. 'all' keeps every non-warmup state with at least "
            "one legal PPO-R candidate (the v1.3 fix). 'deep_path_only' reproduces the "
            "old E=1-only dataset for backward compatibility."
        ),
    )
    parser.add_argument(
        "--feature_mode",
        choices=list(SUPPORTED_FEATURE_MODES.keys()),
        default="v1.3",
        help=(
            "Feature schema for the ranker. 'v1.3' keeps the original 25-d pre-decision "
            "features. 'v1.35_poststate_v1' concatenates the 25-d base features with an "
            "explicit non-mutating optical afterstate vector."
        ),
    )
    return parser


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.warmup >= args.requests_per_episode:
        raise ValueError(
            f"--warmup ({args.warmup}) must be < --requests_per_episode "
            f"({args.requests_per_episode})"
        )
    if args.candidate_mode != "ppo_r_topk_only":
        raise ValueError(
            f"This generator requires candidate_mode='ppo_r_topk_only', "
            f"got {args.candidate_mode}"
        )
    if args.max_candidates > 30:
        raise ValueError(f"--max_candidates must be <= 30, got {args.max_candidates}")

    # Validate group_filter early.
    if args.group_filter not in ("all", "deep_path_only"):
        raise ValueError(f"--group_filter must be 'all' or 'deep_path_only', got {args.group_filter}")
    if args.feature_mode not in SUPPORTED_FEATURE_MODES:
        raise ValueError(f"--feature_mode must be one of {list(SUPPORTED_FEATURE_MODES.keys())}, got {args.feature_mode}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    config = _make_config(args)
    shards = _make_shards(args)
    _, feature_names, _ = _feature_mode_config(args)

    checkpoint_sha256 = {
        "agent_c": _sha256_file(args.agent_c_checkpoint),
        "agent_r": _sha256_file(args.agent_r_checkpoint),
    }

    launcher = ShardedParallelLauncher(
        output_dir=output_dir,
        config=config,
        worker_fn=_shard_worker,
        max_workers=args.max_workers,
        shard_subdir="shards",
        log_interval_sec=args.log_interval,
    )
    results = launcher.run(shards)

    # Aggregate per-shard info for top-level metadata.
    start_time = time.perf_counter()

    shard_meta_list = []
    global_max_action_id = 0
    global_max_path_idx = 0
    global_groups_e0 = 0
    global_groups_e1 = 0
    global_stratum_counts = {0: 0, 1: 0, 2: 0, 3: 0}
    for r in results:
        if r.status not in ("ok", "skipped"):
            continue
        meta_path = output_dir / "shards" / f"{r.shard_id}.metadata.json"
        if not meta_path.exists():
            continue
        shard_meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if shard_meta.get("status") != "ok":
            continue
        npz_path = output_dir / "shards" / f"{r.shard_id}.npz"
        shard_meta["npz_path"] = str(npz_path.relative_to(output_dir))
        shard_meta_list.append(shard_meta)

        if npz_path.exists():
            with np.load(str(npz_path), allow_pickle=True) as data:
                mask = np.asarray(data["mask"], dtype=bool)
                if mask.any():
                    action_ids = np.asarray(data["action_ids"], dtype=np.int32)
                    path_indices = np.asarray(data["path_indices"], dtype=np.int32)
                    global_max_action_id = max(
                        global_max_action_id, int(action_ids[mask].max())
                    )
                    global_max_path_idx = max(
                        global_max_path_idx, int(path_indices[mask].max())
                    )
                if "deep_path_gate" in data.files:
                    dpg = np.asarray(data["deep_path_gate"], dtype=np.int32)
                    global_groups_e0 += int((dpg == 0).sum())
                    global_groups_e1 += int((dpg == 1).sum())
                if "depth_stratum" in data.files:
                    ds = np.asarray(data["depth_stratum"], dtype=np.int32)
                    for stratum in range(4):
                        global_stratum_counts[stratum] += int((ds == stratum).sum())

    # Best-effort modulation count from a prototype env.
    try:
        proto = make_env(
            args.topology,
            args.num_slots,
            args.num_servers,
            0,
            modulation_profile=args.modulation_profile,
        )
        modulation_count = len(proto.mod_reg.formats)
    except Exception:
        modulation_count = 4

    metadata = {
        "topology": args.topology,
        "num_slots": args.num_slots,
        "num_servers": args.num_servers,
        "K_C": args.k_paths_c,
        "K_path": args.k_paths_r,
        "K_prop": args.ppo_top_k,
        "H": args.label_horizon,
        "H_definition": (
            "H is the number of future transitions rolled out *after* the "
            "current R-side action is executed, i.e. additional future requests "
            "beyond the request that owns the candidate set."
        ),
        "gamma": args.gamma,
        "path_sort_strategy_c": args.path_sort_strategy_c,
        "path_sort_strategy_r": args.path_sort_strategy_r,
        "block_sort_strategy_c": args.block_sort_strategy_c,
        "block_sort_strategy_r": args.block_sort_strategy_r,
        "max_blocks": args.max_blocks,
        "modulation_count": modulation_count,
        "candidate_mode": args.candidate_mode,
        "max_candidates": args.max_candidates,
        "ppo_top_k": args.ppo_top_k,
        "num_random_candidates": 0,
        "min_candidates": 1,
        "candidate_seed": 12345,
        "horizon": args.label_horizon,
        "return_coefs": LABEL_COEFS,
        "label_coefs": LABEL_COEFS,
        "continuation_checkpoints": {
            "agent_c": args.agent_c_checkpoint,
            "agent_r": args.agent_r_checkpoint,
        },
        "checkpoint_sha256": checkpoint_sha256,
        "schema_version": "1.3.1",
        "component_arrays": config["component_arrays"],
        "component_shapes": {
            "per_candidate_1d": ["current_block", "current_nsb", "current_overload",
                                   "immediate_delay", "immediate_fs", "actual_horizon",
                                   "ppo_logits", "ppo_rank"],
            "per_candidate_per_step": ["future_blocked_per_step", "future_nsb_per_step",
                                        "future_overload_per_step", "future_delay_per_step",
                                        "future_fs_per_step"],
        },
        "traffic_params": {
            "arrival_interval": args.arrival_interval,
            "holding_min": args.holding_min,
            "holding_max": args.holding_max,
            "deadline_min": args.deadline_min,
            "deadline_max": args.deadline_max,
            "size_min_mb": args.size_min_mb,
            "size_max_mb": args.size_max_mb,
            "edge_cost_min": args.edge_cost_min,
            "edge_cost_max": args.edge_cost_max,
            "num_splits": args.num_splits,
            "split_profile": args.split_profile,
        },
        "max_observed_action_id": global_max_action_id,
        "max_observed_path_idx": global_max_path_idx,
        "group_filter": args.group_filter,
        "groups_e0": global_groups_e0,
        "groups_e1": global_groups_e1,
        "groups_total": global_groups_e0 + global_groups_e1,
        "depth_stratum_counts": global_stratum_counts,
        "common_future_trace_consistency": True,
        "feature_names": list(feature_names),
        "feature_dim": len(feature_names),
        "config": config,
        "config_hash": _config_hash(config),
        "shards": shard_meta_list,
    }

    metadata["generation_seconds"] = time.perf_counter() - start_time

    _atomic_write_json(output_dir / "metadata.json", metadata)
    print(f"[dataset] Wrote top-level metadata to {output_dir / 'metadata.json'}")

    n_ok = sum(1 for r in results if r.status == "ok")
    n_err = sum(1 for r in results if r.status == "error")
    print(
        f"[dataset] Finished: ok={n_ok} error={n_err} shards={len(shard_meta_list)}"
    )


if __name__ == "__main__":
    main()
