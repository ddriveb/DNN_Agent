#!/usr/bin/env python3
"""Oracle Ladder Phase 1: quantify post-v1.2 blocking headroom.

Four modes are evaluated on the low-blocking S100 scenario:
  1. baseline_v12
  2. r_oracle_in_mask
  3. r_expanded_block_oracle
  4. joint_c_topk_r_oracle

No new models are trained; existing checkpoints are used read-only.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
)
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import (
    load_ranking_checkpoint,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


# ---------------------------------------------------------------------------
# Return coefficients (matching the task specification)
# ---------------------------------------------------------------------------
RETURN_COEFS = {
    "current_block": 3.0,
    "future_block": 4.0,
    "future_nsb": 3.0,
    "future_server_overload": 2.0,
    "delay": 0.03,
    "fs": 0.05,
}


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class StateOutcome:
    seed: int
    episode: int
    request_index: int
    req_id: int

    baseline_success: bool
    baseline_reason: str
    baseline_future_blocked: int
    baseline_future_nsb: int
    baseline_future_overload: int
    baseline_delay_mean: float
    baseline_avg_fs: float

    selected_c_idx: int
    selected_split: int
    selected_server: int
    selected_r_idx: int

    raw_c_mask: np.ndarray = field(repr=False)
    raw_r_mask: np.ndarray = field(repr=False)

    # Oracle results (populated for blocked states)
    r_oracle_success: Optional[bool] = None
    r_oracle_r_idx: Optional[int] = None
    r_oracle_future_blocked: Optional[int] = None
    r_oracle_changed: Optional[bool] = None

    expanded_oracle_success: Optional[bool] = None
    expanded_oracle_r_idx: Optional[int] = None
    expanded_oracle_future_blocked: Optional[int] = None
    expanded_oracle_changed: Optional[bool] = None
    expanded_candidate_count: Optional[int] = None

    joint_oracle_success: Optional[bool] = None
    joint_oracle_c_idx: Optional[int] = None
    joint_oracle_r_idx: Optional[int] = None
    joint_oracle_future_blocked: Optional[int] = None
    joint_oracle_changed_c: Optional[bool] = None
    joint_oracle_changed_r: Optional[bool] = None
    joint_pair_count: Optional[int] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "request_index": self.request_index,
            "req_id": self.req_id,
            "baseline_success": self.baseline_success,
            "baseline_reason": self.baseline_reason,
            "baseline_future_blocked": self.baseline_future_blocked,
            "baseline_future_nsb": self.baseline_future_nsb,
            "baseline_future_overload": self.baseline_future_overload,
            "baseline_delay_mean": self.baseline_delay_mean,
            "baseline_avg_fs": self.baseline_avg_fs,
            "selected_c_idx": self.selected_c_idx,
            "selected_split": self.selected_split,
            "selected_server": self.selected_server,
            "selected_r_idx": self.selected_r_idx,
            "r_oracle_success": self.r_oracle_success,
            "r_oracle_r_idx": self.r_oracle_r_idx,
            "r_oracle_future_blocked": self.r_oracle_future_blocked,
            "r_oracle_changed": self.r_oracle_changed,
            "expanded_oracle_success": self.expanded_oracle_success,
            "expanded_oracle_r_idx": self.expanded_oracle_r_idx,
            "expanded_oracle_future_blocked": self.expanded_oracle_future_blocked,
            "expanded_oracle_changed": self.expanded_oracle_changed,
            "expanded_candidate_count": self.expanded_candidate_count,
            "joint_oracle_success": self.joint_oracle_success,
            "joint_oracle_c_idx": self.joint_oracle_c_idx,
            "joint_oracle_r_idx": self.joint_oracle_r_idx,
            "joint_oracle_future_blocked": self.joint_oracle_future_blocked,
            "joint_oracle_changed_c": self.joint_oracle_changed_c,
            "joint_oracle_changed_r": self.joint_oracle_changed_r,
            "joint_pair_count": self.joint_pair_count,
        }


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------
def _score_legal_r_actions(
    env,
    req,
    obs_c,
    obs_r,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    split_id: int,
    server_id: int,
    device: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (legal_r_indices, scores) for the given obs_r."""
    r_features, r_mask = agent_r_for_features.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool))
    if len(legal) == 0:
        return legal, np.array([])
    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - ranker_mean) / ranker_std
    with torch.no_grad():
        scores = ranker_model(
            torch.as_tensor(normalized, dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()
    return legal, scores


def _select_v12_r_action(
    env,
    req,
    obs_c,
    obs_r,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    split_id: int,
    server_id: int,
    device: str,
) -> int:
    legal, scores = _score_legal_r_actions(
        env, req, obs_c, obs_r, agent_r_for_features,
        ranker_model, ranker_mean, ranker_std, split_id, server_id, device,
    )
    if len(legal) == 0:
        return 0
    return int(legal[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# Future rollout with frozen C + v1.2 ranker
# ---------------------------------------------------------------------------
def _rollout_future_ranker(
    env,
    requests: List[Any],
    start_idx: int,
    horizon: int,
    agent_c,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    num_servers: int,
    device: str,
) -> Dict[str, Any]:
    """Roll out H future requests using delay-aware PPO-C + v1.2 ranker."""
    blocked = 0
    raw_empty = 0
    no_suitable_block = 0
    server_overload = 0
    delays: List[float] = []
    fs_values: List[float] = []

    for offset in range(horizon):
        t = start_idx + offset
        if t >= len(requests):
            break
        req = requests[t]
        env.advance_time(req.arrival_time)

        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if int(raw_c_mask.sum()) == 0:
            raw_empty += 1
            blocked += 1
            env.step((0, 0), (0, 0, 0))
            delays.append(0.0)
            fs_values.append(0.0)
            continue

        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split, server = decode_agent_c_action(int(c_idx), num_servers)
        obs_r = build_agent_r_observation(env, req, split, server)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        if int(raw_r_mask.sum()) == 0:
            raw_empty += 1
            blocked += 1
            env.step((split, server), (0, 0, 0))
            delays.append(0.0)
            fs_values.append(0.0)
            continue

        r_idx = _select_v12_r_action(
            env, req, obs_c, obs_r, agent_r_for_features,
            ranker_model, ranker_mean, ranker_std, split, server, device,
        )
        r_action = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split, server), r_action)

        if not info.get("success", False):
            blocked += 1
            reason = info.get("reason", "")
            if reason == "no_suitable_block":
                no_suitable_block += 1
            elif reason in ("server_overload", "server_saturated"):
                server_overload += 1
            delays.append(0.0)
            fs_values.append(0.0)
        else:
            delays.append(float(info.get("delay_ms", 0.0)))
            fs_values.append(float(info.get("num_slots", 0)))

    denom = max(len(delays), 1)
    return {
        "blocked": blocked,
        "raw_mask_empty": raw_empty,
        "no_suitable_block": no_suitable_block,
        "server_overload": server_overload,
        "delay_mean": float(np.sum(delays) / denom),
        "avg_fs": float(np.sum(fs_values) / denom),
    }


def _compute_return(current_block: int, future: Dict[str, Any]) -> float:
    return float(
        -RETURN_COEFS["current_block"] * current_block
        - RETURN_COEFS["future_block"] * future["blocked"]
        - RETURN_COEFS["future_nsb"] * future["no_suitable_block"]
        - RETURN_COEFS["future_server_overload"] * future["server_overload"]
        - RETURN_COEFS["delay"] * future["delay_mean"]
        - RETURN_COEFS["fs"] * future["avg_fs"]
    )


def _failure_reason(info: Dict[str, Any], raw_c_empty: bool, raw_r_empty: bool, success: bool) -> str:
    if success:
        return "success"
    if raw_c_empty:
        return "no_valid_c_action"
    if raw_r_empty:
        return "raw_mask_empty"
    reason = info.get("reason", "other")
    if reason == "no_suitable_block":
        return "no_suitable_block"
    if reason in ("server_overload", "server_saturated"):
        return "server_overload"
    if reason == "deadline_infeasible":
        return "deadline_infeasible"
    if reason == "allocation_failed":
        return "no_suitable_block"
    if reason == "fs_too_large":
        return "fs_too_large"
    return "other"


# ---------------------------------------------------------------------------
# Oracle evaluators
# ---------------------------------------------------------------------------
def _eval_r_oracle_in_mask(
    state: StateOutcome,
    pre_r_snapshot,
    obs_c,
    obs_r,
    agent_c,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    requests: List[Any],
    horizon: int,
    num_servers: int,
    device: str,
) -> None:
    legal = np.flatnonzero(state.raw_r_mask).tolist()
    if not legal:
        state.r_oracle_success = False
        state.r_oracle_r_idx = state.selected_r_idx
        state.r_oracle_changed = False
        return

    best_ret = -np.inf
    best_success = False
    best_r = state.selected_r_idx
    best_future_blocked = None

    for r_idx in legal:
        branch = copy.deepcopy(pre_r_snapshot)
        r_action = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), branch.max_blocks)
        _, _, _, info = branch.step((state.selected_split, state.selected_server), r_action)
        success = bool(info.get("success", False))
        future = _rollout_future_ranker(
            branch, requests, state.request_index + 1, horizon,
            agent_c, agent_r_for_features, ranker_model, ranker_mean, ranker_std,
            num_servers, device,
        )
        ret = _compute_return(0 if success else 1, future)
        if ret > best_ret:
            best_ret = ret
            best_success = success
            best_r = int(r_idx)
            best_future_blocked = future["blocked"]

    state.r_oracle_success = best_success
    state.r_oracle_r_idx = best_r
    state.r_oracle_future_blocked = best_future_blocked
    state.r_oracle_changed = best_r != state.selected_r_idx


def _eval_r_expanded_block_oracle(
    state: StateOutcome,
    pre_r_snapshot,
    obs_c,
    agent_c,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    requests: List[Any],
    horizon: int,
    num_servers: int,
    max_blocks_expanded: int,
    device: str,
) -> None:
    # Build an expanded candidate set by temporarily raising max_blocks on a copy.
    branch_for_obs = copy.deepcopy(pre_r_snapshot)
    branch_for_obs.max_blocks = max_blocks_expanded
    obs_r_exp = build_agent_r_observation(
        branch_for_obs, requests[state.request_index], state.selected_split, state.selected_server
    )
    raw_r_mask_exp = np.asarray(obs_r_exp["agent_r_mask"], dtype=bool)
    legal = np.flatnonzero(raw_r_mask_exp).tolist()
    state.expanded_candidate_count = len(legal)

    if not legal:
        state.expanded_oracle_success = False
        state.expanded_oracle_r_idx = state.selected_r_idx
        state.expanded_oracle_changed = False
        return

    best_ret = -np.inf
    best_success = False
    best_r = state.selected_r_idx
    best_future_blocked = None
    num_mods = len(obs_r_exp["mod_names"])

    for r_idx in legal:
        branch = copy.deepcopy(pre_r_snapshot)
        branch.max_blocks = max_blocks_expanded
        r_action = decode_agent_r_action(int(r_idx), num_mods, branch.max_blocks)
        _, _, _, info = branch.step((state.selected_split, state.selected_server), r_action)
        success = bool(info.get("success", False))
        # Restore max_blocks for future rollout so the ranker stays in-distribution.
        branch.max_blocks = 10
        future = _rollout_future_ranker(
            branch, requests, state.request_index + 1, horizon,
            agent_c, agent_r_for_features, ranker_model, ranker_mean, ranker_std,
            num_servers, device,
        )
        ret = _compute_return(0 if success else 1, future)
        if ret > best_ret:
            best_ret = ret
            best_success = success
            best_r = int(r_idx)
            best_future_blocked = future["blocked"]

    state.expanded_oracle_success = best_success
    state.expanded_oracle_r_idx = best_r
    state.expanded_oracle_future_blocked = best_future_blocked
    state.expanded_oracle_changed = best_r != state.selected_r_idx


def _eval_joint_c_topk_r_oracle(
    state: StateOutcome,
    pre_c_snapshot,
    obs_c,
    agent_c,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    requests: List[Any],
    horizon: int,
    num_servers: int,
    top_k_r: int,
    device: str,
) -> None:
    c_legal = np.flatnonzero(state.raw_c_mask).tolist()
    if not c_legal:
        state.joint_oracle_success = False
        state.joint_oracle_c_idx = state.selected_c_idx
        state.joint_oracle_r_idx = state.selected_r_idx
        state.joint_oracle_changed_c = False
        state.joint_oracle_changed_r = False
        state.joint_pair_count = 0
        return

    best_ret = -np.inf
    best_success = False
    best_c = state.selected_c_idx
    best_r = state.selected_r_idx
    best_future_blocked = None
    pair_count = 0

    for c_idx in c_legal:
        split, server = decode_agent_c_action(int(c_idx), num_servers)
        obs_r_c = build_agent_r_observation(pre_c_snapshot, requests[state.request_index], split, server)
        raw_r_mask_c = np.asarray(obs_r_c["agent_r_mask"], dtype=bool)
        r_legal = np.flatnonzero(raw_r_mask_c).tolist()
        if not r_legal:
            continue

        # Score all legal R actions with v1.2 ranker and keep top-K.
        legal_arr, scores = _score_legal_r_actions(
            pre_c_snapshot, requests[state.request_index], obs_c, obs_r_c,
            agent_r_for_features, ranker_model, ranker_mean, ranker_std,
            split, server, device,
        )
        top_r_indices = legal_arr[np.argsort(-scores)[: min(top_k_r, len(legal_arr))]].tolist()

        for r_idx in top_r_indices:
            pair_count += 1
            branch = copy.deepcopy(pre_c_snapshot)
            r_action = decode_agent_r_action(int(r_idx), len(obs_r_c["mod_names"]), branch.max_blocks)
            _, _, _, info = branch.step((split, server), r_action)
            success = bool(info.get("success", False))
            future = _rollout_future_ranker(
                branch, requests, state.request_index + 1, horizon,
                agent_c, agent_r_for_features, ranker_model, ranker_mean, ranker_std,
                num_servers, device,
            )
            ret = _compute_return(0 if success else 1, future)
            if ret > best_ret:
                best_ret = ret
                best_success = success
                best_c = int(c_idx)
                best_r = int(r_idx)
                best_future_blocked = future["blocked"]

    state.joint_oracle_success = best_success
    state.joint_oracle_c_idx = best_c
    state.joint_oracle_r_idx = best_r
    state.joint_oracle_future_blocked = best_future_blocked
    state.joint_oracle_changed_c = best_c != state.selected_c_idx
    state.joint_oracle_changed_r = best_r != state.selected_r_idx
    state.joint_pair_count = pair_count


# ---------------------------------------------------------------------------
# Episode diagnostic
# ---------------------------------------------------------------------------
def _diagnose_episode(
    env,
    requests: List[Any],
    agent_c,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    num_servers: int,
    max_blocks_expanded: int,
    horizon: int,
    top_k_r: int,
    device: str,
    seed: int,
    episode_index: int,
) -> List[StateOutcome]:
    env.reset(requests)
    states: List[StateOutcome] = []

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        pre_c_snapshot = copy.deepcopy(env)

        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        selected_split, selected_server = decode_agent_c_action(int(c_idx), num_servers)

        obs_r = build_agent_r_observation(env, req, selected_split, selected_server)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        r_idx = _select_v12_r_action(
            env, req, obs_c, obs_r, agent_r_for_features,
            ranker_model, ranker_mean, ranker_std,
            selected_split, selected_server, device,
        )

        pre_r_snapshot = copy.deepcopy(env)
        r_action = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((selected_split, selected_server), r_action)
        success = bool(info.get("success", False))

        # Future rollout from a copy of the post-baseline state so the main env
        # remains intact for the next request.
        post_step_env = copy.deepcopy(env)
        future = _rollout_future_ranker(
            post_step_env, requests, request_index + 1, horizon,
            agent_c, agent_r_for_features, ranker_model, ranker_mean, ranker_std,
            num_servers, device,
        )

        raw_c_empty = int(raw_c_mask.sum()) == 0
        raw_r_empty = int(raw_r_mask.sum()) == 0
        reason = _failure_reason(info, raw_c_empty, raw_r_empty, success)

        state = StateOutcome(
            seed=seed,
            episode=episode_index,
            request_index=request_index,
            req_id=int(req.req_id),
            baseline_success=success,
            baseline_reason=reason,
            baseline_future_blocked=future["blocked"],
            baseline_future_nsb=future["no_suitable_block"],
            baseline_future_overload=future["server_overload"],
            baseline_delay_mean=future["delay_mean"],
            baseline_avg_fs=future["avg_fs"],
            selected_c_idx=int(c_idx),
            selected_split=selected_split,
            selected_server=selected_server,
            selected_r_idx=int(r_idx),
            raw_c_mask=raw_c_mask,
            raw_r_mask=raw_r_mask,
        )

        # Only run expensive oracles on blocked states (the residual floor).
        if not success:
            _eval_r_oracle_in_mask(
                state, pre_r_snapshot, obs_c, obs_r, agent_c, agent_r_for_features,
                ranker_model, ranker_mean, ranker_std, requests, horizon,
                num_servers, device,
            )
            _eval_r_expanded_block_oracle(
                state, pre_r_snapshot, obs_c, agent_c, agent_r_for_features,
                ranker_model, ranker_mean, ranker_std, requests, horizon,
                num_servers, max_blocks_expanded, device,
            )
            _eval_joint_c_topk_r_oracle(
                state, pre_c_snapshot, obs_c, agent_c, agent_r_for_features,
                ranker_model, ranker_mean, ranker_std, requests, horizon,
                num_servers, top_k_r, device,
            )

        states.append(state)

    return states


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------
def _aggregate(states: List[StateOutcome], horizon: int) -> Dict[str, Any]:
    total = len(states)
    blocked_states = [s for s in states if not s.baseline_success]
    blocked = len(blocked_states)
    baseline_rate = blocked / max(total, 1)

    def _current_rate(success_field: str):
        improved = sum(1 for s in blocked_states if getattr(s, success_field))
        return (blocked - improved) / max(total, 1)

    def _future_rate(future_field: str):
        arr = [s for s in blocked_states if getattr(s, future_field) is not None]
        if not arr:
            return 0.0
        total_fb = sum(getattr(s, future_field) for s in arr)
        return total_fb / max(len(arr) * horizon, 1)

    def _headroom(success_field: str):
        return (baseline_rate - _current_rate(success_field)) * 100.0

    reason_counts: Dict[str, int] = {}
    for s in blocked_states:
        reason_counts[s.baseline_reason] = reason_counts.get(s.baseline_reason, 0) + 1

    # Coverage stats (over blocked states, since oracles are only run there).
    legal_r_counts = [int(s.raw_r_mask.sum()) for s in blocked_states]
    expanded_counts = [s.expanded_candidate_count for s in blocked_states if s.expanded_candidate_count is not None]
    joint_pair_counts = [s.joint_pair_count for s in blocked_states if s.joint_pair_count is not None]

    # Change rates over blocked states where oracle was evaluated.
    r_changed = sum(1 for s in blocked_states if s.r_oracle_changed) / max(len(blocked_states), 1)
    expanded_changed = sum(1 for s in blocked_states if s.expanded_oracle_changed) / max(len(blocked_states), 1)
    joint_changed_c = sum(1 for s in blocked_states if s.joint_oracle_changed_c) / max(len(blocked_states), 1)
    joint_changed_r = sum(1 for s in blocked_states if s.joint_oracle_changed_r) / max(len(blocked_states), 1)

    headrooms = {
        "r_in_mask_headroom_pp": _headroom("r_oracle_success"),
        "r_expanded_block_headroom_pp": _headroom("expanded_oracle_success"),
        "joint_c_topk_r_headroom_pp": _headroom("joint_oracle_success"),
    }

    # Verdict.
    max_h = max(headrooms.values())
    if max_h >= 1.0:
        verdict = "PROCEED"
    elif max_h < 0.5:
        verdict = "STOP_ACTION_SPACE_EXPANSION"
    else:
        verdict = "MARGINAL"

    # Determine direction.
    direction = "none"
    if verdict == "PROCEED":
        best_key = max(headrooms, key=headrooms.get)
        if best_key == "r_in_mask_headroom_pp":
            direction = "retrain_R_ranker_on_in_mask_candidates"
        elif best_key == "r_expanded_block_headroom_pp":
            direction = "R-expanded_candidate_ranker"
        else:
            direction = "C-R_joint_topK_ranker"

    return {
        "total_states": total,
        "baseline_blocked": blocked,
        "baseline_blocking_rate": baseline_rate,
        "baseline_future_block_rate": _future_rate("baseline_future_blocked"),
        "r_oracle_blocking_rate": _current_rate("r_oracle_success"),
        "r_oracle_future_block_rate": _future_rate("r_oracle_future_blocked"),
        "expanded_oracle_blocking_rate": _current_rate("expanded_oracle_success"),
        "expanded_oracle_future_block_rate": _future_rate("expanded_oracle_future_blocked"),
        "joint_oracle_blocking_rate": _current_rate("joint_oracle_success"),
        "joint_oracle_future_block_rate": _future_rate("joint_oracle_future_blocked"),
        **headrooms,
        "evaluated_states": len(blocked_states),
        "legal_r_count_mean": float(np.mean(legal_r_counts)) if legal_r_counts else 0.0,
        "expanded_candidate_count_mean": float(np.mean(expanded_counts)) if expanded_counts else 0.0,
        "joint_pair_count_mean": float(np.mean(joint_pair_counts)) if joint_pair_counts else 0.0,
        "r_oracle_changed_rate": r_changed,
        "expanded_oracle_changed_rate": expanded_changed,
        "joint_oracle_changed_c_rate": joint_changed_c,
        "joint_oracle_changed_r_rate": joint_changed_r,
        "failure_reason_counts": reason_counts,
        "failure_reason_rates": {k: v / max(blocked, 1) for k, v in reason_counts.items()},
        "verdict": verdict,
        "direction": direction,
        "state_results": [s.to_dict() for s in states],
    }


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# Oracle Ladder Phase 1 — v1.2 Residual Headroom (S100)",
        "",
        f"- Total states: {summary['total_states']}",
        f"- Baseline v1.2 blocked: {summary['baseline_blocked']} ({summary['baseline_blocking_rate']*100:.3f}%)",
        f"- Oracle evaluated on blocked states: {summary['evaluated_states']}",
        "",
        "## Blocking and future block rates",
        "",
        "| Mode | Current blocking rate | Future block rate | Headroom (pp) |",
        "|---|---:|---:|---:|",
        f"| baseline_v12 | {summary['baseline_blocking_rate']*100:.3f}% | {summary['baseline_future_block_rate']*100:.2f}% | — |",
        f"| r_oracle_in_mask | {summary['r_oracle_blocking_rate']*100:.3f}% | {summary['r_oracle_future_block_rate']*100:.2f}% | {summary['r_in_mask_headroom_pp']:.3f} |",
        f"| r_expanded_block_oracle | {summary['expanded_oracle_blocking_rate']*100:.3f}% | {summary['expanded_oracle_future_block_rate']*100:.2f}% | {summary['r_expanded_block_headroom_pp']:.3f} |",
        f"| joint_c_topk_r_oracle | {summary['joint_oracle_blocking_rate']*100:.3f}% | {summary['joint_oracle_future_block_rate']*100:.2f}% | {summary['joint_c_topk_r_headroom_pp']:.3f} |",
        "",
        "## Coverage and behavior change",
        "",
        "| Metric | Value |",
        "|---|---:|",
        f"| Evaluated blocked states | {summary['evaluated_states']} |",
        f"| Mean legal R actions (in-mask) | {summary['legal_r_count_mean']:.2f} |",
        f"| Mean expanded candidates | {summary['expanded_candidate_count_mean']:.2f} |",
        f"| Mean joint (C, top-3 R) pairs | {summary['joint_pair_count_mean']:.2f} |",
        f"| R-oracle changed action rate | {summary['r_oracle_changed_rate']*100:.2f}% |",
        f"| Expanded-oracle changed action rate | {summary['expanded_oracle_changed_rate']*100:.2f}% |",
        f"| Joint-oracle changed C rate | {summary['joint_oracle_changed_c_rate']*100:.2f}% |",
        f"| Joint-oracle changed R rate | {summary['joint_oracle_changed_r_rate']*100:.2f}% |",
        "",
        "## Failure reason distribution (baseline blocked states)",
        "",
        "| Reason | Count | Share |",
        "|---|---:|---:|",
    ]
    for reason, count in sorted(summary["failure_reason_counts"].items(), key=lambda x: -x[1]):
        lines.append(
            f"| {reason} | {count} | {summary['failure_reason_rates'][reason]*100:.2f}% |"
        )

    lines += [
        "",
        f"## Verdict: {summary['verdict']}",
        f"- Direction: {summary['direction']}",
    ]

    path.write_text("\n".join(lines), encoding="utf-8")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def run(args: argparse.Namespace) -> Dict[str, Any]:
    start = time.time()
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

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
    print(f"Loading PPO-R (feature backbone): {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    print(f"Loading v1.2 ranker: {args.ranking_checkpoint}")
    ranker_model, ranker_mean, ranker_std, _ = load_ranking_checkpoint(
        args.ranking_checkpoint, args.device
    )

    c_ref = {k: v.clone() for k, v in agent_c.policy_net.state_dict().items()}
    r_ref = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}

    all_states: List[StateOutcome] = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        for episode_index in range(args.episodes):
            src = int(rng.randint(0, env_proto.net.NUM_NODES))
            requests = generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            )
            env = make_env(
                topology=args.topology,
                num_slots=args.num_slots,
                num_servers=args.num_servers,
                seed=42,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            t0 = time.time()
            states = _diagnose_episode(
                env, requests, agent_c, agent_r,
                ranker_model, ranker_mean, ranker_std,
                args.num_servers, args.max_blocks_expanded,
                args.horizon, args.top_k_r, args.device,
                seed, episode_index,
            )
            all_states.extend(states)
            print(
                f"[seed={seed}][episode={episode_index+1}/{args.episodes}] "
                f"blocked={sum(1 for s in states if not s.baseline_success)}/{len(states)} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    c_unchanged = all(torch.equal(agent_c.policy_net.state_dict()[k], c_ref[k]) for k in c_ref)
    r_unchanged = all(torch.equal(agent_r.policy_net.state_dict()[k], r_ref[k]) for k in r_ref)

    summary = _aggregate(all_states, args.horizon)
    summary["elapsed_seconds"] = time.time() - start
    summary["frozen_backends_unchanged"] = bool(c_unchanged and r_unchanged)
    summary["config"] = vars(args)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--seeds", default="3030,4040,5050")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=100)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--max_blocks_expanded", type=int, default=32)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.15)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument("--top_k_r", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/oracle_ladder_v12_s100.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/oracle_ladder_v12_s100.md")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    summary = run(args)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    _write_markdown(out_md, summary)
    print(f"Saved {out_json}")
    print(f"Saved {out_md}")
    print(
        f"Baseline blocking: {summary['baseline_blocking_rate']*100:.3f}% | "
        f"R-in-mask headroom: {summary['r_in_mask_headroom_pp']:.3f}pp | "
        f"Expanded headroom: {summary['r_expanded_block_headroom_pp']:.3f}pp | "
        f"Joint headroom: {summary['joint_c_topk_r_headroom_pp']:.3f}pp | "
        f"Verdict: {summary['verdict']}"
    )
