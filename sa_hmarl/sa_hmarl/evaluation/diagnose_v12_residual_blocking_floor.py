#!/usr/bin/env python3
"""Residual blocking floor audit for the v1.2 counterfactual R-ranker.

For every request in a v1.2 closed-loop rollout, we record whether the ranker's
chosen action succeeds.  When it fails, we restore the pre-decision env snapshot
and exhaustively enumerate all raw-mask-legal (split, server, path/mod/block)
combinations to classify the failure as:

  - unavoidable_block:   no C/R combination can make the current request succeed.
  - C_avoidable:         the original split/server cannot be saved by any R
                         action, but another split/server can.
  - R_avoidable:         under the original split/server some R action succeeds,
                         but v1.2 did not pick it.
  - trajectory_avoidable: no current success is possible, yet different failed
                         actions lead to different H-step future blocking counts.

The script also reports the failure reason distribution and per-seed statistics.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import subprocess
import sys
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
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _rollout_future,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
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
# Data structures
# ---------------------------------------------------------------------------
@dataclass
class BlockedRecord:
    seed: int
    episode: int
    request_index: int
    req_id: int

    selected_split: int
    selected_server: int
    selected_r_idx: int

    category: str  # unavoidable_block, C_avoidable, R_avoidable, trajectory_avoidable, unknown
    reason: str

    # Immediate enumeration stats
    total_c_actions: int
    total_r_actions_evaluated: int
    successful_c_actions: int
    successful_r_actions_under_original_c: int

    # Future-oracle stats (populated when no immediate success)
    chosen_future_blocked: Optional[int] = None
    best_future_blocked: Optional[int] = None
    future_blocked_range: Optional[int] = None

    # R-avoidable detail
    successful_alternative_r_idx: Optional[int] = None
    successful_alternative_path_idx: Optional[int] = None
    successful_alternative_mod_idx: Optional[int] = None
    successful_alternative_block_idx: Optional[int] = None
    successful_alternative_path_km: Optional[float] = None
    successful_alternative_required_fs: Optional[float] = None
    successful_alternative_block_waste: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "request_index": self.request_index,
            "req_id": self.req_id,
            "selected_split": self.selected_split,
            "selected_server": self.selected_server,
            "selected_r_idx": self.selected_r_idx,
            "category": self.category,
            "reason": self.reason,
            "total_c_actions": self.total_c_actions,
            "total_r_actions_evaluated": self.total_r_actions_evaluated,
            "successful_c_actions": self.successful_c_actions,
            "successful_r_actions_under_original_c": self.successful_r_actions_under_original_c,
            "chosen_future_blocked": self.chosen_future_blocked,
            "best_future_blocked": self.best_future_blocked,
            "future_blocked_range": self.future_blocked_range,
            "successful_alternative_r_idx": self.successful_alternative_r_idx,
            "successful_alternative_path_idx": self.successful_alternative_path_idx,
            "successful_alternative_mod_idx": self.successful_alternative_mod_idx,
            "successful_alternative_block_idx": self.successful_alternative_block_idx,
            "successful_alternative_path_km": self.successful_alternative_path_km,
            "successful_alternative_required_fs": self.successful_alternative_required_fs,
            "successful_alternative_block_waste": self.successful_alternative_block_waste,
        }


@dataclass
class EpisodeResult:
    seed: int
    episode: int
    total_requests: int
    blocked_count: int
    raw_mask_empty_count: int
    blocked_records: List[BlockedRecord] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "total_requests": self.total_requests,
            "blocked_count": self.blocked_count,
            "raw_mask_empty_count": self.raw_mask_empty_count,
            "blocked_records": [r.to_dict() for r in self.blocked_records],
        }


# ---------------------------------------------------------------------------
# Policy helpers
# ---------------------------------------------------------------------------
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
    """Select the R action that the v1.2 ranker would choose."""
    r_features, r_mask = agent_r_for_features.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])

    online = np.stack([
        _r_feature_vector(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
        for a in legal
    ])
    normalized = (online - ranker_mean) / ranker_std
    with torch.no_grad():
        scores = ranker_model(
            torch.as_tensor(normalized, dtype=torch.float32, device=device)
        ).cpu().numpy()
    return int(legal[int(np.argmax(scores))])


# ---------------------------------------------------------------------------
# Classification helper (pure function, easy to unit test)
# ---------------------------------------------------------------------------
def _classify_from_outcomes(
    original_c: int,
    original_r: int,
    success_matrix: np.ndarray,
    future_blocked_matrix: Optional[np.ndarray] = None,
) -> Tuple[str, Optional[int], Optional[int], Optional[int]]:
    """Return (category, chosen_future, best_future, range).

    success_matrix[c, r] is True if that C/R pair makes the current request succeed.
    future_blocked_matrix[c, r] is the H-step future blocked count from that pair.
    """
    original_success = bool(success_matrix[original_c, original_r])
    if original_success:
        # Should not happen for a blocked request, but keep robust.
        return "unknown", None, None, None

    any_success = bool(success_matrix.any())
    original_c_has_success = bool(success_matrix[original_c, :].any())

    if any_success:
        if original_c_has_success:
            return "R_avoidable", None, None, None
        return "C_avoidable", None, None, None

    # No immediate success -> unavoidable or trajectory avoidable.
    if future_blocked_matrix is None:
        return "unavoidable_block", None, None, None

    chosen_future = int(future_blocked_matrix[original_c, original_r])
    best_future = int(future_blocked_matrix.min())
    range_val = chosen_future - best_future
    if range_val > 0:
        return "trajectory_avoidable", chosen_future, best_future, range_val
    return "unavoidable_block", chosen_future, best_future, range_val


# ---------------------------------------------------------------------------
# Single blocked request diagnostic
# ---------------------------------------------------------------------------
def _diagnose_blocked_request(
    snapshot,
    req,
    original_c_idx: int,
    original_r_idx: int,
    selected_split: int,
    selected_server: int,
    obs_c: Dict[str, Any],
    raw_c_mask: np.ndarray,
    agent_c,
    agent_r_for_features,
    ranker_model,
    ranker_mean,
    ranker_std,
    num_servers: int,
    max_blocks: int,
    requests: List[Any],
    request_index: int,
    horizon: int,
    device: str,
) -> BlockedRecord:
    """Enumerate all C/R combos from snapshot and classify the blocked request."""
    c_legal = np.flatnonzero(raw_c_mask).tolist()
    total_c = len(c_legal)
    total_r_eval = 0
    successful_c_set: set = set()
    successful_r_under_original_c: set = set()

    # Matrices indexed by position in c_legal / r_legal per C.
    # We store results keyed by (c_idx_global, r_idx_global).
    success_by_pair: Dict[Tuple[int, int], bool] = {}
    info_by_pair: Dict[Tuple[int, int], Dict[str, Any]] = {}

    for c_pos, c_idx in enumerate(c_legal):
        split, server = decode_agent_c_action(int(c_idx), num_servers)
        obs_r_c = build_agent_r_observation(snapshot, req, split, server)
        raw_r_mask = np.asarray(obs_r_c["agent_r_mask"], dtype=bool)
        r_legal = np.flatnonzero(raw_r_mask).tolist()
        total_r_eval += len(r_legal)

        for r_idx in r_legal:
            branch = copy.deepcopy(snapshot)
            r_action = decode_agent_r_action(
                int(r_idx), len(obs_r_c["mod_names"]), max_blocks
            )
            _, _, _, info = branch.step((split, server), r_action)
            success = bool(info.get("success", False))
            success_by_pair[(int(c_idx), int(r_idx))] = success
            info_by_pair[(int(c_idx), int(r_idx))] = info
            if success:
                successful_c_set.add(int(c_idx))
                if int(c_idx) == original_c_idx:
                    successful_r_under_original_c.add(int(r_idx))

    record = BlockedRecord(
        seed=-1,
        episode=-1,
        request_index=-1,
        req_id=int(req.req_id),
        selected_split=selected_split,
        selected_server=selected_server,
        selected_r_idx=original_r_idx,
        category="unknown",
        reason="other",
        total_c_actions=total_c,
        total_r_actions_evaluated=total_r_eval,
        successful_c_actions=len(successful_c_set),
        successful_r_actions_under_original_c=len(successful_r_under_original_c),
    )

    # Determine reason from the original (c, r) outcome.
    if int(raw_c_mask.sum()) == 0:
        record.reason = "no_valid_c_action"
    else:
        # Build original R obs to check R mask.
        obs_r_orig = build_agent_r_observation(snapshot, req, selected_split, selected_server)
        raw_r_mask_orig = np.asarray(obs_r_orig["agent_r_mask"], dtype=bool)
        if int(raw_r_mask_orig.sum()) == 0:
            record.reason = "raw_mask_empty"
        else:
            orig_info = info_by_pair.get((original_c_idx, original_r_idx), {})
            reason = orig_info.get("reason", "other")
            if reason in ("server_overload", "server_saturated"):
                record.reason = "server_overload"
            elif reason == "no_suitable_block":
                record.reason = "no_suitable_block"
            elif reason == "deadline_infeasible":
                record.reason = "deadline_infeasible"
            else:
                record.reason = "other"

    any_success = len(successful_c_set) > 0
    original_c_has_success = original_c_idx in successful_c_set

    if any_success:
        if original_c_has_success:
            record.category = "R_avoidable"
            # Pick the successful alternative with the smallest R index as a representative.
            alt_r = min(successful_r_under_original_c)
            record.successful_alternative_r_idx = int(alt_r)
            obs_r_orig = build_agent_r_observation(
                snapshot, req, selected_split, selected_server
            )
            r_features, _ = agent_r_for_features.build_action_features(obs_r_orig)
            num_mods = len(obs_r_orig["mod_names"])
            path_idx, mod_idx, block_idx = decode_agent_r_action(alt_r, num_mods, max_blocks)
            record.successful_alternative_path_idx = path_idx
            record.successful_alternative_mod_idx = mod_idx
            record.successful_alternative_block_idx = block_idx
            record.successful_alternative_path_km = float(
                r_features[alt_r][FEATURE_NAMES.index("path_length_km")]
            )
            record.successful_alternative_required_fs = float(
                r_features[alt_r][FEATURE_NAMES.index("required_fs")]
            )
            record.successful_alternative_block_waste = float(
                r_features[alt_r][FEATURE_NAMES.index("block_waste")]
            )
        else:
            record.category = "C_avoidable"
        return record

    # No immediate success.  Before running futures, handle degenerate cases.
    if not c_legal:
        # No legal C action at all.
        record.category = "unavoidable_block"
        return record

    # Rebuild R-legal lists per C (cheap) and check whether the original C has any R action.
    c_index_map = {int(c_idx): i for i, c_idx in enumerate(c_legal)}
    r_legal_per_c: Dict[int, List[int]] = {}
    max_r_len = 0
    for c_idx in c_legal:
        split, server = decode_agent_c_action(int(c_idx), num_servers)
        obs_r_c = build_agent_r_observation(snapshot, req, split, server)
        raw_r_mask = np.asarray(obs_r_c["agent_r_mask"], dtype=bool)
        r_legal = np.flatnonzero(raw_r_mask).tolist()
        r_legal_per_c[int(c_idx)] = r_legal
        max_r_len = max(max_r_len, len(r_legal))

    r_legal_orig = r_legal_per_c.get(original_c_idx, [])
    if not r_legal_orig:
        # Original (split, server) has no feasible R action at all.
        record.category = "unavoidable_block"
        return record

    if max_r_len == 0:
        # No C has any feasible R action.
        record.category = "unavoidable_block"
        return record

    # Run H-step oracle for each failed C/R pair.
    success_matrix = np.zeros((len(c_legal), max_r_len), dtype=bool)
    future_matrix = np.full(success_matrix.shape, -1, dtype=int)

    for c_pos, c_idx in enumerate(c_legal):
        split, server = decode_agent_c_action(int(c_idx), num_servers)
        r_legal = r_legal_per_c[int(c_idx)]
        obs_r_c = build_agent_r_observation(snapshot, req, split, server)
        for j, r_idx in enumerate(r_legal):
            branch = copy.deepcopy(snapshot)
            r_action = decode_agent_r_action(int(r_idx), len(obs_r_c["mod_names"]), max_blocks)
            branch.step((split, server), r_action)
            future = _rollout_future(
                branch, requests, request_index + 1, horizon,
                agent_c, agent_r_for_features, num_servers,
                util_threshold=0.95, alpha=0.3,
            )
            future_matrix[c_pos, j] = int(future.get("blocked", 0))

    orig_c_pos = c_index_map[original_c_idx]
    orig_r_pos = r_legal_orig.index(original_r_idx) if original_r_idx in r_legal_orig else 0

    category, chosen_future, best_future, range_val = _classify_from_outcomes(
        orig_c_pos, orig_r_pos, success_matrix, future_matrix
    )
    record.category = category
    record.chosen_future_blocked = chosen_future
    record.best_future_blocked = best_future
    record.future_blocked_range = range_val
    return record


# ---------------------------------------------------------------------------
# Episode-level diagnostic
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
    max_blocks: int,
    horizon: int,
    device: str,
    seed: int,
    episode_index: int,
) -> EpisodeResult:
    result = EpisodeResult(
        seed=seed, episode=episode_index, total_requests=len(requests),
        blocked_count=0, raw_mask_empty_count=0,
    )
    env.reset(requests)

    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)

        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        selected_split, selected_server = decode_agent_c_action(int(c_idx), num_servers)

        obs_r = build_agent_r_observation(env, req, selected_split, selected_server)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)

        # v1.2 ranker R selection.
        r_idx = _select_v12_r_action(
            env, req, obs_c, obs_r, agent_r_for_features,
            ranker_model, ranker_mean, ranker_std,
            selected_split, selected_server, device,
        )

        # Snapshot before applying the C/R decision.
        snapshot = copy.deepcopy(env)

        r_action = decode_agent_r_action(int(r_idx), len(obs_r["mod_names"]), max_blocks)
        _, _, _, info = env.step((selected_split, selected_server), r_action)
        success = bool(info.get("success", False))

        if not success:
            result.blocked_count += 1
            if int(raw_c_mask.sum()) == 0 or int(raw_r_mask.sum()) == 0:
                result.raw_mask_empty_count += 1

            record = _diagnose_blocked_request(
                snapshot, req, int(c_idx), int(r_idx),
                selected_split, selected_server, obs_c, raw_c_mask,
                agent_c, agent_r_for_features,
                ranker_model, ranker_mean, ranker_std,
                num_servers, max_blocks, requests, request_index,
                horizon, device,
            )
            record.seed = seed
            record.episode = episode_index
            record.request_index = request_index
            result.blocked_records.append(record)

    return result


# ---------------------------------------------------------------------------
# Aggregation and reporting
# ---------------------------------------------------------------------------
def _aggregate_results(episode_results: List[EpisodeResult]) -> Dict[str, Any]:
    total_requests = sum(r.total_requests for r in episode_results)
    total_blocked = sum(r.blocked_count for r in episode_results)
    total_raw_empty = sum(r.raw_mask_empty_count for r in episode_results)

    all_records: List[BlockedRecord] = []
    for er in episode_results:
        all_records.extend(er.blocked_records)

    categories = ["unavoidable_block", "C_avoidable", "R_avoidable", "trajectory_avoidable", "unknown"]
    category_counts = {c: 0 for c in categories}
    reason_counts: Dict[str, int] = {}
    for rec in all_records:
        category_counts[rec.category] += 1
        reason_counts[rec.reason] = reason_counts.get(rec.reason, 0) + 1

    c_avoidable_records = [r for r in all_records if r.category == "C_avoidable"]
    r_avoidable_records = [r for r in all_records if r.category == "R_avoidable"]
    traj_records = [r for r in all_records if r.category == "trajectory_avoidable"]

    c_avoidable_successful_c_mean = float(np.mean([r.successful_c_actions for r in c_avoidable_records])) if c_avoidable_records else 0.0
    r_path_km_diffs = []
    for r in r_avoidable_records:
        # We need the v1.2 selected path km; fetch from the original observation not stored here.
        # Leave empty for now; can be enriched later if needed.
        pass

    traj_range_mean = float(np.mean([r.future_blocked_range for r in traj_records if r.future_blocked_range is not None])) if traj_records else 0.0
    traj_gain_mean = float(np.mean([
        (r.chosen_future_blocked - r.best_future_blocked)
        for r in traj_records
        if r.chosen_future_blocked is not None and r.best_future_blocked is not None
    ])) if traj_records else 0.0

    # Per-seed summary
    seed_summary: Dict[int, Dict[str, Any]] = {}
    for er in episode_results:
        seed = er.seed
        if seed not in seed_summary:
            seed_summary[seed] = {
                "total_requests": 0,
                "blocked_count": 0,
                "raw_mask_empty_count": 0,
                "category_counts": {c: 0 for c in categories},
                "reason_counts": {},
            }
        seed_summary[seed]["total_requests"] += er.total_requests
        seed_summary[seed]["blocked_count"] += er.blocked_count
        seed_summary[seed]["raw_mask_empty_count"] += er.raw_mask_empty_count
        for rec in er.blocked_records:
            seed_summary[seed]["category_counts"][rec.category] += 1
            seed_summary[seed]["reason_counts"][rec.reason] = seed_summary[seed]["reason_counts"].get(rec.reason, 0) + 1

    return {
        "total_requests": total_requests,
        "total_blocked": total_blocked,
        "blocking_rate": total_blocked / max(total_requests, 1),
        "raw_mask_empty_count": total_raw_empty,
        "raw_mask_empty_rate": total_raw_empty / max(total_requests, 1),
        "category_counts": category_counts,
        "category_rates": {c: category_counts[c] / max(total_blocked, 1) for c in categories},
        "reason_counts": reason_counts,
        "reason_rates": {k: v / max(total_blocked, 1) for k, v in reason_counts.items()},
        "c_avoidable_mean_successful_c_actions": c_avoidable_successful_c_mean,
        "r_avoidable_count": len(r_avoidable_records),
        "trajectory_avoidable_count": len(traj_records),
        "trajectory_future_blocked_range_mean": traj_range_mean,
        "trajectory_oracle_gain_mean": traj_gain_mean,
        "seed_summary": seed_summary,
        "episode_results": [er.to_dict() for er in episode_results],
    }


def _write_markdown(path: Path, summary: Dict[str, Any]) -> None:
    lines = [
        "# v1.2 Residual Blocking Floor Diagnostic",
        "",
        f"- Total requests: {summary['total_requests']}",
        f"- v1.2 blocked: {summary['total_blocked']} ({summary['blocking_rate']*100:.2f}%)",
        f"- Raw mask empty events: {summary['raw_mask_empty_count']} ({summary['raw_mask_empty_rate']*100:.2f}%)",
        "",
        "## Blocked request decomposition",
        "",
        "| Category | Count | Share of blocked |",
        "|---|---:|---:|",
    ]
    for cat in ["unavoidable_block", "C_avoidable", "R_avoidable", "trajectory_avoidable", "unknown"]:
        lines.append(
            f"| {cat} | {summary['category_counts'][cat]} | "
            f"{summary['category_rates'][cat]*100:.2f}% |"
        )

    lines += ["", "## Failure reason distribution", "", "| Reason | Count | Share of blocked |", "|---|---:|---:|"]
    for reason, count in sorted(summary["reason_counts"].items(), key=lambda x: -x[1]):
        lines.append(
            f"| {reason} | {count} | {summary['reason_rates'][reason]*100:.2f}% |"
        )

    lines += [
        "",
        "## Additional diagnostics",
        "",
        f"- Mean successful C actions when C_avoidable: {summary['c_avoidable_mean_successful_c_actions']:.2f}",
        f"- R_avoidable count: {summary['r_avoidable_count']}",
        f"- Trajectory_avoidable count: {summary['trajectory_avoidable_count']}",
        f"- Mean future-blocked range (trajectory avoidable): {summary['trajectory_future_blocked_range_mean']:.3f}",
        f"- Mean oracle gain (trajectory avoidable): {summary['trajectory_oracle_gain_mean']:.3f}",
        "",
        "## Per-seed summary",
        "",
        "| Seed | Requests | Blocked | unavoidable | C_avoidable | R_avoidable | trajectory |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for seed, s in sorted(summary["seed_summary"].items()):
        lines.append(
            f"| {seed} | {s['total_requests']} | {s['blocked_count']} | "
            f"{s['category_counts']['unavoidable_block']} | {s['category_counts']['C_avoidable']} | "
            f"{s['category_counts']['R_avoidable']} | {s['category_counts']['trajectory_avoidable']} |"
        )

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

    # Verify frozen backends are unchanged.
    c_ref = {k: v.clone() for k, v in agent_c.policy_net.state_dict().items()}
    r_ref = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}

    episode_results: List[EpisodeResult] = []
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
            er = _diagnose_episode(
                env, requests, agent_c, agent_r,
                ranker_model, ranker_mean, ranker_std,
                args.num_servers, args.max_blocks, args.horizon, args.device,
                seed, episode_index,
            )
            episode_results.append(er)
            print(
                f"[seed={seed}][episode={episode_index+1}/{args.episodes}] "
                f"blocked={er.blocked_count}/{er.total_requests} "
                f"elapsed={time.time()-t0:.1f}s",
                flush=True,
            )

    c_unchanged = all(torch.equal(agent_c.policy_net.state_dict()[k], c_ref[k]) for k in c_ref)
    r_unchanged = all(torch.equal(agent_r.policy_net.state_dict()[k], r_ref[k]) for k in r_ref)

    summary = _aggregate_results(episode_results)
    summary["elapsed_seconds"] = time.time() - start
    summary["frozen_backends_unchanged"] = bool(c_unchanged and r_unchanged)
    summary["config"] = vars(args)
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranking_checkpoint", default="sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt")
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.09)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=14.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--horizon", type=int, default=3)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output_json", default="sa_hmarl/experiments/v12_residual_blocking_floor_diagnostic.json")
    parser.add_argument("--output_md", default="sa_hmarl/experiments/v12_residual_blocking_floor_diagnostic.md")
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
        f"Blocking rate: {summary['blocking_rate']*100:.2f}% | "
        f"unavoidable={summary['category_counts']['unavoidable_block']} "
        f"C_avoidable={summary['category_counts']['C_avoidable']} "
        f"R_avoidable={summary['category_counts']['R_avoidable']} "
        f"trajectory={summary['category_counts']['trajectory_avoidable']}"
    )
