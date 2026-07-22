"""Exact reference and optimized online selectors for Strict v1.3.

The optimized path only removes repeated work.  It preserves the frozen PPO-R
legal Top-30 proposal, the checkpoint-defined feature schema, normalization,
ranker scores, and stable action ordering used by the reference path.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

import numpy as np
import torch

from sa_hmarl.baselines.rmsa_baselines import (
    ff_ksp_highest_mod_action,
    ksp_ff_highest_mod_action,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import _r_feature_vector
from sa_hmarl.evaluation.r_ranker_features import build_r_ranker_feature_batch


@dataclass
class PPOTopKArtifacts:
    candidate_actions: np.ndarray
    action_features: np.ndarray
    action_mask: np.ndarray
    legal_actions: np.ndarray
    logits: np.ndarray


@dataclass
class StrictSelectionResult:
    action: Optional[int]
    valid: bool
    info: Dict[str, Any]
    timings_ms: Dict[str, float]
    topk: PPOTopKArtifacts
    ranker_features: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    normalized_features: np.ndarray = field(
        default_factory=lambda: np.empty((0, 0), dtype=np.float32)
    )
    ranker_scores: np.ndarray = field(
        default_factory=lambda: np.empty((0,), dtype=np.float32)
    )


def _elapsed_ms(start: float) -> float:
    return (time.perf_counter() - start) * 1000.0


def ppo_r_topk_with_features(agent_r, obs_r: Dict[str, Any], top_k: int = 30):
    """Build PPO-R action features once and return stable legal Top-K."""
    timings: Dict[str, float] = {}

    start = time.perf_counter()
    features, mask = agent_r.build_action_features(obs_r)
    features = np.asarray(features, dtype=np.float32)
    mask = np.asarray(mask, dtype=bool)
    legal = np.flatnonzero(mask)
    timings["ppo_feature_build_ms"] = _elapsed_ms(start)

    if legal.size == 0:
        artifacts = PPOTopKArtifacts(
            candidate_actions=np.empty((0,), dtype=np.int64),
            action_features=features,
            action_mask=mask,
            legal_actions=legal.astype(np.int64, copy=False),
            logits=np.empty((0,), dtype=np.float32),
        )
        timings["ppo_forward_ms"] = 0.0
        timings["legal_top30_select_ms"] = 0.0
        return artifacts, timings

    start = time.perf_counter()
    with torch.no_grad():
        tensor = torch.as_tensor(
            features, dtype=torch.float32, device=agent_r.device
        ).unsqueeze(0)
        logits = agent_r.policy_net(tensor).squeeze(0).cpu().numpy()
    logits = np.asarray(logits, dtype=np.float32)
    logits[~mask] = -np.inf
    timings["ppo_forward_ms"] = _elapsed_ms(start)

    start = time.perf_counter()
    count = min(int(top_k), int(legal.size))
    top_local = np.argsort(-logits[legal], kind="stable")[:count]
    candidates = legal[top_local].astype(np.int64, copy=False)
    timings["legal_top30_select_ms"] = _elapsed_ms(start)

    artifacts = PPOTopKArtifacts(
        candidate_actions=candidates,
        action_features=features,
        action_mask=mask,
        legal_actions=legal.astype(np.int64, copy=False),
        logits=logits,
    )
    return artifacts, timings


def build_strict_minimal_obs_c(
    env,
    req,
    obs_r: Dict[str, Any],
    split_id: int,
    server_id: int,
) -> Dict[str, Any]:
    """Reproduce the existing synthetic C observation without semantic changes."""
    num_servers = len(env.mec.servers)
    num_splits = len(req.splits)
    num_paths = len(obs_r["candidate_paths"])
    num_mods = len(obs_r["mod_names"])
    server_utilizations = [server.utilization for server in env.mec.servers]
    feasible_count = int(obs_r.get("feasible_count", 0))
    feasible_counts = [[0] * num_servers for _ in range(num_splits)]
    if 0 <= split_id < num_splits and 0 <= server_id < num_servers:
        feasible_counts[split_id][server_id] = feasible_count

    candidate_features = []
    for current_split_id in range(num_splits):
        for current_server_id in range(num_servers):
            split = req.splits[current_split_id]
            server = env.mec.servers[current_server_id]
            edge_ms = env._estimate_compute_ms(server, split.edge_compute_cost)
            is_target = (
                current_split_id == split_id and current_server_id == server_id
            )
            candidate_features.append({
                "server_available_compute": server.available_compute,
                "edge_compute_cost": split.edge_compute_cost,
                "edge_compute_ms": edge_ms,
                "_feasible_mask_per_path_mod": (
                    obs_r["feasible_mask_per_path_mod"]
                    if is_target else [[False] * num_mods for _ in range(num_paths)]
                ),
                "_required_fs_per_path_mod": (
                    obs_r["required_fs_per_path_mod"]
                    if is_target else [[None] * num_mods for _ in range(num_paths)]
                ),
                "_candidate_blocks_per_path_mod": (
                    obs_r["candidate_blocks_per_path_mod"]
                    if is_target
                    else [[[] for _ in range(num_mods)] for _ in range(num_paths)]
                ),
                "_mod_formats": num_mods,
            })

    return {
        "server_utilizations": server_utilizations,
        "feasible_counts": feasible_counts,
        "candidate_features": candidate_features,
    }


def _empty_info() -> Dict[str, Any]:
    return {
        "candidate_count": 0,
        "fallback": False,
        "in_candidates": False,
        "score_margin": 0.0,
        "strict_selected_rank_in_ppo_r_top30": -1,
        "strict_top30_contains_ksp_action": None,
        "strict_rank_of_ksp_action_if_contained": None,
        "strict_top30_contains_ffksp_action": None,
        "strict_rank_of_ffksp_action_if_contained": None,
        "action_coverage_diagnostics_collected": False,
    }


def _collect_coverage_diagnostics(
    obs_r: Dict[str, Any], candidates: np.ndarray, info: Dict[str, Any]
) -> Dict[str, float]:
    timings: Dict[str, float] = {}
    for key, selector, timing_name in (
        ("ksp", ksp_ff_highest_mod_action, "diagnostic_ksp_ms"),
        ("ffksp", ff_ksp_highest_mod_action, "diagnostic_ffksp_ms"),
    ):
        start = time.perf_counter()
        heuristic_action = selector(obs_r)
        timings[timing_name] = _elapsed_ms(start)
        contains_key = f"strict_top30_contains_{key}_action"
        rank_key = f"strict_rank_of_{key}_action_if_contained"
        info[contains_key] = False
        info[rank_key] = -1
        if heuristic_action is not None:
            matches = np.flatnonzero(candidates == int(heuristic_action))
            if matches.size:
                info[contains_key] = True
                info[rank_key] = int(matches[0])
    info["action_coverage_diagnostics_collected"] = True
    return timings


def select_strict_v13(
    ranker: Dict[str, Any],
    agent_r,
    env,
    req,
    obs_r: Dict[str, Any],
    split_id: int,
    server_id: int,
    *,
    optimized: bool,
    collect_action_coverage_diagnostics: bool = False,
) -> StrictSelectionResult:
    """Select one Strict v1.3 action using reference or exact optimized code."""
    selector_start = time.perf_counter()
    info = _empty_info()
    timings: Dict[str, float] = {
        "diagnostic_ksp_ms": 0.0,
        "diagnostic_ffksp_ms": 0.0,
        "ranker_base_feature_rebuild_ms": 0.0,
    }

    topk, topk_timings = ppo_r_topk_with_features(agent_r, obs_r, 30)
    timings.update(topk_timings)
    candidates = topk.candidate_actions
    info["candidate_count"] = int(candidates.size)
    info["fallback"] = candidates.size == 0
    if candidates.size == 0:
        timings.update({
            "obs_c_minimal_build_ms": 0.0,
            "ranker_feature_build_ms": 0.0,
            "ranker_normalize_ms": 0.0,
            "ranker_forward_ms": 0.0,
            "selector_total_ms": _elapsed_ms(selector_start),
        })
        return StrictSelectionResult(None, False, info, timings, topk)

    if collect_action_coverage_diagnostics:
        timings.update(_collect_coverage_diagnostics(obs_r, candidates, info))

    start = time.perf_counter()
    obs_c_minimal = build_strict_minimal_obs_c(
        env, req, obs_r, split_id, server_id
    )
    timings["obs_c_minimal_build_ms"] = _elapsed_ms(start)

    if optimized:
        all_r_features = topk.action_features
    else:
        start = time.perf_counter()
        all_r_features, rebuilt_mask = agent_r.build_action_features(obs_r)
        timings["ranker_base_feature_rebuild_ms"] = _elapsed_ms(start)
        if not np.array_equal(np.asarray(rebuilt_mask, dtype=bool), topk.action_mask):
            raise RuntimeError("PPO-R action mask changed during one Strict decision")

    start = time.perf_counter()
    if optimized:
        raw_features = build_r_ranker_feature_batch(
            env,
            req,
            obs_c_minimal,
            obs_r,
            all_r_features,
            candidates,
            split_id,
            server_id,
            feature_names=ranker["feature_names"],
        )
    else:
        raw_features = np.stack([
            _r_feature_vector(
                env,
                req,
                obs_c_minimal,
                obs_r,
                all_r_features,
                int(action),
                split_id,
                server_id,
                feature_names=ranker["feature_names"],
            )
            for action in candidates
        ]).astype(np.float32)
    raw_features = np.asarray(raw_features, dtype=np.float32)
    timings["ranker_feature_build_ms"] = _elapsed_ms(start)

    start = time.perf_counter()
    normalized = (
        raw_features - ranker["feature_mean"]
    ) / ranker["feature_std"]
    normalized = np.asarray(normalized, dtype=np.float32)
    normalized[~np.isfinite(normalized)] = 0.0
    timings["ranker_normalize_ms"] = _elapsed_ms(start)

    start = time.perf_counter()
    with torch.no_grad():
        scores = (
            ranker["model"](
                torch.as_tensor(
                    normalized, dtype=torch.float32, device=ranker["device"]
                )
            )
            .cpu()
            .numpy()
        )
    scores = np.asarray(scores, dtype=np.float32)
    timings["ranker_forward_ms"] = _elapsed_ms(start)

    best_local = int(np.argmax(scores))
    selected = int(candidates[best_local])
    info["in_candidates"] = True
    info["strict_selected_rank_in_ppo_r_top30"] = best_local
    sorted_scores = np.sort(scores)[::-1]
    info["score_margin"] = (
        float(sorted_scores[0] - sorted_scores[1])
        if sorted_scores.size >= 2 else 0.0
    )
    timings["selector_total_ms"] = _elapsed_ms(selector_start)
    return StrictSelectionResult(
        selected,
        True,
        info,
        timings,
        topk,
        ranker_features=raw_features,
        normalized_features=normalized,
        ranker_scores=scores,
    )


__all__ = [
    "PPOTopKArtifacts",
    "StrictSelectionResult",
    "build_strict_minimal_obs_c",
    "ppo_r_topk_with_features",
    "select_strict_v13",
]
