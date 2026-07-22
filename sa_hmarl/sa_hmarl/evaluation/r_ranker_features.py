"""planner-distilled v1.3 R-ranker 的候选动作特征接口。

这个文件是一个轻量兼容层：在线策略通过这里构造 ranker 输入特征，
避免直接依赖离线数据生成脚本。

这些特征是摊销 planner 使用的候选级状态-动作表示：

    phi(s, a_R) -> distilled planner score
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _compute_spectrum_field,
    _phi_spec,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
    structured_feature_names,
)
from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    build_poststate_v1_feature_batch,
)


def build_r_ranker_feature(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    r_action_idx: int,
    split_id: int,
    server_id: int,
    feature_names: Iterable[str] = None,
) -> np.ndarray:
    """为单个 R 候选动作构造 v1.3 的标准 planner-distillation 特征。"""
    return _r_feature_vector(
        env, req, obs_c, obs_r, r_features, int(r_action_idx), split_id, server_id,
        feature_names=feature_names,
    )


def _build_default_feature_batch(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    action_indices: np.ndarray,
    split_id: int,
    server_id: int,
) -> np.ndarray:
    """Vectorized builder for the default FEATURE_NAMES schema.

    Computes per-decision context/field features once and concatenates them with
    per-action base features and normalized action indices.
    """
    action_indices = np.asarray(action_indices, dtype=int)
    num_actions = action_indices.shape[0]

    num_paths = max(len(obs_r["candidate_paths"]), 1)
    num_mods = max(len(obs_r["mod_names"]), 1)
    max_blocks = max(int(env.max_blocks), 1)

    # Per-action base features.
    base = np.asarray(r_features[action_indices], dtype=np.float32)

    # Per-decision context features.
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

    # Per-decision field features.
    field = np.asarray([
        k_c / num_c,
        k_r / max_r_total,
        _phi_spec(k_c, k_r, 0.3) / 8.0,
        int(raw_r_mask.sum()) / max(len(raw_r_mask), 1),
    ], dtype=np.float32)

    # Per-action path/mod/block indices (vectorized decode).
    path_idx = action_indices // (num_mods * max_blocks)
    rem = action_indices % (num_mods * max_blocks)
    mod_idx = rem // max_blocks
    block_idx = rem % max_blocks

    action = np.stack([
        path_idx / max(num_paths - 1, 1),
        mod_idx / max(num_mods - 1, 1),
        block_idx / max(max_blocks - 1, 1),
    ], axis=1).astype(np.float32)

    # Concatenate: base (per-action) + context (broadcast) + field (broadcast) + action (per-action).
    context_broadcast = np.broadcast_to(context, (num_actions, context.shape[0]))
    field_broadcast = np.broadcast_to(field, (num_actions, field.shape[0]))
    return np.concatenate([base, context_broadcast, field_broadcast, action], axis=1).astype(np.float32)


def build_r_ranker_feature_batch(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    action_indices: Iterable[int],
    split_id: int,
    server_id: int,
    feature_names: Iterable[str] = None,
) -> np.ndarray:
    """为同一个决策状态下的一组合法 R 候选动作构造特征矩阵。"""
    if feature_names is None:
        feature_names = FEATURE_NAMES
    feature_names = list(feature_names)

    # Fast vectorized path only for the default schema.
    if tuple(feature_names) == tuple(FEATURE_NAMES):
        return _build_default_feature_batch(
            env, req, obs_c, obs_r, r_features, action_indices, split_id, server_id
        )
    if tuple(feature_names) == tuple(POSTSTATE_V1_FEATURE_NAMES):
        return build_poststate_v1_feature_batch(
            env, req, obs_c, obs_r, r_features, action_indices, split_id, server_id,
            feature_names=feature_names,
        )

    # Fallback for structured/extended checkpoints: preserve exact per-action semantics.
    return np.stack([
        build_r_ranker_feature(
            env, req, obs_c, obs_r, r_features, int(action_idx), split_id, server_id,
            feature_names=feature_names,
        )
        for action_idx in action_indices
    ]).astype(np.float32)


__all__ = [
    "FEATURE_NAMES",
    "structured_feature_names",
    "build_r_ranker_feature",
    "build_r_ranker_feature_batch",
]
