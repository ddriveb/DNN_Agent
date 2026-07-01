"""planner-distilled v1.2 R-ranker 的候选动作特征接口。

这个文件是一个轻量兼容层：在线策略通过这里构造 ranker 输入特征，
避免直接依赖离线数据生成脚本。

这些特征是摊销 planner 使用的候选级状态-动作表示：

    phi(s, a_R) -> distilled planner score
"""
from __future__ import annotations

from typing import Any, Dict, Iterable

import numpy as np

from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
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
) -> np.ndarray:
    """为单个 R 候选动作构造 v1.2 的标准 planner-distillation 特征。"""
    return _r_feature_vector(
        env, req, obs_c, obs_r, r_features, int(r_action_idx), split_id, server_id
    )


def build_r_ranker_feature_batch(
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    action_indices: Iterable[int],
    split_id: int,
    server_id: int,
) -> np.ndarray:
    """为同一个决策状态下的一组合法 R 候选动作构造特征矩阵。"""
    return np.stack([
        build_r_ranker_feature(
            env, req, obs_c, obs_r, r_features, int(action_idx), split_id, server_id
        )
        for action_idx in action_indices
    ]).astype(np.float32)
