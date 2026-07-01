"""final v1.2 R-ranker 的在线策略包装器。"""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from sa_hmarl.evaluation.r_ranker_features import build_r_ranker_feature_batch


class CounterfactualRRankerPolicy:
    """用蒸馏后的有限时域 planner 分数选择 RMSA 动作。

    这个类负责完整的在线 R 端决策流程：

        枚举合法 R 动作 -> 构造候选特征 -> 归一化
        -> 调用 ranker 网络 -> 选择分数最高动作

    被包装的 model 只会打分；它本身不访问环境、不生成 mask，
    在线推理时也不会 rollout 未来请求。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        device: str = "cpu",
        empty_action: int = 0,
    ):
        """保存 ranker 网络、特征归一化统计量和空动作兼容设置。"""
        self.model = model
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.device = device
        self.empty_action = int(empty_action)
        self.model.eval()

    def score_legal_actions(
        self,
        env,
        req,
        obs_c: Dict[str, Any],
        obs_r: Dict[str, Any],
        agent_r,
        split_id: int,
        server_id: int,
    ) -> tuple[list[int], np.ndarray]:
        """枚举当前合法 R 动作，并返回每个动作的 v1.2 打分。"""
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).astype(int).tolist()
        if not legal:
            return [], np.empty((0,), dtype=np.float32)
        if len(legal) == 1:
            return legal, np.asarray([0.0], dtype=np.float32)

        features = build_r_ranker_feature_batch(
            env, req, obs_c, obs_r, r_features, legal, split_id, server_id
        )
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            scores = self.model(
                torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            ).detach().cpu().numpy()
        return legal, np.asarray(scores, dtype=np.float32).reshape(-1)

    def select_action(
        self,
        env,
        req,
        obs_c: Dict[str, Any],
        obs_r: Dict[str, Any],
        agent_r,
        split_id: int,
        server_id: int,
    ) -> Optional[int]:
        """返回最终选择的展平 R 动作 id。

        历史 v1.2 在没有合法 R 动作时返回 ``0``；这里保留该行为，
        方便和既有实验指标保持一致。
        """
        legal, scores = self.score_legal_actions(
            env, req, obs_c, obs_r, agent_r, split_id, server_id
        )
        if not legal:
            return self.empty_action
        if len(legal) == 1:
            return int(legal[0])
        return int(legal[int(np.argmax(scores))])


# Paper-facing alias: online policy for planner-distilled amortized MPC.
PlannerDistilledRPolicy = CounterfactualRRankerPolicy
