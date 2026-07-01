"""规划器蒸馏式 R 端动作价值排序网络。

这个文件定义 final v1.2 的网络本体。它只负责对一个候选
RMSA 动作打分，不负责枚举动作、生成 mask 或执行环境交互。

方法视角：
    离线 H-step 反事实 rollout 充当有限时域 planner；
    该网络把 planner 的候选动作排序能力蒸馏成一次便宜的前向打分。
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn


class CounterfactualActionValueRanker(nn.Module):
    """给每个可行 RMSA 候选动作输出一个 planner-distilled 分数。

    这是候选级打分器：

        score = f_theta(phi(s, a))

    它不是 TD value 网络，也不是模仿 PPO-R 的动作。训练标签来自离线
    H-step 反事实 rollout；在线时只对当前合法候选动作做前向打分，
    然后选择分数最高者。

    内部属性名保留为 ``net``，用于兼容早期 PostDecisionValueNetwork
    生成的 checkpoint。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        layers = []
        previous = input_dim
        for hidden in hidden_dims:
            layers.extend([nn.Linear(previous, hidden), nn.SiLU()])
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            previous = hidden
        layers.append(nn.Linear(previous, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, candidate_features: torch.Tensor) -> torch.Tensor:
        """输入候选动作特征，返回每个候选的标量分数。"""
        return self.net(candidate_features).squeeze(-1)


# Shorter aliases for scripts and paper-facing code.
CounterfactualRRanker = CounterfactualActionValueRanker
PlannerDistilledRRanker = CounterfactualActionValueRanker
AmortizedCounterfactualMPC = CounterfactualActionValueRanker
