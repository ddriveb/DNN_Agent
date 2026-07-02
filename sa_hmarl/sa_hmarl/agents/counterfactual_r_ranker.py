"""规划器蒸馏式 R 端动作价值排序网络。

这个文件定义 final v1.2 的网络本体。它只负责对一个候选
RMSA 动作打分，不负责枚举动作、生成 mask 或执行环境交互。

方法视角：
    离线 H-step 反事实 rollout 充当有限时域 planner；
    该网络把 planner 的候选动作排序能力蒸馏成一次便宜的前向打分。
"""
from __future__ import annotations

from typing import Iterable, Optional

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

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """输入候选动作特征，返回每个候选的标量分数。"""
        return self.net(candidate_features).squeeze(-1)


class DeepSetCounterfactualRRanker(nn.Module):
    """置换不变的 AC-MPC 列表评估器。

    该网络仍然输出每个候选 RMSA 动作的 planner-distilled 分数，但不再
    独立评估每个候选。它先用 ``phi`` 编码每个候选，再对当前合法候选集
    做 mean/max 全局聚合，最后让每个候选结合全局竞争上下文经 ``rho``
    输出分数：

        score_i = rho(phi(s, a_i), Pool_j phi(s, a_j))

    这样候选顺序不会影响结果，同时每个动作的分数会感知同一状态下其他
    合法动作的整体资源竞争格局。
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.0,
    ):
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            hidden_dims = (128, 64)

        phi_layers = []
        previous = input_dim
        for hidden in hidden_dims:
            phi_layers.extend([nn.Linear(previous, hidden), nn.SiLU()])
            if dropout > 0:
                phi_layers.append(nn.Dropout(dropout))
            previous = hidden
        self.phi = nn.Sequential(*phi_layers)

        embed_dim = hidden_dims[-1]
        rho_hidden = max(embed_dim, hidden_dims[0])
        self.local_head = nn.Linear(embed_dim, 1)
        rho_layers = [
            nn.Linear(embed_dim * 3, rho_hidden),
            nn.SiLU(),
        ]
        if dropout > 0:
            rho_layers.append(nn.Dropout(dropout))
        rho_layers.append(nn.Linear(rho_hidden, 1))
        self.rho = nn.Sequential(*rho_layers)
        # Start from an MLP-like local scorer and let training earn the right
        # to use set context. This protects against train/inference candidate
        # set mismatch when offline groups are top-K subsets of all legal actions.
        self.context_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """返回列表中每个候选的分数。

        Args:
            candidate_features: ``[N, F]`` 或 ``[B, N, F]``。
            candidate_mask: 可选 bool mask，形状 ``[N]`` 或 ``[B, N]``。
                训练时用于排除 padded/invalid 候选的全局聚合影响。
        """
        squeeze_batch = False
        if candidate_features.dim() == 2:
            candidate_features = candidate_features.unsqueeze(0)
            squeeze_batch = True
            if candidate_mask is not None and candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.unsqueeze(0)

        local = self.phi(candidate_features)
        batch_size, num_candidates, embed_dim = local.shape

        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch_size, num_candidates, dtype=torch.bool, device=local.device
            )
        else:
            candidate_mask = candidate_mask.to(device=local.device, dtype=torch.bool)

        mask_f = candidate_mask.unsqueeze(-1).float()
        valid_count = mask_f.sum(dim=1)
        denom = valid_count.clamp_min(1.0)
        mean_context = (local * mask_f).sum(dim=1) / denom

        neg_inf = torch.finfo(local.dtype).min
        max_context = local.masked_fill(~candidate_mask.unsqueeze(-1), neg_inf).max(dim=1).values
        has_valid = valid_count > 0.0
        max_context = torch.where(
            has_valid,
            max_context,
            torch.zeros_like(max_context),
        )

        global_context = torch.cat([mean_context, max_context], dim=-1)
        expanded_context = global_context.unsqueeze(1).expand(-1, num_candidates, -1)
        combined = torch.cat([local, expanded_context], dim=-1)
        local_scores = self.local_head(local).squeeze(-1)
        context_delta = self.rho(combined).squeeze(-1)
        scores = local_scores + self.context_scale * context_delta
        return scores.squeeze(0) if squeeze_batch else scores


class SetTransformerCounterfactualRRanker(nn.Module):
    """候选列表自注意力版 AC-MPC 评估器。

    这是第一档低成本 Set Transformer：输入仍是当前 v1.2 的 25 维候选
    摘要特征，不额外加入链路/slot overlap 结构特征。网络先对每个候选
    做局部编码，再通过 masked self-attention 让同一状态下的候选相互
    交换信息，最后输出每个候选的 planner-distilled 分数。

    为了避免注意力分支在候选分布不完全对齐时破坏已有 MLP 能力，输出
    采用 residual 形式：

        score_i = local_i + context_scale * attention_delta_i
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Iterable[int] = (128, 64),
        dropout: float = 0.0,
        num_heads: int = 4,
        num_layers: int = 1,
    ):
        super().__init__()
        hidden_dims = tuple(hidden_dims)
        if not hidden_dims:
            hidden_dims = (128, 64)

        embed_dim = hidden_dims[-1]
        if embed_dim % num_heads != 0:
            valid_heads = [h for h in (8, 4, 2, 1) if embed_dim % h == 0]
            num_heads = valid_heads[0]

        encoder_layers = []
        previous = input_dim
        for hidden in hidden_dims:
            encoder_layers.extend([nn.Linear(previous, hidden), nn.SiLU()])
            if dropout > 0:
                encoder_layers.append(nn.Dropout(dropout))
            previous = hidden
        self.encoder = nn.Sequential(*encoder_layers)

        self.attn_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=embed_dim,
                num_heads=num_heads,
                dropout=dropout,
                batch_first=True,
            )
            for _ in range(max(1, num_layers))
        ])
        self.norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(max(1, num_layers))
        ])
        ff_hidden = max(embed_dim, hidden_dims[0])
        self.ff_layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, ff_hidden),
                nn.SiLU(),
                nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
                nn.Linear(ff_hidden, embed_dim),
            )
            for _ in range(max(1, num_layers))
        ])
        self.ff_norm_layers = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(max(1, num_layers))
        ])

        self.local_head = nn.Linear(embed_dim, 1)
        self.context_head = nn.Sequential(
            nn.Linear(embed_dim, ff_hidden),
            nn.SiLU(),
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(ff_hidden, 1),
        )
        self.context_scale = nn.Parameter(torch.tensor(0.0))

    def forward(
        self,
        candidate_features: torch.Tensor,
        candidate_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """输入 ``[N,F]`` 或 ``[B,N,F]``，输出候选分数。"""
        squeeze_batch = False
        if candidate_features.dim() == 2:
            candidate_features = candidate_features.unsqueeze(0)
            squeeze_batch = True
            if candidate_mask is not None and candidate_mask.dim() == 1:
                candidate_mask = candidate_mask.unsqueeze(0)

        x = self.encoder(candidate_features)
        batch_size, num_candidates, _ = x.shape
        if candidate_mask is None:
            candidate_mask = torch.ones(
                batch_size, num_candidates, dtype=torch.bool, device=x.device
            )
        else:
            candidate_mask = candidate_mask.to(device=x.device, dtype=torch.bool)

        key_padding_mask = ~candidate_mask
        all_padded = key_padding_mask.all(dim=1)
        if all_padded.any():
            # MultiheadAttention cannot handle a row where every key is masked.
            key_padding_mask = key_padding_mask.clone()
            key_padding_mask[all_padded, 0] = False

        local_scores = self.local_head(x).squeeze(-1)
        h = x
        for attn, norm, ff, ff_norm in zip(
            self.attn_layers, self.norm_layers, self.ff_layers, self.ff_norm_layers
        ):
            attn_out, _ = attn(h, h, h, key_padding_mask=key_padding_mask, need_weights=False)
            h = norm(h + attn_out)
            h = ff_norm(h + ff(h))

        context_delta = self.context_head(h).squeeze(-1)
        scores = local_scores + self.context_scale * context_delta
        scores = scores.masked_fill(~candidate_mask, 0.0)
        return scores.squeeze(0) if squeeze_batch else scores


def build_counterfactual_r_ranker(
    model_type: str,
    input_dim: int,
    hidden_dims: Iterable[int] = (128, 64),
    dropout: float = 0.0,
) -> nn.Module:
    """Factory for AC-MPC ranker variants."""
    model_type = (model_type or "mlp").lower()
    if model_type in ("mlp", "candidate_mlp", "candidate"):
        return CounterfactualActionValueRanker(input_dim, hidden_dims, dropout)
    if model_type in ("deepset", "deep_set", "permutation_invariant"):
        return DeepSetCounterfactualRRanker(input_dim, hidden_dims, dropout)
    if model_type in ("set_transformer", "settransformer", "attention", "list_attention"):
        return SetTransformerCounterfactualRRanker(input_dim, hidden_dims, dropout)
    raise ValueError(f"Unknown counterfactual R ranker model_type: {model_type}")


# Shorter aliases for scripts and paper-facing code.
CounterfactualRRanker = CounterfactualActionValueRanker
PlannerDistilledRRanker = CounterfactualActionValueRanker
AmortizedCounterfactualMPC = CounterfactualActionValueRanker
PermutationInvariantListwiseEvaluator = DeepSetCounterfactualRRanker
AttentionListwiseEvaluator = SetTransformerCounterfactualRRanker
