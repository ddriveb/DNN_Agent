"""Legacy post-decision value network definitions.

The final v1.2 R-side method is now named
``CounterfactualActionValueRanker`` in ``counterfactual_r_ranker.py``.  In the
paper-facing story this network is the distilled scorer for an offline H-step
counterfactual planner.  The ``PostDecisionValueNetwork`` name is kept only as a
compatibility alias for older checkpoints and scripts.
"""
from __future__ import annotations

import torch

from sa_hmarl.agents.counterfactual_r_ranker import CounterfactualActionValueRanker


class PostDecisionValueNetwork(CounterfactualActionValueRanker):
    """Backward-compatible name for the v1.2 planner-distilled R-ranker."""


def pairwise_ranking_loss(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    min_label_gap: float = 1e-5,
) -> torch.Tensor:
    """Logistic pairwise loss over candidates from the same request only."""
    losses = []
    for group_id in torch.unique(group_ids):
        indices = torch.nonzero(group_ids == group_id, as_tuple=False).squeeze(-1)
        if indices.numel() < 2:
            continue
        pred = predictions[indices]
        target = labels[indices]
        target_diff = target[:, None] - target[None, :]
        upper = torch.triu(torch.ones_like(target_diff, dtype=torch.bool), diagonal=1)
        valid = upper & (torch.abs(target_diff) > min_label_gap)
        if not torch.any(valid):
            continue
        pred_diff = pred[:, None] - pred[None, :]
        direction = torch.sign(target_diff[valid])
        losses.append(torch.nn.functional.softplus(-direction * pred_diff[valid]).mean())
    if not losses:
        return predictions.sum() * 0.0
    return torch.stack(losses).mean()
