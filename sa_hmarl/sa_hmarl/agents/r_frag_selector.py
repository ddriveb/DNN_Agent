"""Inference-time fragmentation-aware Agent-R action selection."""
from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import torch

from sa_hmarl.agents.ppo_agents import PPOAgentR
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.env.r_frag_aware import rerank_scores


def select_frag_aware_action(
    agent: Any,
    obs_r: Dict[str, Any],
    waste_coef: float = 0.2,
    frag_coef: float = 0.4,
    large_block_coef: float = 0.2,
    lfb_drop_coef: float = 0.2,
    exact_fit_bonus: float = 0.05,
) -> Optional[int]:
    """Select an Agent-R action after preventive-fragmentation reranking.

    The base policy is left unchanged.  We compute its per-action Q/logit
    scores, subtract a local fragmentation-risk penalty, and then choose the
    highest adjusted valid action.
    """
    features, mask = agent.build_action_features(obs_r)
    if len(mask) == 0 or not np.any(mask) or features.size == 0:
        return None

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=agent.device).unsqueeze(0)
        if isinstance(agent, PPOAgentR):
            scores = agent.policy_net(x).squeeze(0).cpu().numpy()
        elif isinstance(agent, AgentR):
            scores = agent.q_net(x).squeeze(0).cpu().numpy()
        else:
            raise TypeError(
                "select_frag_aware_action currently supports PPOAgentR and AgentR"
            )

    adjusted = rerank_scores(
        scores,
        mask,
        obs_r,
        waste_coef=waste_coef,
        frag_coef=frag_coef,
        large_block_coef=large_block_coef,
        lfb_drop_coef=lfb_drop_coef,
        exact_fit_bonus=exact_fit_bonus,
    )
    if not np.any(np.isfinite(adjusted)):
        return None
    return int(np.argmax(adjusted))
