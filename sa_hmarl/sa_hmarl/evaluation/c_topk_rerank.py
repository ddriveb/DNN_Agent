"""Agent-C Top-K rerank with C-candidate pressure scoring.

Takes the top-K Agent-C candidates by policy score, then re-ranks them using
the detailed spectrum-pressure metrics from :mod:`c_candidate_pressure`.
This is a *post-hoc* reranker — it does not modify training.

Usage::

    from sa_hmarl.evaluation.c_topk_rerank import select_c_topk_rerank

    action = select_c_topk_rerank(agent_c, obs_c, env, req, top_k=5)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.evaluation.c_candidate_pressure import compute_c_candidate_pressure


# ---------------------------------------------------------------------------
# Default weights for the rerank score (lower = better)
# ---------------------------------------------------------------------------

DEFAULT_WEIGHTS = {
    "w_block": 10.0,    # penalty for zero valid R actions
    "w_ratio": 1.0,     # penalty for high fs/lfb ratio
    "w_delay": 0.02,    # penalty per ms of deadline violation
    "w_frag": 0.5,      # penalty for fragmentation pressure
    "w_server": 0.5,    # penalty for high server utilisation
    "w_feas": 0.2,      # bonus per log(1 + n_valid) — encourage more options
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_action_scores(
    agent_c: Any,
    obs_c: Dict[str, Any],
) -> Tuple[np.ndarray, np.ndarray]:
    """Return (scores, mask) for all C actions.

    scores[i] is the raw PPO policy logit; higher = better.
    Invalid actions are set to -inf.
    """
    from sa_hmarl.agents.ppo_agents import PPOAgentC

    features, mask = agent_c.build_action_features(obs_c)

    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=agent_c.device).unsqueeze(0)
        if isinstance(agent_c, PPOAgentC):
            raw = agent_c.policy_net(x).squeeze(0).cpu().numpy()
        else:
            raise TypeError(
                f"Unsupported agent type: {type(agent_c)}. "
                f"Expected PPOAgentC."
            )

    scores = raw.copy()
    scores[~mask] = -float("inf")
    return scores, mask


def _compute_rerank_score(
    pressure: Dict[str, Any],
    weights: Dict[str, float],
) -> float:
    """Compute rerank score from a pressure dict (lower = better).

    score = w_block * I[n_valid == 0]
          + w_ratio * fs_lfb_ratio
          + w_delay * max(-delay_slack_ms, 0)
          + w_frag  * frag_pressure
          + w_server * server_utilization
          - w_feas  * log(1 + n_valid_r_actions)
    """
    w = weights
    n_valid = pressure["n_valid_r_actions"]

    score = 0.0
    # Block penalty: heavy cost when no R action is feasible
    score += w["w_block"] * (1.0 if n_valid == 0 else 0.0)
    # Tightness: how close required FS is to largest free block
    # (inf when n_valid==0 → candidate can never win)
    score += w["w_ratio"] * pressure["fs_lfb_ratio"]
    # Deadline violation: positive when delay exceeds deadline
    # (-inf when n_valid==0 → max(+inf,0)=inf, same effect)
    score += w["w_delay"] * max(-pressure["delay_slack_ms"], 0.0)
    # Fragmentation pressure (inf when n_valid==0)
    score += w["w_frag"] * pressure["frag_pressure"]
    # Server load
    score += w["w_server"] * pressure["server_utilization"]
    # Option value: more valid R actions → lower score
    score -= w["w_feas"] * math.log(1.0 + n_valid)

    return float(score)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def select_c_topk_rerank(
    agent_c: Any,
    obs_c: Dict[str, Any],
    env: Any,
    req: Any,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
) -> Optional[int]:
    """Select an Agent-C action by reranking top-K policy candidates with
    spectrum-pressure scores.

    Procedure
    ---------
    1. Score all valid C actions via the agent's policy network.
    2. Keep the *top_k* by policy score.
    3. For each, call :func:`compute_c_candidate_pressure` to get detailed
       R-level spectrum metrics.
    4. Compute a rerank score (lower = better) from the pressure dict.
    5. Return the action with the lowest rerank score.

    Args:
        agent_c: A ``PPOAgentC`` instance.
        obs_c: Pre-built Agent-C observation dict.
        env: The current SMDP environment (read-only).
        req: The current DNN request.
        top_k: Number of top policy candidates to consider for reranking.
            When ``top_k=1`` this is equivalent to the original Agent-C.
        weights: Optional dict overriding default rerank weights.  Keys:
            ``w_block``, ``w_ratio``, ``w_delay``, ``w_frag``,
            ``w_server``, ``w_feas``.

    Returns:
        Flat action index (int) or ``None`` if no valid C action exists.
    """
    # --- 1. Get policy scores for all valid C actions -----------------------
    scores, mask = _get_action_scores(agent_c, obs_c)

    valid_indices = np.where(mask)[0]
    if len(valid_indices) == 0:
        return None

    # --- 2. Top-K by policy score -------------------------------------------
    k = min(top_k, len(valid_indices))
    if k == 1:
        # Shortcut: top_k=1 → original Agent-C (skip pressure computation)
        return int(valid_indices[np.argmax(scores[valid_indices])])

    # argpartition for efficiency: top k by descending score
    topk_local = np.argpartition(-scores[valid_indices], k - 1)[:k]
    topk_global = valid_indices[topk_local]

    # --- 3. Compute pressure for each top-K candidate -----------------------
    w = DEFAULT_WEIGHTS.copy()
    if weights is not None:
        w.update(weights)

    num_servers = len(env.mec.servers)

    best_action: Optional[int] = None
    best_rerank_score: float = float("inf")
    # Fallback: highest policy score among evaluated candidates
    best_policy_score: float = -float("inf")
    fallback_action: Optional[int] = None

    for flat_idx in topk_global:
        flat_idx = int(flat_idx)
        split_id = flat_idx // num_servers
        server_id = flat_idx % num_servers

        pressure = compute_c_candidate_pressure(env, req, split_id, server_id)
        rerank_score = _compute_rerank_score(pressure, w)

        if rerank_score < best_rerank_score:
            best_rerank_score = rerank_score
            best_action = flat_idx

        # Track fallback in case best_action somehow becomes invalid
        policy_score = float(scores[flat_idx])
        if policy_score > best_policy_score:
            best_policy_score = policy_score
            fallback_action = flat_idx

    # --- 4. Return result ---------------------------------------------------
    # Guarantee: always return a valid action (best by rerank, or fallback)
    return best_action if best_action is not None else fallback_action


# ---------------------------------------------------------------------------
# Convenience: select with default agent-C and then rerank
# ---------------------------------------------------------------------------

def select_c_topk_rerank_from_raw(
    agent_c: Any,
    env: Any,
    req: Any,
    top_k: int = 5,
    weights: Optional[Dict[str, float]] = None,
    c_policy: str = "agent",
) -> Optional[int]:
    """Full pipeline: build obs_c, select via top-K rerank.

    This is a drop-in replacement for the standard C-action selection in
    evaluation loops.

    Args:
        agent_c: PPOAgentC instance.
        env: Current SMDP environment.
        req: Current DNN request.
        top_k: Number of top-K candidates.
        weights: Optional rerank weight overrides.
        c_policy: ``"agent"`` uses the provided agent_c; ``"greedy"`` uses
            a simple minimum-edge-compute heuristic as the base policy.

    Returns:
        Flat C action index or None.
    """
    from sa_hmarl.env.observation_builder import build_agent_c_observation

    obs_c = build_agent_c_observation(env, req)

    if c_policy == "greedy":
        # Use a synthetic "greedy" base policy: rank by -edge_compute_ms
        mask = obs_c["agent_c_mask"]
        valid = np.where(mask)[0]
        if len(valid) == 0:
            return None
        # Build synthetic scores: -edge_compute_ms (higher = better)
        synthetic_scores = np.full(len(mask), -float("inf"))
        for idx in valid:
            idx = int(idx)
            ms = obs_c["candidate_features"][idx].get("edge_compute_ms", 1e9)
            if ms == float("inf"):
                ms = 1e9
            synthetic_scores[idx] = -ms

        # Patch _get_action_scores to return our synthetic scores
        # (we inline the top-K logic here for the greedy case)
        scores = synthetic_scores
        valid_indices = np.where(mask)[0]
        k = min(top_k, len(valid_indices))
        if k == 1:
            return int(valid_indices[np.argmax(scores[valid_indices])])

        topk_local = np.argpartition(-scores[valid_indices], k - 1)[:k]
        topk_global = valid_indices[topk_local]

        w = DEFAULT_WEIGHTS.copy()
        if weights is not None:
            w.update(weights)

        num_servers = len(env.mec.servers)
        best_action: Optional[int] = None
        best_rerank: float = float("inf")

        for flat_idx in topk_global:
            flat_idx = int(flat_idx)
            split_id = flat_idx // num_servers
            server_id = flat_idx % num_servers
            pressure = compute_c_candidate_pressure(env, req, split_id, server_id)
            rs = _compute_rerank_score(pressure, w)
            if rs < best_rerank:
                best_rerank = rs
                best_action = flat_idx

        return best_action

    # Standard agent-based path
    obs_c = build_agent_c_observation(env, req)

    # Apply risk mask if the agent has one (from checkpoint args)
    if hasattr(agent_c, "checkpoint_args"):
        ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
        if ckpt_args:
            from sa_hmarl.env.c_action_risk import (
                apply_agent_c_risk_mask,
                risk_kwargs_from_args,
            )
            risk_kwargs = risk_kwargs_from_args(ckpt_args)
            if risk_kwargs:
                features_c, mask_c = agent_c.build_action_features(obs_c)
                min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
                mask_c = apply_agent_c_risk_mask(
                    obs_c, mask_c,
                    num_slots_total=int(ckpt_args.get("num_slots", 24)),
                    min_valid_after_mask=min_valid,
                    **risk_kwargs,
                )
                obs_c = dict(obs_c)  # shallow copy
                obs_c["agent_c_mask"] = mask_c

    return select_c_topk_rerank(agent_c, obs_c, env, req, top_k=top_k, weights=weights)
