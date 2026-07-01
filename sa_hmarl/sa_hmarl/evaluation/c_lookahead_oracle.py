"""Lookahead scoring for a single Agent-C action.

Simulates the current C action plus a fixed-length horizon of future requests
on a deep-copied environment to produce a forward-looking score.  The original
environment is never mutated.

This is **step 1** of the Lookahead-C Oracle — the per-action scorer.
A full Oracle that enumerates all C candidates and uses this scorer for
ranking will follow.
"""
from __future__ import annotations

import copy
import heapq
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.env.request import DNNRequest


# ---------------------------------------------------------------------------
# Default lookahead parameters
# ---------------------------------------------------------------------------

_DEFAULT_LOOKAHEAD = {
    "lookahead_horizon": 5,
    "current_block_penalty": 5.0,
    "future_block_penalty": 2.0,
    "no_block_penalty": 1.0,
    "delay_coef": 0.1,
    "fs_coef": 0.2,
    "waste_coef_score": 0.2,
    "server_overload_coef": 1.0,
}


def _get_lookahead_param(args: Any, name: str) -> Any:
    """Get a lookahead parameter from *args* (namespace or dict), falling
    back to ``_DEFAULT_LOOKAHEAD``."""
    if args is None:
        return _DEFAULT_LOOKAHEAD[name]
    if isinstance(args, dict):
        return args.get(name, _DEFAULT_LOOKAHEAD[name])
    return getattr(args, name, _DEFAULT_LOOKAHEAD[name])


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _select_c_deterministic(
    agent_c: Any,
    obs_c: Dict[str, Any],
) -> Optional[int]:
    """Deterministic Agent-C action selection with risk-mask support."""
    features, mask = agent_c.build_action_features(obs_c)

    # Apply risk mask from checkpoint args (if present)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        from sa_hmarl.env.c_action_risk import (
            apply_agent_c_risk_mask,
            risk_kwargs_from_args,
        )
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        if risk_kwargs:
            min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
            mask = apply_agent_c_risk_mask(
                obs_c, mask,
                num_slots_total=int(ckpt_args.get("num_slots", 24)),
                min_valid_after_mask=min_valid,
                **risk_kwargs,
            )

    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


def _select_r_deterministic(
    agent_r: Any,
    env: SMDPEnv,
    req: DNNRequest,
    split_id: int,
    server_id: int,
) -> Tuple[Optional[int], Optional[Tuple[int, int, int]]]:
    """Deterministic Agent-R action selection.

    Returns (flat_action_idx, decoded_tuple) or (None, None).
    """
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return None, None
    decoded = (
        decode_agent_r_action(
            action_idx, len(obs_r["mod_names"]), getattr(env, "max_blocks", 5),
        )
    )
    return action_idx, decoded


def _push_future_requests(
    env_try: SMDPEnv,
    future_reqs: List[DNNRequest],
) -> None:
    """Push *future_reqs* into the environment's event queue so that
    ``env_try.step()`` can process them."""
    for i, freq in enumerate(future_reqs):
        heapq.heappush(
            env_try.event_queue,
            (freq.arrival_time, 100_000 + i, freq),
        )


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def score_c_action_lookahead(
    agent_c: Any,
    agent_r: Any,
    env: SMDPEnv,
    req: DNNRequest,
    action_idx_c: int,
    future_reqs: List[DNNRequest],
    args: Any = None,
) -> Dict[str, Any]:
    """Score a single C action by simulating it + a horizon of future requests.

    The original *env* is **never** mutated — all simulation happens on a
    deep copy.

    Args:
        agent_c: A ``PPOAgentC`` (or compatible) for split/server selection.
        agent_r: A ``PPOAgentR`` (or compatible) for path/mod/block selection.
        env: The current environment state (before processing *req*).
        req: The current DNN request.
        action_idx_c: Flat index of the C candidate to evaluate.
        future_reqs: List of upcoming DNN requests to simulate.
        args: Optional namespace / dict with lookahead parameters.  Missing
            keys fall back to ``_DEFAULT_LOOKAHEAD``.  Supported keys:

            - ``lookahead_horizon`` (int, default 5)
            - ``current_block_penalty`` (float, default 5.0)
            - ``future_block_penalty`` (float, default 2.0)
            - ``no_block_penalty`` (float, default 1.0)
            - ``delay_coef`` (float, default 0.1)
            - ``fs_coef`` (float, default 0.2)
            - ``waste_coef_score`` (float, default 0.2)
            - ``server_overload_coef`` (float, default 1.0)

    Returns:
        Dict with ``score`` and all diagnostic breakdowns.  Lower score is
        better.
    """
    horizon = int(_get_lookahead_param(args, "lookahead_horizon"))
    current_block_penalty = float(_get_lookahead_param(args, "current_block_penalty"))
    future_block_penalty = float(_get_lookahead_param(args, "future_block_penalty"))
    no_block_penalty = float(_get_lookahead_param(args, "no_block_penalty"))
    delay_coef = float(_get_lookahead_param(args, "delay_coef"))
    fs_coef = float(_get_lookahead_param(args, "fs_coef"))
    waste_coef_score = float(_get_lookahead_param(args, "waste_coef_score"))
    server_overload_coef = float(_get_lookahead_param(args, "server_overload_coef"))

    num_servers = len(env.mec.servers)
    split_id, server_id = decode_agent_c_action(action_idx_c, num_servers)

    # Validate split/server range
    if (split_id < 0 or split_id >= len(req.splits) or
            server_id < 0 or server_id >= num_servers):
        return {
            "score": float(current_block_penalty + future_block_penalty * horizon),
            "action_idx_c": int(action_idx_c),
            "action_c": (int(split_id), int(server_id)),
            "current_success": False,
            "current_reason": "invalid_c_action",
            "current_delay_ms": 0.0,
            "current_fs": 0,
            "current_waste": 0.0,
            "future_block_count": 0,
            "future_no_suitable_block_count": 0,
            "future_overload_count": 0,
            "future_success_count": 0,
            "future_avg_delay_ms": 0.0,
            "future_avg_fs": 0.0,
            "future_avg_waste": 0.0,
        }

    # ------------------------------------------------------------------
    # 1. Deep-copy environment — original never touched
    # ------------------------------------------------------------------
    env_try = copy.deepcopy(env)

    # ------------------------------------------------------------------
    # 2. Execute the current C action on the copy
    # ------------------------------------------------------------------
    _, action_r_decoded = _select_r_deterministic(
        agent_r, env_try, req, split_id, server_id,
    )

    current_success: bool = False
    current_reason: str = "r_action_none"
    current_delay_ms: float = 0.0
    current_fs: int = 0
    current_waste: float = 0.0

    if action_r_decoded is None:
        current_failed = True
    else:
        _, _, _, info = env_try.step((split_id, server_id), action_r_decoded)
        current_success = info.get("success", False)
        current_reason = info.get("reason", "unknown")

        if current_success:
            current_failed = False
            current_delay_ms = float(info.get("delay_ms", 0.0))
            current_fs = int(info.get("num_slots", 0))
            current_waste = float(info.get("block_waste", 0.0))
        else:
            current_failed = True

    # ------------------------------------------------------------------
    # 3. Simulate future requests (up to horizon)
    # ------------------------------------------------------------------
    future_block_count = 0
    future_no_suitable_block_count = 0
    future_overload_count = 0
    future_delay_sum = 0.0
    future_fs_sum = 0.0
    future_waste_sum = 0.0
    future_success_count = 0

    if horizon > 0 and future_reqs:
        _push_future_requests(env_try, future_reqs)

        for _ in range(horizon):
            if not env_try.event_queue:
                break

            # Peek at the next request
            _, _, future_req = env_try.event_queue[0]

            # Agent-C selection
            obs_c = build_agent_c_observation(env_try, future_req)
            action_idx_c_future = _select_c_deterministic(agent_c, obs_c)

            if action_idx_c_future is None:
                # No valid C action → process as blocked via step()
                future_block_count += 1
                _, _, done, _ = env_try.step((0, 0), (0, 0, 0))
                if done:
                    break
                continue

            split_id_f, server_id_f = decode_agent_c_action(
                action_idx_c_future, num_servers,
            )

            # Agent-R selection
            _, action_r_future = _select_r_deterministic(
                agent_r, env_try, future_req, split_id_f, server_id_f,
            )

            if action_r_future is None:
                future_block_count += 1
                _, _, done, _ = env_try.step(
                    (split_id_f, server_id_f), (0, 0, 0),
                )
                if done:
                    break
                continue

            # Execute
            _, _, done, info_future = env_try.step(
                (split_id_f, server_id_f), action_r_future,
            )

            if info_future.get("success", False):
                future_success_count += 1
                future_delay_sum += float(info_future.get("delay_ms", 0.0))
                future_fs_sum += float(info_future.get("num_slots", 0))
                future_waste_sum += float(info_future.get("block_waste", 0.0))
            else:
                future_block_count += 1
                reason = info_future.get("reason", "unknown")
                if reason == "no_suitable_block":
                    future_no_suitable_block_count += 1
                elif reason in ("server_overload", "server_saturated"):
                    future_overload_count += 1

            if done:
                break

    # ------------------------------------------------------------------
    # 4. Compute score (lower = better)
    # ------------------------------------------------------------------
    n_future = max(future_success_count, 1)
    future_avg_delay_ms = future_delay_sum / n_future
    future_avg_fs = future_fs_sum / n_future
    future_avg_waste = future_waste_sum / n_future

    score = (
        current_block_penalty * (1.0 if current_failed else 0.0)
        + future_block_penalty * future_block_count
        + no_block_penalty * future_no_suitable_block_count
        + delay_coef * (current_delay_ms + future_avg_delay_ms) / 100.0
        + fs_coef * (current_fs + future_avg_fs)
        + waste_coef_score * (current_waste + future_avg_waste)
        + server_overload_coef * future_overload_count
    )

    return {
        "score": float(score),
        "action_idx_c": int(action_idx_c),
        "action_c": (int(split_id), int(server_id)),
        "current_success": current_success,
        "current_reason": current_reason,
        "current_delay_ms": float(current_delay_ms),
        "current_fs": int(current_fs),
        "current_waste": float(current_waste),
        "future_block_count": future_block_count,
        "future_no_suitable_block_count": future_no_suitable_block_count,
        "future_overload_count": future_overload_count,
        "future_success_count": future_success_count,
        "future_avg_delay_ms": float(future_avg_delay_ms),
        "future_avg_fs": float(future_avg_fs),
        "future_avg_waste": float(future_avg_waste),
    }


# ---------------------------------------------------------------------------
# Lookahead-C Oracle action selector
# ---------------------------------------------------------------------------

def select_lookahead_c_action(
    agent_c: Any,
    agent_r: Any,
    env: SMDPEnv,
    req: DNNRequest,
    future_reqs: List[DNNRequest],
    args: Any = None,
) -> Tuple[Optional[int], Dict[str, Any]]:
    """Select the best C action by lookahead scoring over all valid candidates.

    Enumerates every entry in ``obs_c["agent_c_mask"]`` (the **original**
    mask — no risk clipping), scores each with
    :func:`score_c_action_lookahead`, and returns the action with the lowest
    score.

    This is an oracle diagnostic: it tries every valid (split, server) and
    picks the one that minimises the lookahead score, giving an upper bound
    on what *any* C policy could achieve under the fixed BC-PPO-R backend
    for the given lookahead horizon.

    Args:
        agent_c: PPOAgentC for split/server selection in future steps.
        agent_r: PPOAgentR for path/mod/block selection.
        env: Current SMDP environment (not mutated).
        req: Current DNN request.
        future_reqs: List of upcoming DNN requests for lookahead simulation.
        args: Optional namespace / dict with lookahead parameters (see
            :func:`score_c_action_lookahead`).

    Returns:
        (best_action_idx, diagnostics) tuple.
        - ``best_action_idx`` is ``None`` when no valid C action exists.
        - ``diagnostics`` dict contains:
            - ``best_score`` (float)
            - ``mean_score`` (float)
            - ``num_candidates`` (int)
            - ``best_score_detail`` (dict) — the full return of
              ``score_c_action_lookahead`` for the winning action
            - ``top3_actions`` (list[dict]) — top-3 actions with their
              scores and decoded (split, server)
            - ``reason`` (str) — ``"no_valid_c_actions"`` when mask is empty
    """
    obs_c = build_agent_c_observation(env, req)
    mask: np.ndarray = obs_c["agent_c_mask"]
    valid = np.where(mask)[0]

    if len(valid) == 0:
        return None, {
            "best_score": float("inf"),
            "mean_score": float("inf"),
            "num_candidates": 0,
            "best_score_detail": {},
            "top3_actions": [],
            "reason": "no_valid_c_actions",
        }

    # Score every valid candidate
    scored: List[Dict[str, Any]] = []
    for action_idx in valid:
        detail = score_c_action_lookahead(
            agent_c, agent_r, env, req, int(action_idx), future_reqs, args,
        )
        scored.append(detail)

    # Sort by score ascending
    scored.sort(key=lambda d: d["score"])

    best = scored[0]
    mean_score = float(np.mean([d["score"] for d in scored]))

    # Top-3 summary
    top3 = []
    for d in scored[:3]:
        top3.append({
            "action_idx_c": d["action_idx_c"],
            "action_c": d["action_c"],
            "score": d["score"],
            "current_success": d["current_success"],
            "current_reason": d["current_reason"],
            "future_block_count": d["future_block_count"],
        })

    return int(best["action_idx_c"]), {
        "best_score": best["score"],
        "mean_score": float(mean_score),
        "num_candidates": len(scored),
        "best_score_detail": best,
        "top3_actions": top3,
    }
