"""Smoke tests for Agent-C Top-K rerank with pressure scoring.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/tests/test_c_topk_rerank.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.agents.ppo_agents import PPOAgentC
from sa_hmarl.evaluation.c_topk_rerank import (
    select_c_topk_rerank,
    select_c_topk_rerank_from_raw,
    DEFAULT_WEIGHTS,
    _get_action_scores,
    _compute_rerank_score,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(num_slots: int = 32):
    net = OpticalNetwork("net1", num_slots=num_slots)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=2, seed=42,
        server_nodes=[0, 3], capacities=[50.0, 50.0],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    return SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)


def _make_request(env: SMDPEnv, src: int = 1,
                  deadline_ms: float = 120.0,
                  num_splits: int = 3) -> DNNRequest:
    splits = [
        SplitProfile(
            split_id=i,
            intermediate_size_mb=10.0 + i * 5.0,
            edge_compute_cost=2.0 + i * 1.0,
            local_compute_cost=5.0 - i * 1.0,
        )
        for i in range(num_splits)
    ]
    return DNNRequest(
        req_id=0, src_node=src, arrival_time=0.0,
        deadline_ms=deadline_ms, holding_time=5.0, splits=splits,
    )


def _make_agent_c() -> PPOAgentC:
    """A fresh randomly-initialised PPOAgentC for testing."""
    return PPOAgentC(input_dim=17, hidden_dims=(32, 16), device="cpu")


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_int_when_valid_actions():
    """select_c_topk_rerank returns an int when valid C actions exist."""
    env = _make_env()
    req = _make_request(env)
    obs_c = build_agent_c_observation(env, req)
    agent_c = _make_agent_c()

    action = select_c_topk_rerank(agent_c, obs_c, env, req, top_k=5)
    assert action is not None, "expected an action, got None"
    assert isinstance(action, (int, np.integer)), f"expected int, got {type(action)}"

    print("  PASS test_returns_int_when_valid_actions")


def test_returns_none_when_no_valid_actions():
    """Returns None when Agent-C mask has no valid actions."""
    env = _make_env()
    req = _make_request(env)
    obs_c = build_agent_c_observation(env, req)

    # Manually zero out the mask to simulate no valid actions
    obs_c["agent_c_mask"][:] = False

    agent_c = _make_agent_c()
    action = select_c_topk_rerank(agent_c, obs_c, env, req, top_k=5)
    assert action is None, f"expected None, got {action}"

    print("  PASS test_returns_none_when_no_valid_actions")


def test_topk_one_equivalent_to_original():
    """top_k=1 returns a valid, policy-consistent action without calling pressure.

    top_k=1 uses a shortcut that skips pressure computation entirely.
    We verify the returned action is (a) valid, (b) non-None, and (c) has
    the maximum policy score among valid actions.
    """
    env = _make_env()
    req = _make_request(env)
    obs_c = build_agent_c_observation(env, req)
    agent_c = _make_agent_c()

    # Top-K rerank with k=1
    rerank_action = select_c_topk_rerank(agent_c, obs_c, env, req, top_k=1)

    assert rerank_action is not None, "top_k=1 should return an action"
    assert isinstance(rerank_action, (int, np.integer))

    # Verify it has the maximum policy score
    scores, mask = _get_action_scores(agent_c, obs_c)
    valid = np.where(mask)[0]
    max_score = np.max(scores[valid])
    assert scores[int(rerank_action)] == max_score, (
        f"rerank_action={rerank_action} score={scores[int(rerank_action)]:.6f} "
        f"!= max={max_score:.6f}"
    )

    print("  PASS test_topk_one_equivalent_to_original")


def test_rerank_score_lower_is_better():
    """_compute_rerank_score gives lower scores to better candidates."""
    # A "good" candidate: many valid R actions, low ratio, slack > 0
    good = {
        "n_valid_r_actions": 20,
        "fs_lfb_ratio": 0.3,
        "delay_slack_ms": 50.0,
        "frag_pressure": 0.2,
        "server_utilization": 0.1,
    }
    # A "bad" candidate: zero valid R actions, tight, deadline exceeded
    bad = {
        "n_valid_r_actions": 0,
        "fs_lfb_ratio": 0.95,
        "delay_slack_ms": -30.0,
        "frag_pressure": 1.5,
        "server_utilization": 0.95,
    }

    good_score = _compute_rerank_score(good, DEFAULT_WEIGHTS)
    bad_score = _compute_rerank_score(bad, DEFAULT_WEIGHTS)

    assert good_score < bad_score, (
        f"good score ({good_score:.3f}) should be < bad score ({bad_score:.3f})"
    )

    print(f"  PASS test_rerank_score_lower_is_better (good={good_score:.3f} < bad={bad_score:.3f})")


def test_weights_affect_score():
    """Changing weights changes the rerank score."""
    pressure = {
        "n_valid_r_actions": 5,
        "fs_lfb_ratio": 0.5,
        "delay_slack_ms": 10.0,
        "frag_pressure": 0.4,
        "server_utilization": 0.3,
    }

    default_score = _compute_rerank_score(pressure, DEFAULT_WEIGHTS)

    # Increase block penalty
    heavy_block = {**DEFAULT_WEIGHTS, "w_block": 100.0}
    heavy_score = _compute_rerank_score(pressure, heavy_block)

    # Scores should be equal when n_valid > 0 (block penalty not triggered)
    assert abs(default_score - heavy_score) < 1e-9, (
        f"block penalty should not affect score when n_valid>0: "
        f"{default_score:.6f} vs {heavy_score:.6f}"
    )

    # But with n_valid=0 they should differ
    pressure_zero = {**pressure, "n_valid_r_actions": 0}
    ds0 = _compute_rerank_score(pressure_zero, DEFAULT_WEIGHTS)
    hs0 = _compute_rerank_score(pressure_zero, heavy_block)
    assert hs0 > ds0, f"heavier block weight should give higher score: {hs0:.3f} > {ds0:.3f}"

    print("  PASS test_weights_affect_score")


def test_get_action_scores_returns_finite():
    """_get_action_scores returns finite values for valid actions."""
    env = _make_env()
    req = _make_request(env)
    obs_c = build_agent_c_observation(env, req)
    agent_c = _make_agent_c()

    scores, mask = _get_action_scores(agent_c, obs_c)

    assert len(scores) == len(mask)
    assert np.all(np.isfinite(scores[mask])), "valid action scores should be finite"
    assert np.all(np.isneginf(scores[~mask])), "invalid action scores should be -inf"

    print("  PASS test_get_action_scores_returns_finite")


def test_select_from_raw_returns_valid():
    """select_c_topk_rerank_from_raw returns a valid action."""
    env = _make_env()
    req = _make_request(env)
    agent_c = _make_agent_c()

    action = select_c_topk_rerank_from_raw(agent_c, env, req, top_k=5)
    assert action is not None
    assert isinstance(action, (int, np.integer))

    # Also test greedy mode
    action_greedy = select_c_topk_rerank_from_raw(
        agent_c, env, req, top_k=5, c_policy="greedy",
    )
    assert action_greedy is not None

    print("  PASS test_select_from_raw_returns_valid")


def test_custom_weights_accepted():
    """Passing custom weights does not raise."""
    env = _make_env()
    req = _make_request(env)
    obs_c = build_agent_c_observation(env, req)
    agent_c = _make_agent_c()

    custom = {"w_block": 5.0, "w_ratio": 2.0}
    action = select_c_topk_rerank(
        agent_c, obs_c, env, req, top_k=5, weights=custom,
    )
    assert action is not None

    print("  PASS test_custom_weights_accepted")


def test_does_not_crash_with_extreme_deadline():
    """Very tight deadline does not crash the reranker."""
    env = _make_env(num_slots=8)
    req = _make_request(env, deadline_ms=0.5)
    obs_c = build_agent_c_observation(env, req)
    agent_c = _make_agent_c()

    # Should not raise
    action = select_c_topk_rerank(agent_c, obs_c, env, req, top_k=5)
    # At minimum, returns something or None without crashing
    print(f"  PASS test_does_not_crash_with_extreme_deadline (action={action})")


def test_different_topk_produce_actions():
    """Different top_k values produce valid results."""
    env = _make_env()
    req = _make_request(env)
    obs_c = build_agent_c_observation(env, req)
    agent_c = _make_agent_c()

    for k in [1, 2, 3, 5, 10]:
        action = select_c_topk_rerank(agent_c, obs_c, env, req, top_k=k)
        assert action is not None, f"top_k={k} returned None"
        assert isinstance(action, (int, np.integer)), f"top_k={k} returned {type(action)}"

    print("  PASS test_different_topk_produce_actions")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("Agent-C Top-K Rerank — Smoke Tests")
    print("=" * 60)

    test_returns_int_when_valid_actions()
    test_returns_none_when_no_valid_actions()
    test_topk_one_equivalent_to_original()
    test_rerank_score_lower_is_better()
    test_weights_affect_score()
    test_get_action_scores_returns_finite()
    test_select_from_raw_returns_valid()
    test_custom_weights_accepted()
    test_does_not_crash_with_extreme_deadline()
    test_different_topk_produce_actions()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
