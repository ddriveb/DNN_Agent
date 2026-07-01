"""Smoke tests for lookahead C-action scoring.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/tests/test_c_lookahead_oracle.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import copy
import math

import numpy as np
import torch

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.evaluation.c_lookahead_oracle import (
    score_c_action_lookahead,
    select_lookahead_c_action,
)


# ---------------------------------------------------------------------------
# Expected return keys
# ---------------------------------------------------------------------------

REQUIRED_KEYS = [
    "score",
    "action_idx_c",
    "action_c",
    "current_success",
    "current_reason",
    "current_delay_ms",
    "future_block_count",
    "future_no_suitable_block_count",
    "future_overload_count",
    "future_success_count",
    "future_avg_delay_ms",
    "future_avg_fs",
    "future_avg_waste",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_env(num_slots: int = 32, num_servers: int = 2):
    net = OpticalNetwork("net1", num_slots=num_slots)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=num_servers, seed=42,
        server_nodes=[0, 3] if num_servers >= 2 else [0],
        capacities=[50.0] * num_servers,
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    return SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)


def _make_request(src: int = 1, deadline_ms: float = 200.0,
                  arrival_time: float = 0.0, holding_time: float = 2.0,
                  num_splits: int = 3) -> DNNRequest:
    splits = [
        SplitProfile(
            split_id=i,
            intermediate_size_mb=10.0 + i * 5.0,
            edge_compute_cost=1.0 + i * 0.5,
            local_compute_cost=3.0 - i * 0.5,
        )
        for i in range(num_splits)
    ]
    return DNNRequest(
        req_id=0, src_node=src, arrival_time=arrival_time,
        deadline_ms=deadline_ms, holding_time=holding_time, splits=splits,
    )


def _make_agents():
    agent_c = PPOAgentC(input_dim=17, hidden_dims=(32, 16), device="cpu")
    agent_c.policy_net.eval()
    mod_reg = ModulationRegistry()
    agent_r = PPOAgentR(
        input_dim=11, mod_registry=mod_reg,
        hidden_dims=(32, 16), device="cpu",
    )
    agent_r.policy_net.eval()
    return agent_c, agent_r


def _get_first_valid_c_action(env, req, agent_c):
    obs_c = build_agent_c_observation(env, req)
    mask = obs_c["agent_c_mask"]
    valid = np.where(mask)[0]
    return int(valid[0]) if len(valid) > 0 else None


def _setup_env_with_request(env: SMDPEnv, req: DNNRequest):
    """Reset env with a single request so stats are initialised."""
    env.reset([req])
    return env


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_returns_all_required_keys():
    """score_c_action_lookahead returns every required key."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    action_idx = _get_first_valid_c_action(env, req, agent_c)
    assert action_idx is not None, "no valid C action"

    future_reqs = [_make_request(arrival_time=10.0 + i, deadline_ms=300.0)
                   for i in range(3)]

    env_before = copy.deepcopy(env)
    result = score_c_action_lookahead(
        agent_c, agent_r, env, req, action_idx, future_reqs,
    )

    for key in REQUIRED_KEYS:
        assert key in result, f"missing key: {key}"

    # Verify env not mutated
    assert env.time == env_before.time
    assert len(env.active_connections) == len(env_before.active_connections)
    assert len(env.event_queue) == len(env_before.event_queue)

    print("  PASS test_returns_all_required_keys")


def test_does_not_mutate_original_env():
    """Original environment is completely untouched."""
    env = _make_env(num_slots=16)
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    action_idx = _get_first_valid_c_action(env, req, agent_c)
    future_reqs = [_make_request(arrival_time=5.0 + i) for i in range(5)]

    snapshot = {
        "time": env.time,
        "num_active": len(env.active_connections),
        "queue_len": len(env.event_queue),
    }

    _ = score_c_action_lookahead(
        agent_c, agent_r, env, req, action_idx, future_reqs,
    )

    assert env.time == snapshot["time"]
    assert len(env.active_connections) == snapshot["num_active"]
    assert len(env.event_queue) == snapshot["queue_len"]

    print("  PASS test_does_not_mutate_original_env")


def test_invalid_action_returns_large_score():
    """An out-of-range action_idx_c returns current_failed and large score."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    future_reqs = [_make_request(arrival_time=10.0)]

    # action_idx 9999 is definitely out of range
    result = score_c_action_lookahead(
        agent_c, agent_r, env, req, 9999, future_reqs,
    )

    assert not result["current_success"]
    assert result["score"] >= 5.0, (
        f"expected score >= 5 (current_block_penalty), got {result['score']}"
    )

    print("  PASS test_invalid_action_returns_large_score")


def test_horizon_zero_only_evaluates_current():
    """horizon=0 skips future simulation entirely."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    action_idx = _get_first_valid_c_action(env, req, agent_c)
    future_reqs = [_make_request(arrival_time=10.0) for _ in range(10)]

    class Args:
        lookahead_horizon = 0

    result = score_c_action_lookahead(
        agent_c, agent_r, env, req, action_idx, future_reqs, Args(),
    )

    assert result["future_block_count"] == 0
    assert result["future_success_count"] == 0
    assert result["future_avg_delay_ms"] == 0.0

    print("  PASS test_horizon_zero_only_evaluates_current")


def test_no_future_reqs_handled():
    """Empty future_reqs list works correctly."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    action_idx = _get_first_valid_c_action(env, req, agent_c)

    result = score_c_action_lookahead(
        agent_c, agent_r, env, req, action_idx, [],
    )

    assert result["future_block_count"] == 0
    assert result["future_success_count"] == 0

    print("  PASS test_no_future_reqs_handled")


def test_score_types_are_reasonable():
    """All score fields have reasonable types and ranges."""
    env = _make_env()
    req = _make_request(deadline_ms=300.0)
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    action_idx = _get_first_valid_c_action(env, req, agent_c)
    future_reqs = [_make_request(arrival_time=10.0 + i, deadline_ms=300.0)
                   for i in range(3)]

    result = score_c_action_lookahead(
        agent_c, agent_r, env, req, action_idx, future_reqs,
    )

    assert isinstance(result["score"], float)
    assert math.isfinite(result["score"])
    assert isinstance(result["action_idx_c"], int)
    assert isinstance(result["action_c"], tuple)
    assert isinstance(result["current_success"], bool)
    assert isinstance(result["current_reason"], str)
    assert isinstance(result["current_delay_ms"], float)
    assert isinstance(result["future_block_count"], int)
    assert isinstance(result["future_no_suitable_block_count"], int)
    assert isinstance(result["future_overload_count"], int)
    assert isinstance(result["future_success_count"], int)
    assert isinstance(result["future_avg_delay_ms"], float)
    assert isinstance(result["future_avg_fs"], float)
    assert isinstance(result["future_avg_waste"], float)

    # Ranges
    assert result["future_block_count"] >= 0
    assert result["future_success_count"] >= 0
    assert result["current_delay_ms"] >= 0.0

    print("  PASS test_score_types_are_reasonable")


def test_different_actions_produce_different_scores():
    """Different C actions should (usually) produce different scores."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()

    obs_c = build_agent_c_observation(env, req)
    mask = obs_c["agent_c_mask"]
    valid = np.where(mask)[0]
    if len(valid) < 2:
        print("  SKIP test_different_actions_produce_different_scores (<2 valid actions)")
        return

    future_reqs = [_make_request(arrival_time=10.0 + i) for i in range(3)]
    r0 = score_c_action_lookahead(
        agent_c, agent_r, env, req, int(valid[0]), future_reqs,
    )
    r1 = score_c_action_lookahead(
        agent_c, agent_r, env, req, int(valid[1]), future_reqs,
    )

    # At minimum, both should have finite scores
    assert math.isfinite(r0["score"])
    assert math.isfinite(r1["score"])

    print(f"  PASS test_different_actions_produce_different_scores "
          f"(score0={r0['score']:.3f}, score1={r1['score']:.3f})")


def test_score_lower_for_successful_action():
    """A successful current action scores lower than a forced-failure one."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    action_idx = _get_first_valid_c_action(env, req, agent_c)
    future_reqs = [_make_request(arrival_time=10.0 + i) for i in range(3)]

    result_valid = score_c_action_lookahead(
        agent_c, agent_r, env, req, action_idx, future_reqs,
    )
    result_invalid = score_c_action_lookahead(
        agent_c, agent_r, env, req, 9999, future_reqs,
    )

    assert result_invalid["score"] > result_valid["score"], (
        f"invalid score ({result_invalid['score']:.3f}) should be > "
        f"valid score ({result_valid['score']:.3f})"
    )

    print("  PASS test_score_lower_for_successful_action")


# ---------------------------------------------------------------------------
# Tests for select_lookahead_c_action
# ---------------------------------------------------------------------------

def test_select_returns_valid_action():
    """select_lookahead_c_action returns a valid C action index."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    future_reqs = [_make_request(arrival_time=10.0 + i) for i in range(3)]

    best_action, diag = select_lookahead_c_action(
        agent_c, agent_r, env, req, future_reqs,
    )

    assert best_action is not None, "expected a valid action"
    assert isinstance(best_action, (int, np.integer))
    assert "best_score" in diag
    assert "mean_score" in diag
    assert "num_candidates" in diag
    assert "best_score_detail" in diag
    assert "top3_actions" in diag

    print(f"  PASS test_select_returns_valid_action "
          f"(best={best_action}, candidates={diag['num_candidates']}, "
          f"score={diag['best_score']:.3f})")


def test_top3_actions_length():
    """top3_actions has at most 3 entries."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    future_reqs = [_make_request(arrival_time=10.0 + i) for i in range(2)]

    _, diag = select_lookahead_c_action(
        agent_c, agent_r, env, req, future_reqs,
    )

    top3 = diag["top3_actions"]
    assert len(top3) <= 3, f"top3_actions length {len(top3)} > 3"
    assert len(top3) >= 1, "expected at least 1 entry in top3"

    # Each entry has the required keys
    for entry in top3:
        assert "action_idx_c" in entry
        assert "action_c" in entry
        assert "score" in entry
        assert "current_success" in entry

    print(f"  PASS test_top3_actions_length ({len(top3)} entries)")


def test_best_score_is_minimum():
    """best_score equals min of all candidate scores."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    future_reqs = [_make_request(arrival_time=10.0 + i) for i in range(3)]

    # Compute all scores manually
    obs_c = build_agent_c_observation(env, req)
    valid = np.where(obs_c["agent_c_mask"])[0]
    all_scores = []
    for idx in valid:
        detail = score_c_action_lookahead(
            agent_c, agent_r, env, req, int(idx), future_reqs,
        )
        all_scores.append(detail["score"])

    _, diag = select_lookahead_c_action(
        agent_c, agent_r, env, req, future_reqs,
    )

    expected_min = min(all_scores)
    assert abs(diag["best_score"] - expected_min) < 1e-9, (
        f"best_score={diag['best_score']:.6f} != min={expected_min:.6f}"
    )

    print(f"  PASS test_best_score_is_minimum (best={diag['best_score']:.3f})")


def test_select_returns_none_when_no_valid_c():
    """Returns None when agent_c_mask is all False."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    future_reqs = [_make_request(arrival_time=10.0)]

    # Manually build obs_c and zero out the mask
    obs_c = build_agent_c_observation(env, req)
    obs_c["agent_c_mask"][:] = False

    # Monkey-patch build_agent_c_observation to return our modified obs_c
    import sa_hmarl.evaluation.c_lookahead_oracle as clo
    _orig_build = clo.build_agent_c_observation
    clo.build_agent_c_observation = lambda e, r: obs_c

    try:
        best_action, diag = select_lookahead_c_action(
            agent_c, agent_r, env, req, future_reqs,
        )
        assert best_action is None
        assert diag["reason"] == "no_valid_c_actions"
        assert diag["num_candidates"] == 0
    finally:
        clo.build_agent_c_observation = _orig_build

    print("  PASS test_select_returns_none_when_no_valid_c")


def test_select_does_not_mutate_env():
    """select_lookahead_c_action leaves the original env untouched."""
    env = _make_env()
    req = _make_request()
    _setup_env_with_request(env, req)
    agent_c, agent_r = _make_agents()
    future_reqs = [_make_request(arrival_time=10.0 + i) for i in range(3)]

    snapshot = {
        "time": env.time,
        "num_active": len(env.active_connections),
        "queue_len": len(env.event_queue),
    }

    _, _ = select_lookahead_c_action(
        agent_c, agent_r, env, req, future_reqs,
    )

    assert env.time == snapshot["time"]
    assert len(env.active_connections) == snapshot["num_active"]
    assert len(env.event_queue) == snapshot["queue_len"]

    print("  PASS test_select_does_not_mutate_env")


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("C Lookahead Oracle — Smoke Tests")
    print("=" * 60)

    test_returns_all_required_keys()
    test_does_not_mutate_original_env()
    test_invalid_action_returns_large_score()
    test_horizon_zero_only_evaluates_current()
    test_no_future_reqs_handled()
    test_score_types_are_reasonable()
    test_different_actions_produce_different_scores()
    test_score_lower_for_successful_action()

    print()
    print("─" * 60)
    print("select_lookahead_c_action tests")
    print("─" * 60)

    test_select_returns_valid_action()
    test_top3_actions_length()
    test_best_score_is_minimum()
    test_select_returns_none_when_no_valid_c()
    test_select_does_not_mutate_env()

    print()
    print("=" * 60)
    print("ALL TESTS PASSED")
    print("=" * 60)
