"""Unit tests for the `mr_feasibility_rule` Agent-C feature mode."""
from __future__ import annotations

import numpy as np
import pytest

from sa_hmarl.agents.ppo_agents import PPOAgentC
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.training.utils import generate_requests, make_env


@pytest.fixture
def base_env():
    return make_env(
        topology="snap24_gnutella_reach",
        num_slots=20,
        num_servers=4,
        seed=42,
        modulation_profile="default",
        max_blocks=10,
        block_sort_strategy="mixed",
        k=5,
    )


@pytest.fixture
def sample_obs(base_env):
    rng = np.random.RandomState(42)
    requests = generate_requests(base_env, rng, src_node=0, num_requests=5, split_profile="default3")
    base_env.reset(requests)
    return build_agent_c_observation(base_env, requests[0])


def test_input_dim():
    agent = PPOAgentC(
        input_dim=17, hidden_dims=(8, 8), device="cpu",
        feature_mode="mr_feasibility_rule", num_servers=4,
    )
    assert agent.input_dim == 33


def test_build_action_features_shape(sample_obs):
    agent = PPOAgentC(
        input_dim=17, hidden_dims=(8, 8), device="cpu",
        feature_mode="mr_feasibility_rule", num_servers=4,
    )
    feat, mask = agent.build_action_features(sample_obs)
    assert feat.ndim == 2
    assert feat.shape[1] == 33
    assert feat.shape[0] == mask.shape[0]


def test_last_six_dims_finite_and_in_unit_interval(sample_obs):
    agent = PPOAgentC(
        input_dim=17, hidden_dims=(8, 8), device="cpu",
        feature_mode="mr_feasibility_rule", num_servers=4,
    )
    feat, _ = agent.build_action_features(sample_obs)
    mr = feat[:, -6:]
    assert np.isfinite(mr).all()
    assert (mr >= -1e-6).all()
    assert (mr <= 1.0 + 1e-6).all()


def test_zero_valid_r_implies_safe_indicator_zero(base_env, sample_obs):
    """Candidates with no valid R actions must have safe_indicator == 0."""
    agent = PPOAgentC(
        input_dim=17, hidden_dims=(8, 8), device="cpu",
        feature_mode="mr_feasibility_rule", num_servers=4,
    )
    feat, _ = agent.build_action_features(sample_obs)

    from sa_hmarl.agents.c_agent import AgentC
    for idx, feat_dict in enumerate(sample_obs["candidate_features"]):
        diag = AgentC.compute_r_feasibility_diagnostics(sample_obs, feat_dict)
        if diag.get("valid_r_actions", 0) == 0:
            assert feat[idx, -1] == pytest.approx(0.0, abs=1e-6), \
                f"candidate {idx}: valid_r_actions=0 but safe_indicator={feat[idx, -1]}"


def test_negative_server_margin_implies_safe_indicator_zero(base_env, sample_obs):
    """Candidates whose post-action server margin is negative must have safe_indicator == 0."""
    agent = PPOAgentC(
        input_dim=17, hidden_dims=(8, 8), device="cpu",
        feature_mode="mr_feasibility_rule", num_servers=4,
    )
    feat, _ = agent.build_action_features(sample_obs)

    for idx, feat_dict in enumerate(sample_obs["candidate_features"]):
        available = float(feat_dict.get("server_available_compute", 0.0))
        capacity = max(float(feat_dict.get("server_capacity", 1.0)), 1e-6)
        edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
        margin_raw = (available - edge_cost) / capacity
        if margin_raw < -1e-6:
            assert feat[idx, -1] == pytest.approx(0.0, abs=1e-6), \
                f"candidate {idx}: margin_raw={margin_raw:.4f} but safe_indicator={feat[idx, -1]}"


def test_safe_indicator_one_requires_all_conditions(base_env, sample_obs):
    """If safe_indicator == 1 then all three raw safety conditions hold."""
    agent = PPOAgentC(
        input_dim=17, hidden_dims=(8, 8), device="cpu",
        feature_mode="mr_feasibility_rule", num_servers=4,
    )
    feat, _ = agent.build_action_features(sample_obs)

    from sa_hmarl.agents.c_agent import AgentC
    from sa_hmarl.env.fs_demand import DEFAULT_PROP_SPEED_KM_S, DEFAULT_SETUP_TIME_S

    deadline_ms = float(sample_obs["request_features"]["deadline_ms"])
    for idx, feat_dict in enumerate(sample_obs["candidate_features"]):
        if feat[idx, -1] < 0.5:
            continue

        available = float(feat_dict.get("server_available_compute", 0.0))
        capacity = max(float(feat_dict.get("server_capacity", 1.0)), 1e-6)
        edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
        margin_raw = (available - edge_cost) / capacity
        assert margin_raw >= -1e-6, f"candidate {idx}: safe=1 but margin_raw={margin_raw:.4f}"

        local_ms = float(feat_dict.get("local_compute_ms", 0.0))
        edge_ms = float(feat_dict.get("edge_compute_ms", 0.0))
        min_path_km = float(feat_dict.get("server_min_path_km", 0.0))
        path_delay_est = (min_path_km / DEFAULT_PROP_SPEED_KM_S) * 1000.0 + DEFAULT_SETUP_TIME_S * 1000.0
        deadline_raw = (deadline_ms - (local_ms + edge_ms + path_delay_est)) / max(deadline_ms, 1e-6)
        assert deadline_raw >= -1e-6, f"candidate {idx}: safe=1 but deadline_raw={deadline_raw:.4f}"

        diag = AgentC.compute_r_feasibility_diagnostics(sample_obs, feat_dict)
        assert diag.get("valid_r_actions", 0) > 0, f"candidate {idx}: safe=1 but valid_r_actions=0"
