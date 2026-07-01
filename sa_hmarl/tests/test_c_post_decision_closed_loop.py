"""Tests for learned post-decision closed-loop integration."""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from sa_hmarl.agents.post_decision_value import PostDecisionValueNetwork
from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.evaluation.eval_c_post_decision_closed_loop import (
    _apply_learned_method_verdict,
    _features_for_post_decision,
    _paired_bootstrap,
    evaluate,
    load_post_decision_checkpoint,
    select_post_decision_action,
)
from sa_hmarl.evaluation.generate_c_post_decision_dataset import FEATURE_NAMES
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.env.observation_builder import build_agent_c_observation


POST_CKPT = "sa_hmarl/checkpoints/post_decision/c_post_decision_joint.pt"


def test_checkpoint_loads_with_matching_schema():
    model, mean, std, checkpoint = load_post_decision_checkpoint(POST_CKPT, "cpu")
    assert isinstance(model, PostDecisionValueNetwork)
    assert mean.shape == std.shape == (50,)
    assert checkpoint["mode"] == "joint"


def test_post_decision_features_can_wrap_default_agent_c():
    env = make_env(
        "snap24_gnutella_reach", num_slots=24, num_servers=4, seed=42,
        max_blocks=10, block_sort_strategy="mixed", k=5,
    )
    rng = np.random.RandomState(123)
    req = generate_requests(
        env, rng, 0, num_requests=1, arrival_interval=0.15,
        holding_min=4.0, holding_max=10.0,
        deadline_min=30.0, deadline_max=100.0,
        size_min_mb=5.0, size_max_mb=30.0,
        edge_cost_min=0.5, edge_cost_max=15.0,
        num_splits=3, split_profile="default3",
    )[0]
    obs_c = build_agent_c_observation(env, req)
    agent_c = type("DefaultFeatureAgent", (), {
        "build_action_features": AgentC.build_action_features,
        "feature_mode": "default",
        "ablation": False,
        "zero_spectrum": False,
    })()
    features = _features_for_post_decision(agent_c, obs_c)
    assert features.shape == (len(obs_c["candidate_features"]), 27)


def test_post_only_selects_highest_prediction():
    model = PostDecisionValueNetwork(2, ())
    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.net[0].bias.zero_()
    action, prediction = select_post_decision_action(
        model, np.zeros(2), np.ones(2),
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        [3, 7], np.array([10.0, -10.0]), "post_only", 1.0, "cpu",
    )
    assert action == 7
    assert prediction.tolist() == [0.0, 2.0]


def test_rerank_lambda_zero_matches_policy_argmax():
    model = PostDecisionValueNetwork(2, ())
    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([[1.0, 0.0]]))
        model.net[0].bias.zero_()
    action, _ = select_post_decision_action(
        model, np.zeros(2), np.ones(2),
        np.array([[0.0, 0.0], [2.0, 0.0]], dtype=np.float32),
        [3, 7], np.array([2.0, 0.0]), "ppo_post_rerank", 0.0, "cpu",
    )
    assert action == 3


def test_post_delay_trades_value_for_lower_delay():
    model = PostDecisionValueNetwork(50, ())
    delay_index = FEATURE_NAMES.index("min_delay_est")
    with torch.no_grad():
        model.net[0].weight.zero_()
        model.net[0].weight[0, 0] = 1.0
        model.net[0].bias.zero_()
    features = np.zeros((2, 50), dtype=np.float32)
    features[:, 0] = [1.0, 0.9]
    features[:, delay_index] = [20.0, 5.0]
    action, _ = select_post_decision_action(
        model, np.zeros(50), np.ones(50), features,
        [3, 7], np.zeros(2), "post_delay", 2.0, "cpu",
    )
    assert action == 7


def test_paired_bootstrap_has_expected_sign():
    result = _paired_bootstrap([0.1, 0.2, 0.3], [0.2, 0.3, 0.4])
    assert result["delta"] < 0
    assert result["ci_95_hi"] < 0


def test_verdict_uses_learned_method_not_explicit_upper_bound():
    report = {
        "methods": {
            "ppo_c": {"aggregate": {
                "blocking_rate": 0.4665,
                "mean_decision_time_ms": 9.0,
                "p95_decision_time_ms": 14.0,
            }},
            "post_only": {"aggregate": {
                "blocking_rate": 0.4456,
                "mean_decision_time_ms": 8.2,
                "p95_decision_time_ms": 12.5,
            }},
            "explicit_spectrum": {"aggregate": {
                "blocking_rate": 0.4399,
                "mean_decision_time_ms": 263.0,
                "p95_decision_time_ms": 633.0,
            }},
        }
    }
    _apply_learned_method_verdict(report)
    assert report["best_learned_method"] == "post_only"
    assert report["explicit_gain_retained"] > 0.70
    assert report["verdict"] == "PASS_POST_DECISION_CLOSED_LOOP"


def test_closed_loop_smoke_and_agent_r_frozen(tmp_path):
    args = Namespace(
        post_checkpoint=POST_CKPT,
        agent_c_checkpoint="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt",
        agent_r_checkpoint="sa_hmarl/checkpoints/agent_r_mixed.pt",
        methods="ppo_c,post_only,ppo_post_rerank,greedy,rf", lambda_values="0.5",
        seeds="3030", episodes=1, requests_per_episode=4,
        topology="snap24_gnutella_reach", num_slots=20, num_servers=4,
        k_paths=5, max_blocks=10, block_sort_strategy="mixed",
        split_profile="default3", num_splits=3, arrival_interval=0.15,
        holding_min=4.0, holding_max=10.0, deadline_min=30.0,
        deadline_max=100.0, size_min_mb=5.0, size_max_mb=30.0,
        edge_cost_min=0.5, edge_cost_max=15.0, traffic_mode="iid",
        regime_stay_prob=0.9, modulation_profile="default",
        slot_bw_hz=1.25e9, guard_band_fs=1,
        potential_history_window=12, potential_probe_limit=2,
        util_threshold=0.95, alpha=0.3, device="cpu",
        output_json=str(tmp_path / "smoke.json"),
        output_md=str(tmp_path / "smoke.md"),
    )
    report = evaluate(args)
    assert report["agent_r_unchanged"] is True
    assert set(report["methods"]) == {
        "ppo_c", "post_only", "ppo_post_rerank_0.5", "greedy", "rf",
    }
    assert (tmp_path / "smoke.json").exists()
