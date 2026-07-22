"""Unit tests for the corrected regret metrics in train_r_counterfactual_ranking."""
from __future__ import annotations

import numpy as np

from sa_hmarl.training.train_r_counterfactual_ranking import _compute_group_metrics


def _make_mask(n_groups: int, n_cand: int):
    return np.ones((n_groups, n_cand), dtype=bool)


def test_aggregate_regret_reduction_positive_when_model_better():
    """If the model's mean regret is lower than PPO's, aggregate reduction is positive."""
    # Two groups. Model picks the best action in both groups -> model regret = 0.
    # PPO picks a suboptimal action -> PPO regret > 0.
    scores = np.array([
        [0.0, 1.0, 0.0],   # model selects candidate 1 (best return)
        [0.0, 0.0, 1.0],   # model selects candidate 2 (best return)
    ], dtype=np.float32)
    returns = np.array([
        [0.0, -1.0, -2.0],  # best return is 0 at index 0, but model selected index 1?
        [-3.0, -2.0, -1.0], # need best at selected index
    ], dtype=np.float32)
    # Adjust: make selected index have max return.
    returns = np.array([
        [-2.0, -1.0, -3.0],  # best at index 1
        [-3.0, -2.0, -1.0],  # best at index 2
    ], dtype=np.float32)
    mask = _make_mask(2, 3)
    ppo_action_index = np.array([0, 0], dtype=np.int64)  # PPO picks worse candidate
    metrics = _compute_group_metrics(scores, returns, mask, ppo_action_index)
    assert metrics["model_regret_mean"] == 0.0
    assert metrics["ppo_regret_mean"] > 0.0
    assert metrics["absolute_regret_improvement"] > 0.0
    assert metrics["aggregate_regret_reduction"] > 0.0
    assert metrics["aggregate_regret_reduction"] <= 1.0 + 1e-6


def test_aggregate_regret_reduction_does_not_explode_with_small_ppo_regret():
    """When PPO regret is tiny but positive, aggregate reduction stays bounded."""
    # Group 0: model picks index 1 (best), PPO picks index 0 (slightly worse).
    # Group 1: model picks index 0 (best), PPO picks index 1 (slightly worse).
    scores = np.array([
        [0.0, 1.0],
        [1.0, 0.0],
    ], dtype=np.float32)
    returns = np.array([
        [-1.0001, -1.0000],
        [-2.0000, -2.0001],
    ], dtype=np.float32)
    mask = _make_mask(2, 2)
    ppo_action_index = np.array([0, 1], dtype=np.int64)
    metrics = _compute_group_metrics(scores, returns, mask, ppo_action_index)
    assert metrics["aggregate_regret_reduction"] > 0.0
    assert metrics["aggregate_regret_reduction"] <= 1.0 + 1e-6
    assert abs(metrics["aggregate_regret_reduction"]) < 10.0


def test_better_equal_worse_rates_sum_to_one():
    """The three rates should sum to approximately 1."""
    np.random.seed(0)
    n_groups, n_cand = 100, 10
    returns = np.random.randn(n_groups, n_cand).astype(np.float32)
    scores = np.random.randn(n_groups, n_cand).astype(np.float32)
    mask = _make_mask(n_groups, n_cand)
    ppo_action_index = np.zeros(n_groups, dtype=np.int64)
    metrics = _compute_group_metrics(scores, returns, mask, ppo_action_index)
    total = (
        metrics["model_better_than_ppo_rate"]
        + metrics["model_equal_to_ppo_rate"]
        + metrics["model_worse_than_ppo_rate"]
    )
    assert abs(total - 1.0) < 1e-6


def test_model_better_rate_reflects_actual_rescues():
    """When model always picks a better action than PPO, better rate = 1."""
    scores = np.array([
        [0.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ], dtype=np.float32)
    returns = np.array([
        [-2.0, -1.0, -3.0],
        [-3.0, -2.0, -1.0],
    ], dtype=np.float32)
    mask = _make_mask(2, 3)
    ppo_action_index = np.array([0, 1], dtype=np.int64)
    metrics = _compute_group_metrics(scores, returns, mask, ppo_action_index)
    assert metrics["model_better_than_ppo_rate"] == 1.0
    assert metrics["model_equal_to_ppo_rate"] == 0.0
    assert metrics["model_worse_than_ppo_rate"] == 0.0
    assert metrics["mean_improvement_on_rescued_groups"] > 0.0
