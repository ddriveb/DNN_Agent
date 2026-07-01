"""Tests for C-side post-decision value learning."""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import torch

from sa_hmarl.agents.post_decision_value import (
    PostDecisionValueNetwork,
    pairwise_ranking_loss,
)
from sa_hmarl.training.train_c_post_decision_value import (
    _group_batches,
    _group_index_map,
    _predict,
    evaluate_predictions,
    train_mode,
)


def test_network_forward_shape():
    model = PostDecisionValueNetwork(50, (32, 16))
    assert model(torch.randn(7, 50)).shape == (7,)


def test_pairwise_loss_rewards_correct_order():
    labels = torch.tensor([0.0, 1.0, 2.0])
    groups = torch.tensor([1, 1, 1])
    correct = pairwise_ranking_loss(torch.tensor([0.0, 1.0, 2.0]), labels, groups)
    reversed_loss = pairwise_ranking_loss(torch.tensor([2.0, 1.0, 0.0]), labels, groups)
    assert correct < reversed_loss


def test_pairwise_loss_never_compares_different_groups():
    labels = torch.tensor([0.0, 1.0])
    predictions = torch.tensor([10.0, -10.0], requires_grad=True)
    groups = torch.tensor([1, 2])
    loss = pairwise_ranking_loss(predictions, labels, groups)
    assert loss.item() == 0.0


def test_group_index_map_preserves_complete_groups():
    group_ids = np.array([2, 1, 2, 1, 3])
    groups = _group_index_map(group_ids)
    assert {tuple(sorted(indices.tolist())) for indices in groups} == {(0, 2), (1, 3), (4,)}


def test_group_batches_do_not_split_groups():
    groups = [np.array([0, 1]), np.array([2, 3, 4]), np.array([5])]
    batches = list(_group_batches(groups, np.random.RandomState(0), 2, False))
    assert batches[0].tolist() == [0, 1, 2, 3, 4]
    assert batches[1].tolist() == [5]


def test_prediction_metrics_are_perfect_for_exact_predictions():
    labels = np.array([0.0, 1.0, 0.5, 0.2], dtype=np.float32)
    groups = np.array([1, 1, 2, 2], dtype=np.int64)
    metrics = evaluate_predictions(labels, labels.copy(), groups)
    assert metrics["mae"] == 0.0
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["mean_regret"] == 0.0
    assert metrics["pairwise_accuracy"] == 1.0


def test_top1_accepts_any_tied_optimal_candidate():
    labels = np.array([1.0, 1.0, 0.0], dtype=np.float32)
    predictions = np.array([0.5, 2.0, 0.0], dtype=np.float32)
    metrics = evaluate_predictions(labels, predictions, np.array([1, 1, 1]))
    assert metrics["top1_accuracy"] == 1.0
    assert metrics["exact_index_top1_accuracy"] == 0.0


def test_predict_applies_training_normalization():
    model = PostDecisionValueNetwork(2, ())
    with torch.no_grad():
        model.net[0].weight.copy_(torch.tensor([[1.0, 1.0]]))
        model.net[0].bias.zero_()
    features = np.array([[3.0, 6.0]], dtype=np.float32)
    prediction = _predict(
        model, features, np.array([1.0, 2.0]), np.array([2.0, 2.0]), "cpu"
    )
    assert prediction.tolist() == [3.0]


def test_training_smoke_reduces_validation_loss():
    rng = np.random.RandomState(1)
    features = rng.randn(48, 4).astype(np.float32)
    labels = (features[:, 0] - 0.5 * features[:, 1]).astype(np.float32)
    groups = np.repeat(np.arange(12), 4).astype(np.int64)
    data = {"features": features, "labels": labels, "group_ids": groups}
    args = Namespace(
        seed=1, hidden_dims=(16, 8), dropout=0.0, device="cpu", lr=1e-2,
        weight_decay=0.0, patience=10, epochs=30, groups_per_batch=4,
        huber_delta=1.0, rank_coef=0.2, max_grad_norm=1.0,
        min_delta=1e-6, log_every=100,
    )
    model, result = train_mode(
        "joint", data, data, features.mean(0), np.maximum(features.std(0), 1e-6), args
    )
    assert result["best_val_loss"] < result["history"][0]["val_loss"]
    assert isinstance(model, PostDecisionValueNetwork)
