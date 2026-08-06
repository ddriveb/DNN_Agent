from __future__ import annotations

import numpy as np
import torch

from sa_hmarl.pure_rmsa_v13.neural_opportunity.model import (
    TinyOpportunityMLP,
    TinyOpportunityWeights,
)
from sa_hmarl.pure_rmsa_v13.neural_opportunity.protocol import (
    FEATURE_DIM,
    HIDDEN_DIM,
    K_PATHS,
    MAX_CANDIDATES,
    PROTOCOL_ID,
)
from sa_hmarl.pure_rmsa_v13.neural_opportunity.training import composite_loss


def test_locked_protocol_shape():
    assert PROTOCOL_ID == "pure_rmsa_neural_opportunity_v1"
    assert K_PATHS == 50
    assert MAX_CANDIDATES == 150
    assert FEATURE_DIM == 110


def test_numpy_forward_matches_torch():
    torch.manual_seed(7)
    model = TinyOpportunityMLP()
    features = np.random.default_rng(7).normal(
        size=(3, FEATURE_DIM)
    ).astype(np.float32)
    weights = TinyOpportunityWeights.from_torch(model)
    with torch.inference_mode():
        expected = model(torch.from_numpy(features)).numpy()
    actual = weights.forward(features)
    np.testing.assert_allclose(actual, expected, rtol=1e-5, atol=1e-6)


def test_zero_weights_are_deterministic():
    features = np.random.default_rng(3).normal(
        size=(3, FEATURE_DIM)
    ).astype(np.float32)
    weights = TinyOpportunityWeights.zeros()
    np.testing.assert_array_equal(
        weights.forward(features), np.zeros((3, 50), dtype=np.float32)
    )
    assert weights.w1.shape == (FEATURE_DIM, HIDDEN_DIM)


def test_checkpoint_round_trip(tmp_path):
    weights = TinyOpportunityWeights.zeros()
    path = tmp_path / "weights.npz"
    weights.save(path)
    loaded = TinyOpportunityWeights.load(path)
    np.testing.assert_array_equal(loaded.w1, weights.w1)


def test_composite_loss_is_finite():
    model = TinyOpportunityMLP()
    features = torch.zeros((2, 3, FEATURE_DIM), dtype=torch.float32)
    valid = torch.zeros((2, 3, 50), dtype=torch.bool)
    valid[:, 0, :3] = True
    targets = torch.zeros((2, 3, 50), dtype=torch.float32)
    targets[:, 0, :3] = torch.tensor([0.0, 1.0, 2.0])
    losses = composite_loss(model, features, valid, targets, 10.0)
    assert all(torch.isfinite(loss) for loss in losses)
