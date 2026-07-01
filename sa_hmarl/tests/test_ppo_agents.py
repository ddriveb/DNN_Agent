"""Smoke tests for masked PPO agents.

Run:
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/tests/test_ppo_agents.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.network.modulation import ModulationRegistry


def test_masked_ppo_selects_only_valid_action():
    agent = PPOAgentC(input_dim=17, hidden_dims=(32,), device="cpu")
    features = np.random.RandomState(1).rand(5, 17).astype(np.float32)
    mask = np.array([False, True, False, True, False], dtype=bool)
    selected = set()
    for _ in range(50):
        action, _, _ = agent.select_from_features(features, mask, deterministic=False)
        selected.add(action)
    assert selected.issubset({1, 3})
    print("test_masked_ppo_selects_only_valid_action PASSED")


def test_masked_ppo_all_false_returns_none():
    agent = PPOAgentR(
        input_dim=11,
        mod_registry=ModulationRegistry(),
        hidden_dims=(32,),
        device="cpu",
    )
    features = np.random.RandomState(2).rand(4, 11).astype(np.float32)
    mask = np.zeros(4, dtype=bool)
    action, log_prob, entropy = agent.select_from_features(features, mask)
    assert action is None
    assert log_prob == 0.0
    assert entropy == 0.0
    print("test_masked_ppo_all_false_returns_none PASSED")


def test_masked_ppo_optimize_finite_loss():
    agent = PPOAgentC(input_dim=17, hidden_dims=(32,), device="cpu")
    rng = np.random.RandomState(3)
    features_list = tuple(rng.rand(4, 17).astype(np.float32) for _ in range(3))
    masks_list = tuple(np.array([True, True, False, True], dtype=bool) for _ in range(3))
    actions = (0, 1, 3)
    old_log_probs = (-1.0, -1.0, -1.0)
    advantages = np.array([0.5, -0.2, 0.1], dtype=np.float32)
    loss, entropy, approx_kl, ref_kl = agent.optimize_ppo(
        features_list,
        masks_list,
        actions,
        old_log_probs,
        advantages,
        clip_coef=0.2,
    )
    assert math.isfinite(loss)
    assert math.isfinite(entropy)
    assert math.isfinite(approx_kl)
    assert ref_kl == 0.0  # no reference policy by default
    print("test_masked_ppo_optimize_finite_loss PASSED")


if __name__ == "__main__":
    test_masked_ppo_selects_only_valid_action()
    test_masked_ppo_all_false_returns_none()
    test_masked_ppo_optimize_finite_loss()
    print("\n=== All PPO agent smoke tests PASSED ===")
