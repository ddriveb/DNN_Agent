"""Smoke tests for Dual-Critic MAPPO training script."""
import numpy as np
import pytest
import torch

from sa_hmarl.training.train_joint_mappo_dual_critic import (
    DualPPOJointTransition,
    DualPPORolloutBuffer,
    DualCentralizedValueCritic,
    CentralizedValueNetwork,
    _filter_actor_batch,
)


def _make_transition(value_c=0.5, value_r=0.3, reward_c=0.2, reward_r=0.1, done=False,
                       c_valid=True, r_valid=True):
    return DualPPOJointTransition(
        c_features=np.random.randn(3, 17).astype(np.float32),
        c_mask=np.array([True, True, False]) if c_valid else np.array([False, False, False]),
        c_action=0,
        c_log_prob=-0.5,
        c_valid=c_valid,
        r_features=np.random.randn(5, 11).astype(np.float32),
        r_mask=np.array([True, False, True, False, True]) if r_valid else np.array([False]*5),
        r_action=2,
        r_log_prob=-0.7,
        r_valid=r_valid,
        value_features_c=np.random.randn(59).astype(np.float32),
        value_c=value_c,
        value_features_r=np.random.randn(70).astype(np.float32),
        value_r=value_r,
        reward_c=reward_c,
        reward_r=reward_r,
        done=done,
        info={"success": True},
    )


# ------------------------------------------------------------------
# Network shapes
# ------------------------------------------------------------------

def test_critic_c_input_dim():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    assert critic.input_dim_c == 17 + 17 + 11 + 14
    assert critic.input_dim_r == 17 + 17 + 11 + 11 + 14


def test_critic_c_forward_shape():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    x = torch.randn(4, critic.input_dim_c)
    out = critic.value_net_c(x)
    assert out.shape == (4,)


def test_critic_r_forward_shape():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    x = torch.randn(4, critic.input_dim_r)
    out = critic.value_net_r(x)
    assert out.shape == (4,)


# ------------------------------------------------------------------
# Encode methods
# ------------------------------------------------------------------

def test_encode_c_shape():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    c_features = np.random.randn(3, 17).astype(np.float32)
    c_mask = np.array([True, True, False])
    r_features = np.random.randn(5, 11).astype(np.float32)
    r_mask = np.array([True, False, True, False, True])
    encoded = critic.encode_c(c_features, c_mask, 0, r_features, r_mask, env=None)
    assert encoded.shape == (critic.input_dim_c,)
    assert encoded.dtype == np.float32


def test_encode_r_shape():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    c_features = np.random.randn(3, 17).astype(np.float32)
    c_mask = np.array([True, True, False])
    r_features = np.random.randn(5, 11).astype(np.float32)
    r_mask = np.array([True, False, True, False, True])
    encoded = critic.encode_r(c_features, c_mask, 0, r_features, r_mask, 2, env=None)
    assert encoded.shape == (critic.input_dim_r,)
    assert encoded.dtype == np.float32


# ------------------------------------------------------------------
# Value estimation
# ------------------------------------------------------------------

def test_value_c_returns_scalar():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    features = np.random.randn(critic.input_dim_c).astype(np.float32)
    v = critic.value_c(features)
    assert isinstance(v, float)


def test_value_r_returns_scalar():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    features = np.random.randn(critic.input_dim_r).astype(np.float32)
    v = critic.value_r(features)
    assert isinstance(v, float)


# ------------------------------------------------------------------
# Critic optimize (finite loss)
# ------------------------------------------------------------------

def test_optimize_c_finite_loss():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    features = [np.random.randn(critic.input_dim_c).astype(np.float32) for _ in range(8)]
    returns = np.random.randn(8).astype(np.float32)
    old_values = np.random.randn(8).astype(np.float32)
    loss = critic.optimize_c(features, returns, old_values=old_values, epochs=2, clip_coef=0.2)
    assert np.isfinite(loss)
    assert loss >= 0.0


def test_optimize_r_finite_loss():
    critic = DualCentralizedValueCritic(c_dim=17, r_dim=11, global_dim=14)
    features = [np.random.randn(critic.input_dim_r).astype(np.float32) for _ in range(8)]
    returns = np.random.randn(8).astype(np.float32)
    old_values = np.random.randn(8).astype(np.float32)
    loss = critic.optimize_r(features, returns, old_values=old_values, epochs=2, clip_coef=0.2)
    assert np.isfinite(loss)
    assert loss >= 0.0


# ------------------------------------------------------------------
# Rollout buffer + GAE
# ------------------------------------------------------------------

def test_rollout_buffer_gae_c():
    buf = DualPPORolloutBuffer()
    for i in range(10):
        buf.add(_make_transition(value_c=float(i) * 0.1, reward_c=1.0, done=(i == 9)))
    advantages, returns, values = buf.compute_gae_c(gamma=0.95, gae_lambda=0.95)
    assert advantages.shape == (10,)
    assert returns.shape == (10,)
    assert np.isfinite(advantages).all()


def test_rollout_buffer_gae_r():
    buf = DualPPORolloutBuffer()
    for i in range(10):
        buf.add(_make_transition(value_r=float(i) * 0.1, reward_r=-0.5, done=(i == 9)))
    advantages, returns, values = buf.compute_gae_r(gamma=0.95, gae_lambda=0.95)
    assert advantages.shape == (10,)
    assert returns.shape == (10,)
    assert np.isfinite(advantages).all()


def test_rollout_buffer_gae_c_normalization():
    buf = DualPPORolloutBuffer()
    for i in range(10):
        buf.add(_make_transition(value_c=0.5, reward_c=1.0, done=(i == 9)))
    advantages, _, _ = buf.compute_gae_c(gamma=0.95, gae_lambda=0.95, normalize=True)
    assert abs(advantages.mean()) < 1e-6
    assert abs(advantages.std() - 1.0) < 1e-6


# ------------------------------------------------------------------
# Actor batch filtering
# ------------------------------------------------------------------

def test_filter_actor_batch_c():
    transitions = [
        _make_transition(c_valid=True, r_valid=True),
        _make_transition(c_valid=False, r_valid=True),
        _make_transition(c_valid=True, r_valid=False),
    ]
    features, masks, actions, logps, indices = _filter_actor_batch(transitions, "c")
    assert len(features) == 2
    assert indices == [0, 2]


def test_filter_actor_batch_r():
    transitions = [
        _make_transition(c_valid=True, r_valid=True),
        _make_transition(c_valid=False, r_valid=True),
        _make_transition(c_valid=True, r_valid=False),
    ]
    features, masks, actions, logps, indices = _filter_actor_batch(transitions, "r")
    assert len(features) == 2
    assert indices == [0, 1]


# ------------------------------------------------------------------
# End-to-end smoke: 3 episodes
# ------------------------------------------------------------------

def test_smoke_train_3_episodes():
    """Run a minimal 3-episode training to verify the full pipeline."""
    from types import SimpleNamespace
    from sa_hmarl.training.train_joint_mappo_dual_critic import train

    args = SimpleNamespace(
        topologies="net1",
        modulation_profile="default",
        warm_start_c=None,
        warm_start_r=None,
        freeze_r_episodes=0,
        r_ref_kl_coef=0.0,
        episodes=3,
        requests_per_episode=4,
        arrival_interval=0.5,
        holding_min=2.0,
        holding_max=4.0,
        deadline_min=30.0,
        deadline_max=60.0,
        size_min_mb=1.0,
        size_max_mb=5.0,
        edge_cost_min=0.5,
        edge_cost_max=5.0,
        waste_coef=0.8,
        team_coef=0.0,
        fs_penalty_coef=0.0,
        c_valid_action_pressure_coef=0.0,
        c_spectrum_block_penalty=0.0,
        pm_bpsk_penalty=0.0,
        r_bc_coef=0.0,
        r_se_bonus_coef=0.0,
        r_fs_usage_coef=0.0,
        r_low_se_penalty=0.0,
        reference_r_checkpoint=None,
        gamma=0.95,
        gae_lambda=0.95,
        clip_coef=0.2,
        entropy_coef=0.01,
        entropy_final_coef=None,
        entropy_decay_episodes=0,
        value_clip_coef=0.2,
        target_kl=0.03,
        max_grad_norm=0.5,
        no_advantage_norm=False,
        ppo_epochs=2,
        lr_actor=3e-4,
        lr_r_actor=None,
        lr_critic=3e-4,
        slot_bw_hz=1.25e9,
        guard_band_fs=1,
        num_slots=32,
        num_servers=2,
        num_splits=3,
        seed=42,
        device="cpu",
        log_interval=1,
        eval_freq=2,
        eval_episodes=2,
        print_eval_diagnostics=False,
        ckpt_prefix="dual_critic_smoke",
    )
    agent_c, agent_r, critic, metrics = train(args)
    assert len(metrics["episode_blocking"]) == 3
    assert len(metrics["losses_critic_c"]) == 3
    assert len(metrics["losses_critic_r"]) == 3
    assert len(metrics["losses_c"]) > 0
    assert len(metrics["losses_r"]) > 0
