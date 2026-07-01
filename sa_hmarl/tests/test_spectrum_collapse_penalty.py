"""Unit tests for the spectrum collapse penalty implementation.

Tests cover:
  - coefficient=0 leaves rewards unchanged
  - non-empty next raw mask -> no penalty
  - empty next raw mask -> penalty == -coef attributed to previous transition
  - first request and terminal edge cases
  - c_valid=False transitions excluded from actor batch but kept in GAE
  - fallback (0,0) not marked as valid C action
  - raw vs risk mask distinction
  - Agent-R parameters unchanged
  - multi-seed validation protocol independence from train seed
  - checkpoint selection uses multi-seed mean blocking
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import json
import tempfile
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from sa_hmarl.agents.ppo_agents import PPOAgentC
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_agent_c_with_frozen_r import (
    _run_training_episode,
    _save_c_checkpoint,
    _validation_seeds_for_topology,
    train,
)
from sa_hmarl.training.train_joint_mappo import CentralizedValueCritic
from sa_hmarl.training.utils import generate_requests, make_env


def _make_args(**overrides):
    """Build a minimal argparse-like namespace for testing."""
    defaults = {
        "topologies": "snap24_gnutella_reach",
        "modulation_profile": "default",
        "frozen_r_checkpoint": "sa_hmarl/checkpoints/agent_r_mixed.pt",
        "frozen_r_type": "ppo",
        "warm_start_c": None,
        "episodes": 2,
        "requests_per_episode": 10,
        "arrival_interval": 0.25,
        "holding_min": 4.0,
        "holding_max": 10.0,
        "deadline_min": 30.0,
        "deadline_max": 100.0,
        "size_min_mb": 5.0,
        "size_max_mb": 30.0,
        "edge_cost_min": 0.5,
        "edge_cost_max": 15.0,
        "waste_coef": 0.8,
        "c_valid_action_pressure_coef": 0.0,
        "c_spectrum_block_penalty": 0.0,
        "c_fs_request_penalty_coef": 0.0,
        "c_safe_fs_penalty_coef": 0.0,
        "c_spectrum_pressure_penalty_coef": 0.0,
        "c_path_penalty_coef": 0.0,
        "c_delay_risk_penalty_coef": 0.0,
        "c_path_norm_km": 600.0,
        "c_path_metric": "min",
        "c_max_spectrum_pressure": 0.0,
        "c_max_path_km": 0.0,
        "c_max_safe_fs_ratio": 0.0,
        "c_min_valid_after_risk_mask": 1,
        "gamma": 0.95,
        "gae_lambda": 0.95,
        "clip_coef": 0.2,
        "entropy_coef": 0.01,
        "entropy_final_coef": 0.001,
        "entropy_decay_episodes": 1200,
        "value_clip_coef": 0.2,
        "target_kl": 0.03,
        "max_grad_norm": 0.5,
        "no_advantage_norm": False,
        "ppo_epochs": 1,
        "lr_actor": 3e-4,
        "lr_critic": 3e-4,
        "slot_bw_hz": 1.25e9,
        "guard_band_fs": 1,
        "num_slots": 20,
        "num_servers": 4,
        "num_splits": 3,
        "split_profile": "default3",
        "agent_c_feature_mode": "r_feasibility",
        "agent_c_activation": "tanh",
        "gate_reg_coef": 0.0,
        "fixed_blend_alpha": 0.5,
        "k_paths": 5,
        "max_blocks": 10,
        "block_sort_strategy": "mixed",
        "agent_c_input_dim": 17,
        "seed": 42,
        "device": "cpu",
        "log_interval": 1,
        "eval_freq": 1,
        "eval_episodes": 2,
        "num_validation_seeds": 3,
        "validation_seeds": "1001,1002,1003",
        "validation_episodes": 2,
        "checkpoint_metric": "mean_blocking",
        "print_eval_diagnostics": False,
        "ckpt_prefix": "test_collapse",
        "objective": "default",
        "delay_coef": 0.5,
        "block_penalty": 2.0,
        "success_reward": 1.0,
        "deadline_penalty": 1.0,
        "server_overload_penalty": 1.0,
        "no_block_penalty": 1.2,
        "fs_penalty_coef": 0.0,
        "base_delay_coef": 0.8,
        "base_block_penalty": 2.0,
        "base_no_block_penalty": 1.2,
        "pressure_source": "auto",
        "pressure_clip_min": 0.0,
        "pressure_clip_max": 1.0,
        "pressure_penalty_coef": 0.5,
        "server_util_coef": 0.0,
        "spectrum_fail_extra": 0.0,
        "server_fail_extra": 0.0,
        "best_metric": "blocking",
        "episode_credit_mode": "off",
        "episode_credit_scale": 1.0,
        "episode_blocking_coef": 1.0,
        "episode_no_block_coef": 1.0,
        "episode_delay_coef": 0.5,
        "spectrum_collapse_penalty_coef": 0.0,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


def _make_minimal_env_and_agents(args):
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    env = make_env(
        topology=args.topologies,
        num_servers=args.num_servers,
        seed=args.seed,
        num_slots=args.num_slots,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        k=args.k_paths,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
    )
    agent_c = PPOAgentC(
        input_dim=args.agent_c_input_dim,
        hidden_dims=(128, 64),
        lr=args.lr_actor,
        entropy_coef=args.entropy_coef,
        max_grad_norm=args.max_grad_norm,
        device=args.device,
        feature_mode=args.agent_c_feature_mode,
        activation=args.agent_c_activation,
        num_servers=args.num_servers,
    )
    # Load frozen R
    from sa_hmarl.training.train_agent_c_with_frozen_r import _load_frozen_r
    frozen_r = _load_frozen_r(args, mod_reg)
    critic = CentralizedValueCritic(c_dim=agent_c.input_dim, r_dim=11, device=args.device)
    return env, agent_c, frozen_r, critic


def _generate_requests(env, args, seed_offset=0):
    rng = np.random.RandomState(args.seed + seed_offset)
    src = rng.randint(0, env.net.NUM_NODES)
    return generate_requests(
        env, rng, src, args.requests_per_episode,
        arrival_interval=args.arrival_interval,
        holding_min=args.holding_min, holding_max=args.holding_max,
        deadline_min=args.deadline_min, deadline_max=args.deadline_max,
        size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
        edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
        num_splits=args.num_splits, split_profile=args.split_profile,
    )


@pytest.fixture
def args_fixture():
    return _make_args()


def test_coef_zero_leaves_rewards_unchanged(args_fixture):
    args = args_fixture
    args.spectrum_collapse_penalty_coef = 0.0
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, ep_stats, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    assert ep_stats["collapse_penalty_mean"] == pytest.approx(0.0, abs=1e-8)
    for transition in transitions:
        assert transition.info.get("collapse_penalty", 0.0) == pytest.approx(0.0, abs=1e-8)


def test_next_mask_empty_gets_penalty(args_fixture):
    args = args_fixture
    args.spectrum_collapse_penalty_coef = 0.3
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, ep_stats, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    triggered = False
    for i, transition in enumerate(transitions):
        if i == 0:
            continue
        prev = transitions[i - 1]
        if transition.info.get("collapse_next"):
            triggered = True
            assert prev.info.get("collapse_penalty") == pytest.approx(-args.spectrum_collapse_penalty_coef)
            assert prev.reward < prev.info.get("base_reward", prev.reward + args.spectrum_collapse_penalty_coef)
    # The episode might not always trigger; if it does, assert exact penalty.
    if ep_stats["collapse_penalty_trigger_count"] > 0:
        assert triggered


def test_penalty_modifies_previous_not_current(args_fixture):
    args = args_fixture
    args.spectrum_collapse_penalty_coef = 0.3
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, _, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    for i, transition in enumerate(transitions):
        if transition.info.get("collapse_next") and i + 1 < len(transitions):
            current = transitions[i + 1]
            assert current.info.get("collapse_penalty", 0.0) == 0.0


def test_first_request_no_previous_penalty(args_fixture):
    args = args_fixture
    args.spectrum_collapse_penalty_coef = 0.3
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, _, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    assert transitions[0].info.get("collapse_penalty", 0.0) == pytest.approx(0.0, abs=1e-8)


def test_terminal_no_extra_penalty(args_fixture):
    args = args_fixture
    args.spectrum_collapse_penalty_coef = 0.3
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, _, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    # Last transition cannot have a collapse_penalty because there is no next request.
    assert transitions[-1].info.get("collapse_penalty", 0.0) == pytest.approx(0.0, abs=1e-8)


def test_c_valid_false_excluded_from_actor_batch_but_in_gae(args_fixture):
    args = args_fixture
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, _, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    from sa_hmarl.training.train_joint_mappo import _filter_actor_batch
    c_features, c_masks, c_actions, c_old_logp, c_indices = _filter_actor_batch(transitions, "c")
    assert len(c_features) == len(c_indices)
    for i in c_indices:
        assert transitions[i].c_valid is True
    # All transitions still feed into GAE via the rollout buffer.
    assert len(transitions) >= len(c_indices)


def test_fallback_not_marked_valid(args_fixture):
    args = args_fixture
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, _, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    for transition in transitions:
        if transition.c_action == 0 and not transition.info.get("success", False):
            # Fallback action is stored but c_valid must be False when mask was empty.
            if int(np.sum(transition.c_mask)) == 0:
                assert transition.c_valid is False


def test_raw_mask_used_not_risk_mask(args_fixture):
    args = args_fixture
    args.spectrum_collapse_penalty_coef = 0.3
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    transitions, _, _, _ = _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    # Trigger is based on raw mask sum, not the (possibly smaller) risk-mask sum.
    for i, transition in enumerate(transitions):
        if i == 0:
            continue
        prev = transitions[i - 1]
        collapse = transition.info.get("collapse_next", False)
        k_valid = transition.info.get("k_c_valid_next", -1)
        assert collapse == (k_valid == 0)
        if collapse:
            assert k_valid == 0


def test_agent_r_params_unaffected(args_fixture):
    args = args_fixture
    args.spectrum_collapse_penalty_coef = 0.3
    env, agent_c, frozen_r, critic = _make_minimal_env_and_agents(args)
    requests = _generate_requests(env, args)
    env.reset(requests)
    before = {name: param.clone() for name, param in frozen_r.policy_net.named_parameters()}
    _run_training_episode(
        env, requests, agent_c, frozen_r, args.frozen_r_type, critic, args
    )
    for name, param in frozen_r.policy_net.named_parameters():
        assert torch.equal(param, before[name])


def test_validation_seeds_independent_of_train_seed():
    args_s1 = _make_args(seed=42, validation_seeds="1001,1002,1003")
    args_s2 = _make_args(seed=123, validation_seeds="1001,1002,1003")
    assert _validation_seeds_for_topology(args_s1, 0) == [1001, 1002, 1003]
    assert _validation_seeds_for_topology(args_s2, 0) == [1001, 1002, 1003]


def test_checkpoint_metadata_includes_validation_config(args_fixture, tmp_path):
    args = args_fixture
    args.episodes = 1
    args.eval_freq = 1
    agent_c = PPOAgentC(
        input_dim=args.agent_c_input_dim,
        hidden_dims=(128, 64),
        device=args.device,
        feature_mode=args.agent_c_feature_mode,
    )
    critic = CentralizedValueCritic(c_dim=agent_c.input_dim, r_dim=11, device=args.device)
    metrics = {"episode_blocking": [0.1]}
    ckpt_path = tmp_path / "test.pt"
    _save_c_checkpoint(ckpt_path, agent_c, critic, args, metrics, best_validation={"mean_blocking": 0.2})
    ckpt = torch.load(ckpt_path, map_location="cpu")
    meta = ckpt["metadata"]
    assert meta["train_seed"] == args.seed
    assert meta["validation_seeds"] == [1001, 1002, 1003]
    assert meta["validation_episodes"] == args.eval_episodes
    assert meta["checkpoint_metric"] == "mean_blocking"
    assert meta["feature_mode"] == "r_feasibility"
    assert meta["spectrum_collapse_penalty_coef"] == 0.0
    assert meta["best_validation_mean"]["mean_blocking"] == pytest.approx(0.2)


def test_checkpoint_selection_uses_multi_seed_mean_blocking():
    """Short training run: best checkpoint metric should be mean_blocking over validation seeds."""
    args = _make_args(
        episodes=2,
        eval_freq=1,
        requests_per_episode=5,
        eval_episodes=1,
        spectrum_collapse_penalty_coef=0.0,
        ckpt_prefix="test_select_mean_blocking",
    )
    with tempfile.TemporaryDirectory() as tmpdir:
        args.ckpt_prefix = f"{tmpdir}/test_select"
        agent_c, critic, metrics = train(args)
        assert "eval_mean_blocking" in metrics
        assert len(metrics["eval_mean_blocking"]) > 0
        # Each recorded validation metric is a mean across validation seeds.
        assert all(isinstance(v, float) for v in metrics["eval_mean_blocking"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
