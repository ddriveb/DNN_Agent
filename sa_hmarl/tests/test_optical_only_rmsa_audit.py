"""Unit tests for the optical-only R-side discriminative audit.

These tests are hard gates.  They must pass before micro, calibration, smoke,
or pilot phases can be executed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import pytest
import torch

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.evaluation.optical_only_rmsa_env import (
    DEFAULT_BITRATE_MAX_GBPS,
    DEFAULT_BITRATE_MIN_GBPS,
    DEFAULT_GUARD_BAND_FS,
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_SLOT_BW_HZ,
    ODRequest,
    OpticalOnlyRMSAEnv,
    generate_od_requests,
)
from sa_hmarl.evaluation.optical_only_rmsa_evaluator import (
    RANKER_CKPTS,
    build_transfer_features,
    load_ppo_r,
    load_ranker,
    run_episode,
    select_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def tiny_config() -> Dict[str, Any]:
    return {
        "topology": "xlron_cost239_ptrnet_real",
        "num_slots": 100,
        "k_paths": 50,
        "max_blocks": 10,
        "path_sort_strategy": "hops",
        "block_sort_strategy": "start_asc",
        "modulation_profile": "default",
        "slot_bw_hz": DEFAULT_SLOT_BW_HZ,
        "guard_band_fs": DEFAULT_GUARD_BAND_FS,
        "mean_holding_time": DEFAULT_MEAN_HOLDING_TIME,
        "bitrate_min_gbps": DEFAULT_BITRATE_MIN_GBPS,
        "bitrate_max_gbps": DEFAULT_BITRATE_MAX_GBPS,
    }


@pytest.fixture
def env(tiny_config: Dict[str, Any]) -> OpticalOnlyRMSAEnv:
    return OpticalOnlyRMSAEnv(
        topology=tiny_config["topology"],
        num_slots=tiny_config["num_slots"],
        k_paths=tiny_config["k_paths"],
        max_blocks=tiny_config["max_blocks"],
        path_sort_strategy=tiny_config["path_sort_strategy"],
        block_sort_strategy=tiny_config["block_sort_strategy"],
        mod_registry=ModulationRegistry.from_profile(tiny_config["modulation_profile"]),
        slot_bw_hz=tiny_config["slot_bw_hz"],
        guard_band_fs=tiny_config["guard_band_fs"],
        seed=42,
    )


@pytest.fixture
def small_requests(tiny_config: Dict[str, Any]) -> list[ODRequest]:
    rng = np.random.RandomState(123)
    num_nodes = 11
    return generate_od_requests(
        num_nodes=num_nodes,
        rng=rng,
        num_requests=64,
        arrival_interval=1.0,
        mean_holding_time=tiny_config["mean_holding_time"],
        bitrate_min_gbps=tiny_config["bitrate_min_gbps"],
        bitrate_max_gbps=tiny_config["bitrate_max_gbps"],
    )


@pytest.fixture
def agent_r(tiny_config: Dict[str, Any]):
    ckpt_path = "sa_hmarl/checkpoints/agent_r_mixed.pt"
    if not Path(ckpt_path).exists():
        pytest.skip(f"PPO-R checkpoint not found: {ckpt_path}")
    return load_ppo_r(ckpt_path, ModulationRegistry.from_profile(tiny_config["modulation_profile"]), device="cpu")


@pytest.fixture
def rankers():
    loaded = {}
    for name, path in RANKER_CKPTS.items():
        if not Path(path).exists():
            pytest.skip(f"Ranker checkpoint not found: {path}")
        loaded[name] = load_ranker(path, device="cpu")
    return loaded


# ---------------------------------------------------------------------------
# Request trace tests
# ---------------------------------------------------------------------------
def test_request_has_no_c_fields(small_requests: list[ODRequest]):
    for req in small_requests:
        assert isinstance(req, ODRequest)
        assert req.src_node != req.dst_node
        assert 25 <= req.bitrate_gbps <= 100
        assert req.arrival_time >= 0
        assert req.holding_time > 0
        assert not hasattr(req, "splits")
        assert not hasattr(req, "deadline_ms")
        assert not hasattr(req, "server_id")


def test_required_fs_formula(env: OpticalOnlyRMSAEnv, tiny_config: Dict[str, Any]):
    req = ODRequest(0, 0, 1, 100, 0.0, 10.0)
    mod_idx = env.mod_reg.names.index("16QAM")  # SE=4, reach=500 km
    mod = env.mod_reg[mod_idx]
    data_fs = int(np.ceil((100e9) / (tiny_config["slot_bw_hz"] * 4)))
    expected = data_fs + tiny_config["guard_band_fs"]
    assert req.required_fs(mod, tiny_config["slot_bw_hz"], tiny_config["guard_band_fs"]) == expected


# ---------------------------------------------------------------------------
# Environment observation tests
# ---------------------------------------------------------------------------
def test_observation_is_optical_only(env: OpticalOnlyRMSAEnv, small_requests: list[ODRequest]):
    env.reset(small_requests)
    req = small_requests[0]
    env.advance_time(req.arrival_time)
    obs = env.build_observation(req)

    assert "c_context" not in obs
    assert "server_utilizations" not in obs
    assert "candidate_features" not in obs
    assert obs["src_node"] == req.src_node
    assert obs["dst_node"] == req.dst_node
    assert len(obs["candidate_paths"]) > 0
    assert len(obs["candidate_paths"]) <= env.k_paths
    assert len(obs["path_features"]) == len(obs["candidate_paths"])

    num_mods = env.mod_reg.num_formats
    expected_mask_len = len(obs["candidate_paths"]) * num_mods * env.max_blocks
    assert obs["agent_r_mask"].shape == (expected_mask_len,)


def test_ksp_ff_highest_selects_legal_action(env: OpticalOnlyRMSAEnv, small_requests: list[ODRequest]):
    env.reset(small_requests)
    legal_count = 0
    admitted = 0
    for req in small_requests:
        env.advance_time(req.arrival_time)
        obs = env.build_observation(req)
        action = ksp_ff_highest_mod_action(obs)
        if action is not None:
            legal_count += 1
            info = env.step(action, obs, req.holding_time)
            if info["success"]:
                admitted += 1
    assert legal_count > 0 or admitted > 0 or len(small_requests) < 10


def test_resource_conservation_after_drain(env: OpticalOnlyRMSAEnv, small_requests: list[ODRequest]):
    env.reset(small_requests)
    for req in small_requests:
        env.advance_time(req.arrival_time)
        obs = env.build_observation(req)
        action = ksp_ff_highest_mod_action(obs)
        if action is not None:
            env.step(action, obs, req.holding_time)
    max_release = max(r.arrival_time + r.holding_time for r in small_requests)
    env.advance_time(max_release + 1.0)
    assert env.get_utilization() == pytest.approx(0.0, abs=1e-9)
    assert len(env.active_connections) == 0


# ---------------------------------------------------------------------------
# Model loading tests
# ---------------------------------------------------------------------------
def test_ppo_r_loads(agent_r):
    assert agent_r is not None
    assert agent_r.input_dim == 11
    assert hasattr(agent_r, "policy_net")
    assert next(agent_r.policy_net.parameters()).requires_grad is False


def test_rankers_load(rankers):
    for name, ranker in rankers.items():
        assert ranker["input_dim"] in (25, 41, 16)
        assert ranker["feature_mean"].shape == (ranker["input_dim"],)
        assert ranker["feature_std"].shape == (ranker["input_dim"],)
        assert len(ranker["feature_names"]) == ranker["input_dim"]
        assert next(ranker["model"].parameters()).requires_grad is False


# ---------------------------------------------------------------------------
# Transfer feature adapter tests
# ---------------------------------------------------------------------------
def test_transfer_features_finite(env: OpticalOnlyRMSAEnv, small_requests: list[ODRequest], agent_r, rankers):
    env.reset(small_requests)
    req = small_requests[0]
    env.advance_time(req.arrival_time)
    obs = env.build_observation(req)
    mask = obs["agent_r_mask"]
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        pytest.skip("No legal actions for the first request")
    candidates = [int(legal[0]), int(legal[-1])]
    for name, ranker in rankers.items():
        x = build_transfer_features(obs, env, req, candidates, ranker)
        assert x.shape == (len(candidates), ranker["input_dim"])
        assert np.all(np.isfinite(x))


def test_c_only_features_normalized_to_zero(env: OpticalOnlyRMSAEnv, small_requests: list[ODRequest], rankers):
    from sa_hmarl.evaluation.optical_only_rmsa_evaluator import normalize_transfer_features
    env.reset(small_requests)
    req = small_requests[0]
    env.advance_time(req.arrival_time)
    obs = env.build_observation(req)
    mask = obs["agent_r_mask"]
    legal = np.flatnonzero(mask)
    if legal.size == 0:
        pytest.skip("No legal actions for the first request")
    candidates = [int(legal[0])]
    for name, ranker in rankers.items():
        x = build_transfer_features(obs, env, req, candidates, ranker)
        x_norm = normalize_transfer_features(x, ranker)
        for i, feat_name in enumerate(ranker["feature_names"]):
            if feat_name in {
                "split_norm", "server_norm", "deadline_norm",
                "intermediate_size_norm", "server_utilization",
                "k_c_valid_ratio", "k_r_total_ratio", "phi_spec_norm",
                "server_util_context", "server_margin_context",
            }:
                assert x_norm[0, i] == pytest.approx(0.0, abs=1e-6), f"{name}:{feat_name} should normalize to zero"


# ---------------------------------------------------------------------------
# Action-selection dispatch tests
# ---------------------------------------------------------------------------
def test_select_r_action_legal(env: OpticalOnlyRMSAEnv, small_requests: list[ODRequest], agent_r, rankers):
    env.reset(small_requests)
    req = small_requests[0]
    env.advance_time(req.arrival_time)
    obs = env.build_observation(req)
    for r_mode in ("ksp_ff_highest", "ppo_r_top1", "strict_v13", "v135_afterstate", "v135_afterstate_explicit"):
        action, valid, info = select_r_action(r_mode, obs, env, req, agent_r, rankers)
        if valid:
            num_mods = len(obs["mod_names"])
            p, m, b = decode_agent_r_action(action, num_mods, env.max_blocks)
            assert 0 <= p < len(obs["candidate_paths"])
            assert 0 <= m < num_mods
            assert 0 <= b < env.max_blocks
            assert obs["agent_r_mask"][action]


# ---------------------------------------------------------------------------
# Episode integration tests
# ---------------------------------------------------------------------------
def test_run_episode_ksp_ff(env: OpticalOnlyRMSAEnv, tiny_config: Dict[str, Any]):
    rng = np.random.RandomState(7)
    requests = generate_od_requests(
        num_nodes=11,
        rng=rng,
        num_requests=200,
        arrival_interval=0.05,
        mean_holding_time=tiny_config["mean_holding_time"],
        bitrate_min_gbps=tiny_config["bitrate_min_gbps"],
        bitrate_max_gbps=tiny_config["bitrate_max_gbps"],
    )
    result = run_episode(7, requests, "ksp_ff_highest", None, {}, tiny_config, warmup_requests=50)
    assert result["schema_version"] == "optical-only-rmsa-audit-v1.0"
    assert result["evaluated_requests"] == 150
    assert result["admitted"] + result["blocked"] == 150
    assert result["r_no_valid_action"] == result["blocked"]
    assert result["final_active_connections"] == 0
    assert result["final_utilization"] == pytest.approx(0.0, abs=1e-9)


def test_run_episode_rankers_produce_results(env: OpticalOnlyRMSAEnv, tiny_config: Dict[str, Any], agent_r, rankers):
    rng = np.random.RandomState(8)
    requests = generate_od_requests(
        num_nodes=11,
        rng=rng,
        num_requests=200,
        arrival_interval=0.05,
        mean_holding_time=tiny_config["mean_holding_time"],
        bitrate_min_gbps=tiny_config["bitrate_min_gbps"],
        bitrate_max_gbps=tiny_config["bitrate_max_gbps"],
    )
    for r_mode in ("ppo_r_top1", "strict_v13", "v135_afterstate", "v135_afterstate_explicit"):
        result = run_episode(8, requests, r_mode, agent_r, rankers, tiny_config, warmup_requests=50)
        assert result["evaluated_requests"] == 150
        assert result["admitted"] + result["blocked"] == 150
        assert result["r_no_valid_action"] == result["blocked"]


def test_episode_results_no_c_fields(env: OpticalOnlyRMSAEnv, tiny_config: Dict[str, Any]):
    rng = np.random.RandomState(9)
    requests = generate_od_requests(
        num_nodes=11,
        rng=rng,
        num_requests=100,
        arrival_interval=0.05,
        mean_holding_time=tiny_config["mean_holding_time"],
        bitrate_min_gbps=tiny_config["bitrate_min_gbps"],
        bitrate_max_gbps=tiny_config["bitrate_max_gbps"],
    )
    result = run_episode(9, requests, "ksp_ff_highest", None, {}, tiny_config, warmup_requests=20)
    for action in result["selected_actions"]:
        assert "server_id" not in action
        assert "split_id" not in action
        assert "deadline_failure" not in action
        assert "c_no_valid_action" not in action
