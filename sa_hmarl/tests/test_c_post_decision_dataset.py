"""Tests for the C-side post-decision supervised dataset generator."""
from __future__ import annotations

import copy

import numpy as np
import torch

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.evaluation.diagnose_c_action_horizon_oracle import (
    _execute_c_with_frozen_r,
    _snapshot_before_c_decision,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import (
    _demand_aware_potential_spectrum_only,
)
from sa_hmarl.evaluation.generate_c_post_decision_dataset import (
    FEATURE_NAMES,
    _atomic_save_npz,
    _candidate_feature_vector,
    _concat_shards,
    _dataset_diagnostics,
    _delay_aware_label,
    _features_for_post_decision,
    _generate_episode,
    _group_id,
    _horizon_delay_aware_label,
    _rank_desc,
    _should_generate_shard,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


C_CKPT = "sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt"
R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"


def _fixture(num_requests=4):
    env = make_env(
        "snap24_gnutella_reach", 20, 4, 42,
        max_blocks=10, block_sort_strategy="mixed", k=5,
    )
    rng = np.random.RandomState(42)
    requests = generate_requests(
        env, rng, 0, num_requests, arrival_interval=0.15,
        holding_min=4, holding_max=10, deadline_min=30, deadline_max=100,
        size_min_mb=5, size_max_mb=30, edge_cost_min=0.5, edge_cost_max=15,
        num_splits=3, split_profile="default3",
    )
    agent_c = _load_ppo_c(C_CKPT, "cpu")
    agent_r = _load_ppo_r(R_CKPT, ModulationRegistry.from_profile("default"), "cpu")
    return env, requests, agent_c, agent_r


def test_feature_schema_excludes_label_fields():
    forbidden = {"explicit_phi_before", "explicit_phi_after", "delta_phi", "label_rank_in_group"}
    assert forbidden.isdisjoint(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES)) == 50


def test_group_ids_are_unique_across_splits():
    ids = {
        _group_id(split, 42, 0, 0)
        for split in ("train", "val", "test")
    }
    assert len(ids) == 3


def test_rank_desc_is_group_local_and_deterministic():
    values = np.array([0.2, 0.8, 0.5], dtype=np.float32)
    assert _rank_desc(values).tolist() == [3, 1, 2]
    assert _rank_desc(values).tolist() == _rank_desc(values).tolist()


def test_candidate_features_are_finite_and_label_free():
    env, requests, agent_c, _ = _fixture(1)
    env.reset(requests)
    req = requests[0]
    env.advance_time(req.arrival_time)
    obs = build_agent_c_observation(env, req)
    features, _ = agent_c.build_action_features(obs)
    candidate_idx = int(np.flatnonzero(obs["agent_c_mask"])[0])
    split_id, server_id = divmod(candidate_idx, 4)
    vector = _candidate_feature_vector(
        env, req, obs, features, candidate_idx, split_id, server_id
    )
    assert vector.shape == (50,)
    assert vector.dtype == np.float32
    assert np.all(np.isfinite(vector))


def test_post_decision_features_wrap_default_policy_features():
    env, requests, _, _ = _fixture(1)
    env.reset(requests)
    req = requests[0]
    env.advance_time(req.arrival_time)
    obs = build_agent_c_observation(env, req)
    agent_c = type("DefaultFeatureAgent", (), {
        "build_action_features": AgentC.build_action_features,
        "feature_mode": "default",
        "ablation": False,
        "zero_spectrum": False,
    })()
    features = _features_for_post_decision(agent_c, obs)
    assert features.shape == (len(obs["candidate_features"]), 27)


def test_delay_aware_label_penalizes_current_delay():
    req = type("Req", (), {"deadline_ms": 100.0})()
    fast = _delay_aware_label(3.0, {"success": True, "delay_ms": 10.0}, req, 1.0)
    slow = _delay_aware_label(3.0, {"success": True, "delay_ms": 50.0}, req, 1.0)
    assert fast > slow


def test_horizon_label_penalizes_future_blocking_and_delay():
    req = type("Req", (), {"deadline_ms": 100.0})()
    good = _horizon_delay_aware_label(
        3.0, {"success": True, "delay_ms": 10.0}, req,
        {"blocked": 0}, 3,
        phi_coef=0.1, current_block_coef=1.0,
        future_block_coef=2.0, delay_coef=0.5,
    )
    future_bad = _horizon_delay_aware_label(
        3.0, {"success": True, "delay_ms": 10.0}, req,
        {"blocked": 2}, 3,
        phi_coef=0.1, current_block_coef=1.0,
        future_block_coef=2.0, delay_coef=0.5,
    )
    slow = _horizon_delay_aware_label(
        3.0, {"success": True, "delay_ms": 80.0}, req,
        {"blocked": 0}, 3,
        phi_coef=0.1, current_block_coef=1.0,
        future_block_coef=2.0, delay_coef=0.5,
    )
    current_fail = _horizon_delay_aware_label(
        3.0, {"success": False}, req,
        {"blocked": 0}, 3,
        phi_coef=0.1, current_block_coef=1.0,
        future_block_coef=2.0, delay_coef=0.5,
    )
    assert good > future_bad
    assert good > slow
    assert good > current_fail


def test_label_matches_existing_spectrum_only_potential():
    env, requests, agent_c, agent_r = _fixture(1)
    data = _generate_episode(env, requests, agent_c, agent_r, "train", 42, 0, 12, 3, 0.3, 0.0)

    check_env, check_requests, _, check_r = _fixture(1)
    check_env.reset(check_requests)
    req = check_requests[0]
    check_env.advance_time(req.arrival_time)
    obs = build_agent_c_observation(check_env, req)
    candidate_idx = int(np.flatnonzero(obs["agent_c_mask"])[0])
    snapshot = _snapshot_before_c_decision(check_env, req.req_id)
    branch = copy.deepcopy(snapshot)
    _execute_c_with_frozen_r(branch, req, candidate_idx, check_r, 4)
    expected = _demand_aware_potential_spectrum_only(branch, [req], 12, 3, 0.3)
    row = np.flatnonzero(
        (data["candidate_actions"][:, 0] == candidate_idx // 4)
        & (data["candidate_actions"][:, 1] == candidate_idx % 4)
    )[0]
    assert data["labels"][row] == np.float32(expected)


def test_all_candidates_leave_predecision_snapshot_unchanged():
    env, requests, _, agent_r = _fixture(1)
    env.reset(requests)
    req = requests[0]
    env.advance_time(req.arrival_time)
    obs = build_agent_c_observation(env, req)
    snapshot = _snapshot_before_c_decision(env, req.req_id)
    before_slots = {key: value.copy() for key, value in snapshot.net.link_states.items()}
    for candidate_idx in np.flatnonzero(obs["agent_c_mask"])[:2]:
        branch = copy.deepcopy(snapshot)
        _execute_c_with_frozen_r(branch, req, int(candidate_idx), agent_r, 4)
    assert all(np.array_equal(before_slots[key], snapshot.net.link_states[key]) for key in before_slots)


def test_generator_is_reproducible_and_agent_r_frozen():
    env1, requests1, c1, r1 = _fixture(2)
    before = {key: value.clone() for key, value in r1.policy_net.state_dict().items()}
    data1 = _generate_episode(env1, requests1, c1, r1, "train", 42, 0, 12, 2, 0.3, 0.0)
    env2, requests2, c2, r2 = _fixture(2)
    data2 = _generate_episode(env2, requests2, c2, r2, "train", 42, 0, 12, 2, 0.3, 0.0)
    assert np.array_equal(data1["features"], data2["features"])
    assert np.array_equal(data1["labels"], data2["labels"])
    assert all(torch.equal(value, before[key]) for key, value in r1.policy_net.state_dict().items())


def test_output_shapes_and_dtypes():
    env, requests, agent_c, agent_r = _fixture(2)
    data = _generate_episode(env, requests, agent_c, agent_r, "train", 42, 0, 12, 2, 0.3, 0.0)
    n = len(data["labels"])
    assert data["features"].shape == (n, 50)
    assert data["candidate_actions"].shape == (n, 2)
    assert data["features"].dtype == np.float32
    assert data["group_ids"].dtype == np.int64
    assert len(data["all_group_ids"]) == 2


def test_atomic_shards_round_trip_and_concat(tmp_path):
    first = tmp_path / "a.npz"
    second = tmp_path / "b.npz"
    _atomic_save_npz(first, x=np.array([[1]], dtype=np.float32))
    _atomic_save_npz(second, x=np.array([[2]], dtype=np.float32))
    assert not (tmp_path / "a.npz.tmp").exists()
    merged = _concat_shards([first, second])
    assert merged["x"].reshape(-1).tolist() == [1.0, 2.0]


def test_resume_skips_existing_shard(tmp_path):
    shard = tmp_path / "episode.npz"
    assert _should_generate_shard(shard, resume=True)
    _atomic_save_npz(shard, x=np.array([1], dtype=np.float32))
    assert not _should_generate_shard(shard, resume=True)
    assert _should_generate_shard(shard, resume=False)


def test_generator_passes_only_current_and_past_history(monkeypatch):
    import sa_hmarl.evaluation.generate_c_post_decision_dataset as generator

    seen_history_lengths = []

    def fake_potential(env, request_history, window, probe_limit, alpha):
        seen_history_lengths.append(len(request_history))
        return float(len(request_history))

    monkeypatch.setattr(generator, "_demand_aware_potential_spectrum_only", fake_potential)
    env, requests, agent_c, agent_r = _fixture(2)
    _generate_episode(env, requests, agent_c, agent_r, "train", 42, 0, 12, 2, 0.3, 0.0)
    assert seen_history_lengths
    assert min(seen_history_lengths) == 1
    assert max(seen_history_lengths) == 2


def test_diagnostics_report_nonconstant_group_labels():
    data = {
        "labels": np.array([1.0, 2.0], dtype=np.float32),
        "group_ids": np.array([10, 10], dtype=np.int64),
        "all_group_ids": np.array([10], dtype=np.int64),
        "group_raw_mask_empty": np.array([False]),
        "current_success": np.array([True, True]),
        "selected_by_ppo": np.array([True, False]),
        "selected_by_explicit_potential": np.array([False, True]),
        "features": np.ones((2, 50), dtype=np.float32),
        "candidate_actions": np.array([[0, 0], [1, 0]], dtype=np.int64),
    }
    diag = _dataset_diagnostics(data)
    assert diag["nonzero_label_range_rate"] == 1.0
    assert diag["top1_gap_mean"] == 1.0
    assert diag["ppo_explicit_top1_agreement"] == 0.0
