"""Unit tests for bottleneck/release/hard-pair diagnostic."""
from __future__ import annotations

import copy

import numpy as np
import pytest

from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
from sa_hmarl.evaluation.diagnose_r_bottleneck_hardpair_signals import (
    CandidateRecord,
    _edges_of_path,
    _edge_array_stats,
    _hard_pairs,
    _path_overlap,
    _post_allocation_edge_stats,
    _release_features,
)
from sa_hmarl.training.utils import generate_requests, make_env


def _minimal_env_and_obs():
    """Create a small env with one request and build an R observation."""
    env = make_env(
        topology="snap24_gnutella_reach",
        num_slots=24,
        num_servers=4,
        seed=42,
        modulation_profile="default",
        max_blocks=10,
        block_sort_strategy="mixed",
        k=5,
    )
    rng = np.random.RandomState(42)
    src = 0
    requests = generate_requests(
        env, rng, src, num_requests=5,
        arrival_interval=0.09, holding_min=4.0, holding_max=14.0,
        deadline_min=30.0, deadline_max=100.0,
        size_min_mb=5.0, size_max_mb=30.0,
        edge_cost_min=0.5, edge_cost_max=15.0,
        num_splits=3, split_profile="default3",
    )
    env.reset(requests)
    req = requests[0]
    env.advance_time(req.arrival_time)
    # Use split 1, server 0 as a representative pair.
    obs_r = build_agent_r_observation(env, req, split_id=1, server_id=0)
    return env, req, obs_r


def test_edges_of_path():
    path = [0, 3, 7]
    edges = _edges_of_path(path)
    assert edges == [(0, 3), (3, 7)]


def test_bottleneck_features_finite():
    env, req, obs_r = _minimal_env_and_obs()
    legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()
    assert legal, "Need at least one legal R action for this test"
    r_idx = legal[0]
    num_mods = len(obs_r["mod_names"])
    path_idx, mod_idx, block_idx = decode_agent_r_action(r_idx, num_mods, env.max_blocks)
    path = obs_r["candidate_paths"][path_idx]

    stats = _edge_array_stats(env, path)
    assert stats["utils"].shape == stats["lfbs"].shape
    assert np.all(np.isfinite(stats["utils"]))
    assert np.all(stats["utils"] >= 0.0) and np.all(stats["utils"] <= 1.0)
    assert np.all(stats["free_ratios"] >= 0.0) and np.all(stats["free_ratios"] <= 1.0)
    assert np.all(stats["lfbs"] >= 0.0) and np.all(stats["lfbs"] <= env.net.num_slots)

    overlap_count, overlap_ratio = _path_overlap(env, path)
    assert isinstance(overlap_count, int)
    assert 0.0 <= overlap_ratio <= 1.0


def test_release_features_finite():
    env, req, obs_r = _minimal_env_and_obs()
    legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()
    assert legal
    r_idx = legal[0]
    num_mods = len(obs_r["mod_names"])
    path_idx, _, _ = decode_agent_r_action(r_idx, num_mods, env.max_blocks)
    path = obs_r["candidate_paths"][path_idx]

    feats = _release_features(env, path, arrival_interval=0.09)
    for k, v in feats.items():
        assert np.isfinite(v), f"{k} is not finite"
        assert v >= 0.0, f"{k} is negative"


def test_post_allocation_no_pollution():
    env, req, obs_r = _minimal_env_and_obs()
    legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()
    assert legal
    r_idx = legal[0]
    num_mods = len(obs_r["mod_names"])
    path_idx, mod_idx, block_idx = decode_agent_r_action(r_idx, num_mods, env.max_blocks)
    path = obs_r["candidate_paths"][path_idx]

    pre = {e: env.net.link_states[e].copy() for e in _edges_of_path(path)}
    branch = copy.deepcopy(env)
    _, _, _, info = branch.step((1, 0), (path_idx, mod_idx, block_idx))
    assert info.get("success", False) is True, "Expected the legal action to succeed"

    post_branch = {e: branch.net.link_states[e].copy() for e in _edges_of_path(path)}
    for e in pre:
        assert np.array_equal(pre[e], env.net.link_states[e]), "Original env was mutated"
        assert not np.array_equal(pre[e], post_branch[e]), "Branch should differ after allocation"

    post_stats = _post_allocation_edge_stats(branch, path)
    assert np.all(np.isfinite(post_stats["lfbs"]))


def test_hard_pair_filter_logic():
    base = CandidateRecord(seed=0, episode=0, request=0, req_id=0, split_id=0, server_id=0,
                           r_action_idx=0, path_idx=0, mod_idx=0, block_idx=0)
    base.current_success = True
    base.delay_ms = 5.0
    base.block_waste = 2.0
    base.future_blocked_count = 0
    base.future_nsb_count = 0
    base.G_v12 = -1.0

    # Identical action -> not a hard pair.
    a = copy.deepcopy(base)
    b = copy.deepcopy(base)
    assert _hard_pairs([a, b], return_gap_threshold=0.1) == []

    # Same current metrics but different future blocking -> hard pair.
    b.future_blocked_count = 1
    b.G_v12 = -5.0
    pairs = _hard_pairs([a, b], return_gap_threshold=0.1)
    assert len(pairs) == 1

    # Future differs but delay is too far apart -> not hard pair.
    b2 = copy.deepcopy(b)
    b2.delay_ms = 10.0
    pairs = _hard_pairs([a, b2], return_gap_threshold=0.1)
    assert len(pairs) == 0

    # Block waste too far apart -> not hard pair.
    b3 = copy.deepcopy(b)
    b3.delay_ms = 5.0
    b3.block_waste = 10.0
    pairs = _hard_pairs([a, b3], return_gap_threshold=0.1)
    assert len(pairs) == 0

    # Different states should not pair.
    c = copy.deepcopy(b)
    c.seed = 1
    pairs = _hard_pairs([a, c], return_gap_threshold=0.1)
    assert len(pairs) == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
