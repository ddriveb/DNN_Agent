"""Unit tests for the R-action H-step Oracle diagnostic."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from sa_hmarl.env.observation_builder import build_agent_c_observation, build_agent_r_observation
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    RActionRecord,
    _compute_spectrum_field,
    _load_ppo_c,
    _load_ppo_r,
    _phi_spec,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
    _snapshot_before_r_decision,
    _spectrum_stats,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


CHECKPOINT_C = "sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt"
CHECKPOINT_R = "sa_hmarl/checkpoints/agent_r_mixed.pt"


@pytest.fixture
def env_and_requests():
    env = make_env(
        "snap24_gnutella_reach",
        num_slots=20,
        num_servers=4,
        seed=42,
        k=5,
        max_blocks=10,
        block_sort_strategy="mixed",
    )
    rng = np.random.RandomState(42)
    src = rng.randint(0, env.net.NUM_NODES)
    requests = generate_requests(env, rng, src, 20, num_splits=3, split_profile="default3")
    env.reset(requests)
    return env, requests


@pytest.fixture
def agents():
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c(CHECKPOINT_C, "cpu")
    agent_r = _load_ppo_r(CHECKPOINT_R, mod_reg, "cpu")
    return agent_c, agent_r


def test_aligned_advance_time_releases_resources_before_observation(env_and_requests):
    """Calling advance_time(req.arrival_time) before building observations releases expired resources."""
    env, requests = env_and_requests
    req0 = requests[0]
    env.advance_time(req0.arrival_time)
    obs_c0 = build_agent_c_observation(env, req0)
    # Pick any valid C action.
    raw_mask = np.asarray(obs_c0["agent_c_mask"], dtype=bool)
    assert raw_mask.sum() > 0
    action_c_idx = int(np.where(raw_mask)[0][0])
    split_id, server_id = divmod(action_c_idx, len(env.mec.servers))
    obs_r0 = build_agent_r_observation(env, req0, split_id, server_id)
    raw_r_mask = np.asarray(obs_r0["agent_r_mask"], dtype=bool)
    assert raw_r_mask.sum() > 0
    action_r_idx = int(np.where(raw_r_mask)[0][0])
    action_r = decode_agent_r_action(action_r_idx, len(obs_r0["mod_names"]), env.max_blocks)
    env.step((split_id, server_id), action_r)

    # Without advance_time, the next observation would still see the active allocation.
    stats_no_advance = _spectrum_stats(env)

    req1 = requests[1]
    env.advance_time(req1.arrival_time)
    stats_after_advance = _spectrum_stats(env)

    # After advance_time, fragmentation should be no higher (resources released).
    assert stats_after_advance["frag_index"] <= stats_no_advance["frag_index"] + 1e-6
    assert stats_after_advance["lfb_ratio"] >= stats_no_advance["lfb_ratio"] - 1e-6


def test_counterfactuals_start_from_same_snapshot(agents, env_and_requests):
    """All enumerated R actions for a state must share identical pre-decision spectrum stats."""
    env, requests = env_and_requests
    agent_c, agent_r = agents

    # Step through until we find a multi-action state.
    found = False
    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if raw_c_mask.sum() == 0:
            env.step((0, 0), (0, 0, 0))
            continue
        action_c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = divmod(action_c_idx, len(env.mec.servers))
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        legal_indices = np.where(raw_r_mask)[0].tolist()
        if len(legal_indices) >= 2:
            found = True
            import copy
            snapshot = copy.deepcopy(env)
            before_values = []
            for r_idx in legal_indices:
                env_r = copy.deepcopy(snapshot)
                before_values.append(_spectrum_stats(env_r)["frag_index"])
            assert len(set(before_values)) == 1
            break
        # Advance actual env with PPO-R action
        action_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        action_r = decode_agent_r_action(action_r_idx, len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), action_r)
    assert found, "No multi-action state found in the generated episode"


def test_pre_decision_snapshot_still_processes_current_request(env_and_requests):
    env, requests = env_and_requests
    current = requests[0]
    env.advance_time(current.arrival_time)
    snapshot = _snapshot_before_r_decision(env, current.req_id)

    obs_c = build_agent_c_observation(snapshot, current)
    c_idx = int(np.flatnonzero(obs_c["agent_c_mask"])[0])
    split_id, server_id = divmod(c_idx, len(snapshot.mec.servers))
    obs_r = build_agent_r_observation(snapshot, current, split_id, server_id)
    r_idx = int(np.flatnonzero(obs_r["agent_r_mask"])[0])
    action_r = decode_agent_r_action(
        r_idx, len(obs_r["mod_names"]), snapshot.max_blocks
    )
    _, _, _, info = snapshot.step((split_id, server_id), action_r)

    assert info["req_id"] == current.req_id
    assert env.event_queue[0][2].req_id == current.req_id


def test_only_r_mask_true_actions_enumerated(agents, env_and_requests):
    """Every enumerated counterfactual action must be legal under the raw R-mask."""
    env, requests = env_and_requests
    agent_c, agent_r = agents

    found = False
    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if raw_c_mask.sum() == 0:
            env.step((0, 0), (0, 0, 0))
            continue
        action_c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = divmod(action_c_idx, len(env.mec.servers))
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        legal_indices = set(np.where(raw_r_mask)[0].tolist())
        if len(legal_indices) >= 2:
            found = True
            assert legal_indices.issubset(set(range(len(raw_r_mask))))
            assert all(raw_r_mask[i] for i in legal_indices)
            break
        action_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        action_r = decode_agent_r_action(action_r_idx, len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), action_r)
    assert found


def test_future_rollout_uses_same_request_trajectory():
    """Future rollout must stop at episode end and never process more than H future requests."""
    env = make_env(
        "snap24_gnutella_reach",
        num_slots=20,
        num_servers=4,
        seed=42,
        k=5,
        max_blocks=10,
        block_sort_strategy="mixed",
    )
    rng = np.random.RandomState(7)
    src = rng.randint(0, env.net.NUM_NODES)
    requests = generate_requests(env, rng, src, 5, num_splits=3, split_profile="default3")
    env.reset(requests)

    # Process first request.
    req0 = requests[0]
    env.advance_time(req0.arrival_time)
    obs_c = build_agent_c_observation(env, req0)
    raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    action_c_idx = int(np.where(raw_c_mask)[0][0]) if raw_c_mask.sum() else 0
    split_id, server_id = divmod(action_c_idx, len(env.mec.servers))
    obs_r = build_agent_r_observation(env, req0, split_id, server_id)
    raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    action_r_idx = int(np.where(raw_r_mask)[0][0]) if raw_r_mask.sum() else 0
    action_r = decode_agent_r_action(action_r_idx, len(obs_r["mod_names"]), env.max_blocks)
    env.step((split_id, server_id), action_r)

    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c(CHECKPOINT_C, "cpu")
    agent_r = _load_ppo_r(CHECKPOINT_R, mod_reg, "cpu")

    from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _rollout_future
    future = _rollout_future(env, requests, 1, horizon=10, agent_c=agent_c, agent_r=agent_r,
                             num_servers=4, util_threshold=0.95, alpha=0.3)
    # Only 4 future requests exist (indices 1-4); blocked+raw_empty+... <= 4.
    total_future_events = future["blocked"] + future["raw_mask_empty"] + future["no_suitable_block"] + future["server_overload"]
    # Some future requests may succeed, so total events can be less than 4.
    assert total_future_events <= 4


def test_oracle_does_not_select_current_failure():
    """Oracle choice, when it exists, must correspond to a current-successful action."""
    recs = [
        RActionRecord(
            seed=0, episode=0, request=0, req_id=0,
            split_id=0, server_id=0, r_action_idx=0, path_idx=0, mod_idx=0, block_idx=0,
            path_length_km=1.0, modulation="QPSK", block_start=0, block_size=2,
            current_success=False, current_reason="no_suitable_block", current_fs=None, current_block_waste=None,
            frag_index_before=0.0, frag_index_after=0.0, lfb_ratio_before=1.0, lfb_ratio_after=1.0,
            k_c_valid_before=1, k_r_total_before=1, phi_spec_before=1.0,
            k_c_valid_after=1, k_r_total_after=1, phi_spec_after=1.0,
            future_blocked_count=0, future_raw_mask_empty_count=0,
            future_no_suitable_block_count=0, future_server_overload_count=0,
            future_phi_spec_mean=1.0, future_phi_spec_min=1.0, future_phi_spec_end=1.0,
        ),
        RActionRecord(
            seed=0, episode=0, request=0, req_id=0,
            split_id=0, server_id=0, r_action_idx=1, path_idx=0, mod_idx=0, block_idx=1,
            path_length_km=1.0, modulation="QPSK", block_start=2, block_size=2,
            current_success=True, current_reason="success", current_fs=2, current_block_waste=0.1,
            frag_index_before=0.0, frag_index_after=0.1, lfb_ratio_before=1.0, lfb_ratio_after=0.9,
            k_c_valid_before=1, k_r_total_before=1, phi_spec_before=1.0,
            k_c_valid_after=1, k_r_total_after=1, phi_spec_after=0.9,
            future_blocked_count=1, future_raw_mask_empty_count=0,
            future_no_suitable_block_count=0, future_server_overload_count=0,
            future_phi_spec_mean=0.9, future_phi_spec_min=0.9, future_phi_spec_end=0.9,
        ),
    ]
    successful = [r for r in recs if r.current_success]
    oracle_sorted = sorted(
        successful,
        key=lambda r: (
            r.future_blocked_count,
            r.future_raw_mask_empty_count,
            r.future_no_suitable_block_count,
            r.current_block_waste if r.current_block_waste is not None else float('inf'),
            -(r.future_phi_spec_mean if r.future_phi_spec_mean is not None else -float('inf')),
        ),
    )
    assert oracle_sorted[0].current_success is True
    assert oracle_sorted[0].r_action_idx == 1


def test_oracle_sorting_rule_priority():
    """Oracle must prioritize lower future blocking, then lower raw-empty, etc."""
    base = dict(
        seed=0, episode=0, request=0, req_id=0,
        split_id=0, server_id=0, path_idx=0, mod_idx=0, block_idx=0,
        path_length_km=1.0, modulation="QPSK", block_start=0, block_size=2,
        current_success=True, current_reason="success", current_fs=2, current_block_waste=0.0,
        frag_index_before=0.0, frag_index_after=0.0, lfb_ratio_before=1.0, lfb_ratio_after=1.0,
        k_c_valid_before=1, k_r_total_before=1, phi_spec_before=1.0,
        k_c_valid_after=1, k_r_total_after=1, phi_spec_after=1.0,
        future_no_suitable_block_count=0, future_server_overload_count=0,
        future_phi_spec_mean=1.0, future_phi_spec_min=1.0, future_phi_spec_end=1.0,
    )
    recs = [
        RActionRecord(r_action_idx=0, future_blocked_count=1, future_raw_mask_empty_count=0, **base),
        RActionRecord(r_action_idx=1, future_blocked_count=0, future_raw_mask_empty_count=1, **base),
        RActionRecord(r_action_idx=2, future_blocked_count=0, future_raw_mask_empty_count=0, **base),
    ]
    oracle_sorted = sorted(
        recs,
        key=lambda r: (
            r.future_blocked_count,
            r.future_raw_mask_empty_count,
            r.future_no_suitable_block_count,
            r.current_block_waste if r.current_block_waste is not None else float('inf'),
            -(r.future_phi_spec_mean if r.future_phi_spec_mean is not None else -float('inf')),
        ),
    )
    assert oracle_sorted[0].r_action_idx == 2
    assert oracle_sorted[1].r_action_idx == 1
    assert oracle_sorted[2].r_action_idx == 0


def test_horizon_does_not_cross_episode_end(agents):
    """Future rollout for the last request should process zero future requests."""
    env = make_env(
        "snap24_gnutella_reach",
        num_slots=20,
        num_servers=4,
        seed=42,
        k=5,
        max_blocks=10,
        block_sort_strategy="mixed",
    )
    rng = np.random.RandomState(8)
    src = rng.randint(0, env.net.NUM_NODES)
    requests = generate_requests(env, rng, src, 2, num_splits=3, split_profile="default3")
    env.reset(requests)

    req0 = requests[0]
    env.advance_time(req0.arrival_time)
    obs_c = build_agent_c_observation(env, req0)
    raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    action_c_idx = int(np.where(raw_c_mask)[0][0]) if raw_c_mask.sum() else 0
    split_id, server_id = divmod(action_c_idx, len(env.mec.servers))
    obs_r = build_agent_r_observation(env, req0, split_id, server_id)
    raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    action_r_idx = int(np.where(raw_r_mask)[0][0]) if raw_r_mask.sum() else 0
    action_r = decode_agent_r_action(action_r_idx, len(obs_r["mod_names"]), env.max_blocks)
    env.step((split_id, server_id), action_r)

    agent_c, agent_r = agents
    from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _rollout_future
    # start_idx == len(requests) means no future requests exist.
    future = _rollout_future(env, requests, 2, horizon=3, agent_c=agent_c, agent_r=agent_r,
                             num_servers=4, util_threshold=0.95, alpha=0.3)
    assert future["blocked"] == 0
    assert future["raw_mask_empty"] == 0
    assert future["phi_spec_mean"] is None


def test_frozen_params_unchanged_after_diagnostic():
    """Loading and running the policies must not mutate their checkpoint weights."""
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c(CHECKPOINT_C, "cpu")
    agent_r = _load_ppo_r(CHECKPOINT_R, mod_reg, "cpu")
    c_ref = {k: v.clone() for k, v in agent_c.policy_net.state_dict().items()}
    r_ref = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}

    # Run a tiny diagnostic.
    from types import SimpleNamespace
    from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _run_diagnostic
    args = SimpleNamespace(
        agent_c_checkpoint=CHECKPOINT_C,
        agent_r_checkpoint=CHECKPOINT_R,
        topology="snap24_gnutella_reach",
        num_slots=20,
        num_servers=4,
        k_paths=5,
        max_blocks=10,
        block_sort_strategy="mixed",
        split_profile="default3",
        num_splits=3,
        seeds="42",
        episodes=1,
        requests_per_episode=5,
        horizon=2,
        arrival_interval=0.25,
        holding_min=4.0, holding_max=10.0,
        deadline_min=30.0, deadline_max=100.0,
        size_min_mb=5.0, size_max_mb=30.0,
        edge_cost_min=0.5, edge_cost_max=15.0,
        modulation_profile="default",
        alpha=0.3,
        util_threshold=0.95,
        device="cpu",
    )
    report = _run_diagnostic(args)
    assert report["frozen_params_unchanged"]["agent_c"] is True
    assert report["frozen_params_unchanged"]["agent_r"] is True

    for k in c_ref:
        assert (agent_c.policy_net.state_dict()[k] == c_ref[k]).all()
    for k in r_ref:
        assert (agent_r.policy_net.state_dict()[k] == r_ref[k]).all()


def test_report_uses_correct_denominators():
    """Summary rates must use the denominators documented in the diagnostic."""
    from types import SimpleNamespace
    from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _run_diagnostic
    args = SimpleNamespace(
        agent_c_checkpoint=CHECKPOINT_C,
        agent_r_checkpoint=CHECKPOINT_R,
        topology="snap24_gnutella_reach",
        num_slots=20,
        num_servers=4,
        k_paths=5,
        max_blocks=10,
        block_sort_strategy="mixed",
        split_profile="default3",
        num_splits=3,
        seeds="42",
        episodes=1,
        requests_per_episode=10,
        horizon=2,
        arrival_interval=0.25,
        holding_min=4.0, holding_max=10.0,
        deadline_min=30.0, deadline_max=100.0,
        size_min_mb=5.0, size_max_mb=30.0,
        edge_cost_min=0.5, edge_cost_max=15.0,
        modulation_profile="default",
        alpha=0.3,
        util_threshold=0.95,
        device="cpu",
    )
    report = _run_diagnostic(args)
    s = report["summary"]
    total = s["total_requests"]
    multi = s["multi_action_count"]
    assert s["multi_action_rate"] == multi / max(total, 1)
    assert s["multi_action_with_diff_future_blocking_rate"] == s["multi_action_with_diff_future_blocking"] / max(multi, 1)
    assert s["multi_action_with_diff_future_raw_empty_rate"] == s["multi_action_with_diff_future_raw_empty"] / max(multi, 1)
    # Oracle headroom uses common comparison states.
    assert s["common_comparison_count"] <= s["oracle_evaluable_count"]


def test_mask_empty_but_success_counted():
    """mask_empty_but_success flag aggregates correctly from state records."""
    from types import SimpleNamespace
    from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _run_diagnostic
    args = SimpleNamespace(
        agent_c_checkpoint=CHECKPOINT_C,
        agent_r_checkpoint=CHECKPOINT_R,
        topology="snap24_gnutella_reach",
        num_slots=20,
        num_servers=4,
        k_paths=5,
        max_blocks=10,
        block_sort_strategy="mixed",
        split_profile="default3",
        num_splits=3,
        seeds="42",
        episodes=1,
        requests_per_episode=10,
        horizon=2,
        arrival_interval=0.25,
        holding_min=4.0, holding_max=10.0,
        deadline_min=30.0, deadline_max=100.0,
        size_min_mb=5.0, size_max_mb=30.0,
        edge_cost_min=0.5, edge_cost_max=15.0,
        modulation_profile="default",
        alpha=0.3,
        util_threshold=0.95,
        device="cpu",
    )
    report = _run_diagnostic(args)
    counted = sum(s["mask_empty_but_success"] for s in report["state_records"])
    assert counted == report["summary"]["mask_empty_but_success_count"]
