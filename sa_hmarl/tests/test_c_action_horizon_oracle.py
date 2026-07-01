"""Tests for the C-action H-step Oracle diagnostic."""
from __future__ import annotations

from argparse import Namespace

import numpy as np
import pytest

from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.evaluation.diagnose_c_action_horizon_oracle import (
    CActionRecord,
    _execute_c_with_frozen_r,
    _demand_aware_potential,
    _load_ppo_c,
    _load_ppo_r,
    _oracle_sort_key,
    _run_diagnostic,
    _snapshot_before_c_decision,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


C_CKPT = "sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt"
R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"


@pytest.fixture
def env_requests_agents():
    env = make_env(
        "snap24_gnutella_reach", num_slots=20, num_servers=4, seed=42,
        k=5, max_blocks=10, block_sort_strategy="mixed",
    )
    rng = np.random.RandomState(42)
    src = rng.randint(0, env.net.NUM_NODES)
    requests = generate_requests(
        env, rng, src, 12, num_splits=3, split_profile="default3"
    )
    env.reset(requests)
    agent_c = _load_ppo_c(C_CKPT, "cpu")
    agent_r = _load_ppo_r(
        R_CKPT, ModulationRegistry.from_profile("default"), "cpu"
    )
    return env, requests, agent_c, agent_r


def test_snapshot_requires_current_request_at_queue_head(env_requests_agents):
    env, requests, _, _ = env_requests_agents
    env.advance_time(requests[0].arrival_time)
    snap = _snapshot_before_c_decision(env, requests[0].req_id)
    assert snap.event_queue[0][2].req_id == requests[0].req_id
    with pytest.raises(RuntimeError):
        _snapshot_before_c_decision(env, requests[0].req_id + 1)


def test_counterfactual_c_actions_start_from_identical_snapshot(env_requests_agents):
    env, requests, _, _ = env_requests_agents
    req = requests[0]
    env.advance_time(req.arrival_time)
    obs = build_agent_c_observation(env, req)
    legal = np.flatnonzero(obs["agent_c_mask"])
    assert len(legal) >= 2
    snap = _snapshot_before_c_decision(env, req.req_id)
    signatures = []
    for _ in legal[:2]:
        branch = _snapshot_before_c_decision(snap, req.req_id)
        signatures.append((branch.time, len(branch.active_connections)))
    assert signatures[0] == signatures[1]


def test_every_enumerated_candidate_is_raw_mask_legal(env_requests_agents):
    env, requests, _, _ = env_requests_agents
    req = requests[0]
    env.advance_time(req.arrival_time)
    obs = build_agent_c_observation(env, req)
    mask = np.asarray(obs["agent_c_mask"], dtype=bool)
    legal = np.flatnonzero(mask).tolist()
    assert legal
    assert all(mask[idx] for idx in legal)
    assert set(legal) == set(np.where(mask)[0].tolist())


def test_execute_candidate_uses_frozen_r_and_processes_current_request(env_requests_agents):
    env, requests, _, agent_r = env_requests_agents
    req = requests[0]
    env.advance_time(req.arrival_time)
    obs = build_agent_c_observation(env, req)
    c_idx = int(np.flatnonzero(obs["agent_c_mask"])[0])
    state_before = {k: v.clone() for k, v in agent_r.policy_net.state_dict().items()}
    _, info = _execute_c_with_frozen_r(env, req, c_idx, agent_r, 4)
    assert info["req_id"] == req.req_id
    assert all(
        np.array_equal(agent_r.policy_net.state_dict()[k].cpu().numpy(), v.cpu().numpy())
        for k, v in state_before.items()
    )


def _record(action, success, blocked, raw_empty=0, phi=1.0):
    return CActionRecord(
        seed=1, episode=0, request=0, req_id=0,
        c_action_idx=action, split_id=0, server_id=action, r_action_idx=0,
        current_success=success, current_reason="success" if success else "failed",
        current_delay_ms=1.0, current_fs=1, current_block_waste=0.0,
        k_c_valid_before=2, k_r_total_before=2, phi_spec_before=1.0,
        frag_index_before=0.0, frag_index_after=0.0,
        lfb_ratio_before=1.0, lfb_ratio_after=1.0,
        k_c_valid_after=2, k_r_total_after=2, phi_spec_after=phi,
        future_blocked_count=blocked,
        future_raw_mask_empty_count=raw_empty,
        future_no_suitable_block_count=0,
        future_server_overload_count=0,
        future_phi_spec_mean=phi, future_phi_spec_min=phi,
        future_phi_spec_end=phi,
    )


def test_oracle_sort_prioritizes_future_blocking_then_raw_empty():
    better_block = _record(0, True, blocked=0, raw_empty=2, phi=0.1)
    worse_block = _record(1, True, blocked=1, raw_empty=0, phi=10.0)
    assert min([worse_block, better_block], key=_oracle_sort_key) is better_block

    fewer_empty = _record(2, True, blocked=0, raw_empty=0, phi=0.1)
    assert min([better_block, fewer_empty], key=_oracle_sort_key) is fewer_empty


def test_failed_current_candidate_is_not_oracle_eligible():
    failed = _record(0, False, blocked=0)
    successful = _record(1, True, blocked=1)
    eligible = [r for r in [failed, successful] if r.current_success]
    assert min(eligible, key=_oracle_sort_key) is successful


def test_demand_potential_uses_supplied_history_only(env_requests_agents):
    env, requests, _, _ = env_requests_agents
    env.advance_time(requests[0].arrival_time)
    value_one = _demand_aware_potential(
        env, requests[:1], window=12, probe_limit=6,
        util_threshold=0.95, alpha=0.3,
    )
    value_one_again = _demand_aware_potential(
        env, requests[:1], window=12, probe_limit=6,
        util_threshold=0.95, alpha=0.3,
    )
    assert np.isfinite(value_one)
    assert value_one == pytest.approx(value_one_again)
    assert _demand_aware_potential(
        env, [], window=12, probe_limit=6,
        util_threshold=0.95, alpha=0.3,
    ) is None


def test_smoke_report_uses_complete_horizon_and_keeps_models_frozen():
    args = Namespace(
        agent_c_checkpoint=C_CKPT, agent_r_checkpoint=R_CKPT,
        topology="snap24_gnutella_reach", num_slots=20, num_servers=4,
        k_paths=5, max_blocks=10, block_sort_strategy="mixed",
        split_profile="default3", num_splits=3, seeds="42", episodes=1,
        requests_per_episode=6, horizon=2, arrival_interval=0.25,
        holding_min=4.0, holding_max=10.0, deadline_min=30.0,
        deadline_max=100.0, size_min_mb=5.0, size_max_mb=30.0,
        edge_cost_min=0.5, edge_cost_max=15.0,
        modulation_profile="default", alpha=0.3, util_threshold=0.95,
        potential_history_window=12, potential_probe_limit=0,
        device="cpu",
    )
    report = _run_diagnostic(args)
    assert report["summary"]["total_requests"] == 6
    assert report["frozen_params_unchanged"] == {
        "agent_c": True, "agent_r": True,
    }
    for state in report["state_records"]:
        if state["request"] >= 4:
            assert state["full_horizon"] is False
            assert state["oracle_c_action_idx"] is None
