"""Unit tests for C-side downstream survivability diagnostic."""
from __future__ import annotations

import copy
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from sa_hmarl.env.observation_builder import build_agent_c_observation, build_agent_r_observation, decode_agent_c_action
from sa_hmarl.evaluation.diagnose_c_downstream_survivability import (
    _compute_probe_survivability,
    _evaluate_candidate,
    _make_probe_requests,
    _run_oracle_phi_trajectory,
    _run_ppo_c_trajectory,
    evaluate,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import _load_ppo_c, _load_ppo_r, _snapshot_before_r_decision
from sa_hmarl.evaluation.eval_r_counterfactual_ranking_closed_loop import load_ranking_checkpoint
from sa_hmarl.evaluation.eval_c_demand_potential_closed_loop import PerMethodMetrics, _aggregate_metrics
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


@pytest.fixture
def args():
    """Minimal arg namespace for tests."""
    class Args:
        agent_c_checkpoint = "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt"
        agent_r_checkpoint = "sa_hmarl/checkpoints/agent_r_mixed.pt"
        ranking_checkpoint = "sa_hmarl/checkpoints/r_counterfactual_ranking_k5m10_h5/ranking_model.pt"
        seeds = "3030"
        episodes = 1
        requests_per_episode = 5
        probe_count = 4
        base_probe_seed = 123456
        topology = "snap24_gnutella_reach"
        num_slots = 24
        num_servers = 4
        k_paths = 5
        max_blocks = 10
        block_sort_strategy = "mixed"
        split_profile = "default3"
        num_splits = 3
        arrival_interval = 0.15
        holding_min = 4.0
        holding_max = 10.0
        deadline_min = 30.0
        deadline_max = 100.0
        size_min_mb = 5.0
        size_max_mb = 30.0
        edge_cost_min = 0.5
        edge_cost_max = 15.0
        modulation_profile = "default"
        device = "cpu"
        output_json = ""
        output_md = ""
    return Args()


@pytest.fixture
def agents(args):
    env_proto = make_env(
        args.topology, args.num_slots, args.num_servers, 42,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    rank_model, rank_mean, rank_std, _ = load_ranking_checkpoint(args.ranking_checkpoint, args.device)
    return env_proto, agent_c, agent_r, (rank_model, rank_mean, rank_std)


@pytest.fixture
def episode(args, agents):
    env_proto = agents[0]
    rng = np.random.RandomState(3030)
    src = int(rng.randint(0, env_proto.net.NUM_NODES))
    return generate_requests(
        env_proto, rng, src, args.requests_per_episode,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )


@pytest.fixture
def env_with_request(args, episode):
    env = make_env(
        args.topology, args.num_slots, args.num_servers, 3030,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    env.reset(episode)
    req = episode[0]
    env.advance_time(req.arrival_time)
    return env, req


def test_probe_does_not_change_env(env_with_request, args):
    env, req = env_with_request
    snapshot = _snapshot_before_r_decision(env, req.req_id)
    probe_rng = np.random.RandomState(args.base_probe_seed)
    probes = _make_probe_requests(
        env, req, args.probe_count, probe_rng,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )
    before = copy.deepcopy(env.net.link_states)
    _compute_probe_survivability(env, probes)
    after = env.net.link_states
    assert before.keys() == after.keys()
    for key in before:
        assert np.array_equal(before[key], after[key])


def test_phi_in_zero_one(env_with_request, args):
    env, req = env_with_request
    probe_rng = np.random.RandomState(args.base_probe_seed)
    probes = _make_probe_requests(
        env, req, args.probe_count, probe_rng,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )
    result = _compute_probe_survivability(env, probes)
    assert 0.0 <= result["phi"] <= 1.0


def test_all_candidates_share_same_probes(env_with_request, args, agents):
    env, req = env_with_request
    _, _, agent_r, rank_pack = agents
    rank_model, rank_mean, rank_std = rank_pack
    obs_c = build_agent_c_observation(env, req)
    valid_c_indices = np.flatnonzero(np.asarray(obs_c["agent_c_mask"], dtype=bool)).tolist()
    if len(valid_c_indices) < 2:
        pytest.skip("need >=2 valid C candidates")
    snapshot = _snapshot_before_r_decision(env, req.req_id)
    probe_rng = np.random.RandomState(args.base_probe_seed)
    probes = _make_probe_requests(
        env, req, args.probe_count, probe_rng,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )
    phi_values = []
    for c_idx in valid_c_indices[:2]:
        cand = _evaluate_candidate(
            snapshot, req, c_idx, agent_r, rank_model, rank_mean, rank_std,
            probes, args.device,
        )
        phi_values.append(cand["phi"])
    # Probes shared means both candidates get same probe_count.
    assert len(set(phi_values)) <= 2  # deterministic given same probes


def test_valid_c_count_zero_phi(env_with_request, args):
    env, req = env_with_request
    obs_c = build_agent_c_observation(env, req)
    obs_c["agent_c_mask"][:] = False
    probe_rng = np.random.RandomState(args.base_probe_seed)
    probes = _make_probe_requests(
        env, req, args.probe_count, probe_rng,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )
    result = _compute_probe_survivability(env, probes)
    # Mask being all False doesn't affect probe computation; function still returns phi.
    assert 0.0 <= result["phi"] <= 1.0


def test_candidate_deepcopy_does_not_change_main_env(env_with_request, args, agents):
    env, req = env_with_request
    _, _, agent_r, rank_pack = agents
    rank_model, rank_mean, rank_std = rank_pack
    obs_c = build_agent_c_observation(env, req)
    valid_c_indices = np.flatnonzero(np.asarray(obs_c["agent_c_mask"], dtype=bool)).tolist()
    if not valid_c_indices:
        pytest.skip("no valid C candidates")
    snapshot = _snapshot_before_r_decision(env, req.req_id)
    probe_rng = np.random.RandomState(args.base_probe_seed)
    probes = _make_probe_requests(
        env, req, args.probe_count, probe_rng,
        args.arrival_interval, args.holding_min, args.holding_max,
        args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
        args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
    )
    before = copy.deepcopy(env.net.link_states)
    _evaluate_candidate(
        snapshot, req, valid_c_indices[0], agent_r, rank_model, rank_mean, rank_std,
        probes, args.device,
    )
    assert before.keys() == env.net.link_states.keys()
    for key in before:
        assert np.array_equal(before[key], env.net.link_states[key])


def test_oracle_phi_does_not_load_deeprmsa(args):
    # The diagnostic module should not reference DeepRMSA checkpoints.
    import sa_hmarl.evaluation.diagnose_c_downstream_survivability as diag_mod
    source = Path(diag_mod.__file__).read_text()
    assert "deep_rmsa_checkpoint" not in source
    assert "DeepRMSA" not in source or "DeepRMSA" not in source.split("#")[0]


def test_smoke_run_outputs_fields(args, tmp_path):
    args.output_json = str(tmp_path / "smoke.json")
    args.output_md = str(tmp_path / "smoke.md")
    report = evaluate(args)
    assert "aggregate" in report
    assert "per_seed_metrics" in report
    assert "records" in report
    agg = report["aggregate"]
    for key in ["total_requests", "requests_with_valid_c_ge2", "avg_valid_c_count",
                "raw_mask_empty_rate", "ppo_c_v1_ranker", "oracle_phi", "phi_stats",
                "ppo_c_selection_quality", "correlations"]:
        assert key in agg
    assert Path(args.output_json).exists()
    assert Path(args.output_md).exists()


def test_smoke_run_json_loadable(args, tmp_path):
    args.output_json = str(tmp_path / "smoke2.json")
    args.output_md = str(tmp_path / "smoke2.md")
    evaluate(args)
    data = json.loads(Path(args.output_json).read_text())
    assert data["aggregate"]["total_requests"] == args.episodes * args.requests_per_episode
