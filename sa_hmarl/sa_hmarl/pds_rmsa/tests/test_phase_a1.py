"""Phase A.1 hardening tests for PDS-RMSA."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from sa_hmarl.pds_rmsa.baselines.ksp_ff import ksp_ff_action
from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, PDSRMSAEnv, RMSAAction
from sa_hmarl.pds_rmsa.env.reservations import PostDecisionState, Reservation
from sa_hmarl.pds_rmsa.env.topology import make_ksp_k5_test, make_tiny_ring3
from sa_hmarl.pds_rmsa.env.traffic_trace import (
    Trace,
    generate_trace,
    load_trace,
    save_trace,
    trace_sha256,
)
from sa_hmarl.pds_rmsa.features.afterstate import (
    build_afterstate_features,
    build_probe_suite,
    build_schema,
    probe_suite_sha256,
)
from sa_hmarl.pds_rmsa.protocol import BITRATE_CHOICES_GBPS, K_PATHS, Request
from sa_hmarl.pds_rmsa.run_phase0 import run_phase0


# 1

def test_allocate_release_restores_state(tiny_env):
    env = tiny_env
    env.reset()
    path = (0, 1)
    start = 2
    count = 3
    ok = env.spectrum.allocate(path, start, count)
    assert ok
    row = env.spectrum.directed_link_ids().index((0, 1))
    assert np.all(env.spectrum.as_bitmap()[row, 2:5])

    res = Reservation(
        request_id=0,
        path=path,
        start_slot=start,
        required_fs=count,
        release_time=5.0,
    )
    env.ledger.add(res)
    env.advance_external(env.trace.requests[0])
    if env.time >= 5.0:
        assert not np.any(env.spectrum.as_bitmap()[row, 2:5])
        assert len(env.ledger.active_at(env.time)) == 0


# 2

def test_all_legal_candidates_succeed_in_trial(tiny_env):
    env = tiny_env
    env.reset()
    for request in env.trace.requests[:10]:
        env.advance_external(request)
        candidates = env.build_candidates(request)
        for action in candidates:
            pds = env.trial_afterstate(request, action)
            if isinstance(action, RMSAAction):
                assert pds.admitted_this_step
                assert pds.allocated_action == action
            else:
                assert not pds.admitted_this_step
                assert pds.allocated_action is None


# 3

def test_fabricated_action_rejected(tiny_env):
    env = tiny_env
    env.reset()
    request = env.trace.requests[0]
    env.advance_external(request)
    candidates = env.build_candidates(request)
    if len(candidates) == 1 and isinstance(candidates[0], BlockedAction):
        pytest.skip("no legal candidates to fabricate against")
    legal = candidates[0]
    fabricated = RMSAAction(
        action_id=(legal.path_rank, legal.modulation, legal.start_slot + 1, legal.required_fs),
        path_rank=legal.path_rank,
        path=legal.path,
        modulation=legal.modulation,
        start_slot=legal.start_slot + 1,
        required_fs=legal.required_fs,
    )
    with pytest.raises(ValueError, match="RMSAAction not in current candidate set"):
        env.step(fabricated, request=request)


# 4

def test_blocked_action_rejected_when_legal_candidates_exist(tiny_env):
    env = tiny_env
    env.reset()
    request = env.trace.requests[0]
    env.advance_external(request)
    candidates = env.build_candidates(request)
    if len(candidates) == 1 and isinstance(candidates[0], BlockedAction):
        pytest.skip("no legal candidates")
    with pytest.raises(ValueError, match="BlockedAction chosen while legal candidates exist"):
        env.step(BlockedAction(reason="insufficient_spectrum"), request=request)


# 5

def test_blocked_action_only_when_no_candidates(nsfnet_env):
    env = nsfnet_env
    env.reset()
    # Saturate all slots on a simple path by repeatedly allocating on every directed link.
    link_ids = env.spectrum.directed_link_ids()
    for link in link_ids:
        env.spectrum.allocate(link, 0, env.num_slots)
    req = env.trace.requests[0]
    candidates = env.build_candidates(req)
    assert len(candidates) == 1
    assert isinstance(candidates[0], BlockedAction)
    assert candidates[0].reason in {
        "no_candidate_path",
        "no_reach_feasible_path_mod",
        "insufficient_spectrum",
    }


# 6

def test_trial_afterstate_no_side_effects(tiny_env):
    env = tiny_env
    env.reset()
    request = env.trace.requests[0]
    env.advance_external(request)
    candidates = env.build_candidates(request)
    action = candidates[0]
    fp_before = env.state_fingerprint()
    pds = env.trial_afterstate(request, action)
    assert env.state_fingerprint() == fp_before


# 7

def test_advance_external_reproduces_next_prestate(tiny_env):
    env = tiny_env
    env.reset()
    # Inject a few reservations with known release times.
    link_ids = env.spectrum.directed_link_ids()
    res_a = Reservation(
        request_id=1,
        path=(0, 1),
        start_slot=0,
        required_fs=2,
        release_time=2.0,
    )
    res_b = Reservation(
        request_id=2,
        path=(1, 2),
        start_slot=1,
        required_fs=2,
        release_time=5.0,
    )
    env.ledger.add(res_a)
    env.ledger.add(res_b)
    env.spectrum.allocate(res_a.path, res_a.start_slot, res_a.required_fs)
    env.spectrum.allocate(res_b.path, res_b.start_slot, res_b.required_fs)

    snapshot = env.snapshot_true_state()
    base_request = env.trace.requests[0]
    next_request = Request(
        request_id=base_request.request_id,
        src_node=base_request.src_node,
        dst_node=base_request.dst_node,
        bitrate_gbps=base_request.bitrate_gbps,
        arrival_time=3.0,
        holding_time=base_request.holding_time,
    )

    # Manual release of reservations with release_time <= 3.0.
    expected_bitmap = env.spectrum.as_bitmap().copy()
    expected_ledger = [r for r in env.ledger.active_at(env.time)]
    for r in env.ledger.active_at(env.time):
        if r.release_time <= 3.0:
            row = link_ids.index((r.path[0], r.path[1]))
            expected_bitmap[row, r.start_slot : r.start_slot + r.required_fs] = False
            expected_ledger.remove(r)

    env.advance_external(next_request)
    assert env.time == 3.0
    assert np.array_equal(env.spectrum.as_bitmap(), expected_bitmap)
    assert env.ledger.as_sorted_tuple() == tuple(sorted(
        expected_ledger, key=lambda r: (r.release_time, r.request_id, r.start_slot, r.path)
    ))


# 8

def test_bitmap_equals_active_reservations(tiny_env):
    env = tiny_env
    env.reset()
    for request in env.trace.requests[:15]:
        env.advance_external(request)
        action = ksp_ff_action(env, request)
        env.step(action, request=request)
        expected = np.zeros_like(env.spectrum.as_bitmap())
        link_ids = env.spectrum.directed_link_ids()
        for res in env.ledger.active_at(env.time):
            for i in range(len(res.path) - 1):
                row = link_ids.index((res.path[i], res.path[i + 1]))
                expected[row, res.start_slot : res.start_slot + res.required_fs] = True
        assert np.array_equal(expected, env.spectrum.as_bitmap())


def test_bitmap_all_free_after_final_drain(tiny_env):
    env = tiny_env
    env.reset()
    for request in env.trace.requests:
        env.advance_external(request)
        action = ksp_ff_action(env, request)
        env.step(action, request=request)
    max_release = max((r.release_time for r in env.ledger), default=0.0)
    fake_request = env.trace.requests[-1].__class__(
        request_id=-1,
        src_node=0,
        dst_node=1,
        bitrate_gbps=25,
        arrival_time=max_release + 1.0,
        holding_time=1.0,
    )
    env.advance_external(fake_request)
    assert env.spectrum.free_ratio() == 1.0


# 9

def test_trace_reproducibility_fingerprints():
    trace = generate_trace("tiny_ring3", 3, 20, 10.0, seed=55, num_slots=8)
    env1 = PDSRMSAEnv("tiny_ring3", 8, 10.0, 0, trace=trace)
    env2 = PDSRMSAEnv("tiny_ring3", 8, 10.0, 0, trace=trace)
    env1.reset()
    env2.reset()
    for request in trace.requests:
        env1.advance_external(request)
        action1 = ksp_ff_action(env1, request)
        env1.step(action1, request=request)

        env2.advance_external(request)
        action2 = ksp_ff_action(env2, request)
        env2.step(action2, request=request)

        assert env1.state_fingerprint() == env2.state_fingerprint()


# 10

def test_identical_post_state_identical_features(tiny_probe_suite):
    config = {"num_slots": 8}
    bitmap = np.random.RandomState(1).randint(0, 2, size=(6, 8), dtype=bool)
    reservations = (
        Reservation(
            request_id=0,
            path=(0, 1),
            start_slot=1,
            required_fs=2,
            release_time=3.0,
        ),
    )
    pds1 = PostDecisionState(
        time=1.0,
        bitmap=bitmap,
        reservations=reservations,
        admitted_this_step=True,
        allocated_action=None,
    )
    pds2 = PostDecisionState(
        time=1.0,
        bitmap=bitmap.copy(),
        reservations=reservations,
        admitted_this_step=False,
        allocated_action=None,
    )
    feat1 = build_afterstate_features(pds1, tiny_probe_suite, config)
    feat2 = build_afterstate_features(pds2, tiny_probe_suite, config)
    assert np.allclose(feat1, feat2)


# 11

def test_raw_bitmap_segment_matches_post_state_bitmap(tiny_env, tiny_probe_suite):
    env = tiny_env
    env.reset()
    request = env.trace.requests[0]
    env.advance_external(request)
    action = ksp_ff_action(env, request)
    pds = env.trial_afterstate(request, action)
    config = {"num_slots": env.num_slots}
    features = build_afterstate_features(pds, tiny_probe_suite, config)
    bitmap_len = pds.bitmap.size
    raw_segment = features[:bitmap_len]
    expected = pds.bitmap.flatten().astype(np.float32)
    assert np.array_equal(raw_segment, expected)


# 12

def test_probe_suite_independent_of_traffic_seed():
    suite1 = build_probe_suite(num_nodes=14, topology_name="nsfnet")
    suite2 = build_probe_suite(num_nodes=14, topology_name="nsfnet")
    assert probe_suite_sha256(suite1) == probe_suite_sha256(suite2)


# 13

def test_schema_dimension_matches_feature_vector(tiny_env, tiny_probe_suite):
    env = tiny_env
    env.reset()
    request = env.trace.requests[0]
    env.advance_external(request)
    action = ksp_ff_action(env, request)
    pds = env.trial_afterstate(request, action)
    config = {"num_slots": env.num_slots}
    features = build_afterstate_features(pds, tiny_probe_suite, config)
    schema = build_schema(
        topology_name="tiny_ring3",
        feature_config=config,
        probe_count=len(tiny_probe_suite.probes),
    )
    assert len(schema.feature_names) == len(features)


def test_schema_has_no_request_or_action_specific_names(tiny_env, tiny_probe_suite):
    env = tiny_env
    env.reset()
    request = env.trace.requests[0]
    env.advance_external(request)
    action = ksp_ff_action(env, request)
    pds = env.trial_afterstate(request, action)
    config = {"num_slots": env.num_slots}
    schema = build_schema(
        topology_name="tiny_ring3",
        feature_config=config,
        probe_count=len(tiny_probe_suite.probes),
    )
    forbidden = {
        "src", "dst", "bitrate", "request_id", "holding",
        "path_rank", "modulation", "start_slot", "action_id",
        "admitted_this_step", "allocated_action",
    }
    for name in schema.feature_names:
        assert not any(term in name for term in forbidden), f"forbidden term in {name}"


# 14

def test_km_ksp_ordering_and_k5_truncation():
    topology = make_ksp_k5_test()
    paths = topology.k_shortest_paths(0, 4, K_PATHS)
    expected = [
        (0, 8, 4),       # length 4, hops 2
        (0, 1, 2, 3, 4), # length 4, hops 4, tuple smaller
        (0, 5, 6, 7, 4), # length 4, hops 4
        (0, 4),          # length 5, hops 1
        (0, 9, 4),       # length 6, hops 2
    ]
    assert paths == expected


def test_hops_ordering_collects_kth_hop_boundary_before_truncation():
    topology = make_ksp_k5_test()
    paths = topology.k_shortest_paths(0, 4, 5, sort_by="hops")
    expected = [
        (0, 4),
        (0, 8, 4),
        (0, 9, 4),
        (0, 1, 2, 3, 4),
        (0, 5, 6, 7, 4),
    ]
    assert paths == expected


def test_configurable_env_defaults_and_clone_preserve_protocol(tiny_env):
    assert tiny_env.k_paths == K_PATHS
    assert tiny_env.path_sort_strategy == "km"
    clone = tiny_env.clone()
    assert clone.k_paths == K_PATHS
    assert clone.path_sort_strategy == "km"


# 15

def test_runner_executes_only_warmup_plus_eval(tmp_path):
    output_dir = tmp_path / "run_out"
    args = argparse.Namespace(
        topology="tiny_ring3",
        num_slots=8,
        load_erlang=10.0,
        warmup_requests=10,
        eval_requests=20,
        seed=123,
        trace_in=None,
        trace_out=None,
        output_dir=str(output_dir),
    )
    run_phase0(args)
    manifest_path = output_dir / "PHASE0_MANIFEST.json"
    results_path = output_dir / "PHASE0_RESULTS.json"
    assert manifest_path.exists()
    assert results_path.exists()
    manifest = json.loads(manifest_path.read_text())
    results = json.loads(results_path.read_text())
    assert manifest["trace_total_requests"] == 30
    assert manifest["trace_executed_requests"] == 30
    assert len(results["results"]) == 30
    assert results["summary"]["admitted_eval"] + results["summary"]["blocked_eval"] == 20


# 16

def _mutate_and_save(trace, mutation, tmp_path):
    raw = json.loads(trace_sha256(trace))  # not used, placeholder for deep copy
    requests = list(trace.requests)
    mutated = mutation(requests)
    new_trace = Trace(
        topology=trace.topology,
        num_slots=trace.num_slots,
        load_erlang=trace.load_erlang,
        seed=trace.seed,
        requests=tuple(mutated),
    )
    path = tmp_path / "invalid_trace.jsonl"
    save_trace(path, new_trace)
    return path


def test_invalid_trace_fail_fast(tmp_path):
    trace = generate_trace("tiny_ring3", 3, 10, 10.0, seed=1, num_slots=8)
    # non-monotonic arrivals
    requests = list(trace.requests)
    requests[2] = requests[2].__class__(
        **{**requests[2].__dict__, "arrival_time": requests[1].arrival_time - 1.0}
    )
    bad_path = tmp_path / "bad_arrivals.jsonl"
    save_trace(bad_path, Trace(
        topology=trace.topology,
        num_slots=trace.num_slots,
        load_erlang=trace.load_erlang,
        seed=trace.seed,
        requests=tuple(requests),
    ))
    with pytest.raises(ValueError):
        load_trace(bad_path)

    # invalid bitrate
    requests = list(trace.requests)
    requests[0] = requests[0].__class__(
        **{**requests[0].__dict__, "bitrate_gbps": 30}
    )
    bad_path = tmp_path / "bad_bitrate.jsonl"
    save_trace(bad_path, Trace(
        topology=trace.topology,
        num_slots=trace.num_slots,
        load_erlang=trace.load_erlang,
        seed=trace.seed,
        requests=tuple(requests),
    ))
    with pytest.raises(ValueError):
        load_trace(bad_path)

    # src == dst
    requests = list(trace.requests)
    requests[0] = requests[0].__class__(
        **{**requests[0].__dict__, "src_node": 1, "dst_node": 1}
    )
    bad_path = tmp_path / "bad_src_dst.jsonl"
    save_trace(bad_path, Trace(
        topology=trace.topology,
        num_slots=trace.num_slots,
        load_erlang=trace.load_erlang,
        seed=trace.seed,
        requests=tuple(requests),
    ))
    with pytest.raises(ValueError):
        load_trace(bad_path)

    # non-positive holding
    requests = list(trace.requests)
    requests[0] = requests[0].__class__(
        **{**requests[0].__dict__, "holding_time": 0.0}
    )
    bad_path = tmp_path / "bad_holding.jsonl"
    save_trace(bad_path, Trace(
        topology=trace.topology,
        num_slots=trace.num_slots,
        load_erlang=trace.load_erlang,
        seed=trace.seed,
        requests=tuple(requests),
    ))
    with pytest.raises(ValueError):
        load_trace(bad_path)

    # generate_trace parameter validation
    with pytest.raises(ValueError):
        generate_trace("tiny_ring3", 1, 10, 10.0, seed=1)
    with pytest.raises(ValueError):
        generate_trace("tiny_ring3", 3, 10, 0.0, seed=1)
    with pytest.raises(ValueError):
        generate_trace("tiny_ring3", 3, -1, 10.0, seed=1)
