"""Tests for observation builder (Agent-C and Agent-R)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)


# ------------------------------------------------------------------
# Helper: build a standard env + request
# ------------------------------------------------------------------

def _make_env(num_servers=2, capacities=None):
    net = OpticalNetwork("net1", num_slots=32)
    all_nodes = [0, 3, 5, 7, 9]
    mec = MECCluster(
        num_nodes=net.NUM_NODES,
        num_servers=num_servers,
        seed=42,
        server_nodes=all_nodes[:num_servers],
        capacities=capacities,
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)
    return env


def _make_request(num_splits=2, edge_costs=None, local_costs=None):
    if edge_costs is None:
        edge_costs = [1.0] * num_splits
    if local_costs is None:
        local_costs = [0.5] * num_splits
    splits = [
        SplitProfile(i, 1.0, local_costs[i], edge_costs[i])
        for i in range(num_splits)
    ]
    return DNNRequest(
        req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
        deadline_ms=100.0, splits=splits,
    )


# ------------------------------------------------------------------
# 1. Agent-C observation fields complete
# ------------------------------------------------------------------

def test_agent_c_observation_fields():
    env = _make_env()
    request = _make_request(num_splits=2)
    obs = build_agent_c_observation(env, request)

    required_top = [
        "request_features", "candidate_features", "server_utilizations",
        "spectrum_summaries", "feasible_counts", "agent_c_mask",
    ]
    for key in required_top:
        assert key in obs, f"Missing top-level key: {key}"

    required_feat = [
        "intermediate_size_mb", "local_compute_ms", "edge_compute_ms",
        "server_utilization", "best_fs_estimate", "safe_fs_estimate",
        "feasible_count", "spectrum_summary",
    ]
    for feat in obs["candidate_features"]:
        for key in required_feat:
            assert key in feat, f"Missing candidate feature key: {key}"

    print("test_agent_c_observation_fields PASSED")


# ------------------------------------------------------------------
# 2. Agent-C candidate_features shape = num_splits * num_servers
# ------------------------------------------------------------------

def test_agent_c_candidate_features_shape():
    env = _make_env(num_servers=3)
    request = _make_request(num_splits=2)
    obs = build_agent_c_observation(env, request)

    expected = 2 * 3
    assert len(obs["candidate_features"]) == expected, \
        f"Expected {expected} candidate features, got {len(obs['candidate_features'])}"
    print("test_agent_c_candidate_features_shape PASSED")


# ------------------------------------------------------------------
# 3. Agent-C mask masks infeasible split-server combos
# ------------------------------------------------------------------

def test_agent_c_mask_masks_infeasible():
    """Split with edge_compute_cost > server capacity should be masked."""
    env = _make_env(num_servers=2, capacities=[50.0, 50.0])
    # split0: cost=10 (feasible), split1: cost=60 (infeasible on both servers)
    request = _make_request(num_splits=2, edge_costs=[10.0, 60.0])
    obs = build_agent_c_observation(env, request)

    mask = obs["agent_c_mask"]
    num_splits = 2
    num_servers = 2
    assert len(mask) == num_splits * num_servers

    # split0 on both servers should be feasible (True)
    assert mask[0] == True, "split0-server0 should be feasible"
    assert mask[1] == True, "split0-server1 should be feasible"

    # split1 on both servers should be infeasible (False) because cost > capacity
    assert mask[2] == False, "split1-server0 should be masked (cost > cap)"
    assert mask[3] == False, "split1-server1 should be masked (cost > cap)"

    print("test_agent_c_mask_masks_infeasible PASSED")


# ------------------------------------------------------------------
# 4. Agent-R observation generates paths / required_fs / blocks
# ------------------------------------------------------------------

def test_agent_r_observation_structure():
    env = _make_env(num_servers=2)
    request = _make_request(num_splits=2)
    obs = build_agent_r_observation(env, request, split_id=0, server_id=0)

    required_top = [
        "candidate_paths", "path_features", "mod_names",
        "feasible_mask_per_path_mod", "required_fs_per_path_mod",
        "candidate_blocks_per_path_mod", "agent_r_mask",
    ]
    for key in required_top:
        assert key in obs, f"Missing Agent-R key: {key}"

    # candidate_paths should be list of lists of ints
    assert isinstance(obs["candidate_paths"], list)
    assert all(isinstance(p, list) and all(isinstance(n, int) for n in p)
               for p in obs["candidate_paths"])

    # required_fs_per_path_mod should be 2D
    assert isinstance(obs["required_fs_per_path_mod"], list)
    for row in obs["required_fs_per_path_mod"]:
        assert isinstance(row, list)
        for val in row:
            assert val is None or isinstance(val, int)

    # candidate_blocks_per_path_mod should be 3D list of tuples
    assert isinstance(obs["candidate_blocks_per_path_mod"], list)
    for p_idx, path_blocks in enumerate(obs["candidate_blocks_per_path_mod"]):
        assert isinstance(path_blocks, list)
        for m_idx, mod_blocks in enumerate(path_blocks):
            assert isinstance(mod_blocks, list)
            for block in mod_blocks:
                assert isinstance(block, tuple) and len(block) == 2
                assert isinstance(block[0], int) and isinstance(block[1], int)

    print("test_agent_r_observation_structure PASSED")


# ------------------------------------------------------------------
# 5. Agent-R mask shape = K * num_modulations * max_blocks
# ------------------------------------------------------------------

def test_agent_r_mask_shape():
    env = _make_env(num_servers=2)
    request = _make_request(num_splits=2)
    obs = build_agent_r_observation(env, request, split_id=0, server_id=0)

    num_paths = len(obs["candidate_paths"])
    num_mods = len(obs["mod_names"])
    num_blocks = env.max_blocks
    expected_len = num_paths * num_mods * num_blocks

    assert len(obs["agent_r_mask"]) == expected_len, \
        f"Expected mask length {expected_len}, got {len(obs['agent_r_mask'])}"
    print("test_agent_r_mask_shape PASSED")


# ------------------------------------------------------------------
# 6. Agent-R mask has at least one True on empty network
# ------------------------------------------------------------------

def test_agent_r_mask_has_true():
    env = _make_env(num_servers=2)
    request = _make_request(num_splits=1)
    obs = build_agent_r_observation(env, request, split_id=0, server_id=0)

    assert np.any(obs["agent_r_mask"]), \
        "Expected at least one True in Agent-R mask on empty network"
    print("test_agent_r_mask_has_true PASSED")


# ------------------------------------------------------------------
# 7. decode_agent_c_action correct
# ------------------------------------------------------------------

def test_decode_agent_c_action():
    # 3 splits, 4 servers → idx = split*4 + server
    assert decode_agent_c_action(0, 4) == (0, 0)
    assert decode_agent_c_action(3, 4) == (0, 3)
    assert decode_agent_c_action(4, 4) == (1, 0)
    assert decode_agent_c_action(7, 4) == (1, 3)
    assert decode_agent_c_action(11, 4) == (2, 3)
    print("test_decode_agent_c_action PASSED")


# ------------------------------------------------------------------
# 8. decode_agent_r_action correct
# ------------------------------------------------------------------

def test_decode_agent_r_action():
    # 2 paths, 3 mods, 4 blocks → idx = path*(3*4) + mod*4 + block
    assert decode_agent_r_action(0, 3, 4) == (0, 0, 0)
    assert decode_agent_r_action(3, 3, 4) == (0, 0, 3)
    assert decode_agent_r_action(4, 3, 4) == (0, 1, 0)
    assert decode_agent_r_action(7, 3, 4) == (0, 1, 3)
    assert decode_agent_r_action(12, 3, 4) == (1, 0, 0)
    assert decode_agent_r_action(23, 3, 4) == (1, 2, 3)
    print("test_decode_agent_r_action PASSED")


# ------------------------------------------------------------------
# 9. Observation builder does NOT mutate env state
# ------------------------------------------------------------------

def test_observation_builder_no_mutation():
    env = _make_env(num_servers=2, capacities=[50.0, 50.0])
    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=5.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 10.0)]),
        DNNRequest(req_id=1, src_node=2, arrival_time=1.0, holding_time=3.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 5.0)]),
    ]
    env.reset(requests)

    # Step once to mutate state
    env.step((0, 0), (0, 0, 0))

    # Snapshot state
    active_conn_before = len(env.active_connections)
    server_loads_before = [s.current_load for s in env.mec.servers]
    total_requests_before = env.stats["total_requests"]
    accepted_before = env.stats["accepted"]

    # Build observations (should NOT mutate)
    req = requests[1]
    obs_c = build_agent_c_observation(env, req)
    obs_r = build_agent_r_observation(env, req, split_id=0, server_id=0)

    # Verify state unchanged
    assert len(env.active_connections) == active_conn_before, \
        f"active_connections changed: {active_conn_before} -> {len(env.active_connections)}"
    for i, s in enumerate(env.mec.servers):
        assert s.current_load == server_loads_before[i], \
            f"Server {i} load changed: {server_loads_before[i]} -> {s.current_load}"
    assert env.stats["total_requests"] == total_requests_before
    assert env.stats["accepted"] == accepted_before

    print("test_observation_builder_no_mutation PASSED")


def test_agent_r_observation_paths_match_env_ordering():
    env = _make_env(num_servers=2)
    request = _make_request(num_splits=1)

    for sort_by in ("km", "hops"):
        env.path_sort_strategy = sort_by
        obs = build_agent_r_observation(env, request, split_id=0, server_id=0)
        expected_paths = env.get_candidate_paths(request.src_node, env.mec.servers[0].node_id)
        assert obs["candidate_paths"] == expected_paths, (
            f"Agent-R candidate paths must match env.get_candidate_paths() for sort={sort_by}"
        )

    print("test_agent_r_observation_paths_match_env_ordering PASSED")


def test_agent_c_observation_server_paths_match_env_ordering():
    env = _make_env(num_servers=2)
    request = _make_request(num_splits=1)

    for sort_by in ("km", "hops"):
        env.path_sort_strategy = sort_by
        obs = build_agent_c_observation(env, request)
        expected = [
            env.get_candidate_paths(request.src_node, server.node_id)
            for server in env.mec.servers
        ]
        assert obs["_per_server_paths"] == expected, (
            f"Agent-C per-server paths must match env.get_candidate_paths() for sort={sort_by}"
        )

    print("test_agent_c_observation_server_paths_match_env_ordering PASSED")


def test_agent_c_mask_masks_overloaded_server():
    """When server is already loaded so that cost > available_compute,
    Agent-C mask should mask that split-server.
    """
    env = _make_env(num_servers=2, capacities=[50.0, 50.0])
    # First request fully loads server0 (cost=50)
    requests_load = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=10.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 50.0)]),
    ]
    env.reset(requests_load)
    env.step((0, 0), (0, 0, 0))

    # Server0 now has current_load=50, available=0
    assert env.mec.servers[0].current_load == 50.0
    assert env.mec.servers[0].available_compute == 0.0

    # New request wants edge_compute_cost=40 on server0 → exceeds available
    new_req = DNNRequest(req_id=1, src_node=2, arrival_time=1.0, holding_time=2.0,
                         deadline_ms=100.0,
                         splits=[SplitProfile(0, 1.0, 0.5, 40.0)])

    obs = build_agent_c_observation(env, new_req)
    mask = obs["agent_c_mask"]

    # split0-server0 should be masked (cost=40 > available=0)
    assert mask[0] == False, f"Expected mask[0]=False for overloaded server0, got {mask[0]}"
    # split0-server1 should still be feasible (server1 is idle)
    assert mask[1] == True, f"Expected mask[1]=True for idle server1, got {mask[1]}"
    print("test_agent_c_mask_masks_overloaded_server PASSED")


def test_agent_r_observation_invalid_split_id():
    """Negative or out-of-range split_id should raise ValueError."""
    env = _make_env(num_servers=2)
    request = _make_request(num_splits=2)

    try:
        build_agent_r_observation(env, request, split_id=-1, server_id=0)
        assert False, "Expected ValueError for split_id=-1"
    except ValueError as e:
        assert "Invalid split_id" in str(e)
        print(f"test_agent_r_observation_invalid_split_id: {e}")

    try:
        build_agent_r_observation(env, request, split_id=5, server_id=0)
        assert False, "Expected ValueError for split_id=5"
    except ValueError as e:
        assert "Invalid split_id" in str(e)
        print("test_agent_r_observation_invalid_split_id PASSED")


def test_agent_r_observation_invalid_server_id():
    """Negative or out-of-range server_id should raise ValueError."""
    env = _make_env(num_servers=2)
    request = _make_request(num_splits=2)

    try:
        build_agent_r_observation(env, request, split_id=0, server_id=-1)
        assert False, "Expected ValueError for server_id=-1"
    except ValueError as e:
        assert "Invalid server_id" in str(e)
        print(f"test_agent_r_observation_invalid_server_id: {e}")

    try:
        build_agent_r_observation(env, request, split_id=0, server_id=5)
        assert False, "Expected ValueError for server_id=5"
    except ValueError as e:
        assert "Invalid server_id" in str(e)
        print("test_agent_r_observation_invalid_server_id PASSED")


if __name__ == "__main__":
    test_agent_c_observation_fields()
    test_agent_c_candidate_features_shape()
    test_agent_c_mask_masks_infeasible()
    test_agent_r_observation_structure()
    test_agent_r_mask_shape()
    test_agent_r_mask_has_true()
    test_decode_agent_c_action()
    test_decode_agent_r_action()
    test_observation_builder_no_mutation()
    test_agent_r_observation_paths_match_env_ordering()
    test_agent_c_observation_server_paths_match_env_ordering()
    test_agent_c_mask_masks_overloaded_server()
    test_agent_r_observation_invalid_split_id()
    test_agent_r_observation_invalid_server_id()
    print("\n=== All observation builder tests PASSED ===")
