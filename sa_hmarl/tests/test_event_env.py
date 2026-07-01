"""Integration tests for the minimal event-driven SMDP environment."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv


def test_env_basic():
    """Minimal integration: 3 requests, all succeed, then release all."""
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES,
        num_servers=3,
        seed=42,
        server_nodes=[0, 3, 5],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)

    # 3 staggered requests
    requests = [
        DNNRequest(
            req_id=0, src_node=1, arrival_time=0.0, holding_time=5.0,
            deadline_ms=100.0,
            splits=[
                SplitProfile(0, 1.0, 0.5, 0.5),   # 1 MB, low bandwidth need
                SplitProfile(1, 2.0, 0.3, 0.7),
            ],
        ),
        DNNRequest(
            req_id=1, src_node=2, arrival_time=2.0, holding_time=3.0,
            deadline_ms=80.0,
            splits=[
                SplitProfile(0, 1.5, 0.5, 0.5),
            ],
        ),
        DNNRequest(
            req_id=2, src_node=4, arrival_time=4.0, holding_time=2.0,
            deadline_ms=60.0,
            splits=[
                SplitProfile(0, 1.0, 0.5, 0.5),
            ],
        ),
    ]

    env.reset(requests)

    # --- Request 0: src=1 → server 0 (node 0) ---
    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"], f"Request 0 failed: {info}"
    print(f"Request 0: success, delay={info['delay_ms']:.3f}ms, fs={info['num_slots']}, mod={info['modulation']}")
    assert not done

    # --- Request 1: src=2 → server 0 (node 0) ---
    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"], f"Request 1 failed: {info}"
    print(f"Request 1: success, delay={info['delay_ms']:.3f}ms, fs={info['num_slots']}, mod={info['modulation']}")
    assert not done

    # --- Request 2: src=4 → server 1 (node 3) ---
    obs, reward, done, info = env.step((0, 1), (0, 0, 0))
    assert info["success"], f"Request 2 failed: {info}"
    print(f"Request 2: success, delay={info['delay_ms']:.3f}ms, fs={info['num_slots']}, mod={info['modulation']}")
    assert done

    # --- Check metrics before release ---
    metrics = env.get_metrics()
    assert metrics["total_requests"] == 3
    assert metrics["accepted"] == 3
    assert metrics["blocked"] == 0
    assert metrics["success_rate"] == 1.0
    assert metrics["num_active_connections"] == 3
    print(f"Metrics before release: accepted={metrics['accepted']}, blocked={metrics['blocked']}, "
          f"avg_delay={metrics['avg_delay_ms']:.3f}ms, active={metrics['num_active_connections']}")

    # --- Advance time to trigger all releases ---
    # Request 0 releases at 0+5=5.0
    # Request 1 releases at 2+3=5.0
    # Request 2 releases at 4+2=6.0
    env.advance_time(10.0)
    assert len(env.active_connections) == 0, "Expected all connections released"

    metrics = env.get_metrics()
    assert metrics["num_active_connections"] == 0
    print(f"Metrics after release: active={metrics['num_active_connections']}, current_time={metrics['current_time']}")
    print("test_env_basic PASSED")


def test_env_block_due_to_server_overload():
    """Server overloaded → block."""
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=1, seed=42,
        server_nodes=[0],
        capacities=[60.0],  # 60 GFLOPS capacity
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    # Two requests both targeting the single server with huge compute cost
    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=5.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 50.0)]),  # 50 GFLOPS
        DNNRequest(req_id=1, src_node=2, arrival_time=1.0, holding_time=3.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 50.0)]),
    ]

    env.reset(requests)

    # Request 0 should succeed (server capacity >= 50)
    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"], f"R0 should succeed: {info}"

    # Request 1 should be blocked (server now full)
    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert not info["success"], f"R1 should be blocked: {info}"
    assert info["reason"] == "server_overload"

    metrics = env.get_metrics()
    assert metrics["accepted"] == 1
    assert metrics["blocked"] == 1
    print(f"test_env_block_due_to_server_overload: accepted={metrics['accepted']}, blocked={metrics['blocked']}")
    print("test_env_block_due_to_server_overload PASSED")


def test_env_release_during_step():
    """Resource release happens automatically between request arrivals."""
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=1, seed=42,
        server_nodes=[0],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    # R0: short holding time, arrives at 0.0
    # R1: arrives at 5.0, after R0 has expired
    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 1.0)]),
        DNNRequest(req_id=1, src_node=2, arrival_time=5.0, holding_time=2.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 1.0)]),
    ]

    env.reset(requests)

    # R0 at t=0.0
    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"]
    assert len(env.active_connections) == 1
    print(f"After R0: active_connections={len(env.active_connections)}, time={env.time}")

    # R1 at t=5.0 — R0 (holding=2.0) should have been released at t=2.0
    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"]
    assert len(env.active_connections) == 1  # R1 only
    assert env.time == 5.0
    print(f"After R1: active_connections={len(env.active_connections)}, time={env.time}")
    print("test_env_release_during_step PASSED")


def test_env_metrics():
    """Metrics accumulate correctly across steps."""
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=2, seed=42,
        server_nodes=[0, 3],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
                   deadline_ms=100.0, splits=[SplitProfile(0, 1.0, 0.5, 1.0)]),
        DNNRequest(req_id=1, src_node=2, arrival_time=1.0, holding_time=2.0,
                   deadline_ms=50.0, splits=[SplitProfile(0, 1.0, 0.5, 1.0)]),
    ]

    env.reset(requests)

    # R0 succeeds
    _, _, _, info0 = env.step((0, 0), (0, 0, 0))
    assert info0["success"]
    delay0 = info0["delay_ms"]

    # R1 succeeds
    _, _, _, info1 = env.step((0, 1), (0, 0, 0))
    assert info1["success"]
    delay1 = info1["delay_ms"]

    metrics = env.get_metrics()
    assert metrics["total_requests"] == 2
    assert metrics["accepted"] == 2
    assert metrics["blocked"] == 0
    assert abs(metrics["avg_delay_ms"] - (delay0 + delay1) / 2.0) < 1e-6
    print(f"test_env_metrics: avg_delay={metrics['avg_delay_ms']:.3f}ms, total={metrics['total_requests']}")
    print("test_env_metrics PASSED")


def test_env_negative_path_index():
    """path_idx=-1 must be rejected (no Python negative indexing)."""
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=1, seed=42,
                     server_nodes=[0])
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 1.0)]),
    ]
    env.reset(requests)

    obs, reward, done, info = env.step((0, 0), (-1, 0, 0))
    assert not info["success"], f"Expected failure for path_idx=-1: {info}"
    assert info["reason"] == "invalid_path"
    print("test_env_negative_path_index PASSED")


def test_env_negative_block_index():
    """block_idx=-1 must be rejected (no Python negative indexing)."""
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=1, seed=42,
                     server_nodes=[0])
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 1.0)]),
    ]
    env.reset(requests)

    obs, reward, done, info = env.step((0, 0), (0, 0, -1))
    assert not info["success"], f"Expected failure for block_idx=-1: {info}"
    assert info["reason"] == "no_suitable_block"
    print("test_env_negative_block_index PASSED")


def test_env_compute_delay_before_allocate():
    """Compute delay must be estimated *before* allocate_task so that a task
    which exactly fills the server's remaining capacity gets a finite delay.
    """
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=1, seed=42,
        server_nodes=[0],
        capacities=[50.0],  # 50 GFLOPS
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    # compute_cost == capacity (50 GFLOPS) → should succeed with finite delay
    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
                   deadline_ms=200.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 50.0)]),
    ]
    env.reset(requests)

    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"], f"Expected success: {info}"
    assert info["edge_compute_ms"] < float('inf'), \
        f"Delay should be finite, got {info['edge_compute_ms']}"
    # delay = (50 / 50) * 20ms = 20ms
    assert abs(info["edge_compute_ms"] - 20.0) < 1e-6, \
        f"Expected ~20ms, got {info['edge_compute_ms']}"
    print(f"test_env_compute_delay_before_allocate: edge_compute_ms={info['edge_compute_ms']:.3f}ms, success")
    print("test_env_compute_delay_before_allocate PASSED")


def test_env_done_with_active_connections_drain():
    """After the last request, done=True but active_connections may still be
    non-empty.  drain() must release all of them.
    """
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=1, seed=42,
                     server_nodes=[0])
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    # One request with long holding time
    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=10.0,
                   deadline_ms=100.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 1.0)]),
    ]
    env.reset(requests)

    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"]
    assert done, "Single request → queue empty → done should be True"
    assert len(env.active_connections) == 1, \
        "Active connection should still exist after last step"

    env.drain()
    assert len(env.active_connections) == 0, "drain() should release all connections"
    print("test_env_done_with_active_connections_drain PASSED")


def test_delay_includes_local_compute():
    """total_delay_ms must include local_compute_ms, not just edge_compute_ms."""
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=1, seed=42,
        server_nodes=[0],
        capacities=[50.0],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    # local_compute_cost=10, edge_compute_cost=1 on 50 GFLOPS server
    # local_compute_ms  = (10 / 100) * 20 = 2.0 ms   (terminal = 2x capacity)
    # edge_compute_ms   = (1  / 50)  * 20 = 0.4 ms
    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
                   deadline_ms=200.0,
                   splits=[SplitProfile(0, 1.0, 10.0, 1.0)]),
    ]
    env.reset(requests)

    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"], f"Expected success: {info}"
    assert "local_compute_ms" in info
    assert "edge_compute_ms" in info
    assert info["delay_ms"] > info["edge_compute_ms"], \
        f"delay_ms ({info['delay_ms']}) should be > edge_compute_ms ({info['edge_compute_ms']})"
    # delay_ms = prop + proc + setup + local + edge
    expected_delay = info["delay_ms"]  # exact value depends on path
    assert expected_delay >= info["local_compute_ms"] + info["edge_compute_ms"]
    print(f"test_delay_includes_local_compute: delay_ms={info['delay_ms']:.3f}, "
          f"local={info['local_compute_ms']:.3f}, edge={info['edge_compute_ms']:.3f}")
    print("test_delay_includes_local_compute PASSED")


def test_edge_compute_exceeds_capacity():
    """If edge_compute_cost > server.compute_capacity, step should fail and
    the server's load must remain unchanged (allocate_task never called).
    """
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=1, seed=42,
        server_nodes=[0],
        capacities=[50.0],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)

    # edge_compute_cost=60 > capacity=50 → infeasible
    requests = [
        DNNRequest(req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
                   deadline_ms=200.0,
                   splits=[SplitProfile(0, 1.0, 0.5, 60.0)]),
    ]
    env.reset(requests)

    server = mec.get_server_by_id(0)
    assert server.current_load == 0.0, "Server should start empty"

    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert not info["success"], f"Expected failure: {info}"
    assert info["reason"] in ("server_saturated", "server_overload")
    assert server.current_load == 0.0, "Server load must not change when task exceeds capacity"
    print(f"test_edge_compute_exceeds_capacity: reason={info['reason']}, load={server.current_load}")
    print("test_edge_compute_exceeds_capacity PASSED")


if __name__ == "__main__":
    test_env_basic()
    test_env_block_due_to_server_overload()
    test_env_release_during_step()
    test_env_metrics()
    test_env_negative_path_index()
    test_env_negative_block_index()
    test_env_compute_delay_before_allocate()
    test_env_done_with_active_connections_drain()
    test_delay_includes_local_compute()
    test_edge_compute_exceeds_capacity()
    print("\n=== All event env tests PASSED ===")
