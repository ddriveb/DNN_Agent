"""Unit tests for multi-resource mean-field statistics (future-work Step 1)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.multi_resource_mean_field import (
    compute_spectrum_mean_field,
    compute_compute_mean_field,
    compute_queue_mean_field,
    compute_multi_resource_mean_field,
    multi_resource_mean_field_to_vector,
)


def _make_env(num_servers=3, num_slots=32):
    net = OpticalNetwork("net1", num_slots=num_slots)
    mec = MECCluster(
        num_nodes=net.NUM_NODES,
        num_servers=num_servers,
        seed=42,
        server_nodes=list(range(num_servers)),
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    return SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)


def test_empty_environment():
    """Multi-resource MF can be computed on a freshly reset env without error."""
    env = _make_env()
    env.reset([])
    mf = compute_multi_resource_mean_field(env)
    assert "spec" in mf
    assert "compute" in mf
    assert "queue" in mf
    vec = multi_resource_mean_field_to_vector(mf)
    assert vec.shape == (15,)
    assert vec.dtype == np.float32
    assert np.all(np.isfinite(vec))
    assert np.all((vec >= 0.0) & (vec <= 1.0))
    print("test_empty_environment PASSED")


def test_vector_shape_dtype_and_range():
    """Vector is 15-dim float32, finite, and in [0, 1]."""
    env = _make_env()
    env.reset([])
    vec = multi_resource_mean_field_to_vector(compute_multi_resource_mean_field(env))
    assert vec.shape == (15,), f"Expected shape (15,), got {vec.shape}"
    assert vec.dtype == np.float32, f"Expected float32, got {vec.dtype}"
    assert np.all(np.isfinite(vec)), f"Non-finite values: {vec}"
    assert np.all(vec >= 0.0) and np.all(vec <= 1.0), f"Values out of [0,1]: {vec}"
    print("test_vector_shape_dtype_and_range PASSED")


def test_compute_ratios_sum_to_one():
    """Idle/normal/busy/overload ratios partition the server set."""
    env = _make_env(num_servers=4)
    env.reset([])
    # Manually set utilizations to cover all bins.
    env.mec.servers[0].current_load = 0.10 * env.mec.servers[0].compute_capacity  # idle
    env.mec.servers[1].current_load = 0.50 * env.mec.servers[1].compute_capacity  # normal
    env.mec.servers[2].current_load = 0.80 * env.mec.servers[2].compute_capacity  # busy
    env.mec.servers[3].current_load = 0.95 * env.mec.servers[3].compute_capacity  # overload

    cmf = compute_compute_mean_field(env)
    ratio_sum = (
        cmf["idle_server_ratio"]
        + cmf["normal_server_ratio"]
        + cmf["busy_server_ratio"]
        + cmf["overload_risk_server_ratio"]
    )
    assert abs(ratio_sum - 1.0) < 1e-6, f"Ratios sum to {ratio_sum}"
    assert cmf["idle_server_ratio"] == 0.25
    assert cmf["normal_server_ratio"] == 0.25
    assert cmf["busy_server_ratio"] == 0.25
    assert cmf["overload_risk_server_ratio"] == 0.25
    print("test_compute_ratios_sum_to_one PASSED")


def test_utilization_change_affects_compute_ratios():
    """Increasing server utilization moves servers from idle to busy/overload."""
    env = _make_env(num_servers=2)
    env.reset([])
    env.mec.servers[0].current_load = 0.0
    env.mec.servers[1].current_load = 0.0
    cmf_low = compute_compute_mean_field(env)
    assert cmf_low["idle_server_ratio"] == 1.0
    assert cmf_low["overload_risk_server_ratio"] == 0.0

    env.mec.servers[0].current_load = 0.95 * env.mec.servers[0].compute_capacity
    env.mec.servers[1].current_load = 0.95 * env.mec.servers[1].compute_capacity
    cmf_high = compute_compute_mean_field(env)
    assert cmf_high["idle_server_ratio"] == 0.0
    assert cmf_high["overload_risk_server_ratio"] == 1.0
    print("test_utilization_change_affects_compute_ratios PASSED")


def test_spectrum_occupancy_reduces_free_ratio():
    """Allocating slots on a link reduces mean_free_slot_ratio."""
    env = _make_env(num_slots=32)
    env.reset([])
    free_before = compute_spectrum_mean_field(env)["mean_free_slot_ratio"]

    # Allocate half the slots on one link.
    edges = list(env.net.G.edges())
    assert len(edges) > 0
    u, v = edges[0]
    path = [u, v]
    success = env.net.allocate(path, start_slot=0, num_slots=16)
    assert success

    free_after = compute_spectrum_mean_field(env)["mean_free_slot_ratio"]
    assert free_after < free_before, (
        f"Expected free ratio to drop, got before={free_before}, after={free_after}"
    )
    print("test_spectrum_occupancy_reduces_free_ratio PASSED")


def test_queue_delay_change():
    """Increasing queue delay EMA raises long_queue_ratio and max delay norm."""
    env = _make_env(num_servers=2)
    env.reset([])
    env.mec.servers[0]._queue_delay_ema = 5.0
    env.mec.servers[1]._queue_delay_ema = 5.0
    qmf_low = compute_queue_mean_field(env)

    env.mec.servers[0]._queue_delay_ema = 100.0
    env.mec.servers[1]._queue_delay_ema = 5.0
    qmf_high = compute_queue_mean_field(env)

    assert qmf_high["long_queue_ratio"] >= qmf_low["long_queue_ratio"]
    assert qmf_high["max_queue_delay_norm"] > qmf_low["max_queue_delay_norm"]
    print("test_queue_delay_change PASSED")


def test_vector_order():
    """Vector ordering matches the specification exactly."""
    env = _make_env()
    env.reset([])
    mf = compute_multi_resource_mean_field(env)
    vec = multi_resource_mean_field_to_vector(mf)

    expected = [
        mf["spec"]["small_block_ratio"],
        mf["spec"]["medium_block_ratio"],
        mf["spec"]["large_block_ratio"],
        mf["spec"]["mean_largest_free_block_norm"],
        mf["spec"]["mean_fragmentation"],
        mf["spec"]["mean_free_slot_ratio"],
        mf["compute"]["idle_server_ratio"],
        mf["compute"]["normal_server_ratio"],
        mf["compute"]["busy_server_ratio"],
        mf["compute"]["overload_risk_server_ratio"],
        mf["compute"]["mean_available_compute_ratio"],
        mf["queue"]["mean_queue_delay_norm"],
        mf["queue"]["max_queue_delay_norm"],
        mf["queue"]["short_queue_ratio"],
        mf["queue"]["long_queue_ratio"],
    ]
    assert np.allclose(vec, np.array(expected, dtype=np.float32)), "Vector order mismatch"
    print("test_vector_order PASSED")


if __name__ == "__main__":
    test_empty_environment()
    test_vector_shape_dtype_and_range()
    test_compute_ratios_sum_to_one()
    test_utilization_change_affects_compute_ratios()
    test_spectrum_occupancy_reduces_free_ratio()
    test_queue_delay_change()
    test_vector_order()
    print("\n=== All multi-resource mean-field tests PASSED ===")
