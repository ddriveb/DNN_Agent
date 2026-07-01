"""Test spectrum summary extraction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.env.spectrum_summary import compute_server_direction_summary
from sa_hmarl.env.fs_demand import (
    DEFAULT_PROP_SPEED_KM_S,
    DEFAULT_PROC_PER_HOP_S,
    DEFAULT_SETUP_TIME_S,
)


def test_summary_empty_network():
    net = OpticalNetwork("nsfnet", num_slots=32, seed=42)
    paths = [[0, 1, 2], [0, 5, 4, 2]]
    summary = compute_server_direction_summary(net, paths)

    assert summary.lfb_max == 32.0
    assert summary.lfb_mean == 32.0
    assert summary.frag_mean == 0.0
    assert summary.free_mean == 1.0
    assert summary.num_feas_paths == 2
    assert summary.dim == 10
    vec = summary.to_vector()
    assert len(vec) == 10
    print("test_summary_empty_network PASSED")


def test_summary_loaded_network():
    net = OpticalNetwork("nsfnet", num_slots=32, seed=42)
    path1 = [0, 1, 2]
    path2 = [0, 5, 4, 2]

    net.allocate(path1, 5, 10)

    summary = compute_server_direction_summary(net, [path1, path2])
    assert summary.lfb_max == 32.0
    assert summary.lfb_min < 32.0
    assert summary.lfb_mean < 32.0
    print(f"test_summary_loaded_network: lfb_mean={summary.lfb_mean}, frag_mean={summary.frag_mean:.3f}")
    print("test_summary_loaded_network PASSED")


def test_min_delay_est_components():
    """min_delay_est includes propagation + hop_processing + setup."""
    net = OpticalNetwork("nsfnet", num_slots=32, seed=42)
    # Path [0, 1]: 1 hop, 1100 km
    summary = compute_server_direction_summary(net, [[0, 1]])

    expected_prop = 1100.0 / DEFAULT_PROP_SPEED_KM_S
    expected_proc = 1 * DEFAULT_PROC_PER_HOP_S
    expected_setup = DEFAULT_SETUP_TIME_S
    expected = expected_prop + expected_proc + expected_setup

    assert abs(summary.min_delay_est - expected) < 1e-9
    print(f"test_min_delay_est_components: {summary.min_delay_est*1e3:.3f}ms "
          f"(prop={expected_prop*1e3:.3f}ms, proc={expected_proc*1e3:.3f}ms, setup={expected_setup*1e3:.3f}ms)")
    print("test_min_delay_est_components PASSED")


def test_min_delay_est_increases_with_path_length():
    """Longer physical distance → larger min_delay_est."""
    net = OpticalNetwork("nsfnet", num_slots=32, seed=42)
    # [0, 1] = 1100 km (1 hop)
    # [0, 5] = 2800 km (1 hop)
    summary_short = compute_server_direction_summary(net, [[0, 1]])
    summary_long = compute_server_direction_summary(net, [[0, 5]])

    assert summary_long.min_delay_est > summary_short.min_delay_est
    print(f"test_min_delay_est_increases_with_path_length: "
          f"short={summary_short.min_delay_est*1e3:.3f}ms, long={summary_long.min_delay_est*1e3:.3f}ms")
    print("test_min_delay_est_increases_with_path_length PASSED")


def test_min_delay_est_increases_with_hops():
    """More hops → larger min_delay_est (even if distance is shorter)."""
    # Use net1 where we can construct paths with different hop counts
    net = OpticalNetwork("net1", num_slots=32, seed=42)
    # [0, 1] = 1 hop, 10 km
    # [0, 2, 1] = 2 hops, 6+8 = 14 km
    summary_1hop = compute_server_direction_summary(net, [[0, 1]])
    summary_2hop = compute_server_direction_summary(net, [[0, 2, 1]])

    assert summary_2hop.min_delay_est > summary_1hop.min_delay_est
    print(f"test_min_delay_est_increases_with_hops: "
          f"1hop={summary_1hop.min_delay_est*1e3:.3f}ms, 2hop={summary_2hop.min_delay_est*1e3:.3f}ms")
    print("test_min_delay_est_increases_with_hops PASSED")


def test_min_delay_est_same_hops_different_dist():
    """Same hop count but longer distance → larger delay."""
    net = OpticalNetwork("nsfnet", num_slots=32, seed=42)
    # Both are 1 hop, but different distances
    # [0, 1] = 1100 km
    # [11, 12] = 300 km
    summary_long = compute_server_direction_summary(net, [[0, 1]])
    summary_short = compute_server_direction_summary(net, [[11, 12]])

    assert summary_long.min_delay_est > summary_short.min_delay_est
    print(f"test_min_delay_est_same_hops_different_dist: "
          f"long={summary_long.min_delay_est*1e3:.3f}ms, short={summary_short.min_delay_est*1e3:.3f}ms")
    print("test_min_delay_est_same_hops_different_dist PASSED")


if __name__ == "__main__":
    test_summary_empty_network()
    test_summary_loaded_network()
    test_min_delay_est_components()
    test_min_delay_est_increases_with_path_length()
    test_min_delay_est_increases_with_hops()
    test_min_delay_est_same_hops_different_dist()
    print("\n=== All spectrum summary tests PASSED ===")
