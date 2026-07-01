"""Test enhanced optical network with fixed topologies."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.topology_data import list_supported_topologies


def test_supported_topologies():
    names = list_supported_topologies()
    assert "nsfnet" in names
    assert "usnet" in names
    assert "net1" in names
    assert "net2" in names
    assert "net3" in names
    assert "metro24_c_sensitive" in names
    print(f"test_supported_topologies: {names}")


def test_basic_allocation():
    net = OpticalNetwork("nsfnet", num_slots=32)
    path = [0, 1, 2]

    # Allocate 4 slots starting at 5
    ok = net.allocate(path, 5, 4)
    assert ok

    # Try overlapping allocation
    ok2 = net.allocate(path, 7, 2)
    assert not ok2

    # Non-overlapping should succeed
    ok3 = net.allocate(path, 10, 2)
    assert ok3

    net.release(path, 5, 4)
    ok4 = net.allocate(path, 7, 2)
    assert ok4

    print("test_basic_allocation PASSED")


def test_candidate_blocks_on_path():
    net = OpticalNetwork("nsfnet", num_slots=32)
    path = [0, 1, 2]

    # Occupy slots 5-8 on the path
    net.allocate(path, 5, 4)

    blocks = net.get_candidate_blocks(path, req_fs=3, max_candidates=5)
    assert len(blocks) > 0
    for b in blocks:
        assert b.size >= 3

    print("test_candidate_blocks_on_path PASSED")


def test_modulation_feasibility():
    net = OpticalNetwork("nsfnet", num_slots=32)
    reg = ModulationRegistry()
    path = [0, 1, 2]
    feas = net.get_feasible_modulations(path, reg)
    assert len(feas) >= 1
    print("test_modulation_feasibility PASSED")


def test_path_stats():
    net = OpticalNetwork("nsfnet", num_slots=32)
    path = [0, 1, 2]
    stats = net.get_path_spectrum_stats(path)
    assert "frag_index" in stats
    assert stats["max_free_slots"] == 32  # Empty network
    print("test_path_stats PASSED")


# ------------------------------------------------------------------
# Fixed-topology specific tests
# ------------------------------------------------------------------

def test_nsfnet_fixed_lengths():
    """NSFNET link lengths are fixed and known."""
    net = OpticalNetwork("nsfnet", num_slots=32)
    # 0-1 should be 1100 km
    assert net.G.edges[(0, 1)]["length_km"] == 1100.0
    assert net.G.edges[(0, 1)]["weight"] == 1100.0
    # 0-5 should be 2800 km
    assert net.G.edges[(0, 5)]["length_km"] == 2800.0
    # 11-12 should be 300 km
    assert net.G.edges[(11, 12)]["length_km"] == 300.0
    print("test_nsfnet_fixed_lengths PASSED")


def test_usnet_fixed_lengths():
    """USNET link lengths are fixed."""
    net = OpticalNetwork("usnet", num_slots=32)
    # Check a few edges
    assert net.G.edges[(0, 1)]["length_km"] == 300.0
    assert net.G.edges[(3, 20)]["length_km"] == 700.0
    assert net.G.edges[(26, 27)]["length_km"] == 300.0
    print("test_usnet_fixed_lengths PASSED")


def test_net1_net2_net3_load():
    """Net-1/2/3 from Yin 2024 can be loaded successfully."""
    for topo in ["net1", "net2", "net3"]:
        net = OpticalNetwork(topo, num_slots=100)
        assert net.NUM_NODES > 0
        assert net.G.number_of_edges() > 0
        # All edges should have positive length
        for u, v, data in net.G.edges(data=True):
            assert data["length_km"] > 0
            assert data["weight"] == data["length_km"]
    print("test_net1_net2_net3_load PASSED")


def test_reproducible_across_seeds():
    """Same topology with different seeds produces identical edge lengths."""
    net1 = OpticalNetwork("nsfnet", num_slots=32, seed=42)
    net2 = OpticalNetwork("nsfnet", num_slots=32, seed=999)

    for u, v in net1.G.edges():
        assert net1.G.edges[(u, v)]["length_km"] == net2.G.edges[(u, v)]["length_km"]
        assert net1.G.edges[(u, v)]["weight"] == net2.G.edges[(u, v)]["weight"]

    # Also verify path length is identical
    path = [0, 1, 2]
    assert net1.path_length_km(path) == net2.path_length_km(path)
    assert net1.path_length_km(path) == 1100.0 + 600.0  # 0-1 + 1-2

    print("test_reproducible_across_seeds PASSED")


def test_path_length_deterministic():
    """path_length_km is determined by fixed edge lengths, not random."""
    net = OpticalNetwork("nsfnet", num_slots=32)
    path = [0, 5, 7, 8]
    length = net.path_length_km(path)
    expected = 2800.0 + 1200.0 + 800.0  # 0-5 + 5-7 + 7-8
    assert length == expected, f"Expected {expected}, got {length}"
    print(f"test_path_length_deterministic: path {path} = {length} km")


def test_net1_path_lengths():
    """Net-1 path lengths match Yin 2024 data."""
    net = OpticalNetwork("net1", num_slots=32)
    # 0-1-3: 10 + 12 = 22 km
    assert net.path_length_km([0, 1, 3]) == 22.0
    # 2-5-7: 4 + 12 = 16 km
    assert net.path_length_km([2, 5, 7]) == 16.0
    print("test_net1_path_lengths PASSED")


def test_metro24_c_sensitive_design():
    """C-sensitive metro topology should stay metro-scale but non-trivial."""
    net = OpticalNetwork("metro24_c_sensitive", num_slots=24)
    assert net.NUM_NODES == 24
    assert net.G.number_of_edges() >= 35
    for _, _, data in net.G.edges(data=True):
        assert 20.0 <= data["length_km"] <= 250.0
    # Sparse gateways from server0's region into the core.
    assert net.G.has_edge(3, 6)
    assert net.G.has_edge(5, 10)
    # Spectrum-friendly core around server1 has multiple alternatives.
    assert net.G.has_edge(6, 9)
    assert net.G.has_edge(7, 10)
    assert net.G.has_edge(8, 11)
    print("test_metro24_c_sensitive_design PASSED")


def test_routing_uses_length_as_weight():
    """KSP should use length_km as weight, not uniform weight."""
    import networkx as nx
    net = OpticalNetwork("nsfnet", num_slots=32)
    # Shortest path from 0 to 2 by hop count: 0-1-2 (2 hops) or 0-2 (1 hop)
    # By length: 0-2 = 1600, 0-1-2 = 1100+600 = 1700
    # So 0-2 should be shortest by length
    sp = nx.shortest_path(net.G, 0, 2, weight="weight")
    assert sp == [0, 2], f"Expected [0, 2] by length, got {sp}"
    print("test_routing_uses_length_as_weight PASSED")


if __name__ == "__main__":
    test_supported_topologies()
    test_basic_allocation()
    test_candidate_blocks_on_path()
    test_modulation_feasibility()
    test_path_stats()
    test_nsfnet_fixed_lengths()
    test_usnet_fixed_lengths()
    test_net1_net2_net3_load()
    test_reproducible_across_seeds()
    test_path_length_deterministic()
    test_net1_path_lengths()
    test_metro24_c_sensitive_design()
    test_routing_uses_length_as_weight()
    print("\n=== All optical network tests PASSED ===")
