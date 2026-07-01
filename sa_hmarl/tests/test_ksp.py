"""Test K-Shortest Paths module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import networkx as nx
from sa_hmarl.network.ksp import get_k_shortest_paths
from sa_hmarl.network.optical_network import OpticalNetwork


def _path_cost(graph, path, weight="length_km") -> float:
    """Compute total cost of a path."""
    if len(path) <= 1:
        return 0.0
    return sum(graph[u][v][weight] for u, v in zip(path[:-1], path[1:]))


def _is_simple(path: list) -> bool:
    """Check if path has no repeated nodes."""
    return len(path) == len(set(path))


def test_ksp_basic():
    """On a simple mesh, request k=3 paths."""
    G = nx.Graph()
    # 4-node mesh
    # 0 --10-- 1
    # |        |
    # 10       10
    # |        |
    # 2 --10-- 3
    # Also diagonals
    edges = [
        (0, 1, 10.0), (0, 2, 10.0), (1, 3, 10.0), (2, 3, 10.0),
        (0, 3, 25.0), (1, 2, 25.0),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, length_km=w, weight=w)

    paths = get_k_shortest_paths(G, 0, 3, k=3, weight="length_km")
    assert len(paths) == 3, f"Expected 3 paths, got {len(paths)}: {paths}"
    print(f"test_ksp_basic: {len(paths)} paths found")
    for p in paths:
        print(f"  {p} cost={_path_cost(G, p)}")
    print("test_ksp_basic PASSED")


def test_ksp_simple_paths():
    """All returned paths must be simple (no repeated nodes)."""
    G = nx.Graph()
    edges = [
        (0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0),
        (0, 3, 50.0), (0, 2, 25.0),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, length_km=w, weight=w)

    paths = get_k_shortest_paths(G, 0, 3, k=5, weight="length_km")
    for p in paths:
        assert _is_simple(p), f"Path {p} is not simple"
    print(f"test_ksp_simple_paths: all {len(paths)} paths are simple")
    print("test_ksp_simple_paths PASSED")


def test_ksp_sorted():
    """Paths are returned in ascending order of total cost."""
    G = nx.Graph()
    edges = [
        (0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0),
        (0, 3, 50.0), (0, 2, 25.0), (1, 3, 25.0),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, length_km=w, weight=w)

    paths = get_k_shortest_paths(G, 0, 3, k=5, weight="length_km")
    costs = [_path_cost(G, p) for p in paths]
    assert costs == sorted(costs), f"Costs not sorted: {costs}"
    print(f"test_ksp_sorted: costs = {costs}")
    print("test_ksp_sorted PASSED")


def test_ksp_deduplicated():
    """No duplicate paths in the result."""
    G = nx.Graph()
    edges = [
        (0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0),
        (0, 3, 50.0), (0, 2, 25.0), (1, 3, 25.0),
    ]
    for u, v, w in edges:
        G.add_edge(u, v, length_km=w, weight=w)

    paths = get_k_shortest_paths(G, 0, 3, k=10, weight="length_km")
    unique = set(tuple(p) for p in paths)
    assert len(unique) == len(paths), f"Duplicate paths found: {paths}"
    print(f"test_ksp_deduplicated: {len(paths)} unique paths")
    print("test_ksp_deduplicated PASSED")


def test_ksp_no_path():
    """When src and dst are disconnected, return []."""
    G = nx.Graph()
    G.add_edge(0, 1, length_km=10.0, weight=10.0)
    G.add_edge(2, 3, length_km=10.0, weight=10.0)

    paths = get_k_shortest_paths(G, 0, 3, k=3, weight="length_km")
    assert paths == [], f"Expected [], got {paths}"
    print("test_ksp_no_path PASSED")


def test_ksp_same_node():
    """When src == dst, return [[src]]."""
    G = nx.Graph()
    G.add_edge(0, 1, length_km=10.0, weight=10.0)

    paths = get_k_shortest_paths(G, 0, 0, k=3, weight="length_km")
    assert paths == [[0]], f"Expected [[0]], got {paths}"
    print("test_ksp_same_node PASSED")


def test_ksp_fewer_than_k():
    """When fewer than k paths exist, return all found paths."""
    G = nx.Graph()
    # A simple line: 0--1--2--3
    edges = [(0, 1, 10.0), (1, 2, 10.0), (2, 3, 10.0)]
    for u, v, w in edges:
        G.add_edge(u, v, length_km=w, weight=w)

    paths = get_k_shortest_paths(G, 0, 3, k=10, weight="length_km")
    assert len(paths) == 1, f"Expected 1 path, got {len(paths)}"
    assert paths[0] == [0, 1, 2, 3]
    print("test_ksp_fewer_than_k PASSED")


def test_ksp_nsfnet():
    """On real NSFNET topology, verify k=3 paths."""
    net = OpticalNetwork("nsfnet", num_slots=32)
    paths = get_k_shortest_paths(net.G, 0, 13, k=3, weight="length_km")

    assert len(paths) >= 1, "Expected at least 1 path"
    assert len(paths) <= 3, f"Expected at most 3 paths, got {len(paths)}"

    costs = [_path_cost(net.G, p, "length_km") for p in paths]
    assert costs == sorted(costs), f"Costs not sorted: {costs}"

    for p in paths:
        assert _is_simple(p), f"Path {p} is not simple"

    print(f"test_ksp_nsfnet: {len(paths)} paths from 0→13")
    for p, c in zip(paths, costs):
        print(f"  {p} cost={c:.1f}km")
    print("test_ksp_nsfnet PASSED")


def test_ksp_net1():
    """On Net-1 topology, verify multiple paths exist between nodes."""
    net = OpticalNetwork("net1", num_slots=32)
    paths = get_k_shortest_paths(net.G, 0, 8, k=3, weight="length_km")

    assert len(paths) >= 2, f"Expected >=2 paths, got {len(paths)}"
    costs = [_path_cost(net.G, p, "length_km") for p in paths]
    assert costs == sorted(costs)

    for p in paths:
        assert _is_simple(p)

    print(f"test_ksp_net1: {len(paths)} paths from 0→8")
    for p, c in zip(paths, costs):
        print(f"  {p} cost={c:.1f}km")
    print("test_ksp_net1 PASSED")


def test_ksp_zero_or_negative_k():
    """k <= 0 should return [] immediately."""
    G = nx.Graph()
    G.add_edge(0, 1, length_km=10.0, weight=10.0)
    G.add_edge(1, 2, length_km=10.0, weight=10.0)

    assert get_k_shortest_paths(G, 0, 2, k=0, weight="length_km") == []
    assert get_k_shortest_paths(G, 0, 2, k=-1, weight="length_km") == []
    assert get_k_shortest_paths(G, 0, 2, k=-5, weight="length_km") == []
    print("test_ksp_zero_or_negative_k PASSED")


if __name__ == "__main__":
    test_ksp_basic()
    test_ksp_simple_paths()
    test_ksp_sorted()
    test_ksp_deduplicated()
    test_ksp_no_path()
    test_ksp_same_node()
    test_ksp_fewer_than_k()
    test_ksp_nsfnet()
    test_ksp_net1()
    test_ksp_zero_or_negative_k()
    print("\n=== All KSP tests PASSED ===")
