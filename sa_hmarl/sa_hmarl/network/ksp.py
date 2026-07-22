"""K-Shortest Paths (Yen's algorithm) for Agent-R candidate path generation.

Returns up to k simple paths.  By default paths are ordered by total edge
weight.  Evaluation baselines can optionally re-order the returned candidate
set by hop count to reproduce hops-ordered KSP-FF comparisons.
"""
import networkx as nx
from typing import List


def get_k_shortest_paths(
    graph,
    src: int,
    dst: int,
    k: int,
    weight: str = "length_km",
    sort_by: str = "km",
) -> List[List[int]]:
    """Return up to k shortest simple paths from src to dst.

    Uses Yen's algorithm with graph copies.  Paths are guaranteed to be
    simple (no repeated nodes), deduplicated, and sorted by the requested
    output criterion.

    Args:
        graph: A networkx Graph.  Each edge must have the specified weight attribute.
        src: Source node.
        dst: Destination node.
        k: Maximum number of paths to return.
        weight: Edge attribute to use as path cost.  Default "length_km".
        sort_by: Output ordering.  ``"km"`` preserves the classic weighted
            KSP order.  ``"hops"`` sorts by hop count, then by path length.

    Returns:
        List of paths, each path is a list of node IDs.
        - If k <= 0: returns [].
        - If src == dst: returns [[src]].
        - If no path exists: returns [].
        - If fewer than k paths exist: returns all found paths.
    """
    if k <= 0:
        return []

    if src == dst:
        return [[src]]
    if sort_by not in ("km", "hops"):
        raise ValueError(f"Unknown path sort strategy: {sort_by}")

    def _path_cost(path: List[int]) -> float:
        return sum(graph[u][v][weight] for u, v in zip(path[:-1], path[1:]))

    if sort_by == "hops":
        try:
            path_iter = nx.shortest_simple_paths(graph, src, dst, weight=None)
            paths = []
            for path in path_iter:
                paths.append(path)
                if len(paths) >= k:
                    break
        except nx.NetworkXNoPath:
            return []
        paths.sort(key=lambda path: (max(len(path) - 1, 0), _path_cost(path), tuple(path)))
        return paths

    try:
        shortest = nx.shortest_path(graph, src, dst, weight=weight)
    except nx.NetworkXNoPath:
        return []

    paths = [shortest]
    candidates = []

    for i in range(1, k):
        if i > len(paths):
            break

        prev_path = paths[i - 1]

        for j in range(len(prev_path) - 1):
            spur_node = prev_path[j]
            root_path = prev_path[:j + 1]

            # Work on a fresh copy for this spur iteration
            H = graph.copy()

            # Remove edges that would recreate already-found paths with same root
            for p in paths:
                if len(p) > j and p[:j + 1] == root_path:
                    u, v = p[j], p[j + 1]
                    if H.has_edge(u, v):
                        H.remove_edge(u, v)

            # Remove root_path nodes (except spur_node) to prevent cycles
            for node in root_path[:-1]:
                if node in H and node != spur_node:
                    H.remove_node(node)

            try:
                spur_path = nx.shortest_path(H, spur_node, dst, weight=weight)
                total_path = root_path[:-1] + spur_path

                # Ensure simple path (no repeated nodes)
                if len(total_path) == len(set(total_path)):
                    if total_path not in candidates and total_path not in paths:
                        candidates.append(total_path)
            except nx.NetworkXNoPath:
                pass

        if not candidates:
            break

        candidates.sort(key=_path_cost)
        paths.append(candidates.pop(0))

    paths.sort(key=lambda path: (_path_cost(path), tuple(path)))

    return paths
