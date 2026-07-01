#!/usr/bin/env python3
"""Generate SA-HMARL topology snippets from SNAP-style edge lists.

SNAP datasets usually provide graph structure only. SA-HMARL optical-network
experiments additionally need reproducible link distances and MEC server
placements. This script bridges that gap by:

1. Loading a SNAP-style edge list.
2. Extracting a connected induced subgraph of a target size.
3. Relabeling nodes to contiguous IDs.
4. Assigning synthetic metro-scale link lengths.
5. Suggesting server nodes and heterogeneous capacities.
6. Exporting a Python snippet that can be pasted into topology_data.py.

The output is intentionally "SNAP-derived synthetic optical topology", not a
claim that the original SNAP graph contains physical optical-link distances.
"""
from __future__ import annotations

import argparse
import math
from collections import Counter, deque
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Set, Tuple

import numpy as np

Edge = Tuple[int, int]
WeightedEdge = Tuple[int, int, float]


def parse_edge_list(path: Path, undirected: bool = True) -> Tuple[Set[int], Set[Edge]]:
    """Read a whitespace/comma separated edge list.

    Lines beginning with # or % are ignored. Only the first two columns are
    used, matching common SNAP text files.
    """
    nodes: Set[int] = set()
    edges: Set[Edge] = set()

    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or line.startswith("%"):
                continue
            parts = line.replace(",", " ").split()
            if len(parts) < 2:
                continue
            try:
                u = int(parts[0])
                v = int(parts[1])
            except ValueError:
                continue
            if u == v:
                continue
            nodes.update((u, v))
            if undirected:
                a, b = sorted((u, v))
                edges.add((a, b))
            else:
                edges.add((u, v))

    return nodes, edges


def build_adj(nodes: Iterable[int], edges: Iterable[Edge]) -> Dict[int, Set[int]]:
    adj = {n: set() for n in nodes}
    for u, v in edges:
        adj.setdefault(u, set()).add(v)
        adj.setdefault(v, set()).add(u)
    return adj


def connected_components(adj: Dict[int, Set[int]]) -> List[Set[int]]:
    seen: Set[int] = set()
    comps: List[Set[int]] = []

    for start in adj:
        if start in seen:
            continue
        q = deque([start])
        seen.add(start)
        comp = {start}
        while q:
            u = q.popleft()
            for v in adj[u]:
                if v not in seen:
                    seen.add(v)
                    comp.add(v)
                    q.append(v)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def bfs_subgraph(adj: Dict[int, Set[int]], start: int, target_nodes: int,
                 rng: np.random.RandomState) -> Set[int]:
    """Grow a connected node set by randomized BFS."""
    selected = {start}
    frontier = list(adj[start])
    rng.shuffle(frontier)

    while frontier and len(selected) < target_nodes:
        u = frontier.pop(0)
        if u in selected:
            continue
        selected.add(u)
        nbrs = [v for v in adj[u] if v not in selected]
        rng.shuffle(nbrs)
        frontier.extend(nbrs)

    return selected


def sample_connected_subgraph(nodes: Set[int], edges: Set[Edge], target_nodes: int,
                              seed: int, min_avg_degree: float) -> Tuple[List[int], Set[Edge]]:
    """Sample a connected subgraph, preferring non-tree-like samples."""
    adj = build_adj(nodes, edges)
    comps = connected_components(adj)
    if not comps or len(comps[0]) < target_nodes:
        raise ValueError(
            f"Largest connected component has {len(comps[0]) if comps else 0} nodes, "
            f"cannot extract {target_nodes} nodes."
        )

    giant = comps[0]
    giant_adj = {n: adj[n] & giant for n in giant}
    rng = np.random.RandomState(seed)
    candidates = list(giant)

    best_nodes: Set[int] = set()
    best_edges: Set[Edge] = set()
    best_avg_deg = -1.0

    for _ in range(200):
        start = int(rng.choice(candidates))
        sub_nodes = bfs_subgraph(giant_adj, start, target_nodes, rng)
        if len(sub_nodes) < target_nodes:
            continue
        sub_edges = {
            tuple(sorted((u, v)))
            for u in sub_nodes
            for v in giant_adj[u]
            if u < v and v in sub_nodes
        }
        avg_deg = 2.0 * len(sub_edges) / len(sub_nodes)
        if avg_deg > best_avg_deg:
            best_nodes = sub_nodes
            best_edges = sub_edges
            best_avg_deg = avg_deg
        if avg_deg >= min_avg_degree:
            break

    if not best_nodes:
        raise RuntimeError("Failed to sample a connected subgraph.")

    return sorted(best_nodes), best_edges


def relabel(nodes: Sequence[int], edges: Set[Edge]) -> Tuple[Dict[int, int], Set[Edge]]:
    mapping = {old: new for new, old in enumerate(nodes)}
    relabeled = {tuple(sorted((mapping[u], mapping[v]))) for u, v in edges}
    return mapping, relabeled


def bfs_distances(adj: Dict[int, Set[int]], source: int) -> Dict[int, int]:
    dist = {source: 0}
    q = deque([source])
    while q:
        u = q.popleft()
        for v in adj[u]:
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def closeness_scores(adj: Dict[int, Set[int]]) -> Dict[int, float]:
    scores = {}
    n = len(adj)
    for u in adj:
        dist = bfs_distances(adj, u)
        total = sum(dist.values())
        scores[u] = (n - 1) / total if total > 0 and len(dist) == n else 0.0
    return scores


def approximate_betweenness(adj: Dict[int, Set[int]]) -> Dict[int, float]:
    """Small-graph Brandes betweenness centrality."""
    bet = {v: 0.0 for v in adj}
    for s in adj:
        stack: List[int] = []
        pred = {w: [] for w in adj}
        sigma = {w: 0.0 for w in adj}
        sigma[s] = 1.0
        dist = {w: -1 for w in adj}
        dist[s] = 0
        q = deque([s])
        while q:
            v = q.popleft()
            stack.append(v)
            for w in adj[v]:
                if dist[w] < 0:
                    q.append(w)
                    dist[w] = dist[v] + 1
                if dist[w] == dist[v] + 1:
                    sigma[w] += sigma[v]
                    pred[w].append(v)
        delta = {w: 0.0 for w in adj}
        while stack:
            w = stack.pop()
            for v in pred[w]:
                if sigma[w] > 0:
                    delta[v] += (sigma[v] / sigma[w]) * (1.0 + delta[w])
            if w != s:
                bet[w] += delta[w]
    return {k: v / 2.0 for k, v in bet.items()}


def bridge_edges(adj: Dict[int, Set[int]]) -> Set[Edge]:
    """Tarjan bridge detection for an undirected graph."""
    time = 0
    tin: Dict[int, int] = {}
    low: Dict[int, int] = {}
    bridges: Set[Edge] = set()

    def dfs(u: int, parent: int) -> None:
        nonlocal time
        time += 1
        tin[u] = low[u] = time
        for v in adj[u]:
            if v == parent:
                continue
            if v in tin:
                low[u] = min(low[u], tin[v])
            else:
                dfs(v, u)
                low[u] = min(low[u], low[v])
                if low[v] > tin[u]:
                    bridges.add(tuple(sorted((u, v))))

    for node in adj:
        if node not in tin:
            dfs(node, -1)
    return bridges


def assign_lengths(edges: Set[Edge], adj: Dict[int, Set[int]], seed: int,
                   min_km: float, max_km: float) -> List[WeightedEdge]:
    """Assign deterministic synthetic metro-scale lengths.

    Bridges and low-degree peripheral edges are made slightly longer to create
    optical pressure without introducing NSFNET-like reach domination.
    """
    rng = np.random.RandomState(seed)
    bridges = bridge_edges(adj)
    lengths: List[WeightedEdge] = []

    for u, v in sorted(edges):
        du = len(adj[u])
        dv = len(adj[v])
        base = rng.uniform(min_km, max_km)
        if (u, v) in bridges:
            base *= 1.25
        if min(du, dv) <= 2:
            base *= 1.10
        length = float(np.clip(round(base / 5.0) * 5.0, min_km, max_km))
        lengths.append((u, v, length))
    return lengths


def choose_servers(adj: Dict[int, Set[int]], num_servers: int) -> Tuple[List[int], List[float]]:
    """Pick heterogeneous server nodes from structural roles."""
    degree = {n: len(adj[n]) for n in adj}
    close = closeness_scores(adj)
    between = approximate_betweenness(adj)

    nodes = list(adj)
    high_compute_bottleneck = max(nodes, key=lambda n: (between[n], -degree[n]))
    meshed_core = max(nodes, key=lambda n: (degree[n], close[n]))
    short_local = min(nodes, key=lambda n: (degree[n], -close[n]))
    peripheral_high_compute = min(nodes, key=lambda n: (close[n], -degree[n]))

    ordered = [high_compute_bottleneck, meshed_core, short_local, peripheral_high_compute]
    servers: List[int] = []
    for node in ordered:
        if node not in servers:
            servers.append(node)
    for node in sorted(nodes, key=lambda n: (-degree[n], -close[n])):
        if len(servers) >= num_servers:
            break
        if node not in servers:
            servers.append(node)

    capacities_template = [90.0, 55.0, 32.0, 85.0, 60.0, 45.0, 75.0, 40.0]
    capacities = capacities_template[:num_servers]
    return servers[:num_servers], capacities


def format_python_snippet(name: str, weighted_edges: Sequence[WeightedEdge],
                          servers: Sequence[int], capacities: Sequence[float]) -> str:
    const_name = name.upper().replace("-", "_") + "_EDGES"
    lines = [
        f"# SNAP-derived synthetic optical topology: {name}",
        f"# Suggested server_nodes={list(servers)}",
        f"# Suggested capacities={list(capacities)}",
        f"{const_name}: List[Tuple[int, int, float]] = [",
    ]
    for u, v, length in weighted_edges:
        lines.append(f"    ({u:2d}, {v:2d}, {length:6.1f}),")
    lines.append("]")
    lines.append("")
    lines.append(f'# Add to TOPOLOGY_REGISTRY: "{name}": {const_name},')
    return "\n".join(lines)


def write_edge_csv(path: Path, weighted_edges: Sequence[WeightedEdge]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        f.write("u,v,length_km\n")
        for u, v, length in weighted_edges:
            f.write(f"{u},{v},{length:.1f}\n")


def print_stats(adj: Dict[int, Set[int]], weighted_edges: Sequence[WeightedEdge],
                servers: Sequence[int], capacities: Sequence[float],
                original_nodes: int, original_edges: int) -> None:
    degrees = [len(adj[n]) for n in adj]
    lengths = [e[2] for e in weighted_edges]
    bridges = bridge_edges(adj)
    degree_hist = Counter(degrees)

    print("=" * 80)
    print("SNAP-Derived Topology Summary")
    print("=" * 80)
    print(f"Original graph: nodes={original_nodes}, edges={original_edges}")
    print(f"Subgraph:       nodes={len(adj)}, edges={len(weighted_edges)}")
    print(f"Avg degree:     {2 * len(weighted_edges) / len(adj):.2f}")
    print(f"Degree range:   {min(degrees)}..{max(degrees)}")
    print(f"Degree hist:    {dict(sorted(degree_hist.items()))}")
    print(f"Bridge edges:   {len(bridges)}")
    print(f"Length range:   {min(lengths):.1f}..{max(lengths):.1f} km")
    print(f"Length avg:     {np.mean(lengths):.1f} km")
    print(f"Servers:        {list(servers)}")
    print(f"Capacities:     {list(capacities)}")
    print("=" * 80)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert a SNAP-style edge list into an SA-HMARL topology snippet."
    )
    parser.add_argument("--input", required=True, type=Path,
                        help="SNAP-style edge list path")
    parser.add_argument("--name", default="snap24_synthetic",
                        help="Topology name / Python constant prefix")
    parser.add_argument("--nodes", type=int, default=24,
                        help="Number of nodes to extract")
    parser.add_argument("--servers", type=int, default=4,
                        help="Number of MEC servers to suggest")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min_avg_degree", type=float, default=3.0,
                        help="Try to sample a connected subgraph at least this dense")
    parser.add_argument("--min_km", type=float, default=20.0)
    parser.add_argument("--max_km", type=float, default=250.0)
    parser.add_argument("--directed", action="store_true",
                        help="Keep directed edges while parsing, then symmetrize for analysis")
    parser.add_argument("--output_py", type=Path, default=None,
                        help="Write topology_data.py snippet to this file")
    parser.add_argument("--output_csv", type=Path, default=None,
                        help="Write generated weighted edge CSV")
    args = parser.parse_args()

    nodes, edges = parse_edge_list(args.input, undirected=not args.directed)
    if args.directed:
        edges = {tuple(sorted((u, v))) for u, v in edges if u != v}
        nodes = {n for e in edges for n in e}

    sub_nodes, sub_edges = sample_connected_subgraph(
        nodes, edges, args.nodes, args.seed, args.min_avg_degree
    )
    _, relabeled_edges = relabel(sub_nodes, sub_edges)
    relabeled_nodes = set(range(len(sub_nodes)))
    adj = build_adj(relabeled_nodes, relabeled_edges)
    weighted = assign_lengths(
        relabeled_edges, adj, args.seed + 17, args.min_km, args.max_km
    )
    servers, capacities = choose_servers(adj, args.servers)

    print_stats(adj, weighted, servers, capacities, len(nodes), len(edges))
    snippet = format_python_snippet(args.name, weighted, servers, capacities)
    print(snippet)

    if args.output_py:
        args.output_py.parent.mkdir(parents=True, exist_ok=True)
        args.output_py.write_text(snippet + "\n", encoding="utf-8")
        print(f"\nWrote Python snippet to {args.output_py}")

    if args.output_csv:
        write_edge_csv(args.output_csv, weighted)
        print(f"Wrote weighted edge CSV to {args.output_csv}")


if __name__ == "__main__":
    main()
