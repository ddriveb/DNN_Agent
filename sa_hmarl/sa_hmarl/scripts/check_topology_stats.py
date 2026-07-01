"""Topology sanity checker: prints stats and modulation reach feasibility.

Usage:
    PYTHONPATH=sa_hmarl python sa_hmarl/scripts/check_topology_stats.py --topology metro24
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import numpy as np
import networkx as nx

from sa_hmarl.network.topology_data import get_topology_edges
from sa_hmarl.network.modulation import ModulationRegistry


def main():
    parser = argparse.ArgumentParser(description="Topology sanity checker")
    parser.add_argument("--topology", type=str, default="metro24")
    parser.add_argument("--k", type=int, default=5, help="K for K-shortest paths")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    args = parser.parse_args()

    edges = get_topology_edges(args.topology)
    G = nx.Graph()
    for u, v, length in edges:
        G.add_edge(u, v, length=length)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    degrees = [d for _, d in G.degree()]
    avg_degree = np.mean(degrees)

    # Shortest path lengths (km)
    all_pairs = dict(nx.all_pairs_dijkstra_path_length(G, weight="length"))
    sp_lengths = []
    for src in all_pairs:
        for dst in all_pairs[src]:
            if src != dst:
                sp_lengths.append(all_pairs[src][dst])
    avg_sp = np.mean(sp_lengths)
    max_sp = np.max(sp_lengths)

    # KSP path lengths (sample all node pairs)
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    reach_map = {m.name: m.reach_km for m in mod_reg.mod_table}

    ksp_lengths = []
    feasible_counts = {name: 0 for name in reach_map}
    total_paths = 0

    # Sample all node pairs but cap KSP computation per pair
    for src in range(num_nodes):
        for dst in range(src + 1, num_nodes):
            try:
                gen = nx.shortest_simple_paths(G, src, dst, weight="length")
                for _ in range(args.k):
                    try:
                        path = next(gen)
                        length = sum(
                            G[path[i]][path[i + 1]]["length"] for i in range(len(path) - 1)
                        )
                        ksp_lengths.append(length)
                        total_paths += 1
                        for name, reach in reach_map.items():
                            if length <= reach:
                                feasible_counts[name] += 1
                    except StopIteration:
                        break
            except nx.NetworkXNoPath:
                pass

    print("=" * 60)
    print(f"Topology: {args.topology}")
    print("=" * 60)
    print(f"Nodes:            {num_nodes}")
    print(f"Edges:            {num_edges}")
    print(f"Avg node degree:  {avg_degree:.2f}")
    print(f"Min degree:       {min(degrees)}")
    print(f"Max degree:       {max(degrees)}")
    print(f"Connected:        {nx.is_connected(G)}")
    print()
    print(f"Shortest path (km)")
    print(f"  Average:        {avg_sp:.1f}")
    print(f"  Max:            {max_sp:.1f}")
    print()
    print(f"KSP path lengths (k={args.k}, all node pairs)")
    print(f"  Count:          {len(ksp_lengths)}")
    print(f"  Average:        {np.mean(ksp_lengths):.1f}")
    print(f"  Max:            {np.max(ksp_lengths):.1f}")
    print(f"  Median:         {np.median(ksp_lengths):.1f}")
    print(f"  P95:            {np.percentile(ksp_lengths, 95):.1f}")
    print(f"  P99:            {np.percentile(ksp_lengths, 99):.1f}")
    print()
    print("Modulation reach feasibility (% of KSP paths)")
    for name in mod_reg.names:
        pct = feasible_counts[name] / total_paths * 100 if total_paths > 0 else 0
        print(f"  {name:6s} ({reach_map[name]:4.0f} km):  {pct:5.1f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
