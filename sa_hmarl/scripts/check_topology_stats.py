"""Topology sanity check script.

Outputs statistics for a given topology to verify it meets design criteria.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/scripts/check_topology_stats.py --topology metro24
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import networkx as nx

from sa_hmarl.network.topology_data import get_topology_edges, list_supported_topologies
from sa_hmarl.network.ksp import get_k_shortest_paths
from sa_hmarl.network.modulation import ModulationRegistry


def analyze_topology(topology: str, k: int = 5, modulation_profile: str = "default"):
    edges = get_topology_edges(topology)
    G = nx.Graph()
    for u, v, length_km in edges:
        G.add_edge(u, v, length_km=float(length_km), weight=float(length_km))

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    is_connected = nx.is_connected(G)
    degrees = [d for _, d in G.degree()]
    avg_degree = sum(degrees) / len(degrees)

    # Shortest path lengths (km)
    asp = dict(nx.shortest_path_length(G, weight="length_km"))
    all_sp_dists = []
    for src in asp:
        for dst in asp[src]:
            if src != dst:
                all_sp_dists.append(asp[src][dst])
    avg_sp = sum(all_sp_dists) / len(all_sp_dists) if all_sp_dists else 0.0
    max_sp = max(all_sp_dists) if all_sp_dists else 0.0
    min_sp = min(all_sp_dists) if all_sp_dists else 0.0

    # KSP path lengths
    ksp_lengths = []
    for src in range(num_nodes):
        for dst in range(num_nodes):
            if src != dst:
                paths = get_k_shortest_paths(G, src, dst, k=k)
                for p in paths:
                    length = sum(G[u][v]["length_km"] for u, v in zip(p[:-1], p[1:]))
                    ksp_lengths.append(length)

    avg_ksp = sum(ksp_lengths) / len(ksp_lengths) if ksp_lengths else 0.0
    max_ksp = max(ksp_lengths) if ksp_lengths else 0.0
    min_ksp = min(ksp_lengths) if ksp_lengths else 0.0

    # Modulation feasibility
    mod_reg = ModulationRegistry.from_profile(modulation_profile)
    mod_reaches = {m.name: m.reach_km for m in mod_reg.mod_table}

    feasible_counts = {name: 0 for name in mod_reaches}
    for length in ksp_lengths:
        mods = mod_reg.feasible_for_path(length)
        mod_names = {m.name for m in mods}
        for name in feasible_counts:
            if name in mod_names:
                feasible_counts[name] += 1

    total_ksp = len(ksp_lengths)

    print(f"{'='*60}")
    print(f"Topology: {topology}")
    print(f"{'='*60}")
    print(f"  Nodes                : {num_nodes}")
    print(f"  Edges                : {num_edges}")
    print(f"  Connected            : {is_connected}")
    print(f"  Average node degree  : {avg_degree:.2f}")
    print(f"  Min / Max degree     : {min(degrees)} / {max(degrees)}")
    print()
    print("Shortest Path Lengths (km)")
    print(f"  Average              : {avg_sp:.2f}")
    print(f"  Minimum              : {min_sp:.2f}")
    print(f"  Maximum              : {max_sp:.2f}")
    print()
    print(f"KSP Path Lengths (k={k})")
    print(f"  Total KSP paths      : {total_ksp}")
    print(f"  Average              : {avg_ksp:.2f}")
    print(f"  Minimum              : {min_ksp:.2f}")
    print(f"  Maximum              : {max_ksp:.2f}")
    print()
    print("Modulation Reach Feasibility (KSP paths)")
    for name, reach in mod_reaches.items():
        count = feasible_counts[name]
        pct = count / total_ksp * 100 if total_ksp > 0 else 0.0
        print(f"  {name:<6} (reach {reach:>6.0f} km) : {count:>6}/{total_ksp:<6} = {pct:>5.1f}%")
    print(f"{'='*60}")

    # Pass/fail summary
    passes = True
    checks = []
    if max_sp > 700:
        checks.append(f"FAIL: max shortest path {max_sp:.0f} km > 700 km")
        passes = False
    else:
        checks.append(f"PASS: max shortest path {max_sp:.0f} km <= 700 km")

    if feasible_counts["16QAM"] / total_ksp < 0.70:
        checks.append(f"FAIL: 16QAM feasible {feasible_counts['16QAM']/total_ksp*100:.1f}% < 70%")
        passes = False
    else:
        checks.append(f"PASS: 16QAM feasible {feasible_counts['16QAM']/total_ksp*100:.1f}% >= 70%")

    if not is_connected:
        checks.append("FAIL: graph is not connected")
        passes = False
    else:
        checks.append("PASS: graph is connected")

    print("\nSanity Checks:")
    for c in checks:
        print(f"  {c}")
    print(f"\nOverall: {'PASS' if passes else 'FAIL'}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Topology sanity check")
    parser.add_argument("--topology", type=str, default="metro24",
                        help="Topology name to analyze")
    parser.add_argument("--k", type=int, default=5,
                        help="Number of KSP paths per src-dst pair")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"],
                        help="Modulation profile: default (4) or extended (7 formats)")
    parser.add_argument("--list", action="store_true",
                        help="List supported topologies and exit")
    args = parser.parse_args()

    if args.list:
        print("Supported topologies:")
        for t in list_supported_topologies():
            print(f"  {t}")
        return

    if args.topology not in list_supported_topologies():
        print(f"ERROR: Unknown topology '{args.topology}'")
        print(f"Supported: {list_supported_topologies()}")
        sys.exit(1)

    analyze_topology(args.topology, k=args.k, modulation_profile=args.modulation_profile)


if __name__ == "__main__":
    main()
