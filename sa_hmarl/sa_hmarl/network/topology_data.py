"""Fixed topology data with reproducible link lengths (km).

All topologies use explicit length_km per edge.
No random generation — initialization with the same topology name always
produces the same graph.

Sources:
- NSFNET: widely-cited EON literature (e.g. "Spectrum-Efficient Survivable
  Routing in Elastic Optical Networks")
- USNET: relative weights from literature scaled to plausible km range
- Net-1/2/3: digitized from Yin 2024 (Fig. 7)
"""
from typing import List, Tuple, Dict
from pathlib import Path
import json

# ------------------------------------------------------------------
# NSFNET (14 nodes, 21 links)
# ------------------------------------------------------------------
NSFNET_EDGES: List[Tuple[int, int, float]] = [
    (0,  1,  1100.0),
    (0,  2,  1600.0),
    (0,  5,  2800.0),
    (1,  2,   600.0),
    (1,  3,  1000.0),
    (2,  4,   800.0),
    (2,  8,  2200.0),
    (3,  4,   400.0),
    (3, 10,  1600.0),
    (4,  5,   800.0),
    (4,  6,  1100.0),
    (5,  7,  1200.0),
    (6,  7,   400.0),
    (6, 13,  2000.0),
    (7,  8,   800.0),
    (8,  9,   700.0),
    (9, 10,   500.0),
    (9, 12,   800.0),
    (10, 11,  500.0),
    (11, 12,  300.0),
    (12, 13,  300.0),
]

# ------------------------------------------------------------------
# USNET (28 nodes, 45 links)
# ------------------------------------------------------------------
USNET_EDGES: List[Tuple[int, int, float]] = [
    (0,  1,   300.0),
    (0,  2,   500.0),
    (1,  2,   400.0),
    (1,  3,   600.0),
    (2,  4,   800.0),
    (3,  4,   300.0),
    (3,  5,   500.0),
    (4,  6,   400.0),
    (5,  6,   200.0),
    (5,  7,   400.0),
    (6,  8,   300.0),
    (7,  8,   500.0),
    (7,  9,   400.0),
    (8, 10,   300.0),
    (9, 10,   200.0),
    (9, 11,   400.0),
    (10, 12,  500.0),
    (11, 12,  300.0),
    (11, 13,  400.0),
    (12, 14,  300.0),
    (13, 14,  200.0),
    (13, 15,  500.0),
    (14, 16,  400.0),
    (15, 16,  200.0),
    (15, 17,  300.0),
    (16, 18,  400.0),
    (17, 18,  300.0),
    (17, 19,  400.0),
    (18, 20,  300.0),
    (19, 20,  200.0),
    (19, 21,  400.0),
    (20, 22,  300.0),
    (21, 22,  200.0),
    (21, 23,  500.0),
    (22, 24,  300.0),
    (23, 24,  200.0),
    (23, 25,  400.0),
    (24, 26,  300.0),
    (25, 26,  200.0),
    (25, 27,  400.0),
    (26, 27,  300.0),
    (0, 24,   600.0),
    (3, 20,   700.0),
    (5, 22,   500.0),
    (7, 26,   400.0),
]

# ------------------------------------------------------------------
# Net-1 (9 nodes, 14 links) — from Yin 2024
# ------------------------------------------------------------------
NET1_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 10.0),
    (0, 2,  6.0),
    (1, 2,  8.0),
    (1, 3, 12.0),
    (1, 4,  9.0),
    (3, 4,  6.0),
    (2, 5,  4.0),
    (5, 4, 10.0),
    (5, 6,  5.0),
    (5, 7, 12.0),
    (4, 6, 12.0),
    (4, 8, 11.0),
    (6, 7,  8.0),
    (7, 8, 12.0),
]

# ------------------------------------------------------------------
# Net-2 (17 nodes, 23 links) — from Yin 2024
# ------------------------------------------------------------------
NET2_EDGES: List[Tuple[int, int, float]] = [
    (0,  1, 10.0),
    (0,  2,  7.0),
    (0,  3,  6.0),
    (1,  3, 15.0),
    (1,  7, 24.0),
    (2,  4,  6.0),
    (4,  5,  6.0),
    (5,  6, 12.0),
    (3,  6, 18.0),
    (2,  8, 20.0),
    (6, 13, 10.0),
    (6, 12, 18.0),
    (7,  8,  6.0),
    (8,  9,  6.0),
    (8, 11,  7.0),
    (9, 10,  7.0),
    (9, 11,  3.0),
    (10, 15,  7.0),
    (11, 15,  3.0),
    (12, 14,  7.0),
    (13, 14,  3.0),
    (14, 15,  3.0),
    (15, 16,  2.0),
]

# ------------------------------------------------------------------
# Net-3 (20 nodes, 27 links) — from Yin 2024
# ------------------------------------------------------------------
NET3_EDGES: List[Tuple[int, int, float]] = [
    (0,  1,  6.0),
    (1,  2,  3.0),
    (2,  3,  3.0),
    (2,  4,  2.0),
    (0,  5,  5.0),
    (5,  6,  4.0),
    (4,  6,  4.0),
    (4,  7,  5.0),
    (3,  7,  2.0),
    (6,  9,  4.0),
    (7,  8,  3.0),
    (8,  9,  3.0),
    (8, 11,  3.0),
    (8, 12,  4.0),
    (7, 12,  7.0),
    (5, 10,  4.0),
    (9, 10,  5.0),
    (10, 16,  5.0),
    (16, 17,  3.0),
    (11, 13,  3.0),
    (13, 14,  2.0),
    (14, 18,  4.0),
    (11, 12,  3.0),
    (12, 15,  4.0),
    (15, 19,  4.0),
    (18, 19,  4.0),
    (17, 18,  3.0),
]

# ------------------------------------------------------------------
# Metro-24 (24 nodes, 43 links) — metro-scale ring + chords
# ------------------------------------------------------------------
METRO24_EDGES: List[Tuple[int, int, float]] = [
    # Ring backbone (24 edges, ~25-40 km each)
    (0,  1,  25.0), (1,  2,  30.0), (2,  3,  35.0), (3,  4,  30.0),
    (4,  5,  25.0), (5,  6,  40.0), (6,  7,  30.0), (7,  8,  35.0),
    (8,  9,  30.0), (9,  10, 25.0), (10, 11, 35.0), (11, 12, 30.0),
    (12, 13, 25.0), (13, 14, 30.0), (14, 15, 35.0), (15, 16, 30.0),
    (16, 17, 25.0), (17, 18, 40.0), (18, 19, 30.0), (19, 20, 35.0),
    (20, 21, 30.0), (21, 22, 25.0), (22, 23, 35.0), (23, 0,  30.0),
    # East-west shortcuts (8 edges)
    (0,  8,  70.0), (1,  9,  65.0), (2,  10, 75.0), (3,  11, 70.0),
    (12, 20, 70.0), (13, 21, 65.0), (14, 22, 75.0), (15, 23, 70.0),
    # Diagonal cross-links (6 edges)
    (0,  12, 60.0), (4,  16, 55.0), (6,  18, 65.0), (8,  20, 60.0),
    (2,  6,  45.0), (10, 14, 45.0),
    # Local chords (5 edges)
    (0,  4,  45.0), (4,  8,  50.0), (12, 16, 45.0), (16, 20, 50.0),
    (18, 22, 45.0),
]

# ------------------------------------------------------------------
# Metro-24 Bottleneck (24 nodes, 32 links)
# 4 regions x 6 nodes, interleaved IDs so first 4 nodes span 4 regions.
# Intra-region ring + chord; inter-region: single bridge per adjacency.
# Creates directional bottlenecks where server choice determines bridge crossing.
# ------------------------------------------------------------------
METRO24_BOTTLENECK_EDGES: List[Tuple[int, int, float]] = [
    # Region A: nodes 0, 4, 8, 12, 16, 20
    (0,  4,  30.0), (4,  8,  30.0), (8,  12, 30.0),
    (12, 16, 30.0), (16, 20, 30.0), (20, 0,  30.0),
    (0,  12, 45.0),                    # chord
    # Region B: nodes 1, 5, 9, 13, 17, 21
    (1,  5,  30.0), (5,  9,  30.0), (9,  13, 30.0),
    (13, 17, 30.0), (17, 21, 30.0), (21, 1,  30.0),
    (1,  13, 45.0),                    # chord
    # Region C: nodes 2, 6, 10, 14, 18, 22
    (2,  6,  30.0), (6,  10, 30.0), (10, 14, 30.0),
    (14, 18, 30.0), (18, 22, 30.0), (22, 2,  30.0),
    (2,  14, 45.0),                    # chord
    # Region D: nodes 3, 7, 11, 15, 19, 23
    (3,  7,  30.0), (7,  11, 30.0), (11, 15, 30.0),
    (15, 19, 30.0), (19, 23, 30.0), (23, 3,  30.0),
    (3,  15, 45.0),                    # chord
    # Inter-region bridges (4 links => ring of regions A-B-C-D-A)
    (8,  9,  70.0),   # A-B bridge
    (13, 14, 70.0),   # B-C bridge
    (18, 19, 70.0),   # C-D bridge
    (23, 0,  70.0),   # D-A bridge
]

# ------------------------------------------------------------------
# Metro-24 C-Sensitive (24 nodes, 42 links)
# Designed to amplify Agent-C's split/server trade-off rather than simply
# increasing topology size.
#
# Server defaults in training.utils.make_env:
#   s0 @ node 0  : high compute, sparse gateway / bottleneck region
#   s1 @ node 8  : medium compute, highly meshed spectrum-friendly core
#   s2 @ node 13 : low compute, short paths / local access friendly
#   s3 @ node 20 : high compute, longer peripheral region, fragmentation-prone
#
# All links are metro-scale (20-250 km). The graph intentionally contains
# narrow inter-region gateways plus backup alternatives so Agent-R still has
# room to route around pressure while Agent-C's server choice matters.
# ------------------------------------------------------------------
METRO24_C_SENSITIVE_EDGES: List[Tuple[int, int, float]] = [
    # Region A: high-compute server0 behind sparse gateways (nodes 0-5)
    (0,  1,  30.0), (1,  2,  35.0), (2,  3,  30.0),
    (3,  4,  40.0), (4,  5,  35.0), (5,  0,  45.0),
    (1,  4,  55.0),

    # Region B: spectrum-friendly meshed core around server1 (nodes 6-11)
    (6,  7,  35.0), (7,  8,  25.0), (8,  9,  30.0),
    (9,  10, 35.0), (10, 11, 25.0), (11, 6,  30.0),
    (6,  9,  45.0), (7,  10, 50.0), (8,  11, 45.0),
    (7,  11, 55.0),

    # Region C: low-compute server2 with short, locally diverse paths (12-17)
    (12, 13, 25.0), (13, 14, 25.0), (14, 15, 30.0),
    (15, 16, 25.0), (16, 17, 30.0), (17, 12, 35.0),
    (12, 15, 45.0), (13, 16, 40.0),

    # Region D: high-compute server3 in longer, more fragile periphery (18-23)
    (18, 19, 50.0), (19, 20, 45.0), (20, 21, 50.0),
    (21, 22, 55.0), (22, 23, 60.0), (23, 18, 65.0),
    (18, 21, 95.0),

    # Sparse A<->core gateways: compute-greedy traffic to server0 congests here
    (3,  6,  90.0), (5,  10, 110.0),

    # Core<->C multiple short alternatives: server1/server2 have path diversity
    (8,  12, 60.0), (9,  12, 45.0), (10, 13, 40.0),
    (11, 14, 50.0),

    # C<->D longer alternatives: server3 is powerful but path/fragments matter
    (16, 18, 100.0), (17, 19, 110.0), (14, 22, 130.0),
    (15, 23, 140.0),
]

# ------------------------------------------------------------------
# SNAP24 Gnutella Reach-Sensitive (24 nodes, 40 links)
# ------------------------------------------------------------------
# Derived from SNAP p2p-Gnutella04 graph structure with synthetic optical
# distances. This topology is designed for reach-constrained comparison:
# shortest-path pairs are roughly 23% 16QAM, 62% 8QAM, 15% QPSK under the
# default modulation registry. No pair requires BPSK on shortest paths, so the
# experiment stresses modulation choice without becoming physically infeasible.
#
# Suggested server defaults in training.utils.make_env:
#   server_nodes=[6, 1, 7, 12]
#   capacities=[90.0, 55.0, 32.0, 85.0]
# ------------------------------------------------------------------
SNAP24_GNUTELLA_REACH_EDGES: List[Tuple[int, int, float]] = [
    (0,  2,  490.0),
    (0,  6,  425.0),
    (0,  9,  475.0),
    (0, 19,  320.0),
    (1,  6,  465.0),
    (2,  4,  305.0),
    (2,  5,  370.0),
    (2, 10,  145.0),
    (2, 11,  635.0),
    (2, 12,  360.0),
    (2, 13,  625.0),
    (3,  6,  350.0),
    (4,  6,  520.0),
    (4, 19,  260.0),
    (5,  6,  250.0),
    (5,  9,  520.0),
    (6,  8,  380.0),
    (6, 10,  635.0),
    (6, 11,   95.0),
    (6, 12,  380.0),
    (6, 13,  480.0),
    (6, 15,  345.0),
    (6, 18,  160.0),
    (6, 21,  600.0),
    (6, 22,  515.0),
    (6, 23,  125.0),
    (7, 12,  225.0),
    (9, 10,  445.0),
    (9, 11,  560.0),
    (9, 12,  310.0),
    (9, 13,  495.0),
    (10, 17,  200.0),
    (10, 19,   95.0),
    (12, 14,  225.0),
    (12, 16,  485.0),
    (12, 17,  195.0),
    (12, 19,  235.0),
    (12, 20,  125.0),
    (13, 17,  215.0),
    (13, 19,  580.0),
]

# ------------------------------------------------------------------
# XLRON / TopologyBench imports
# ------------------------------------------------------------------
# Imported from micdoh/XLRON ``xlron/data/topologies`` undirected JSON files.
# XLRON stores several classic topology node IDs as 1-based labels; these lists
# are relabelled to contiguous 0-based IDs to match SA-HMARL request sampling.

# XLRON NSFNET DeepRMSA (14 nodes, 22 links)
XLRON_NSFNET_DEEPRMSA_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 1050.0),
    (0, 2, 1500.0),
    (0, 7, 2400.0),
    (1, 2, 600.0),
    (1, 3, 750.0),
    (2, 5, 1800.0),
    (3, 4, 600.0),
    (3, 10, 1950.0),
    (4, 5, 1200.0),
    (4, 6, 600.0),
    (5, 9, 1050.0),
    (5, 13, 1800.0),
    (6, 7, 750.0),
    (6, 9, 1350.0),
    (7, 8, 750.0),
    (8, 9, 750.0),
    (8, 11, 300.0),
    (8, 12, 300.0),
    (10, 11, 600.0),
    (10, 12, 750.0),
    (11, 13, 300.0),
    (12, 13, 150.0),
]

# XLRON COST239 PtrNet real distances (11 nodes, 26 links)
XLRON_COST239_PTRNET_REAL_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 1310.0),
    (0, 2, 760.0),
    (0, 3, 390.0),
    (0, 6, 740.0),
    (1, 2, 550.0),
    (1, 4, 390.0),
    (1, 7, 450.0),
    (2, 3, 660.0),
    (2, 4, 210.0),
    (2, 5, 390.0),
    (3, 6, 340.0),
    (3, 7, 1090.0),
    (3, 9, 660.0),
    (4, 5, 294.0),
    (4, 7, 220.0),
    (4, 10, 900.0),
    (5, 6, 350.0),
    (5, 7, 730.0),
    (5, 8, 350.0),
    (6, 8, 560.0),
    (6, 9, 320.0),
    (7, 8, 600.0),
    (7, 10, 820.0),
    (8, 9, 730.0),
    (8, 10, 320.0),
    (9, 10, 820.0),
]

# XLRON German17 (17 nodes, 24 links)
XLRON_GERMAN17_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 123.0),
    (0, 7, 332.0),
    (1, 2, 130.0),
    (1, 3, 255.0),
    (2, 3, 237.0),
    (2, 4, 312.0),
    (3, 4, 75.0),
    (3, 5, 95.0),
    (3, 7, 548.0),
    (4, 5, 45.0),
    (5, 11, 494.0),
    (6, 7, 551.0),
    (6, 8, 199.0),
    (7, 9, 656.0),
    (8, 9, 248.0),
    (9, 10, 196.0),
    (10, 11, 298.0),
    (10, 12, 245.0),
    (11, 14, 582.0),
    (11, 16, 130.0),
    (12, 13, 70.0),
    (13, 14, 137.0),
    (14, 15, 81.0),
    (15, 16, 680.0),
]

# XLRON JPN48 (48 nodes, 82 links)
XLRON_JPN48_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 476.0),
    (0, 2, 409.0),
    (1, 2, 178.0),
    (1, 4, 181.0),
    (2, 3, 183.0),
    (2, 4, 127.0),
    (3, 5, 61.0),
    (3, 6, 79.0),
    (3, 7, 245.0),
    (4, 5, 211.0),
    (4, 15, 273.0),
    (5, 15, 187.0),
    (6, 8, 163.0),
    (6, 15, 180.0),
    (7, 8, 95.0),
    (7, 10, 117.0),
    (7, 11, 127.0),
    (8, 9, 106.0),
    (8, 10, 79.0),
    (9, 10, 74.0),
    (9, 13, 96.0),
    (9, 15, 228.0),
    (9, 20, 117.0),
    (10, 11, 66.0),
    (10, 12, 30.0),
    (11, 12, 39.0),
    (12, 13, 47.0),
    (12, 14, 28.0),
    (13, 14, 36.0),
    (13, 19, 86.0),
    (14, 22, 151.0),
    (15, 16, 254.0),
    (15, 20, 211.0),
    (16, 17, 59.0),
    (16, 20, 192.0),
    (17, 18, 76.0),
    (18, 26, 148.0),
    (19, 20, 164.0),
    (19, 22, 122.0),
    (19, 23, 262.0),
    (20, 23, 250.0),
    (21, 23, 30.0),
    (21, 25, 107.0),
    (22, 23, 185.0),
    (23, 24, 66.0),
    (24, 25, 84.0),
    (24, 29, 89.0),
    (24, 30, 365.0),
    (25, 26, 10.0),
    (26, 27, 39.0),
    (26, 28, 77.0),
    (26, 29, 41.0),
    (26, 31, 253.0),
    (27, 28, 36.0),
    (27, 29, 52.0),
    (27, 30, 76.0),
    (28, 33, 143.0),
    (30, 36, 65.0),
    (31, 32, 121.0),
    (31, 33, 141.0),
    (32, 35, 256.0),
    (33, 34, 161.0),
    (33, 37, 71.0),
    (34, 35, 132.0),
    (34, 38, 66.0),
    (35, 40, 147.0),
    (36, 37, 74.0),
    (36, 39, 156.0),
    (37, 38, 194.0),
    (37, 39, 159.0),
    (38, 39, 251.0),
    (38, 44, 166.0),
    (40, 41, 53.0),
    (40, 43, 118.0),
    (40, 44, 198.0),
    (41, 42, 100.0),
    (42, 47, 758.0),
    (43, 44, 148.0),
    (43, 46, 170.0),
    (44, 45, 207.0),
    (45, 46, 125.0),
    (46, 47, 673.0),
]

COST239_DEEPRMSA_EDGES: List[Tuple[int, int, float]] = [
    (0, 1, 900.0),
    (0, 2, 780.0),
    (0, 3, 1100.0),
    (0, 7, 2620.0),
    (1, 2, 600.0),
    (1, 4, 800.0),
    (1, 5, 1200.0),
    (1, 6, 1640.0),
    (1, 8, 2180.0),
    (2, 3, 420.0),
    (2, 4, 440.0),
    (2, 6, 1860.0),
    (3, 4, 780.0),
    (3, 7, 1520.0),
    (3, 8, 1320.0),
    (4, 5, 700.0),
    (4, 9, 1460.0),
    (5, 6, 640.0),
    (5, 9, 1130.0),
    (5, 10, 1460.0),
    (6, 10, 1640.0),
    (7, 8, 780.0),
    (7, 9, 1480.0),
    (8, 9, 680.0),
    (8, 10, 1320.0),
    (9, 10, 640.0),
]

# ------------------------------------------------------------------
# XLRON USNet GCN-RMSA (GCN-RNN) / GCN-RMSA paper (Doherty et al. 2025)
# 24 nodes, 43 undirected links
# Source: https://github.com/micdoh/XLRON xlron/data/topologies/usnet_gcnrnn_undirected.json
# Commit: d07980b3233b1edc93507f60dc4a1c64b37af2e9
# ------------------------------------------------------------------
_XLRON_USNET_GCNRMSA_PATH = Path(__file__).with_name("topology_data_usnet_gcnrmsa.json")
if _XLRON_USNET_GCNRMSA_PATH.exists():
    with open(_XLRON_USNET_GCNRMSA_PATH) as _f:
        _XLRON_USNET_GCNRMSA_DATA = json.load(_f)
    XLRON_USNET_GCNRMSA_EDGES: List[Tuple[int, int, float]] = [
        (int(u), int(v), float(d)) for u, v, d in _XLRON_USNET_GCNRMSA_DATA["edges"]
    ]
else:
    XLRON_USNET_GCNRMSA_EDGES: List[Tuple[int, int, float]] = []

# ------------------------------------------------------------------
# Registry
# ------------------------------------------------------------------
TOPOLOGY_REGISTRY: Dict[str, List[Tuple[int, int, float]]] = {
    "nsfnet": NSFNET_EDGES,
    "usnet": USNET_EDGES,
    "net1": NET1_EDGES,
    "net2": NET2_EDGES,
    "net3": NET3_EDGES,
    "metro24": METRO24_EDGES,
    "metro24_bottleneck": METRO24_BOTTLENECK_EDGES,
    "metro24_c_sensitive": METRO24_C_SENSITIVE_EDGES,
    "snap24_gnutella_reach": SNAP24_GNUTELLA_REACH_EDGES,
    "xlron_nsfnet_deeprmsa": XLRON_NSFNET_DEEPRMSA_EDGES,
    "xlron_cost239_ptrnet_real": XLRON_COST239_PTRNET_REAL_EDGES,
    "xlron_german17": XLRON_GERMAN17_EDGES,
    "xlron_jpn48": XLRON_JPN48_EDGES,
    "cost239_deeprmsa": COST239_DEEPRMSA_EDGES,
    "xlron_usnet_gcnrmsa": XLRON_USNET_GCNRMSA_EDGES,
}


def get_topology_edges(name: str) -> List[Tuple[int, int, float]]:
    """Return fixed edge list (u, v, length_km) for a topology name.

    Raises:
        ValueError: if topology name is not supported.
    """
    if name not in TOPOLOGY_REGISTRY:
        raise ValueError(
            f"Unknown topology: {name!r}. "
            f"Supported: {list(TOPOLOGY_REGISTRY.keys())}"
        )
    return TOPOLOGY_REGISTRY[name]


def list_supported_topologies() -> List[str]:
    """Return list of supported topology names."""
    return list(TOPOLOGY_REGISTRY.keys())
