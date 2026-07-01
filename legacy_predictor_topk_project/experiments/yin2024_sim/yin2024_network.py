"""Yin 2024 network topologies (Net-1/Net-2/Net-3) + spectrum initialization.

All edge lists are manually digitized from Fig. 7 of Yin 2024.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import networkx as nx
from env import OpticalNetwork
from mec_servers import MECServer


# ------------------------------------------------------------------
# Topologies (node_u, node_v, length_km)
# ------------------------------------------------------------------

NET1_EDGES = [
    (0, 1, 10),
    (0, 2, 6),
    (1, 2, 8),
    (1, 3, 12),
    (1, 4, 9),
    (3, 4, 6),
    (2, 5, 4),
    (5, 4, 10),
    (5, 6, 5),
    (5, 7, 12),
    (4, 6, 12),
    (4, 8, 11),
    (6, 7, 8),
    (7, 8, 12),
]

NET1_SERVERS = [0, 3, 5, 6, 8]

NET2_EDGES = [
    (0, 1, 10),
    (0, 2, 7),
    (0, 3, 6),
    (1, 3, 15),
    (1, 7, 24),
    (2, 4, 6),
    (4, 5, 6),
    (5, 6, 12),
    (3, 6, 18),
    (2, 8, 20),
    (6, 13, 10),
    (6, 12, 18),
    (7, 8, 6),
    (8, 9, 6),
    (8, 11, 7),
    (9, 10, 7),
    (9, 11, 3),
    (10, 15, 7),
    (11, 15, 3),
    (12, 14, 7),
    (13, 14, 3),
    (14, 15, 3),
    (15, 16, 2),
]

NET2_SERVERS = [1, 3, 4, 8, 11, 16]

NET3_EDGES = [
    (0, 1, 6),
    (1, 2, 3),
    (2, 3, 3),
    (2, 4, 2),
    (0, 5, 5),
    (5, 6, 4),
    (4, 6, 4),
    (4, 7, 5),
    (3, 7, 2),
    (6, 9, 4),
    (7, 8, 3),
    (8, 9, 3),
    (8, 11, 3),
    (8, 12, 4),
    (7, 12, 7),
    (5, 10, 4),
    (9, 10, 5),
    (10, 16, 5),
    (16, 17, 3),
    (11, 13, 3),
    (13, 14, 2),
    (14, 18, 4),
    (11, 12, 3),
    (12, 15, 4),
    (15, 19, 4),
    (18, 19, 4),
    (17, 18, 3),
]

NET3_SERVERS = [0, 1, 3, 6, 8, 17, 19]


# ------------------------------------------------------------------
# Spectrum initialization
# ------------------------------------------------------------------

def init_spectrum(num_slots, load_factor=0.6, frag_level=0.2, seed=None):
    """Generate an initial spectrum occupancy pattern.

    Returns a bool array where True = occupied.
    """
    rng = np.random.RandomState(seed)
    occupied = int(num_slots * load_factor)
    slots = np.zeros(num_slots, dtype=bool)

    if frag_level <= 0.25:
        num_blocks = max(2, int(num_slots * 0.03))
    else:
        num_blocks = max(8, int(num_slots * 0.10))

    remaining = occupied
    for b in range(num_blocks):
        if remaining <= 0:
            break
        block_len = max(1, remaining // max(1, num_blocks - b))
        block_len = min(block_len, remaining, num_slots)
        start = rng.randint(0, num_slots - block_len + 1)
        slots[start:start + block_len] = True
        remaining = occupied - int(np.sum(slots))

    # Fill any remaining gaps randomly
    free_idx = np.where(~slots)[0]
    if len(free_idx) > 0 and remaining > 0:
        chosen = rng.choice(free_idx, size=min(remaining, len(free_idx)), replace=False)
        slots[chosen] = True

    return slots


# ------------------------------------------------------------------
# Yin 2024 Optical Network
# ------------------------------------------------------------------

class Yin2024OpticalNetwork(OpticalNetwork):
    """OpticalNetwork extended with Net-1/2/3 topologies + initialized spectrum."""

    def __init__(self, topology="net1", num_slots=100, seed=None,
                 init_load_factor=0.6, frag_level=0.2):
        self._init_load_factor = init_load_factor
        self._frag_level = frag_level
        super().__init__(topology=topology, num_slots=num_slots, seed=seed)
        self._init_spectrum(seed)

    def _build_topology(self):
        if self.topology == "net1":
            edges = NET1_EDGES
        elif self.topology == "net2":
            edges = NET2_EDGES
        elif self.topology == "net3":
            edges = NET3_EDGES
        else:
            return super()._build_topology()

        for u, v, length in edges:
            self.G.add_edge(u, v, length=float(length), weight=float(length))

    def _init_spectrum(self, seed):
        for link in self.link_states:
            self.link_states[link] = init_spectrum(
                self.num_slots,
                load_factor=self._init_load_factor,
                frag_level=self._frag_level,
                seed=(seed + hash(link)) % (2**31) if seed is not None else None,
            )

    def reset(self):
        super().reset()
        # Re-apply initialized spectrum after clearing
        self._init_spectrum(None)


# ------------------------------------------------------------------
# Yin 2024 MEC Cluster (fixed server placement + paper parameters)
# ------------------------------------------------------------------

class Yin2024MECServer(MECServer):
    """MECServer adapted for Yin 2024 parameters.

    - compute_delay_ms no longer includes data_size_mb * 0.5 (that penalty
      was calibrated for tiny intermediate sizes in the original framework).
    - The 20.0 reference factor is kept for interface compatibility.
    """

    def compute_delay_ms(self, compute_cost, data_size_mb):
        if self.available_compute <= 0:
            return float('inf')
        # Pure compute delay; transfer latency is modeled via net_delay
        base_ms = (compute_cost / self.available_compute) * 20.0
        return base_ms


class Yin2024MECCluster:
    """MEC cluster with fixed server positions and Yin 2024 parameters."""

    def __init__(self, server_nodes, loads_hz, caps_hz_per_s):
        self.num_nodes = max(server_nodes) + 1
        self.servers = []
        for i, (node_id, load_hz, cap_hz) in enumerate(zip(server_nodes, loads_hz, caps_hz_per_s)):
            srv = Yin2024MECServer(
                i, int(node_id),
                compute_capacity_gflops=cap_hz / 1e9,
                memory_capacity_gb=16.0,
            )
            srv.current_load = load_hz / 1e9
            srv._initial_load = load_hz / 1e9
            srv.active_tasks = 0
            self.servers.append(srv)

    def get_server(self, server_id: int):
        return self.servers[server_id]

    def get_server_at_node(self, node_id: int):
        for s in self.servers:
            if s.node_id == node_id:
                return s
        return None

    def get_server_by_id(self, server_id: int):
        return self.servers[server_id]

    @property
    def server_node_ids(self):
        return [s.node_id for s in self.servers]

    def reset(self):
        for s in self.servers:
            s.current_load = s._initial_load
            s.active_tasks = 0

    def server_load_vector(self, normalize=True):
        vec = np.array([s.utilization for s in self.servers], dtype=np.float32)
        if normalize and len(vec) > 0:
            m = vec.max()
            if m > 0:
                vec /= m
        return vec
