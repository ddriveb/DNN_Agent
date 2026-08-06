"""XLRON-compatible RMSA environment (pure NumPy).

State mirrors XLRON RSAEnvState:
- link_slot_array:        (num_links, num_slots) 0=free, 1=occupied
- link_slot_departure_array: (num_links, num_slots) release time
- request:                (src, dst, bitrate)

KSP pools: hops-first (XLRON `_get_k_shortest_paths` with weight=None
sorts by path length), built once per topology from the project's
directed edge set (xlron_nsfnet_deeprmsa: 14 nodes, 44 directed links).

Modulation (XLRON defaults): BPSK/QPSK/8QAM/16QAM reach
10000/2500/1250/625 km, spectral efficiency 1/2/3/4.
required_slots = ceil(bitrate / (se * slot_size)) + guardband,
slot_size = 12.5 GHz, guardband = 1.
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import networkx as nx
import numpy as np

from .traffic import Request, Trace

SLOT_SIZE_GHZ = 12.5
GUARDBAND = 1
MODULATION_TABLE = (
    ("BPSK", 10000.0, 1.0),
    ("QPSK", 2500.0, 2.0),
    ("8QAM", 1250.0, 3.0),
    ("16QAM", 625.0, 4.0),
)


def load_topology_edges(topology_name: str = "xlron_nsfnet_deeprmsa"):
    """Directed edge list (u, v, km) from the project's topology registry."""
    from sa_hmarl.network.topology_data import get_topology_edges
    undirected = get_topology_edges(topology_name)
    edges = []
    for u, v, d in undirected:
        edges.append((int(u), int(v), float(d)))
        edges.append((int(v), int(u), float(d)))
    return edges


def highest_modulation(path_km: float) -> Optional[Tuple[str, float]]:
    """Highest-SE reachable format; None if no format reaches the path
    (XLRON marks such paths unavailable, mod_format_mask = -1)."""
    chosen = None
    for name, reach, se in MODULATION_TABLE:
        if reach >= path_km and (chosen is None or se > chosen[1]):
            chosen = (name, se)
    return chosen


def required_slots(bitrate: float, se: float) -> int:
    base = bitrate / (se * SLOT_SIZE_GHZ) + GUARDBAND
    return int(np.ceil(base))


class RMSAEnv:
    """XLRON-style RMSA environment (dynamic arrivals, first-fit KSP-FF
    allocation, time-driven departures)."""

    def __init__(self, trace: Trace, k_paths: int, topology_name: str):
        self.trace = trace
        self.k_paths = int(k_paths)
        self.num_slots = trace.num_slots
        self.edges = load_topology_edges(topology_name)
        self.num_links = len(self.edges)
        self.edge_index = {e[:2]: i for i, e in enumerate(self.edges)}
        self.num_nodes = max(max(u, v) for u, v, _ in self.edges) + 1
        self._graph = nx.DiGraph()
        for u, v, d in self.edges:
            self._graph.add_edge(u, v, distance=d)
        self._ksp_cache: dict = {}
        # state
        self.link_slot_array = np.zeros((self.num_links, self.num_slots),
                                        dtype=np.int8)
        self.link_slot_departure_array = np.full(
            (self.num_links, self.num_slots), np.inf)
        self.time = 0.0
        self.current_request: Optional[Request] = None
        self.request_pointer = 0

    # ---- KSP pools (hops-first, cached) ----
    def ksp_paths(self, src: int, dst: int) -> List[List[int]]:
        key = (int(src), int(dst))
        if key not in self._ksp_cache:
            # islice, NOT list()[...]: full materialization hangs on dense
            # topologies (USNET: millions of simple paths per OD pair).
            # Semantics identical — generator yields in non-decreasing hops.
            import itertools
            paths = list(itertools.islice(
                nx.shortest_simple_paths(
                    self._graph, src, dst, weight=None), self.k_paths))
            # weight=None -> sorted by hops; deterministic tie-break
            paths.sort(key=lambda p: (len(p), tuple(p)))
            self._ksp_cache[key] = paths
        return self._ksp_cache[key]

    def path_links(self, path: List[int]) -> List[int]:
        return [self.edge_index[(u, v)]
                for u, v in zip(path[:-1], path[1:])]

    def path_km(self, path: List[int]) -> float:
        return sum(self._graph[u][v]["distance"]
                   for u, v in zip(path[:-1], path[1:]))

    # ---- time advance: release expired slots ----
    def advance_time(self, next_time: float) -> None:
        if next_time < self.time:
            raise ValueError(
                f"cannot advance backwards {next_time} < {self.time}")
        self.time = float(next_time)
        expired = self.link_slot_departure_array <= self.time
        self.link_slot_array[expired] = 0
        self.link_slot_departure_array[expired] = np.inf

    # ---- feasibility mask (spectrum continuity + adjacency) ----
    def path_mask(self, path: List[int], num_slots_needed: int) -> np.ndarray:
        """(num_slots,) bool: True where a window of num_slots_needed is
        free on every link of the path (cumsum window check)."""
        links = self.path_links(path)
        free = np.all(self.link_slot_array[links] == 0, axis=0).astype(
            np.int64)
        c = np.concatenate([[0], np.cumsum(free)])
        ns = int(num_slots_needed)
        mask = np.zeros(self.num_slots, dtype=bool)
        if ns <= self.num_slots:
            mask[:self.num_slots - ns + 1] = (c[ns:] - c[:-ns]) == ns
        return mask

    def allocate(self, path: List[int], start_slot: int,
                 num_slots_needed: int, holding_time: float) -> bool:
        links = self.path_links(path)
        window = self.link_slot_array[links, start_slot:
                                      start_slot + num_slots_needed]
        if np.any(window != 0):
            return False
        self.link_slot_array[links, start_slot:
                             start_slot + num_slots_needed] = 1
        release = self.time + holding_time
        self.link_slot_departure_array[links, start_slot:
                                       start_slot + num_slots_needed] = release
        return True

    # ---- request advancement ----
    def next_request(self) -> Optional[Request]:
        if self.request_pointer >= len(self.trace.requests):
            return None
        req = self.trace.requests[self.request_pointer]
        self.request_pointer += 1
        self.advance_time(req.arrival_time)
        self.current_request = req
        return req

    # ---- per-path action candidates ----
    def path_actions(self, req: Request) -> List[dict]:
        """List of (path, required_slots) for feasible KSP paths.
        Paths beyond the maximum modulation reach are unavailable
        (XLRON mod_format_mask = -1)."""
        out = []
        for path in self.ksp_paths(req.src_node, req.dst_node):
            mod = highest_modulation(self.path_km(path))
            if mod is None:
                continue
            _, se = mod
            ns = required_slots(req.bitrate_gbps, se)
            if ns > self.num_slots:
                continue
            out.append({"path": path, "required_slots": ns})
        return out
