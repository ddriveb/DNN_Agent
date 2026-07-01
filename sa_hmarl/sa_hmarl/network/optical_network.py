"""Enhanced Optical Network for SA-HMARL.

Uses fixed, reproducible link lengths from topology_data.py.
No random link length generation — same topology name always produces
the same graph structure and edge lengths.
"""
import numpy as np
import networkx as nx
from typing import List, Tuple, Optional
from sa_hmarl.network.modulation import ModulationRegistry, ModulationFormat
from sa_hmarl.network.spectrum_blocks import extract_candidate_blocks, SpectrumBlock
from sa_hmarl.network.topology_data import get_topology_edges, list_supported_topologies


class OpticalNetwork:
    """Elastic Optical Network with FS allocation and modulation awareness.

    Link lengths are fixed and determined solely by the topology name.
    The ``seed`` parameter only affects random behaviours *outside* topology
    construction (e.g. spectrum initialization if added later).
    """

    def __init__(self, topology: str = "nsfnet", num_slots: int = 128, seed=None):
        if topology not in list_supported_topologies():
            raise ValueError(
                f"Unknown topology: {topology!r}. "
                f"Supported: {list_supported_topologies()}"
            )
        self.topology = topology
        self.num_slots = num_slots
        self.rng = np.random.RandomState(seed)
        self.G = nx.Graph()
        self._build_topology()
        self.NUM_NODES = self.G.number_of_nodes()
        self.reset()

    def _build_topology(self):
        """Build graph from fixed topology data with explicit length_km."""
        edges = get_topology_edges(self.topology)
        for u, v, length_km in edges:
            self.G.add_edge(u, v, length_km=float(length_km), weight=float(length_km))

    def reset(self):
        self.link_states = {}
        for u, v in self.G.edges():
            key = (min(u, v), max(u, v))
            self.link_states[key] = np.zeros(self.num_slots, dtype=bool)

    # ------------------------------------------------------------------
    # Basic allocation
    # ------------------------------------------------------------------
    def allocate(self, path: List[int], start_slot: int, num_slots: int) -> bool:
        if start_slot < 0 or start_slot + num_slots > self.num_slots:
            return False
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            if np.any(self.link_states[link][start_slot:start_slot + num_slots]):
                return False
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            self.link_states[link][start_slot:start_slot + num_slots] = True
        return True

    def release(self, path: List[int], start_slot: int, num_slots: int):
        if path is None or start_slot is None:
            return
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            self.link_states[link][start_slot:start_slot + num_slots] = False

    def get_available_slots(self, path: List[int]) -> np.ndarray:
        avail = np.ones(self.num_slots, dtype=bool)
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            avail &= ~self.link_states[link]
        return avail

    # ------------------------------------------------------------------
    # Path properties
    # ------------------------------------------------------------------
    def path_length_km(self, path: List[int]) -> float:
        """Total physical length of a path (km)."""
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += self.G.edges[(min(u, v), max(u, v))]["length_km"]
        return total

    def path_num_hops(self, path: List[int]) -> int:
        return max(0, len(path) - 1)

    # ------------------------------------------------------------------
    # Candidate blocks for Agent-R
    # ------------------------------------------------------------------
    def get_candidate_blocks(self,
                             path: List[int],
                             req_fs: int,
                             max_candidates: int = 5,
                             sort_by: str = "size_desc") -> List[SpectrumBlock]:
        """Extract candidate contiguous free blocks on a path."""
        avail = self.get_available_slots(path)
        return extract_candidate_blocks(avail, req_fs, max_candidates, sort_by)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------
    @staticmethod
    def _max_consecutive(arr: np.ndarray) -> int:
        if not np.any(arr):
            return 0
        max_len = curr = 0
        for v in arr:
            if v:
                curr += 1
                max_len = max(max_len, curr)
            else:
                curr = 0
        return max_len

    @staticmethod
    def _count_blocks(arr: np.ndarray) -> int:
        count = 0
        in_block = False
        for v in arr:
            if v and not in_block:
                count += 1
                in_block = True
            elif not v:
                in_block = False
        return count

    def summarize_availability(self, free_arr: np.ndarray) -> dict:
        total_free = int(np.sum(free_arr))
        max_free = self._max_consecutive(free_arr)
        free_blocks = self._count_blocks(free_arr)
        if total_free == 0:
            frag = 1.0
        elif total_free == self.num_slots:
            frag = 0.0
        else:
            frag = 1.0 - (max_free / total_free)
        return {
            "total_free_slots": total_free,
            "max_free_slots": max_free,
            "largest_free_block_ratio": max_free / max(self.num_slots, 1),
            "free_block_count": free_blocks,
            "frag_index": float(frag),
        }

    def get_path_spectrum_stats(self, path: List[int]) -> dict:
        free_arr = self.get_available_slots(path)
        stats = self.summarize_availability(free_arr)
        stats["path_length"] = max(len(path) - 1, 0)
        stats["path_dist_km"] = self.path_length_km(path)
        return stats

    def get_global_spectrum_stats(self) -> dict:
        total_slots = len(self.link_states) * self.num_slots
        occupied = 0
        frag_vals = []
        max_free_vals = []
        free_block_vals = []
        for slots in self.link_states.values():
            occupied += int(np.sum(slots))
            free_arr = ~slots
            stats = self.summarize_availability(free_arr)
            frag_vals.append(stats["frag_index"])
            max_free_vals.append(stats["max_free_slots"])
            free_block_vals.append(stats["free_block_count"])
        return {
            "spectrum_utilization": occupied / max(total_slots, 1),
            "avg_frag_index": float(np.mean(frag_vals)) if frag_vals else 0.0,
            "largest_free_block_ratio": (max(max_free_vals) / max(self.num_slots, 1)) if max_free_vals else 0.0,
            "avg_free_block_count": float(np.mean(free_block_vals)) if free_block_vals else 0.0,
            "num_links": len(self.link_states),
        }

    # ------------------------------------------------------------------
    # Modulation-aware helpers
    # ------------------------------------------------------------------
    def get_feasible_modulations(self, path: List[int],
                                 mod_registry: ModulationRegistry) -> List[ModulationFormat]:
        """Return modulation formats feasible for this path length."""
        dist = self.path_length_km(path)
        return mod_registry.feasible_for_path(dist)
