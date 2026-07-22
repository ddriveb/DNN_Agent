"""Doherty paper-parity pure RMSA environment with full KxMxB actions."""
from __future__ import annotations

import heapq
import math
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

from sa_hmarl.network.ksp import get_k_shortest_paths
from sa_hmarl.network.modulation import ModulationFormat, ModulationRegistry
from sa_hmarl.network.topology_data import get_topology_edges


SLOT_BW_HZ = 12.5e9
GUARD_BAND_FS = 1
MEAN_HOLDING_TIME = 10.0
BITRATE_MIN_GBPS = 25
BITRATE_MAX_GBPS = 100

PRIMARY_CAUSE_ORDER = (
    "no_candidate_path",
    "no_reach_feasible_path_mod",
    "fs_exceeds_total_slots",
    "insufficient_link_capacity",
    "spectrum_continuity_failure",
    "spectrum_contiguity_failure",
    "k50_path_truncation",
)


def paper_modulation_registry() -> ModulationRegistry:
    return ModulationRegistry(mod_table=[
        ModulationFormat("BPSK", 10000.0, 1.0),
        ModulationFormat("QPSK", 2500.0, 2.0),
        ModulationFormat("8QAM", 1250.0, 3.0),
        ModulationFormat("16QAM", 625.0, 4.0),
    ])


@dataclass(frozen=True, slots=True)
class PureRMSARequest:
    req_id: int
    src_node: int
    dst_node: int
    bitrate_gbps: int
    arrival_time: float
    holding_time: float

    def required_fs(self, spectral_efficiency: float) -> int:
        return int(math.ceil(
            self.bitrate_gbps * 1e9 / (SLOT_BW_HZ * spectral_efficiency)
        ) + GUARD_BAND_FS)


def generate_paper_requests(
    num_nodes: int,
    seed: int,
    num_requests: int,
    load_erlang: float,
) -> List[PureRMSARequest]:
    """Generate the exact upstream-style trace used by the parity pipeline."""
    if load_erlang <= 0:
        raise ValueError("load_erlang must be positive")
    rng = np.random.RandomState(seed)
    pairs = [(s, d) for s in range(num_nodes) for d in range(num_nodes) if s != d]
    arrival_interval = MEAN_HOLDING_TIME / float(load_erlang)
    now = 0.0
    requests: List[PureRMSARequest] = []
    for req_id in range(num_requests):
        inter = 0.0
        while inter == 0.0:
            inter = float(rng.exponential(arrival_interval))
        now += inter
        src, dst = pairs[int(rng.randint(0, len(pairs)))]
        bitrate = int(rng.randint(BITRATE_MIN_GBPS, BITRATE_MAX_GBPS + 1))
        holding = 0.0
        while holding == 0.0 or holding >= 2.0 * MEAN_HOLDING_TIME:
            holding = float(rng.exponential(MEAN_HOLDING_TIME))
        requests.append(PureRMSARequest(
            req_id=req_id,
            src_node=int(src),
            dst_node=int(dst),
            bitrate_gbps=bitrate,
            arrival_time=now,
            holding_time=holding,
        ))
    return requests


def _free_blocks(available: np.ndarray) -> List[Tuple[int, int]]:
    blocks: List[Tuple[int, int]] = []
    index = 0
    while index < len(available):
        if not available[index]:
            index += 1
            continue
        start = index
        while index < len(available) and available[index]:
            index += 1
        blocks.append((start, index - start))
    return blocks


def _max_consecutive(available: np.ndarray) -> int:
    return max((size for _, size in _free_blocks(available)), default=0)


def build_pure_rmsa_mask(
    num_paths: int,
    num_modulations: int,
    num_blocks: int,
    reachable: Sequence[Sequence[bool]],
    required_fs: Sequence[Sequence[Optional[int]]],
    candidate_blocks: Sequence[Sequence[Sequence[Tuple[int, int]]]],
) -> np.ndarray:
    mask = np.zeros(num_paths * num_modulations * num_blocks, dtype=bool)
    for path_idx in range(num_paths):
        for mod_idx in range(num_modulations):
            req_fs = required_fs[path_idx][mod_idx]
            if not reachable[path_idx][mod_idx] or req_fs is None or req_fs <= 0:
                continue
            base = path_idx * num_modulations * num_blocks + mod_idx * num_blocks
            for block_idx, (_, size) in enumerate(candidate_blocks[path_idx][mod_idx][:num_blocks]):
                if size >= req_fs:
                    mask[base + block_idx] = True
    return mask


class DirectedSpectrumNetwork:
    """Dual-fiber spectrum: each directed arc owns an independent slot array."""

    def __init__(self, topology: str, num_slots: int):
        self.topology = topology
        self.num_slots = int(num_slots)
        self.G = nx.Graph()
        for u, v, distance in get_topology_edges(topology):
            self.G.add_edge(int(u), int(v), length_km=float(distance), weight=float(distance))
        self.reset()

    def reset(self) -> None:
        self.link_states: Dict[Tuple[int, int], np.ndarray] = {}
        for u, v in self.G.edges():
            self.link_states[(u, v)] = np.zeros(self.num_slots, dtype=bool)
            self.link_states[(v, u)] = np.zeros(self.num_slots, dtype=bool)

    def path_length_km(self, path: Sequence[int]) -> float:
        return float(sum(self.G.edges[(u, v)]["length_km"] for u, v in zip(path[:-1], path[1:])))

    def get_available_slots(self, path: Sequence[int]) -> np.ndarray:
        available = np.ones(self.num_slots, dtype=bool)
        for u, v in zip(path[:-1], path[1:]):
            available &= ~self.link_states[(u, v)]
        return available

    def allocate(self, path: Sequence[int], start: int, count: int) -> bool:
        if start < 0 or start + count > self.num_slots:
            return False
        if any(np.any(self.link_states[(u, v)][start:start + count]) for u, v in zip(path[:-1], path[1:])):
            return False
        for u, v in zip(path[:-1], path[1:]):
            self.link_states[(u, v)][start:start + count] = True
        return True

    def release(self, path: Sequence[int], start: int, count: int) -> None:
        for u, v in zip(path[:-1], path[1:]):
            self.link_states[(u, v)][start:start + count] = False

    def utilization(self) -> float:
        total = len(self.link_states) * self.num_slots
        return sum(int(x.sum()) for x in self.link_states.values()) / max(total, 1)

    def spectrum_summary(self) -> Dict[str, float]:
        free_ratios = []
        largest_blocks = []
        fragmentations = []
        for occupied in self.link_states.values():
            available = ~occupied
            free_count = int(available.sum())
            largest = _max_consecutive(available)
            free_ratios.append(free_count / self.num_slots)
            largest_blocks.append(float(largest))
            fragmentations.append(
                1.0 if free_count == 0
                else (0.0 if free_count == self.num_slots else 1.0 - largest / free_count)
            )
        return {
            "occupancy": self.utilization(),
            "free_ratio": float(np.mean(free_ratios)),
            "fragmentation": float(np.mean(fragmentations)),
            "largest_free_block": float(np.mean(largest_blocks)),
        }


class PureRMSAPaperEnv:
    def __init__(
        self,
        topology: str,
        num_slots: int = 100,
        k_paths: int = 50,
        max_blocks: int = 10,
        path_sort: str = "hops",
    ):
        self.topology = topology
        self.num_slots = int(num_slots)
        self.k_paths = int(k_paths)
        self.max_blocks = int(max_blocks)
        self.path_sort = path_sort
        self.mod_reg = paper_modulation_registry()
        self.net = DirectedSpectrumNetwork(topology, num_slots)
        self.time = 0.0
        self._release_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        self._counter = 0
        self.active_connections: List[Tuple[float, int, Dict[str, Any]]] = []
        self._path_cache: Dict[Tuple[int, int, int, str], List[List[int]]] = {}

    @property
    def num_nodes(self) -> int:
        return self.net.G.number_of_nodes()

    def reset(self) -> None:
        self.net.reset()
        self.time = 0.0
        self._release_heap = []
        self.active_connections = []
        self._counter = 0

    def advance_time(self, target_time: float) -> None:
        while self._release_heap and self._release_heap[0][0] <= target_time:
            _, _, conn = heapq.heappop(self._release_heap)
            self.net.release(conn["path"], conn["start_slot"], conn["required_fs"])
        self.time = float(target_time)
        self.active_connections = list(self._release_heap)

    def _paths(self, src: int, dst: int, k: Optional[int] = None) -> List[List[int]]:
        count = self.k_paths if k is None else int(k)
        key = (int(src), int(dst), count, self.path_sort)
        if key not in self._path_cache:
            self._path_cache[key] = [list(path) for path in get_k_shortest_paths(
                self.net.G, src, dst, count, weight="length_km", sort_by=self.path_sort
            )]
        return [list(path) for path in self._path_cache[key]]

    def build_observation(self, req: PureRMSARequest, k: Optional[int] = None) -> Dict[str, Any]:
        paths = self._paths(req.src_node, req.dst_node, k)
        path_features = []
        reachable: List[List[bool]] = []
        required: List[List[Optional[int]]] = []
        blocks_by_action: List[List[List[Tuple[int, int]]]] = []
        for path in paths:
            available = self.net.get_available_slots(path)
            free_count = int(available.sum())
            lfb = _max_consecutive(available)
            frag = 1.0 if free_count == 0 else (0.0 if free_count == self.num_slots else 1.0 - lfb / free_count)
            distance = self.net.path_length_km(path)
            path_features.append({
                "path_length_km": distance,
                "hop_count": max(0, len(path) - 1),
                "lfb": lfb,
                "free_ratio": free_count / self.num_slots,
                "frag_index": frag,
            })
            path_reachable: List[bool] = []
            path_required: List[Optional[int]] = []
            path_blocks: List[List[Tuple[int, int]]] = []
            all_blocks = _free_blocks(available)
            for mod_idx in range(self.mod_reg.num_formats):
                mod = self.mod_reg[mod_idx]
                is_reachable = distance <= mod.reach_km
                req_fs = req.required_fs(mod.spectral_efficiency) if is_reachable else None
                eligible = [] if req_fs is None else [b for b in all_blocks if b[1] >= req_fs]
                path_reachable.append(is_reachable)
                path_required.append(req_fs)
                path_blocks.append(eligible[:self.max_blocks])
            reachable.append(path_reachable)
            required.append(path_required)
            blocks_by_action.append(path_blocks)

        mask = build_pure_rmsa_mask(
            len(paths), self.mod_reg.num_formats, self.max_blocks,
            reachable, required, blocks_by_action,
        )
        diagnosis = self.diagnose_empty(req, paths, required, mask)
        return {
            "src_node": req.src_node,
            "dst_node": req.dst_node,
            "num_slots": self.num_slots,
            "candidate_paths": paths,
            "path_features": path_features,
            "mod_names": list(self.mod_reg.names),
            "feasible_mask_per_path_mod": reachable,
            "required_fs_per_path_mod": required,
            "candidate_blocks_per_path_mod": blocks_by_action,
            "agent_r_mask": mask,
            "request_id": req.req_id,
            "pure_mask_diagnosis": diagnosis,
        }

    def diagnose_empty(
        self,
        req: PureRMSARequest,
        paths: Sequence[Sequence[int]],
        required: Sequence[Sequence[Optional[int]]],
        mask: np.ndarray,
    ) -> Dict[str, Any]:
        flags = {name: False for name in PRIMARY_CAUSE_ORDER}
        reachable_count = 0
        demand_fitting = 0
        if not paths:
            flags["no_candidate_path"] = True
        for path_idx, path in enumerate(paths):
            distance = self.net.path_length_km(path)
            for mod_idx in range(self.mod_reg.num_formats):
                mod = self.mod_reg[mod_idx]
                if distance > mod.reach_km:
                    continue
                reachable_count += 1
                req_fs = required[path_idx][mod_idx]
                if req_fs is None or req_fs > self.num_slots:
                    flags["fs_exceeds_total_slots"] = True
                    continue
                demand_fitting += 1
                edge_free = [~self.net.link_states[(u, v)] for u, v in zip(path[:-1], path[1:])]
                if any(int(free.sum()) < req_fs for free in edge_free):
                    flags["insufficient_link_capacity"] = True
                    continue
                common = np.logical_and.reduce(edge_free)
                if int(common.sum()) < req_fs:
                    flags["spectrum_continuity_failure"] = True
                elif _max_consecutive(common) < req_fs:
                    flags["spectrum_contiguity_failure"] = True
        if paths and reachable_count == 0:
            flags["no_reach_feasible_path_mod"] = True
        empty = not np.asarray(mask, dtype=bool).any()
        primary = next((name for name in PRIMARY_CAUSE_ORDER if flags[name]), None) if empty else None
        if empty and primary is None:
            raise RuntimeError(f"No cause identified for empty mask, request={asdict(req)}")
        return {
            "mask_empty": empty,
            "primary_reason": primary,
            "flags": flags,
            "reachable_path_mods": reachable_count,
            "demand_fitting_path_mods": demand_fitting,
            "legal_actions": int(np.asarray(mask, dtype=bool).sum()),
        }

    def step(self, action_idx: int, obs: Dict[str, Any], req: PureRMSARequest) -> Dict[str, Any]:
        num_mods = self.mod_reg.num_formats
        path_idx = int(action_idx) // (num_mods * self.max_blocks)
        rem = int(action_idx) % (num_mods * self.max_blocks)
        mod_idx, block_idx = divmod(rem, self.max_blocks)
        mask = np.asarray(obs["agent_r_mask"], dtype=bool)
        if action_idx < 0 or action_idx >= len(mask) or not mask[action_idx]:
            return {"success": False, "reason": "illegal_action"}
        path = obs["candidate_paths"][path_idx]
        start, block_size = obs["candidate_blocks_per_path_mod"][path_idx][mod_idx][block_idx]
        req_fs = int(obs["required_fs_per_path_mod"][path_idx][mod_idx])
        if block_size < req_fs or not self.net.allocate(path, start, req_fs):
            return {"success": False, "reason": "allocation_failed"}
        conn = {
            "req_id": req.req_id,
            "path": list(path),
            "start_slot": int(start),
            "required_fs": req_fs,
        }
        heapq.heappush(self._release_heap, (self.time + req.holding_time, self._counter, conn))
        self._counter += 1
        self.active_connections = list(self._release_heap)
        return {
            "success": True,
            "reason": "admitted",
            "path_idx": path_idx,
            "mod_idx": mod_idx,
            "block_idx": block_idx,
            "start_slot": int(start),
            "block_size": int(block_size),
            "required_fs": req_fs,
            "path_hops": max(0, len(path) - 1),
            "path_km": self.net.path_length_km(path),
        }


__all__ = [
    "BITRATE_MAX_GBPS", "BITRATE_MIN_GBPS", "GUARD_BAND_FS",
    "MEAN_HOLDING_TIME", "PRIMARY_CAUSE_ORDER", "PureRMSAPaperEnv",
    "PureRMSARequest", "SLOT_BW_HZ", "build_pure_rmsa_mask",
    "generate_paper_requests", "paper_modulation_registry",
]
