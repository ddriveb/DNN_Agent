"""Optical-only RMSA environment for the strict R-side discriminative audit.

This module deliberately avoids Agent-C, DF_C, MEC servers, splits, deadlines,
and any C-side observation builder.  Requests are pure optical demands
(src, dst, bitrate, arrival, holding).  The environment only allocates and
releases spectrum on an OpticalNetwork.
"""
from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.action_mask import build_agent_r_mask
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.env.spectrum_summary import _compute_path_stats
from sa_hmarl.network.ksp import get_k_shortest_paths
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.optical_network import OpticalNetwork


# Protocol constants fixed by the audit charter.
DEFAULT_SLOT_BW_HZ = 12.5e9
DEFAULT_GUARD_BAND_FS = 1
DEFAULT_MEAN_HOLDING_TIME = 10.0
DEFAULT_BITRATE_MIN_GBPS = 25
DEFAULT_BITRATE_MAX_GBPS = 100


@dataclass(frozen=True, slots=True)
class ODRequest:
    """Optical demand request with no C-side semantics."""

    req_id: int
    src_node: int
    dst_node: int
    bitrate_gbps: int
    arrival_time: float
    holding_time: float

    def required_fs(self, mod, slot_bw_hz: float = DEFAULT_SLOT_BW_HZ,
                    guard_band_fs: int = DEFAULT_GUARD_BAND_FS) -> int:
        """Required FS count for a given modulation format."""
        bitrate_bps = float(self.bitrate_gbps) * 1e9
        data_fs = math.ceil(bitrate_bps / (slot_bw_hz * mod.spectral_efficiency))
        return int(data_fs + guard_band_fs)


def generate_od_requests(
    num_nodes: int,
    rng: np.random.RandomState,
    num_requests: int,
    arrival_interval: float,
    mean_holding_time: float = DEFAULT_MEAN_HOLDING_TIME,
    bitrate_min_gbps: int = DEFAULT_BITRATE_MIN_GBPS,
    bitrate_max_gbps: int = DEFAULT_BITRATE_MAX_GBPS,
    poisson_arrivals: bool = True,
    exponential_holding: bool = True,
) -> List[ODRequest]:
    """Generate a deterministic optical-demand request trace."""
    if poisson_arrivals:
        inter_arrivals = rng.exponential(scale=arrival_interval, size=num_requests)
    else:
        inter_arrivals = np.full(num_requests, arrival_interval, dtype=float)
    arrival_times = np.cumsum(inter_arrivals)

    if exponential_holding:
        holdings = rng.exponential(scale=mean_holding_time, size=num_requests)
    else:
        holdings = np.full(num_requests, mean_holding_time, dtype=float)

    requests: List[ODRequest] = []
    for i in range(num_requests):
        while True:
            src = rng.randint(0, num_nodes)
            dst = rng.randint(0, num_nodes)
            if src != dst:
                break
        bitrate = int(rng.randint(bitrate_min_gbps, bitrate_max_gbps + 1))
        requests.append(ODRequest(
            req_id=i,
            src_node=int(src),
            dst_node=int(dst),
            bitrate_gbps=bitrate,
            arrival_time=float(arrival_times[i]),
            holding_time=float(max(holdings[i], 1e-3)),
        ))
    return requests


class OpticalOnlyRMSAEnv:
    """Minimal optical RMSA environment: no C-side, no MEC."""

    def __init__(
        self,
        topology: str = "xlron_cost239_ptrnet_real",
        num_slots: int = 100,
        k_paths: int = 50,
        max_blocks: int = 10,
        path_sort_strategy: str = "hops",
        block_sort_strategy: str = "start_asc",
        mod_registry: Optional[ModulationRegistry] = None,
        slot_bw_hz: float = DEFAULT_SLOT_BW_HZ,
        guard_band_fs: int = DEFAULT_GUARD_BAND_FS,
        seed: Optional[int] = None,
    ):
        self.topology = topology
        self.num_slots = num_slots
        self.k_paths = k_paths
        self.max_blocks = max_blocks
        self.path_sort_strategy = path_sort_strategy
        self.block_sort_strategy = block_sort_strategy
        self.mod_reg = mod_registry if mod_registry is not None else ModulationRegistry.from_profile("default")
        self.slot_bw_hz = slot_bw_hz
        self.guard_band_fs = guard_band_fs
        self.seed = seed

        self.net = OpticalNetwork(topology, num_slots, seed=seed)
        # Candidate paths depend only on the fixed topology and OD pair.  Keeping
        # them across resets makes native-policy training practical without
        # changing any spectrum-dependent observation field.
        self._path_cache: Dict[Tuple[int, int, int, str], List[List[int]]] = {}
        self.time: float = 0.0
        # Heap of (release_time, counter, connection_dict).  Counter breaks ties.
        self._release_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        self._counter = 0
        self.active_connections: List[Tuple[float, int, Dict[str, Any]]] = []
        self.allocations = 0
        self.releases = 0

    def reset(self, requests: Optional[List[ODRequest]] = None):
        """Reset link states and release all connections."""
        self.net.reset()
        self._release_heap = []
        self._counter = 0
        self.active_connections = []
        self.allocations = 0
        self.releases = 0
        self.time = 0.0
        if requests:
            self.time = float(requests[0].arrival_time)

    def advance_time(self, target_time: float):
        """Advance simulation time and release expired lightpaths."""
        target_time = float(target_time)
        while self._release_heap and self._release_heap[0][0] <= target_time:
            release_time, _, conn = heapq.heappop(self._release_heap)
            self.net.release(conn["path"], conn["start_slot"], conn["num_slots"])
            self.releases += 1
            # Keep active_connections consistent: rebuild lazily on demand.
        self.time = target_time
        self._rebuild_active_connections()

    def _rebuild_active_connections(self):
        """Sync active_connections with the release heap."""
        now = self.time
        self.active_connections = [
            (rt, c, conn) for (rt, c, conn) in self._release_heap if rt > now
        ]

    def build_observation(self, req: ODRequest) -> Dict[str, Any]:
        """Build an R-side observation for the current request."""
        path_key = (
            int(req.src_node),
            int(req.dst_node),
            int(self.k_paths),
            str(self.path_sort_strategy),
        )
        cached_paths = self._path_cache.get(path_key)
        if cached_paths is None:
            cached_paths = get_k_shortest_paths(
                self.net.G,
                req.src_node,
                req.dst_node,
                self.k_paths,
                weight="length_km",
                sort_by=self.path_sort_strategy,
            )
            self._path_cache[path_key] = [list(path) for path in cached_paths]
        candidate_paths = [list(path) for path in cached_paths]

        path_features = []
        for path in candidate_paths:
            stats = _compute_path_stats(self.net, path)
            path_features.append({
                "path_length_km": float(stats.path_dist_km),
                "hop_count": int(stats.path_length),
                "lfb": int(stats.lfb),
                "free_ratio": float(stats.free_ratio),
                "frag_index": float(stats.frag_index),
            })

        num_paths = len(candidate_paths)
        num_mods = self.mod_reg.num_formats

        feasible_mask: List[List[bool]] = []
        required_fs: List[List[Optional[int]]] = []
        candidate_blocks: List[List[List[Tuple[int, int]]]] = []

        for p_idx, path in enumerate(candidate_paths):
            path_dist = self.net.path_length_km(path)
            path_feasible: List[bool] = []
            path_req_fs: List[Optional[int]] = []
            path_blocks: List[List[Tuple[int, int]]] = []
            for m_idx in range(num_mods):
                mod = self.mod_reg[m_idx]
                if mod.reach_km < path_dist:
                    path_feasible.append(False)
                    path_req_fs.append(None)
                    path_blocks.append([])
                    continue
                req_fs = req.required_fs(mod, self.slot_bw_hz, self.guard_band_fs)
                blocks = self.net.get_candidate_blocks(
                    path,
                    req_fs,
                    max_candidates=self.max_blocks,
                    sort_by=self.block_sort_strategy,
                )
                block_tuples = [(int(b.start_slot), int(b.size)) for b in blocks]
                path_feasible.append(True)
                path_req_fs.append(req_fs)
                path_blocks.append(block_tuples)
            feasible_mask.append(path_feasible)
            required_fs.append(path_req_fs)
            candidate_blocks.append(path_blocks)

        mask = build_agent_r_mask(
            num_paths,
            num_mods,
            self.max_blocks,
            self.mod_reg.names,
            feasible_mask,
            required_fs,
            candidate_blocks,
        )

        return {
            "src_node": req.src_node,
            "dst_node": req.dst_node,
            "num_slots": self.num_slots,
            "candidate_paths": candidate_paths,
            "path_features": path_features,
            "mod_names": list(self.mod_reg.names),
            "feasible_mask_per_path_mod": feasible_mask,
            "required_fs_per_path_mod": required_fs,
            "candidate_blocks_per_path_mod": candidate_blocks,
            "agent_r_mask": mask,
            "request_id": req.req_id,
        }

    def step(self, action_idx: int, obs: Dict[str, Any], holding_time: float) -> Dict[str, Any]:
        """Execute a decoded R action on the optical layer."""
        num_mods = len(obs["mod_names"])
        path_idx, mod_idx, block_idx = decode_agent_r_action(
            action_idx, num_mods, self.max_blocks
        )

        if path_idx >= len(obs["candidate_paths"]):
            return {"success": False, "reason": "invalid_path"}

        path = obs["candidate_paths"][path_idx]
        blocks = obs["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        if block_idx >= len(blocks):
            return {"success": False, "reason": "no_suitable_block"}

        start_slot, block_size = blocks[block_idx]
        req_fs = obs["required_fs_per_path_mod"][path_idx][mod_idx]
        if req_fs is None or req_fs <= 0:
            return {"success": False, "reason": "invalid_required_fs"}
        if block_size < req_fs:
            return {"success": False, "reason": "no_suitable_block"}

        ok = self.net.allocate(path, start_slot, req_fs)
        if not ok:
            return {"success": False, "reason": "no_suitable_block"}

        self.allocations += 1
        release_time = self.time + float(holding_time)
        conn = {
            "req_id": obs.get("request_id", -1),
            "path": list(path),
            "start_slot": int(start_slot),
            "num_slots": int(req_fs),
            "path_idx": int(path_idx),
            "mod_idx": int(mod_idx),
            "mod_name": self.mod_reg.names[mod_idx],
            "block_idx": int(block_idx),
            "path_length_km": self.net.path_length_km(path),
            "hop_count": max(0, len(path) - 1),
            "required_fs": int(req_fs),
        }
        heapq.heappush(self._release_heap, (release_time, self._counter, conn))
        self._counter += 1
        self._rebuild_active_connections()

        return {
            "success": True,
            "reason": "admitted",
            "path_length_km": conn["path_length_km"],
            "hop_count": conn["hop_count"],
            "required_fs": conn["required_fs"],
            "start_slot": conn["start_slot"],
        }

    def get_global_spectrum_stats(self) -> Dict[str, Any]:
        return self.net.get_global_spectrum_stats()

    def get_utilization(self) -> float:
        total = len(self.net.link_states) * self.num_slots
        occupied = sum(int(np.sum(v)) for v in self.net.link_states.values())
        return occupied / max(total, 1)
