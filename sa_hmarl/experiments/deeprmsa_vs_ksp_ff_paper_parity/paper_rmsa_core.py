"""Paper-standard pure RMSA simulation core (Doherty et al. 2025 reproduction).

Self-contained core for the DeepRMSA vs KSP-FF paper-parity pipeline.  It
deliberately does NOT use the SA-HMARL coupled environment: the paper protocol
is optical-only (no C-side, no MEC, no split, no deadline, no server
capacity), uses dual-fiber direction-independent spectrum, paper modulation
reaches, and upstream DeepRMSA traffic truncation.

Reused shared components (unmodified):
- ``sa_hmarl.network.topology_data`` (fixed topology edges)
- ``sa_hmarl.network.ksp.get_k_shortest_paths`` (km / hops orderings)
- ``sa_hmarl.network.modulation`` (ModulationFormat / Registry containers)
"""
from __future__ import annotations

import heapq
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import networkx as nx
import numpy as np

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "sa_hmarl"))

from sa_hmarl.network.ksp import get_k_shortest_paths
from sa_hmarl.network.modulation import ModulationFormat, ModulationRegistry
from sa_hmarl.network.topology_data import get_topology_edges

from paper_topologies import LOCAL_TOPOLOGIES


def _topology_edges(topology: str) -> List[Tuple[int, int, float]]:
    """Local paper topologies take precedence over the shared registry."""
    if topology in LOCAL_TOPOLOGIES:
        return LOCAL_TOPOLOGIES[topology]
    return get_topology_edges(topology)


EXP_DIR = Path(__file__).resolve().parent
RUN_CONFIG_PATH = EXP_DIR / "RUN_CONFIG.json"


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
def load_run_config() -> Dict[str, Any]:
    return json.loads(RUN_CONFIG_PATH.read_text(encoding="utf-8"))


def paper_modulation_registry() -> ModulationRegistry:
    """Paper-standard modulation table: reach 10000/2500/1250/625, SE 1/2/3/4."""
    cfg = load_run_config()
    return ModulationRegistry(
        mod_table=[
            ModulationFormat(m["name"], float(m["reach_km"]), float(m["spectral_efficiency"]))
            for m in cfg["modulations"]
        ]
    )


# ---------------------------------------------------------------------------
# Requests
# ---------------------------------------------------------------------------
@dataclass(frozen=True, slots=True)
class PaperRequest:
    """Pure optical demand: no C-side/MEC/split/deadline semantics."""

    req_id: int
    src_node: int
    dst_node: int
    bitrate_gbps: int
    arrival_time: float
    holding_time: float


def generate_paper_requests(
    num_nodes: int,
    seed: int,
    num_requests: int,
    arrival_interval: float,
    mean_holding_time: float = 10.0,
    bitrate_min_gbps: int = 25,
    bitrate_max_gbps: int = 100,
) -> List[PaperRequest]:
    """Upstream DeepRMSA traffic generation (paper-standard).

    - Uniform over all N*(N-1) ordered OD pairs via a single pair-index draw
      (upstream: ``Src_Dest_Pair[np.random.randint(0, num_src_dest_pair)]``).
    - Poisson arrivals: exponential inter-arrival, zero draws resampled.
    - Holding: exponential(mean), resampled while ``ttl == 0 or ttl >= 2*mean``.
    - Bit rate: integer uniform in [25, 100] Gbps.
    """
    rng = np.random.RandomState(seed)
    # Lexicographic ordered-pair list, same order as upstream Src_Dest_Pair.
    pairs = [(s, d) for s in range(num_nodes) for d in range(num_nodes) if s != d]
    n_pairs = len(pairs)

    requests: List[PaperRequest] = []
    now = 0.0
    for i in range(num_requests):
        while True:
            inter = float(rng.exponential(arrival_interval))
            if inter != 0.0:
                break
        now += inter
        src, dst = pairs[int(rng.randint(0, n_pairs))]
        bitrate = int(rng.randint(bitrate_min_gbps, bitrate_max_gbps + 1))
        ttl = 0.0
        while ttl == 0.0 or ttl >= 2.0 * mean_holding_time:
            ttl = float(rng.exponential(mean_holding_time))
        requests.append(PaperRequest(i, src, dst, bitrate, now, ttl))
    return requests


# ---------------------------------------------------------------------------
# Modulation / FS demand (upstream cal_FS)
# ---------------------------------------------------------------------------
def best_modulation_index(
    path_len_km: float, mod_reg: ModulationRegistry
) -> Optional[int]:
    """Highest spectral-efficiency format whose reach covers the path (inclusive)."""
    best_idx = None
    best_se = -1.0
    for idx in range(mod_reg.num_formats):
        mod = mod_reg[idx]
        if path_len_km <= mod.reach_km and mod.spectral_efficiency > best_se:
            best_se = mod.spectral_efficiency
            best_idx = idx
    return best_idx


def required_fs(
    bitrate_gbps: int,
    spectral_efficiency: float,
    slot_bw_hz: float = 12.5e9,
    guard_band_fs: int = 1,
) -> int:
    """Upstream cal_FS: ceil(bitrate / (SE * 12.5GHz)) + 1 guard FS."""
    data_fs = math.ceil(float(bitrate_gbps) * 1e9 / (slot_bw_hz * spectral_efficiency))
    return int(data_fs + guard_band_fs)


# ---------------------------------------------------------------------------
# Directed-arc optical network (dual-fiber direction-independent spectrum)
# ---------------------------------------------------------------------------
class PaperOpticalNetwork:
    """EON with one independent spectrum array per DIRECTED arc."""

    def __init__(self, topology: str, num_slots: int = 100):
        self.topology = topology
        self.num_slots = int(num_slots)
        self.G = nx.Graph()
        for u, v, length_km in _topology_edges(topology):
            self.G.add_edge(int(u), int(v), length_km=float(length_km), weight=float(length_km))
        self.num_nodes = self.G.number_of_nodes()
        self.reset()

    def reset(self) -> None:
        self.arc_states: Dict[Tuple[int, int], np.ndarray] = {}
        for u, v in self.G.edges():
            self.arc_states[(u, v)] = np.zeros(self.num_slots, dtype=bool)
            self.arc_states[(v, u)] = np.zeros(self.num_slots, dtype=bool)

    @property
    def num_directed_arcs(self) -> int:
        return len(self.arc_states)

    def path_length_km(self, path: Sequence[int]) -> float:
        total = 0.0
        for u, v in zip(path[:-1], path[1:]):
            total += self.G.edges[(min(u, v), max(u, v))]["length_km"]
        return total

    def path_hops(self, path: Sequence[int]) -> int:
        return max(0, len(path) - 1)

    def get_availability(self, path: Sequence[int]) -> np.ndarray:
        """Slots free on EVERY directed arc of the path (continuity)."""
        avail = np.ones(self.num_slots, dtype=bool)
        for u, v in zip(path[:-1], path[1:]):
            avail &= ~self.arc_states[(u, v)]
        return avail

    def can_allocate(self, path: Sequence[int], start: int, num_slots: int) -> bool:
        if start < 0 or start + num_slots > self.num_slots:
            return False
        for u, v in zip(path[:-1], path[1:]):
            if np.any(self.arc_states[(u, v)][start:start + num_slots]):
                return False
        return True

    def allocate(self, path: Sequence[int], start: int, num_slots: int) -> bool:
        if not self.can_allocate(path, start, num_slots):
            return False
        for u, v in zip(path[:-1], path[1:]):
            self.arc_states[(u, v)][start:start + num_slots] = True
        return True

    def release(self, path: Sequence[int], start: int, num_slots: int) -> None:
        for u, v in zip(path[:-1], path[1:]):
            self.arc_states[(u, v)][start:start + num_slots] = False

    def utilization(self) -> float:
        total = len(self.arc_states) * self.num_slots
        used = sum(int(np.sum(state)) for state in self.arc_states.values())
        return used / max(total, 1)


# ---------------------------------------------------------------------------
# Free-block extraction (First-Fit order = ascending start slot)
# ---------------------------------------------------------------------------
def contiguous_free_blocks(avail: np.ndarray) -> List[Tuple[int, int]]:
    """All contiguous free blocks as (start, size), ascending start."""
    blocks: List[Tuple[int, int]] = []
    i = 0
    n = len(avail)
    while i < n:
        if avail[i]:
            start = i
            while i < n and avail[i]:
                i += 1
            blocks.append((start, i - start))
        else:
            i += 1
    return blocks


def eligible_blocks(avail: np.ndarray, req_fs: int) -> List[Tuple[int, int]]:
    """Contiguous free blocks with size >= req_fs, ascending start (First-Fit order)."""
    return [b for b in contiguous_free_blocks(avail) if b[1] >= req_fs]


def first_fit_start(avail: np.ndarray, req_fs: int) -> Optional[int]:
    """First-Fit: lowest start slot among blocks that fit req_fs, else None."""
    blocks = eligible_blocks(avail, req_fs)
    return blocks[0][0] if blocks else None


# ---------------------------------------------------------------------------
# Candidate paths (cached per topology/OD/k/sort)
# ---------------------------------------------------------------------------
_PATH_CACHE: Dict[Tuple[str, int, int, int, str], List[List[int]]] = {}


def get_candidate_paths(
    net: PaperOpticalNetwork, src: int, dst: int, k: int, sort_by: str
) -> List[List[int]]:
    key = (net.topology, int(src), int(dst), int(k), str(sort_by))
    cached = _PATH_CACHE.get(key)
    if cached is None:
        cached = [
            list(p)
            for p in get_k_shortest_paths(
                net.G, src, dst, k, weight="length_km", sort_by=sort_by
            )
        ]
        _PATH_CACHE[key] = cached
    return [list(p) for p in cached]


# ---------------------------------------------------------------------------
# Event-driven RMSA environment
# ---------------------------------------------------------------------------
class PaperRMSAEnv:
    """Event-driven pure-RMSA environment on directed arcs.

    Release discipline: all lightpaths with release_time <= arrival_time are
    released before the arrival is processed; ties are FIFO (insertion order).
    """

    def __init__(
        self,
        topology: str,
        num_slots: int = 100,
        mod_reg: Optional[ModulationRegistry] = None,
        slot_bw_hz: float = 12.5e9,
        guard_band_fs: int = 1,
    ):
        self.net = PaperOpticalNetwork(topology, num_slots)
        self.num_slots = int(num_slots)
        self.mod_reg = mod_reg if mod_reg is not None else paper_modulation_registry()
        self.slot_bw_hz = float(slot_bw_hz)
        self.guard_band_fs = int(guard_band_fs)
        self.time = 0.0
        self._release_heap: List[Tuple[float, int, Dict[str, Any]]] = []
        self._counter = 0
        self.allocations = 0
        self.releases = 0

    def reset(self) -> None:
        self.net.reset()
        self._release_heap = []
        self._counter = 0
        self.allocations = 0
        self.releases = 0
        self.time = 0.0

    # -- time ------------------------------------------------------------
    def advance_time(self, target_time: float) -> None:
        target_time = float(target_time)
        while self._release_heap and self._release_heap[0][0] <= target_time:
            _, _, conn = heapq.heappop(self._release_heap)
            self.net.release(conn["path"], conn["start_slot"], conn["num_slots"])
            self.releases += 1
        self.time = target_time

    # -- request view ------------------------------------------------------
    def build_view(
        self, req: PaperRequest, k_paths: int, path_sort: str
    ) -> Dict[str, Any]:
        """Per-candidate-path view: best modulation, FS demand, First-Fit state."""
        paths = get_candidate_paths(self.net, req.src_node, req.dst_node, k_paths, path_sort)
        view_paths: List[Dict[str, Any]] = []
        for path in paths:
            path_len = self.net.path_length_km(path)
            mod_idx = best_modulation_index(path_len, self.mod_reg)
            if mod_idx is None:
                view_paths.append({
                    "path": path, "path_len_km": path_len,
                    "hop_count": self.net.path_hops(path),
                    "best_mod_idx": None, "required_fs": None,
                    "first_fit_start": None, "eligible_blocks": [],
                    "all_free_blocks": [],
                })
                continue
            fs = required_fs(
                req.bitrate_gbps,
                self.mod_reg[mod_idx].spectral_efficiency,
                self.slot_bw_hz,
                self.guard_band_fs,
            )
            avail = self.net.get_availability(path)
            elig = eligible_blocks(avail, fs)
            view_paths.append({
                "path": path, "path_len_km": path_len,
                "hop_count": self.net.path_hops(path),
                "best_mod_idx": mod_idx, "required_fs": fs,
                "first_fit_start": elig[0][0] if elig else None,
                "eligible_blocks": elig,
                "all_free_blocks": contiguous_free_blocks(avail),
            })
        return {
            "req_id": req.req_id,
            "src_node": req.src_node,
            "dst_node": req.dst_node,
            "bitrate_gbps": req.bitrate_gbps,
            "paths": view_paths,
        }

    # -- commitment ----------------------------------------------------------
    def commit(
        self,
        req: PaperRequest,
        path_info: Dict[str, Any],
        start_slot: int,
    ) -> Dict[str, Any]:
        """Allocate and register release. Caller guarantees feasibility."""
        fs = int(path_info["required_fs"])
        ok = self.net.allocate(path_info["path"], start_slot, fs)
        if not ok:
            return {"success": False, "reason": "allocation_conflict"}
        self.allocations += 1
        conn = {
            "req_id": req.req_id,
            "path": list(path_info["path"]),
            "start_slot": int(start_slot),
            "num_slots": fs,
            "mod_idx": int(path_info["best_mod_idx"]),
            "path_len_km": float(path_info["path_len_km"]),
            "hop_count": int(path_info["hop_count"]),
        }
        release_time = self.time + float(req.holding_time)
        heapq.heappush(self._release_heap, (release_time, self._counter, conn))
        self._counter += 1
        return {
            "success": True,
            "reason": "admitted",
            "required_fs": fs,
            "start_slot": int(start_slot),
            "mod_idx": int(path_info["best_mod_idx"]),
            "path_len_km": float(path_info["path_len_km"]),
            "hop_count": int(path_info["hop_count"]),
        }

    def drain(self, horizon: float) -> None:
        self.advance_time(horizon)


# ---------------------------------------------------------------------------
# KSP-FF policy
# ---------------------------------------------------------------------------
def ksp_ff_decide(view: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    """Path-ordered KSP-FF with highest feasible modulation and First-Fit.

    Returns (path_idx, start_slot) of the first path that fits, else None.
    """
    for path_idx, path_info in enumerate(view["paths"]):
        if path_info["first_fit_start"] is not None:
            return path_idx, int(path_info["first_fit_start"])
    return None


# ---------------------------------------------------------------------------
# Simulation driver
# ---------------------------------------------------------------------------
def run_simulation(
    topology: str,
    requests: List[PaperRequest],
    decide: Callable[[Dict[str, Any], PaperRMSAEnv, PaperRequest], Optional[Tuple[int, int]]],
    k_paths: int,
    path_sort: str,
    warmup: int,
    num_slots: int = 100,
    mod_reg: Optional[ModulationRegistry] = None,
    env: Optional[PaperRMSAEnv] = None,
) -> Dict[str, Any]:
    """Run one simulation over a pre-generated trace.

    ``decide(view, env, req)`` returns (path_idx, start_slot) or None (block).
    """
    if env is None:
        env = PaperRMSAEnv(topology, num_slots, mod_reg=mod_reg)
    env.reset()

    evaluated = admitted = blocked = 0
    fs_sum = slot_hop_sum = path_len_sum = hop_sum = 0.0
    mod_counts: Dict[str, int] = {}
    block_reasons: Dict[str, int] = {}
    util_samples: List[float] = []

    for idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        view = env.build_view(req, k_paths, path_sort)
        decision = decide(view, env, req)
        result: Dict[str, Any]
        if decision is None:
            result = {"success": False, "reason": "no_feasible_path"}
        else:
            path_idx, start_slot = decision
            if path_idx >= len(view["paths"]):
                result = {"success": False, "reason": "invalid_path_choice"}
            else:
                path_info = view["paths"][path_idx]
                if path_info["required_fs"] is None:
                    result = {"success": False, "reason": "no_feasible_modulation"}
                elif start_slot is None:
                    result = {"success": False, "reason": "no_spectrum_on_chosen_path"}
                else:
                    result = env.commit(req, path_info, int(start_slot))

        if idx >= warmup:
            evaluated += 1
            if result["success"]:
                admitted += 1
                fs_sum += result["required_fs"]
                slot_hop_sum += result["required_fs"] * result["hop_count"]
                path_len_sum += result["path_len_km"]
                hop_sum += result["hop_count"]
                mod_name = env.mod_reg.names[result["mod_idx"]]
                mod_counts[mod_name] = mod_counts.get(mod_name, 0) + 1
            else:
                blocked += 1
                reason = result.get("reason", "unknown")
                block_reasons[reason] = block_reasons.get(reason, 0) + 1
            util_samples.append(env.net.utilization())

    if requests:
        env.drain(max(r.arrival_time + r.holding_time for r in requests) + 1.0)

    return {
        "topology": topology,
        "k_paths": k_paths,
        "path_sort": path_sort,
        "warmup": warmup,
        "evaluated": evaluated,
        "admitted": admitted,
        "blocked": blocked,
        "blocking_rate": blocked / max(evaluated, 1),
        "avg_fs": fs_sum / max(admitted, 1),
        "avg_slot_hops": slot_hop_sum / max(admitted, 1),
        "avg_path_len_km": path_len_sum / max(admitted, 1),
        "avg_hop_count": hop_sum / max(admitted, 1),
        "mod_counts": mod_counts,
        "block_reasons": block_reasons,
        "mean_utilization": float(np.mean(util_samples)) if util_samples else 0.0,
        "final_utilization": env.net.utilization(),
        "conservation": {
            "allocations": env.allocations,
            "releases": env.releases,
            "final_active": len(env._release_heap),
            "spectrum_empty_after_drain": bool(
                all(not np.any(s) for s in env.net.arc_states.values())
            ),
        },
    }
