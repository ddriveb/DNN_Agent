"""Paper-condition topology loader and env factory for cross-paper comparison."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List

import numpy as np
import networkx as nx

import sys

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))
sys.path.insert(0, str(Path(__file__).parent.parent / "agent_mvp"))

from encoder import Encoder
from env_wrapper import DNNOpticalEnv
from mapper import KSPMapper
from mec_servers import MECServer


@dataclass
class ParsedServer:
    server_id: int
    node_id: int
    compute_capacity_hz: float
    current_load_hz: float


class PaperTopologyNetwork:
    """Optical network with topology loaded from the paper-style text format."""

    def __init__(
        self,
        topology_file: str | Path,
        *,
        load_factor: float = 0.6,
        fragmentation: float = 0.2,
        seed: int = 42,
    ):
        self.topology_file = Path(topology_file)
        self.topology = self.topology_file.stem
        self.rng = np.random.RandomState(seed)
        self._load_factor = float(load_factor)
        self._fragmentation = float(fragmentation)
        parsed = self._parse(self.topology_file)
        self.name = parsed["name"]
        self.nodes = parsed["nodes"]
        self.servers = parsed["servers"]
        self.G = nx.Graph()
        self.num_slots = parsed["num_slots"]
        self.NUM_NODES = len(self.nodes)
        self._build_graph(parsed["links"])
        self.reset()

    def _parse(self, path: Path) -> Dict:
        lines = [line.strip() for line in path.read_text().splitlines() if line.strip()]
        idx = 0
        name = lines[idx].split()[1]
        idx += 1

        n_nodes = int(lines[idx].split()[1])
        idx += 1
        nodes = []
        for _ in range(n_nodes):
            parts = lines[idx].split()
            nodes.append(
                {
                    "id": int(parts[0]),
                    "x": float(parts[1]),
                    "y": float(parts[2]),
                    "has_server": int(parts[3]),
                }
            )
            idx += 1

        n_links = int(lines[idx].split()[1])
        idx += 1
        links = []
        num_slots = None
        for _ in range(n_links):
            parts = lines[idx].split()
            u, v = int(parts[0]), int(parts[1])
            length, slots = float(parts[2]), int(parts[3])
            links.append((u, v, length, slots))
            num_slots = slots if num_slots is None else num_slots
            idx += 1

        n_servers = int(lines[idx].split()[1])
        idx += 1
        servers: List[ParsedServer] = []
        for _ in range(n_servers):
            parts = lines[idx].split()
            servers.append(
                ParsedServer(
                    server_id=int(parts[0]),
                    node_id=int(parts[1]),
                    compute_capacity_hz=float(parts[2]),
                    current_load_hz=float(parts[3]),
                )
            )
            idx += 1

        return {
            "name": name,
            "nodes": nodes,
            "links": links,
            "servers": servers,
            "num_slots": int(num_slots or 100),
        }

    def _build_graph(self, links: List[tuple[int, int, float, int]]) -> None:
        for node in self.nodes:
            self.G.add_node(node["id"], pos=(node["x"], node["y"]))
        for u, v, length, _slots in links:
            self.G.add_edge(u, v, length=length, weight=length)

    def reset(self):
        self.link_states = {}
        for u, v in self.G.edges():
            self.link_states[(min(u, v), max(u, v))] = np.zeros(self.num_slots, dtype=bool)
        self.initialize_link_load(self._load_factor, self._fragmentation)

    def initialize_link_load(self, load_factor: float = 0.6, fragmentation: float = 0.2) -> None:
        for link, bitmap in self.link_states.items():
            bitmap[:] = False
            n_fs = len(bitmap)
            n_occupied = int(n_fs * load_factor)
            n_frag = int(n_occupied * fragmentation)
            n_cont = n_occupied - n_frag

            if n_cont > 0:
                max_start = max(0, n_fs - n_cont)
                start = int(self.rng.randint(0, max_start + 1))
                bitmap[start : start + n_cont] = True

            free_positions = np.where(~bitmap)[0]
            if len(free_positions) > 0 and n_frag > 0:
                chosen = self.rng.choice(
                    free_positions,
                    size=min(n_frag, len(free_positions)),
                    replace=False,
                )
                bitmap[chosen] = True

    def allocate(self, path, start_slot, num_slots):
        if start_slot < 0 or start_slot + num_slots > self.num_slots:
            return False
        for u, v in zip(path[:-1], path[1:]):
            link = (min(u, v), max(u, v))
            if np.any(self.link_states[link][start_slot : start_slot + num_slots]):
                return False
        for u, v in zip(path[:-1], path[1:]):
            link = (min(u, v), max(u, v))
            self.link_states[link][start_slot : start_slot + num_slots] = True
        return True

    def release(self, path, start_slot, num_slots):
        if path is None:
            return
        for u, v in zip(path[:-1], path[1:]):
            link = (min(u, v), max(u, v))
            self.link_states[link][start_slot : start_slot + num_slots] = False

    def get_available_slots(self, path):
        avail = np.ones(self.num_slots, dtype=bool)
        for u, v in zip(path[:-1], path[1:]):
            link = (min(u, v), max(u, v))
            avail &= ~self.link_states[link]
        return avail

    @staticmethod
    def _max_consecutive(arr) -> int:
        max_len = curr = 0
        for v in arr:
            if v:
                curr += 1
                max_len = max(max_len, curr)
            else:
                curr = 0
        return max_len

    @staticmethod
    def _count_blocks(arr) -> int:
        count = 0
        in_block = False
        for v in arr:
            if v and not in_block:
                count += 1
                in_block = True
            elif not v:
                in_block = False
        return count

    def summarize_availability(self, free_arr):
        total_free = int(np.sum(free_arr))
        max_free = self._max_consecutive(free_arr)
        free_blocks = self._count_blocks(free_arr)
        frag = 1.0 if total_free == 0 else (0.0 if total_free == self.num_slots else 1.0 - (max_free / total_free))
        return {
            "total_free_slots": total_free,
            "max_free_slots": max_free,
            "largest_free_block_ratio": max_free / max(self.num_slots, 1),
            "free_block_count": free_blocks,
            "frag_index": float(frag),
        }

    def get_path_spectrum_stats(self, path):
        stats = self.summarize_availability(self.get_available_slots(path))
        stats["path_length"] = max(len(path) - 1, 0)
        return stats

    def get_global_spectrum_stats(self):
        total_slots = len(self.link_states) * self.num_slots
        occupied = 0
        frag_vals, max_free_vals, free_block_vals = [], [], []
        for slots in self.link_states.values():
            occupied += int(np.sum(slots))
            stats = self.summarize_availability(~slots)
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


class FixedMECCluster:
    """MEC cluster with fixed server placement and load from the paper config."""

    def __init__(self, servers: List[ParsedServer]):
        self.servers = []
        for spec in servers:
            srv = MECServer(
                spec.server_id,
                spec.node_id,
                compute_capacity_gflops=spec.compute_capacity_hz / 1e9,
                memory_capacity_gb=16.0,
            )
            srv.current_load = spec.current_load_hz / 1e9
            srv._initial_load = srv.current_load
            self.servers.append(srv)

    def get_server(self, server_id: int):
        return self.servers[server_id]

    def get_server_by_id(self, server_id: int):
        return self.servers[server_id]

    def get_server_at_node(self, node_id: int):
        for srv in self.servers:
            if srv.node_id == node_id:
                return srv
        return None

    @property
    def server_node_ids(self):
        return [srv.node_id for srv in self.servers]

    def reset(self):
        for srv in self.servers:
            srv.current_load = srv._initial_load
            srv.active_tasks = 0

    def server_load_vector(self, normalize: bool = True):
        vec = np.array([srv.utilization for srv in self.servers], dtype=np.float32)
        if normalize and len(vec) > 0:
            m = vec.max()
            if m > 0:
                vec /= m
        return vec


def create_paper_env(
    topology_file: str | Path,
    *,
    load_factor: float = 0.6,
    fragmentation: float = 0.2,
    seed: int = 42,
):
    net = PaperTopologyNetwork(
        topology_file,
        load_factor=load_factor,
        fragmentation=fragmentation,
        seed=seed,
    )
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    mec = FixedMECCluster(net.servers)
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version="v1")
    return env, encoder, mec
