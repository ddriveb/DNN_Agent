"""MEC server model: compute capacity, load tracking, queuing delay."""
import numpy as np
from typing import List, Optional


class MECServer:
    """A single MEC/edge server with compute resources."""

    def __init__(self, server_id: int, node_id: int,
                 compute_capacity_gflops: float = 50.0,
                 memory_capacity_gb: float = 16.0):
        self.server_id = server_id
        self.node_id = node_id          # Which network node this server is attached to
        self.compute_capacity = compute_capacity_gflops
        self.memory_capacity = memory_capacity_gb
        self.current_load = 0.0         # Current GFLOPS load
        self.active_tasks = 0

    @property
    def available_compute(self) -> float:
        return max(self.compute_capacity - self.current_load, 0.0)

    @property
    def utilization(self) -> float:
        return self.current_load / self.compute_capacity if self.compute_capacity > 0 else 1.0

    def compute_delay_ms(self, compute_cost: float, data_size_mb: float) -> float:
        """Estimate processing delay for a given compute cost.

        Uses M/M/1-inspired approximation:
            delay = compute_cost / available_compute * base_factor + memory_overhead
        """
        if self.available_compute <= 0:
            return float('inf')
        # Base compute time: normalized cost / available capacity
        # Scale so that compute_cost=1.0 on full capacity takes ~20ms
        base_ms = (compute_cost / self.available_compute) * 20.0
        # Small memory transfer overhead
        mem_ms = data_size_mb * 0.5   # 0.5 ms per MB
        return base_ms + mem_ms

    def allocate_task(self, compute_cost: float):
        """Book compute resources for an incoming task."""
        self.current_load += compute_cost * self.compute_capacity
        self.current_load = min(self.current_load, self.compute_capacity)
        self.active_tasks += 1

    def release_task(self, compute_cost: float):
        """Release compute resources when task finishes."""
        self.current_load -= compute_cost * self.compute_capacity
        self.current_load = max(self.current_load, 0.0)
        self.active_tasks = max(self.active_tasks - 1, 0)

    def reset(self):
        self.current_load = 0.0
        self.active_tasks = 0


class MECCluster:
    """Collection of MEC servers distributed across network nodes."""

    def __init__(self, num_nodes: int, num_servers: int = 5, seed: Optional[int] = None):
        self.rng = np.random.RandomState(seed)
        self.num_nodes = num_nodes
        self.servers: List[MECServer] = []

        # Randomly place servers on network nodes
        server_nodes = self.rng.choice(num_nodes, size=min(num_servers, num_nodes), replace=False)
        for i, node_id in enumerate(server_nodes):
            cap = float(self.rng.uniform(20, 100))   # 20-100 GFLOPS
            mem = float(self.rng.uniform(8, 32))     # 8-32 GB
            self.servers.append(MECServer(i, int(node_id), cap, mem))

    def get_server(self, server_id: int) -> MECServer:
        return self.servers[server_id]

    def get_server_at_node(self, node_id: int) -> Optional[MECServer]:
        for s in self.servers:
            if s.node_id == node_id:
                return s
        return None

    def get_server_by_id(self, server_id: int) -> MECServer:
        return self.servers[server_id]

    @property
    def server_node_ids(self) -> List[int]:
        return [s.node_id for s in self.servers]

    def reset(self):
        for s in self.servers:
            s.reset()

    def server_load_vector(self, normalize: bool = True) -> np.ndarray:
        """Return utilization vector for all servers."""
        vec = np.array([s.utilization for s in self.servers], dtype=np.float32)
        if normalize and len(vec) > 0:
            m = vec.max()
            if m > 0:
                vec /= m
        return vec
