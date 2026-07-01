"""MEC cluster with multiple servers distributed across network nodes."""
import numpy as np
from typing import List, Optional
from sa_hmarl.mec.server import MECServer


class MECCluster:
    """Collection of MEC servers."""

    def __init__(self, num_nodes: int, num_servers: int = 5,
                 seed: Optional[int] = None,
                 server_nodes: Optional[List[int]] = None,
                 capacities: Optional[List[float]] = None):
        self.rng = np.random.RandomState(seed)
        self.num_nodes = num_nodes
        self.servers: List[MECServer] = []

        if server_nodes is not None:
            nodes = list(server_nodes)
        else:
            nodes = self.rng.choice(num_nodes, size=min(num_servers, num_nodes), replace=False)

        for i, node_id in enumerate(nodes):
            if capacities is not None and i < len(capacities):
                cap = float(capacities[i])
            else:
                cap = float(self.rng.uniform(20, 100))
            mem = float(self.rng.uniform(8, 32))
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
        vec = np.array([s.utilization for s in self.servers], dtype=np.float32)
        if normalize and len(vec) > 0:
            m = vec.max()
            if m > 0:
                vec /= m
        return vec

    def server_utilizations(self) -> List[float]:
        return [s.utilization for s in self.servers]
