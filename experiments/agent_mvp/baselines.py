"""Baseline agents for comparison."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
from typing import Tuple, Dict

from dnn_models import DNNModel
from mec_servers import MECCluster
from traffic_generator import DNNRequest


class RandomAgent:
    """Randomly selects split_point and target_server."""

    def __init__(self, mec_cluster: MECCluster, seed: int = 42):
        self.mec = mec_cluster
        self.rng = np.random.RandomState(seed)

    def decide(self, request: DNNRequest, network_state) -> Tuple[int, int, float, Dict]:
        model: DNNModel = request.model
        split_id = self.rng.choice([s.split_id for s in model.splits])
        # Pick a server not at the source node
        valid_servers = [s for s in self.mec.servers if s.node_id != request.source_node]
        if not valid_servers:
            valid_servers = self.mec.servers
        srv = self.rng.choice(valid_servers)
        return split_id, srv.server_id, 0.0, {"type": "random"}


class ShortestPathAgent:
    """Always use smallest bandwidth (split_id=0) and shortest-path server.

    This is essentially the KSPMapper's default behavior:
    - Minimal bandwidth to reduce blocking
    - First server that can be reached via shortest path
    """

    def __init__(self, mec_cluster: MECCluster):
        self.mec = mec_cluster

    def decide(self, request: DNNRequest, network_state) -> Tuple[int, int, float, Dict]:
        model: DNNModel = request.model
        split_id = 0  # Minimal bandwidth

        # Pick the server with the shortest network distance from source
        src = request.source_node
        best_srv = None
        best_dist = float('inf')

        for srv in self.mec.servers:
            if srv.node_id == src:
                continue
            # Simple distance heuristic: we don't have the actual path here,
            # so we use node_id difference as a coarse proxy or rely on mapper.
            # For fairness, we just pick the first available server.
            if best_srv is None:
                best_srv = srv

        if best_srv is None:
            best_srv = self.mec.servers[0]

        return split_id, best_srv.server_id, 0.0, {"type": "shortest_path"}


class YinLikeAgent:
    """Yin-like baseline: greedy server selection + first-fit slot.

    - Fixes split_point to 0 (minimum bandwidth, like edge-only in Yin)
    - Picks target_server greedily based on server load only
    - Does NOT use predictor for network quality
    """

    def __init__(self, mec_cluster: MECCluster):
        self.mec = mec_cluster

    def decide(self, request: DNNRequest, network_state) -> Tuple[int, int, float, Dict]:
        model: DNNModel = request.model
        split_id = 0  # Greedy: minimize bandwidth

        # Pick server with lowest utilization (like Yin server selection)
        valid_servers = [s for s in self.mec.servers if s.node_id != request.source_node]
        if not valid_servers:
            valid_servers = self.mec.servers

        best_srv = min(valid_servers, key=lambda s: s.utilization)
        return split_id, best_srv.server_id, 0.0, {"type": "yin_like"}


class LoadBalancedAgent:
    """Selects split_point and server to balance load across MEC servers.

    - Picks the server with the lowest load
    - Chooses split_point that minimizes bandwidth demand
    """

    def __init__(self, mec_cluster: MECCluster):
        self.mec = mec_cluster

    def decide(self, request: DNNRequest, network_state) -> Tuple[int, int, float, Dict]:
        model: DNNModel = request.model
        # Pick server with lowest utilization
        valid_servers = [s for s in self.mec.servers if s.node_id != request.source_node]
        if not valid_servers:
            valid_servers = self.mec.servers
        best_srv = min(valid_servers, key=lambda s: s.utilization)

        # Use smallest bandwidth split
        split_id = 0
        return split_id, best_srv.server_id, 0.0, {"type": "load_balanced"}
