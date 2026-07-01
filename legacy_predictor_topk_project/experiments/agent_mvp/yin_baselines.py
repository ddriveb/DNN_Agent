"""Yin-protocol-style baselines: WO, DF, RF, IWD-Approx.

All baselines share:
- Greedy partition for split selection (except WO, which uses local-only split=0)
- Shortest-path + First-Fit for spectrum allocation (except WO, no lightpath)
- No predictor, no learned model, no CorrectionNet.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import networkx as nx
from typing import Tuple, Dict, Optional, List

from dnn_models import DNNModel, get_split_bandwidth_map
from mec_servers import MECCluster, MECServer
from traffic_generator import DNNRequest


# ------------------------------------------------------------------
# Shared: greedy partition (estimates delay for each split)
# ------------------------------------------------------------------

def greedy_partition(request: DNNRequest,
                     source_node: int,
                     target_server: MECServer,
                     env) -> Tuple[int, float]:
    """Select split_id that minimizes estimated total delay + spectrum penalty.

    Heavily penalizes large bandwidth to avoid spectrum blocking,
    aligning with the observation that YinLike (always split=0, bw=1)
    achieves much lower blocking than greedy delay-only partitioning.
    """
    src = source_node
    dst = target_server.node_id
    model: DNNModel = request.model

    # Estimate shortest-path network delay (same formula as mapper)
    if src == dst:
        net_delay_ms = 0.0
    else:
        path = nx.shortest_path(env.net.G, src, dst, weight="weight")
        dist = sum(env.net.G[u][v]["length"] for u, v in zip(path[:-1], path[1:]))
        net_delay_s = dist / 2e5 + (len(path) - 1) * 1e-3
        net_delay_ms = net_delay_s * 1000.0

    best_split_id = 0
    best_score = float('inf')

    for split in model.splits:
        compute_delay = target_server.compute_delay_ms(
            split.compute_cost, split.intermediate_size_mb
        )
        total_delay = net_delay_ms + compute_delay

        # Score = total delay (bandwidth differences now naturally reflected in model)
        score = total_delay

        if score < best_score:
            best_score = score
            best_split_id = split.split_id

    return best_split_id, best_score


def estimate_shortest_path_delay_ms(env, src: int, dst: int) -> float:
    """Return estimated one-way propagation + processing delay in ms."""
    if src == dst:
        return 0.0
    path = nx.shortest_path(env.net.G, src, dst, weight="weight")
    dist = sum(env.net.G[u][v]["length"] for u, v in zip(path[:-1], path[1:]))
    net_delay_s = dist / 2e5 + (len(path) - 1) * 1e-3
    return net_delay_s * 1000.0


# ------------------------------------------------------------------
# 1. WO — Without Offloading (local execution only)
# ------------------------------------------------------------------

class WOAgent:
    """Without Offloading: execute entirely on the source node's local MEC server.

    No lightpath is allocated; network delay is zero.
    If the source node has no MEC server, falls back to the nearest server
    (this is a graceful fallback, but conceptually still "local-first").
    """

    def __init__(self, mec_cluster: MECCluster):
        self.mec = mec_cluster

    def decide(self, request: DNNRequest, env) -> Tuple[int, int, float, Dict]:
        src = request.source_node
        local_srv = self.mec.get_server_at_node(src)

        if local_srv is not None:
            # Local execution: split=0 minimizes bandwidth (not used anyway)
            # but compute-wise split=0 has highest local compute cost.
            # For pure local, we just use split=0 and target the local server.
            split_id = 0
            server_id = local_srv.server_id
            # Estimate delay
            _, est_delay = greedy_partition(request, src, local_srv, env)
            return split_id, server_id, -est_delay, {"type": "wo", "local": True}

        # Fallback: nearest server (should not happen in standard setups)
        best_srv = None
        best_dist = float('inf')
        for srv in self.mec.servers:
            if srv.node_id == src:
                continue
            try:
                d = nx.shortest_path_length(env.net.G, src, srv.node_id, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if d < best_dist:
                best_dist = d
                best_srv = srv

        if best_srv is None:
            best_srv = self.mec.servers[0]

        split_id, est_delay = greedy_partition(request, src, best_srv, env)
        return split_id, best_srv.server_id, -est_delay, {"type": "wo", "local": False}


# ------------------------------------------------------------------
# 2. DF — Distance First
# ------------------------------------------------------------------

class DFAgent:
    """Distance First: choose the MEC server with shortest path distance from source."""

    def __init__(self, mec_cluster: MECCluster):
        self.mec = mec_cluster

    def decide(self, request: DNNRequest, env) -> Tuple[int, int, float, Dict]:
        src = request.source_node
        best_srv = None
        best_dist = float('inf')

        for srv in self.mec.servers:
            if srv.node_id == src:
                continue
            try:
                d = nx.shortest_path_length(env.net.G, src, srv.node_id, weight="weight")
            except nx.NetworkXNoPath:
                continue
            if d < best_dist:
                best_dist = d
                best_srv = srv

        if best_srv is None:
            # Fallback to first available server
            valid = [s for s in self.mec.servers if s.node_id != src]
            best_srv = valid[0] if valid else self.mec.servers[0]

        split_id, est_delay = greedy_partition(request, src, best_srv, env)
        return split_id, best_srv.server_id, -est_delay, {"type": "df", "dist": best_dist}


# ------------------------------------------------------------------
# 3. RF — Resource First
# ------------------------------------------------------------------

class RFAgent:
    """Resource First: choose the MEC server with lowest utilization."""

    def __init__(self, mec_cluster: MECCluster):
        self.mec = mec_cluster

    def decide(self, request: DNNRequest, env) -> Tuple[int, int, float, Dict]:
        src = request.source_node
        valid_servers = [s for s in self.mec.servers if s.node_id != src]
        if not valid_servers:
            valid_servers = self.mec.servers

        best_srv = min(valid_servers, key=lambda s: s.utilization)
        split_id, est_delay = greedy_partition(request, src, best_srv, env)
        return split_id, best_srv.server_id, -est_delay, {
            "type": "rf",
            "util": best_srv.utilization,
        }


# ------------------------------------------------------------------
# 4. IWD-Approx — IWD-inspired heuristic search
# ------------------------------------------------------------------

class IWDApproxAgent:
    """IWD-inspired heuristic: search target server by composite cost.

    Cost = w1 * normalized_delay + w2 * server_load
         + w3 * spectrum_shortage + w4 * deadline_penalty

    Simplified: single-pass evaluation over all valid servers + small perturbations.
    """

    def __init__(self,
                 mec_cluster: MECCluster,
                 w1: float = 1.0,   # delay weight
                 w2: float = 0.5,   # load weight
                 w3: float = 0.3,   # spectrum weight
                 w4: float = 0.5,   # deadline penalty weight
                 num_perturbations: int = 3,
                 seed: int = 42):
        self.mec = mec_cluster
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.num_perturbations = num_perturbations
        self.rng = np.random.RandomState(seed)

    def _server_cost(self, request: DNNRequest, srv: MECServer, env) -> Tuple[float, int, float]:
        """Return (cost, split_id, est_delay) for assigning request to srv."""
        src = request.source_node
        dst = srv.node_id

        if dst == src:
            return float('inf'), 0, float('inf')

        # 1. Estimate shortest-path delay
        net_delay_ms = estimate_shortest_path_delay_ms(env, src, dst)

        # 2. Greedy partition (best split for this server)
        split_id, _ = greedy_partition(request, src, srv, env)
        split = next(s for s in request.model.splits if s.split_id == split_id)
        compute_delay = srv.compute_delay_ms(split.compute_cost, split.intermediate_size_mb)
        total_delay = net_delay_ms + compute_delay

        # 3. Normalized delay
        delay_norm = total_delay / max(request.deadline_ms, 1.0)

        # 4. Load
        load = srv.utilization

        # 5. Spectrum shortage on shortest path
        try:
            path = nx.shortest_path(env.net.G, src, dst, weight="weight")
            avail = env.net.get_available_slots(path)
            avail_ratio = float(np.sum(avail)) / env.net.num_slots
            spectrum_penalty = 1.0 - avail_ratio
        except Exception:
            spectrum_penalty = 1.0

        # 6. Deadline penalty
        deadline_penalty = max(0.0, (total_delay - request.deadline_ms) / max(request.deadline_ms, 1.0))

        cost = (self.w1 * delay_norm
                + self.w2 * load
                + self.w3 * spectrum_penalty
                + self.w4 * deadline_penalty)
        return cost, split_id, total_delay

    def decide(self, request: DNNRequest, env) -> Tuple[int, int, float, Dict]:
        src = request.source_node
        valid_servers = [s for s in self.mec.servers if s.node_id != src]
        if not valid_servers:
            valid_servers = self.mec.servers

        # --- Phase 1: evaluate all candidates ---
        candidates = []
        for srv in valid_servers:
            cost, split_id, est_delay = self._server_cost(request, srv, env)
            candidates.append({
                "server_id": srv.server_id,
                "node_id": srv.node_id,
                "cost": cost,
                "split_id": split_id,
                "est_delay": est_delay,
                "util": srv.utilization,
            })

        # --- Phase 2: small perturbation search (IWD-like exploration) ---
        best = min(candidates, key=lambda c: c["cost"])
        best_cost = best["cost"]
        best_choice = best

        for _ in range(self.num_perturbations):
            # Pick a random candidate and slightly perturb its cost
            cand = self.rng.choice(candidates)
            noise = self.rng.uniform(-0.05, 0.05)
            perturbed_cost = cand["cost"] + noise
            if perturbed_cost < best_cost:
                best_cost = perturbed_cost
                best_choice = cand

        return best_choice["split_id"], best_choice["server_id"], -best_cost, {
            "type": "iwd_approx",
            "cost": best_cost,
            "est_delay": best_choice["est_delay"],
            "candidates": len(candidates),
        }
