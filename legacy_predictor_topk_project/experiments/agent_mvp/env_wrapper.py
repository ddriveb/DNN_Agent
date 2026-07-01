"""Environment wrapper: integrates network, encoder, mapper, predictor, MEC.

Provides a unified step() interface for agent evaluation.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from dnn_models import get_split_bandwidth_map
from mec_servers import MECCluster, MECServer
from traffic_generator import DNNRequest


@dataclass
class StepResult:
    """Result of executing one action."""
    success: bool
    reward: float
    real_delay_ms: float
    path: Optional[List[int]]
    start_slot: Optional[int]
    bandwidth_slots: int
    info: Dict


class DNNOpticalEnv:
    """Gym-like environment for DNN offloading over elastic optical networks."""

    def __init__(self,
                 network: OpticalNetwork,
                 encoder: Encoder,
                 mapper: KSPMapper,
                 mec_cluster: MECCluster,
                 reward_version: str = "v1"):
        self.net = network
        self.encoder = encoder
        self.mapper = mapper
        self.mec = mec_cluster
        self.reward_version = reward_version
        self.bw_map = get_split_bandwidth_map()
        self.active_connections: List[Tuple[List[int], int, int, float]] = []
        self.time = 0.0
        self.stats = {
            "total_requests": 0,
            "accepted": 0,
            "blocked": 0,
            "total_delay": 0.0,
            "total_slots_used": 0,
            "deadline_violations": 0,
            "sum_frag_index": 0.0,
            "sum_largest_free_block_ratio": 0.0,
            "sum_spectrum_utilization": 0.0,
            "sum_free_block_count": 0.0,
            "num_observations": 0,
            "sum_delta_frag_on_accept": 0.0,
            "sum_delta_lfb_on_accept": 0.0,
            "sum_future_risk_on_accept": 0.0,
        }

    def reset(self):
        self.net.reset()
        self.mec.reset()
        self.active_connections = []
        self.time = 0.0
        self.stats = {
            "total_requests": 0,
            "accepted": 0,
            "blocked": 0,
            "total_delay": 0.0,
            "total_slots_used": 0,
            "deadline_violations": 0,
            "sum_frag_index": 0.0,
            "sum_largest_free_block_ratio": 0.0,
            "sum_spectrum_utilization": 0.0,
            "sum_free_block_count": 0.0,
            "num_observations": 0,
            "sum_delta_frag_on_accept": 0.0,
            "sum_delta_lfb_on_accept": 0.0,
            "sum_future_risk_on_accept": 0.0,
        }

    def advance_time(self, new_time: float):
        """Release expired connections up to new_time."""
        still_active = []
        for conn in self.active_connections:
            path, start, bw, release_t = conn[:4]
            if release_t <= new_time:
                self.net.release(path, start, bw)
                # Release MEC compute if tracked
                if len(conn) >= 6 and conn[4] >= 0:
                    srv_id, compute_cost = conn[4], conn[5]
                    self.mec.get_server_by_id(srv_id).release_task(compute_cost)
            else:
                still_active.append(conn)
        self.active_connections = still_active
        self.time = new_time

    def step(self, request: DNNRequest,
             split_id: int,
             target_server_id: int) -> StepResult:
        """Execute one high-level action.

        Args:
            request: The DNN inference request
            split_id: Which split point (0,1,2)
            target_server_id: Index into mec_cluster.servers

        Returns:
            StepResult with success flag, reward, delay, etc.
        """
        self.stats["total_requests"] += 1

        model = request.model
        src = request.source_node
        srv = self.mec.get_server_by_id(target_server_id)
        dst = srv.node_id
        bw_slots = self.bw_map[split_id]
        global_before = self.net.get_global_spectrum_stats()

        # --- 1. Try to allocate lightpath via KSPMapper ---
        success, path, start_slot, net_delay = self.mapper.map(src, dst, bw_slots)

        if not success:
            self.stats["blocked"] += 1
            self._record_observation(global_before)
            reward = self._compute_reward(False, None, None, request, srv, split_id)
            return StepResult(
                success=False,
                reward=reward,
                real_delay_ms=0.0,
                path=None,
                start_slot=None,
                bandwidth_slots=bw_slots,
                info={"blocking_reason": "no_path", "target_node": dst},
            )

        # --- 2. Compute total delay = network + MEC processing ---
        split = next(s for s in model.splits if s.split_id == split_id)
        compute_delay = srv.compute_delay_ms(split.compute_cost, split.intermediate_size_mb)
        total_delay_ms = (net_delay * 1000.0) + compute_delay  # net_delay is in seconds

        # --- 3. Check deadline ---
        deadline_met = total_delay_ms <= request.deadline_ms
        if not deadline_met:
            # Release the allocation since we can't meet SLA
            self.net.release(path, start_slot, bw_slots)
            self.stats["blocked"] += 1
            self.stats["deadline_violations"] += 1
            self._record_observation(global_before)
            reward = self._compute_reward(False, total_delay_ms, deadline_met, request, srv, split_id)
            return StepResult(
                success=False,
                reward=reward,
                real_delay_ms=total_delay_ms,
                path=path,
                start_slot=start_slot,
                bandwidth_slots=bw_slots,
                info={"blocking_reason": "deadline_violation",
                      "target_node": dst,
                      "delay_ms": total_delay_ms},
            )

        # --- 4. Accept: book resources ---
        self.stats["accepted"] += 1
        self.stats["total_delay"] += total_delay_ms
        self.stats["total_slots_used"] += bw_slots * (len(path) - 1)

        # Hold the lightpath and MEC compute
        release_t = self.time + request.holding_time
        self.active_connections.append((path, start_slot, bw_slots, release_t,
                                        target_server_id, split.compute_cost))

        # Book MEC compute
        srv.allocate_task(split.compute_cost)
        global_after = self.net.get_global_spectrum_stats()
        delta_frag = global_after["avg_frag_index"] - global_before["avg_frag_index"]
        delta_lfb = (
            global_before["largest_free_block_ratio"] - global_after["largest_free_block_ratio"]
        )
        future_risk = max(
            0.0,
            (bw_slots - (global_after["largest_free_block_ratio"] * self.net.num_slots)) / max(float(bw_slots), 1.0),
        )
        self.stats["sum_delta_frag_on_accept"] += delta_frag
        self.stats["sum_delta_lfb_on_accept"] += delta_lfb
        self.stats["sum_future_risk_on_accept"] += future_risk
        self._record_observation(global_after)

        reward = self._compute_reward(True, total_delay_ms, deadline_met, request, srv, split_id)

        return StepResult(
            success=True,
            reward=reward,
            real_delay_ms=total_delay_ms,
            path=path,
            start_slot=start_slot,
            bandwidth_slots=bw_slots,
            info={
                "target_node": dst,
                "delay_ms": total_delay_ms,
                "net_delay_ms": net_delay * 1000.0,
                "compute_delay_ms": compute_delay,
                "server_util_after": srv.utilization,
                "path_length": len(path) if path else 0,
                "frag_before": global_before["avg_frag_index"],
                "frag_after": global_after["avg_frag_index"],
                "delta_frag": delta_frag,
                "lfb_before": global_before["largest_free_block_ratio"],
                "lfb_after": global_after["largest_free_block_ratio"],
                "delta_lfb": delta_lfb,
                "future_risk": future_risk,
            },
        )

    def _compute_reward(self, success: bool, delay_ms: Optional[float],
                        deadline_met: Optional[bool],
                        request: DNNRequest, srv: MECServer, split_id: int) -> float:
        """Compute reward based on the configured reward version."""
        if self.reward_version == "v1":
            # Simplest: success/fail + normalized delay
            if not success:
                return -2.0
            delay_norm = delay_ms / request.deadline_ms
            return 1.0 - delay_norm

        elif self.reward_version == "v2":
            # Add resource consumption penalty
            if not success:
                return -5.0
            delay_norm = delay_ms / request.deadline_ms
            bw_slots = get_split_bandwidth_map()[split_id]
            resource_penalty = (bw_slots / 8.0) * 0.2  # max 0.2 for 8 slots
            load_penalty = srv.utilization * 0.1
            return 1.0 - 0.5 * delay_norm - resource_penalty - load_penalty

        elif self.reward_version == "v3":
            # Add fragmentation penalty (computed from global state)
            if not success:
                return -5.0
            delay_norm = delay_ms / request.deadline_ms if delay_ms else 0.0
            bw_slots = get_split_bandwidth_map()[split_id]
            resource_penalty = (bw_slots / 8.0) * 0.2
            load_penalty = srv.utilization * 0.1
            # Fragmentation: reuse encoder's global frag index
            frag = self._global_frag_index()
            frag_penalty = frag * 0.1
            return 1.0 - 0.5 * delay_norm - resource_penalty - load_penalty - frag_penalty

        else:
            raise ValueError(f"Unknown reward version: {self.reward_version}")

    def _global_frag_index(self) -> float:
        """Compute average fragmentation across all links."""
        return self.net.get_global_spectrum_stats()["avg_frag_index"]

    def _record_observation(self, spectrum_stats: Dict):
        self.stats["sum_frag_index"] += spectrum_stats["avg_frag_index"]
        self.stats["sum_largest_free_block_ratio"] += spectrum_stats["largest_free_block_ratio"]
        self.stats["sum_spectrum_utilization"] += spectrum_stats["spectrum_utilization"]
        self.stats["sum_free_block_count"] += spectrum_stats["avg_free_block_count"]
        self.stats["num_observations"] += 1

    @staticmethod
    def _max_consecutive(arr) -> int:
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

    def get_state_vector(self, request: DNNRequest, for_server_id: int) -> np.ndarray:
        """Get the state vector (z) for a given request and target server."""
        srv = self.mec.get_server_by_id(for_server_id)
        return self.encoder.encode(request.source_node, srv.node_id)

    def get_metrics(self) -> Dict:
        """Return current evaluation metrics."""
        total = self.stats["total_requests"]
        accepted = self.stats["accepted"]
        obs = self.stats["num_observations"]
        return {
            "blocking_rate": self.stats["blocked"] / total if total > 0 else 0.0,
            "acceptance_rate": accepted / total if total > 0 else 0.0,
            "avg_delay_ms": self.stats["total_delay"] / accepted if accepted > 0 else 0.0,
            "avg_slots_per_req": self.stats["total_slots_used"] / accepted if accepted > 0 else 0.0,
            "deadline_violation_rate": self.stats["deadline_violations"] / total if total > 0 else 0.0,
            "avg_frag_index": self.stats["sum_frag_index"] / obs if obs > 0 else 0.0,
            "avg_largest_free_block_ratio": self.stats["sum_largest_free_block_ratio"] / obs if obs > 0 else 0.0,
            "avg_spectrum_utilization": self.stats["sum_spectrum_utilization"] / obs if obs > 0 else 0.0,
            "avg_free_block_count": self.stats["sum_free_block_count"] / obs if obs > 0 else 0.0,
            "avg_delta_frag_on_accept": self.stats["sum_delta_frag_on_accept"] / accepted if accepted > 0 else 0.0,
            "avg_delta_lfb_on_accept": self.stats["sum_delta_lfb_on_accept"] / accepted if accepted > 0 else 0.0,
            "avg_future_risk_on_accept": self.stats["sum_future_risk_on_accept"] / accepted if accepted > 0 else 0.0,
            "total_requests": total,
            "accepted": accepted,
            "blocked": self.stats["blocked"],
        }
