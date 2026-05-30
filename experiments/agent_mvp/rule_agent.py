"""Phase A: Predictor-guided Rule Agent.

Enumerates all (split_point, target_server) combinations,
queries the predictor for each, and selects by a heuristic score.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
from typing import List, Tuple, Optional, Dict

from dnn_models import DNNModel, SplitProfile
from mec_servers import MECCluster
from traffic_generator import DNNRequest


class RuleAgent:
    """Predictor-guided heuristic agent.

    Score = alpha * P_success - beta * delay_pred - gamma * server_load_penalty
    """

    def __init__(self,
                 predictor,
                 encoder,
                 mec_cluster: MECCluster,
                 alpha: float = 1.0,      # P_success weight
                 beta: float = 0.5,       # delay weight
                 gamma: float = 0.3,      # server load penalty
                 delta: float = 0.1,      # frag risk penalty (optional)
                 device: str = "cpu"):
        self.predictor = predictor
        self.encoder = encoder
        self.mec = mec_cluster
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.device = device
        self.predictor.eval()

    def decide(self, request: DNNRequest,
               network_state) -> Tuple[int, int, float, Dict]:
        """Select best (split_id, target_server_id) for the request.

        Returns:
            split_id: chosen split point (0,1,2)
            target_server_id: index into mec_cluster.servers
            best_score: the score of the chosen action
            info: dict with all evaluated candidates
        """
        model: DNNModel = request.model
        src = request.source_node
        best_score = -float('inf')
        best_action = None
        candidates = []

        with torch.no_grad():
            for split in model.splits:
                split_id = split.split_id
                bw_slots = split.bandwidth_slots

                for srv in self.mec.servers:
                    dst = srv.node_id
                    if dst == src:
                        continue  # Skip same-node offloading for now

                    # Encode network state for this src-dst pair
                    z = self.encoder.encode(src, dst)
                    z_t = torch.from_numpy(z).unsqueeze(0).float().to(self.device)

                    partition = torch.tensor([split_id], dtype=torch.long, device=self.device)
                    src_t = torch.tensor([src], dtype=torch.long, device=self.device)
                    dst_t = torch.tensor([dst], dtype=torch.long, device=self.device)

                    success_logit, delay_pred = self.predictor(partition, src_t, dst_t, z_t)
                    p_success = torch.sigmoid(success_logit).item()
                    delay_pred_val = delay_pred.item()

                    # Server compute delay
                    compute_delay = srv.compute_delay_ms(split.compute_cost, split.intermediate_size_mb)

                    # Total estimated delay = network + compute
                    total_delay_est = delay_pred_val + compute_delay

                    # Server load penalty (0-1 normalized)
                    load_penalty = srv.utilization

                    # Deadline penalty: how close are we to the deadline?
                    deadline_slack = request.deadline_ms - total_delay_est
                    deadline_penalty = max(0.0, 1.0 - deadline_slack / request.deadline_ms)

                    # Composite score (higher is better)
                    score = (self.alpha * p_success
                             - self.beta * (total_delay_est / 100.0)   # normalize delay to ~0-1
                             - self.gamma * load_penalty
                             - self.delta * deadline_penalty)

                    candidates.append({
                        "split_id": split_id,
                        "server_id": srv.server_id,
                        "node_id": dst,
                        "p_success": p_success,
                        "delay_net": delay_pred_val,
                        "delay_compute": compute_delay,
                        "delay_total": total_delay_est,
                        "server_util": srv.utilization,
                        "deadline_slack": deadline_slack,
                        "score": score,
                    })

                    if score > best_score:
                        best_score = score
                        best_action = (split_id, srv.server_id)

        if best_action is None:
            # Fallback: random valid action
            best_action = (0, 0)
            best_score = 0.0

        info = {
            "candidates": candidates,
            "num_evaluated": len(candidates),
        }
        return best_action[0], best_action[1], best_score, info
