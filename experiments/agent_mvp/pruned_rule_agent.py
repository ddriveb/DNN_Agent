"""RuleAgent with action space pruning.

Pruning rules:
  1. Deadline urgency: if deadline < threshold, only keep nearby servers
  2. Server overload: skip servers with util > threshold
  3. Bandwidth risk: if bw_slots > threshold and best P_success < threshold, skip
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
from typing import List, Tuple, Dict, Optional

from dnn_models import DNNModel
from mec_servers import MECCluster
from traffic_generator import DNNRequest
from rule_agent import RuleAgent


class PrunedRuleAgent(RuleAgent):
    """RuleAgent with configurable action pruning."""

    def __init__(self, *args,
                 prune_deadline_ms: float = 80.0,
                 prune_max_hops: int = 3,
                 prune_server_util: float = 0.9,
                 prune_bw_threshold: int = 6,
                 prune_min_p_success: float = 0.5,
                 track_pruning: bool = True,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.prune_deadline_ms = prune_deadline_ms
        self.prune_max_hops = prune_max_hops
        self.prune_server_util = prune_server_util
        self.prune_bw_threshold = prune_bw_threshold
        self.prune_min_p_success = prune_min_p_success
        self.track_pruning = track_pruning
        self.prune_stats = {
            "total_candidates": 0,
            "pruned_deadline": 0,
            "pruned_overload": 0,
            "pruned_bw_risk": 0,
            "evaluated": 0,
        }

    def reset_prune_stats(self):
        self.prune_stats = {
            "total_candidates": 0,
            "pruned_deadline": 0,
            "pruned_overload": 0,
            "pruned_bw_risk": 0,
            "evaluated": 0,
        }

    def _estimate_hops(self, src: int, dst: int) -> int:
        """Coarse hop estimate using node ID difference (fallback)."""
        # In a real system this would use the path cache, but we don't have
        # direct access to KSPMapper here. Use a simple heuristic.
        # Actually, we can use the encoder's path_cache if available.
        if hasattr(self.encoder, '_path_cache'):
            paths = self.encoder._path_cache.get((src, dst), [])
            if paths and len(paths[0]) > 1:
                return len(paths[0]) - 1
        return abs(src - dst)  # very coarse fallback

    def decide(self, request: DNNRequest, network_state) -> Tuple[int, int, float, Dict]:
        model: DNNModel = request.model
        src = request.source_node
        best_score = -float('inf')
        best_action = None
        candidates = []

        # For mis-pruning detection: evaluate ALL candidates first
        all_evaluated = []

        with torch.no_grad():
            for split in model.splits:
                split_id = split.split_id
                bw_slots = split.bandwidth_slots

                for srv in self.mec.servers:
                    dst = srv.node_id
                    if dst == src:
                        continue

                    # --- Pruning checks ---
                    self.prune_stats["total_candidates"] += 1
                    pruned = False
                    prune_reason = None

                    # Rule 1: deadline urgency
                    if request.deadline_ms < self.prune_deadline_ms:
                        hops = self._estimate_hops(src, dst)
                        if hops > self.prune_max_hops:
                            self.prune_stats["pruned_deadline"] += 1
                            pruned = True
                            prune_reason = "deadline"

                    # Rule 2: server overload
                    if not pruned and srv.utilization > self.prune_server_util:
                        self.prune_stats["pruned_overload"] += 1
                        pruned = True
                        prune_reason = "overload"

                    # Encode and predict
                    z = self.encoder.encode(src, dst)
                    z_t = torch.from_numpy(z).unsqueeze(0).float().to(self.device)
                    partition = torch.tensor([split_id], dtype=torch.long, device=self.device)
                    src_t = torch.tensor([src], dtype=torch.long, device=self.device)
                    dst_t = torch.tensor([dst], dtype=torch.long, device=self.device)

                    success_logit, delay_pred = self.predictor(partition, src_t, dst_t, z_t)
                    p_success = torch.sigmoid(success_logit).item()
                    delay_pred_val = delay_pred.item()

                    # Rule 3: bandwidth risk
                    if not pruned and bw_slots >= self.prune_bw_threshold:
                        if p_success < self.prune_min_p_success:
                            self.prune_stats["pruned_bw_risk"] += 1
                            pruned = True
                            prune_reason = "bw_risk"

                    compute_delay = srv.compute_delay_ms(split.compute_cost, split.intermediate_size_mb)
                    total_delay_est = delay_pred_val + compute_delay
                    load_penalty = srv.utilization
                    deadline_slack = request.deadline_ms - total_delay_est
                    deadline_penalty = max(0.0, 1.0 - deadline_slack / request.deadline_ms)

                    score = (self.alpha * p_success
                             - self.beta * (total_delay_est / 100.0)
                             - self.gamma * load_penalty
                             - self.delta * deadline_penalty)

                    cand = {
                        "split_id": split_id,
                        "server_id": srv.server_id,
                        "node_id": dst,
                        "p_success": p_success,
                        "delay_total": total_delay_est,
                        "server_util": srv.utilization,
                        "score": score,
                        "pruned": pruned,
                        "prune_reason": prune_reason,
                    }
                    all_evaluated.append(cand)

                    if not pruned:
                        self.prune_stats["evaluated"] += 1
                        if score > best_score:
                            best_score = score
                            best_action = (split_id, srv.server_id)

        # Fallback: if all pruned, use best from all (ignore pruning)
        if best_action is None and all_evaluated:
            best_cand = max(all_evaluated, key=lambda x: x["score"])
            best_action = (best_cand["split_id"], best_cand["server_id"])
            best_score = best_cand["score"]

        if best_action is None:
            best_action = (0, 0)
            best_score = 0.0

        # Mis-pruning analysis: among pruned candidates, how many had higher score than chosen?
        chosen = next(c for c in all_evaluated
                      if c["split_id"] == best_action[0] and c["server_id"] == best_action[1])
        mispruned = [c for c in all_evaluated if c["pruned"] and c["score"] > chosen["score"]]

        info = {
            "candidates": all_evaluated,
            "num_total": len(all_evaluated),
            "num_pruned": sum(1 for c in all_evaluated if c["pruned"]),
            "num_evaluated": sum(1 for c in all_evaluated if not c["pruned"]),
            "mispruned_count": len(mispruned),
            "mispruned_max_score_diff": max((c["score"] - chosen["score"] for c in mispruned), default=0.0),
            "prune_stats": dict(self.prune_stats),
        }
        return best_action[0], best_action[1], best_score, info
