"""Top-K Selector Agent.

Neural proposal (ImitationAgent outputs top-K actions)
+ Predictor-guided re-ranking (RuleAgent score on each candidate)
= Final action selection.

This leverages the 96.3% top-3 hit rate while using the predictor
for precise final selection.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch

from predictor import Predictor
from eval_imitation import build_state_vector_for_imitation
from enhance_state_and_retrain import build_enhanced_state


class TopKSelectorAgent:
    """Hybrid agent: neural top-K proposal + predictor re-ranking."""

    def __init__(
        self,
        imitation_checkpoint_path: str,
        predictor,
        encoder,
        mec_cluster,
        num_servers: int = 5,
        top_k: int = 3,
        alpha: float = 2.0,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.05,
        frag_weight: float = 0.0,
        lfb_weight: float = 0.0,
        risk_weight: float = 0.0,
        enforce_mapper_feasibility: bool = False,
        use_fragmentation_tiebreaker: bool = False,
        p_success_min: float = 0.0,
        p_success_tie_epsilon: float = 0.0,
        use_enhanced_state: bool = False,
        device: str = "cpu",
    ):
        self.predictor = predictor
        self.encoder = encoder
        self.mec = mec_cluster
        self.num_servers = num_servers
        self.top_k = top_k
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.frag_weight = frag_weight
        self.lfb_weight = lfb_weight
        self.risk_weight = risk_weight
        self.enforce_mapper_feasibility = enforce_mapper_feasibility
        self.use_fragmentation_tiebreaker = use_fragmentation_tiebreaker
        self.p_success_min = p_success_min
        self.p_success_tie_epsilon = p_success_tie_epsilon
        self.use_enhanced_state = use_enhanced_state
        self.device = device

        # Load imitation agent
        from train_imitation import ImitationAgent
        ckpt = torch.load(imitation_checkpoint_path, map_location=device, weights_only=False)
        self.imitation_model = ImitationAgent(
            ckpt["state_dim"],
            ckpt["num_actions"],
            hidden_dims=ckpt.get("hidden_dims", (128, 128)),
            dropout=0.0,
        )
        self.imitation_model.load_state_dict(ckpt["model_state"])
        self.imitation_model.eval()
        self.imitation_model.to(device)

    def _score_candidate(self, request, split_id, server_id, env):
        """Compute RuleAgent-style score for a (split, server) candidate."""
        src = request.source_node
        srv = self.mec.get_server_by_id(server_id)
        if srv is None or srv.node_id == src:
            return None

        dst = srv.node_id
        z = self.encoder.encode(src, dst)
        z_t = torch.from_numpy(z).unsqueeze(0).float().to(self.device)

        partition = torch.tensor([split_id], dtype=torch.long, device=self.device)
        src_t = torch.tensor([src], dtype=torch.long, device=self.device)
        dst_t = torch.tensor([dst], dtype=torch.long, device=self.device)

        with torch.no_grad():
            success_logit, delay_pred = self.predictor(partition, src_t, dst_t, z_t)
            p_success = torch.sigmoid(success_logit).item()
            delay_pred_val = delay_pred.item()

        split = request.model.splits[split_id]
        compute_delay = srv.compute_delay_ms(split.compute_cost, split.intermediate_size_mb)
        total_delay_est = delay_pred_val + compute_delay
        load_penalty = srv.utilization
        deadline_slack = request.deadline_ms - total_delay_est
        deadline_penalty = max(0.0, 1.0 - deadline_slack / request.deadline_ms)
        preview = env.mapper.preview(src, dst, split.bandwidth_slots)

        if self.enforce_mapper_feasibility and not preview["feasible"]:
            return {
                "score": -1e9,
                "base_score": -1e9,
                "p_success": p_success,
                "delay_total": total_delay_est,
                "server_util": srv.utilization,
                "mapper_feasible": False,
                "frag_penalty": 0.0,
                "lfb_penalty": 0.0,
                "risk_penalty": 1.0,
                "secondary_cost": 1.0,
                "preview": preview,
            }

        score = (
            self.alpha * p_success
            - self.beta * (total_delay_est / 100.0)
            - self.gamma * load_penalty
            - self.delta * deadline_penalty
        )
        frag_penalty = max(0.0, preview["delta_path_frag"])
        lfb_penalty = max(0.0, preview["delta_path_lfb_ratio"])
        risk_penalty = max(0.0, preview["future_blocking_risk"])
        secondary_cost = (
            self.frag_weight * frag_penalty
            + self.lfb_weight * lfb_penalty
            + self.risk_weight * risk_penalty
        )
        if not self.use_fragmentation_tiebreaker:
            score -= secondary_cost

        return {
            "score": score,
            "base_score": score,
            "p_success": p_success,
            "delay_total": total_delay_est,
            "server_util": srv.utilization,
            "mapper_feasible": preview["feasible"],
            "frag_penalty": frag_penalty,
            "lfb_penalty": lfb_penalty,
            "risk_penalty": risk_penalty,
            "secondary_cost": secondary_cost,
            "preview": preview,
        }

    def _pick_best_evaluated(self, evaluated):
        """Choose best candidate, optionally using fragmentation as a secondary tie-break."""
        if not evaluated:
            return None

        candidate_pool = evaluated
        if self.enforce_mapper_feasibility:
            feasible_only = [item for item in evaluated if item["mapper_feasible"]]
            if feasible_only:
                candidate_pool = feasible_only

        if not self.use_fragmentation_tiebreaker:
            return max(candidate_pool, key=lambda item: item["final_score"])

        best_p_success = max(item["p_success"] for item in candidate_pool)
        feasible_band = [
            item for item in candidate_pool
            if item["p_success"] >= self.p_success_min
            and item["p_success"] >= best_p_success - self.p_success_tie_epsilon
        ]
        if not feasible_band:
            return max(candidate_pool, key=lambda item: item["final_score"])

        return min(
            feasible_band,
            key=lambda item: (
                item["secondary_cost"],
                -item["final_score"],
                item["delay_total"],
            ),
        )

    def decide(self, request, env):
        """Select best (split_id, server_id) via top-K neural + predictor re-rank."""
        # 1. Build state vector
        state = build_state_vector_for_imitation(request, env, self.encoder, self.num_servers)
        if self.use_enhanced_state:
            state = build_enhanced_state(state, request.model_name)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # 2. Get top-K actions from imitation model
        with torch.no_grad():
            logits = self.imitation_model(state_t)
            top_k_values, top_k_indices = torch.topk(logits, k=min(self.top_k, logits.shape[1]), dim=1)

        # 3. Re-rank with predictor score
        best_action = None
        best_score = -float('inf')
        best_details = None
        evaluated = []

        for action_id in top_k_indices[0].cpu().tolist():
            split_id = action_id // self.num_servers
            server_id = action_id % self.num_servers

            if split_id >= len(request.model.splits):
                continue
            if server_id >= len(self.mec.servers):
                continue

            result = self._score_candidate(request, split_id, server_id, env)
            if result is None:
                continue

            evaluated.append({
                "action_id": action_id,
                "split_id": split_id,
                "server_id": server_id,
                "score": result["score"],
                "final_score": result["score"],
                "p_success": result["p_success"],
                "delay_total": result["delay_total"],
                "server_util": result["server_util"],
                "mapper_feasible": result["mapper_feasible"],
                "frag_penalty": result["frag_penalty"],
                "lfb_penalty": result["lfb_penalty"],
                "risk_penalty": result["risk_penalty"],
                "secondary_cost": result["secondary_cost"],
            })

        chosen = self._pick_best_evaluated(evaluated)
        if chosen is not None:
            best_score = chosen["final_score"]
            best_action = (chosen["split_id"], chosen["server_id"])
            best_details = chosen

        # Fallback: if no valid candidate in top-K, use top-1 action
        if best_action is None and len(top_k_indices[0]) > 0:
            action_id = int(top_k_indices[0][0].item())
            split_id = action_id // self.num_servers
            server_id = action_id % self.num_servers
            best_action = (split_id, server_id)
            best_score = 0.0
            best_details = {}

        if best_action is None:
            # Ultimate fallback
            best_action = (0, 0)
            best_score = 0.0
            best_details = {}

        return best_action[0], best_action[1], best_score, {
            "type": "topk_selector",
            "top_k": self.top_k,
            "evaluated": evaluated,
            "best_details": best_details,
        }
