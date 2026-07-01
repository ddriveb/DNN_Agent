"""CorrectionNet Agent — aligned with TopKSelectorAgent baseline.

Design principle:
  λ=0.0 must be exactly equivalent to TopKSelectorAgent.
  Only the correction term differs.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch

from eval_imitation import build_state_vector_for_imitation
from enhance_state_and_retrain import build_enhanced_state
from topk_selector_agent import TopKSelectorAgent


class TopK2CorrectionAgent:
    """TopK2 selector with CorrectionNet residual correction.

    When lambda_corr=0.0, this is EXACTLY equivalent to TopKSelectorAgent
    with the same (alpha, beta, gamma, delta, top_k, use_enhanced_state).
    """

    def __init__(
        self,
        correction_ckpt: str,
        predictor,
        encoder,
        mec_cluster,
        num_servers: int = 5,
        top_k: int = 2,
        lambda_corr: float = 0.0,
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
        use_enhanced_state: bool = True,
        device: str = "cpu",
    ):
        self.predictor = predictor
        self.encoder = encoder
        self.mec = mec_cluster
        self.num_servers = num_servers
        self.top_k = top_k
        self.lambda_corr = lambda_corr
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

        # Load CorrectionNet
        from correction_net import CorrectionNet
        ckpt = torch.load(correction_ckpt, map_location=device, weights_only=False)
        self.correction_net = CorrectionNet(
            ckpt.get("input_dim", 10),
            hidden_dims=(64, 64),
        ).to(device)
        self.correction_net.load_state_dict(ckpt["model_state"])
        self.correction_net.eval()
        self.target_mean = ckpt.get("target_mean", 0.0)
        self.target_std = ckpt.get("target_std", 1.0)
        self.input_mean = np.array(ckpt.get("input_mean", [0.0]*ckpt.get("input_dim", 10)), dtype=np.float32)
        self.input_std = np.array(ckpt.get("input_std", [1.0]*ckpt.get("input_dim", 10)), dtype=np.float32)
        self.input_dim = int(ckpt.get("input_dim", len(self.input_mean)))

        # Load imitation model for top-k (same as TopKSelectorAgent)
        from train_imitation import ImitationAgent
        imitation_ckpt_path = Path(__file__).parent / "checkpoints" / "imitation_agent_enhanced.pt"
        imitation_ckpt = torch.load(imitation_ckpt_path, map_location=device, weights_only=False)
        self.imitation_model = ImitationAgent(
            imitation_ckpt["state_dim"],
            imitation_ckpt["num_actions"],
            hidden_dims=imitation_ckpt.get("hidden_dims", (128, 128)),
            dropout=0.0,
        )
        self.imitation_model.load_state_dict(imitation_ckpt["model_state"])
        self.imitation_model.eval()
        self.imitation_model.to(device)

    def _score_candidate(self, request, split_id, server_id, env):
        """EXACT copy of TopKSelectorAgent._score_candidate."""
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

    def _compute_correction(self, request, split_id, server_id, score_result, global_state):
        """Compute CorrectionNet correction for a single candidate."""
        from correction_net import compose_correction_input

        split = request.model.splits[split_id]
        deadline_slack_norm = (request.deadline_ms - score_result["delay_total"]) / max(request.deadline_ms, 1.0)
        if not np.isfinite(deadline_slack_norm):
            deadline_slack_norm = -1.0

        preview = score_result.get("preview", {})
        path_before = preview.get("path_stats_before") or {}
        path_total_free = float(path_before.get("total_free_slots", 0.0))
        path_utilization = 1.0 - (path_total_free / max(float(getattr(self.encoder.net, "num_slots", 1)), 1.0))

        if self.input_dim <= 10:
            action_feat = np.array([
                split.bandwidth_slots / 8.0,
                split.compute_cost,
                split.intermediate_size_mb / 3.0,
                score_result["p_success"],
                score_result["delay_total"] / 200.0,
                score_result["server_util"],
                deadline_slack_norm,
            ], dtype=np.float32)
        else:
            action_feat = np.array([
                split.bandwidth_slots / 8.0,
                split.compute_cost,
                split.intermediate_size_mb / 3.0,
                score_result["p_success"],
                score_result["delay_total"] / 200.0,
                score_result["server_util"],
                deadline_slack_norm,
                max(0.0, float(score_result.get("frag_penalty", 0.0))),
                max(0.0, float(score_result.get("lfb_penalty", 0.0))),
                max(0.0, float(score_result.get("risk_penalty", 0.0))),
                float(path_before.get("largest_free_block_ratio", 0.0)),
                float(np.clip(path_utilization, 0.0, 1.0)),
                1.0 if score_result.get("mapper_feasible", False) else 0.0,
            ], dtype=np.float32)

        feat = compose_correction_input(action_feat, global_state)
        feat = (feat - self.input_mean) / self.input_std
        feat_t = torch.from_numpy(feat).float().unsqueeze(0).to(self.device)

        with torch.no_grad():
            correction_norm = self.correction_net(feat_t).item()
            correction = correction_norm * self.target_std + self.target_mean

        return correction

    def decide(self, request, env):
        """EXACT TopKSelectorAgent decide logic, plus correction term."""
        # 1. Build state vector (same as TopKSelectorAgent)
        state = build_state_vector_for_imitation(request, env, self.encoder, self.num_servers)
        if self.use_enhanced_state:
            state = build_enhanced_state(state, request.model_name)
        state_t = torch.from_numpy(state).float().unsqueeze(0).to(self.device)

        # 2. Get top-K actions from imitation model (same as TopKSelectorAgent)
        with torch.no_grad():
            logits = self.imitation_model(state_t)
            top_k_values, top_k_indices = torch.topk(logits, k=min(self.top_k, logits.shape[1]), dim=1)

        # 3. Re-rank with predictor score + correction
        best_action = None
        best_score = -float('inf')
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

            # Add correction term
            if self.lambda_corr != 0.0:
                correction = self._compute_correction(request, split_id, server_id, result, state)
                final_score = result["score"] + self.lambda_corr * correction
            else:
                final_score = result["score"]
                correction = 0.0

            evaluated.append({
                "action_id": action_id,
                "split_id": split_id,
                "server_id": server_id,
                "pred_score": result["score"],
                "p_success": result["p_success"],
                "delay_total": result["delay_total"],
                "mapper_feasible": result["mapper_feasible"],
                "frag_penalty": result["frag_penalty"],
                "lfb_penalty": result["lfb_penalty"],
                "risk_penalty": result["risk_penalty"],
                "secondary_cost": result["secondary_cost"],
                "correction": correction,
                "final_score": final_score,
            })

        chosen = self._pick_best_evaluated(evaluated)
        if chosen is not None:
            best_score = chosen["final_score"]
            best_action = (chosen["split_id"], chosen["server_id"])

        # Fallback: EXACT same as TopKSelectorAgent
        if best_action is None and len(top_k_indices[0]) > 0:
            action_id = int(top_k_indices[0][0].item())
            split_id = action_id // self.num_servers
            server_id = action_id % self.num_servers
            best_action = (split_id, server_id)
            best_score = 0.0

        if best_action is None:
            best_action = (0, 0)
            best_score = 0.0

        return best_action[0], best_action[1], best_score, {
            "type": "topk2_correction",
            "lambda": self.lambda_corr,
            "evaluated": evaluated,
        }
