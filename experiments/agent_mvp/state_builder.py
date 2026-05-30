"""Action-aware state builder for replay collection and correction training.

Builds a rich state vector that includes:
  - Global/request features (30D)
  - Per-action candidate features (num_actions x feat_dim)
  - Action validity mask
  - Top-K mask from imitation agent

This version extends the original 7D action feature with fragmentation-aware
preview statistics so CorrectionNet can learn when fragmentation matters.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch

from correction_net import ACTION_FEATURE_DIM_V2
from eval_imitation import build_state_vector_for_imitation
from enhance_state_and_retrain import build_enhanced_state
from train_imitation import ImitationAgent


class ActionAwareStateBuilder:
    """Builds state vectors with per-action preview features."""

    def __init__(
        self,
        encoder,
        predictor,
        mec_cluster,
        imitation_checkpoint_path: str = None,
        num_servers: int = 5,
        use_enhanced_state: bool = True,
        load_imitation_mask: bool = True,
        use_predictor_action_features: bool = True,
        drop_predictor_action_features: bool = False,
        device: str = "cpu",
    ):
        self.encoder = encoder
        self.predictor = predictor
        self.mec = mec_cluster
        self.num_servers = num_servers
        self.use_enhanced_state = use_enhanced_state
        self.load_imitation_mask = load_imitation_mask
        self.use_predictor_action_features = use_predictor_action_features
        self.drop_predictor_action_features = drop_predictor_action_features
        self.device = device

        self.imitation_model = None
        if (
            self.load_imitation_mask
            and imitation_checkpoint_path is not None
            and Path(imitation_checkpoint_path).exists()
        ):
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

    def _build_action_feature(self, request, env, split_id, server_id):
        """Build 13D per-action feature with fragmentation-aware preview."""
        src = request.source_node
        srv = self.mec.get_server_by_id(server_id)
        if srv is None or srv.node_id == src:
            return np.zeros(ACTION_FEATURE_DIM_V2, dtype=np.float32), False

        dst = srv.node_id
        split = request.model.splits[split_id]
        z = self.encoder.encode(src, dst)
        if self.use_predictor_action_features and self.predictor is not None:
            z_t = torch.from_numpy(z).unsqueeze(0).float().to(self.device)

            partition = torch.tensor([split_id], dtype=torch.long, device=self.device)
            src_t = torch.tensor([src], dtype=torch.long, device=self.device)
            dst_t = torch.tensor([dst], dtype=torch.long, device=self.device)

            with torch.no_grad():
                success_logit, delay_pred = self.predictor(partition, src_t, dst_t, z_t)
                p_success = float(torch.sigmoid(success_logit).item())
                delay_pred_val = float(delay_pred.item())

            delay_pred_val = float(np.clip(delay_pred_val, 0.0, 500.0))
            p_success = float(np.clip(p_success, 0.0, 1.0))
        else:
            p_success = 0.0
            delay_pred_val = 0.0

        compute_delay = srv.compute_delay_ms(split.compute_cost, split.intermediate_size_mb)
        total_delay = delay_pred_val + compute_delay
        if not np.isfinite(total_delay):
            total_delay = 200.0

        deadline_slack = request.deadline_ms - total_delay
        deadline_slack_norm = deadline_slack / max(request.deadline_ms, 1.0)
        if not np.isfinite(deadline_slack_norm):
            deadline_slack_norm = -1.0

        server_load = float(srv.utilization)
        if not np.isfinite(server_load):
            server_load = 0.5

        preview = env.mapper.preview(src, dst, split.bandwidth_slots)
        path_before = preview.get("path_stats_before") or {}
        path_total_free = float(path_before.get("total_free_slots", 0.0))
        path_utilization = 1.0 - (path_total_free / max(float(env.net.num_slots), 1.0))

        feat = np.array([
            split.bandwidth_slots / 8.0,
            split.compute_cost,
            split.intermediate_size_mb / 3.0,
            p_success,
            total_delay / 200.0,
            server_load,
            deadline_slack_norm,
            max(0.0, float(preview.get("delta_path_frag", 0.0))),
            max(0.0, float(preview.get("delta_path_lfb_ratio", 0.0))),
            max(0.0, float(preview.get("future_blocking_risk", 0.0))),
            float(path_before.get("largest_free_block_ratio", 0.0)),
            float(np.clip(path_utilization, 0.0, 1.0)),
            1.0 if preview.get("feasible", False) else 0.0,
        ], dtype=np.float32)
        if self.drop_predictor_action_features:
            # Remove the most direct predictor-driven action priors while
            # keeping the feature dimensionality fixed for DQN ablations.
            feat[3] = 0.0  # p_success
            feat[4] = 0.0  # delay_norm
            feat[6] = 0.0  # deadline_slack_norm (depends on predicted delay)
        return feat, True

    def build(self, request, env, top_k: int = 2):
        """Build full action-aware state."""
        global_state = build_state_vector_for_imitation(request, env, self.encoder, self.num_servers)
        if self.use_enhanced_state:
            global_state = build_enhanced_state(global_state, request.model_name)

        src = request.source_node
        num_splits = len(request.model.splits)
        num_actions = num_splits * self.num_servers

        action_features = []
        valid_mask = np.zeros(num_actions, dtype=bool)

        for split_id in range(num_splits):
            for server_id in range(self.num_servers):
                action_id = split_id * self.num_servers + server_id
                feat, valid = self._build_action_feature(request, env, split_id, server_id)
                action_features.append(feat)
                valid_mask[action_id] = valid

        action_features = np.stack(action_features)
        flat_state = np.concatenate([global_state, action_features.flatten()]).astype(np.float32)

        topk_mask = np.zeros(num_actions, dtype=bool)
        if self.imitation_model is not None:
            state_t = torch.from_numpy(global_state).float().unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.imitation_model(state_t)
                top_k_actions = torch.topk(logits, k=min(top_k, num_actions), dim=1)[1]
                for action_id in top_k_actions[0].cpu().tolist():
                    if 0 <= action_id < num_actions:
                        topk_mask[action_id] = True

        return {
            "global_state": global_state,
            "action_features": action_features,
            "flat_state": flat_state,
            "valid_mask": valid_mask,
            "topk_mask": topk_mask,
            "num_actions": num_actions,
            "action_feat_dim": ACTION_FEATURE_DIM_V2,
            "source_node": src,
        }

    @property
    def flat_state_dim(self):
        return 30 + 15 * ACTION_FEATURE_DIM_V2
