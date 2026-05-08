"""
PINN-Predictor: Physics-Informed Neural Network for RMSA feasibility prediction.

Design:
  - External interface (forward) unchanged: returns (success_logit, delay_pred)
  - Internal Head 2 (physics auxiliary) only active during training
  - Three physics losses:
    1. Overlap: penalize occupancy on already-used slots
    2. Continuity: TV norm encouraging contiguous blocks
    3. Sparsity: total occupied slots must equal bandwidth demand
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class PINNPredictor(nn.Module):
    """
    PINN-enhanced predictor.
    Forward (inference): same as base Predictor
    Forward_train (training): returns dict with physics_loss
    """
    def __init__(self, state_dim, num_links, num_slots,
                 num_paths=3, max_servers=64, hidden_dim=64,
                 lambda_overlap=10.0, lambda_tv=1.0, lambda_sparsity=5.0):
        super().__init__()
        self.num_links = num_links
        self.num_slots = num_slots
        self.lambda_overlap = lambda_overlap
        self.lambda_tv = lambda_tv
        self.lambda_sparsity = lambda_sparsity

        # Action embeddings (same as base Predictor)
        self.path_embed = nn.Embedding(num_paths, 8)
        self.bw_proj = nn.Linear(1, 8)
        self.src_embed = nn.Embedding(max_servers, 16)
        self.dst_embed = nn.Embedding(max_servers, 16)

        # Shared encoder
        action_dim = 8 + 8 + 16 + 16
        self.encoder = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
        )

        # Head 1: scalar outputs (original)
        self.head_p = nn.Linear(hidden_dim, 1)   # success logit
        self.head_d = nn.Linear(hidden_dim, 1)   # delay

        # Head 2: physics auxiliary (slot occupancy probability)
        self.head_x = nn.Linear(hidden_dim, num_slots)

    def _embed_actions(self, path_id, src, dst, bw):
        """Consistent action embedding logic."""
        p_emb = self.path_embed(path_id)
        if bw.dim() == 1:
            bw = bw.unsqueeze(-1)
        bw_emb = self.bw_proj(bw.float())
        s_emb = self.src_embed(src)
        d_emb = self.dst_embed(dst)
        return torch.cat([p_emb, bw_emb, s_emb, d_emb], dim=-1)

    def forward(self, path_id, src, dst, bw, state):
        """
        Standard inference interface. Returns (success_logit, delay_pred).
        Compatible with base Predictor.
        """
        actions = self._embed_actions(path_id, src, dst, bw)
        h = self.encoder(torch.cat([actions, state], dim=-1))
        success_logit = self.head_p(h).squeeze(-1)
        delay_pred = self.head_d(h).squeeze(-1)
        return success_logit, delay_pred

    def forward_train(self, path_id, src, dst, bw, state, S_current, path_link_mask, start_slot=None):
        """
        Training interface with physics loss + X_hat supervision.

        Args:
            path_id, src, dst, bw, state: same as forward
            S_current: (num_links, num_slots) current spectrum occupancy [0/1]
            path_link_mask: (B, num_links) binary mask for links in candidate path
            start_slot: (B,) actual start slot from Mapper (-1 if blocked)
        Returns:
            dict with keys: success_logit, delay_pred, physics_loss, loss_xhat, X_hat
        """
        actions = self._embed_actions(path_id, src, dst, bw)
        h = self.encoder(torch.cat([actions, state], dim=-1))

        # Head 1
        success_logit = self.head_p(h).squeeze(-1)
        delay_pred = self.head_d(h).squeeze(-1)

        # Head 2: physics auxiliary
        X_logits = self.head_x(h)                 # (B, num_slots)
        X_hat = torch.sigmoid(X_logits)           # (B, num_slots)
        X_hat = X_hat.clamp(1e-6, 1.0 - 1e-6)

        # Head 2 data supervision: X_hat should match actual First-Fit allocation
        loss_xhat = torch.tensor(0.0, device=h.device)
        B = h.size(0)
        if start_slot is not None and (start_slot >= 0).any():
            X_target = torch.zeros_like(X_hat)
            bw_slots_raw = bw.squeeze(-1) if bw.dim() == 2 else bw
            for b in range(B):
                if start_slot[b] >= 0:
                    s = int(start_slot[b].item())
                    bw_val = int(bw_slots_raw[b].item())
                    if s + bw_val <= X_target.size(1):
                        X_target[b, s:s + bw_val] = 1.0
            loss_xhat = F.binary_cross_entropy(X_hat, X_target)

        # Compute physics loss
        physics_loss = torch.tensor(0.0, device=h.device)
        if self.training and S_current is not None and path_link_mask is not None:
            # 1. Overlap loss: path_link_mask [B, num_links] @ S_current [num_links, F]
            S_path = torch.matmul(path_link_mask.float(), S_current.float())  # (B, F)
            # FIXED: use conflict_mask instead of additive formulation
            conflict_mask = (S_path > 0).float()  # (B, F)
            overlap = (conflict_mask * X_hat).pow(2).sum()

            # 2. Continuity loss: TV norm along frequency axis
            tv = (X_hat[:, 1:] - X_hat[:, :-1]).pow(2).sum()

            # 3. Sparsity loss: total occupied slots should equal bw
            bw_slots = bw.squeeze(-1) if bw.dim() == 2 else bw
            sparsity = (X_hat.sum(dim=1) - bw_slots).pow(2).sum()

            physics_loss = (
                self.lambda_overlap * overlap +
                self.lambda_tv * tv +
                self.lambda_sparsity * sparsity
            )

        return {
            'success_logit': success_logit,
            'delay_pred': delay_pred,
            'physics_loss': physics_loss,
            'loss_xhat': loss_xhat,
            'X_hat': X_hat,
        }


class GradNormPINNPredictor(PINNPredictor):
    """
    PINN-Predictor with GradNorm adaptive loss weighting.
    Reference: Chen et al. "GradNorm: Gradient Normalization for Adaptive Loss Balancing"
    """
    def __init__(self, state_dim, num_links, num_slots,
                 num_paths=3, max_servers=64, hidden_dim=64,
                 lambda_overlap=10.0, lambda_tv=1.0, lambda_sparsity=5.0,
                 alpha=0.16):
        super().__init__(state_dim, num_links, num_slots,
                         num_paths, max_servers, hidden_dim,
                         lambda_overlap, lambda_tv, lambda_sparsity)
        self.alpha = alpha
        # Learnable weights for GradNorm
        self.w_data = nn.Parameter(torch.ones(1))
        self.w_physics = nn.Parameter(torch.ones(1))
        # Initial reference losses (updated online)
        self.register_buffer('l0_data', torch.ones(1))
        self.register_buffer('l0_physics', torch.ones(1))
        self.gradnorm_step = 0

    def forward_train_gradnorm(self, path_id, src, dst, bw, state,
                               S_current, path_link_mask,
                               y_success, y_delay, start_slot=None):
        """
        Training with GradNorm adaptive weighting.
        Returns dict with losses and adaptive weights.
        """
        out = self.forward_train(path_id, src, dst, bw, state,
                                 S_current, path_link_mask, start_slot)

        # Data loss (includes X_hat supervision)
        bce = F.binary_cross_entropy_with_logits(out['success_logit'], y_success)
        mask = y_success > 0.5
        mse_delay = F.mse_loss(out['delay_pred'][mask], y_delay[mask]) if mask.sum() > 0 else 0.0
        loss_data = bce + mse_delay + out['loss_xhat']

        # Physics loss
        loss_physics = out['physics_loss']

        # GradNorm adaptive weighting
        if self.gradnorm_step == 0:
            # Initialize reference losses
            self.l0_data = loss_data.detach().clone()
            self.l0_physics = loss_physics.detach().clone()

        # Weighted losses
        weighted_data = self.w_data.pow(2) * loss_data
        weighted_physics = self.w_physics.pow(2) * loss_physics

        # Total loss
        total_loss = weighted_data + weighted_physics

        self.gradnorm_step += 1

        return {
            'success_logit': out['success_logit'],
            'delay_pred': out['delay_pred'],
            'loss_data': loss_data,
            'loss_physics': loss_physics,
            'total_loss': total_loss,
            'w_data': self.w_data.item(),
            'w_physics': self.w_physics.item(),
        }
