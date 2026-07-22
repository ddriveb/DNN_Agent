"""Masked PPO actors for Agent-C and Agent-R.

These actors keep the existing variable-length action feature design but use a
categorical policy over masked valid actions instead of DQN argmax scores.
"""
from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from sa_hmarl.agents.action_feature_builders import (
    AgentCFeatureBuilder,
    AgentRFeatureBuilder,
)
from sa_hmarl.network.modulation import ModulationRegistry


ACTIVATION_MAP = {
    "tanh": nn.Tanh,
    "relu": nn.ReLU,
    "silu": nn.SiLU,
    "gelu": nn.GELU,
    "leaky_relu": lambda: nn.LeakyReLU(0.01),
}


class MaskedPPOActorNetwork(nn.Module):
    """Shared MLP producing one policy logit per candidate action."""

    def __init__(self, input_dim: int, hidden_dims=(128, 64), activation: str = "tanh"):
        super().__init__()
        act_fn = ACTIVATION_MAP.get(activation, nn.Tanh)
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), act_fn()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_actions, input_dim = x.shape
        return self.net(x.reshape(-1, input_dim)).view(batch_size, num_actions)


class GatedMaskedPPOActorNetwork(nn.Module):
    """Gated shared MLP that learns when to use mean-field features.

    Architecture:
        h_base = base_encoder(x_base)
        h_mf   = mf_encoder(x_mf)
        gate   = sigmoid(gate_net(concat(x_base, x_mf)))   # scalar per action
        h      = h_base + gate * h_mf
        logit  = output_head(h)

    The scalar gate is initialized near 0 (bias = -2.0 => sigmoid(-2) ~ 0.12)
    so the network defaults to base features and must learn to open the gate
    when mean-field information is useful.

    The input tensor is the concatenation [x_base, x_mf] along the last axis,
    matching the existing feature construction in AgentC.build_action_features.
    """

    def __init__(
        self,
        base_dim: int,
        mf_dim: int,
        hidden_dims=(128, 64),
        activation: str = "tanh",
        gate_init_bias: float = -2.0,
    ):
        super().__init__()
        self.base_dim = base_dim
        self.mf_dim = mf_dim
        self.gate_init_bias = gate_init_bias
        act_fn = ACTIVATION_MAP.get(activation, nn.Tanh)

        # Base feature encoder
        base_layers = []
        prev = base_dim
        for h in hidden_dims:
            base_layers.extend([nn.Linear(prev, h), act_fn()])
            prev = h
        self.base_encoder = nn.Sequential(*base_layers)

        # Mean-field feature encoder
        mf_layers = []
        prev = mf_dim
        for h in hidden_dims:
            mf_layers.extend([nn.Linear(prev, h), act_fn()])
            prev = h
        self.mf_encoder = nn.Sequential(*mf_layers)

        # Gate network: scalar gate per action
        self.gate_net = nn.Sequential(
            nn.Linear(base_dim + mf_dim, hidden_dims[0]),
            act_fn(),
            nn.Linear(hidden_dims[0], 1),
            nn.Sigmoid(),
        )
        # Initialize gate near closed (trust base features by default)
        self.gate_net[-2].bias.data.fill_(self.gate_init_bias)

        # Output head shared by both encoders
        self.output_head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_actions, total_dim = x.shape
        x = x.reshape(-1, total_dim)
        x_base = x[..., : self.base_dim]
        x_mf = x[..., self.base_dim :]

        h_base = self.base_encoder(x_base)
        h_mf = self.mf_encoder(x_mf)
        gate = self.gate_net(torch.cat([x_base, x_mf], dim=-1))
        h = h_base + gate * h_mf

        logits = self.output_head(h).view(batch_size, num_actions)
        return logits

    def forward_with_gate(
        self, x: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Return policy logits and per-action scalar gate values.

        Gate shape: (batch_size, num_actions).  Values in [0, 1].
        """
        batch_size, num_actions, total_dim = x.shape
        x = x.reshape(-1, total_dim)
        x_base = x[..., : self.base_dim]
        x_mf = x[..., self.base_dim :]

        h_base = self.base_encoder(x_base)
        h_mf = self.mf_encoder(x_mf)
        gate = self.gate_net(torch.cat([x_base, x_mf], dim=-1))
        h = h_base + gate * h_mf

        logits = self.output_head(h).view(batch_size, num_actions)
        gate = gate.view(batch_size, num_actions)
        return logits, gate


class _MaskedPPOBase:
    """Common masked categorical PPO actor logic."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims=(128, 64),
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        device: str = "cpu",
        activation: str = "tanh",
        policy_net: Optional[nn.Module] = None,
        gate_reg_coef: float = 0.0,
    ):
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.activation = activation
        self.gate_reg_coef = gate_reg_coef
        self.last_gate_loss = 0.0
        if policy_net is not None:
            self.policy_net = policy_net.to(device)
        else:
            self.policy_net = MaskedPPOActorNetwork(input_dim, self.hidden_dims, activation=activation).to(device)
        self.optimizer = torch.optim.Adam(self.policy_net.parameters(), lr=lr)

    def _distribution_from_features(
        self,
        features: np.ndarray,
        mask: np.ndarray,
    ) -> Optional[torch.distributions.Categorical]:
        if len(mask) == 0 or not np.any(mask) or features.size == 0:
            return None
        x = torch.tensor(features, dtype=torch.float32, device=self.device).unsqueeze(0)
        mask_t = torch.tensor(mask, dtype=torch.bool, device=self.device).unsqueeze(0)
        logits = self.policy_net(x).masked_fill(~mask_t, -1e9).squeeze(0)
        return torch.distributions.Categorical(logits=logits)

    def select_from_features(
        self,
        features: np.ndarray,
        mask: np.ndarray,
        deterministic: bool = False,
    ) -> Tuple[Optional[int], float, float]:
        dist = self._distribution_from_features(features, mask)
        if dist is None:
            return None, 0.0, 0.0
        if deterministic:
            action_t = torch.argmax(dist.logits)
        else:
            action_t = dist.sample()
        log_prob = dist.log_prob(action_t)
        entropy = dist.entropy()
        return int(action_t.item()), float(log_prob.item()), float(entropy.item())

    def _pad_features_masks(self, features_list, masks_list):
        """Pad variable-length action features to a uniform batch tensor."""
        max_actions = max(len(m) for m in masks_list)
        padded_features = []
        padded_masks = []
        for features, mask in zip(features_list, masks_list):
            pad_len = max_actions - len(mask)
            if pad_len > 0:
                features = np.concatenate(
                    [features, np.zeros((pad_len, features.shape[1]), dtype=np.float32)],
                    axis=0,
                )
                mask = np.concatenate([mask, np.zeros(pad_len, dtype=bool)], axis=0)
            padded_features.append(features)
            padded_masks.append(mask)

        features_t = torch.tensor(
            np.stack(padded_features), dtype=torch.float32, device=self.device
        )
        masks_t = torch.tensor(np.stack(padded_masks), dtype=torch.bool, device=self.device)
        return features_t, masks_t

    def _log_probs_entropy_batch(self, features_list, masks_list, actions):
        features_t, masks_t = self._pad_features_masks(features_list, masks_list)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)

        logits = self.policy_net(features_t).masked_fill(~masks_t, -1e9)
        dist = torch.distributions.Categorical(logits=logits)
        return dist.log_prob(actions_t), dist.entropy()

    def _gate_binary_loss(self, features_list, masks_list) -> torch.Tensor:
        """Encourage scalar gates to be near 0 or 1, not 0.5.

        Returns 0 for non-gated policies.
        """
        if self.gate_reg_coef <= 0.0:
            return torch.tensor(0.0, device=self.device)
        if not isinstance(self.policy_net, GatedMaskedPPOActorNetwork):
            return torch.tensor(0.0, device=self.device)
        features_t, masks_t = self._pad_features_masks(features_list, masks_list)
        _, gate = self.policy_net.forward_with_gate(features_t)
        gate_valid = gate[masks_t]
        if gate_valid.numel() == 0:
            return torch.tensor(0.0, device=self.device)
        # Maximized at 0.5, minimized at 0 or 1
        return (gate_valid * (1.0 - gate_valid)).mean()

    def optimize_ppo(
        self,
        features_list,
        masks_list,
        actions,
        old_log_probs,
        advantages,
        clip_coef: float,
        epochs: int = 1,
        target_kl: Optional[float] = None,
        reference_policy: Optional[MaskedPPOActorNetwork] = None,
        ref_kl_coef: float = 0.0,
    ) -> Tuple[float, float, float, float]:
        old_log_probs_t = torch.tensor(old_log_probs, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)

        # Pre-compute reference distribution once (frozen, no grad needed)
        ref_dist = None
        features_t_for_kl = None
        masks_t_for_kl = None
        if reference_policy is not None and ref_kl_coef > 0:
            features_t_for_kl, masks_t_for_kl = self._pad_features_masks(
                features_list, masks_list
            )
            with torch.no_grad():
                ref_logits = reference_policy(features_t_for_kl).masked_fill(
                    ~masks_t_for_kl, -1e9
                )
            ref_dist = torch.distributions.Categorical(logits=ref_logits)

        last_policy_loss = 0.0
        last_entropy = 0.0
        last_kl = 0.0
        last_ref_kl = 0.0
        for _ in range(max(1, epochs)):
            new_log_probs, entropy = self._log_probs_entropy_batch(
                features_list, masks_list, actions
            )
            log_ratio = new_log_probs - old_log_probs_t
            ratio = torch.exp(log_ratio)
            with torch.no_grad():
                # Schulman approximation used by PPO implementations.
                approx_kl = ((ratio - 1.0) - log_ratio).mean()
            unclipped = ratio * advantages_t
            clipped = torch.clamp(ratio, 1.0 - clip_coef, 1.0 + clip_coef) * advantages_t
            policy_loss = -torch.min(unclipped, clipped).mean()
            entropy_loss = entropy.mean()
            loss = policy_loss - self.entropy_coef * entropy_loss

            # KL reference regularization (Agent-R only)
            if ref_dist is not None:
                cur_logits = self.policy_net(features_t_for_kl).masked_fill(
                    ~masks_t_for_kl, -1e9
                )
                cur_dist = torch.distributions.Categorical(logits=cur_logits)
                # KL(reference || current) — mode-seeking; penalise forgetting
                ref_kl = torch.distributions.kl_divergence(ref_dist, cur_dist).mean()
                loss = loss + ref_kl_coef * ref_kl
                last_ref_kl = float(ref_kl.item())

            # Gate binary regularization (gated policies only)
            gate_loss = self._gate_binary_loss(features_list, masks_list)
            if self.gate_reg_coef > 0.0:
                loss = loss + self.gate_reg_coef * gate_loss
                self.last_gate_loss = float(gate_loss.item())
            else:
                self.last_gate_loss = 0.0

            self.optimizer.zero_grad()
            loss.backward()
            if self.max_grad_norm is not None and self.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(
                    self.policy_net.parameters(), self.max_grad_norm
                )
            self.optimizer.step()

            last_policy_loss = float(policy_loss.item())
            last_entropy = float(entropy_loss.item())
            last_kl = float(approx_kl.item())
            if target_kl is not None and approx_kl.item() > target_kl:
                break

        return last_policy_loss, last_entropy, last_kl, last_ref_kl


class PPOAgentC(_MaskedPPOBase):
    """PPO actor for high-level split/server selection."""

    def __init__(
        self,
        input_dim: int = 17,
        hidden_dims=(128, 64),
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        device: str = "cpu",
        ablation: bool = False,
        zero_spectrum: bool = False,
        feature_mode: str = "default",
        activation: str = "tanh",
        num_servers: Optional[int] = None,
        gate_reg_coef: float = 0.0,
        fixed_blend_alpha: float = 0.5,
    ):
        if ablation and zero_spectrum:
            raise ValueError("Cannot use both ablation and zero_spectrum")
        if ablation and feature_mode in ("mean_field", "typed_mean_field", "gated_typed_mean_field"):
            raise ValueError(
                "mean_field/typed_mean_field/gated_typed_mean_field feature_modes are not supported with ablation"
            )
        if feature_mode not in (
            "default",
            "enhanced",
            "pressure_aware",
            "overload_aware",
            "cross_pressure",
            "r_feasibility",
            "r_feasibility_safe",
            "r_feasibility_spectrum_impact",
            "r_feasibility_mr_spec_compute",
            "mr_feasibility_rule",
            "mean_field",
            "typed_mean_field",
            "gated_typed_mean_field",
            "fixed_blend_typed_mean_field",
            "candidate_mean_field",
            "candidate_mean_field_count_only",
        ):
            raise ValueError(f"Unknown Agent-C feature_mode: {feature_mode}")
        self.ablation = ablation
        self.zero_spectrum = zero_spectrum
        self.feature_mode = feature_mode
        self.fixed_blend_alpha = float(fixed_blend_alpha)
        actual_dim = 7 if ablation else input_dim
        mf_dim = None
        # Auto-adjust dimensions for feature modes that add extra features.
        # Only add if the caller passed the base dimension (<=17); if they passed
        # a larger value, assume they already computed the total.
        if actual_dim <= 17:
            if feature_mode == "enhanced":
                actual_dim = 17 + 7
            elif feature_mode == "pressure_aware":
                actual_dim = 17 + 8
            elif feature_mode == "overload_aware":
                actual_dim = 17 + 9
            elif feature_mode == "r_feasibility":
                actual_dim = 17 + 10
            elif feature_mode == "r_feasibility_safe":
                actual_dim = 17 + 10 + 2
            elif feature_mode == "r_feasibility_spectrum_impact":
                actual_dim = 17 + 10 + 5
            elif feature_mode == "r_feasibility_mr_spec_compute":
                actual_dim = 17 + 10 + 6 + 5
            elif feature_mode == "mr_feasibility_rule":
                actual_dim = 17 + 10 + 6
            elif feature_mode == "cross_pressure":
                actual_dim = 29  # cross_pressure replaces entire feature vector
            elif feature_mode == "mean_field":
                if num_servers is None:
                    raise ValueError(
                        "mean_field feature_mode requires num_servers to compute input_dim"
                    )
                actual_dim = 17 + num_servers + 2
            elif feature_mode in ("typed_mean_field", "gated_typed_mean_field", "fixed_blend_typed_mean_field"):
                if num_servers is None:
                    raise ValueError(
                        f"{feature_mode} feature_mode requires num_servers to compute input_dim"
                    )
                mf_dim = 3 * (num_servers + 2)
                actual_dim = 17 + mf_dim
            elif feature_mode == "candidate_mean_field":
                actual_dim = 17 + 14
            elif feature_mode == "candidate_mean_field_count_only":
                actual_dim = 17 + 7

        # Gated typed mean-field always uses a dedicated gated network, even when
        # the caller passed the full input_dim explicitly (e.g. from a checkpoint).
        policy_net = None
        if feature_mode == "gated_typed_mean_field":
            if mf_dim is None:
                if actual_dim < 17:
                    raise ValueError(
                        "gated_typed_mean_field requires actual_dim >= 17"
                    )
                mf_dim = actual_dim - 17
            policy_net = GatedMaskedPPOActorNetwork(
                base_dim=17,
                mf_dim=mf_dim,
                hidden_dims=hidden_dims,
                activation=activation,
            )

        super().__init__(
            actual_dim, hidden_dims, lr, entropy_coef, max_grad_norm, device,
            activation=activation, policy_net=policy_net, gate_reg_coef=gate_reg_coef,
        )

    def build_action_features(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        return AgentCFeatureBuilder.build_action_features(self, obs)

    def select_action(self, obs: Dict[str, Any], deterministic: bool = True) -> Optional[int]:
        features, mask = self.build_action_features(obs)
        action, _, _ = self.select_from_features(features, mask, deterministic)
        return action

    def select_action_topk_rerank(
        self,
        obs: Dict[str, Any],
        top_k: int = 3,
        pressure_alpha: float = 2.0,
        w_spectrum: float = 0.35,
        w_server: float = 0.25,
        w_frag: float = 0.15,
        w_delay: float = 0.25,
        deterministic: bool = True,
    ) -> Optional[int]:
        """Select action by reranking top-K policy candidates with pressure scores.

        Args:
            obs: Agent-C observation dict.
            top_k: Number of top policy-probability candidates to consider.
            pressure_alpha: Weight for pressure penalty in reranking.
                Final score = log_prob - alpha * pressure_score.
            w_spectrum, w_server, w_frag, w_delay: Pressure component weights.
            deterministic: If True, top-K by probability; if False, sample then rerank.

        Returns:
            Selected flat action index or None.
        """
        import numpy as np

        features, mask = self.build_action_features(obs)
        dist = self._distribution_from_features(features, mask)
        if dist is None:
            return None

        # Get probabilities for all valid actions
        probs = torch.softmax(dist.logits, dim=-1).detach().cpu().numpy()
        valid_indices = np.where(mask)[0]
        if len(valid_indices) == 0:
            return None

        valid_probs = probs[valid_indices]

        if deterministic:
            # Top-K by probability
            top_k = min(top_k, len(valid_indices))
            topk_local_idx = np.argpartition(-valid_probs, top_k - 1)[:top_k]
            topk_global = valid_indices[topk_local_idx]
        else:
            # Sample up to top_k unique actions
            sampled = set()
            for _ in range(top_k * 10):
                if len(sampled) >= min(top_k, len(valid_indices)):
                    break
                a = int(dist.sample().item())
                sampled.add(a)
            topk_global = np.array(list(sampled), dtype=int)

        num_servers = len(obs["server_utilizations"])
        candidate_features = obs["candidate_features"]

        # Normalize delay across top-k
        max_delay = 1e-6
        for idx in topk_global:
            edge_ms = candidate_features[int(idx)]["edge_compute_ms"]
            if edge_ms != float("inf"):
                max_delay = max(max_delay, edge_ms)

        best_action = None
        best_score = float("-inf")

        for idx in topk_global:
            idx = int(idx)
            feat = candidate_features[idx]
            _, server_id = divmod(idx, num_servers)  # split_id, server_id

            spec = feat.get("spectrum_summary", [])
            if isinstance(spec, np.ndarray):
                spec = spec.tolist()

            lfb_max = float(spec[0]) if len(spec) > 0 else 24.0
            frag_mean = float(spec[4]) if len(spec) > 4 else 0.0
            best_fs = feat.get("best_fs_estimate")
            if best_fs is None:
                best_fs = 24.0

            server_util = obs["server_utilizations"][server_id]
            edge_ms = feat["edge_compute_ms"]
            if edge_ms == float("inf"):
                edge_ms = 1e9

            spectrum_pressure = min(float(best_fs) / max(lfb_max, 1.0), 1.0)
            server_pressure = min(server_util, 1.0)
            frag_pressure = min(frag_mean, 1.0)
            delay_pressure = min(edge_ms / max_delay, 1.0) if max_delay > 0 else 0.0

            pressure_score = (
                w_spectrum * spectrum_pressure
                + w_server * server_pressure
                + w_frag * frag_pressure
                + w_delay * delay_pressure
            )

            # Rerank: policy logit minus pressure penalty
            logit = float(dist.logits[idx].item())
            score = logit - pressure_alpha * pressure_score

            if score > best_score:
                best_score = score
                best_action = idx

        return best_action


class PPOAgentR(_MaskedPPOBase):
    """PPO actor for low-level RMSA selection."""

    def __init__(
        self,
        input_dim: int,
        mod_registry: ModulationRegistry,
        hidden_dims=(128, 64),
        lr: float = 3e-4,
        entropy_coef: float = 0.01,
        max_grad_norm: Optional[float] = 0.5,
        device: str = "cpu",
        feature_mode: str = "default",
    ):
        if feature_mode not in ("default", "frag_aware", "c_aware"):
            raise ValueError(f"Unknown Agent-R feature_mode: {feature_mode}")
        self.mod_registry = mod_registry
        self.feature_mode = feature_mode
        super().__init__(input_dim, hidden_dims, lr, entropy_coef, max_grad_norm, device)

    def build_action_features(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        return AgentRFeatureBuilder.build_action_features(self, obs)

    def select_action(self, obs: Dict[str, Any], deterministic: bool = True) -> Optional[int]:
        features, mask = self.build_action_features(obs)
        action, _, _ = self.select_from_features(features, mask, deterministic)
        return action
