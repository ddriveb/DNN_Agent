"""Online residual RL utilities.

This module implements a minimal online residual-DQN setup over the existing
Top-K candidate pipeline:

  request/env -> ActionAwareStateBuilder -> top-k candidate actions
              -> base predictor score + lambda * residual Q(s, a)
              -> epsilon-greedy exploration

The residual network operates on the same per-action feature view already used
by CorrectionNet, but is now updated online with TD targets and a target net.
"""
from __future__ import annotations

import random
from collections import deque
from typing import Deque, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from correction_net import compose_correction_input
from state_builder import ActionAwareStateBuilder


def serialize_state_dict(state_dict: Dict) -> Dict:
    """Store only the pieces needed for TD learning."""
    return {
        "global_state": np.asarray(state_dict["global_state"], dtype=np.float32).copy(),
        "action_features": np.asarray(state_dict["action_features"], dtype=np.float32).copy(),
        "valid_mask": np.asarray(state_dict["valid_mask"], dtype=bool).copy(),
        "topk_mask": np.asarray(state_dict["topk_mask"], dtype=bool).copy(),
        "num_actions": int(state_dict["num_actions"]),
        "action_feat_dim": int(state_dict["action_feat_dim"]),
    }


def candidate_action_ids(
    state_dict: Dict,
    *,
    p_success_min: float = 0.0,
    require_mapper_feasible: bool = False,
    use_topk: bool = True,
) -> np.ndarray:
    """Return candidate action ids after feasibility-aware masking."""
    valid_mask = np.asarray(state_dict["valid_mask"], dtype=bool)
    topk_mask = np.asarray(state_dict["topk_mask"], dtype=bool)
    candidate_mask = valid_mask & topk_mask if use_topk else valid_mask.copy()
    action_features = np.asarray(state_dict.get("action_features"), dtype=np.float32)

    if action_features.size > 0:
        if p_success_min > 0.0:
            candidate_mask &= action_features[:, 3] >= float(p_success_min)
        if require_mapper_feasible:
            candidate_mask &= action_features[:, 12] >= 0.5

    if np.any(candidate_mask):
        return np.flatnonzero(candidate_mask)
    fallback_mask = valid_mask.copy()
    if action_features.size > 0 and require_mapper_feasible:
        fallback_mask &= action_features[:, 12] >= 0.5
    if np.any(fallback_mask):
        return np.flatnonzero(fallback_mask)
    if np.any(valid_mask):
        return np.flatnonzero(valid_mask)
    return np.arange(int(state_dict["num_actions"]), dtype=np.int64)


def base_score_from_action_feature(
    action_feat: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    enforce_mapper_feasibility: bool,
) -> float:
    """Reconstruct the TopK predictor score from action-aware features.

    action_feat layout follows ACTION_FEATURE_DIM_V2:
      [bw_norm, compute_cost, interm_norm, p_success, delay_norm, server_load,
       deadline_slack_norm, delta_frag, delta_lfb_ratio, future_blocking_risk,
       path_lfb_ratio, path_utilization, mapper_feasible]
    """
    mapper_feasible = float(action_feat[12]) > 0.5
    if enforce_mapper_feasibility and not mapper_feasible:
        return -1e9

    p_success = float(action_feat[3])
    total_delay_over_100 = float(action_feat[4]) * 2.0
    server_load = float(action_feat[5])
    deadline_penalty = max(0.0, 1.0 - float(action_feat[6]))

    return (
        alpha * p_success
        - beta * total_delay_over_100
        - gamma * server_load
        - delta * deadline_penalty
    )


def soft_predictor_score_from_action_feature(
    action_feat: np.ndarray,
    *,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
) -> float:
    """Bounded predictor score used as an input signal, not a hard decision."""
    score = base_score_from_action_feature(
        action_feat,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
        enforce_mapper_feasibility=False,
    )
    return float(np.clip(score, -5.0, 5.0))


class ResidualQNet(nn.Module):
    """Small MLP for residual Q-values over action-aware inputs."""

    def __init__(self, input_dim: int = 16, hidden_dims=(64, 64)):
        super().__init__()
        layers = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class GatedResidualQNet(nn.Module):
    """Residual Q network with a learned state-dependent gate.

    The network predicts:
      - raw_delta(s, a): unrestricted residual value
      - gate(s, a) in [0, 2]: multiplicative gate

    The final network output is gate * raw_delta, so the training loop stays
    unchanged while residual strength becomes state-dependent.
    """

    def __init__(self, input_dim: int = 16, hidden_dims=(64, 64)):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = tuple(hidden_dims)
        layers = []
        prev = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev, hidden_dim))
            layers.append(nn.ReLU())
            prev = hidden_dim
        self.backbone = nn.Sequential(*layers)
        self.delta_head = nn.Linear(prev, 1)
        self.gate_head = nn.Linear(prev, 1)
        self.gated_residual = True

    def forward_details(self, x: torch.Tensor):
        h = self.backbone(x)
        raw_delta = self.delta_head(h).squeeze(-1)
        gate = 2.0 * torch.sigmoid(self.gate_head(h).squeeze(-1))
        gated_delta = gate * raw_delta
        return gated_delta, raw_delta, gate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gated_delta, _raw_delta, _gate = self.forward_details(x)
        return gated_delta


class DuelingResidualQNet(nn.Module):
    """Dueling-style Q network for action-aware scalar scoring.

    The value stream only looks at compact global features, while the advantage
    stream sees the full state-action input. Since actions are scored one by one,
    we use V(s) + A(s,a) without the across-action mean subtraction.
    """

    def __init__(self, input_dim: int = 16, hidden_dims=(64, 64), global_dim: int = 3):
        super().__init__()
        self.input_dim = input_dim
        self.global_dim = global_dim
        self.action_dim = max(1, input_dim - global_dim)

        hidden_1, hidden_2 = hidden_dims

        self.adv_backbone = nn.Sequential(
            nn.Linear(input_dim, hidden_1),
            nn.ReLU(),
            nn.Linear(hidden_1, hidden_2),
            nn.ReLU(),
        )
        self.adv_head = nn.Linear(hidden_2, 1)

        self.value_backbone = nn.Sequential(
            nn.Linear(global_dim, hidden_1 // 2),
            nn.ReLU(),
            nn.Linear(hidden_1 // 2, hidden_2 // 2),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_2 // 2, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        global_x = x[:, -self.global_dim :]
        adv = self.adv_head(self.adv_backbone(x))
        value = self.value_head(self.value_backbone(global_x))
        return (value + adv).squeeze(-1)


class SetAwareQNet(nn.Module):
    """DeepSets-style Q network over a candidate action set."""

    def __init__(self, input_dim: int = 16, hidden_dim: int = 128):
        super().__init__()
        self.set_aware = True
        self.input_dim = input_dim
        self.action_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, action_feats: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        squeeze = False
        if action_feats.dim() == 2:
            action_feats = action_feats.unsqueeze(0)
            valid_mask = valid_mask.unsqueeze(0)
            squeeze = True

        h = self.action_encoder(action_feats)
        mask = valid_mask.float().unsqueeze(-1)
        masked_h = h * mask
        denom = mask.sum(dim=1).clamp(min=1.0)
        context = masked_h.sum(dim=1) / denom
        context_expand = context.unsqueeze(1).expand_as(h)
        q = self.q_head(torch.cat([h, context_expand], dim=-1)).squeeze(-1)
        q = q.masked_fill(~valid_mask, -1e9)
        if squeeze:
            return q.squeeze(0)
        return q


class TSPQNet(nn.Module):
    """Two-stream predictor-informed Q network.

    Stream 1 provides a fixed, differentiable predictor-like score from action features.
    Stream 2 learns a deep RL representation from the full state-action vector.
    Fusion is non-linear, so this is not a residual additive structure.
    """

    def __init__(
        self,
        input_dim: int = 16,
        hidden_dims=(128, 128),
        *,
        alpha: float = 2.0,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.05,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        h1, h2 = hidden_dims
        self.rl_backbone = nn.Sequential(
            nn.Linear(input_dim, h1),
            nn.LayerNorm(h1),
            nn.ReLU(),
            nn.Linear(h1, h2),
            nn.LayerNorm(h2),
            nn.ReLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(h2 + 1, h2),
            nn.ReLU(),
            nn.Linear(h2, 1),
        )

    def predictor_score(self, x: torch.Tensor) -> torch.Tensor:
        p_success = x[:, 3]
        delay = x[:, 4] * 2.0
        load = x[:, 5]
        deadline_slack = x[:, 6]
        deadline_penalty = torch.clamp(1.0 - deadline_slack, min=0.0)
        score = (
            self.alpha * p_success
            - self.beta * delay
            - self.gamma * load
            - self.delta * deadline_penalty
        )
        return torch.clamp(score, -5.0, 5.0).unsqueeze(-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_score = self.predictor_score(x)
        rl_emb = self.rl_backbone(x)
        fused = torch.cat([rl_emb, base_score], dim=-1)
        return self.fusion(fused).squeeze(-1)


def build_q_net(
    input_dim: int = 16,
    hidden_dims=(64, 64),
    *,
    dueling: bool = False,
    set_aware: bool = False,
    tspq: bool = False,
    gated_residual: bool = False,
):
    if set_aware:
        return SetAwareQNet(input_dim=input_dim, hidden_dim=max(hidden_dims))
    if tspq:
        return TSPQNet(input_dim=input_dim, hidden_dims=hidden_dims)
    if gated_residual:
        return GatedResidualQNet(input_dim=input_dim, hidden_dims=hidden_dims)
    if dueling:
        return DuelingResidualQNet(input_dim=input_dim, hidden_dims=hidden_dims)
    return ResidualQNet(input_dim=input_dim, hidden_dims=hidden_dims)


def soft_update(target_net: nn.Module, online_net: nn.Module, tau: float) -> None:
    with torch.no_grad():
        for target_param, online_param in zip(target_net.parameters(), online_net.parameters()):
            target_param.data.mul_(1.0 - tau).add_(tau * online_param.data)


def compose_action_inputs(state_dict: Dict) -> np.ndarray:
    """Compose 16D per-action inputs for all actions in a state dict."""
    global_state = state_dict["global_state"]
    action_feats = state_dict["action_features"]
    inputs = [compose_correction_input(action_feat, global_state) for action_feat in action_feats]
    return np.stack(inputs).astype(np.float32)


def compose_relative_features(
    state_dict: Dict,
    action_id: int,
    *,
    candidate_ids: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    enforce_mapper_feasibility: bool,
) -> np.ndarray:
    """Build candidate-relative features for one action within a candidate set."""
    action_features = np.asarray(state_dict["action_features"], dtype=np.float32)
    if candidate_ids.size == 0:
        return np.zeros(7, dtype=np.float32)

    cand_feats = action_features[candidate_ids]
    predictor_scores = np.array(
        [
            soft_predictor_score_from_action_feature(
                feat,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                delta=delta,
            )
            for feat in cand_feats
        ],
        dtype=np.float32,
    )
    order = np.argsort(-predictor_scores)
    rank_pos = int(np.where(candidate_ids[order] == action_id)[0][0]) if action_id in candidate_ids else len(candidate_ids) - 1
    rank_norm = 1.0 if len(candidate_ids) <= 1 else 1.0 - (rank_pos / float(len(candidate_ids) - 1))

    best_action_id = int(candidate_ids[int(np.argmax(predictor_scores))])
    curr_feat = action_features[action_id]
    best_feat = action_features[best_action_id]
    best_pred = float(np.max(predictor_scores))
    curr_pred = soft_predictor_score_from_action_feature(
        curr_feat,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        delta=delta,
    )
    return np.array(
        [
            rank_norm,
            1.0 if action_id == best_action_id else 0.0,
            float(np.clip(curr_pred - best_pred, -2.0, 2.0)),
            float(np.clip(curr_feat[3] - best_feat[3], -1.0, 1.0)),   # p_success gap
            float(np.clip(curr_feat[4] - best_feat[4], -1.0, 1.0)),   # delay gap
            float(np.clip(curr_feat[5] - best_feat[5], -1.0, 1.0)),   # server load gap
            float(np.clip(curr_feat[7] - best_feat[7], -1.0, 1.0)),   # frag gap
        ],
        dtype=np.float32,
    )


def compose_augmented_q_input(
    state_dict: Dict,
    action_id: int,
    *,
    relative_features: bool,
    predictor_input_feature: bool,
    candidate_ids: np.ndarray,
    alpha: float,
    beta: float,
    gamma: float,
    delta: float,
    enforce_mapper_feasibility: bool,
) -> np.ndarray:
    """Compose Q-network input with optional candidate-relative features."""
    action_feat = state_dict["action_features"][action_id]
    global_state = state_dict["global_state"]
    base_input = compose_correction_input(action_feat, global_state)
    extras = []
    if predictor_input_feature:
        extras.append(
            np.array(
                [
                    soft_predictor_score_from_action_feature(
                        action_feat,
                        alpha=alpha,
                        beta=beta,
                        gamma=gamma,
                        delta=delta,
                    )
                ],
                dtype=np.float32,
            )
        )
    if not relative_features and not extras:
        return base_input
    if relative_features:
        rel = compose_relative_features(
            state_dict,
            action_id,
            candidate_ids=candidate_ids,
            alpha=alpha,
            beta=beta,
            gamma=gamma,
            delta=delta,
            enforce_mapper_feasibility=enforce_mapper_feasibility,
        )
        extras.append(rel)
    return np.concatenate([base_input, *extras]).astype(np.float32)


class ReplayBuffer:
    """Simple uniform replay buffer for online TD training."""

    def __init__(self, capacity: int = 50000):
        self.capacity = int(capacity)
        self.buffer: Deque[Dict] = deque(maxlen=self.capacity)

    def add(self, transition: Dict) -> None:
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Dict]:
        return random.sample(self.buffer, batch_size)

    def __len__(self) -> int:
        return len(self.buffer)


class OnlineResidualDQNAgent:
    """Residual agent that acts over Top-K candidates with online Q-learning."""

    def __init__(
        self,
        imitation_checkpoint_path: str,
        predictor,
        encoder,
        mec_cluster,
        q_net: ResidualQNet,
        *,
        num_servers: int = 5,
        top_k: int = 2,
        lambda_residual: float = 0.2,
        decision_mode: str = "residual",
        p_success_min: float = 0.0,
        alpha: float = 2.0,
        beta: float = 0.3,
        gamma: float = 0.2,
        delta: float = 0.05,
        enforce_mapper_feasibility: bool = True,
        use_enhanced_state: bool = True,
        relative_features: bool = False,
        predictor_input_feature: bool = False,
        load_imitation_mask: bool = True,
        use_predictor_action_features: bool = True,
        drop_predictor_action_features: bool = False,
        exploration_mode: str = "epsilon_greedy",
        exploration_temp: float = 1.0,
        device: str = "cpu",
    ):
        self.predictor = predictor
        self.encoder = encoder
        self.mec = mec_cluster
        self.q_net = q_net
        self.num_servers = num_servers
        self.top_k = top_k
        self.lambda_residual = lambda_residual
        self.decision_mode = decision_mode
        self.p_success_min = p_success_min
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.enforce_mapper_feasibility = enforce_mapper_feasibility
        self.use_enhanced_state = use_enhanced_state
        self.relative_features = relative_features
        self.predictor_input_feature = predictor_input_feature
        self.load_imitation_mask = load_imitation_mask
        self.use_predictor_action_features = use_predictor_action_features
        self.drop_predictor_action_features = drop_predictor_action_features
        self.exploration_mode = exploration_mode
        self.exploration_temp = exploration_temp
        self.device = device

        self.builder = ActionAwareStateBuilder(
            encoder=self.encoder,
            predictor=self.predictor,
            mec_cluster=self.mec,
            imitation_checkpoint_path=imitation_checkpoint_path,
            num_servers=self.num_servers,
            use_enhanced_state=self.use_enhanced_state,
            load_imitation_mask=self.load_imitation_mask,
            use_predictor_action_features=self.use_predictor_action_features,
            drop_predictor_action_features=self.drop_predictor_action_features,
            device=self.device,
        )

    def bind_runtime(self, encoder, mec_cluster) -> None:
        """Bind the latest env-backed encoder/MEC references before rollout."""
        self.encoder = encoder
        self.mec = mec_cluster
        self.builder.encoder = encoder
        self.builder.mec = mec_cluster

    def _candidate_action_ids(self, state_dict: Dict) -> np.ndarray:
        """Resolve the candidate action set for the current decision mode."""
        if self.decision_mode == "masked_q":
            return candidate_action_ids(
                state_dict,
                p_success_min=self.p_success_min,
                require_mapper_feasible=self.enforce_mapper_feasibility,
                use_topk=True,
            )
        if self.decision_mode == "pure_q_feasible":
            return candidate_action_ids(
                state_dict,
                p_success_min=self.p_success_min,
                require_mapper_feasible=self.enforce_mapper_feasibility,
                use_topk=False,
            )
        return candidate_action_ids(state_dict)

    def _compose_q_input(self, state_dict: Dict, action_id: int) -> np.ndarray:
        candidate_ids = self._candidate_action_ids(state_dict)
        return compose_augmented_q_input(
            state_dict,
            action_id,
            relative_features=self.relative_features,
            predictor_input_feature=self.predictor_input_feature,
            candidate_ids=candidate_ids,
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            enforce_mapper_feasibility=self.enforce_mapper_feasibility,
        )

    def _base_score(self, state_dict: Dict, action_id: int) -> float:
        return base_score_from_action_feature(
            state_dict["action_features"][action_id],
            alpha=self.alpha,
            beta=self.beta,
            gamma=self.gamma,
            delta=self.delta,
            enforce_mapper_feasibility=self.enforce_mapper_feasibility,
        )

    def build_state(self, request, env) -> Dict:
        self.builder.encoder = self.encoder
        self.builder.mec = self.mec
        return self.builder.build(request, env, top_k=self.top_k)

    def _final_score(self, state_dict: Dict, action_id: int, residual_q: float) -> float:
        if self.decision_mode in {"masked_q", "pure_q_feasible"}:
            return float(residual_q)
        base_score = self._base_score(state_dict, action_id)
        return base_score + self.lambda_residual * float(residual_q)

    def evaluate_state(self, state_dict: Dict) -> List[Dict]:
        """Evaluate candidate actions for the current state."""
        candidate_ids = self._candidate_action_ids(state_dict)
        if candidate_ids.size == 0:
            return []

        if getattr(self.q_net, "set_aware", False):
            all_inputs = np.stack([self._compose_q_input(state_dict, int(a)) for a in range(int(state_dict["num_actions"]))])
            valid_mask = np.zeros(int(state_dict["num_actions"]), dtype=bool)
            valid_mask[candidate_ids] = True
            q_inputs_t = torch.from_numpy(all_inputs).float().to(self.device)
            valid_mask_t = torch.from_numpy(valid_mask).bool().to(self.device)
            with torch.no_grad():
                q_all = self.q_net(q_inputs_t, valid_mask_t).cpu().numpy()
            q_values = q_all[candidate_ids]
        else:
            q_inputs = np.stack([self._compose_q_input(state_dict, int(a)) for a in candidate_ids])
            q_inputs_t = torch.from_numpy(q_inputs).float().to(self.device)
            with torch.no_grad():
                q_values = self.q_net(q_inputs_t).cpu().numpy()

        evaluated = []
        for local_idx, action_id in enumerate(candidate_ids.tolist()):
            split_id = action_id // self.num_servers
            server_id = action_id % self.num_servers
            base_score = self._base_score(state_dict, action_id)
            residual_q = float(q_values[local_idx])
            gate_value = None
            raw_delta = residual_q
            if not getattr(self.q_net, "set_aware", False) and hasattr(self.q_net, "forward_details"):
                single_input = torch.from_numpy(self._compose_q_input(state_dict, action_id)).float().unsqueeze(0).to(self.device)
                with torch.no_grad():
                    _gated_delta_t, raw_delta_t, gate_t = self.q_net.forward_details(single_input)
                gate_value = float(gate_t.squeeze(0).cpu().item())
                raw_delta = float(raw_delta_t.squeeze(0).cpu().item())
            final_score = self._final_score(state_dict, action_id, residual_q)
            feat = state_dict["action_features"][action_id]
            evaluated.append(
                {
                    "action_id": action_id,
                    "split_id": split_id,
                    "server_id": server_id,
                    "base_score": base_score,
                    "residual_q": residual_q,
                    "raw_delta": raw_delta,
                    "gate": gate_value,
                    "effective_lambda": (self.lambda_residual * gate_value) if gate_value is not None else self.lambda_residual,
                    "final_score": final_score,
                    "p_success": float(feat[3]),
                    "delay_norm": float(feat[4]),
                    "server_load": float(feat[5]),
                    "mapper_feasible": bool(float(feat[12]) > 0.5),
                }
            )
        return evaluated

    def select_action_from_state(
        self,
        state_dict: Dict,
        *,
        epsilon: float = 0.0,
        rng: Optional[np.random.RandomState] = None,
    ) -> Dict:
        evaluated = self.evaluate_state(state_dict)
        if not evaluated:
            return {
                "action_id": 0,
                "split_id": 0,
                "server_id": 0,
                "base_score": 0.0,
                "residual_q": 0.0,
                "final_score": 0.0,
                "explore": False,
                "evaluated": [],
            }

        if rng is None:
            rng = np.random.RandomState()

        if epsilon <= 0.0:
            chosen = dict(max(evaluated, key=lambda item: item["final_score"]))
            chosen["explore"] = False
        elif self.exploration_mode == "epsilon_boltzmann":
            if rng.rand() < epsilon:
                scores = np.asarray([item["final_score"] for item in evaluated], dtype=np.float32)
                temp = max(float(self.exploration_temp), 1e-3)
                logits = (scores - float(np.max(scores))) / temp
                probs = np.exp(logits)
                probs_sum = float(np.sum(probs))
                if not np.isfinite(probs_sum) or probs_sum <= 0.0:
                    probs = np.ones_like(probs) / max(len(probs), 1)
                else:
                    probs = probs / probs_sum
                chosen_idx = int(rng.choice(len(evaluated), p=probs))
                chosen = dict(evaluated[chosen_idx])
                chosen["explore"] = True
                chosen["explore_probs"] = probs.tolist()
            else:
                chosen = dict(max(evaluated, key=lambda item: item["final_score"]))
                chosen["explore"] = False
        else:
            if rng.rand() < epsilon:
                chosen = evaluated[int(rng.randint(0, len(evaluated)))]
                chosen = dict(chosen)
                chosen["explore"] = True
            else:
                chosen = dict(max(evaluated, key=lambda item: item["final_score"]))
                chosen["explore"] = False
        chosen["evaluated"] = evaluated
        return chosen

    def act(self, request, env, *, epsilon: float = 0.0, rng: Optional[np.random.RandomState] = None):
        state_dict = self.build_state(request, env)
        chosen = self.select_action_from_state(state_dict, epsilon=epsilon, rng=rng)
        return chosen["split_id"], chosen["server_id"], state_dict, chosen

    def decide(self, request, env):
        split_id, server_id, _state_dict, chosen = self.act(request, env, epsilon=0.0)
        return split_id, server_id, chosen["final_score"], {
            "type": "online_residual_dqn",
            "lambda": self.lambda_residual,
            "decision_mode": self.decision_mode,
            "evaluated": chosen["evaluated"],
            "explore": chosen["explore"],
        }
