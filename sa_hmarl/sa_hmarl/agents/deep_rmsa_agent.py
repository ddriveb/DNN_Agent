"""DeepRMSA Agent: A3C (Actor-Critic) adapted for SA-HMARL's Agent-R slot.

This is a PyTorch re-implementation of the DeepRMSA A3C algorithm
(Xiaoliang Chen et al., IEEE TNSM 2019) to replace Agent-R's DQN
within the SA-HMARL framework.

Key adaptations:
- Masked categorical action sampling: only valid DeepRMSA actions (those
  that decode to feasible SA-HMARL flat actions) are considered.
- Training: sample from policy; Evaluation: argmax over masked policy.
- Episode-level A2C with proper n-step/bootstrap value handling.
- Modulation selection: follows DeepRMSA's implicit rule (highest feasible SE).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List

from sa_hmarl.network.modulation import ModulationRegistry


class ACNetwork(nn.Module):
    """Actor-Critic network: 5-layer ELU MLP matching DeepRMSA architecture."""

    def __init__(self, input_dim: int, output_dim: int,
                 num_layers: int = 5, layer_size: int = 128):
        super().__init__()
        layers = []
        prev = input_dim
        for _ in range(num_layers):
            layers.append(nn.Linear(prev, layer_size))
            layers.append(nn.ELU())
            prev = layer_size
        self.backbone = nn.Sequential(*layers)
        self.head = nn.Linear(layer_size, output_dim)

        # DeepRMSA-style normalized column initialization for policy head
        if output_dim > 1:
            with torch.no_grad():
                std = 0.01
                w = torch.randn(layer_size, output_dim)
                w *= std / torch.sqrt(torch.sum(w ** 2, dim=0, keepdim=True))
                self.head.weight.copy_(w.t())
                self.head.bias.zero_()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.backbone(x))


class DeepRMSAAgent:
    """DeepRMSA A3C agent compatible with SA-HMARL Agent-R interface.

    Provides ``select_action`` and ``optimize`` methods that can be
    dropped into existing Agent-R training/evaluation loops.
    """

    def __init__(self,
                 num_nodes: int,
                 num_slots: int,
                 k_path: int = 3,
                 m_blocks: int = 1,
                 mod_registry: ModulationRegistry = None,
                 gamma: float = 0.95,
                 lr: float = 1e-5,
                 entropy_coef: float = 0.01,
                 value_loss_coef: float = 0.5,
                 max_grad_norm: float = 40.0,
                 num_layers: int = 5,
                 layer_size: int = 128,
                 device: str = 'cpu'):
        self.num_nodes = num_nodes
        self.num_slots = num_slots
        self.k_path = k_path
        self.m_blocks = m_blocks
        self.n_actions = k_path * m_blocks
        self.mod_registry = mod_registry
        self.gamma = gamma
        self.entropy_coef = entropy_coef
        self.value_loss_coef = value_loss_coef
        self.max_grad_norm = max_grad_norm
        self.device = device
        self.step_count = 0
        self.training = True  # True → sample; False → argmax

        # State dimension: NODE_NUM*2 + k_path*(1 + M*2 + 2)
        self.state_dim = num_nodes * 2 + k_path * (1 + m_blocks * 2 + 2)

        # Policy and value networks (non-shared, matching DeepRMSA)
        self.policy_net = ACNetwork(self.state_dim, self.n_actions,
                                    num_layers, layer_size).to(device)
        self.value_net = ACNetwork(self.state_dim, 1,
                                   num_layers, layer_size).to(device)

        # Single optimizer for both networks
        self.optimizer = torch.optim.Adam(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            lr=lr
        )

        # Episode buffer for on-policy A2C updates
        # Each entry: (state, action_id, reward, value, done)
        self.episode_buffer: List[Tuple] = []

    # ------------------------------------------------------------------
    # State encoding: SA-HMARL obs → DeepRMSA state vector
    # ------------------------------------------------------------------

    def encode_state(self, obs: Dict[str, Any]) -> np.ndarray:
        """Convert SA-HMARL Agent-R observation to DeepRMSA state vector."""
        src = obs.get("src_node", 0)
        dst = obs.get("dst_node", 0)
        num_paths = len(obs["candidate_paths"])
        num_mods = len(obs["mod_names"])

        # Node one-hot
        src_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        dst_onehot = np.zeros(self.num_nodes, dtype=np.float32)
        if 0 <= src < self.num_nodes:
            src_onehot[src] = 1.0
        if 0 <= dst < self.num_nodes:
            dst_onehot[dst] = 1.0

        features: List[float] = []
        features.extend(src_onehot.tolist())
        features.extend(dst_onehot.tolist())

        # Per-path features (matching DeepRMSA normalization)
        for p_idx in range(num_paths):
            best_mod_idx = self._best_mod_for_path(obs, p_idx)

            if best_mod_idx is None:
                features.extend([-1.0] * (1 + self.m_blocks * 2 + 2))
                continue

            req_fs = obs["required_fs_per_path_mod"][p_idx][best_mod_idx]
            if req_fs is None:
                req_fs = 0
            blocks = obs["candidate_blocks_per_path_mod"][p_idx][best_mod_idx]

            # 1) Required FS count (normalized, approx range 1-16)
            features.append((req_fs - 8.5) / 7.5)

            # 2) First M available FS-block start indices and sizes
            for b_idx in range(self.m_blocks):
                if b_idx < len(blocks):
                    start_idx, size = blocks[b_idx]
                    features.append(2.0 * (start_idx - 0.5 * self.num_slots) / self.num_slots)
                    features.append((size - self.num_slots / 2) / (self.num_slots / 2))
                else:
                    features.extend([-1.0, -1.0])

                # 3) Total available FS among candidate blocks (may be subset of full path)
            total_avail = sum(size for _, size in blocks)
            features.append(2.0 * (total_avail - 0.5 * self.num_slots) / self.num_slots)

            # 4) Mean size of FS-blocks
            avg_size = np.mean([size for _, size in blocks]) if blocks else 0.0
            features.append((avg_size - self.num_slots / 4) / (self.num_slots / 4))

        # Pad to fixed state_dim if fewer paths than k_path
        expected_per_path = 1 + self.m_blocks * 2 + 2
        actual_paths = len(features) - self.num_nodes * 2
        expected_paths = self.k_path * expected_per_path
        if actual_paths < expected_paths:
            features.extend([-1.0] * (expected_paths - actual_paths))

        return np.array(features, dtype=np.float32)

    def _best_mod_for_path(self, obs: Dict[str, Any], path_idx: int) -> Optional[int]:
        """Return highest-SE feasible modulation index for a path."""
        num_mods = len(obs["mod_names"])
        best_mod_idx = None
        best_se = -1.0
        for m_idx in range(num_mods):
            if obs["feasible_mask_per_path_mod"][path_idx][m_idx]:
                if self.mod_registry is not None and m_idx < self.mod_registry.num_formats:
                    se = self.mod_registry[m_idx].spectral_efficiency
                else:
                    se = 1.0
                if se > best_se:
                    best_se = se
                    best_mod_idx = m_idx
        return best_mod_idx

    # ------------------------------------------------------------------
    # Action validity mask: which DeepRMSA actions decode to valid SA-HMARL actions?
    # ------------------------------------------------------------------

    def _build_action_mask(self, obs: Dict[str, Any]) -> np.ndarray:
        """Build a mask over DeepRMSA action_ids.

        Returns bool array of shape (n_actions,) where True means the
        action_id can be decoded to a valid SA-HMARL flat action.
        """
        mask = np.zeros(self.n_actions, dtype=bool)
        num_paths = len(obs["candidate_paths"])
        num_mods = len(obs["mod_names"])
        num_blocks = len(obs["agent_r_mask"]) // (num_paths * num_mods) if num_paths > 0 else 0

        for action_id in range(self.n_actions):
            path_id = action_id // self.m_blocks
            fs_id = action_id % self.m_blocks

            if path_id >= num_paths:
                continue

            best_mod_idx = self._best_mod_for_path(obs, path_id)
            if best_mod_idx is None:
                continue

            blocks = obs["candidate_blocks_per_path_mod"][path_id][best_mod_idx]
            if fs_id >= len(blocks):
                continue

            flat_idx = path_id * (num_mods * num_blocks) + best_mod_idx * num_blocks + fs_id
            # Also check SA-HMARL's own mask
            if flat_idx < len(obs["agent_r_mask"]) and obs["agent_r_mask"][flat_idx]:
                mask[action_id] = True

        return mask

    def _decode_to_sahmarl(self, action_id: int, obs: Dict[str, Any]) -> Optional[int]:
        """Convert DeepRMSA action_id to SA-HMARL flat action index."""
        num_paths = len(obs["candidate_paths"])
        num_mods = len(obs["mod_names"])
        num_blocks = len(obs["agent_r_mask"]) // (num_paths * num_mods) if num_paths > 0 else 0

        path_id = action_id // self.m_blocks
        fs_id = action_id % self.m_blocks

        if path_id >= num_paths:
            return None

        best_mod_idx = self._best_mod_for_path(obs, path_id)
        if best_mod_idx is None:
            return None

        blocks = obs["candidate_blocks_per_path_mod"][path_id][best_mod_idx]
        if fs_id >= len(blocks):
            return None

        flat_idx = path_id * (num_mods * num_blocks) + best_mod_idx * num_blocks + fs_id
        return int(flat_idx)

    # ------------------------------------------------------------------
    # Action selection: masked categorical sampling
    # ------------------------------------------------------------------

    def select_action(self, obs: Dict[str, Any]) -> Optional[int]:
        """Select action using masked categorical sampling.

        - Training mode (self.training=True): sample from masked policy.
        - Eval mode (self.training=False): argmax over masked policy.

        Returns:
            SA-HMARL flat action index, or None if no valid DeepRMSA action exists.
        """
        state = self.encode_state(obs)
        state_t = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

        with torch.no_grad():
            logits = self.policy_net(state_t)
            value = self.value_net(state_t).cpu().item()

        # Build valid action mask
        action_mask = self._build_action_mask(obs)
        if not np.any(action_mask):
            self._last_valid = False
            return None

        # Apply mask: set invalid logits to -inf
        logits_np = logits.cpu().numpy().flatten()
        masked_logits = np.full_like(logits_np, -np.inf)
        masked_logits[action_mask] = logits_np[action_mask]

        # Convert to probabilities
        max_logit = np.max(masked_logits)
        exp_logits = np.exp(masked_logits - max_logit)
        probs = exp_logits / np.sum(exp_logits)

        if self.training:
            # Categorical sampling from masked distribution
            action_id = int(np.random.choice(self.n_actions, p=probs))
        else:
            # Greedy argmax
            action_id = int(np.argmax(probs))

        flat_action = self._decode_to_sahmarl(action_id, obs)

        # Store trajectory info for on-policy training
        self._last_valid = True
        self._last_state = state
        self._last_action_id = action_id
        self._last_value = value
        self._last_action_mask = action_mask

        return flat_action

    def store_transition(self, reward: float, done: bool):
        """Store a transition in the episode buffer for A2C updates.

        Call this after env.step() in the training loop.
        Skips storage if the last select_action had no valid action,
        preventing stale (_last_state, _last_action) from being reused.
        """
        if not getattr(self, '_last_valid', False):
            return
        self.episode_buffer.append((
            self._last_state,
            self._last_action_id,
            reward,
            self._last_value,
            self._last_action_mask,
            done,
        ))

    # ------------------------------------------------------------------
    # Training: episode-level A2C with proper bootstrap
    # ------------------------------------------------------------------

    def optimize(self, bootstrap_value: Optional[float] = None) -> Optional[Dict[str, float]]:
        """Run episode-level A2C optimization on the full episode buffer.

        Automatically bootstraps from the value network if the episode
        was truncated (last transition done=False). Pass an explicit
        ``bootstrap_value`` to override (e.g. 0.0 for terminal episodes).

        Args:
            bootstrap_value: Optional override for the value estimate of
                the state after the last transition. If None:
                - if last done is True → bootstrap_value = 0.0
                - else → query value_net on the last state.

        Returns:
            Dict of loss scalars, or None if buffer is empty.
        """
        if len(self.episode_buffer) == 0:
            return None

        states = np.stack([t[0] for t in self.episode_buffer])
        actions = np.array([t[1] for t in self.episode_buffer], dtype=np.int64)
        rewards = np.array([t[2] for t in self.episode_buffer], dtype=np.float32)
        values = np.array([t[3] for t in self.episode_buffer], dtype=np.float32)
        action_masks = np.stack([t[4] for t in self.episode_buffer])
        dones = np.array([t[5] for t in self.episode_buffer], dtype=np.float32)

        # Determine bootstrap value
        if bootstrap_value is None:
            if dones[-1] > 0.5:
                bootstrap_value = 0.0
            else:
                last_state = torch.tensor(states[-1], dtype=torch.float32).unsqueeze(0).to(self.device)
                with torch.no_grad():
                    bootstrap_value = self.value_net(last_state).cpu().item()

        # Compute discounted returns with done masking
        returns = self._compute_returns_with_dones(rewards, dones, bootstrap_value)
        advantages = returns - values

        # Convert to tensors
        states_t = torch.tensor(states, dtype=torch.float32, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        returns_t = torch.tensor(returns, dtype=torch.float32, device=self.device)
        advantages_t = torch.tensor(advantages, dtype=torch.float32, device=self.device)
        action_masks_t = torch.tensor(action_masks, dtype=torch.bool, device=self.device)

        # Policy loss: re-apply action masks so training distribution matches sampling distribution
        logits = self.policy_net(states_t)
        # Mask invalid actions with -1e9 (same as select_action)
        masked_logits = logits.masked_fill(~action_masks_t, -1e9)

        log_probs = F.log_softmax(masked_logits, dim=-1)
        log_probs_actions = log_probs.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        probs = F.softmax(masked_logits, dim=-1)
        entropy = -torch.sum(probs * log_probs, dim=-1).mean()

        policy_loss = -(log_probs_actions * advantages_t).mean()
        policy_loss = policy_loss - self.entropy_coef * entropy

        # Value loss
        values_pred = self.value_net(states_t).squeeze(-1)
        value_loss = F.mse_loss(values_pred, returns_t)

        # Total loss
        loss = policy_loss + self.value_loss_coef * value_loss

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(self.policy_net.parameters()) + list(self.value_net.parameters()),
            self.max_grad_norm
        )
        self.optimizer.step()
        self.step_count += 1

        # Episode-level update: clear buffer after optimization
        self.clear_buffer()

        return {
            "loss": loss.item(),
            "policy_loss": policy_loss.item(),
            "value_loss": value_loss.item(),
            "entropy": entropy.item(),
        }

    def _compute_returns_with_dones(self, rewards: np.ndarray,
                                    dones: np.ndarray,
                                    bootstrap_value: float) -> np.ndarray:
        """Compute discounted returns with proper done masking."""
        returns = np.zeros_like(rewards)
        running = bootstrap_value
        for t in reversed(range(len(rewards))):
            running = rewards[t] + self.gamma * running * (1.0 - dones[t])
            returns[t] = running
        return returns

    def clear_buffer(self):
        """Clear the episode buffer (call at episode start)."""
        self.episode_buffer.clear()

    def train(self, mode: bool = True):
        """Set training/eval mode for action sampling."""
        self.training = mode
        self.policy_net.train(mode)
        self.value_net.train(mode)
        return self

    def eval(self):
        """Set eval mode (argmax action selection)."""
        return self.train(False)

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, Any]:
        return {
            "policy_net": self.policy_net.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "state_dim": self.state_dim,
            "num_nodes": self.num_nodes,
            "num_slots": self.num_slots,
            "k_path": self.k_path,
            "m_blocks": self.m_blocks,
            "n_actions": self.n_actions,
            "gamma": self.gamma,
            "step_count": self.step_count,
        }

    def load_state_dict(self, state_dict: Dict[str, Any]):
        self.policy_net.load_state_dict(state_dict["policy_net"])
        self.value_net.load_state_dict(state_dict["value_net"])
        self.optimizer.load_state_dict(state_dict["optimizer"])
        self.step_count = state_dict.get("step_count", 0)
