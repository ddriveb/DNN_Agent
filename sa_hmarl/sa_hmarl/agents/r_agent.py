"""Agent-R: 旧 DQN-R 与 R 端候选动作特征构造器。

历史上这个文件实现 DQN 风格的 R 端 RMSA 选择器；当前 final v1.2
仍复用其中的 ``build_action_features`` 来枚举/描述合法 R 候选动作。
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple

from sa_hmarl.env.r_frag_aware import build_frag_feature_tail
from sa_hmarl.network.modulation import ModulationRegistry


class AgentRNetwork(nn.Module):
    """旧 DQN-R 使用的共享 MLP，对每个候选 R 动作独立输出 Q 值。

    Input:  (batch, num_actions, input_dim)
    Output: (batch, num_actions)
    """

    def __init__(self, input_dim: int, hidden_dims=(128, 64)):
        """按输入维度和隐藏层配置创建候选动作打分网络。"""
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([nn.Linear(prev, h), nn.ReLU()])
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """把一批候选动作特征映射成对应的 Q 值矩阵。"""
        batch_size, num_actions, input_dim = x.shape
        x = x.view(-1, input_dim)
        out = self.net(x).view(batch_size, num_actions)
        return out


class AgentR:
    """旧 DQN-R agent，动作是 ``(path, modulation, block)``。

    目前主方法不直接用它的 DQN 策略做最终 R 决策，但 v1.2 ranker
    会调用本类的 ``build_action_features`` 获取候选动作特征和 mask。
    """

    def __init__(self,
                 input_dim: int,
                 mod_registry: ModulationRegistry,
                 hidden_dims=(128, 64),
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 lr: float = 1e-3,
                 device: str = 'cpu',
                 feature_mode: str = "default"):
        """初始化旧 DQN-R 网络、target 网络和特征模式。"""
        if feature_mode not in ("default", "frag_aware", "c_aware"):
            raise ValueError(f"Unknown Agent-R feature_mode: {feature_mode}")
        self.input_dim = input_dim
        self.mod_registry = mod_registry
        self.gamma = gamma
        self.epsilon = epsilon
        self.device = device
        self.step_count = 0
        self.feature_mode = feature_mode

        self.q_net = AgentRNetwork(input_dim, hidden_dims).to(device)
        self.target_net = AgentRNetwork(input_dim, hidden_dims).to(device)
        self.target_net.load_state_dict(self.q_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)

    # ------------------------------------------------------------------
    # Action-feature construction
    # ------------------------------------------------------------------

    def build_action_features(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """从 R 端观测构造每个 ``path/mod/block`` 候选动作的特征。

        这是当前代码里非常关键的函数：PPO-R baseline、旧 DQN-R、
        以及 final v1.2 ranker 的在线候选枚举都会依赖它。基础特征
        覆盖路径长度、hop、路径空闲/碎片化、调制格式、所需 FS、
        block 大小、浪费率和 path/mod 可行性。

        Returns:
            action_features: (num_actions, input_dim) float32 array
            mask:            (num_actions,) bool array (alias of obs["agent_r_mask"])
        """
        num_paths = len(obs["candidate_paths"])
        num_mods = len(obs["mod_names"])
        mask = obs["agent_r_mask"]

        if num_paths == 0 or num_mods == 0:
            return np.zeros((0, self.input_dim), dtype=np.float32), mask

        num_blocks = len(mask) // (num_paths * num_mods)

        features = []
        action_idx = 0
        for p_idx in range(num_paths):
            path_feat = obs["path_features"][p_idx]
            for m_idx in range(num_mods):
                mod = self.mod_registry[m_idx]
                req_fs = obs["required_fs_per_path_mod"][p_idx][m_idx]
                feasible = obs["feasible_mask_per_path_mod"][p_idx][m_idx]

                blocks = obs["candidate_blocks_per_path_mod"][p_idx][m_idx]
                for b_idx in range(num_blocks):
                    if b_idx < len(blocks):
                        block_size = blocks[b_idx][1]
                        if req_fs is not None and req_fs > 0:
                            waste = (block_size - req_fs) / block_size
                        else:
                            waste = 1.0
                    else:
                        block_size = 0
                        waste = 1.0

                    feature = [
                        path_feat["path_length_km"],
                        path_feat["hop_count"],
                        path_feat["lfb"],
                        path_feat["free_ratio"],
                        path_feat["frag_index"],
                        mod.spectral_efficiency,
                        mod.reach_km,
                        req_fs if req_fs is not None else 0,
                        block_size,
                        waste,
                        1.0 if feasible else 0.0,
                    ]
                    if self.feature_mode == "frag_aware":
                        feature.extend(build_frag_feature_tail(obs, action_idx).tolist())
                    elif self.feature_mode == "c_aware":
                        feature.extend(AgentR._build_c_aware_tail(obs))
                    features.append(feature)
                    action_idx += 1

        return np.array(features, dtype=np.float32), mask

    # ------------------------------------------------------------------
    # C-aware context tail (8-dim, appended to base 11 features)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_c_aware_tail(obs: Dict[str, Any]) -> list:
        """构造所有 R 候选动作共享的 C 端上下文特征。

        这些 8 维特征来自 ``build_agent_r_observation`` 塞入的
        ``c_context``，让 R 端知道当前动作服务的是哪个 split/server
        和什么 deadline/holding-time 请求。
        """
        ctx = obs.get("c_context", {})
        _INF = 1e6

        def _fin(v, default=0.0):
            try: v = float(v)
            except (TypeError, ValueError): return default
            if not np.isfinite(v): return default
            return v

        # Normalise to roughly [0, 1] or small range
        split_norm = _fin(ctx.get("split_id", 0)) / max(_fin(ctx.get("num_splits", 1)), 1)
        size_norm = _fin(ctx.get("intermediate_size_mb", 0)) / 100.0
        deadline_norm = _fin(ctx.get("deadline_ms", 100.0)) / 200.0
        slack_norm = _fin(ctx.get("deadline_slack_ms", 0.0)) / 200.0
        server_norm = _fin(ctx.get("server_id", 0)) / max(_fin(ctx.get("num_servers", 1)), 1)
        util = _fin(ctx.get("server_utilization", 0.0))
        queue_norm = _fin(ctx.get("server_queue_delay_ms", 0.0)) / 100.0
        holding_norm = _fin(ctx.get("holding_time_s", 10.0)) / 20.0

        return [
            split_norm, size_norm, deadline_norm, slack_norm,
            server_norm, util, queue_norm, holding_norm,
        ]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: Dict[str, Any], epsilon: Optional[float] = None) -> Optional[int]:
        """旧 DQN-R 的 epsilon-greedy 动作选择。

        只在 DQN-R baseline/旧训练中使用；final v1.2 在线选择动作时
        走 ``CounterfactualRRankerPolicy``，不是这里的 argmax Q。
        """
        if epsilon is None:
            epsilon = self.epsilon

        action_features, mask = self.build_action_features(obs)

        if len(mask) == 0 or not np.any(mask):
            return None

        if np.random.random() < epsilon:
            valid_actions = np.where(mask)[0]
            return int(np.random.choice(valid_actions))

        # Exploit: argmax Q over valid actions
        with torch.no_grad():
            x = torch.tensor(action_features, dtype=torch.float32).unsqueeze(0).to(self.device)
            q_values = self.q_net(x).squeeze(0).cpu().numpy()
            q_values[~mask] = -np.inf
            return int(np.argmax(q_values))

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def optimize(self, batch: Tuple, batch_size: int) -> Optional[float]:
        """对旧 DQN-R 执行一次 replay-buffer mini-batch 优化。

        因为不同状态下候选动作数量可能不同，这里先把 action feature
        和 mask pad 到 batch 内最大动作数，再计算 DQN TD 目标。

        Args:
            batch: Tuple of (obs_features_list, masks_list, actions, rewards,
                   next_obs_features_list, next_masks_list, dones) from
                   ReplayBuffer.sample().
            batch_size: Number of transitions in the batch.

        Returns:
            Loss scalar or None if batch is too small.
        """
        if batch is None:
            return None

        (obs_features_list, masks_list, actions,
         rewards, next_obs_features_list, next_masks_list, dones) = batch

        if len(actions) < batch_size:
            return None

        # Determine max action count for padding
        max_actions = max(
            max(len(m) for m in masks_list),
            max(len(m) for m in next_masks_list),
        )

        def _pad(features: np.ndarray, mask: np.ndarray, target_size: int):
            pad_len = target_size - len(mask)
            if pad_len > 0:
                features = np.concatenate([
                    features,
                    np.zeros((pad_len, features.shape[1]), dtype=np.float32)
                ], axis=0)
                mask = np.concatenate([mask, np.zeros(pad_len, dtype=bool)], axis=0)
            return features, mask

        padded_obs = []
        padded_masks = []
        for obs_f, m in zip(obs_features_list, masks_list):
            f, m = _pad(obs_f, m, max_actions)
            padded_obs.append(f)
            padded_masks.append(m)

        padded_next_obs = []
        padded_next_masks = []
        for obs_f, m in zip(next_obs_features_list, next_masks_list):
            f, m = _pad(obs_f, m, max_actions)
            padded_next_obs.append(f)
            padded_next_masks.append(m)

        obs_t = torch.tensor(np.stack(padded_obs), dtype=torch.float32, device=self.device)
        masks_t = torch.tensor(np.stack(padded_masks), dtype=torch.bool, device=self.device)
        actions_t = torch.tensor(actions, dtype=torch.long, device=self.device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=self.device)
        next_obs_t = torch.tensor(np.stack(padded_next_obs), dtype=torch.float32, device=self.device)
        next_masks_t = torch.tensor(np.stack(padded_next_masks), dtype=torch.bool, device=self.device)
        dones_t = torch.tensor(dones, dtype=torch.float32, device=self.device)

        # Current Q(s, a)
        current_q_all = self.q_net(obs_t)  # (batch, max_actions)
        current_q = current_q_all.gather(1, actions_t.unsqueeze(1)).squeeze(1)

        # Target: r + gamma * max_a' Q_target(s', a')
        # If next_mask has no valid actions, future value is 0.
        with torch.no_grad():
            next_q = self.target_net(next_obs_t)
            next_q = next_q.masked_fill(~next_masks_t, -1e9)
            next_q_max = next_q.max(1)[0]
            has_next_action = next_masks_t.any(dim=1)
            next_q_max = torch.where(has_next_action, next_q_max,
                                     torch.zeros_like(next_q_max))
            target_q = rewards_t + self.gamma * next_q_max * (1.0 - dones_t)

        loss = F.smooth_l1_loss(current_q, target_q)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.step_count += 1

        return loss.item()

    def update_target(self):
        """硬更新 target network，用当前 Q 网络覆盖 target Q 网络。"""
        self.target_net.load_state_dict(self.q_net.state_dict())
