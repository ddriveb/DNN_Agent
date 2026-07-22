"""Legacy-compatible Agent-R feature helper.

The old DQN Agent-R policy has been removed. This module is kept only as a
feature-builder shim for diagnostics and backward-compatible imports.
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple

from sa_hmarl.env.r_frag_aware import build_frag_feature_tail
from sa_hmarl.network.modulation import ModulationRegistry


class AgentR:
    """Feature-only Agent-R helper kept for compatibility."""


    def __init__(self,
                 input_dim: int,
                 mod_registry: ModulationRegistry,
                 hidden_dims=(128, 64),
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 lr: float = 1e-3,
                 device: str = 'cpu',
                 feature_mode: str = "default"):
        """Initialize the feature-helper state used by PPO-R and diagnostics."""
        if feature_mode not in ("default", "frag_aware", "c_aware"):
            raise ValueError(f"Unknown Agent-R feature_mode: {feature_mode}")
        self.input_dim = input_dim
        self.mod_registry = mod_registry
        self.device = device
        self.feature_mode = feature_mode
        # Legacy DQN args are accepted for caller compatibility but unused.
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr

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
        raise RuntimeError(
            "Legacy DQN Agent-R has been removed. Use PPOAgentR or the v1.2 "
            "ranker for policy selection; AgentR now only provides feature construction helpers."
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def optimize(self, batch: Tuple, batch_size: int) -> Optional[float]:
        raise RuntimeError(
            "Legacy DQN Agent-R optimization has been removed from the active codebase."
        )

    def update_target(self):
        raise RuntimeError(
            "Legacy DQN Agent-R target updates have been removed from the active codebase."
        )
