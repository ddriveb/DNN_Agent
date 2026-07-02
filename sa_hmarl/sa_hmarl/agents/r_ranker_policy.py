"""final v1.2 R-ranker 的在线策略包装器。"""
from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

import numpy as np
import torch

from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import R_FEATURE_ORDER
from sa_hmarl.evaluation.r_ranker_features import build_r_ranker_feature_batch


class CounterfactualRRankerPolicy:
    """用蒸馏后的有限时域 planner 分数选择 RMSA 动作。

    这个类负责完整的在线 R 端决策流程：

        枚举合法 R 动作 -> 构造候选特征 -> 归一化
        -> 调用 ranker 网络 -> 选择分数最高动作

    被包装的 model 只会打分；它本身不访问环境、不生成 mask，
    在线推理时也不会 rollout 未来请求。

    支持按 checkpoint 的 candidate_mode 在线构造与训练时相同的候选集合
    （例如 ``legalctx48``），解决训练/在线候选分布不一致问题。
    """

    def __init__(
        self,
        model: torch.nn.Module,
        feature_mean: np.ndarray,
        feature_std: np.ndarray,
        device: str = "cpu",
        empty_action: int = 0,
        feature_names: Optional[Iterable[str]] = None,
        candidate_mode: str = "all_legal",
        max_candidates: int = 48,
        ppo_top_k: int = 8,
        num_random_candidates: int = 5,
        min_candidates: int = 15,
        candidate_seed: int = 12345,
    ):
        """保存 ranker 网络、特征归一化统计量和空动作兼容设置。"""
        self.model = model
        self.feature_mean = np.asarray(feature_mean, dtype=np.float32)
        self.feature_std = np.asarray(feature_std, dtype=np.float32)
        self.device = device
        self.empty_action = int(empty_action)
        self.feature_names = list(feature_names) if feature_names is not None else None
        self.candidate_mode = candidate_mode
        self.max_candidates = int(max_candidates)
        self.ppo_top_k = int(ppo_top_k)
        self.num_random_candidates = int(num_random_candidates)
        self.min_candidates = int(min_candidates)
        self.candidate_seed = int(candidate_seed)
        self.model.eval()

    def _select_online_candidates(
        self,
        obs_r: Dict[str, Any],
        r_features: np.ndarray,
        legal: list,
        max_blocks: int,
        agent_r,
    ) -> list:
        """Return the candidate set the ranker was trained on."""
        if self.candidate_mode == "all_legal":
            return legal

        num_mods = len(obs_r["mod_names"])

        # 1. v1 core subset.
        # PPO-R top-K.
        logits = self._ppo_logits(agent_r, obs_r)
        top_k = min(self.ppo_top_k, len(legal))
        ppo_top = np.argsort(-logits[legal], kind="stable")[:top_k]
        candidates = [int(legal[i]) for i in ppo_top]

        # Per-path heuristics.
        by_path: Dict[int, list] = {}
        for a in legal:
            path_idx, _, block_idx = decode_agent_r_action(a, num_mods, max_blocks)
            by_path.setdefault(path_idx, []).append(a)

        for actions in by_path.values():
            feats = {a: r_features[a] for a in actions}
            candidates.append(min(actions, key=lambda a: decode_agent_r_action(a, num_mods, max_blocks)[2]))
            candidates.append(min(actions, key=lambda a: abs(feats[a][R_FEATURE_ORDER["block_size"]] - feats[a][R_FEATURE_ORDER["required_fs"]])))
            candidates.append(max(actions, key=lambda a: feats[a][R_FEATURE_ORDER["block_size"]]))
            candidates.append(min(actions, key=lambda a: feats[a][R_FEATURE_ORDER["block_waste"]]))

        # Shortest path candidate.
        candidates.append(min(legal, key=lambda a: r_features[a][R_FEATURE_ORDER["path_length_km"]]))

        # Random fillers.
        rng = np.random.RandomState(self.candidate_seed)
        remaining = [a for a in legal if a not in candidates]
        if remaining:
            n_random = min(self.num_random_candidates, len(remaining))
            candidates.extend([int(a) for a in rng.choice(remaining, size=n_random, replace=False)])

        # Deduplicate.
        seen = set()
        deduped = [a for a in candidates if a not in seen and a in legal and not seen.add(a)]

        if len(deduped) < self.min_candidates and len(legal) <= self.max_candidates:
            deduped = list(dict.fromkeys(legal))

        core = deduped[: self.max_candidates]

        if self.candidate_mode == "legalctx48":
            # Fill remaining slots with highest-logit legal actions.
            if len(core) < self.max_candidates:
                core_set = set(core)
                remaining = [a for a in legal if a not in core_set]
                # np.argsort(logits[legal]) returns positions inside `legal`,
                # not flat R action ids.  Sort the actual action ids so online
                # inference matches dataset generation exactly.
                remaining_sorted = sorted(remaining, key=lambda a: -logits[int(a)])
                fill = remaining_sorted[: self.max_candidates - len(core)]
                core = core + fill
        return core

    @staticmethod
    def _ppo_logits(agent_r, obs_r: Dict[str, Any]) -> np.ndarray:
        """Return PPO-R logits for every flat R action."""
        features, mask = agent_r.build_action_features(obs_r)
        with torch.no_grad():
            x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
            logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
        logits[~np.asarray(mask, dtype=bool)] = -np.inf
        return logits

    def score_legal_actions(
        self,
        env,
        req,
        obs_c: Dict[str, Any],
        obs_r: Dict[str, Any],
        agent_r,
        split_id: int,
        server_id: int,
    ) -> tuple[list[int], np.ndarray]:
        """枚举当前合法 R 动作（或训练时一致的候选子集），并返回每个动作的 v1.2 打分。"""
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).astype(int).tolist()
        if not legal:
            return [], np.empty((0,), dtype=np.float32)
        if len(legal) == 1:
            return legal, np.asarray([0.0], dtype=np.float32)

        candidate_actions = self._select_online_candidates(
            obs_r, r_features, legal, env.max_blocks, agent_r
        )

        features = build_r_ranker_feature_batch(
            env, req, obs_c, obs_r, r_features, candidate_actions, split_id, server_id,
            feature_names=self.feature_names,
        )
        normalized = (features - self.feature_mean) / self.feature_std
        with torch.no_grad():
            scores = self.model(
                torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
            ).detach().cpu().numpy()
        return candidate_actions, np.asarray(scores, dtype=np.float32).reshape(-1)

    def select_action(
        self,
        env,
        req,
        obs_c: Dict[str, Any],
        obs_r: Dict[str, Any],
        agent_r,
        split_id: int,
        server_id: int,
    ) -> Optional[int]:
        """返回最终选择的展平 R 动作 id。

        历史 v1.2 在没有合法 R 动作时返回 ``0``；这里保留该行为，
        方便和既有实验指标保持一致。
        """
        legal, scores = self.score_legal_actions(
            env, req, obs_c, obs_r, agent_r, split_id, server_id
        )
        if not legal:
            return self.empty_action
        if len(legal) == 1:
            return int(legal[0])
        return int(legal[int(np.argmax(scores))])


# Paper-facing alias: online policy for planner-distilled amortized MPC.
PlannerDistilledRPolicy = CounterfactualRRankerPolicy
