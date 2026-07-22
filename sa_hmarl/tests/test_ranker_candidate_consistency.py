"""Consistency test for optimized vs reference candidate construction in v1.3 ranker."""
from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from sa_hmarl.agents.r_ranker_policy import CounterfactualRRankerPolicy
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import R_FEATURE_ORDER


def _old_build_core_candidates(
    policy: CounterfactualRRankerPolicy,
    obs_r: Dict[str, Any],
    r_features: np.ndarray,
    legal: list,
    max_blocks: int,
    agent_r,
) -> list:
    """Reference implementation copied before vectorization optimizations."""
    num_mods = len(obs_r["mod_names"])

    logits = policy._ppo_logits(agent_r, obs_r)
    top_k = min(policy.ppo_top_k, len(legal))
    ppo_top = np.argsort(-logits[legal], kind="stable")[:top_k]
    candidates = [int(legal[i]) for i in ppo_top]

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

    candidates.append(min(legal, key=lambda a: r_features[a][R_FEATURE_ORDER["path_length_km"]]))

    rng = np.random.RandomState(policy.candidate_seed)
    remaining = [a for a in legal if a not in candidates]
    if remaining:
        n_random = min(policy.num_random_candidates, len(remaining))
        candidates.extend([int(a) for a in rng.choice(remaining, size=n_random, replace=False)])

    seen = set()
    deduped = [a for a in candidates if a not in seen and a in legal and not seen.add(a)]

    if len(deduped) < policy.min_candidates and len(legal) <= policy.max_candidates:
        deduped = list(dict.fromkeys(legal))

    core = deduped[: policy.max_candidates]

    if policy.candidate_mode == "legalctx48":
        if len(core) < policy.max_candidates:
            core_set = set(core)
            remaining = [a for a in legal if a not in core_set]
            remaining_sorted = sorted(remaining, key=lambda a: -logits[int(a)])
            fill = remaining_sorted[: policy.max_candidates - len(core)]
            core = core + fill
    return core


class _DummyAgentR:
    device = "cpu"

    def __init__(self, n_actions: int):
        self.n_actions = n_actions
        self.policy_net = _DummyNet(n_actions)

    def build_action_features(self, obs_r):
        mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        features = np.random.RandomState(7).randn(mask.shape[0], 11).astype(np.float32)
        return features, mask


class _DummyNet:
    def __init__(self, n_actions: int):
        self.n_actions = n_actions

    def __call__(self, x):
        # x shape: (batch=1, n_actions, input_dim)
        # Return deterministic pseudo-logits of shape (batch=1, n_actions).
        vals = x.sum(dim=-1)  # (1, n_actions)
        vals = vals + torch.arange(vals.shape[1], dtype=vals.dtype, device=vals.device).unsqueeze(0)
        return vals


def test_optimized_candidates_match_reference():
    np.random.seed(0)

    num_paths, num_mods, max_blocks = 5, 3, 10
    n_actions = num_paths * num_mods * max_blocks
    obs_r = {
        "candidate_paths": [[0, 1], [0, 2], [1, 2], [0, 1, 3], [2, 3]],
        "mod_names": ["QPSK", "16QAM", "64QAM"],
        "agent_r_mask": np.asarray([i % 3 != 0 for i in range(n_actions)], dtype=bool),
    }
    r_features = np.random.RandomState(42).randn(n_actions, 11).astype(np.float32)
    r_features[:, R_FEATURE_ORDER["block_size"]] = np.abs(r_features[:, R_FEATURE_ORDER["block_size"]]) * 5 + 1
    r_features[:, R_FEATURE_ORDER["required_fs"]] = np.abs(r_features[:, R_FEATURE_ORDER["required_fs"]]) * 3 + 1
    r_features[:, R_FEATURE_ORDER["path_length_km"]] = np.abs(r_features[:, R_FEATURE_ORDER["path_length_km"]]) * 100
    r_features[:, R_FEATURE_ORDER["block_waste"]] = np.clip(r_features[:, R_FEATURE_ORDER["block_waste"]], 0, 1)
    legal = np.flatnonzero(obs_r["agent_r_mask"]).tolist()

    dummy_model = torch.nn.Linear(25, 1)
    for candidate_mode in ["v1", "legalctx48"]:
        for max_candidates in [16, 24, 32, 48]:
            policy = CounterfactualRRankerPolicy(
                dummy_model,
                feature_mean=np.zeros(25, dtype=np.float32),
                feature_std=np.ones(25, dtype=np.float32),
                candidate_mode=candidate_mode,
                max_candidates=max_candidates,
                ppo_top_k=8,
                num_random_candidates=5,
                min_candidates=15,
                candidate_seed=12345,
                ensure_ksp_action=False,
            )
            agent_r = _DummyAgentR(n_actions)
            ref = _old_build_core_candidates(policy, obs_r, r_features, legal, max_blocks, agent_r)
            opt = policy._build_core_candidates(obs_r, r_features, legal, max_blocks, agent_r)
            assert ref == opt, (
                f"Mismatch for {candidate_mode}, K={max_candidates}: "
                f"ref={ref[:10]}... opt={opt[:10]}..."
            )
