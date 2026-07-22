"""Consistency test for vectorized vs per-action R-ranker feature batch builder."""
from __future__ import annotations

import numpy as np

from sa_hmarl.evaluation.generate_r_post_decision_dataset import FEATURE_NAMES
from sa_hmarl.evaluation.r_ranker_features import (
    build_r_ranker_feature,
    build_r_ranker_feature_batch,
)


def _make_dummy_obs_c(num_servers: int = 4):
    return {
        "candidate_features": [{}] * 12,
        "server_utilizations": np.asarray([0.2 * i for i in range(num_servers)], dtype=float),
        "feasible_counts": [[2, 3, 1, 0], [1, 2, 2, 1], [0, 1, 1, 2]],
    }


def _make_dummy_obs_r(num_actions: int = 60):
    return {
        "candidate_paths": [[0, 1, 2], [0, 2], [0, 1, 3], [1, 2, 3], [0, 3]],
        "mod_names": ["QPSK", "16QAM", "64QAM"],
        "agent_r_mask": np.asarray([i % 3 != 0 for i in range(num_actions)], dtype=bool),
        "num_slots": 320,
        "c_context": {"intermediate_size_mb": 12.5},
        "candidate_blocks_per_path_mod": [
            [
                [(i, 4 + (i + j) % 4) for i in range(10)]
                for j in range(3)
            ]
            for _ in range(5)
        ],
    }


def test_vectorized_matches_per_action():
    """Vectorized batch builder must match the legacy per-action builder."""
    class DummyEnv:
        max_blocks = 10

        class _Net:
            num_slots = 320

            class _G:
                def edges(self):
                    return [(0, 1), (1, 2), (2, 3)]
            G = _G()
        net = _Net()

        class _Mec:
            servers = [type("S", (), {"node_id": i}) for i in range(4)]
        mec = _Mec()

    class DummyReq:
        splits = [0, 1, 2]
        deadline_ms = 80.0
        holding_time = 25.0

    env = DummyEnv()
    req = DummyReq()
    obs_c = _make_dummy_obs_c()
    obs_r = _make_dummy_obs_r()
    r_features = np.random.RandomState(42).randn(60, 11).astype(np.float32)
    legal = np.flatnonzero(obs_r["agent_r_mask"]).tolist()[:20]

    for split_id in range(3):
        for server_id in range(4):
            per_action = np.stack([
                build_r_ranker_feature(
                    env, req, obs_c, obs_r, r_features, a, split_id, server_id,
                    feature_names=FEATURE_NAMES,
                )
                for a in legal
            ]).astype(np.float32)
            batch = build_r_ranker_feature_batch(
                env, req, obs_c, obs_r, r_features, legal, split_id, server_id,
                feature_names=FEATURE_NAMES,
            )
            assert per_action.shape == batch.shape, (
                f"shape mismatch at split={split_id}, server={server_id}"
            )
            np.testing.assert_allclose(
                per_action, batch, rtol=1e-6, atol=1e-6,
                err_msg=f"feature mismatch at split={split_id}, server={server_id}",
            )


def test_structured_feature_fallback():
    """Non-default feature_names must still use the per-action fallback."""
    class DummyEnv:
        max_blocks = 10

        class _Net:
            num_slots = 320

            class _G:
                def edges(self):
                    return [(0, 1), (1, 2)]
            G = _G()
        net = _Net()

        class _Mec:
            servers = [type("S", (), {"node_id": i}) for i in range(4)]
        mec = _Mec()

    class DummyReq:
        splits = [0, 1, 2]
        deadline_ms = 80.0
        holding_time = 25.0

    env = DummyEnv()
    req = DummyReq()
    obs_c = _make_dummy_obs_c()
    obs_r = _make_dummy_obs_r()
    r_features = np.random.RandomState(123).randn(60, 11).astype(np.float32)
    legal = [0, 5, 10]
    feature_names = list(FEATURE_NAMES) + ["block_start_norm"]

    per_action = np.stack([
        build_r_ranker_feature(
            env, req, obs_c, obs_r, r_features, a, 1, 2,
            feature_names=feature_names,
        )
        for a in legal
    ]).astype(np.float32)
    batch = build_r_ranker_feature_batch(
        env, req, obs_c, obs_r, r_features, legal, 1, 2,
        feature_names=feature_names,
    )
    np.testing.assert_allclose(per_action, batch, rtol=1e-6, atol=1e-6)
