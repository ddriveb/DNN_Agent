"""SA-HMARL v1.35 explicit afterstate feature diagnostic tests.

These tests guard the core invariants of the v1.35 diagnostic pilot:
  - The first 25 dimensions of poststate_v1 are bit-identical to v1.3 features.
  - Afterstate computation is deterministic and does not mutate the environment.
  - The strict evaluator can route v135_afterstate through the poststate builder.
"""
from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_r_action,
)
from sa_hmarl.evaluation import eval_strict_v13_multitopology_cside_verified as ev
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    build_poststate_v1_feature,
    build_poststate_v1_feature_batch,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


CHECKPOINT_C = "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt"
CHECKPOINT_R = "sa_hmarl/checkpoints/agent_r_mixed.pt"


@pytest.fixture(scope="module")
def env_and_requests():
    env = make_env(
        "xlron_cost239_ptrnet_real",
        num_slots=320,
        num_servers=4,
        seed=42,
        k=50,
        max_blocks=10,
        block_sort_strategy="start_asc",
        path_sort_strategy="hops",
    )
    rng = np.random.RandomState(42)
    src = rng.randint(0, env.net.NUM_NODES)
    requests = generate_requests(
        env, rng, src, 30, num_splits=3, split_profile="default3"
    )
    env.reset(requests)
    return env, requests


@pytest.fixture(scope="module")
def agents():
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c(CHECKPOINT_C, "cpu")
    agent_r = _load_ppo_r(CHECKPOINT_R, mod_reg, "cpu")
    return agent_c, agent_r


def _find_multi_action_state(env, requests, agent_c, agent_r):
    for req in requests:
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        raw_c_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
        if raw_c_mask.sum() == 0:
            env.step((0, 0), (0, 0, 0))
            continue
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = divmod(c_idx, len(env.mec.servers))
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        raw_r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        legal = np.where(raw_r_mask)[0].tolist()
        if len(legal) >= 2:
            return req, split_id, server_id, obs_c, obs_r, legal
        action_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        action_r = decode_agent_r_action(
            action_r_idx, len(obs_r["mod_names"]), env.max_blocks
        )
        env.step((split_id, server_id), action_r)
    pytest.skip("No multi-action R state found in fixture episode")


class TestPoststateV1Builder:
    def test_first_25_dims_match_v13(self, env_and_requests, agents):
        env, requests = env_and_requests
        agent_c, agent_r = agents
        snapshot = copy.deepcopy(env)
        req, split_id, server_id, obs_c, obs_r, legal = _find_multi_action_state(
            snapshot, requests, agent_c, agent_r
        )
        r_features, _ = agent_r.build_action_features(obs_r)

        for r_idx in legal[:5]:
            v13 = _r_feature_vector(
                snapshot, req, obs_c, obs_r, r_features, r_idx, split_id, server_id
            )
            v135 = build_poststate_v1_feature(
                snapshot, req, obs_c, obs_r, r_features, r_idx, split_id, server_id
            )
            assert v135.shape == (len(POSTSTATE_V1_FEATURE_NAMES),)
            assert np.allclose(v135[: len(FEATURE_NAMES)], v13, atol=1e-6)

    def test_batch_matches_single(self, env_and_requests, agents):
        env, requests = env_and_requests
        agent_c, agent_r = agents
        snapshot = copy.deepcopy(env)
        req, split_id, server_id, obs_c, obs_r, legal = _find_multi_action_state(
            snapshot, requests, agent_c, agent_r
        )
        r_features, _ = agent_r.build_action_features(obs_r)
        single = np.stack([
            build_poststate_v1_feature(
                snapshot, req, obs_c, obs_r, r_features, a, split_id, server_id
            )
            for a in legal[:5]
        ])
        batch = build_poststate_v1_feature_batch(
            snapshot, req, obs_c, obs_r, r_features, legal[:5], split_id, server_id
        )
        assert np.allclose(single, batch, atol=1e-6)

    def test_no_env_mutation(self, env_and_requests, agents):
        env, requests = env_and_requests
        agent_c, agent_r = agents
        snapshot = copy.deepcopy(env)
        req, split_id, server_id, obs_c, obs_r, legal = _find_multi_action_state(
            snapshot, requests, agent_c, agent_r
        )
        r_features, _ = agent_r.build_action_features(obs_r)

        # Capture a hashable view of every link state.
        before = {k: v.copy() for k, v in snapshot.net.link_states.items()}
        before_active = list(snapshot.active_connections)

        for r_idx in legal[:5]:
            build_poststate_v1_feature(
                snapshot, req, obs_c, obs_r, r_features, r_idx, split_id, server_id
            )

        after = {k: v.copy() for k, v in snapshot.net.link_states.items()}
        after_active = list(snapshot.active_connections)
        for k in before:
            assert np.array_equal(before[k], after[k]), f"link state mutated for edge {k}"
        assert before_active == after_active


class TestV135AfterstateRMode:
    def test_select_r_action_v135_afterstate_routes_through_poststate(self, env_and_requests, agents):
        env, requests = env_and_requests
        agent_c, agent_r = agents
        snapshot = copy.deepcopy(env)
        req, split_id, server_id, obs_c, obs_r, legal = _find_multi_action_state(
            snapshot, requests, agent_c, agent_r
        )

        # Build a deterministic dummy ranker with the v1.35 input dimension.
        model = build_counterfactual_r_ranker("mlp", len(POSTSTATE_V1_FEATURE_NAMES), (32, 16), 0.0)
        model.eval()
        ranker = {
            "model": model,
            "device": "cpu",
            "feature_mean": np.zeros(len(POSTSTATE_V1_FEATURE_NAMES), dtype=np.float32),
            "feature_std": np.ones(len(POSTSTATE_V1_FEATURE_NAMES), dtype=np.float32),
            "feature_names": list(POSTSTATE_V1_FEATURE_NAMES),
            "input_dim": len(POSTSTATE_V1_FEATURE_NAMES),
            "ckpt_sha256": "dummy",
        }

        ppo_idx = int(legal[0])
        r_idx, valid, info = ev._select_r_action(
            "v135_afterstate", agent_r, ranker, snapshot, req, obs_c, obs_r,
            ppo_idx, split_id, server_id,
        )
        assert valid
        assert r_idx in legal
        assert info["feature_finite"]
        assert info["candidate_count"] > 0

    def test_r_mode_in_ranker_modes(self):
        assert "v135_afterstate" in ev.RANKER_MODES
