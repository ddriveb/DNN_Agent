import argparse

import numpy as np

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.agents.deep_rmsa_source_semantic_agent import DeepRMSASourceSemanticAgent
from sa_hmarl.evaluation.optical_only_rmsa_env import (
    ODRequest,
    OpticalOnlyRMSAEnv,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.train_deeprmsa_optical_k50 import (
    _credit_forced_block,
    build_agent,
)


def _args():
    return argparse.Namespace(
        gamma=0.95,
        lr=1e-5,
        entropy_coef=0.01,
        value_loss_coef=0.5,
        max_grad_norm=40.0,
    )


def test_path_cache_preserves_observation():
    env = OpticalOnlyRMSAEnv(k_paths=50, max_blocks=10, seed=42)
    req = ODRequest(0, 0, 4, 50, 0.0, 10.0)
    first = env.build_observation(req)
    second = env.build_observation(req)
    assert first["candidate_paths"] == second["candidate_paths"]
    assert first["path_features"] == second["path_features"]
    assert np.array_equal(first["agent_r_mask"], second["agent_r_mask"])
    assert len(env._path_cache) == 1


def test_k50_agent_selects_current_legal_action():
    env = OpticalOnlyRMSAEnv(k_paths=50, max_blocks=10, seed=42)
    req = ODRequest(0, 0, 4, 75, 0.0, 10.0)
    obs = env.build_observation(req)
    agent = build_agent(42, 10, "cpu", _args())
    agent.eval()
    action = agent.select_action(obs)
    assert action is not None
    assert bool(obs["agent_r_mask"][action])


def test_m1_and_m10_have_locked_k50_dimensions():
    m1 = build_agent(42, 1, "cpu", _args())
    m10 = build_agent(42, 10, "cpu", _args())
    assert m1.k_path == m10.k_path == 50
    assert m1.n_actions == 50
    assert m10.n_actions == 500
    assert m1.num_slots == m10.num_slots == 100


def test_forced_block_credit_is_discounted_into_previous_action():
    agent = DeepRMSAAgent(
        num_nodes=11,
        num_slots=100,
        k_path=2,
        m_blocks=1,
        mod_registry=ModulationRegistry.from_profile("default"),
        gamma=0.95,
    )
    state = np.zeros(agent.state_dim, dtype=np.float32)
    mask = np.ones(agent.n_actions, dtype=bool)
    agent.episode_buffer.append((state, 0, 1.0, 0.0, mask, False))
    _credit_forced_block(agent, -1.0)
    assert np.isclose(agent.episode_buffer[-1][2], 0.05)


def test_source_semantic_agent_does_not_mask_unavailable_action():
    env = OpticalOnlyRMSAEnv(k_paths=50, max_blocks=10, seed=42)
    for slots in env.net.link_states.values():
        slots[:] = True
    obs = env.build_observation(ODRequest(0, 0, 4, 75, 0.0, 10.0))
    agent = DeepRMSASourceSemanticAgent(
        num_nodes=11,
        num_slots=100,
        k_path=50,
        m_blocks=1,
        mod_registry=ModulationRegistry.from_profile("default"),
    )
    agent.eval()
    assert not np.any(obs["agent_r_mask"])
    assert agent.select_action(obs) is None
    assert agent._last_valid is True
    assert np.all(agent._last_action_mask)
    agent.store_transition(-1.0, False)
    assert len(agent.episode_buffer) == 1
    assert agent.episode_buffer[0][2] == -1.0


def test_source_semantic_unavailable_path_features_are_negative_one():
    env = OpticalOnlyRMSAEnv(k_paths=50, max_blocks=10, seed=42)
    for slots in env.net.link_states.values():
        slots[:] = True
    obs = env.build_observation(ODRequest(0, 0, 4, 50, 0.0, 10.0))
    agent = DeepRMSASourceSemanticAgent(
        num_nodes=11,
        num_slots=100,
        k_path=50,
        m_blocks=1,
        mod_registry=ModulationRegistry.from_profile("default"),
    )
    state = agent.encode_state(obs)
    assert state.shape == (agent.state_dim,)
    assert np.all(state[22:] == -1.0)
