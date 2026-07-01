"""Unit tests for DeepRMSA A3C Agent."""
import numpy as np
import pytest
import torch

from sa_hmarl.agents.deep_rmsa_agent import ACNetwork, DeepRMSAAgent
from sa_hmarl.network.modulation import ModulationRegistry


def _make_obs(feasible=True):
    """Build a sample observation for testing."""
    return {
        "src_node": 0,
        "dst_node": 5,
        "candidate_paths": [[0, 1, 5], [0, 2, 5], [0, 3, 5]],
        "path_features": [
            {"path_length_km": 1000, "hop_count": 2, "lfb": 5,
             "free_ratio": 0.8, "frag_index": 0.2},
            {"path_length_km": 1200, "hop_count": 2, "lfb": 3,
             "free_ratio": 0.7, "frag_index": 0.3},
            {"path_length_km": 1500, "hop_count": 2, "lfb": 2,
             "free_ratio": 0.6, "frag_index": 0.4},
        ],
        "mod_names": ["BPSK", "QPSK", "8QAM", "16QAM"],
        "feasible_mask_per_path_mod": [
            [True, True, True, True],
            [True, True, True, False],
            [True, True, False, False],
        ],
        "required_fs_per_path_mod": [
            [8, 6, 5, 4],
            [8, 6, 5, None],
            [8, 6, None, None],
        ],
        "candidate_blocks_per_path_mod": [
            [[(0, 10), (12, 8)], [(0, 10), (12, 8)], [(0, 10), (12, 8)], [(0, 10)]],
            [[(2, 6)], [(2, 6)], [(2, 6)], []],
            [[(5, 4)], [(5, 4)], [], []],
        ],
        "agent_r_mask": np.array([True] * 60, dtype=bool),
    }


def _make_infeasible_obs():
    """Build an observation where no path is feasible."""
    return {
        "src_node": 0,
        "dst_node": 5,
        "candidate_paths": [[0, 1, 5]],
        "path_features": [
            {"path_length_km": 1000, "hop_count": 2, "lfb": 5,
             "free_ratio": 0.8, "frag_index": 0.2},
        ],
        "mod_names": ["BPSK", "QPSK", "8QAM", "16QAM"],
        "feasible_mask_per_path_mod": [
            [False, False, False, False],
        ],
        "required_fs_per_path_mod": [
            [None, None, None, None],
        ],
        "candidate_blocks_per_path_mod": [
            [[], [], [], []],
        ],
        "agent_r_mask": np.array([False] * 20, dtype=bool),
    }


# ------------------------------------------------------------------
# ACNetwork
# ------------------------------------------------------------------

def test_ac_network_policy_shape():
    net = ACNetwork(input_dim=43, output_dim=5, num_layers=5, layer_size=128)
    x = torch.randn(2, 43)
    out = net(x)
    assert out.shape == (2, 5)


def test_ac_network_value_shape():
    net = ACNetwork(input_dim=43, output_dim=1, num_layers=5, layer_size=128)
    x = torch.randn(2, 43)
    out = net(x)
    assert out.shape == (2, 1)


# ------------------------------------------------------------------
# State encoding
# ------------------------------------------------------------------

def test_encode_state_shape():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    state = agent.encode_state(obs)
    assert state.shape == (agent.state_dim,)
    assert state.dtype == np.float32


def test_encode_state_src_dst_onehot():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=4, num_slots=32, k_path=2, m_blocks=1,
                          mod_registry=mod_reg)
    obs = {
        "src_node": 1,
        "dst_node": 3,
        "candidate_paths": [[1, 3], [1, 2, 3]],
        "path_features": [
            {"path_length_km": 500, "hop_count": 1, "lfb": 10,
             "free_ratio": 0.9, "frag_index": 0.1},
            {"path_length_km": 1000, "hop_count": 2, "lfb": 5,
             "free_ratio": 0.8, "frag_index": 0.2},
        ],
        "mod_names": ["BPSK", "QPSK"],
        "feasible_mask_per_path_mod": [
            [True, True],
            [True, False],
        ],
        "required_fs_per_path_mod": [
            [4, 3],
            [4, None],
        ],
        "candidate_blocks_per_path_mod": [
            [[(0, 10)], [(0, 10)]],
            [[(5, 6)], []],
        ],
        "agent_r_mask": np.array([True] * 4, dtype=bool),
    }
    state = agent.encode_state(obs)
    assert state[0] == 0.0  # src != 0
    assert state[1] == 1.0  # src == 1
    assert state[4] == 0.0  # dst != 0
    assert state[5] == 0.0  # dst != 1
    assert state[6] == 0.0  # dst != 2
    assert state[7] == 1.0  # dst == 3


# ------------------------------------------------------------------
# Action mask
# ------------------------------------------------------------------

def test_build_action_mask_all_valid():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    mask = agent._build_action_mask(obs)
    # All 3 paths have at least one feasible mod and one block
    assert mask.shape == (3,)
    assert np.all(mask == True)


def test_build_action_mask_infeasible():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_infeasible_obs()
    mask = agent._build_action_mask(obs)
    assert np.all(mask == False)


def test_build_action_mask_respects_sahmarl_mask():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    # action_id=0 → path=0, fs=0 → best mod for path 0 = 16QAM (idx 3)
    # flat_idx = 0*(4*5) + 3*5 + 0 = 15
    obs["agent_r_mask"][15] = False
    mask = agent._build_action_mask(obs)
    assert mask[0] == False
    assert mask[1] == True
    assert mask[2] == True


# ------------------------------------------------------------------
# Action decoding
# ------------------------------------------------------------------

def test_decode_to_sahmarl_basic():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    # action_id=0 → path=0, fs=0 → best mod for path 0 = 16QAM (idx 3)
    flat = agent._decode_to_sahmarl(0, obs)
    num_mods = 4
    num_blocks = 5
    assert flat == 0 * (num_mods * num_blocks) + 3 * num_blocks + 0


def test_decode_infeasible_path_returns_none():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_infeasible_obs()
    flat = agent._decode_to_sahmarl(0, obs)
    assert flat is None


# ------------------------------------------------------------------
# select_action: training vs eval mode
# ------------------------------------------------------------------

def test_select_action_training_returns_valid():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    agent.train()
    action_idx = agent.select_action(obs)
    assert action_idx is None or (0 <= action_idx < 60)


def test_select_action_eval_returns_valid():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    agent.eval()
    action_idx = agent.select_action(obs)
    assert action_idx is None or (0 <= action_idx < 60)


def test_select_action_all_invalid_returns_none():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_infeasible_obs()
    agent.train()
    action_idx = agent.select_action(obs)
    assert action_idx is None


def test_select_action_eval_deterministic():
    """Eval mode should return the same action for the same obs (argmax)."""
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    agent.eval()
    actions = [agent.select_action(obs) for _ in range(10)]
    assert all(a == actions[0] for a in actions)


def test_select_action_training_stochastic():
    """Training mode may return different actions (sampling)."""
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    agent.train()
    actions = [agent.select_action(obs) for _ in range(20)]
    # With 3 valid actions, sampling should produce variation occasionally
    assert any(a is not None for a in actions)


# ------------------------------------------------------------------
# store_transition: no-valid-action should skip (P1a)
# ------------------------------------------------------------------

def test_store_transition_skips_after_invalid_select():
    """If select_action returns None, store_transition must not pollute buffer."""
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)

    # First, a valid action
    obs_valid = _make_obs()
    agent.train()
    agent.select_action(obs_valid)
    assert agent._last_valid is True
    agent.store_transition(reward=1.0, done=False)
    assert len(agent.episode_buffer) == 1

    # Then, an invalid action (no valid mask)
    obs_invalid = _make_infeasible_obs()
    agent.select_action(obs_invalid)
    assert getattr(agent, '_last_valid', False) is False
    agent.store_transition(reward=-1.0, done=False)
    # Buffer should NOT grow — stale _last_state must not be reused
    assert len(agent.episode_buffer) == 1


def test_store_transition_stores_mask_in_buffer():
    """Buffer entries should include the action mask (P1b)."""
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    obs = _make_obs()
    agent.train()
    agent.select_action(obs)
    agent.store_transition(reward=1.0, done=False)

    assert len(agent.episode_buffer) == 1
    entry = agent.episode_buffer[0]
    # (state, action_id, reward, value, action_mask, done)
    assert len(entry) == 6
    assert isinstance(entry[4], np.ndarray)
    assert entry[4].dtype == bool
    assert entry[4].shape == (3,)


# ------------------------------------------------------------------
# optimize: episode-level with done handling and masked policy (P1b)
# ------------------------------------------------------------------

def test_optimize_terminal_episode():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)

    # Fill buffer with a short terminal episode
    for i in range(5):
        state = np.random.randn(agent.state_dim).astype(np.float32)
        mask = np.array([True, False, True], dtype=bool)
        agent.episode_buffer.append((state, 0, 1.0, 0.5, mask, True))

    loss_dict = agent.optimize(bootstrap_value=0.0)
    assert loss_dict is not None
    assert "loss" in loss_dict
    assert "policy_loss" in loss_dict
    assert "value_loss" in loss_dict
    assert "entropy" in loss_dict
    assert len(agent.episode_buffer) == 0  # cleared after optimize


def test_optimize_truncated_episode_uses_bootstrap():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)

    # Truncated episode: last done=False
    for i in range(4):
        state = np.random.randn(agent.state_dim).astype(np.float32)
        mask = np.array([True, True, False], dtype=bool)
        agent.episode_buffer.append((state, 0, 1.0, 0.5, mask, False))

    # bootstrap_value=None → should query value_net for last state
    loss_dict = agent.optimize(bootstrap_value=None)
    assert loss_dict is not None
    assert len(agent.episode_buffer) == 0


def test_optimize_empty_buffer_returns_none():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)
    loss_dict = agent.optimize(bootstrap_value=0.0)
    assert loss_dict is None


def test_optimize_ignores_masked_actions_in_loss():
    """Masked actions must not contribute to log_prob normalization (P1b)."""
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)

    # Create a buffer where only action 0 is valid for all transitions
    for _ in range(3):
        state = np.random.randn(agent.state_dim).astype(np.float32)
        mask = np.array([True, False, False], dtype=bool)
        agent.episode_buffer.append((state, 0, 1.0, 0.5, mask, True))

    loss_dict = agent.optimize(bootstrap_value=0.0)
    assert loss_dict is not None
    # Entropy should be low (~0) because only one action is valid
    assert loss_dict["entropy"] < 0.5


def test_compute_returns_with_dones():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)

    rewards = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)
    dones = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    returns = agent._compute_returns_with_dones(rewards, dones, 0.0)

    # t=1 has done=1, so return at t=1 should be just r_1 = 1.0
    assert abs(returns[1] - 1.0) < 1e-6
    # t=0: r_0 + gamma * return_1 * (1-done_0) = 1 + 0.95*1 = 1.95
    assert abs(returns[0] - 1.95) < 1e-6
    # t=2,3 processed in reverse after t=1 done reset
    assert abs(returns[3] - 1.0) < 1e-6
    assert abs(returns[2] - 1.95) < 1e-6


# ------------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------------

def test_state_dict_roundtrip():
    mod_reg = ModulationRegistry.from_profile("default")
    agent = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                          mod_registry=mod_reg)

    sd = agent.state_dict()
    assert "policy_net" in sd
    assert "value_net" in sd
    assert "optimizer" in sd
    assert sd["state_dim"] == agent.state_dim
    assert sd["n_actions"] == agent.n_actions

    agent2 = DeepRMSAAgent(num_nodes=14, num_slots=32, k_path=3, m_blocks=1,
                           mod_registry=mod_reg)
    agent2.load_state_dict(sd)
    assert agent2.step_count == agent.step_count
