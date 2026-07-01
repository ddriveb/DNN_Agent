"""Tests for Agent-C DQN agent."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import torch

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.replay_buffer import ReplayBuffer


def _make_env():
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES, num_servers=2, seed=42,
        server_nodes=[0, 3], capacities=[50.0, 50.0],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)
    return env


def _make_request():
    return DNNRequest(
        req_id=0, src_node=1, arrival_time=0.0, holding_time=2.0,
        deadline_ms=100.0,
        splits=[
            SplitProfile(0, 1.0, 0.5, 1.0),
            SplitProfile(1, 2.0, 1.0, 0.5),
            SplitProfile(2, 3.0, 1.5, 0.2),
        ],
    )


# ------------------------------------------------------------------
# 1. Q-network forward shape correct
# ------------------------------------------------------------------

def test_q_network_forward_shape():
    agent = AgentC(input_dim=17, device='cpu')

    # Batch input: (batch=4, num_actions=6, input_dim=17)
    x = torch.randn(4, 6, 17)
    q = agent.q_net(x)
    assert q.shape == (4, 6), f"Expected (4, 6), got {q.shape}"

    # Single input
    x2 = torch.randn(1, 6, 17)
    q2 = agent.q_net(x2)
    assert q2.shape == (1, 6), f"Expected (1, 6), got {q2.shape}"

    print("test_q_network_forward_shape PASSED")


# ------------------------------------------------------------------
# 2. Action-feature construction length stable and dim=17
# ------------------------------------------------------------------

def test_action_feature_dim():
    env = _make_env()
    request = _make_request()
    obs = build_agent_c_observation(env, request)

    agent = AgentC(input_dim=17, device='cpu')
    features, mask = agent.build_action_features(obs)

    num_splits = len(request.splits)
    num_servers = len(env.mec.servers)
    expected_actions = num_splits * num_servers

    assert features.shape[0] == expected_actions, \
        f"Expected {expected_actions} actions, got {features.shape[0]}"
    assert features.shape[1] == 17, \
        f"Expected input_dim=17, got {features.shape[1]}"
    assert len(mask) == expected_actions

    print(f"test_action_feature_dim: actions={features.shape[0]}, dim={features.shape[1]}")
    print("test_action_feature_dim PASSED")


# ------------------------------------------------------------------
# 3. epsilon=0 selects highest-Q valid action
# ------------------------------------------------------------------

def test_epsilon_zero_selects_best():
    env = _make_env()
    request = _make_request()
    obs = build_agent_c_observation(env, request)

    agent = AgentC(input_dim=17, device='cpu')

    # Manually set Q-net weights so that all valid actions have equal high Q
    with torch.no_grad():
        for p in agent.q_net.parameters():
            p.zero_()
        last_bias = agent.q_net.net[-1].bias
        last_bias[0] = 100.0

    action_idx = agent.select_action(obs, epsilon=0.0)

    valid_actions = np.where(obs["agent_c_mask"])[0]
    assert action_idx in valid_actions, \
        f"Selected action {action_idx} not in valid set {valid_actions}"

    print("test_epsilon_zero_selects_best PASSED")


# ------------------------------------------------------------------
# 4. epsilon=1 selects only from valid mask
# ------------------------------------------------------------------

def test_epsilon_one_random_valid():
    env = _make_env()
    request = _make_request()
    obs = build_agent_c_observation(env, request)

    agent = AgentC(input_dim=17, device='cpu')

    valid_actions = set(np.where(obs["agent_c_mask"])[0])
    assert len(valid_actions) > 0, "Need at least one valid action for this test"

    selected = set()
    for _ in range(50):
        a = agent.select_action(obs, epsilon=1.0)
        assert a in valid_actions, f"Selected {a} not in valid {valid_actions}"
        selected.add(a)

    print(f"test_epsilon_one_random_valid: selected {len(selected)} unique actions from {len(valid_actions)} valid")
    print("test_epsilon_one_random_valid PASSED")


# ------------------------------------------------------------------
# 5. All-False mask returns None
# ------------------------------------------------------------------

def test_all_false_mask_returns_none():
    env = _make_env()
    request = _make_request()
    obs = build_agent_c_observation(env, request)

    # Force mask to all False
    obs["agent_c_mask"] = np.zeros_like(obs["agent_c_mask"], dtype=bool)

    agent = AgentC(input_dim=17, device='cpu')

    action = agent.select_action(obs, epsilon=0.0)
    assert action is None, f"Expected None for all-False mask, got {action}"
    print("test_all_false_mask_returns_none PASSED")


# ------------------------------------------------------------------
# 6. optimize_step returns finite loss
# ------------------------------------------------------------------

def test_optimize_step_returns_loss():
    agent = AgentC(input_dim=17, device='cpu')

    # Synthetic batch: 2 transitions, 6 actions each
    batch_size = 2
    num_actions = 6
    obs_features = [np.random.randn(num_actions, 17).astype(np.float32) for _ in range(batch_size)]
    masks = [np.array([True, True, False, True, False, True], dtype=bool)
             for _ in range(batch_size)]
    actions = [0, 3]
    rewards = [1.0, -1.0]
    next_obs_features = [np.random.randn(num_actions, 17).astype(np.float32) for _ in range(batch_size)]
    next_masks = [np.array([True, False, True, True, False, True], dtype=bool)
                  for _ in range(batch_size)]
    dones = [True, True]

    batch = (obs_features, masks, actions, rewards, next_obs_features, next_masks, dones)
    loss = agent.optimize(batch, batch_size)

    assert loss is not None, "Expected a loss value"
    assert isinstance(loss, float), f"Expected float loss, got {type(loss)}"
    assert loss >= 0.0, f"Expected non-negative loss, got {loss}"
    print(f"test_optimize_step_returns_loss: loss={loss:.4f}")
    print("test_optimize_step_returns_loss PASSED")


# ------------------------------------------------------------------
# 7. Invalid action with highest raw Q must still be masked
# ------------------------------------------------------------------

def test_mask_blocks_invalid_high_q():
    """Even if an invalid action has the highest raw Q value,
    select_action(epsilon=0) must pick from valid actions only."""
    env = _make_env()
    request = _make_request()
    obs = build_agent_c_observation(env, request)

    agent = AgentC(input_dim=17, device='cpu')

    features, mask = agent.build_action_features(obs)
    num_actions = len(mask)
    invalid_actions = np.where(~mask)[0]
    valid_actions = np.where(mask)[0]

    if len(invalid_actions) == 0 or len(valid_actions) == 0:
        print("test_mask_blocks_invalid_high_q SKIPPED (need mixed valid/invalid)")
        return

    # Craft fake Q values: one invalid action gets Q=1000 (far above others)
    fake_q = np.random.randn(num_actions).astype(np.float32) * 0.1
    fake_q[int(invalid_actions[0])] = 1000.0

    original_forward = agent.q_net.forward
    def patched_forward(x):
        batch = x.shape[0]
        return torch.tensor(fake_q, dtype=torch.float32).unsqueeze(0).expand(batch, -1)

    agent.q_net.forward = patched_forward
    try:
        selected = agent.select_action(obs, epsilon=0.0)
        assert selected in valid_actions, \
            f"Selected invalid action {selected} (Q=1000) instead of valid {valid_actions}"
        print(f"test_mask_blocks_invalid_high_q: invalid_Q=1000, selected={selected} (valid)")
        print("test_mask_blocks_invalid_high_q PASSED")
    finally:
        agent.q_net.forward = original_forward


def test_r_feasibility_safe_feature():
    env = _make_env()
    request = _make_request()
    obs = build_agent_c_observation(env, request)

    agent = AgentC(input_dim=17, device='cpu', feature_mode="r_feasibility_safe")
    features, mask = agent.build_action_features(obs)

    num_splits = len(request.splits)
    num_servers = len(env.mec.servers)
    expected_actions = num_splits * num_servers

    assert features.shape[0] == expected_actions, \
        f"Expected {expected_actions} actions, got {features.shape[0]}"
    assert features.shape[1] == 29, \
        f"Expected input_dim=29, got {features.shape[1]}"
    assert len(mask) == expected_actions

    # Server margin features should be in [-1, 1]
    for i in range(expected_actions):
        for j in range(27, 29):
            assert -1.0 <= features[i, j] <= 1.0, \
                f"Feature [{i},{j}]={features[i,j]} out of [-1,1]"

    print("test_r_feasibility_safe_feature PASSED")


if __name__ == "__main__":
    test_q_network_forward_shape()
    test_action_feature_dim()
    test_epsilon_zero_selects_best()
    test_epsilon_one_random_valid()
    test_all_false_mask_returns_none()
    test_optimize_step_returns_loss()
    test_mask_blocks_invalid_high_q()
    test_r_feasibility_safe_feature()
    print("\n=== All Agent-C tests PASSED ===")
