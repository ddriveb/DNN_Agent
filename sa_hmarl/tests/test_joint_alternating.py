"""Smoke tests for joint alternating co-training (Agent-C + Agent-R).

Run:
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/tests/test_joint_alternating.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.agents.joint_replay_buffer import JointReplayBuffer, JointTransition
from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.training.utils import compute_agent_c_reward, compute_reward


def _make_env():
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES,
        num_servers=2,
        seed=42,
        server_nodes=[0, 3],
        capacities=[50.0, 50.0],
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)
    return env


def _make_requests(env, num=10):
    rng = np.random.RandomState(42)
    src = 1
    requests = []
    for i in range(num):
        splits = [
            SplitProfile(0, 1.0, 0.5, 1.0),
            SplitProfile(1, 2.0, 1.0, 0.5),
            SplitProfile(2, 3.0, 1.5, 0.2),
        ]
        req = DNNRequest(
            req_id=i,
            src_node=src,
            arrival_time=i * 0.25,
            holding_time=rng.uniform(4.0, 10.0),
            deadline_ms=rng.uniform(30.0, 100.0),
            splits=splits,
        )
        requests.append(req)
    return requests


def test_joint_training_5_episodes():
    """Smoke test: run 5 episodes of joint alternating training without crashing."""
    env = _make_env()
    mod_reg = ModulationRegistry()
    agent_c = AgentC(
        input_dim=17, hidden_dims=(128, 64), gamma=0.95, epsilon=0.3, device="cpu"
    )
    agent_r = AgentR(
        input_dim=11,
        mod_registry=mod_reg,
        hidden_dims=(128, 64),
        gamma=0.95,
        epsilon=0.3,
        device="cpu",
    )
    replay_buffer = JointReplayBuffer(capacity=1000, seed=42)
    rng = np.random.RandomState(42)

    num_episodes = 5
    requests_per_episode = 8

    for episode in range(num_episodes):
        update_c = episode % 2 == 0
        for p in agent_c.q_net.parameters():
            p.requires_grad = update_c
        for p in agent_r.q_net.parameters():
            p.requires_grad = not update_c

        src = rng.randint(0, env.net.NUM_NODES)
        requests = _make_requests(env, requests_per_episode)
        env.reset(requests)

        for t, req in enumerate(requests):
            obs_c = build_agent_c_observation(env, req)
            action_features_c, mask_c = agent_c.build_action_features(obs_c)
            action_idx_c = agent_c.select_action(obs_c, epsilon=0.0)
            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, len(env.mec.servers))

            split_id, server_id = action_c
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_features_r, mask_r = agent_r.build_action_features(obs_r)
            action_idx_r = agent_r.select_action(obs_r, epsilon=0.0)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(
                    action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                )

            _, _, done, info = env.step(action_c, action_r)
            assert isinstance(done, bool)
            assert isinstance(info, dict)

            reward_c = compute_agent_c_reward(
                info, req.deadline_ms, 0.8, env.mec.servers[server_id].utilization
            )
            reward_r = compute_reward(info, 0.8) + (
                0.2 if info.get("success") else -0.2
            )

            if t + 1 < len(requests):
                next_req = requests[t + 1]
                next_obs_c = build_agent_c_observation(env, next_req)
                next_action_features_c, next_mask_c = agent_c.build_action_features(
                    next_obs_c
                )
                next_action_idx_c = agent_c.select_action(next_obs_c, epsilon=0.0)
                if next_action_idx_c is None:
                    next_split_id, next_server_id = 0, 0
                else:
                    next_split_id, next_server_id = decode_agent_c_action(
                        next_action_idx_c, len(env.mec.servers)
                    )
                next_obs_r = build_agent_r_observation(
                    env, next_req, next_split_id, next_server_id
                )
                next_action_features_r, next_mask_r = agent_r.build_action_features(
                    next_obs_r
                )
                transition_done = False
            else:
                next_action_features_c = np.zeros_like(action_features_c)
                next_mask_c = np.zeros_like(mask_c)
                next_action_features_r = np.zeros_like(action_features_r)
                next_mask_r = np.zeros_like(mask_r)
                transition_done = True

            replay_buffer.push(
                obs_c_features=action_features_c,
                mask_c=mask_c,
                action_c_idx=action_idx_c if action_idx_c is not None else 0,
                obs_r_features=action_features_r,
                mask_r=mask_r,
                action_r_idx=action_idx_r if action_idx_r is not None else 0,
                reward_global=reward_c + reward_r,
                reward_c=reward_c,
                reward_r=reward_r,
                next_obs_c_features=next_action_features_c,
                next_mask_c=next_mask_c,
                next_obs_r_features=next_action_features_r,
                next_mask_r=next_mask_r,
                done=transition_done,
                info=info,
            )

            if len(replay_buffer) >= 4 and t % 2 == 0:
                if update_c:
                    batch = replay_buffer.sample_agent_c(2)
                    if batch is not None:
                        agent_c.optimize(batch, 2)
                else:
                    batch = replay_buffer.sample_agent_r(2)
                    if batch is not None:
                        agent_r.optimize(batch, 2)

    assert len(replay_buffer) > 0, "Replay buffer should contain transitions"
    print(f"test_joint_training_5_episodes PASSED (buffer size={len(replay_buffer)})")


def test_replay_buffer_transition_fields_complete():
    """Verify every JointTransition has all required fields."""
    buffer = JointReplayBuffer(capacity=4, seed=1)
    obs_c = np.ones((6, 17), dtype=np.float32)
    mask_c = np.ones(6, dtype=bool)
    obs_r = np.ones((12, 11), dtype=np.float32)
    mask_r = np.ones(12, dtype=bool)
    next_obs_c = np.zeros((6, 17), dtype=np.float32)
    next_mask_c = np.zeros(6, dtype=bool)
    next_obs_r = np.zeros((12, 11), dtype=np.float32)
    next_mask_r = np.zeros(12, dtype=bool)

    buffer.push(
        obs_c,
        mask_c,
        0,
        obs_r,
        mask_r,
        0,
        0.5,
        0.3,
        0.2,
        next_obs_c,
        next_mask_c,
        next_obs_r,
        next_mask_r,
        False,
        {"success": True},
    )

    assert len(buffer) == 1
    batch = buffer.sample(1)
    assert batch is not None
    transition = batch[0]
    assert isinstance(transition, JointTransition)

    # Check all expected fields exist
    required_attrs = [
        "obs_c_features",
        "mask_c",
        "action_c_idx",
        "obs_r_features",
        "mask_r",
        "action_r_idx",
        "reward_c",
        "reward_r",
        "next_obs_c_features",
        "next_obs_r_features",
        "next_mask_c",
        "next_mask_r",
        "done",
        "info",
    ]
    for attr in required_attrs:
        assert hasattr(transition, attr), f"Missing attribute: {attr}"

    print("test_replay_buffer_transition_fields_complete PASSED")


def test_all_false_mask_does_not_crash():
    """When agent_c_mask or agent_r_mask is all False, the pipeline must not crash."""
    env = _make_env()
    mod_reg = ModulationRegistry()
    agent_c = AgentC(input_dim=17, device="cpu")
    agent_r = AgentR(input_dim=11, mod_registry=mod_reg, device="cpu")

    requests = _make_requests(env, num=3)
    env.reset(requests)

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        obs_c["agent_c_mask"] = np.zeros_like(obs_c["agent_c_mask"], dtype=bool)
        action_idx_c = agent_c.select_action(obs_c, epsilon=0.0)
        assert action_idx_c is None
        action_c = (0, 0)

        split_id, server_id = action_c
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        obs_r["agent_r_mask"] = np.zeros_like(obs_r["agent_r_mask"], dtype=bool)
        action_idx_r = agent_r.select_action(obs_r, epsilon=0.0)
        assert action_idx_r is None
        action_r = (0, 0, 0)

        _, _, done, info = env.step(action_c, action_r)
        assert isinstance(done, bool)
        assert isinstance(info, dict)

    print("test_all_false_mask_does_not_crash PASSED")


if __name__ == "__main__":
    test_joint_training_5_episodes()
    test_replay_buffer_transition_fields_complete()
    test_all_false_mask_does_not_crash()
    print("\n=== All joint alternating smoke tests PASSED ===")
