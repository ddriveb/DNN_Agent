"""Smoke tests for CTDE centralized critic.

Run:
    PYTHONPATH=sa_hmarl .venv/bin/python sa_hmarl/tests/test_centralized_critic.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import math
import numpy as np

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.centralized_critic import CentralizedCritic
from sa_hmarl.agents.joint_replay_buffer import JointReplayBuffer
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.network.modulation import ModulationRegistry


def _make_transition(buffer, action_c=1, action_r=2, done=False):
    rng = np.random.RandomState(7 + action_c + action_r)
    obs_c = rng.rand(6, 17).astype(np.float32)
    mask_c = np.array([True, True, False, True, False, True], dtype=bool)
    obs_r = rng.rand(12, 11).astype(np.float32)
    mask_r = np.array(
        [True, False, True, True, False, True, True, False, True, True, False, True],
        dtype=bool,
    )
    next_obs_c = rng.rand(5, 17).astype(np.float32)
    next_mask_c = np.array([True, False, True, True, False], dtype=bool)
    next_obs_r = rng.rand(9, 11).astype(np.float32)
    next_mask_r = np.array([True, True, False, True, False, True, True, False, True], dtype=bool)

    buffer.push(
        obs_c_features=obs_c,
        mask_c=mask_c,
        action_c_idx=action_c,
        obs_r_features=obs_r,
        mask_r=mask_r,
        action_r_idx=action_r,
        reward_global=0.7,
        reward_c=0.3,
        reward_r=0.4,
        next_obs_c_features=next_obs_c,
        next_mask_c=next_mask_c,
        next_obs_r_features=next_obs_r,
        next_mask_r=next_mask_r,
        done=done,
        info={"success": not done},
    )


def test_centralized_critic_optimize_and_distill():
    agent_c = AgentC(input_dim=17, hidden_dims=(32,), device="cpu")
    agent_r = AgentR(
        input_dim=11,
        mod_registry=ModulationRegistry(),
        hidden_dims=(32,),
        device="cpu",
    )
    critic = CentralizedCritic(c_dim=17, r_dim=11, hidden_dims=(64,), device="cpu")
    buffer = JointReplayBuffer(capacity=8, seed=3)

    _make_transition(buffer, action_c=1, action_r=2, done=False)
    _make_transition(buffer, action_c=3, action_r=5, done=True)

    batch = buffer.sample(2)
    loss = critic.optimize(batch, agent_c, agent_r)
    assert loss is not None and math.isfinite(loss)

    distill_loss = critic.distill_agents(batch, agent_c, agent_r, coef=0.01)
    assert distill_loss is not None and math.isfinite(distill_loss)
    assert critic.input_dim == 56

    print("test_centralized_critic_optimize_and_distill PASSED")


def test_centralized_critic_handles_no_next_action():
    agent_c = AgentC(input_dim=17, hidden_dims=(32,), device="cpu")
    agent_r = AgentR(
        input_dim=11,
        mod_registry=ModulationRegistry(),
        hidden_dims=(32,),
        device="cpu",
    )
    critic = CentralizedCritic(c_dim=17, r_dim=11, hidden_dims=(64,), device="cpu")
    buffer = JointReplayBuffer(capacity=4, seed=4)
    _make_transition(buffer, action_c=1, action_r=2, done=False)

    transition = buffer.sample(1)[0]
    transition.next_mask_c[:] = False
    transition.next_mask_r[:] = False
    loss = critic.optimize([transition], agent_c, agent_r)
    assert loss is not None and math.isfinite(loss)

    print("test_centralized_critic_handles_no_next_action PASSED")


if __name__ == "__main__":
    test_centralized_critic_optimize_and_distill()
    test_centralized_critic_handles_no_next_action()
    print("\n=== All centralized critic smoke tests PASSED ===")
