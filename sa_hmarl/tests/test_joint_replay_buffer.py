"""Tests for the joint Agent-C / Agent-R replay buffer."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.joint_replay_buffer import JointReplayBuffer
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.network.modulation import ModulationRegistry


def _push_transition(buffer: JointReplayBuffer, idx: int, done: bool = False):
    """Push a small but valid synthetic joint transition."""
    c_actions = 6
    r_actions = 12
    obs_c = np.full((c_actions, 17), idx, dtype=np.float32)
    mask_c = np.array([True, True, False, True, False, True], dtype=bool)
    obs_r = np.full((r_actions, 11), idx + 0.5, dtype=np.float32)
    mask_r = np.array(
        [True, False, True, True, False, True, False, True, False, True, True, False],
        dtype=bool,
    )

    next_obs_c = np.full((c_actions, 17), idx + 1, dtype=np.float32)
    next_mask_c = np.array([True, False, True, True, False, True], dtype=bool)
    next_obs_r = np.full((r_actions, 11), idx + 1.5, dtype=np.float32)
    next_mask_r = np.array(
        [False, True, True, False, True, True, False, True, False, True, False, True],
        dtype=bool,
    )

    buffer.push(
        obs_c_features=obs_c,
        mask_c=mask_c,
        action_c_idx=1,
        obs_r_features=obs_r,
        mask_r=mask_r,
        action_r_idx=2,
        reward_global=0.25 + idx,
        reward_c=0.5 + idx,
        reward_r=0.75 + idx,
        next_obs_c_features=next_obs_c,
        next_mask_c=next_mask_c,
        next_obs_r_features=next_obs_r,
        next_mask_r=next_mask_r,
        done=done,
        info={"req_id": idx, "success": True},
    )


def test_push_and_sample_complete_transition():
    buffer = JointReplayBuffer(capacity=4, seed=1)
    _push_transition(buffer, 0)

    batch = buffer.sample(1)
    assert batch is not None
    transition = batch[0]

    assert transition.obs_c_features.shape == (6, 17)
    assert transition.obs_r_features.shape == (12, 11)
    assert transition.action_c_idx == 1
    assert transition.action_r_idx == 2
    assert transition.reward_global == 0.25
    assert transition.reward_c == 0.5
    assert transition.reward_r == 0.75
    assert transition.info["req_id"] == 0

    print("test_push_and_sample_complete_transition PASSED")


def test_capacity_overwrite():
    buffer = JointReplayBuffer(capacity=2, seed=2)
    _push_transition(buffer, 0)
    _push_transition(buffer, 1)
    _push_transition(buffer, 2)

    assert len(buffer) == 2
    batch = buffer.sample(2)
    assert batch is not None
    req_ids = sorted(t.info["req_id"] for t in batch)
    assert req_ids == [1, 2], f"Expected newest req_ids [1, 2], got {req_ids}"

    print("test_capacity_overwrite PASSED")


def test_agent_c_batch_compatible_with_optimize():
    buffer = JointReplayBuffer(capacity=8, seed=3)
    for i in range(4):
        _push_transition(buffer, i, done=(i == 3))

    agent = AgentC(input_dim=17, device="cpu")
    batch = buffer.sample_agent_c(batch_size=2)
    loss = agent.optimize(batch, batch_size=2)

    assert loss is not None
    assert np.isfinite(loss)

    print(f"test_agent_c_batch_compatible_with_optimize: loss={loss:.4f}")
    print("test_agent_c_batch_compatible_with_optimize PASSED")


def test_agent_r_batch_compatible_with_optimize():
    buffer = JointReplayBuffer(capacity=8, seed=4)
    for i in range(4):
        _push_transition(buffer, i, done=(i == 3))

    agent = AgentR(input_dim=11, mod_registry=ModulationRegistry(), device="cpu")
    batch = buffer.sample_agent_r(batch_size=2)
    loss = agent.optimize(batch, batch_size=2)

    assert loss is not None
    assert np.isfinite(loss)

    print(f"test_agent_r_batch_compatible_with_optimize: loss={loss:.4f}")
    print("test_agent_r_batch_compatible_with_optimize PASSED")


def test_inserted_arrays_are_copied():
    buffer = JointReplayBuffer(capacity=2, seed=5)
    obs_c = np.ones((6, 17), dtype=np.float32)
    mask_c = np.array([True, True, False, True, False, True], dtype=bool)
    obs_r = np.ones((12, 11), dtype=np.float32)
    mask_r = np.array([True] * 12, dtype=bool)

    buffer.push(
        obs_c, mask_c, 0,
        obs_r, mask_r, 0,
        1.0, 1.0, 1.0,
        obs_c, mask_c,
        obs_r, mask_r,
        False,
        {"req_id": 7},
    )

    obs_c[:] = 99.0
    mask_r[:] = False
    transition = buffer.sample(1)[0]  # type: ignore[index]

    assert float(transition.obs_c_features[0, 0]) == 1.0
    assert np.any(transition.mask_r)

    print("test_inserted_arrays_are_copied PASSED")


if __name__ == "__main__":
    test_push_and_sample_complete_transition()
    test_capacity_overwrite()
    test_agent_c_batch_compatible_with_optimize()
    test_agent_r_batch_compatible_with_optimize()
    test_inserted_arrays_are_copied()
    print("\n=== All JointReplayBuffer tests PASSED ===")
