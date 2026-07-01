import numpy as np

from sa_hmarl.env.request_demand_mean_field import RequestDemandMeanField
from sa_hmarl.training.train_joint_mappo import CentralizedValueCritic
from sa_hmarl.training.utils import generate_requests


def _requests(seed=42, count=8):
    return generate_requests(
        env=None,
        rng=np.random.RandomState(seed),
        src_node=0,
        num_requests=count,
        deadline_min=30.0,
        deadline_max=100.0,
        size_min_mb=5.0,
        size_max_mb=30.0,
        num_splits=3,
        split_profile="default3",
        traffic_mode="markov_regime",
    )


def test_request_mean_field_starts_empty_and_normalizes_groups():
    tracker = RequestDemandMeanField.from_config(4, 5.0, 30.0, 30.0, 100.0, "default3")
    assert np.array_equal(tracker.vector(), np.zeros(6, dtype=np.float32))
    tracker.observe(_requests(count=1)[0])
    vector = tracker.vector()
    assert np.isclose(vector[:3].sum(), 1.0)
    assert np.isclose(vector[3:].sum(), 1.0)


def test_request_mean_field_window_evicts_old_requests():
    requests = _requests(count=5)
    tracker = RequestDemandMeanField.from_config(2, 5.0, 30.0, 30.0, 100.0, "default3")
    for request in requests:
        tracker.observe(request)
    expected = RequestDemandMeanField.from_config(2, 5.0, 30.0, 30.0, 100.0, "default3")
    expected.observe(requests[-2])
    expected.observe(requests[-1])
    assert len(tracker) == 2
    assert np.array_equal(tracker.vector(), expected.vector())


def test_critic_appends_context_without_changing_actor_features():
    critic = CentralizedValueCritic(c_dim=3, r_dim=2, global_dim=14, context_dim=6)
    c_features = np.ones((2, 3), dtype=np.float32)
    r_features = np.ones((2, 2), dtype=np.float32)
    mask = np.ones(2, dtype=bool)
    context = np.arange(6, dtype=np.float32) / 5.0
    encoded = critic.encode(
        c_features, mask, r_features, mask, env=None, extra_context=context
    )
    assert critic.input_dim == 25
    assert encoded.shape == (25,)
    assert np.array_equal(encoded[-6:], context)
    assert c_features.shape == (2, 3)


def test_critic_rejects_wrong_context_dimension():
    critic = CentralizedValueCritic(c_dim=3, r_dim=2, context_dim=6)
    with np.testing.assert_raises(ValueError):
        critic.encode(
            np.ones((1, 3), dtype=np.float32),
            np.ones(1, dtype=bool),
            np.ones((1, 2), dtype=np.float32),
            np.ones(1, dtype=bool),
            extra_context=np.ones(5, dtype=np.float32),
        )
