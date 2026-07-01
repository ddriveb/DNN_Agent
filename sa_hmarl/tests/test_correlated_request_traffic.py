import numpy as np

from sa_hmarl.training.utils import generate_requests


COMMON = dict(
    env=None,
    src_node=0,
    num_requests=120,
    deadline_min=30.0,
    deadline_max=100.0,
    size_min_mb=5.0,
    size_max_mb=30.0,
    edge_cost_min=0.5,
    edge_cost_max=15.0,
    num_splits=3,
    split_profile="default3",
)


def _signature(requests):
    return np.asarray(
        [
            (
                np.mean([split.intermediate_size_mb for split in request.splits]),
                request.deadline_ms,
            )
            for request in requests
        ]
    )


def test_default_traffic_mode_preserves_iid_sequence():
    implicit = generate_requests(rng=np.random.RandomState(42), **COMMON)
    explicit = generate_requests(
        rng=np.random.RandomState(42), traffic_mode="iid", **COMMON
    )
    assert np.array_equal(_signature(implicit), _signature(explicit))


def test_markov_regime_is_reproducible():
    first = generate_requests(
        rng=np.random.RandomState(123),
        traffic_mode="markov_regime",
        regime_stay_prob=0.9,
        **COMMON,
    )
    second = generate_requests(
        rng=np.random.RandomState(123),
        traffic_mode="markov_regime",
        regime_stay_prob=0.9,
        **COMMON,
    )
    assert np.array_equal(_signature(first), _signature(second))


def test_markov_regime_adds_lag_one_persistence():
    iid = _signature(generate_requests(rng=np.random.RandomState(7), **COMMON))
    correlated = _signature(
        generate_requests(
            rng=np.random.RandomState(7),
            traffic_mode="markov_regime",
            regime_stay_prob=0.95,
            **COMMON,
        )
    )
    iid_corr = np.corrcoef(iid[:-1, 0], iid[1:, 0])[0, 1]
    correlated_corr = np.corrcoef(correlated[:-1, 0], correlated[1:, 0])[0, 1]
    assert correlated_corr > iid_corr + 0.4


def test_invalid_traffic_arguments_are_rejected():
    with np.testing.assert_raises(ValueError):
        generate_requests(
            rng=np.random.RandomState(1), traffic_mode="unknown", **COMMON
        )
    with np.testing.assert_raises(ValueError):
        generate_requests(
            rng=np.random.RandomState(1),
            traffic_mode="markov_regime",
            regime_stay_prob=1.1,
            **COMMON,
        )
