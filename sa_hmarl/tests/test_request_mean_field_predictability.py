import numpy as np

from sa_hmarl.evaluation.diagnose_request_mean_field_predictability import (
    _generate_episode,
    evaluate_predictability,
    fit_thresholds,
    request_mean_field,
)


def test_request_mean_field_marginals_sum_to_one():
    episode = _generate_episode(42, 30)
    thresholds = fit_thresholds([episode])
    vector = request_mean_field(episode, thresholds)
    assert vector.shape == (6,)
    assert np.isclose(vector[:3].sum(), 1.0)
    assert np.isclose(vector[3:].sum(), 1.0)


def test_constant_sequence_is_predictable():
    source = _generate_episode(7, 1, size_range=(5.0, 5.001), deadline_range=(99.0, 99.001))[0]
    episode = [source] * 40
    thresholds = {"size_mb": [10.0, 20.0], "deadline_ms": [40.0, 70.0]}
    prior = np.asarray([0.0, 0.0, 1.0, 1.0, 0.0, 0.0])
    metrics = evaluate_predictability([episode], thresholds, prior, 10, 5)
    assert metrics["rolling_mae"] == 0.0
    assert metrics["forecast_skill"] == 1.0
