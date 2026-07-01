import numpy as np

from sa_hmarl.evaluation.diagnose_candidate_horizon_impact import (
    HorizonOutcome,
    _average_group_rank_correlation,
    _cross_features,
    _make_recommendation,
    _ridge_predict,
    _summarize_decisions,
)


def test_oracle_key_prioritizes_future_blocking_after_current_success():
    better = HorizonOutcome(True, 0, 1, 0, 0, 5, 2.0, 1.0, 2.0)
    worse = HorizonOutcome(True, 1, 0, 0, 0, 5, 4.0, 3.0, 4.0)
    assert better.oracle_key() < worse.oracle_key()


def test_cross_features_include_candidate_mean_field_interactions():
    base = np.asarray([[2.0, 3.0]])
    mean_field = np.asarray([[5.0, 7.0]])
    result = _cross_features(base, mean_field)
    assert result.shape == (1, 8)
    np.testing.assert_allclose(result[0, -4:], [10.0, 14.0, 15.0, 21.0])


def test_ridge_predict_recovers_simple_linear_signal():
    x = np.arange(20, dtype=np.float64).reshape(-1, 1)
    y = 1.0 + 2.0 * x[:, 0]
    prediction = _ridge_predict(x, y, x, ridge=1e-8)
    np.testing.assert_allclose(prediction, y, atol=1e-5)


def test_group_rank_correlation_is_one_for_matching_rankings():
    truth = np.asarray([0.0, 1.0, 2.0, 4.0, 3.0])
    pred = truth * 2.0
    groups = np.asarray([0, 0, 0, 1, 1])
    assert np.isclose(_average_group_rank_correlation(truth, pred, groups), 1.0)


def test_decision_summary_reports_oracle_headroom():
    summary = _summarize_decisions([
        {
            "n_candidates": 3,
            "blocked_range": 2,
            "collapse_range": 1,
            "reference_blocked_rate": 0.4,
            "oracle_blocked_rate": 0.0,
            "reference_collapse_rate": 0.2,
            "oracle_collapse_rate": 0.0,
        }
    ])
    assert np.isclose(summary["oracle_blocking_headroom_pp"], 40.0)
    assert summary["fraction_with_blocked_difference"] == 1.0


def test_recommendation_stops_when_headroom_and_mf_increment_are_small():
    recommendation = _make_recommendation(
        {
            "oracle_blocking_headroom_pp": 0.4,
            "fraction_with_blocked_difference": 0.05,
        },
        {
            "aggregate": {
                "candidate_resource": {
                    "mean_within_request_rank_correlation": 0.22,
                },
                "candidate_conditioned_mf": {
                    "mean_within_request_rank_correlation": 0.07,
                },
                "candidate_conditioned_shuffled_mf": {
                    "mean_within_request_rank_correlation": 0.26,
                },
            }
        },
    )
    assert recommendation["verdict"] == "STOP_BEFORE_PREDICTOR"
