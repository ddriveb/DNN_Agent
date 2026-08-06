"""traffic.py tests: truncation, rejection sampling, reproducibility."""
from __future__ import annotations

import numpy as np
import pytest

from sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.traffic import (
    MEAN_HOLDING_TIME,
    TRUNCATION_FACTOR,
    _holding_rejection_sampling,
    generate_trace,
)


def test_all_holding_times_below_cap():
    t = generate_trace("t", 14, 5000, 250, 69301, num_slots=100)
    hs = [r.holding_time for r in t.requests]
    assert max(hs) < TRUNCATION_FACTOR * MEAN_HOLDING_TIME


def test_truncated_mean_lower_than_untruncated():
    t1 = generate_trace("t", 14, 20000, 250, 69301, num_slots=100,
                        truncate_holding_time=False)
    t2 = generate_trace("t", 14, 20000, 250, 69301, num_slots=100,
                        truncate_holding_time=True)
    h1 = np.mean([r.holding_time for r in t1.requests])
    h2 = np.mean([r.holding_time for r in t2.requests])
    # truncated mean ~0.69x untruncated (XLRON/DeepRMSA ~31% reduction)
    assert h2 / h1 < 0.75
    assert h2 / h1 > 0.60


def test_rejection_sampling_distribution():
    """First-nonzero-of-5 with cap at 2*mean: CDF matches theory.
    P(H > x) for x < 2*mean equals the exponential survival conditioned
    on H < 2*mean, i.e. (e^-x/m - e^-2)/(1 - e^-2)."""
    rng = np.random.RandomState(0)
    samples = np.array([_holding_rejection_sampling(rng)
                        for _ in range(200000)])
    for x in (0.5, 1.0, 1.5 * MEAN_HOLDING_TIME):
        emp = np.mean(samples > x)
        theory = (np.exp(-x / MEAN_HOLDING_TIME) - np.exp(-2.0)) / (
            1.0 - np.exp(-2.0))
        assert abs(emp - theory) < 0.01


def test_reproducible_trace():
    t1 = generate_trace("t", 14, 2000, 250, 69301, num_slots=100)
    t2 = generate_trace("t", 14, 2000, 250, 69301, num_slots=100)
    r1 = [(r.arrival_time, r.holding_time, r.src_node, r.dst_node,
           r.bitrate_gbps) for r in t1.requests]
    r2 = [(r.arrival_time, r.holding_time, r.src_node, r.dst_node,
           r.bitrate_gbps) for r in t2.requests]
    assert r1 == r2


def test_bitrate_range_uniform():
    t = generate_trace("t", 14, 5000, 250, 69302, num_slots=100)
    bw = np.array([r.bitrate_gbps for r in t.requests])
    assert bw.min() >= 25 and bw.max() <= 100
    assert len(set(bw.tolist())) > 60  # 25..100 step 1 -> ~76 distinct


def test_arrival_rate():
    t = generate_trace("t", 14, 100000, 250, 69303, num_slots=100)
    span = t.requests[-1].arrival_time
    # expected span = n / arrival_rate = n * mean / load
    expected = 100000 * MEAN_HOLDING_TIME / 250.0
    assert abs(span - expected) / expected < 0.02
