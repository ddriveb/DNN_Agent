"""ksp_ff.py tests: baseline behaviour, determinism, edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.env import RMSAEnv
from sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.ksp_ff import (
    first_fit_starts,
    ksp_ff,
    run_episode,
)
from sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.traffic import generate_trace


def _env(k=5, seed=69301, n_req=3000):
    t = generate_trace("xlron_nsfnet_deeprmsa", 14, n_req, 250, seed,
                       num_slots=100)
    return RMSAEnv(t, k_paths=k, topology_name="xlron_nsfnet_deeprmsa")


def test_first_fit_lowest_start():
    env = _env()
    env.next_request()
    req = env.current_request
    pas = env.path_actions(req)
    starts = first_fit_starts(env, pas)
    for i, pa in enumerate(pas):
        m = env.path_mask(pa["path"], pa["required_slots"])
        if m.any():
            assert starts[i] == np.flatnonzero(m)[0]
        else:
            assert starts[i] == env.num_slots


def test_ksp_ff_first_feasible_path():
    env = _env()
    env.next_request()
    req = env.current_request
    chosen = ksp_ff(env, req)
    assert chosen is not None
    path, start, ns = chosen
    pas = env.path_actions(req)
    # chosen path must be the first feasible in KSP order
    for pa in pas:
        m = env.path_mask(pa["path"], pa["required_slots"])
        if pa["path"] == path:
            assert m.any()
            break
        assert not m.any(), "an earlier path was feasible"


def test_deterministic_episode():
    r1 = run_episode(_env(k=5), warmup=3000, eval_requests=2000)
    r2 = run_episode(_env(k=5), warmup=3000, eval_requests=2000)
    assert r1 == r2


def test_blocked_when_saturated():
    env = _env()
    env.link_slot_array[:] = 1  # saturate everything
    env.next_request()
    assert ksp_ff(env, env.current_request) is None


def test_k5_vs_k50_episodes_ran():
    """Both K values run cleanly on a trace long enough for warmup+eval."""
    for k in (5, 50):
        env = _env(k=k, n_req=7000)
        r = run_episode(env, warmup=3000, eval_requests=3000)
        assert r["total"] == 3000
        assert 0.0 <= r["blocked_rate"] <= 100.0
