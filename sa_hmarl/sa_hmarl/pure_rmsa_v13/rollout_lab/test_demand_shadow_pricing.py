from __future__ import annotations

import numpy as np

from sa_hmarl.pds_rmsa.baselines.ksp_ff import ksp_ff_action
from sa_hmarl.pds_rmsa.env.rmsa_env import RMSAAction
from sa_hmarl.pure_rmsa_v13.core import execute
from sa_hmarl.pure_rmsa_v13.rollout_lab.demand_shadow_pricing import (
    DemandShadowPricer,
    _destroyed_windows,
    build_diverse_shadow_pool,
    demand_shadow_action,
    demand_shadow_multipath_action,
    rank_demand_shadow_actions,
)
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env


def test_destroyed_windows_matches_bruteforce() -> None:
    run_left, run_right = 2, 12
    start, allocated = 5, 3
    for future_width in range(1, 11):
        windows = [
            set(range(left, left + future_width))
            for left in range(run_left, run_right - future_width + 1)
        ]
        occupied = set(range(start, start + allocated))
        expected = sum(bool(window & occupied) for window in windows)
        assert (
            _destroyed_windows(
                run_left,
                run_right,
                start,
                allocated,
                future_width,
            )
            == expected
        )


def test_shadow_action_preserves_ksp_route_and_does_not_mutate() -> None:
    env = make_env(
        42,
        100,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    pricer = DemandShadowPricer(env)
    assert np.all(pricer.demand_weights >= 0)
    for request in env.trace.requests[:20]:
        env.advance_external(request)
        before = env.state_fingerprint()
        ff = ksp_ff_action(env, request)
        shadow = demand_shadow_action(env, request, pricer)
        assert env.state_fingerprint() == before
        assert isinstance(ff, RMSAAction) == isinstance(shadow, RMSAAction)
        if isinstance(ff, RMSAAction):
            assert shadow.path == ff.path
            assert shadow.path_rank == ff.path_rank
            assert shadow.modulation == ff.modulation
            assert shadow.required_fs == ff.required_fs
            available = env.spectrum.get_available_slots(shadow.path)
            assert np.all(
                available[
                    shadow.start_slot : shadow.start_slot + shadow.required_fs
                ]
            )
            assert execute(env, request, shadow)["success"]
        else:
            break


def test_one_path_multipath_selector_matches_locked_selector() -> None:
    env = make_env(
        43,
        100,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    pricer = DemandShadowPricer(env)
    for request in env.trace.requests[:30]:
        env.advance_external(request)
        locked = demand_shadow_action(env, request, pricer)
        experimental = demand_shadow_multipath_action(
            env,
            request,
            pricer,
            max_feasible_paths=1,
            path_penalty=0.0,
        )
        assert getattr(locked, "action_id", None) == getattr(
            experimental, "action_id", None
        )
        assert execute(env, request, locked)["success"] == isinstance(
            locked, RMSAAction
        )


def test_ranked_shadow_top1_matches_confirmed_multipath_selector() -> None:
    env = make_env(
        44,
        100,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    pricer = DemandShadowPricer(env)
    for request in env.trace.requests[:30]:
        env.advance_external(request)
        selected = demand_shadow_multipath_action(
            env,
            request,
            pricer,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
        ranked = rank_demand_shadow_actions(
            env,
            request,
            pricer,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
        if isinstance(selected, RMSAAction):
            assert ranked
            assert ranked[0][1].action_id == selected.action_id
        else:
            assert not ranked
        execute(env, request, selected)


def test_diverse_pool_preserves_top1_and_unique_actions() -> None:
    env = make_env(
        45,
        100,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    pricer = DemandShadowPricer(env)
    for request in env.trace.requests[:30]:
        env.advance_external(request)
        selected = demand_shadow_multipath_action(
            env,
            request,
            pricer,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
        pool = build_diverse_shadow_pool(
            env, request, pricer, top_n=10
        )
        if isinstance(selected, RMSAAction):
            assert pool[0][1].action_id == selected.action_id
            assert len({action.action_id for _, action in pool}) == len(pool)
            assert len(pool) <= 10
        else:
            assert not pool
        execute(env, request, selected)


def test_batch_marginal_losses_match_scalar_oracle() -> None:
    env = make_env(
        46,
        100,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    pricer = DemandShadowPricer(env)
    for request in env.trace.requests[:10]:
        env.advance_external(request)
        ranked = rank_demand_shadow_actions(
            env,
            request,
            pricer,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
        by_path = {}
        for _, action in ranked:
            by_path.setdefault(action.path, []).append(action)
        for path, actions in by_path.items():
            starts = np.asarray(
                [action.start_slot for action in actions], dtype=np.int16
            )
            actual = pricer.marginal_losses(
                env, path, starts, actions[0].required_fs
            )
            expected = np.asarray(
                [
                    pricer.marginal_loss(
                        env,
                        path,
                        action.start_slot,
                        action.required_fs,
                    )
                    for action in actions
                ]
            )
            assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
        selected = demand_shadow_multipath_action(
            env,
            request,
            pricer,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
        execute(env, request, selected)
