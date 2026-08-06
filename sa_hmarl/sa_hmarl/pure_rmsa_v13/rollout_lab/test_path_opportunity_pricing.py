import numpy as np

from sa_hmarl.pds_rmsa.baselines.ksp_ff import ksp_ff_action
from sa_hmarl.pds_rmsa.env.rmsa_env import RMSAAction
from sa_hmarl.pure_rmsa_v13.core import execute
from sa_hmarl.pure_rmsa_v13.rollout_lab.bitparallel_path_opportunity import (
    BitParallelPathOpportunityPricer,
    bitparallel_path_opportunity_action,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.block_heuristics import (
    ksp_best_fit_action,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import (
    CompiledKSPFFSelector,
    PythonKSPFFSelector,
    _ksp_first_fit_python,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.demand_shadow_pricing import (
    DemandShadowPricer,
    _free_runs,
    _window_count,
    build_diverse_shadow_pool,
    demand_shadow_multipath_action,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.feasible_window_sketch import (
    DirectCompressedSketchSelector,
    ExactOptimizedDirectCompressedSketchSelector,
    _legal_starts_from_word,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.incremental_shadow_pricing import (
    IncrementalDemandShadowPricer,
    _build_price_geometry,
    _build_price_geometry_from_word,
    _build_price_row_from_geometry,
    _build_price_row_python,
    _build_price_rows_python,
    _free_runs_from_occupied_word,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.path_opportunity_pricing import (
    PathOpportunityPricer,
    path_opportunity_action,
)
from sa_hmarl.pure_rmsa_v13.train_proposer import make_env


def _probe_windows(env, probe) -> int:
    available = env.spectrum.get_available_slots(probe.path)
    return sum(
        _window_count(right - left, probe.width)
        for left, right in _free_runs(available)
    )


def test_zero_weight_exactly_preserves_shadow_top1() -> None:
    env = make_env(
        8101,
        100,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    shadow = DemandShadowPricer(env)
    opportunity = PathOpportunityPricer(env)
    for request in env.trace.requests[:30]:
        env.advance_external(request)
        expected = demand_shadow_multipath_action(
            env,
            request,
            shadow,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
        actual = path_opportunity_action(
            env,
            request,
            shadow,
            opportunity,
            top_n=12,
            opportunity_weight=0.0,
        )
        assert actual == expected
        execute(env, request, expected)


def test_path_opportunity_matches_temporary_allocation() -> None:
    env = make_env(
        8101,
        50,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    shadow = DemandShadowPricer(env)
    opportunity = PathOpportunityPricer(env, route_depth=2)
    request = env.trace.requests[0]
    env.advance_external(request)
    action = demand_shadow_multipath_action(
        env,
        request,
        shadow,
        max_feasible_paths=3,
        path_penalty=1.0,
    )
    fingerprint = env.state_fingerprint()
    score = opportunity.score_actions(env, [action])[0]
    action_arcs = frozenset(zip(action.path[:-1], action.path[1:]))
    before = []
    for probe in opportunity._probes:
        if probe.arcs & action_arcs:
            before.append((probe, _probe_windows(env, probe)))
    trial = env.clone()
    assert trial.spectrum.allocate(
        action.path, action.start_slot, action.required_fs
    )
    expected = 0.0
    for probe, total in before:
        if total:
            expected += (
                probe.weight
                * (total - _probe_windows(trial, probe))
                / total
            )
    assert np.isclose(score, expected)
    assert env.state_fingerprint() == fingerprint


def test_normalized_weight_is_action_equivalent() -> None:
    raw_env = make_env(
        8101,
        50,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    normalized_env = raw_env.clone()
    raw_shadow = DemandShadowPricer(raw_env)
    normalized_shadow = DemandShadowPricer(normalized_env)
    raw_pricer = PathOpportunityPricer(raw_env)
    normalized_pricer = PathOpportunityPricer(normalized_env)
    normalized_weight = 10.0 * raw_pricer.total_probe_weight
    for request in raw_env.trace.requests[:30]:
        raw_env.advance_external(request)
        normalized_env.advance_external(request)
        raw_action = path_opportunity_action(
            raw_env,
            request,
            raw_shadow,
            raw_pricer,
            top_n=12,
            opportunity_weight=10.0,
        )
        normalized_action = path_opportunity_action(
            normalized_env,
            request,
            normalized_shadow,
            normalized_pricer,
            top_n=12,
            opportunity_weight=normalized_weight,
            normalize_opportunity=True,
        )
        assert raw_action == normalized_action
        execute(raw_env, request, raw_action)
        execute(normalized_env, request, normalized_action)


def test_optimized_scores_match_original_formula_for_full_pool() -> None:
    env = make_env(
        14301,
        20,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    shadow = DemandShadowPricer(env)
    opportunity = PathOpportunityPricer(env)
    for request in env.trace.requests[:5]:
        env.advance_external(request)
        pool = build_diverse_shadow_pool(
            env, request, shadow, top_n=12
        )
        actions = [action for _, action in pool]
        actual = opportunity.score_actions(env, actions)
        expected = []
        for action in actions:
            action_arcs = frozenset(
                zip(action.path[:-1], action.path[1:])
            )
            loss = 0.0
            for probe in opportunity._probes:
                if not (probe.arcs & action_arcs):
                    continue
                available = env.spectrum.get_available_slots(probe.path)
                total = sum(
                    _window_count(right - left, probe.width)
                    for left, right in _free_runs(available)
                )
                if total == 0:
                    continue
                after = available.copy()
                after[
                    action.start_slot :
                    action.start_slot + action.required_fs
                ] = False
                surviving = sum(
                    _window_count(right - left, probe.width)
                    for left, right in _free_runs(after)
                )
                loss += probe.weight * (total - surviving) / total
            expected.append(loss)
        assert np.allclose(actual, expected, atol=1e-12, rtol=1e-12)
        execute(env, request, pool[0][1])


def test_incremental_cache_hits_and_invalidates_on_spectrum_events() -> None:
    env = make_env(
        14302,
        100,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    shadow = DemandShadowPricer(env)
    opportunity = PathOpportunityPricer(env)
    first = env.trace.requests[0]
    env.advance_external(first)
    pool = build_diverse_shadow_pool(env, first, shadow, top_n=12)
    actions = [action for _, action in pool]
    initial = opportunity.score_actions(env, actions)
    stats_after_first = opportunity.cache_stats
    repeated = opportunity.score_actions(env, actions)
    stats_after_repeat = opportunity.cache_stats
    assert np.array_equal(initial, repeated)
    assert (
        stats_after_repeat["probe_refreshes"]
        == stats_after_first["probe_refreshes"]
    )

    execute(env, first, actions[0])
    second = env.trace.requests[1]
    env.advance_external(second)
    second_pool = build_diverse_shadow_pool(env, second, shadow, top_n=12)
    opportunity.score_actions(
        env, [action for _, action in second_pool]
    )
    stats_after_allocate = opportunity.cache_stats
    assert (
        stats_after_allocate["arc_change_events"]
        > stats_after_repeat["arc_change_events"]
    )

    later = next(
        request
        for request in env.trace.requests[2:]
        if request.arrival_time
        >= first.arrival_time + first.holding_time
    )
    env.advance_external(later)
    later_pool = build_diverse_shadow_pool(env, later, shadow, top_n=12)
    opportunity.score_actions(
        env, [action for _, action in later_pool]
    )
    stats_after_release = opportunity.cache_stats
    assert (
        stats_after_release["arc_change_events"]
        > stats_after_allocate["arc_change_events"]
    )


def test_disabling_incremental_cache_preserves_scores_and_actions() -> None:
    cached_env = make_env(
        14303,
        50,
        topology="xlron_cost239_ptrnet_real",
        num_slots=50,
        load_erlang=300.0,
    )
    uncached_env = cached_env.clone()
    cached_shadow = DemandShadowPricer(cached_env)
    uncached_shadow = DemandShadowPricer(uncached_env)
    cached = PathOpportunityPricer(cached_env)
    uncached = PathOpportunityPricer(
        uncached_env, incremental_cache=False
    )
    for request in cached_env.trace.requests[:20]:
        cached_env.advance_external(request)
        uncached_env.advance_external(request)
        cached_pool = build_diverse_shadow_pool(
            cached_env, request, cached_shadow, top_n=12
        )
        uncached_pool = build_diverse_shadow_pool(
            uncached_env, request, uncached_shadow, top_n=12
        )
        assert [
            action.action_id for _, action in cached_pool
        ] == [action.action_id for _, action in uncached_pool]
        cached_scores = cached.score_actions(
            cached_env, [action for _, action in cached_pool]
        )
        uncached_scores = uncached.score_actions(
            uncached_env, [action for _, action in uncached_pool]
        )
        assert np.array_equal(cached_scores, uncached_scores)
        execute(cached_env, request, cached_pool[0][1])
        execute(uncached_env, request, uncached_pool[0][1])


def test_bitparallel_scores_and_actions_match_exact_on_three_topologies() -> None:
    cases = (
        ("xlron_nsfnet_deeprmsa", 3000.0),
        ("xlron_usnet_gcnrmsa", 8000.0),
        ("xlron_jpn48", 5000.0),
    )
    for topology, weight in cases:
        env = make_env(
            21001,
            12,
            topology=topology,
            num_slots=50,
            load_erlang=150.0,
        )
        shadow = DemandShadowPricer(env)
        exact = PathOpportunityPricer(env, route_depth=3)
        bitparallel = BitParallelPathOpportunityPricer(
            env,
            route_depth=3,
        )
        for request in env.trace.requests[:4]:
            env.advance_external(request)
            fingerprint = env.state_fingerprint()
            pool = build_diverse_shadow_pool(
                env,
                request,
                shadow,
                top_n=12,
            )
            actions = [action for _, action in pool]
            exact_scores = exact.score_actions(
                env,
                actions,
                normalize=True,
            )
            bitparallel_scores = bitparallel.score_actions(
                env,
                actions,
                normalize=True,
            )
            assert np.array_equal(exact_scores, bitparallel_scores)
            assert env.state_fingerprint() == fingerprint

            expected = path_opportunity_action(
                env,
                request,
                shadow,
                exact,
                top_n=12,
                opportunity_weight=weight,
                normalize_opportunity=True,
            )
            actual = bitparallel_path_opportunity_action(
                env,
                request,
                shadow,
                bitparallel,
                top_n=12,
                opportunity_weight=weight,
                normalize_opportunity=True,
            )
            assert actual == expected
            assert env.state_fingerprint() == fingerprint
            execute(env, request, expected)


def test_bitparallel_cache_reuses_and_invalidates_exactly() -> None:
    env = make_env(
        21002,
        30,
        topology="xlron_nsfnet_deeprmsa",
        num_slots=50,
        load_erlang=150.0,
    )
    shadow = DemandShadowPricer(env)
    exact = PathOpportunityPricer(env)
    bitparallel = BitParallelPathOpportunityPricer(env)
    for request in env.trace.requests[:10]:
        env.advance_external(request)
        pool = build_diverse_shadow_pool(env, request, shadow, top_n=12)
        actions = [action for _, action in pool]
        expected = exact.score_actions(env, actions)
        actual = bitparallel.score_actions(env, actions)
        assert np.array_equal(expected, actual)
        repeated = bitparallel.score_actions(env, actions)
        assert np.array_equal(actual, repeated)
        execute(env, request, pool[0][1])
    stats = bitparallel.cache_stats
    assert stats["path_cache_hits"] > 0
    assert stats["arc_change_events"] > 0


def test_incremental_shadow_kernel_matches_exact_on_three_topologies() -> None:
    for topology in (
        "xlron_nsfnet_deeprmsa",
        "xlron_usnet_gcnrmsa",
        "xlron_jpn48",
    ):
        env = make_env(
            23003,
            20,
            topology=topology,
            num_slots=50,
            load_erlang=150.0,
        )
        exact = DemandShadowPricer(env)
        incremental = IncrementalDemandShadowPricer(env)
        opportunity = BitParallelPathOpportunityPricer(env)
        for request in env.trace.requests[:5]:
            env.advance_external(request)
            exact_pool = build_diverse_shadow_pool(
                env,
                request,
                exact,
                top_n=12,
            )
            incremental_pool = build_diverse_shadow_pool(
                env,
                request,
                incremental,
                top_n=12,
            )
            assert np.allclose(
                [item[0] for item in incremental_pool],
                [item[0] for item in exact_pool],
                atol=1e-12,
                rtol=1e-12,
            )
            assert [
                action.action_id for _, action in incremental_pool
            ] == [action.action_id for _, action in exact_pool]
            expected = bitparallel_path_opportunity_action(
                env,
                request,
                exact,
                opportunity,
                top_n=12,
                opportunity_weight=5000.0,
                normalize_opportunity=True,
            )
            actual = bitparallel_path_opportunity_action(
                env,
                request,
                incremental,
                opportunity,
                top_n=12,
                opportunity_weight=5000.0,
                normalize_opportunity=True,
            )
            assert actual == expected
            before = incremental.cache_stats
            repeated = build_diverse_shadow_pool(
                env,
                request,
                incremental,
                top_n=12,
            )
            after = incremental.cache_stats
            assert repeated == incremental_pool
            assert after["cache_hits"] > before["cache_hits"]
            execute(
                env,
                request,
                exact_pool[0][1]
                if exact_pool
                else ksp_ff_action(env, request),
            )


def test_formal_baseline_is_ksp_ff_k50_hops() -> None:
    env = make_env(
        23001,
        5,
        topology="xlron_nsfnet_deeprmsa",
        num_slots=50,
        load_erlang=100.0,
    )
    assert env.k_paths == 50
    assert env.path_sort_strategy == "hops"
    request = env.trace.requests[0]
    env.advance_external(request)
    paths = env._cached_k_paths(
        request.src_node,
        request.dst_node,
        env.k_paths,
    )
    keys = [
        (
            len(path) - 1,
            env.topology.path_length_km(path),
            path,
        )
        for path in paths
    ]
    assert keys == sorted(keys)
    candidates = env.build_candidates(request)
    first_legal = next(
        action for action in candidates if isinstance(action, RMSAAction)
    )
    assert ksp_ff_action(env, request) == first_legal


def test_ksp_ff_k5_environment_uses_only_five_hop_sorted_paths() -> None:
    env = make_env(
        23001,
        5,
        topology="xlron_nsfnet_deeprmsa",
        num_slots=50,
        load_erlang=100.0,
        k_paths=5,
    )
    assert env.k_paths == 5
    assert env.path_sort_strategy == "hops"
    request = env.trace.requests[0]
    env.advance_external(request)
    paths = env._cached_k_paths(
        request.src_node,
        request.dst_node,
        env.k_paths,
    )
    assert len(paths) <= 5
    action = ksp_ff_action(env, request)
    if isinstance(action, RMSAAction):
        assert action.path_rank < 5


def test_diverse_pool_preserves_independent_ff_and_bf_anchors() -> None:
    for topology in (
        "xlron_nsfnet_deeprmsa",
        "xlron_usnet_gcnrmsa",
        "xlron_jpn48",
    ):
        env = make_env(
            23002,
            30,
            topology=topology,
            num_slots=50,
            load_erlang=150.0,
        )
        shadow = DemandShadowPricer(env)
        for request in env.trace.requests[:10]:
            env.advance_external(request)
            expected_ff = ksp_ff_action(env, request)
            expected_bf = ksp_best_fit_action(env, request)
            pool = build_diverse_shadow_pool(
                env,
                request,
                shadow,
                top_n=12,
            )
            pool_ids = {action.action_id for _, action in pool}
            if isinstance(expected_ff, RMSAAction):
                assert expected_ff.action_id in pool_ids
            if isinstance(expected_bf, RMSAAction):
                assert expected_bf.action_id in pool_ids
            execute(env, request, pool[0][1] if pool else expected_ff)


def test_direct_sketch_exact_optimization_preserves_actions() -> None:
    reference_env = make_env(
        23001,
        50,
        topology="xlron_nsfnet_deeprmsa",
        num_slots=50,
        load_erlang=100.0,
    )
    optimized_env = make_env(
        23001,
        50,
        topology="xlron_nsfnet_deeprmsa",
        num_slots=50,
        load_erlang=100.0,
    )
    reference = DirectCompressedSketchSelector(
        reference_env,
        budget=256,
        opportunity_weight=3000.0,
    )
    optimized = ExactOptimizedDirectCompressedSketchSelector(
        optimized_env,
        budget=256,
        opportunity_weight=3000.0,
    )
    for reference_request, optimized_request in zip(
        reference_env.trace.requests,
        optimized_env.trace.requests,
    ):
        reference_env.advance_external(reference_request)
        optimized_env.advance_external(optimized_request)
        reference_action = reference.select(
            reference_env,
            reference_request,
        )
        optimized_action = optimized.select(
            optimized_env,
            optimized_request,
        )
        assert reference_action == optimized_action
        execute(reference_env, reference_request, reference_action)
        execute(optimized_env, optimized_request, optimized_action)
        assert (
            reference_env.state_fingerprint()
            == optimized_env.state_fingerprint()
        )


def test_uint64_legal_start_enumeration_matches_free_runs() -> None:
    rng = np.random.default_rng(20260730)
    for _ in range(500):
        available = rng.random(50) > rng.uniform(0.1, 0.9)
        available_word = sum(
            int(free) << index
            for index, free in enumerate(available)
        )
        for width in range(1, 9):
            expected = [
                start
                for left, right in _free_runs(available)
                if right - left >= width
                for start in range(left, right - width + 1)
            ]
            actual = _legal_starts_from_word(
                available_word,
                width,
                50,
            )
            assert actual.tolist() == expected


def test_compiled_shadow_rows_are_bitwise_identical() -> None:
    from sa_hmarl.pure_rmsa_v13.rollout_lab._direct_sketch_kernel import (
        build_shadow_price_row,
    )

    rng = np.random.default_rng(20260731)
    for _ in range(500):
        bitmap = rng.random(50) < rng.uniform(0.05, 0.95)
        weights = np.zeros(51, dtype=np.float64)
        widths = rng.choice(
            np.arange(1, 16),
            size=int(rng.integers(1, 10)),
            replace=False,
        )
        weights[widths] = rng.random(len(widths))
        for allocated_width in range(1, 9):
            expected = _build_price_row_python(
                bitmap,
                weights,
                allocated_width,
            )
            actual = build_shadow_price_row(
                bitmap,
                weights,
                allocated_width,
            )
            assert np.array_equal(
                actual.view(np.uint64),
                expected.view(np.uint64),
            )


def test_batched_python_shadow_rows_are_bitwise_identical() -> None:
    rng = np.random.default_rng(20260802)
    allocated_widths = np.asarray(
        [2, 3, 4, 5, 6, 7, 9],
        dtype=np.int16,
    )
    for _ in range(500):
        bitmap = rng.random(50) < rng.uniform(0.05, 0.95)
        weights = np.zeros(51, dtype=np.float64)
        future_widths = rng.choice(
            np.arange(1, 16),
            size=int(rng.integers(1, 10)),
            replace=False,
        )
        weights[future_widths] = rng.random(len(future_widths))
        actual = _build_price_rows_python(
            bitmap,
            weights,
            allocated_widths,
        )
        for row, allocated_width in enumerate(allocated_widths):
            expected = _build_price_row_python(
                bitmap,
                weights,
                int(allocated_width),
            )
            assert np.array_equal(
                actual[row].view(np.uint64),
                expected.view(np.uint64),
            )


def test_lazy_geometry_shadow_rows_are_bitwise_identical() -> None:
    rng = np.random.default_rng(20260803)
    for _ in range(500):
        bitmap = rng.random(50) < rng.uniform(0.05, 0.95)
        weights = np.zeros(51, dtype=np.float64)
        future_widths = rng.choice(
            np.arange(1, 16),
            size=int(rng.integers(1, 10)),
            replace=False,
        )
        weights[future_widths] = rng.random(len(future_widths))
        geometry = _build_price_geometry(bitmap, weights)
        occupied_word = sum(
            int(occupied) << slot
            for slot, occupied in enumerate(bitmap)
        )
        word_geometry = _build_price_geometry_from_word(
            occupied_word,
            len(bitmap),
            np.flatnonzero(weights).astype(np.int16, copy=False),
        )
        assert _free_runs_from_occupied_word(
            occupied_word,
            len(bitmap),
        ) == tuple(_free_runs(~bitmap))
        assert word_geometry[0] == geometry[0]
        assert np.array_equal(word_geometry[1], geometry[1])
        assert np.array_equal(word_geometry[2], geometry[2])
        for allocated_width in range(1, 10):
            expected = _build_price_row_python(
                bitmap,
                weights,
                allocated_width,
            )
            actual = _build_price_row_from_geometry(
                len(bitmap),
                weights,
                allocated_width,
                geometry,
            )
            assert np.array_equal(
                actual.view(np.uint64),
                expected.view(np.uint64),
            )
            word_actual = _build_price_row_from_geometry(
                len(bitmap),
                weights,
                allocated_width,
                word_geometry,
            )
            assert np.array_equal(
                word_actual.view(np.uint64),
                expected.view(np.uint64),
            )


def test_compiled_ksp_kernel_matches_python_reference() -> None:
    from sa_hmarl.pure_rmsa_v13.rollout_lab._direct_sketch_kernel import (
        ksp_first_fit,
    )

    rng = np.random.default_rng(20260801)
    full_word = (1 << 50) - 1
    for _ in range(2000):
        arc_count = int(rng.integers(4, 64))
        path_count = int(rng.integers(1, 51))
        max_hops = int(rng.integers(1, 9))
        sentinel = arc_count
        arc_words = np.zeros(arc_count + 1, dtype=np.uint64)
        occupied = rng.random((arc_count, 50)) < rng.uniform(0.0, 0.9)
        for arc_index, row in enumerate(occupied):
            arc_words[arc_index] = np.uint64(
                sum(int(value) << slot for slot, value in enumerate(row))
            )
        path_arcs = np.full(
            (path_count, max_hops),
            sentinel,
            dtype=np.intp,
        )
        for path_index in range(path_count):
            hops = int(rng.integers(1, max_hops + 1))
            path_arcs[path_index, :hops] = rng.integers(
                0,
                arc_count,
                size=hops,
            )
        widths = rng.integers(1, 10, size=path_count, dtype=np.intp)
        expected = _ksp_first_fit_python(
            arc_words,
            path_arcs,
            widths,
            full_word,
        )
        actual = ksp_first_fit(
            arc_words,
            path_arcs,
            widths,
            full_word,
        )
        assert actual == expected


def test_compiled_ksp_selector_preserves_actions_and_states() -> None:
    topology_loads = (
        ("xlron_nsfnet_deeprmsa", 100.0),
        ("xlron_usnet_gcnrmsa", 220.0),
        ("xlron_jpn48", 150.0),
    )
    for topology, load_erlang in topology_loads:
        reference_env = make_env(
            23001,
            100,
            topology=topology,
            num_slots=50,
            load_erlang=load_erlang,
        )
        compiled_env = make_env(
            23001,
            100,
            topology=topology,
            num_slots=50,
            load_erlang=load_erlang,
        )
        selector = CompiledKSPFFSelector(compiled_env)
        for reference_request, compiled_request in zip(
            reference_env.trace.requests,
            compiled_env.trace.requests,
        ):
            reference_env.advance_external(reference_request)
            compiled_env.advance_external(compiled_request)
            reference_action = ksp_ff_action(
                reference_env,
                reference_request,
            )
            compiled_action = selector.select(
                compiled_env,
                compiled_request,
            )
            assert compiled_action == reference_action
            execute(reference_env, reference_request, reference_action)
            execute(compiled_env, compiled_request, compiled_action)
            assert (
                compiled_env.state_fingerprint()
                == reference_env.state_fingerprint()
            )


def test_python_early_exit_ksp_preserves_k5_and_k50_actions() -> None:
    for k_paths in (5, 50):
        reference_env = make_env(
            23001,
            100,
            topology="xlron_nsfnet_deeprmsa",
            num_slots=50,
            load_erlang=100.0,
            k_paths=k_paths,
        )
        optimized_env = make_env(
            23001,
            100,
            topology="xlron_nsfnet_deeprmsa",
            num_slots=50,
            load_erlang=100.0,
            k_paths=k_paths,
        )
        selector = PythonKSPFFSelector(optimized_env)
        for reference_request, optimized_request in zip(
            reference_env.trace.requests,
            optimized_env.trace.requests,
        ):
            reference_env.advance_external(reference_request)
            optimized_env.advance_external(optimized_request)
            reference_action = ksp_ff_action(
                reference_env,
                reference_request,
            )
            optimized_action = selector.select(
                optimized_env,
                optimized_request,
            )
            assert optimized_action == reference_action
            execute(reference_env, reference_request, reference_action)
            execute(optimized_env, optimized_request, optimized_action)
            assert (
                optimized_env.state_fingerprint()
                == reference_env.state_fingerprint()
            )
