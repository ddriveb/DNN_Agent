"""N1-FUSE0 tests: fused-vs-original math and small-scale action parity."""

import numpy as np
import pytest

from sa_hmarl.pure_rmsa_v13.neural_opportunity.model import TinyOpportunityWeights

N1_PATH = (
    "sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/"
    "training_conflict_v1/nsfnet/deployment_weights.npz"
)


@pytest.fixture(scope="module")
def selector_classes():
    from sa_hmarl.pure_rmsa_v13.neural_opportunity.selector import (
        NeuralOpportunitySelector,
    )
    from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_fuse0.selector import (
        FusedNeuralOpportunitySelector,
    )
    return NeuralOpportunitySelector, FusedNeuralOpportunitySelector


def _fresh_env():
    from sa_hmarl.pure_rmsa_v13.train_proposer import make_env
    from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.run_phaseA import (
        _prewarm_routes,
    )

    env = make_env(61201, 400, topology="xlron_nsfnet_deeprmsa",
                   num_slots=50, load_erlang=100, k_paths=50)
    _prewarm_routes(env)
    return env


class TestFusedMath:
    def test_prices_match_original_on_live_states(self, selector_classes):
        """Fused prices equal the 110-dim matmul within float tolerance."""
        NeuralOpportunitySelector, FusedNeuralOpportunitySelector = selector_classes
        env = _fresh_env()
        weights = TinyOpportunityWeights.load(N1_PATH)
        original = NeuralOpportunitySelector(env, weights)
        fused = FusedNeuralOpportunitySelector(env, weights)
        from sa_hmarl.pure_rmsa_v13.core import execute

        checked = 0
        for index, request in enumerate(env.trace.requests):
            env.advance_external(request)
            batch = original.prepare_batch(env, request)
            if batch.features.shape[0] == 0:
                a = original.select(env, request)
                execute(env, request, a)
                continue
            # original prices (pre-rank, pre-mask)
            expected = original.weights.forward(batch.features)
            # fused forward on the same state
            bitmap, versions, changed = fused._detect_changed_arcs(env)
            fused._sync_words(env, changed)
            fused._mark_versions_seen(versions, changed)
            cands = fused._scan_feature_candidates(env, request)
            cw = fused._cw_matrix[[fused._path_row[c[1]] for c in cands]]
            profiles = cw @ bitmap.astype(np.float32, copy=False)
            avail_words = np.asarray([c[5] for c in cands], dtype=np.uint64)
            free = ((avail_words[:, None] >> fused._shift) & np.uint64(1)).astype(np.float32)
            dyn = np.asarray([
                [o / 2.0, c[3] / 50, float(request.bitrate_gbps) / 100.0,
                 float(np.bitwise_count(avail_words[o])) / 50]
                for o, c in enumerate(cands)
            ], dtype=np.float32)
            static = fused._static_matrix[[fused._path_row[c[1]] for c in cands]]
            fused_in = np.concatenate([dyn, free, profiles], axis=1)
            h_pre = static + fused_in @ fused._W_fused + fused._b1
            actual = np.maximum(h_pre, 0.0) @ fused._W2 + fused._b2
            np.testing.assert_allclose(actual, expected, rtol=1e-3, atol=1e-3)
            checked += 1
            execute(env, request, original.select(env, request))
            if checked >= 60:
                break
        assert checked >= 30

    def test_action_parity_small(self, selector_classes):
        """Per-request action identity on a short live window."""
        NeuralOpportunitySelector, FusedNeuralOpportunitySelector = selector_classes
        env = _fresh_env()
        weights = TinyOpportunityWeights.load(N1_PATH)
        original = NeuralOpportunitySelector(env, weights)
        fused = FusedNeuralOpportunitySelector(env, weights)
        from sa_hmarl.pure_rmsa_v13.core import execute

        mismatches = 0
        for index, request in enumerate(env.trace.requests):
            env.advance_external(request)
            a = original.select(env, request)
            b = fused.select(env, request)
            sig_a = (a.path_rank, a.start_slot) if hasattr(a, "path") else None
            sig_b = (b.path_rank, b.start_slot) if hasattr(b, "path") else None
            mismatches += int(sig_a != sig_b)
            execute(env, request, a)
            if index >= 350:
                break
        assert mismatches == 0

    def test_no_dynamic_cache_mutation(self, selector_classes):
        """Static tables are identical before/after live requests (read-only)."""
        _, FusedNeuralOpportunitySelector = selector_classes
        env = _fresh_env()
        weights = TinyOpportunityWeights.load(N1_PATH)
        fused = FusedNeuralOpportunitySelector(env, weights)
        from sa_hmarl.pure_rmsa_v13.core import execute

        static_before = fused._static_matrix.copy()
        cw_before = fused._cw_matrix.copy()
        for index, request in enumerate(env.trace.requests):
            env.advance_external(request)
            execute(env, request, fused.select(env, request))
            if index >= 50:
                break
        np.testing.assert_array_equal(static_before, fused._static_matrix)
        np.testing.assert_array_equal(cw_before, fused._cw_matrix)
