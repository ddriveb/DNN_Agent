"""Wide-bitmap parity tests: force coverage of slots 64-99.

The real 250-Erlang traces barely touch slots >= 64 (occupancy <5%, FF
starts from the low end), so ordinary parity does not exercise the
100-bit path.  These tests (a) unit-check the wide bit functions against
brute-force references, and (b) run a real env with the LOW 64 slots
forcibly occupied so every decision must use slots 64-99.
"""
from __future__ import annotations

import numpy as np
import pytest

from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import (
    _ksp_first_fit_python,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.feasible_window_sketch import (
    _legal_starts_from_word,
)
from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.fused_selector import (
    _word_to_bits,
)


def _brute_first_fit(arc_words, path_arcs, widths, full_word):
    """Reference: scan slots one by one, lowest feasible (path, start)."""
    for path_index, (arc_indices, width) in enumerate(
        zip(path_arcs, widths)
    ):
        occupied = 0
        for arc_index in arc_indices:
            occupied |= int(arc_words[int(arc_index)])
        if occupied & full_word == full_word:
            continue
        for start in range(0, 100 - width + 1):
            window = ((1 << width) - 1) << start
            if (occupied & window) == 0:
                return path_index, start
    return -1, -1


class TestWideFirstFit:
    def test_high_slot_occupancy_parity(self):
        rng = np.random.default_rng(0)
        for trial in range(50):
            n_arcs = 6
            # words with random HIGH-bit (64-99) occupancy, low bits partly free
            words = np.zeros(n_arcs + 1, dtype=object)
            for a in range(n_arcs):
                low = rng.integers(0, 2**10)          # some low slots busy
                high = rng.integers(0, 2**36) << 64   # slots 64-99 busy
                words[a] = int(low) | int(high)
            full = (1 << 100) - 1
            path_arcs = np.asarray([[0, 5], [1, 2], [3, 4]], dtype=np.intp)
            widths = np.asarray([3, 4, 2], dtype=np.intp)
            got = _ksp_first_fit_python(words, path_arcs, widths, full)
            ref = _brute_first_fit(words, path_arcs, widths, full)
            assert got == ref, f"trial {trial}: {got} vs {ref}"

    def test_high_slot_forced_start(self):
        # low 64 slots fully busy on the only path -> must pick a start in
        # slots 64-99
        words = np.zeros(2, dtype=object)
        words[0] = (1 << 64) - 1  # slots 0-63 busy
        path_arcs = np.asarray([[0]], dtype=np.intp)
        widths = np.asarray([5], dtype=np.intp)
        idx, start = _ksp_first_fit_python(
            words, path_arcs, widths, (1 << 100) - 1)
        assert idx == 0 and start >= 64 and start + 5 <= 100


class TestWideLegalStarts:
    def test_high_bit_word(self):
        # available-word semantics: bit = 1 means FREE.  Only slots
        # 70,71,72 are free, so the only width-3 window is [70, 73).
        word = (1 << 70) | (1 << 71) | (1 << 72)
        starts = _legal_starts_from_word(word, 3, 100)
        assert list(starts) == [70]

    def test_matches_reference(self):
        rng = np.random.default_rng(1)
        for trial in range(30):
            word = int.from_bytes(rng.bytes(13), "little") & ((1 << 100) - 1)
            width = int(rng.integers(1, 8))
            got = set(_legal_starts_from_word(word, width, 100))
            ref = {
                s for s in range(0, 100 - width + 1)
                if (word >> s) & ((1 << width) - 1) == (1 << width) - 1
            }
            assert got == ref, f"trial {trial}"


class TestWordToBits:
    def test_high_slots(self):
        word = (1 << 64) | (1 << 99)
        bits = _word_to_bits(word)
        assert bits.shape == (100,)
        assert bits[64] == 1.0 and bits[99] == 1.0
        assert bits[63] == 0.0 and bits[98] == 0.0


class TestRealEnvHighSlotParity:
    def test_forced_high_slot_decisions_match(self):
        """Fill slots 0-63 on every arc, then verify N1 == FUSE0 decisions
        and that starts land in slots 64-99."""
        from sa_hmarl.pure_rmsa_v13.core import execute
        from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.run_phaseA import (
            _prewarm_routes,
        )
        from sa_hmarl.pure_rmsa_v13.train_proposer import make_env
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.model import (
            TinyOpportunityWeights,
        )
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.selector import (
            NeuralOpportunitySelector,
        )
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.fused_selector import (
            FusedNeuralOpportunitySelector100,
        )

        env = make_env(61401, 2000, topology="xlron_nsfnet_deeprmsa",
                       num_slots=100, load_erlang=250, k_paths=50)
        _prewarm_routes(env)
        # occupy slots 0-63 on every directed link
        for link in env.spectrum.directed_link_ids():
            for s in range(0, 64, 2):
                env.spectrum.allocate((link[0], link[1]), s, 2)
        dep = TinyOpportunityWeights.load(
            "sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/"
            "n1_100/nsfnet/training/deployment_weights.npz")
        plain = NeuralOpportunitySelector(env, dep)
        fused = FusedNeuralOpportunitySelector100(env, dep)
        high_used = 0
        mismatches = 0
        checked = 0
        for i, req in enumerate(env.trace.requests):
            if i >= 500:
                env.advance_external(req)
                a1 = plain.select(env, req)
                a2 = fused.select(env, req)
                if hasattr(a1, "path_rank"):
                    checked += 1
                    if a1.start_slot >= 64:
                        high_used += 1
                    sig1 = (a1.path_rank, a1.start_slot, a1.required_fs)
                    sig2 = (a2.path_rank, a2.start_slot, a2.required_fs)
                    if sig1 != sig2:
                        mismatches += 1
                execute(env, req, a1)
            else:
                env.advance_external(req)
                execute(env, req, plain.select(env, req))
        assert checked > 0
        assert high_used >= 10, (
            f"expected decisions in slots 64-99, got {high_used}/{checked}")
        assert mismatches == 0, f"{mismatches} mismatches at high slots"
