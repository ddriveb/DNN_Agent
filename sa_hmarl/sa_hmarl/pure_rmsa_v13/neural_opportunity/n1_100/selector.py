"""Online selector: legal candidate construction plus one tiny MLP call."""
from __future__ import annotations

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import (
    BlockedAction,
    PDSRMSAEnv,
    RMSAAction,
)
from sa_hmarl.pds_rmsa.protocol import Request
from sa_hmarl.pure_rmsa_v13.direct_sketch_exact_optimization.selectors_phaseA import (
    _DirectStaticRoutes,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.feasible_window_sketch import (
    _legal_starts_from_word,
)

from .features import PathSlotFeatureExtractor
from .model import TinyOpportunityWeights
from .protocol import MAX_FEASIBLE_PATHS


class NeuralOpportunitySelector(_DirectStaticRoutes):
    """Approximate opportunity pricing without an online exact pricer."""

    def __init__(self, env: PDSRMSAEnv, weights: TinyOpportunityWeights) -> None:
        self.weights = weights
        self._arcs = tuple(env.spectrum.directed_link_ids())
        self._arc_index = {arc: index for index, arc in enumerate(self._arcs)}
        self._slot_bits = np.asarray(
            [1 << slot for slot in range(env.num_slots)],
            dtype=object,
        )
        self._sentinel_arc = len(self._arcs)
        self._arc_words = np.zeros(len(self._arcs) + 1, dtype=object)
        self._init_route_state(env)
        self._sync_words(env, np.arange(len(self._arcs), dtype=np.intp))
        self._precompute_static_routes(env)
        self._build_fast_routes()
        self._path_km = {
            spec[1]: env.topology.path_length_km(spec[1])
            for specs in self._route_specs.values()
            for spec in specs
        }
        self.feature_extractor = PathSlotFeatureExtractor(env, self._arc_index)

    def _build_fast_routes(self) -> None:
        self._fast_routes = {}
        for key, specs in self._route_specs.items():
            max_hops = max((len(spec[4]) for spec in specs), default=1)
            path_arcs = np.full(
                (len(specs), max_hops),
                self._sentinel_arc,
                dtype=np.intp,
            )
            widths = np.empty(len(specs), dtype=np.intp)
            for index, spec in enumerate(specs):
                path_arcs[index, : len(spec[4])] = spec[4]
                widths[index] = spec[3]
            max_width = int(np.max(widths)) if widths.size else 0
            active_by_shift = tuple(
                np.flatnonzero(widths > shift)
                for shift in range(1, max_width)
            )
            self._fast_routes[key] = (
                specs,
                path_arcs,
                widths,
                active_by_shift,
            )

    def _sync_words(self, env: PDSRMSAEnv, changed: np.ndarray) -> None:
        if changed.size == 0:
            return
        bitmap = env.spectrum.occupied_bitmap_view()
        words = np.asarray(
            bitmap[changed].astype(object, copy=False) @ self._slot_bits,
            dtype=object,
        )
        self._arc_words[changed] = words

    def _scan_feature_candidates(self, env, request) -> list[tuple]:
        """Bit-parallel candidate scan (object-dtype numpy, 100-bit ints).

        Measured faster than a uint64 word-pair formulation: the pair version
        needed ~70 small-array numpy calls per request (fancy-indexed shifts
        with boundary carry), and per-call overhead exceeded the Python-int
        element work the object arrays already do in C.
        """
        specs, path_arcs, widths, active_by_shift = self._fast_routes[
            (request.src_node, request.dst_node, int(request.bitrate_gbps))
        ]
        occupied = np.bitwise_or.reduce(self._arc_words[path_arcs], axis=1)
        available = np.bitwise_not(occupied) & int(self._full_word_int)
        start_words = available.copy()
        for shift, active in enumerate(active_by_shift, start=1):
            start_words[active] &= available[active] >> int(shift)
        feasible_indices = np.flatnonzero(start_words)[:MAX_FEASIBLE_PATHS]
        candidates = []
        for spec_index in feasible_indices:
            path_rank, path, modulation, width, arc_indices = specs[
                int(spec_index)
            ]
            candidates.append(
                (
                    path_rank,
                    path,
                    modulation,
                    width,
                    int(start_words[int(spec_index)]),
                    int(available[int(spec_index)]),
                    arc_indices,
                    self._path_km[path],
                )
            )
        return candidates

    def prepare_batch(self, env: PDSRMSAEnv, request: Request):
        """Synchronize state and build the single-forward input batch."""
        bitmap, spectrum_versions, changed = self._detect_changed_arcs(env)
        self._sync_words(env, changed)
        self._mark_versions_seen(spectrum_versions, changed)
        candidates = self._scan_feature_candidates(env, request)
        return self.feature_extractor.build(request, bitmap, candidates)

    def select(self, env: PDSRMSAEnv, request: Request):
        batch = self.prepare_batch(env, request)
        if batch.features.shape[0] == 0:
            return BlockedAction(
                reason=self._blocked_reasons[
                    (request.src_node, request.dst_node)
                ]
            )
        predicted_opportunity = self.weights.forward(batch.features)
        for row, candidate in enumerate(batch.candidates):
            predicted_opportunity[row] += float(candidate[0])
        predicted_opportunity[~batch.valid_starts] = np.inf
        best = int(np.argmin(predicted_opportunity))
        path_index = best // env.num_slots
        start = best - path_index * env.num_slots
        candidate = batch.candidates[int(path_index)]
        path_rank, path, modulation, width = candidate[:4]
        start = int(start)
        return RMSAAction(
            action_id=(path_rank, modulation, start, width),
            path_rank=path_rank,
            path=path,
            modulation=modulation,
            start_slot=start,
            required_fs=width,
        )
