"""N1-FUSE0: fused feature-construction + first layer for Neural Opportunity.

Computes the first-layer pre-activation directly:

    h_pre[p] = S1[path]                       # path-static features @ W_s rows
             + dyn_statics[p] @ W_s_dyn       # ordinal/width/bitrate/free_frac
             + common_free[p] @ W_f           # 50-bit map @ W_f
             + conflict_weights[p] @ M        # M = occupied_bitmap @ W_c
             + b1

with M computed ONCE per request.  Weights, candidates, and feature
definitions are exactly N1's; no cross-request dynamic caching (the S1
table is static route knowledge computed at init).  Floating-point
summation order differs from the materialized 110-dim matmul, so
equivalence is established by large-scale action parity, not assumed.
"""
from __future__ import annotations

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, PDSRMSAEnv, RMSAAction
from sa_hmarl.pds_rmsa.protocol import BITRATE_CHOICES_GBPS, Request

from ..protocol import NUM_SLOTS, STATIC_FEATURE_DIM
from ..selector import NeuralOpportunitySelector


class FusedNeuralOpportunitySelector(NeuralOpportunitySelector):
    """N1 selector with fused feature+fc1 path.  Same decisions, less work."""

    def __init__(self, env: PDSRMSAEnv, weights) -> None:
        super().__init__(env, weights)
        w1 = weights.w1  # (110, 16), N1 NumPy layout
        # Fused input layout: [dyn(1,4,5,6) | free(10..59) | conflict(60..109)]
        self._W_fused = np.concatenate(
            [w1[[1, 4, 5, 6], :], w1[STATIC_FEATURE_DIM :, :]], axis=0
        ).copy()  # (104, 16)
        self._b1 = weights.b1.copy()
        self._W2 = weights.w2.copy()
        self._b2 = weights.b2.copy()
        self._shift = np.arange(NUM_SLOTS, dtype=np.uint64)
        self._max_bitrate = float(max(BITRATE_CHOICES_GBPS))
        self._build_static_contrib(env)

    def _build_static_contrib(self, env) -> None:
        """Per-path static tables (computed once; static route knowledge).

        _paths: path tuples addressable by row index.
        _cw_matrix: (n_paths, n_arcs) conflict-weight rows.
        _static_matrix: (n_paths, 16) static-feature first-layer contribution.
        """
        w1 = self.weights.w1
        w_static = w1[[0, 2, 3, 7, 8, 9], :]  # (6, 16)
        meta = self.feature_extractor._path_meta
        rows: dict[tuple[int, ...], int] = {}
        cw_rows = []
        static_rows = []
        for specs in self._route_specs.values():
            for spec in specs:
                path_rank, path = spec[0], spec[1]
                if path in rows:
                    continue
                km = self._path_km[path]
                centrality_sum, centrality_max, overlap, conflict_weights = meta[path]
                statics = np.asarray([
                    path_rank / 49.0,
                    min(1.0, (len(path) - 1) / 16.0),
                    min(1.0, float(km) / 5000.0),
                    centrality_sum,
                    centrality_max,
                    overlap,
                ], dtype=np.float32)
                rows[path] = len(cw_rows)
                cw_rows.append(conflict_weights)
                static_rows.append(statics @ w_static)
        self._path_row = rows
        self._cw_matrix = np.stack(cw_rows)
        self._static_matrix = np.stack(static_rows)

    def select(self, env: PDSRMSAEnv, request: Request):
        bitmap, spectrum_versions, changed = self._detect_changed_arcs(env)
        self._sync_words(env, changed)
        self._mark_versions_seen(spectrum_versions, changed)
        candidates = self._scan_feature_candidates(env, request)
        if not candidates:
            return BlockedAction(
                reason=self._blocked_reasons[(request.src_node, request.dst_node)]
            )

        n = len(candidates)
        path_rows = [self._path_row[c[1]] for c in candidates]
        cw = self._cw_matrix[path_rows]                        # (n, arcs)
        profiles = cw @ bitmap.astype(np.float32, copy=False)  # (n, 50)
        start_words = np.asarray([c[4] for c in candidates], dtype=np.uint64)
        avail_words = np.asarray([c[5] for c in candidates], dtype=np.uint64)
        free = ((avail_words[:, None] >> self._shift) & np.uint64(1)).astype(
            np.float32
        )                                                     # (n, 50)
        valid = ((start_words[:, None] >> self._shift) & np.uint64(1)) != 0
        dyn = np.asarray([
            [
                ordinal / 2.0,
                c[3] / NUM_SLOTS,
                float(request.bitrate_gbps) / self._max_bitrate,
                float(np.bitwise_count(avail_words[ordinal])) / NUM_SLOTS,
            ]
            for ordinal, c in enumerate(candidates)
        ], dtype=np.float32)                                  # (n, 4)
        static = self._static_matrix[path_rows]               # (n, 16)

        fused_in = np.concatenate([dyn, free, profiles], axis=1)  # (n, 104)
        h_pre = static + fused_in @ self._W_fused + self._b1
        prices = np.maximum(h_pre, 0.0) @ self._W2 + self._b2  # (n, 50)
        ranks = np.asarray([c[0] for c in candidates], dtype=np.float32)
        prices += ranks[:, None]
        prices[~valid] = np.inf
        best = int(np.argmin(prices))
        path_index = best // NUM_SLOTS
        start = best - path_index * NUM_SLOTS
        path_rank, path, modulation, width = candidates[int(path_index)][:4]
        return RMSAAction(
            action_id=(path_rank, modulation, int(start), width),
            path_rank=path_rank,
            path=path,
            modulation=modulation,
            start_slot=int(start),
            required_fs=width,
        )
