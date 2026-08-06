"""N1-100 FUSE0: fused feature-construction + first layer at 100 slots.

Same fusion idea as n1_fuse0 (precomputed per-path static first-layer
contribution; per-request dynamic block [dyn | common_free | conflict]
fused matmul) with Python-int 100-bit words: the 100-slot map is extracted
via to_bytes + unpackbits instead of uint64 shifts.  Equivalence with the
plain N1-100 selector is established by action parity (0 mismatch).
"""
from __future__ import annotations

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, PDSRMSAEnv, RMSAAction
from sa_hmarl.pds_rmsa.protocol import BITRATE_CHOICES_GBPS, Request

from .protocol import NUM_SLOTS, STATIC_FEATURE_DIM
from .selector import NeuralOpportunitySelector

_WIDE_BYTES = (NUM_SLOTS + 7) // 8  # 13 bytes for 100 slots


def _word_to_bits(word: int) -> np.ndarray:
    """100-bit word -> (NUM_SLOTS,) bool via little-endian bytes."""
    raw = int(word).to_bytes(_WIDE_BYTES, "little")
    return np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8), bitorder="little"
    )[:NUM_SLOTS].astype(np.float32)


class FusedNeuralOpportunitySelector100(NeuralOpportunitySelector):
    """N1-100 selector with fused feature+fc1 path (100-slot variant)."""

    def __init__(self, env: PDSRMSAEnv, weights) -> None:
        super().__init__(env, weights)
        w1 = weights.w1  # (210, 16)
        self._W_fused = np.concatenate(
            [w1[[1, 4, 5, 6], :], w1[STATIC_FEATURE_DIM:, :]], axis=0
        ).copy()  # (204, 16)
        self._b1 = weights.b1.copy()
        self._W2 = weights.w2.copy()
        self._b2 = weights.b2.copy()
        self._max_bitrate = float(max(BITRATE_CHOICES_GBPS))
        self._build_static_contrib(env)

    def _build_static_contrib(self, env) -> None:
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
        profiles = cw @ bitmap.astype(np.float32, copy=False)  # (n, 100)
        free = np.stack([_word_to_bits(c[5]) for c in candidates])     # (n, 100)
        valid = np.stack([_word_to_bits(c[4]) for c in candidates]) != 0
        dyn = np.asarray([
            [
                ordinal / 2.0,
                c[3] / NUM_SLOTS,
                float(request.bitrate_gbps) / self._max_bitrate,
                float(int(c[5]).bit_count()) / NUM_SLOTS,
            ]
            for ordinal, c in enumerate(candidates)
        ], dtype=np.float32)                                  # (n, 4)
        static = self._static_matrix[path_rows]               # (n, 16)

        fused_in = np.concatenate([dyn, free, profiles], axis=1)  # (n, 204)
        h_pre = static + fused_in @ self._W_fused + self._b1
        prices = np.maximum(h_pre, 0.0) @ self._W2 + self._b2  # (n, 100)
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
