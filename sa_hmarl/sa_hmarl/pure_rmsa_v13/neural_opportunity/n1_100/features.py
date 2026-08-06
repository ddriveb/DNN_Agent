"""Three path rows in, fifty slot prices per path out."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import PDSRMSAEnv
from sa_hmarl.pds_rmsa.protocol import BITRATE_CHOICES_GBPS

from .protocol import FEATURE_DIM, NUM_SLOTS, ROUTE_DEPTH, STATIC_FEATURE_DIM


def _ints_to_bools(words: list[int]) -> np.ndarray:
    """list of 100-bit Python ints -> (n, NUM_SLOTS) bool; slot s at column s.

    One vectorized unpackbits call over the concatenated little-endian bytes,
    replacing per-candidate object-dtype bit masking (`(word & slot_bits)!=0`
    boxes a Python big-int per slot per candidate).
    """
    if not words:
        return np.empty((0, NUM_SLOTS), dtype=np.bool_)
    raw = b"".join(int(word).to_bytes(13, "little") for word in words)
    return np.unpackbits(
        np.frombuffer(raw, dtype=np.uint8).reshape(-1, 13),
        axis=1,  # without axis, unpackbits flattens the input
        bitorder="little",
    )[:, :NUM_SLOTS].astype(np.bool_, copy=False)


@dataclass(frozen=True)
class PathSlotBatch:
    features: np.ndarray
    valid_starts: np.ndarray
    candidates: tuple[tuple, ...]


class PathSlotFeatureExtractor:
    """Build three dense path descriptors without per-start Python work."""

    def __init__(self, env: PDSRMSAEnv, arc_index: dict[tuple[int, int], int]):
        if env.num_slots != NUM_SLOTS:
            raise ValueError(f"Neural Opportunity v1 requires {NUM_SLOTS} slots")
        self._slot_bits = np.asarray(
            [1 << slot for slot in range(NUM_SLOTS)],
            dtype=object,
        )
        self._arc_index = arc_index
        self._centrality = self._build_route_centrality(env)
        self._conflict_kernel = self._build_conflict_kernel(env)
        self._path_meta = self._build_path_meta(env)

    def _build_route_centrality(self, env: PDSRMSAEnv) -> np.ndarray:
        centrality = np.zeros(len(self._arc_index), dtype=np.float32)
        for src in range(env.topology.num_nodes):
            for dst in range(env.topology.num_nodes):
                if src == dst:
                    continue
                for path in env._cached_k_paths(src, dst, env.k_paths)[:ROUTE_DEPTH]:
                    weight = 1.0 / max(1, len(path) - 1)
                    for arc in zip(path[:-1], path[1:]):
                        centrality[self._arc_index[arc]] += weight
        maximum = float(np.max(centrality)) if centrality.size else 0.0
        if maximum > 0.0:
            centrality /= maximum
        return centrality

    def _build_conflict_kernel(self, env: PDSRMSAEnv) -> np.ndarray:
        """Static arc co-occurrence among likely future end-to-end routes."""
        arc_count = len(self._arc_index)
        kernel = np.zeros((arc_count, arc_count), dtype=np.float32)
        for src in range(env.topology.num_nodes):
            for dst in range(env.topology.num_nodes):
                if src == dst:
                    continue
                for path in env._cached_k_paths(src, dst, env.k_paths)[:ROUTE_DEPTH]:
                    indices = np.fromiter(
                        (
                            self._arc_index[arc]
                            for arc in zip(path[:-1], path[1:])
                        ),
                        dtype=np.intp,
                    )
                    weight = 1.0 / max(1, len(indices))
                    kernel[np.ix_(indices, indices)] += weight
        return kernel

    def _build_path_meta(self, env: PDSRMSAEnv) -> dict[tuple[int, ...], tuple]:
        result = {}
        for src in range(env.topology.num_nodes):
            for dst in range(env.topology.num_nodes):
                if src == dst:
                    continue
                paths = env._cached_k_paths(src, dst, env.k_paths)
                top_arc_sets = [
                    set(zip(other[:-1], other[1:])) for other in paths[:ROUTE_DEPTH]
                ]
                for path in paths:
                    if path in result:
                        continue
                    path_arcs = set(zip(path[:-1], path[1:]))
                    arc_indices = np.fromiter(
                        (
                            self._arc_index[arc]
                            for arc in zip(path[:-1], path[1:])
                        ),
                        dtype=np.intp,
                    )
                    centrality = self._centrality[arc_indices]
                    conflict_weights = np.sum(
                        self._conflict_kernel[arc_indices], axis=0
                    )
                    conflict_total = float(np.sum(conflict_weights))
                    if conflict_total > 0.0:
                        conflict_weights /= conflict_total
                    others: set[tuple[int, int]] = set()
                    for other_path, other_arcs in zip(paths[:ROUTE_DEPTH], top_arc_sets):
                        if other_path != path:
                            others.update(other_arcs)
                    result[path] = (
                        min(1.0, float(np.sum(centrality)) / 8.0),
                        float(np.max(centrality)),
                        len(path_arcs & others) / max(1, len(path_arcs)),
                        conflict_weights.astype(np.float32, copy=False),
                    )
        return result

    def build(self, request, bitmap: np.ndarray, candidates: list[tuple]):
        if not candidates:
            return PathSlotBatch(
                features=np.empty((0, FEATURE_DIM), dtype=np.float32),
                valid_starts=np.empty((0, NUM_SLOTS), dtype=np.bool_),
                candidates=(),
            )
        count = len(candidates)
        features = np.empty((count, FEATURE_DIM), dtype=np.float32)
        valid_starts = np.empty((count, NUM_SLOTS), dtype=np.bool_)
        conflict_weights = np.stack(
            [self._path_meta[candidate[1]][3] for candidate in candidates]
        )
        conflict_profiles = conflict_weights @ bitmap.astype(
            np.float32, copy=False
        )
        avail_words = [candidate[5] for candidate in candidates]
        start_words = [candidate[4] for candidate in candidates]
        common_free_all = _ints_to_bools(avail_words)
        valid_all = _ints_to_bools(start_words)
        free_counts = [int(word).bit_count() for word in avail_words]
        max_bitrate = max(BITRATE_CHOICES_GBPS)
        for ordinal, candidate in enumerate(candidates):
            (
                path_rank,
                path,
                _,
                width,
                _,
                _,
                arc_indices,
                km,
            ) = candidate
            common_free = common_free_all[ordinal]
            centrality_sum, centrality_max, overlap, _ = self._path_meta[path]
            features[ordinal, :STATIC_FEATURE_DIM] = (
                path_rank / 49.0,
                ordinal / 2.0,
                min(1.0, (len(path) - 1) / 16.0),
                min(1.0, float(km) / 5000.0),
                width / NUM_SLOTS,
                float(request.bitrate_gbps) / max_bitrate,
                free_counts[ordinal] / NUM_SLOTS,
                centrality_sum,
                centrality_max,
                overlap,
            )
            offset = STATIC_FEATURE_DIM
            features[ordinal, offset : offset + NUM_SLOTS] = common_free
            offset += NUM_SLOTS
            features[ordinal, offset : offset + NUM_SLOTS] = (
                conflict_profiles[ordinal]
            )
            valid_starts[ordinal] = valid_all[ordinal]
        return PathSlotBatch(
            features=features,
            valid_starts=valid_starts,
            candidates=tuple(candidates),
        )
