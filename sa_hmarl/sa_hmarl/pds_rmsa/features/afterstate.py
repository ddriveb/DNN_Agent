"""Post-decision-state feature extraction for RMSA."""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

from sa_hmarl.pds_rmsa.env.reservations import PostDecisionState
from sa_hmarl.pds_rmsa.env.topology import Topology, load_topology
from sa_hmarl.pds_rmsa.protocol import (
    K_PATHS,
    PROBE_BANDWIDTHS_GBPS,
    PROBE_OD_COUNT,
    PROBE_SEED,
    highest_modulation_for_distance,
    required_fs,
)


@dataclass(frozen=True)
class Probe:
    src: int
    dst: int
    bandwidth_gbps: int
    paths: Tuple[Tuple[Tuple[int, ...], int], ...]


@dataclass(frozen=True)
class ProbeSuite:
    topology_name: str
    num_nodes: int
    seed: int
    probes: Tuple[Probe, ...]
    k_paths: int = K_PATHS
    path_sort_strategy: str = "km"


@dataclass
class FeatureSchema:
    feature_names: List[str]
    dims: Dict[str, Tuple[int, int]]
    link_order: List[Tuple[int, int]]
    slot_order: str
    dtype: str
    config: dict

    def to_dict(self) -> dict:
        return {
            "feature_names": self.feature_names,
            "dims": self.dims,
            "link_order": self.link_order,
            "slot_order": self.slot_order,
            "dtype": self.dtype,
            "config": self.config,
        }

    def save(self, path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)


def _directed_link_ids(topology: Topology) -> List[Tuple[int, int]]:
    return sorted(
        [(u, v) for u, v in topology.edges()] + [(v, u) for u, v in topology.edges()]
    )


def _largest_free_block(occupied: np.ndarray) -> int:
    free = ~occupied
    max_run = 0
    run = 0
    for is_free in free:
        if is_free:
            run += 1
            max_run = max(max_run, run)
        else:
            run = 0
    return max_run


def _free_block_histogram(occupied: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """Count contiguous free block sizes into inclusive bins (minimum run length is 1)."""
    free = ~occupied
    hist = np.zeros(len(bin_edges), dtype=int)
    run = 0
    for is_free in free:
        if is_free:
            run += 1
        else:
            if run > 0:
                for idx, edge in enumerate(bin_edges):
                    if run <= edge:
                        hist[idx] += 1
                        break
            run = 0
    if run > 0:
        for idx, edge in enumerate(bin_edges):
            if run <= edge:
                hist[idx] += 1
                break
    return hist


def _path_fits(
    bitmap: np.ndarray,
    link_ids: List[Tuple[int, int]],
    topology: Topology,
    path: Tuple[int, ...],
    req_fs: int,
) -> bool:
    indices = [link_ids.index((path[i], path[i + 1])) for i in range(len(path) - 1)]
    available = ~np.any(bitmap[indices, :], axis=0)
    run = 0
    for free in available:
        if free:
            run += 1
            if run >= req_fs:
                return True
        else:
            run = 0
    return False


def build_probe_suite(
    num_nodes: int,
    topology_name: str,
    num_od_pairs: int = PROBE_OD_COUNT,
    bandwidths: Tuple[int, ...] = PROBE_BANDWIDTHS_GBPS,
    k_paths: int = K_PATHS,
    path_sort_strategy: str = "km",
) -> ProbeSuite:
    """Generate a fixed, serializable probe suite using the locked PROBE_SEED."""
    if path_sort_strategy not in {"km", "hops"}:
        raise ValueError("path_sort_strategy must be 'km' or 'hops'")
    topology = load_topology(topology_name)
    rng = np.random.RandomState(int(PROBE_SEED))
    pairs = [
        (s, d) for s in range(int(num_nodes)) for d in range(int(num_nodes)) if s != d
    ]
    chosen_pairs = [
        pairs[i]
        for i in rng.choice(
            len(pairs), size=min(num_od_pairs, len(pairs)), replace=False
        )
    ]

    probes: List[Probe] = []
    for src, dst in chosen_pairs:
        raw_paths = topology.k_shortest_paths(
            src, dst, int(k_paths), sort_by=path_sort_strategy
        )
        for bw in bandwidths:
            paths_info: List[Tuple[Tuple[int, ...], int]] = []
            for path in raw_paths:
                path_len = topology.path_length_km(path)
                mod = highest_modulation_for_distance(path_len)
                if mod is None:
                    continue
                req_fs = required_fs(bw, mod[1])
                paths_info.append((path, req_fs))
            probes.append(
                Probe(
                    src=src,
                    dst=dst,
                    bandwidth_gbps=bw,
                    paths=tuple(paths_info),
                )
            )
    return ProbeSuite(
        topology_name=topology_name,
        num_nodes=int(num_nodes),
        seed=int(PROBE_SEED),
        probes=tuple(probes),
        k_paths=int(k_paths),
        path_sort_strategy=str(path_sort_strategy),
    )


def probe_suite_sha256(probe_suite: ProbeSuite) -> str:
    payload = json.dumps(
        asdict(probe_suite), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def save_probe_suite(path, probe_suite: ProbeSuite) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(asdict(probe_suite), f, indent=2, sort_keys=True)


def _largest_free_block_and_hist_batch(free: np.ndarray, bin_edges: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Vectorized largest free block and free-block histogram over a batch of rows.

    Parameters
    ----------
    free
        Boolean array of shape (B, L, F) where True indicates a free slot.
    bin_edges
        Histogram bin edges (ascending, last edge should be np.inf or float('inf')).

    Returns
    -------
    largest
        Integer array of shape (B, L) with the largest contiguous free block per row.
    hist
        Integer array of shape (B, L, len(bin_edges)) with per-row histograms.
    """
    B, L, F = free.shape
    rows = free.reshape(B * L, F)
    padded = np.concatenate(
        [np.zeros((B * L, 1), dtype=bool), rows, np.zeros((B * L, 1), dtype=bool)],
        axis=1,
    )
    diff = np.diff(padded.astype(np.int8), axis=1)
    start_idx, start_pos = np.where(diff == 1)
    end_idx, end_pos = np.where(diff == -1)
    lengths = end_pos - start_pos

    row_indices = start_idx
    num_rows = B * L

    largest = np.zeros(num_rows, dtype=np.int64)
    if len(lengths) > 0:
        np.maximum.at(largest, row_indices, lengths)

    hist = np.zeros((num_rows, len(bin_edges)), dtype=np.int64)
    if len(lengths) > 0:
        lengths_col = lengths[:, np.newaxis]  # (N, 1)
        edges_row = bin_edges[np.newaxis, :]  # (1, nbins)
        # Find the first bin edge >= length for each run.
        bins = np.argmax(lengths_col <= edges_row, axis=1)
        np.add.at(hist, (row_indices, bins), 1)

    return largest.reshape(B, L), hist.reshape(B, L, len(bin_edges))


def build_afterstate_features_batch(
    post_states: List[PostDecisionState],
    probe_suite: ProbeSuite,
    feature_config: dict,
) -> np.ndarray:
    """Vectorized feature extraction for a batch of post-decision states.

    The output is identical in semantics to ``build_afterstate_features`` applied
    to each state individually, but the per-link statistics are computed in a single
    vectorized pass over all states.
    """
    if len(post_states) == 1:
        return _build_afterstate_features_single(
            post_states[0], probe_suite, feature_config
        )[np.newaxis, :]
    if not post_states:
        return np.zeros((0, 0), dtype=np.float32)

    bitmaps = np.stack([np.asarray(s.bitmap, dtype=bool) for s in post_states])
    B, num_links, num_slots = bitmaps.shape

    topology = load_topology(probe_suite.topology_name)
    link_ids = _directed_link_ids(topology)
    link_index = {arc: idx for idx, arc in enumerate(link_ids)}
    if len(link_ids) != num_links:
        raise ValueError("Bitmap link count does not match topology directed link count")

    hist_edges = np.asarray(
        feature_config.get("free_block_bins", [1, 2, 4, 8, 16, 32, np.inf])
    )
    residual_edges = feature_config.get(
        "residual_bucket_edges", [0, 2, 5, 10, 20, float("inf")]
    )

    free = ~bitmaps
    free_count = free.sum(axis=2)  # (B, num_links)
    free_ratio = free_count / num_slots if num_slots > 0 else np.zeros((B, num_links), dtype=float)
    largest, hist = _largest_free_block_and_hist_batch(free, hist_edges)
    frag = np.zeros((B, num_links), dtype=float)
    valid_mask = (free_count > 0) & (free_count < num_slots)
    frag[valid_mask] = 1.0 - largest[valid_mask].astype(float) / free_count[valid_mask]

    # 1. Raw bitmap segment
    raw = bitmaps.reshape(B, -1).astype(np.float32)

    # 2. Per-link summary
    per_link = np.concatenate(
        [
            free_ratio[:, :, np.newaxis],
            largest[:, :, np.newaxis].astype(float),
            hist.astype(float),
            frag[:, :, np.newaxis],
        ],
        axis=2,
    ).reshape(B, -1)

    # 3. Global summary
    occupancy = bitmaps.mean(axis=(1, 2))[:, np.newaxis]  # (B, 1)
    frag_mean = frag.mean(axis=1, keepdims=True)  # (B, 1)
    largest_global = largest.max(axis=1, keepdims=True).astype(float)  # (B, 1)
    total_occupied_slot_hops = bitmaps.sum(axis=(1, 2))[:, np.newaxis].astype(float)  # (B, 1)
    global_feats = np.concatenate(
        [occupancy, 1.0 - occupancy, frag_mean, largest_global, total_occupied_slot_hops],
        axis=1,
    )

    # 4. Fixed probe summary (vectorized over states for each probe path)
    probe_features = np.zeros((B, 2 * len(probe_suite.probes)), dtype=np.float32)
    for probe_idx, probe in enumerate(probe_suite.probes):
        any_serviceable = np.zeros(B, dtype=bool)
        feasible_path_counts = np.zeros(B, dtype=int)
        for path, req_fs in probe.paths:
            arcs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            indices = [link_index[arc] for arc in arcs]
            available = ~np.any(bitmaps[:, indices, :], axis=1)  # (B, F)
            if req_fs <= num_slots:
                prefix = np.pad(
                    np.cumsum(available, axis=1, dtype=np.int16),
                    ((0, 0), (1, 0)),
                )
                window_free_counts = prefix[:, req_fs:] - prefix[:, :-req_fs]
                fits = np.any(window_free_counts == req_fs, axis=1)
            else:
                fits = np.zeros(B, dtype=bool)
            any_serviceable |= fits
            feasible_path_counts += fits.astype(int)
        path_ratio = feasible_path_counts / len(probe.paths) if probe.paths else np.zeros(B, dtype=float)
        probe_features[:, 2 * probe_idx] = any_serviceable.astype(float)
        probe_features[:, 2 * probe_idx + 1] = path_ratio.astype(float)

    # 5. Residual reservation summary
    residual_dim = 2 * len(residual_edges) + 2
    residual_features = np.zeros((B, residual_dim), dtype=np.float32)
    residual_cache = {}

    def summarize_reservations(reservations, current_time):
        key = (float(current_time), tuple(reservations))
        cached = residual_cache.get(key)
        if cached is not None:
            return cached.copy()
        summary = np.zeros(residual_dim, dtype=np.float32)
        for res in reservations:
            if res.release_time <= current_time:
                continue
            remaining = res.release_time - current_time
            hops = max(0, len(res.path) - 1)
            slot_hops = res.required_fs * hops
            for b_idx, edge in enumerate(residual_edges):
                if remaining < edge or (edge == float("inf") and remaining >= 0):
                    summary[2 * b_idx] += 1.0
                    summary[2 * b_idx + 1] += float(slot_hops)
                    break
            summary[-2] += 1.0
            summary[-1] += float(slot_hops)
        residual_cache[key] = summary
        return summary.copy()

    for b, state in enumerate(post_states):
        reservations = tuple(state.reservations)
        if (
            isinstance(state, PostDecisionState)
            and state.admitted_this_step
            and reservations
        ):
            newest = max(reservations, key=lambda r: r.request_id)
            base = tuple(r for r in reservations if r is not newest)
            summary = summarize_reservations(base, state.time)
            summary += summarize_reservations((newest,), state.time)
            residual_features[b] = summary
        else:
            residual_features[b] = summarize_reservations(
                reservations, state.time
            )

    return np.concatenate(
        [raw, per_link, global_feats, probe_features, residual_features],
        axis=1,
    )


def build_afterstate_features(
    post_state: PostDecisionState,
    probe_suite: ProbeSuite,
    feature_config: dict,
) -> np.ndarray:
    """Build a deterministic feature vector from a single post-decision state."""
    return _build_afterstate_features_single(post_state, probe_suite, feature_config)


def _build_afterstate_features_single(
    post_state: PostDecisionState,
    probe_suite: ProbeSuite,
    feature_config: dict,
) -> np.ndarray:
    """Fast single-state feature extraction."""
    bitmap = np.asarray(post_state.bitmap, dtype=bool)
    num_links, num_slots = bitmap.shape

    topology = load_topology(probe_suite.topology_name)
    link_ids = _directed_link_ids(topology)
    if len(link_ids) != num_links:
        raise ValueError("Bitmap link count does not match topology directed link count")

    hist_edges = np.asarray(
        feature_config.get("free_block_bins", [1, 2, 4, 8, 16, 32, np.inf])
    )
    residual_edges = feature_config.get(
        "residual_bucket_edges", [0, 2, 5, 10, 20, float("inf")]
    )

    features: List[float] = []

    # 1. Raw bitmap segment
    features.extend(bitmap.flatten().astype(float))

    # 2. Per-link summary
    per_link_free_ratios = []
    per_link_largest = []
    per_link_fragmentation = []
    for link_idx in range(num_links):
        occupied = bitmap[link_idx]
        total_free = int(np.sum(~occupied))
        free_ratio = total_free / num_slots if num_slots > 0 else 0.0
        largest = _largest_free_block(occupied)
        frag = 0.0
        if total_free > 0 and total_free < num_slots:
            frag = 1.0 - largest / total_free
        hist = _free_block_histogram(occupied, hist_edges)
        per_link_free_ratios.append(free_ratio)
        per_link_largest.append(largest)
        per_link_fragmentation.append(frag)
        features.append(free_ratio)
        features.append(largest)
        for b_idx in range(len(hist_edges)):
            features.append(float(hist[b_idx]))
        features.append(frag)

    # 3. Global summary
    occupancy = float(np.mean(bitmap)) if bitmap.size else 0.0
    free_ratio_global = 1.0 - occupancy
    frag_mean = float(np.mean(per_link_fragmentation)) if per_link_fragmentation else 0.0
    largest_global = int(max(per_link_largest)) if per_link_largest else 0
    total_occupied_slot_hops = int(bitmap.sum())
    features.extend(
        [occupancy, free_ratio_global, frag_mean, largest_global, total_occupied_slot_hops]
    )

    # 4. Fixed probe summary
    for probe_idx, probe in enumerate(probe_suite.probes):
        feasible_paths = 0
        serviceable = 0.0
        for path, req_fs in probe.paths:
            if _path_fits(bitmap, link_ids, topology, path, req_fs):
                feasible_paths += 1
                serviceable = 1.0
        path_ratio = feasible_paths / len(probe.paths) if probe.paths else 0.0
        features.extend([serviceable, path_ratio])

    # 5. Residual reservation summary
    active = [r for r in post_state.reservations if r.release_time > post_state.time]
    bucket_counts = [0] * len(residual_edges)
    bucket_slot_hops = [0.0] * len(residual_edges)
    total_active_count = 0
    total_active_slot_hops = 0.0
    for res in active:
        remaining = res.release_time - post_state.time
        hops = max(0, len(res.path) - 1)
        slot_hops = res.required_fs * hops
        total_active_count += 1
        total_active_slot_hops += slot_hops
        for b_idx, edge in enumerate(residual_edges):
            if remaining < edge or (edge == float("inf") and remaining >= 0):
                bucket_counts[b_idx] += 1
                bucket_slot_hops[b_idx] += slot_hops
                break
    for b_idx in range(len(residual_edges)):
        features.append(float(bucket_counts[b_idx]))
        features.append(float(bucket_slot_hops[b_idx]))
    features.append(float(total_active_count))
    features.append(float(total_active_slot_hops))

    return np.asarray(features, dtype=np.float32)


def build_schema(
    topology_name: str,
    feature_config: dict,
    probe_count: int = 1,
) -> FeatureSchema:
    """Build a feature schema without needing an actual post-decision state."""
    topology = load_topology(topology_name)
    link_ids = _directed_link_ids(topology)
    num_links = len(link_ids)
    num_slots = int(feature_config.get("num_slots", 0))

    hist_edges = np.asarray(
        feature_config.get("free_block_bins", [1, 2, 4, 8, 16, 32, np.inf])
    )
    residual_edges = feature_config.get(
        "residual_bucket_edges", [0, 2, 5, 10, 20, float("inf")]
    )
    names = []
    dims = {}

    start = len(names)
    for link_idx in range(num_links):
        for slot in range(num_slots):
            names.append(f"link_{link_idx}_slot_{slot}")
    dims["bitmap"] = (start, len(names))

    start = len(names)
    for link_idx in range(num_links):
        names.append(f"link_{link_idx}_free_ratio")
        names.append(f"link_{link_idx}_largest_free")
        for b_idx in range(len(hist_edges)):
            names.append(f"link_{link_idx}_hist_bin_{b_idx}")
        names.append(f"link_{link_idx}_fragmentation")
    dims["per_link"] = (start, len(names))

    start = len(names)
    names.extend(
        [
            "global_occupancy",
            "global_free_ratio",
            "global_fragmentation_mean",
            "global_largest_free_block",
            "global_total_occupied_slot_hops",
        ]
    )
    dims["global"] = (start, len(names))

    start = len(names)
    for probe_idx in range(probe_count):
        names.append(f"probe_{probe_idx}_serviceable")
        names.append(f"probe_{probe_idx}_feasible_path_ratio")
    dims["probe"] = (start, len(names))

    start = len(names)
    for b_idx in range(len(residual_edges)):
        names.append(f"residual_bucket_{b_idx}_count")
        names.append(f"residual_bucket_{b_idx}_slot_hops")
    names.append("residual_total_active_count")
    names.append("residual_total_active_slot_hops")
    dims["residual"] = (start, len(names))

    return FeatureSchema(
        feature_names=names,
        dims=dims,
        link_order=link_ids,
        slot_order="0..num_slots-1",
        dtype="float32",
        config=feature_config,
    )
