"""Compressed end-to-end feasible-window sketches for opportunity pricing."""
from __future__ import annotations

import copy
import hashlib
import os

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import (
    BlockedAction,
    PDSRMSAEnv,
    RMSAAction,
)
from sa_hmarl.pds_rmsa.protocol import Request
from sa_hmarl.pds_rmsa.protocol import (
    BITRATE_CHOICES_GBPS,
    highest_modulation_for_distance,
    required_fs,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.bitparallel_path_opportunity import (
    BitParallelPathOpportunityPricer,
    _popcount_u64,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.demand_shadow_pricing import (
    _free_runs,
    rank_demand_shadow_actions,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.incremental_shadow_pricing import (
    IncrementalDemandShadowPricer,
)

try:
    from sa_hmarl.pure_rmsa_v13.rollout_lab._direct_sketch_kernel import (
        legal_starts_from_word as _compiled_legal_starts_from_word,
    )
except ImportError:
    _compiled_legal_starts_from_word = None

_NATIVE_KERNEL_ENABLED = (
    os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL") == "1"
)
LEGAL_START_BACKEND = (
    "c_extension"
    if _NATIVE_KERNEL_ENABLED
    and _compiled_legal_starts_from_word is not None
    else "python"
)

SKETCH_PROTOCOL = "pure_rmsa_compressed_feasible_window_sketch_v1"
EXACT_OPT_PROTOCOL = (
    "pure_rmsa_compressed_feasible_window_sketch_exact_opt_v1"
)
_BYTE_SET_BITS = tuple(
    tuple(bit for bit in range(8) if value & (1 << bit))
    for value in range(256)
)


def _legal_starts_from_word(
    available_word: int,
    width: int,
    num_slots: int,
) -> np.ndarray:
    """Return legal contiguous-window starts in ascending order."""
    if (
        _NATIVE_KERNEL_ENABLED
        and _compiled_legal_starts_from_word is not None
    ):
        return _compiled_legal_starts_from_word(
            available_word,
            width,
            num_slots,
        )
    if width <= 0 or width > num_slots:
        return np.zeros(0, dtype=np.intp)
    starts_word = int(available_word)
    for shift in range(1, width):
        starts_word &= int(available_word) >> shift
    starts_word &= (1 << (num_slots - width + 1)) - 1
    if starts_word == 0:
        return np.zeros(0, dtype=np.intp)
    if starts_word.bit_count() <= 12:
        starts: list[int] = []
        while starts_word:
            lowest = starts_word & -starts_word
            starts.append(lowest.bit_length() - 1)
            starts_word ^= lowest
        return np.asarray(starts, dtype=np.intp)
    starts: list[int] = []
    for byte_index in range((num_slots + 7) // 8):
        value = (starts_word >> (8 * byte_index)) & 0xFF
        base = 8 * byte_index
        starts.extend(base + bit for bit in _BYTE_SET_BITS[value])
    return np.asarray(starts, dtype=np.intp)


def _stable_key(path: tuple[int, ...], width: int, seed: int) -> bytes:
    payload = (
        f"{seed}|{width}|" + ",".join(str(node) for node in path)
    ).encode("ascii")
    return hashlib.blake2b(payload, digest_size=16).digest()


def stratified_probe_subset(
    pricer: BitParallelPathOpportunityPricer,
    budget: int,
    *,
    seed: int = 20260730,
) -> np.ndarray:
    """Choose a deterministic width-stratified probe subset."""
    probe_count = len(pricer._scoring_probes)
    if budget <= 0:
        raise ValueError("budget must be positive")
    if budget >= probe_count:
        return np.arange(probe_count, dtype=np.intp)
    widths = np.unique(pricer._probe_widths)
    selected: list[int] = []
    remaining_budget = budget
    remaining_probes = probe_count
    for width_index, width in enumerate(widths):
        indices = np.flatnonzero(pricer._probe_widths == width)
        if width_index == len(widths) - 1:
            allocation = remaining_budget
        else:
            allocation = max(
                1,
                int(round(remaining_budget * len(indices) / remaining_probes)),
            )
            allocation = min(allocation, len(indices), remaining_budget)
        ranked = sorted(
            (int(index) for index in indices),
            key=lambda index: _stable_key(
                pricer._scoring_probes[index].path,
                int(width),
                seed,
            ),
        )
        selected.extend(ranked[:allocation])
        remaining_budget -= allocation
        remaining_probes -= len(indices)
    if remaining_budget > 0:
        unselected = sorted(
            set(range(probe_count)) - set(selected),
            key=lambda index: _stable_key(
                pricer._scoring_probes[index].path,
                int(pricer._probe_widths[index]),
                seed + 1,
            ),
        )
        selected.extend(unselected[:remaining_budget])
    return np.asarray(sorted(selected), dtype=np.intp)


class CompressedFeasibleWindowSketch:
    """Incremental exact-window scorer over a fixed compressed probe subset."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        *,
        budget: int,
        route_depth: int = 3,
        seed: int = 20260730,
    ) -> None:
        self.pricer = BitParallelPathOpportunityPricer(
            env,
            route_depth=route_depth,
        )
        self.indices = stratified_probe_subset(
            self.pricer,
            budget,
            seed=seed,
        )
        self.budget = int(len(self.indices))
        self.full_probe_count = len(self.pricer._scoring_probes)
        self.selected_weight = float(
            np.sum(self.pricer._probe_weights[self.indices])
        )
        mutable: dict[tuple[int, int], list[int]] = {}
        selected_set = set(int(index) for index in self.indices)
        for arc, probe_indices in self.pricer._by_arc.items():
            kept = [
                int(index)
                for index in probe_indices
                if int(index) in selected_set
            ]
            if kept:
                mutable[arc] = kept
        self._by_arc = {
            arc: tuple(indices) for arc, indices in mutable.items()
        }
        self._relevant_by_action_path: dict[
            tuple[int, ...], tuple[int, ...]
        ] = {}
        self._relevant_array_by_action_path: dict[
            tuple[int, ...], np.ndarray
        ] = {}
        self._path_ids_by_action_path: dict[
            tuple[int, ...], np.ndarray
        ] = {}
        self._path_id_mark = np.zeros(
            len(self.pricer._probe_paths),
            dtype=bool,
        )

    def fork_for_env(
        self,
        env: PDSRMSAEnv,
    ) -> "CompressedFeasibleWindowSketch":
        """Fork mutable probe caches without rebuilding static probes."""
        fork = copy.copy(self)
        fork.pricer = self.pricer.fork_for_env(env)
        fork._relevant_by_action_path = (
            self._relevant_by_action_path.copy()
        )
        fork._relevant_array_by_action_path = (
            self._relevant_array_by_action_path.copy()
        )
        fork._path_ids_by_action_path = (
            self._path_ids_by_action_path.copy()
        )
        fork._path_id_mark = np.zeros_like(self._path_id_mark)
        return fork

    @property
    def compression_ratio(self) -> float:
        return self.budget / max(1, self.full_probe_count)

    def _relevant_indices(
        self,
        path: tuple[int, ...],
    ) -> np.ndarray:
        cached = self._relevant_array_by_action_path.get(path)
        if cached is not None:
            return cached
        relevant = tuple(
            sorted(
                {
                    index
                    for arc in zip(path[:-1], path[1:])
                    for index in self._by_arc.get(arc, ())
                }
            )
        )
        array = np.asarray(relevant, dtype=np.intp)
        self._relevant_by_action_path[path] = relevant
        self._relevant_array_by_action_path[path] = array
        self._path_ids_by_action_path[path] = np.unique(
            self.pricer._probe_path_ids[array]
        )
        return array

    def sync_state(
        self,
        env: PDSRMSAEnv,
        bitmap: np.ndarray,
        changed: np.ndarray | None = None,
    ) -> None:
        """Synchronize dirty probe paths against a shared state bitmap."""
        self.pricer._score_calls += 1
        self.pricer._sync_dirty_probes(env, bitmap, changed)

    def score_path_starts(
        self,
        path: tuple[int, ...],
        starts: np.ndarray,
        allocated_width: int,
        *,
        normalize: bool = True,
        cache_ready: bool = False,
    ) -> np.ndarray:
        """Score every start for one path without constructing actions."""
        starts = np.asarray(starts, dtype=np.intp)
        if starts.size == 0:
            return np.zeros(0, dtype=np.float64)
        probe_indices = self._relevant_indices(path)
        if probe_indices.size == 0:
            return np.zeros(starts.shape, dtype=np.float64)
        pricer = self.pricer
        if not cache_ready:
            path_ids = self._path_ids_by_action_path[path]
            pricer._ensure_path_ids_bit_cache(
                path_ids,
                bitwise_refresh=True,
            )
        destroy_masks = pricer._destroy_mask_table[
            pricer._probe_widths[probe_indices, None],
            starts[None, :],
            int(allocated_width),
        ]
        intersections = np.bitwise_and(
            pricer._feasible_start_masks[probe_indices, None],
            destroy_masks,
        )
        destroyed = _popcount_u64(intersections)
        scores = np.sum(
            destroyed
            * pricer._weighted_inverse_total[probe_indices, None],
            axis=0,
        )
        if normalize and self.selected_weight > 0.0:
            scores /= self.selected_weight
        return scores

    def ensure_paths(
        self,
        paths: list[tuple[int, ...]],
    ) -> None:
        """Refresh the union of all probe paths needed by candidate paths."""
        mark = self._path_id_mark
        mark.fill(False)
        has_paths = False
        for path in paths:
            self._relevant_indices(path)
            path_ids = self._path_ids_by_action_path[path]
            if path_ids.size:
                mark[path_ids] = True
                has_paths = True
        if not has_paths:
            return
        path_ids = np.flatnonzero(mark)
        self.pricer._ensure_path_ids_bit_cache(
            path_ids,
            bitwise_refresh=True,
        )

    def score_actions(
        self,
        env: PDSRMSAEnv,
        actions: list[RMSAAction],
        *,
        normalize: bool = True,
    ) -> np.ndarray:
        if not actions:
            return np.zeros(0, dtype=np.float64)
        pricer = self.pricer
        pricer._score_calls += 1
        pricer._sync_dirty_probes(env)
        relevant_by_action = []
        for action in actions:
            self._relevant_indices(action.path)
            relevant = self._relevant_by_action_path[action.path]
            relevant_by_action.append(relevant)
        affected = tuple(
            sorted(
                {
                    index
                    for relevant in relevant_by_action
                    for index in relevant
                }
            )
        )
        pricer._ensure_bit_cache(affected)

        scores = np.zeros(len(actions), dtype=np.float64)
        actions_by_path: dict[tuple[int, ...], list[int]] = {}
        for action_index, action in enumerate(actions):
            actions_by_path.setdefault(action.path, []).append(action_index)
        for path, action_indices in actions_by_path.items():
            relevant = self._relevant_by_action_path[path]
            if not relevant:
                continue
            probe_indices = np.asarray(relevant, dtype=np.intp)
            path_actions = [actions[index] for index in action_indices]
            destroy_masks = pricer._destroy_masks(
                pricer._probe_widths[probe_indices],
                path_actions,
            )
            intersections = np.bitwise_and(
                pricer._feasible_start_masks[probe_indices, None],
                destroy_masks,
            )
            destroyed = _popcount_u64(intersections)
            scores[action_indices] = np.sum(
                destroyed
                * pricer._weighted_inverse_total[probe_indices, None],
                axis=0,
            )
        if normalize and self.selected_weight > 0.0:
            scores /= self.selected_weight
        return scores


class DirectCompressedSketchSelector:
    """Analytical selector using exact shadow and a compressed probe sketch."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        *,
        budget: int,
        opportunity_weight: float,
    ) -> None:
        self.shadow = IncrementalDemandShadowPricer(env)
        self.sketch = CompressedFeasibleWindowSketch(
            env,
            budget=budget,
            route_depth=3,
            seed=20260730,
        )
        self.opportunity_weight = float(opportunity_weight)

    def select(
        self,
        env: PDSRMSAEnv,
        request: Request,
    ) -> RMSAAction | BlockedAction:
        ranked = rank_demand_shadow_actions(
            env,
            request,
            self.shadow,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
        if not ranked:
            legal = env.build_candidates(request)
            if len(legal) == 1 and isinstance(legal[0], BlockedAction):
                return legal[0]
            raise AssertionError("sketch pool empty while legal action exists")
        actions = [action for _, action in ranked]
        shadow_scores = np.asarray(
            [score for score, _ in ranked],
            dtype=np.float64,
        )
        opportunity = self.sketch.score_actions(
            env,
            actions,
            normalize=True,
        )
        total = shadow_scores + self.opportunity_weight * opportunity
        best = min(
            range(len(actions)),
            key=lambda index: (
                float(total[index]),
                actions[index].path_rank,
                actions[index].start_slot,
            ),
        )
        return actions[best]


class ExactOptimizedDirectCompressedSketchSelector:
    """Action-identical Direct Sketch with an optimized online hot path."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        *,
        budget: int,
        opportunity_weight: float,
    ) -> None:
        self.shadow = IncrementalDemandShadowPricer(env)
        self.sketch = CompressedFeasibleWindowSketch(
            env,
            budget=budget,
            route_depth=3,
            seed=20260730,
        )
        self.opportunity_weight = float(opportunity_weight)
        self._full_word_int = (1 << env.num_slots) - 1
        self._route_specs: dict[tuple[int, int, int], tuple] = {}
        self._blocked_reasons: dict[tuple[int, int], str] = {}
        self._spectrum_versions_seen = (
            env.spectrum.arc_versions_view().copy()
        )
        self._precompute_static_routes(env)

    def fork_for_env(
        self,
        env: PDSRMSAEnv,
    ) -> "ExactOptimizedDirectCompressedSketchSelector":
        """Fork dynamic selector state while sharing static route geometry."""
        if env.num_slots != self._full_word_int.bit_count():
            raise ValueError("cannot fork selector across slot counts")
        fork = copy.copy(self)
        fork.shadow = self.shadow.fork_for_env(env)
        fork.sketch = self.sketch.fork_for_env(env)
        fork._spectrum_versions_seen = (
            self._spectrum_versions_seen.copy()
        )
        return fork

    def _precompute_static_routes(self, env: PDSRMSAEnv) -> None:
        for src in range(env.topology.num_nodes):
            for dst in range(env.topology.num_nodes):
                if src == dst:
                    continue
                paths = env._cached_k_paths(
                    src,
                    dst,
                    env.k_paths,
                )
                base_specs = []
                for path_rank, path in enumerate(paths):
                    modulation = highest_modulation_for_distance(
                        env.topology.path_length_km(path)
                    )
                    if modulation is None:
                        continue
                    modulation_name, spectral_efficiency = modulation
                    arc_indices = tuple(
                        self.shadow._arc_index[arc]
                        for arc in zip(path[:-1], path[1:])
                    )
                    base_specs.append(
                        (
                            path_rank,
                            path,
                            modulation_name,
                            spectral_efficiency,
                            arc_indices,
                        )
                    )
                for bitrate in BITRATE_CHOICES_GBPS:
                    self._route_specs[(src, dst, bitrate)] = tuple(
                        (
                            path_rank,
                            path,
                            modulation_name,
                            required_fs(bitrate, spectral_efficiency),
                            arc_indices,
                        )
                        for (
                            path_rank,
                            path,
                            modulation_name,
                            spectral_efficiency,
                            arc_indices,
                        ) in base_specs
                    )
                if not paths:
                    reason = "no_candidate_path"
                elif not base_specs:
                    reason = "no_reach_feasible_path_mod"
                else:
                    reason = "insufficient_spectrum"
                self._blocked_reasons[(src, dst)] = reason

    def select(
        self,
        env: PDSRMSAEnv,
        request: Request,
    ) -> RMSAAction | BlockedAction:
        bitmap = env.spectrum.occupied_bitmap_view()
        spectrum_versions = env.spectrum.arc_versions_view()
        changed = np.flatnonzero(
            spectrum_versions != self._spectrum_versions_seen
        )
        self.sketch.sync_state(env, bitmap, changed)
        pricer = self.sketch.pricer
        self.shadow.sync_state(
            env,
            bitmap,
            changed,
            pricer._arc_words,
        )
        if changed.size:
            self._spectrum_versions_seen[changed] = spectrum_versions[changed]
        candidates = []
        feasible_paths = 0
        arc_words_int = pricer._arc_words_int
        legal_starts = _legal_starts_from_word
        full_word = self._full_word_int
        num_slots = env.num_slots
        for (
            path_rank,
            path,
            modulation_name,
            width,
            arc_indices,
        ) in self._route_specs[(
            request.src_node,
            request.dst_node,
            int(request.bitrate_gbps),
        )]:
            occupied_word = 0
            for arc_index in arc_indices:
                occupied_word |= arc_words_int[arc_index]
            starts = legal_starts(
                ~occupied_word & full_word,
                width,
                num_slots,
            )
            if starts.size == 0:
                continue
            candidates.append(
                (
                    path_rank,
                    path,
                    modulation_name,
                    width,
                    starts,
                )
            )
            feasible_paths += 1
            if feasible_paths >= 3:
                break

        self.sketch.ensure_paths(
            [candidate[1] for candidate in candidates]
        )
        best_key = None
        best_action = None
        for (
            path_rank,
            path,
            modulation_name,
            width,
            starts,
        ) in candidates:
            shadow_scores = self.shadow.marginal_losses(
                env,
                path,
                starts,
                width,
                bitmap=bitmap,
                assume_synced=True,
            )
            shadow_scores += float(path_rank)
            opportunity = self.sketch.score_path_starts(
                path,
                starts,
                width,
                normalize=True,
                cache_ready=True,
            )
            total = (
                shadow_scores
                + self.opportunity_weight * opportunity
            )
            local = int(np.argmin(total))
            key = (
                float(total[local]),
                path_rank,
                int(starts[local]),
            )
            if best_key is None or key < best_key:
                best_key = key
                best_action = RMSAAction(
                    action_id=(
                        path_rank,
                        modulation_name,
                        int(starts[local]),
                        width,
                    ),
                    path_rank=path_rank,
                    path=path,
                    modulation=modulation_name,
                    start_slot=int(starts[local]),
                    required_fs=width,
                )
        if best_action is not None:
            return best_action
        return BlockedAction(
            reason=self._blocked_reasons[
                (request.src_node, request.dst_node)
            ]
        )
