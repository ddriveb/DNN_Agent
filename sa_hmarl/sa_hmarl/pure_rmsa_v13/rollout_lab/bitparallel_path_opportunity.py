"""Exact uint64 implementation of incremental path-opportunity pricing."""
from __future__ import annotations

import copy

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import (
    BlockedAction,
    PDSRMSAEnv,
    RMSAAction,
)
from sa_hmarl.pds_rmsa.protocol import Request
from sa_hmarl.pure_rmsa_v13.rollout_lab.demand_shadow_pricing import (
    DemandShadowPricer,
    build_diverse_shadow_pool,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.path_opportunity_pricing import (
    PathOpportunityPricer,
)

def _window_mask(
    num_slots: int,
    probe_width: int,
    action_start: int,
    action_width: int,
) -> int:
    """Return probe starts destroyed by one action interval (Python int)."""
    lower = max(0, action_start - probe_width + 1)
    upper = min(
        num_slots - probe_width + 1,
        action_start + action_width,
    )
    if upper <= lower:
        return 0
    return ((1 << (upper - lower)) - 1) << lower


def _popcount_u64(values: np.ndarray) -> np.ndarray:
    """Vectorized uint64 population count with a portable SWAR fallback.

    Wide (object / >64-bit) inputs use int.bit_count(); the uint64 SWAR
    path is bitwise unchanged for <=64-slot runs.
    """
    arr = np.asarray(values)
    if arr.dtype == object or (
        arr.size and int(arr.max()) > (1 << 64) - 1
    ):
        return np.fromiter(
            (int(v).bit_count() for v in arr.reshape(-1)),
            dtype=np.int16,
        ).reshape(arr.shape)
    bitwise_count = getattr(np, "bitwise_count", None)
    if bitwise_count is not None:
        return bitwise_count(
            np.asarray(values, dtype=np.uint64)
        ).astype(np.int16, copy=False)
    result = np.asarray(values, dtype=np.uint64).copy()
    result -= (result >> np.uint64(1)) & np.uint64(
        0x5555555555555555
    )
    result = (result & np.uint64(0x3333333333333333)) + (
        (result >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    result = (result + (result >> np.uint64(4))) & np.uint64(
        0x0F0F0F0F0F0F0F0F
    )
    return (
        (result * np.uint64(0x0101010101010101)) >> np.uint64(56)
    ).astype(np.int16)


class BitParallelPathOpportunityPricer(PathOpportunityPricer):
    """Path-opportunity pricer using feasible-start uint64 masks."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        *,
        route_depth: int = 3,
        route_decay: float = 0.5,
        incremental_cache: bool = True,
    ) -> None:
        if not env.num_slots > 0:
            raise ValueError("pricer requires a positive slot count")
        # >64 slots use Python-int masks (arbitrary precision); <=64 keeps
        # the uint64 path bitwise unchanged.
        self._wide = env.num_slots > 64
        super().__init__(
            env,
            route_depth=route_depth,
            route_decay=route_decay,
            incremental_cache=incremental_cache,
        )
        if self._wide:
            self._full_word = (1 << env.num_slots) - 1
            self._slot_bits = np.asarray(
                [1 << index for index in range(env.num_slots)],
                dtype=object,
            )
        else:
            self._full_word = np.uint64((1 << env.num_slots) - 1)
            self._slot_bits = np.asarray(
                [np.uint64(1 << index) for index in range(env.num_slots)],
                dtype=np.uint64,
            )
        self._arc_words = self._bitmap_to_words(self._arc_bitmap)
        self._arc_words_int = [
            int(value) for value in self._arc_words
        ]
        self._arc_words_padded = np.zeros(
            len(self._arc_words) + 1,
            dtype=object if self._wide else np.uint64,
        )
        self._arc_words_padded[:-1] = self._arc_words
        sentinel_arc = len(self._arc_words)
        max_path_hops = max(
            (len(indices) for indices in self._path_arc_indices),
            default=1,
        )
        self._path_arc_index_matrix = np.full(
            (len(self._path_arc_indices), max_path_hops),
            sentinel_arc,
            dtype=np.intp,
        )
        for path_id, arc_indices in enumerate(self._path_arc_indices):
            self._path_arc_index_matrix[
                path_id,
                : len(arc_indices),
            ] = arc_indices
        paths_by_arc: list[list[int]] = [
            [] for _ in range(len(self._arc_ids))
        ]
        for path_id, arc_indices in enumerate(self._path_arc_indices):
            for arc_index in arc_indices:
                paths_by_arc[int(arc_index)].append(path_id)
        self._paths_by_arc = tuple(
            np.asarray(path_ids, dtype=np.intp)
            for path_ids in paths_by_arc
        )
        self._dirty_paths = np.ones(
            len(self._probe_paths),
            dtype=bool,
        )
        self._feasible_start_masks = np.zeros(
            len(self._scoring_probes),
            dtype=object if self._wide else np.uint64,
        )
        self._probe_weights = np.asarray(
            [probe.weight for probe in self._scoring_probes],
            dtype=np.float64,
        )
        self._max_probe_width = int(
            np.max(self._probe_widths, initial=0)
        )
        self._probe_index_by_path_width = np.full(
            (
                len(self._probe_paths),
                self._max_probe_width + 1,
            ),
            -1,
            dtype=np.intp,
        )
        for probe_index, (path_id, width) in enumerate(
            zip(self._probe_path_ids, self._probe_widths)
        ):
            self._probe_index_by_path_width[
                int(path_id),
                int(width),
            ] = probe_index
        self._destroy_mask_table = np.zeros(
            (env.num_slots + 1, env.num_slots, env.num_slots + 1),
            dtype=object if self._wide else np.uint64,
        )
        for probe_width in range(1, env.num_slots + 1):
            for action_start in range(env.num_slots):
                for action_width in range(1, env.num_slots + 1):
                    self._destroy_mask_table[
                        probe_width,
                        action_start,
                        action_width,
                    ] = _window_mask(
                        env.num_slots,
                        probe_width,
                        action_start,
                        action_width,
                    )

    def fork_for_env(
        self,
        env: PDSRMSAEnv,
    ) -> "BitParallelPathOpportunityPricer":
        """Fork mutable spectrum caches and share immutable probe geometry."""
        if env.num_slots != int(self._full_word).bit_count():
            raise ValueError("cannot fork opportunity pricer across slot counts")
        if tuple(env.spectrum.directed_link_ids()) != self._arc_ids:
            raise ValueError("cannot fork opportunity pricer across topologies")

        fork = copy.copy(self)
        fork._relevant_by_action_path = (
            self._relevant_by_action_path.copy()
        )
        fork._window_prefix = self._window_prefix.copy()
        fork._weighted_inverse_total = (
            self._weighted_inverse_total.copy()
        )
        fork._initialized = self._initialized.copy()
        fork._arc_bitmap = self._arc_bitmap.copy()
        fork._arc_versions = self._arc_versions.copy()
        fork._path_versions = self._path_versions.copy()
        fork._arc_words = self._arc_words.copy()
        fork._arc_words_int = self._arc_words_int.copy()
        fork._arc_words_padded = self._arc_words_padded.copy()
        fork._dirty_paths = self._dirty_paths.copy()
        fork._feasible_start_masks = self._feasible_start_masks.copy()
        return fork

    def _bitmap_to_words(self, bitmap: np.ndarray) -> np.ndarray:
        if self._wide:
            return np.asarray(
                bitmap.astype(object, copy=False) @ self._slot_bits,
                dtype=object,
            )
        return np.asarray(
            bitmap.astype(np.uint64, copy=False) @ self._slot_bits,
            dtype=np.uint64,
        )

    def _sync_dirty_probes(
        self,
        env: PDSRMSAEnv,
        bitmap: np.ndarray | None = None,
        changed: np.ndarray | None = None,
    ) -> None:
        current = (
            env.spectrum.as_bitmap()
            if bitmap is None
            else np.asarray(bitmap, dtype=bool)
        )
        if current.shape != self._arc_bitmap.shape:
            raise ValueError("spectrum shape changed after pricer construction")
        if changed is None:
            changed = np.flatnonzero(
                np.any(current != self._arc_bitmap, axis=1)
            )
        else:
            changed = np.asarray(changed, dtype=np.intp)
        if changed.size == 0:
            return
        self._state_version += 1
        self._arc_versions[changed] = self._state_version
        self._arc_bitmap[changed] = current[changed]
        self._arc_words[changed] = self._bitmap_to_words(current[changed])
        self._arc_words_padded[changed] = self._arc_words[changed]
        for arc_index in changed:
            index = int(arc_index)
            self._arc_words_int[index] = int(self._arc_words[index])
        for arc_index in changed:
            path_ids = self._paths_by_arc[int(arc_index)]
            if path_ids.size:
                self._dirty_paths[path_ids] = True
        self._arc_change_events += int(changed.size)

    def _ensure_bit_cache(
        self,
        indices: tuple[int, ...],
    ) -> None:
        if not indices:
            return
        probe_indices = np.asarray(indices, dtype=np.intp)
        path_ids = np.unique(self._probe_path_ids[probe_indices])
        self._ensure_path_ids_bit_cache(path_ids)

    def _ensure_path_ids_bit_cache(
        self,
        path_ids: np.ndarray,
        *,
        bitwise_refresh: bool = False,
    ) -> None:
        """Refresh exact path masks for pre-resolved path identifiers."""
        path_ids = np.asarray(path_ids, dtype=np.intp)
        if path_ids.size == 0:
            return
        self._path_cache_checks += int(len(path_ids))
        if self._incremental_cache:
            refresh_ids = path_ids[self._dirty_paths[path_ids]]
            self._path_cache_hits += int(len(path_ids) - len(refresh_ids))
        else:
            refresh_ids = path_ids
        if refresh_ids.size == 0:
            return

        if bitwise_refresh:
            occupied = np.bitwise_or.reduce(
                self._arc_words_padded[
                    self._path_arc_index_matrix[refresh_ids]
                ],
                axis=1,
            )
            free_words = (
                np.bitwise_not(occupied) & self._full_word
            )
        else:
            occupied_counts = self._path_incidence[refresh_ids].dot(
                self._arc_bitmap.astype(np.uint8, copy=False)
            )
            free_words = self._bitmap_to_words(
                np.asarray(occupied_counts == 0)
            )
        starts = free_words.copy()
        refreshed_probes = 0
        for width in range(1, self._max_probe_width + 1):
            if width > 1:
                starts = np.bitwise_and(
                    starts,
                    free_words >> (
                        int(width - 1) if self._wide else np.uint64(width - 1)
                    ),
                )
            probe_indices = self._probe_index_by_path_width[
                refresh_ids,
                width,
            ]
            valid = probe_indices >= 0
            if not np.any(valid):
                continue
            selected_indices = probe_indices[valid]
            selected_starts = starts[valid]
            totals = (
                np.fromiter(
                    (int(value).bit_count() for value in selected_starts),
                    dtype=np.float64,
                )
                if self._wide else _popcount_u64(selected_starts)
            )
            inverse = np.zeros(len(selected_indices), dtype=np.float64)
            positive = totals > 0
            inverse[positive] = (
                self._probe_weights[selected_indices[positive]]
                / totals[positive]
            )
            self._feasible_start_masks[selected_indices] = selected_starts
            self._weighted_inverse_total[selected_indices] = inverse
            self._initialized[selected_indices] = True
            refreshed_probes += int(len(selected_indices))

        self._path_versions[refresh_ids] = self._state_version
        self._dirty_paths[refresh_ids] = False
        self._probe_refreshes += refreshed_probes
        self._path_refreshes += int(len(refresh_ids))

    def _destroy_masks(
        self,
        probe_widths: np.ndarray,
        actions: list[RMSAAction],
    ) -> np.ndarray:
        starts = np.asarray(
            [action.start_slot for action in actions],
            dtype=np.intp,
        )
        action_widths = np.asarray(
            [action.required_fs for action in actions],
            dtype=np.intp,
        )
        return self._destroy_mask_table[
            probe_widths[:, None],
            starts[None, :],
            action_widths[None, :],
        ]

    def score_actions(
        self,
        env: PDSRMSAEnv,
        actions: list[RMSAAction],
        *,
        normalize: bool = False,
    ) -> np.ndarray:
        if not actions:
            return np.zeros(0, dtype=np.float64)
        self._score_calls += 1
        self._sync_dirty_probes(env)
        relevant_by_action = []
        for action in actions:
            relevant = self._relevant_by_action_path.get(action.path)
            if relevant is None:
                arcs = zip(action.path[:-1], action.path[1:])
                relevant = tuple(
                    sorted(
                        {
                            index
                            for arc in arcs
                            for index in self._by_arc.get(arc, ())
                        }
                    )
                )
                self._relevant_by_action_path[action.path] = relevant
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
        self._ensure_bit_cache(affected)

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
            destroy_masks = self._destroy_masks(
                self._probe_widths[probe_indices],
                path_actions,
            )
            intersections = np.bitwise_and(
                self._feasible_start_masks[probe_indices, None],
                destroy_masks,
            )
            destroyed = (
                np.fromiter(
                    (int(value).bit_count() for value in intersections.reshape(-1)),
                    dtype=np.float64,
                ).reshape(intersections.shape)
                if self._wide else _popcount_u64(intersections)
            )
            scores[action_indices] = np.sum(
                destroyed
                * self._weighted_inverse_total[probe_indices, None],
                axis=0,
            )
        if normalize and self._total_probe_weight > 0.0:
            scores /= self._total_probe_weight
        return scores


def rank_bitparallel_path_opportunity_actions(
    env: PDSRMSAEnv,
    request: Request,
    shadow_pricer: DemandShadowPricer,
    opportunity_pricer: BitParallelPathOpportunityPricer,
    *,
    top_n: int = 12,
    opportunity_weight: float = 0.0,
    normalize_opportunity: bool = False,
    shadow_weight: float = 1.0,
) -> list[tuple[float, RMSAAction]]:
    """Rerank the canonical diverse shadow pool with uint64 pricing."""
    if opportunity_weight < 0.0:
        raise ValueError("opportunity_weight must be non-negative")
    if shadow_weight < 0.0:
        raise ValueError("shadow_weight must be non-negative")
    pool = build_diverse_shadow_pool(
        env,
        request,
        shadow_pricer,
        top_n=top_n,
    )
    if not pool:
        return []
    actions = [action for _, action in pool]
    opportunity = opportunity_pricer.score_actions(
        env,
        actions,
        normalize=normalize_opportunity,
    )
    ranked = [
        (
            float(
                shadow_weight * shadow_score
                + opportunity_weight * opportunity_score
            ),
            action,
        )
        for (shadow_score, action), opportunity_score in zip(
            pool,
            opportunity,
        )
    ]
    ranked.sort(
        key=lambda item: (
            item[0],
            item[1].path_rank,
            item[1].start_slot,
        )
    )
    return ranked


def bitparallel_path_opportunity_action(
    env: PDSRMSAEnv,
    request: Request,
    shadow_pricer: DemandShadowPricer,
    opportunity_pricer: BitParallelPathOpportunityPricer,
    *,
    top_n: int = 12,
    opportunity_weight: float = 0.0,
    normalize_opportunity: bool = False,
    shadow_weight: float = 1.0,
) -> RMSAAction | BlockedAction:
    ranked = rank_bitparallel_path_opportunity_actions(
        env,
        request,
        shadow_pricer,
        opportunity_pricer,
        top_n=top_n,
        opportunity_weight=opportunity_weight,
        normalize_opportunity=normalize_opportunity,
        shadow_weight=shadow_weight,
    )
    if ranked:
        return ranked[0][1]
    legal = env.build_candidates(request)
    if len(legal) == 1 and isinstance(legal[0], BlockedAction):
        return legal[0]
    raise AssertionError("bit-parallel pool empty while legal actions exist")
