"""End-to-end route opportunity loss on top of demand-shadow pricing."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix

from sa_hmarl.pds_rmsa.env.rmsa_env import (
    BlockedAction,
    PDSRMSAEnv,
    RMSAAction,
)
from sa_hmarl.pds_rmsa.protocol import (
    BITRATE_CHOICES_GBPS,
    Request,
    highest_modulation_for_distance,
    required_fs,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.demand_shadow_pricing import (
    DemandShadowPricer,
    _free_runs,
    _window_count,
    build_diverse_shadow_pool,
)


@dataclass(frozen=True)
class OpportunityProbe:
    path: tuple[int, ...]
    arcs: frozenset[tuple[int, int]]
    width: int
    weight: float


class PathOpportunityPricer:
    """Price exact end-to-end service windows destroyed by an action."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        *,
        route_depth: int = 3,
        route_decay: float = 0.5,
        incremental_cache: bool = True,
    ) -> None:
        if route_depth <= 0:
            raise ValueError("route_depth must be positive")
        if not 0.0 < route_decay <= 1.0:
            raise ValueError("route_decay must be in (0, 1]")
        self._incremental_cache = bool(incremental_cache)
        probes = []
        depth = min(route_depth, env.k_paths)
        for src in range(env.topology.num_nodes):
            for dst in range(env.topology.num_nodes):
                if src == dst:
                    continue
                paths = env._cached_k_paths(src, dst, depth)
                if not paths:
                    continue
                route_weights = np.asarray(
                    [route_decay**rank for rank in range(len(paths))],
                    dtype=np.float64,
                )
                route_weights /= route_weights.sum()
                for route_weight, path in zip(route_weights, paths):
                    modulation = highest_modulation_for_distance(
                        env.topology.path_length_km(path)
                    )
                    if modulation is None:
                        continue
                    _, spectral_efficiency = modulation
                    arcs = frozenset(zip(path[:-1], path[1:]))
                    for bitrate in BITRATE_CHOICES_GBPS:
                        width = required_fs(
                            bitrate, spectral_efficiency
                        )
                        if width <= env.num_slots:
                            probes.append(
                                OpportunityProbe(
                                    path=path,
                                    arcs=arcs,
                                    width=width,
                                    weight=float(
                                        route_weight
                                        / len(BITRATE_CHOICES_GBPS)
                                    ),
                                )
                            )
        self._probes = tuple(probes)
        self._total_probe_weight = float(
            sum(probe.weight for probe in self._probes)
        )
        aggregated: dict[
            tuple[tuple[int, ...], int], OpportunityProbe
        ] = {}
        for probe in self._probes:
            key = (probe.path, probe.width)
            previous = aggregated.get(key)
            aggregated[key] = OpportunityProbe(
                path=probe.path,
                arcs=probe.arcs,
                width=probe.width,
                weight=probe.weight
                + (previous.weight if previous is not None else 0.0),
            )
        self._scoring_probes = tuple(aggregated.values())
        mutable: dict[tuple[int, int], list[int]] = {}
        for index, probe in enumerate(self._scoring_probes):
            for arc in probe.arcs:
                mutable.setdefault(arc, []).append(index)
        self._by_arc = {
            arc: tuple(indices) for arc, indices in mutable.items()
        }
        self._relevant_by_action_path: dict[
            tuple[int, ...], tuple[int, ...]
        ] = {}
        self._indices_by_probe_path: dict[
            tuple[int, ...], tuple[int, ...]
        ] = {}
        mutable_paths: dict[tuple[int, ...], list[int]] = {}
        for index, probe in enumerate(self._scoring_probes):
            mutable_paths.setdefault(probe.path, []).append(index)
        self._indices_by_probe_path = {
            path: tuple(indices)
            for path, indices in mutable_paths.items()
        }
        self._probe_paths = tuple(self._indices_by_probe_path)
        path_id_by_path = {
            path: index for index, path in enumerate(self._probe_paths)
        }
        self._probe_path_ids = np.asarray(
            [
                path_id_by_path[probe.path]
                for probe in self._scoring_probes
            ],
            dtype=np.int32,
        )
        self._probe_widths = np.asarray(
            [probe.width for probe in self._scoring_probes],
            dtype=np.int16,
        )
        probe_count = len(self._scoring_probes)
        self._window_prefix = np.zeros(
            (probe_count, env.num_slots + 1), dtype=np.int16
        )
        self._weighted_inverse_total = np.zeros(
            probe_count, dtype=np.float64
        )
        self._initialized = np.zeros(probe_count, dtype=bool)
        self._arc_ids = tuple(env.spectrum.directed_link_ids())
        arc_index_by_id = {
            arc: index for index, arc in enumerate(self._arc_ids)
        }
        self._path_arc_indices = tuple(
            np.asarray(
                [
                    arc_index_by_id[arc]
                    for arc in zip(path[:-1], path[1:])
                ],
                dtype=np.intp,
            )
            for path in self._probe_paths
        )
        incidence_rows = []
        incidence_cols = []
        for path_id, arc_indices in enumerate(self._path_arc_indices):
            incidence_rows.extend([path_id] * len(arc_indices))
            incidence_cols.extend(int(index) for index in arc_indices)
        self._path_incidence = csr_matrix(
            (
                np.ones(len(incidence_rows), dtype=np.uint8),
                (incidence_rows, incidence_cols),
            ),
            shape=(len(self._probe_paths), len(self._arc_ids)),
            dtype=np.uint8,
        )
        self._arc_bitmap = env.spectrum.as_bitmap().copy()
        self._arc_versions = np.zeros(len(self._arc_ids), dtype=np.int64)
        self._path_versions = np.full(
            len(self._probe_paths), -1, dtype=np.int64
        )
        self._state_version = 0
        self._score_calls = 0
        self._path_refreshes = 0
        self._probe_refreshes = 0
        self._arc_change_events = 0
        self._path_cache_checks = 0
        self._path_cache_hits = 0

    @property
    def probe_count(self) -> int:
        return len(self._probes)

    @property
    def total_probe_weight(self) -> float:
        return self._total_probe_weight

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "score_calls": self._score_calls,
            "path_refreshes": self._path_refreshes,
            "probe_refreshes": self._probe_refreshes,
            "arc_change_events": self._arc_change_events,
            "cached_paths": int(np.sum(self._path_versions >= 0)),
            "path_cache_checks": self._path_cache_checks,
            "path_cache_hits": self._path_cache_hits,
        }

    def _sync_dirty_probes(self, env: PDSRMSAEnv) -> None:
        current = env.spectrum.as_bitmap()
        if current.shape != self._arc_bitmap.shape:
            raise ValueError("spectrum shape changed after pricer construction")
        changed = np.flatnonzero(np.any(current != self._arc_bitmap, axis=1))
        if changed.size == 0:
            return
        self._state_version += 1
        self._arc_versions[changed] = self._state_version
        self._arc_bitmap[changed] = current[changed]
        self._arc_change_events += int(changed.size)

    def _ensure_probe_cache(
        self,
        env: PDSRMSAEnv,
        indices: tuple[int, ...],
    ) -> None:
        if not indices:
            return
        path_ids = np.unique(
            self._probe_path_ids[np.asarray(indices, dtype=np.intp)]
        )
        self._path_cache_checks += int(len(path_ids))
        refresh_ids = []
        refresh_versions = []
        for path_id_value in path_ids:
            path_id = int(path_id_value)
            arc_indices = self._path_arc_indices[path_id]
            current_version = int(
                np.max(self._arc_versions[arc_indices], initial=0)
            )
            if (
                self._incremental_cache
                and self._path_versions[path_id] >= current_version
            ):
                self._path_cache_hits += 1
                continue
            refresh_ids.append(path_id)
            refresh_versions.append(current_version)
        if not refresh_ids:
            return
        occupied_counts = self._path_incidence[
            np.asarray(refresh_ids, dtype=np.intp)
        ].dot(self._arc_bitmap.astype(np.uint8, copy=False))
        available_by_path = np.asarray(occupied_counts == 0)
        for row_index, (path_id, current_version) in enumerate(
            zip(refresh_ids, refresh_versions)
        ):
            path = self._probe_paths[path_id]
            available = available_by_path[row_index]
            runs = _free_runs(available)
            path_indices = self._indices_by_probe_path[path]
            for index in path_indices:
                probe = self._scoring_probes[index]
                num_starts = env.num_slots - probe.width + 1
                windows = np.zeros(num_starts, dtype=np.int16)
                for left, right in runs:
                    last = right - probe.width + 1
                    if last > left:
                        windows[left:last] = 1
                prefix = self._window_prefix[index]
                prefix.fill(0)
                prefix[1 : len(windows) + 1] = np.cumsum(
                    windows, dtype=np.int16
                )
                total = int(prefix[len(windows)])
                if total > 0:
                    self._weighted_inverse_total[index] = (
                        probe.weight / total
                    )
                else:
                    self._weighted_inverse_total[index] = 0.0
                self._initialized[index] = True
                self._probe_refreshes += 1
            self._path_versions[path_id] = current_version
            self._path_refreshes += 1

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
        self._ensure_probe_cache(env, affected)
        scores = np.zeros(len(actions), dtype=np.float64)
        actions_by_path: dict[tuple[int, ...], list[int]] = {}
        for action_index, action in enumerate(actions):
            actions_by_path.setdefault(action.path, []).append(action_index)
        for path, action_indices in actions_by_path.items():
            relevant = self._relevant_by_action_path[path]
            if not relevant:
                continue
            probe_indices = np.asarray(relevant, dtype=np.intp)
            widths = self._probe_widths[probe_indices, None]
            starts = np.asarray(
                [actions[index].start_slot for index in action_indices],
                dtype=np.int16,
            )[None, :]
            ends = np.asarray(
                [
                    actions[index].start_slot
                    + actions[index].required_fs
                    for index in action_indices
                ],
                dtype=np.int16,
            )[None, :]
            lower = np.maximum(0, starts - widths + 1)
            upper = np.minimum(env.num_slots - widths + 1, ends)
            prefix = self._window_prefix[probe_indices]
            destroyed = (
                np.take_along_axis(prefix, upper, axis=1)
                - np.take_along_axis(prefix, lower, axis=1)
            )
            scores[action_indices] = np.sum(
                destroyed
                * self._weighted_inverse_total[probe_indices, None],
                axis=0,
            )
        if normalize and self._total_probe_weight > 0.0:
            scores /= self._total_probe_weight
        return scores


def rank_path_opportunity_actions(
    env: PDSRMSAEnv,
    request: Request,
    shadow_pricer: DemandShadowPricer,
    opportunity_pricer: PathOpportunityPricer,
    *,
    top_n: int = 12,
    opportunity_weight: float = 0.0,
    normalize_opportunity: bool = False,
    shadow_weight: float = 1.0,
) -> list[tuple[float, RMSAAction]]:
    """Rerank a diverse shadow pool by exact path-level opportunity loss."""
    if opportunity_weight < 0.0:
        raise ValueError("opportunity_weight must be non-negative")
    if shadow_weight < 0.0:
        raise ValueError("shadow_weight must be non-negative")
    pool = build_diverse_shadow_pool(
        env, request, shadow_pricer, top_n=top_n
    )
    if not pool:
        return []
    actions = [action for _, action in pool]
    opportunity = opportunity_pricer.score_actions(
        env, actions, normalize=normalize_opportunity
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
            pool, opportunity
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


def path_opportunity_action(
    env: PDSRMSAEnv,
    request: Request,
    shadow_pricer: DemandShadowPricer,
    opportunity_pricer: PathOpportunityPricer,
    *,
    top_n: int = 12,
    opportunity_weight: float = 0.0,
    normalize_opportunity: bool = False,
    shadow_weight: float = 1.0,
) -> RMSAAction | BlockedAction:
    ranked = rank_path_opportunity_actions(
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
    candidates = env.build_candidates(request)
    if len(candidates) == 1 and isinstance(candidates[0], BlockedAction):
        return candidates[0]
    raise AssertionError("opportunity pool empty while legal actions exist")
