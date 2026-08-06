"""Phase A component-ablation selectors (behavior-changing).

Protocol: `direct_sketch_exact_optimization_v1` Phase A.

- `OpportunityOnlyDirectSelector`: score = 1.0 * path_rank +
  opportunity_weight * sketch opportunity. MUST NOT instantiate
  `IncrementalDemandShadowPricer` or call any shadow API.
- `ShadowOnlyDirectSelector`: score = shadow marginal_losses +
  1.0 * path_rank. MUST NOT instantiate `CompressedFeasibleWindowSketch`
  or any probe component.
- `OpportunityOnlyDepth1Selector`: opportunity-only with sketch
  route_depth=1 (budget unchanged).

All three keep the locked Full Direct (`ExactOptimizedDirectCompressed
SketchSelector`) enumeration semantics: identical static route specs
(K=50 hops order, highest reachable modulation, required_fs widths),
identical bit-parallel legal-start enumeration (`_legal_starts_from_word`),
at most 3 feasible paths, all legal start slots, identical deterministic
tie-break (total, path_rank, start_slot), and identical RMSAAction /
BlockedAction construction.

Code reality note (vs. task brief): the sketch pricer
(`BitParallelPathOpportunityPricer` / `PathOpportunityPricer`) does NOT
expose an `_arc_index` attribute; it stores `_arc_ids` (tuple from
`env.spectrum.directed_link_ids()`) and only builds a local
`arc_index_by_id` during probe construction. The arc-index map used here
is therefore built as `{arc: i for i, arc in enumerate(pricer._arc_ids)}`,
which is the identical mapping to `DemandShadowPricer._arc_index` (both
enumerate `env.spectrum.directed_link_ids()` in order).
"""
from __future__ import annotations

import numpy as np

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
from sa_hmarl.pure_rmsa_v13.rollout_lab.feasible_window_sketch import (
    CompressedFeasibleWindowSketch,
    _legal_starts_from_word,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.incremental_shadow_pricing import (
    IncrementalDemandShadowPricer,
)


class _DirectStaticRoutes:
    """Static route specs + candidate scan, mirrored from the locked
    `ExactOptimizedDirectCompressedSketchSelector` but indexing arcs
    through `self._arc_index` instead of `self.shadow._arc_index`."""

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
                        self._arc_index[arc]
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

    def _init_route_state(self, env: PDSRMSAEnv) -> None:
        self._full_word_int = (1 << env.num_slots) - 1
        self._route_specs: dict[tuple[int, int, int], tuple] = {}
        self._blocked_reasons: dict[tuple[int, int], str] = {}
        self._spectrum_versions_seen = (
            env.spectrum.arc_versions_view().copy()
        )

    def _detect_changed_arcs(
        self,
        env: PDSRMSAEnv,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        bitmap = env.spectrum.occupied_bitmap_view()
        spectrum_versions = env.spectrum.arc_versions_view()
        changed = np.flatnonzero(
            spectrum_versions != self._spectrum_versions_seen
        )
        return bitmap, spectrum_versions, changed

    def _mark_versions_seen(
        self,
        spectrum_versions: np.ndarray,
        changed: np.ndarray,
    ) -> None:
        if changed.size:
            self._spectrum_versions_seen[changed] = spectrum_versions[changed]

    def _scan_candidates(
        self,
        env: PDSRMSAEnv,
        request: Request,
        arc_words_int: list[int],
    ) -> list[tuple]:
        """First 3 feasible paths with all legal starts, Full Direct order."""
        candidates = []
        feasible_paths = 0
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
        return candidates

    def _finalize(
        self,
        request: Request,
        best_action: RMSAAction | None,
    ) -> RMSAAction | BlockedAction:
        if best_action is not None:
            return best_action
        return BlockedAction(
            reason=self._blocked_reasons[
                (request.src_node, request.dst_node)
            ]
        )


class OpportunityOnlyDirectSelector(_DirectStaticRoutes):
    """rank + opportunity_weight * sketch opportunity; no shadow anywhere."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        *,
        budget: int,
        opportunity_weight: float,
        route_depth: int = 3,
    ) -> None:
        self.sketch = CompressedFeasibleWindowSketch(
            env,
            budget=budget,
            route_depth=route_depth,
            seed=20260730,
        )
        self.opportunity_weight = float(opportunity_weight)
        self._arc_index = {
            arc: index
            for index, arc in enumerate(self.sketch.pricer._arc_ids)
        }
        self._init_route_state(env)
        self._precompute_static_routes(env)

    def select(
        self,
        env: PDSRMSAEnv,
        request: Request,
    ) -> RMSAAction | BlockedAction:
        bitmap, spectrum_versions, changed = self._detect_changed_arcs(env)
        self.sketch.sync_state(env, bitmap, changed)
        self._mark_versions_seen(spectrum_versions, changed)
        candidates = self._scan_candidates(
            env,
            request,
            self.sketch.pricer._arc_words_int,
        )
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
            opportunity = self.sketch.score_path_starts(
                path,
                starts,
                width,
                normalize=True,
                cache_ready=True,
            )
            total = (
                float(path_rank)
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
        return self._finalize(request, best_action)


class OpportunityOnlyDepth1Selector(OpportunityOnlyDirectSelector):
    """Opportunity-only with sketch route_depth=1 (budget unchanged)."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        *,
        budget: int,
        opportunity_weight: float,
    ) -> None:
        super().__init__(
            env,
            budget=budget,
            opportunity_weight=opportunity_weight,
            route_depth=1,
        )


class ShadowOnlyDirectSelector(_DirectStaticRoutes):
    """shadow marginal_losses + rank; no sketch/probe anywhere."""

    def __init__(self, env: PDSRMSAEnv) -> None:
        self.shadow = IncrementalDemandShadowPricer(env)
        self._arc_index = self.shadow._arc_index
        self._init_route_state(env)
        self._precompute_static_routes(env)

    def select(
        self,
        env: PDSRMSAEnv,
        request: Request,
    ) -> RMSAAction | BlockedAction:
        bitmap, spectrum_versions, changed = self._detect_changed_arcs(env)
        # No sketch exists, so occupied_words cannot be shared; the shadow
        # recomputes its own arc words from the bitmap (occupied_words=None).
        self.shadow.sync_state(env, bitmap, changed)
        self._mark_versions_seen(spectrum_versions, changed)
        candidates = self._scan_candidates(
            env,
            request,
            self.shadow._arc_words_int,
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
            total = shadow_scores
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
        return self._finalize(request, best_action)
