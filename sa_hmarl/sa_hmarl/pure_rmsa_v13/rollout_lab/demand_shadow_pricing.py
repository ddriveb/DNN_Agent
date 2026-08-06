"""Demand-aware spectrum placement on the first feasible KSP path.

The heuristic keeps the KSP-FF route and modulation decisions fixed.  It only
changes the start slot by minimizing the marginal loss of future contiguous
windows on the directed links used by that path.
"""
from __future__ import annotations

from dataclasses import dataclass

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


def _free_runs(available: np.ndarray) -> list[tuple[int, int]]:
    """Return half-open free runs as ``(left, right)``."""
    runs: list[tuple[int, int]] = []
    left: int | None = None
    for slot, free in enumerate(available):
        if bool(free) and left is None:
            left = slot
        elif not bool(free) and left is not None:
            runs.append((left, slot))
            left = None
    if left is not None:
        runs.append((left, len(available)))
    return runs


def _window_count(run_length: int, width: int) -> int:
    return max(0, run_length - width + 1)


def _destroyed_windows(
    run_left: int,
    run_right: int,
    start_slot: int,
    allocated_width: int,
    future_width: int,
) -> int:
    """Count future windows destroyed by one allocation inside a free run."""
    total = _window_count(run_right - run_left, future_width)
    left_remaining = start_slot - run_left
    right_remaining = run_right - (start_slot + allocated_width)
    surviving = _window_count(left_remaining, future_width) + _window_count(
        right_remaining, future_width
    )
    return total - surviving


@dataclass(frozen=True)
class ShadowPriceConfig:
    """Static demand model used to price dynamic spectrum windows."""

    route_depth: int = 10
    route_decay: float = 0.5


class DemandShadowPricer:
    """Topology-demand prior plus exact per-link marginal window loss."""

    def __init__(
        self,
        env: PDSRMSAEnv,
        config: ShadowPriceConfig = ShadowPriceConfig(),
    ) -> None:
        if config.route_depth <= 0:
            raise ValueError("route_depth must be positive")
        if not 0.0 < config.route_decay <= 1.0:
            raise ValueError("route_decay must be in (0, 1]")
        self.config = config
        self._arcs = tuple(env.spectrum.directed_link_ids())
        self._arc_index = {arc: index for index, arc in enumerate(self._arcs)}
        self._weights = self._build_demand_weights(env)

    def _build_demand_weights(self, env: PDSRMSAEnv) -> np.ndarray:
        weights = np.zeros(
            (len(self._arcs), env.num_slots + 1), dtype=np.float64
        )
        depth = min(self.config.route_depth, env.k_paths)
        for src in range(env.topology.num_nodes):
            for dst in range(env.topology.num_nodes):
                if src == dst:
                    continue
                paths = env._cached_k_paths(src, dst, depth)
                if not paths:
                    continue
                route_weights = np.asarray(
                    [
                        self.config.route_decay**rank
                        for rank in range(len(paths))
                    ],
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
                    for bitrate in BITRATE_CHOICES_GBPS:
                        width = required_fs(bitrate, spectral_efficiency)
                        if width > env.num_slots:
                            continue
                        demand_weight = route_weight / len(BITRATE_CHOICES_GBPS)
                        for arc in zip(path[:-1], path[1:]):
                            weights[self._arc_index[arc], width] += demand_weight

        positive = weights[weights > 0]
        if positive.size:
            weights /= float(positive.mean())
        return weights

    @property
    def demand_weights(self) -> np.ndarray:
        return self._weights.copy()

    def marginal_loss(
        self,
        env: PDSRMSAEnv,
        path: tuple[int, ...],
        start_slot: int,
        allocated_width: int,
    ) -> float:
        """Return normalized future-window loss caused by an allocation."""
        loss = 0.0
        for arc in zip(path[:-1], path[1:]):
            row = self._arc_index[arc]
            available = ~env.spectrum.as_bitmap()[row]
            runs = _free_runs(available)
            containing = next(
                (
                    (left, right)
                    for left, right in runs
                    if left <= start_slot
                    and start_slot + allocated_width <= right
                ),
                None,
            )
            if containing is None:
                return float("inf")

            totals = np.zeros(env.num_slots + 1, dtype=np.float64)
            for left, right in runs:
                run_length = right - left
                for width in np.flatnonzero(self._weights[row]):
                    totals[width] += _window_count(run_length, int(width))

            run_left, run_right = containing
            for width in np.flatnonzero(self._weights[row]):
                total = totals[width]
                if total <= 0:
                    continue
                destroyed = _destroyed_windows(
                    run_left,
                    run_right,
                    start_slot,
                    allocated_width,
                    int(width),
                )
                loss += self._weights[row, width] * destroyed / total
        return float(loss)

    def marginal_losses(
        self,
        env: PDSRMSAEnv,
        path: tuple[int, ...],
        start_slots: np.ndarray,
        allocated_width: int,
        *,
        bitmap: np.ndarray | None = None,
    ) -> np.ndarray:
        """Vectorized exact marginal loss for starts on one path."""
        starts = np.asarray(start_slots, dtype=np.int16)
        losses = np.zeros(starts.shape, dtype=np.float64)
        if starts.size == 0:
            return losses
        state = env.spectrum.as_bitmap() if bitmap is None else bitmap
        for arc in zip(path[:-1], path[1:]):
            row = self._arc_index[arc]
            runs = _free_runs(~state[row])
            run_left = np.full(starts.shape, -1, dtype=np.int16)
            run_right = np.full(starts.shape, -1, dtype=np.int16)
            for left, right in runs:
                contained = (starts >= left) & (
                    starts + allocated_width <= right
                )
                run_left[contained] = left
                run_right[contained] = right
            invalid = run_left < 0
            if np.any(invalid):
                losses[invalid] = np.inf

            positive_widths = np.flatnonzero(self._weights[row])
            for future_width in positive_widths:
                total = sum(
                    _window_count(right - left, int(future_width))
                    for left, right in runs
                )
                if total <= 0:
                    continue
                run_total = np.maximum(
                    0, run_right - run_left - future_width + 1
                )
                left_surviving = np.maximum(
                    0, starts - run_left - future_width + 1
                )
                right_surviving = np.maximum(
                    0,
                    run_right
                    - (starts + allocated_width)
                    - future_width
                    + 1,
                )
                destroyed = (
                    run_total - left_surviving - right_surviving
                )
                destroyed[invalid] = 0
                losses += (
                    self._weights[row, future_width]
                    * destroyed
                    / total
                )
        return losses


def demand_shadow_action(
    env: PDSRMSAEnv,
    request: Request,
    pricer: DemandShadowPricer,
) -> RMSAAction | BlockedAction:
    """Keep KSP-FF path/modulation and choose the lowest-price legal start."""
    paths = env._cached_k_paths(request.src_node, request.dst_node, env.k_paths)
    reachable = 0
    bitmap = env.spectrum.as_bitmap()
    for path_rank, path in enumerate(paths):
        modulation = highest_modulation_for_distance(
            env.topology.path_length_km(path)
        )
        if modulation is None:
            continue
        reachable += 1
        modulation_name, spectral_efficiency = modulation
        width = required_fs(request.bitrate_gbps, spectral_efficiency)
        available = env.spectrum.get_available_slots(path)
        candidates: list[tuple[tuple[float, float, int], RMSAAction]] = []
        pending = []
        for run_left, run_right in _free_runs(available):
            if run_right - run_left < width:
                continue
            run_size = run_right - run_left
            waste = (run_size - width) / run_size
            for start_slot in range(run_left, run_right - width + 1):
                action = RMSAAction(
                    action_id=(
                        path_rank,
                        modulation_name,
                        start_slot,
                        width,
                    ),
                    path_rank=path_rank,
                    path=path,
                    modulation=modulation_name,
                    start_slot=start_slot,
                    required_fs=width,
                )
                pending.append((waste, start_slot, action))
        prices = pricer.marginal_losses(
            env,
            path,
            np.asarray([item[1] for item in pending]),
            width,
            bitmap=bitmap,
        )
        for price, (waste, start_slot, action) in zip(prices, pending):
                candidates.append(((price, waste, start_slot), action))
        if candidates:
            return min(candidates, key=lambda item: item[0])[1]

    reason = (
        "no_candidate_path"
        if not paths
        else "no_reach_feasible_path_mod"
        if reachable == 0
        else "insufficient_spectrum"
    )
    return BlockedAction(reason=reason)


def demand_shadow_multipath_action(
    env: PDSRMSAEnv,
    request: Request,
    pricer: DemandShadowPricer,
    *,
    max_feasible_paths: int = 3,
    path_penalty: float = 0.0,
) -> RMSAAction | BlockedAction:
    """Select across a small feasible-path prefix with a KSP-rank anchor."""
    if max_feasible_paths <= 0:
        raise ValueError("max_feasible_paths must be positive")
    if path_penalty < 0:
        raise ValueError("path_penalty must be non-negative")

    paths = env._cached_k_paths(request.src_node, request.dst_node, env.k_paths)
    reachable = 0
    feasible_paths = 0
    candidates: list[
        tuple[tuple[float, float, int, float, int], RMSAAction]
    ] = []
    bitmap = env.spectrum.as_bitmap()
    for path_rank, path in enumerate(paths):
        modulation = highest_modulation_for_distance(
            env.topology.path_length_km(path)
        )
        if modulation is None:
            continue
        reachable += 1
        modulation_name, spectral_efficiency = modulation
        width = required_fs(request.bitrate_gbps, spectral_efficiency)
        available = env.spectrum.get_available_slots(path)
        path_candidates = []
        pending = []
        for run_left, run_right in _free_runs(available):
            if run_right - run_left < width:
                continue
            run_size = run_right - run_left
            waste = (run_size - width) / run_size
            for start_slot in range(run_left, run_right - width + 1):
                action = RMSAAction(
                    action_id=(
                        path_rank,
                        modulation_name,
                        start_slot,
                        width,
                    ),
                    path_rank=path_rank,
                    path=path,
                    modulation=modulation_name,
                    start_slot=start_slot,
                    required_fs=width,
                )
                pending.append((waste, start_slot, action))
        prices = pricer.marginal_losses(
            env,
            path,
            np.asarray([item[1] for item in pending]),
            width,
            bitmap=bitmap,
        )
        for price, (waste, start_slot, action) in zip(prices, pending):
                total = price + path_penalty * path_rank
                path_candidates.append(
                    (
                        (total, price, path_rank, waste, start_slot),
                        action,
                    )
                )
        if path_candidates:
            candidates.extend(path_candidates)
            feasible_paths += 1
            if feasible_paths >= max_feasible_paths:
                break

    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    reason = (
        "no_candidate_path"
        if not paths
        else "no_reach_feasible_path_mod"
        if reachable == 0
        else "insufficient_spectrum"
    )
    return BlockedAction(reason=reason)


def rank_demand_shadow_actions(
    env: PDSRMSAEnv,
    request: Request,
    pricer: DemandShadowPricer,
    *,
    max_feasible_paths: int = 3,
    path_penalty: float = 1.0,
) -> list[tuple[float, RMSAAction]]:
    """Return every expanded action in the exact multipath selector order."""
    ranked, _, _ = _rank_demand_shadow_actions_with_anchors(
        env,
        request,
        pricer,
        max_feasible_paths=max_feasible_paths,
        path_penalty=path_penalty,
    )
    return ranked


def _rank_demand_shadow_actions_with_anchors(
    env: PDSRMSAEnv,
    request: Request,
    pricer: DemandShadowPricer,
    *,
    max_feasible_paths: int,
    path_penalty: float,
) -> tuple[
    list[tuple[float, RMSAAction]],
    RMSAAction | None,
    RMSAAction | None,
]:
    """Rank actions and derive first-path FF/BF anchors in one scan."""
    if max_feasible_paths <= 0:
        raise ValueError("max_feasible_paths must be positive")
    if path_penalty < 0:
        raise ValueError("path_penalty must be non-negative")
    ranked: list[
        tuple[tuple[float, float, int, float, int], RMSAAction]
    ] = []
    feasible_paths = 0
    ksp_anchor = None
    best_fit_anchor = None
    paths = env._cached_k_paths(request.src_node, request.dst_node, env.k_paths)
    bitmap = env.spectrum.as_bitmap()
    for path_rank, path in enumerate(paths):
        modulation = highest_modulation_for_distance(
            env.topology.path_length_km(path)
        )
        if modulation is None:
            continue
        modulation_name, spectral_efficiency = modulation
        width = required_fs(request.bitrate_gbps, spectral_efficiency)
        available = env.spectrum.get_available_slots(path)
        path_actions = []
        pending = []
        path_ksp_anchor = None
        path_best_fit: tuple[tuple[int, int], RMSAAction] | None = None
        for run_left, run_right in _free_runs(available):
            if run_right - run_left < width:
                continue
            run_size = run_right - run_left
            waste = (run_size - width) / run_size
            for start_slot in range(run_left, run_right - width + 1):
                action = RMSAAction(
                    action_id=(
                        path_rank,
                        modulation_name,
                        start_slot,
                        width,
                    ),
                    path_rank=path_rank,
                    path=path,
                    modulation=modulation_name,
                    start_slot=start_slot,
                    required_fs=width,
                )
                pending.append((waste, start_slot, action))
                if start_slot == run_left:
                    if path_ksp_anchor is None:
                        path_ksp_anchor = action
                    best_fit_key = (run_size, run_left)
                    if (
                        path_best_fit is None
                        or best_fit_key < path_best_fit[0]
                    ):
                        path_best_fit = (best_fit_key, action)
        prices = pricer.marginal_losses(
            env,
            path,
            np.asarray([item[1] for item in pending]),
            width,
            bitmap=bitmap,
        )
        for price, (waste, start_slot, action) in zip(prices, pending):
                total = price + path_penalty * path_rank
                path_actions.append(
                    (
                        (total, price, path_rank, waste, start_slot),
                        action,
                    )
                )
        if path_actions:
            ranked.extend(path_actions)
            if ksp_anchor is None:
                ksp_anchor = path_ksp_anchor
                best_fit_anchor = (
                    path_best_fit[1]
                    if path_best_fit is not None
                    else None
                )
            feasible_paths += 1
            if feasible_paths >= max_feasible_paths:
                break
    ranked.sort(key=lambda item: item[0])
    return (
        [(float(key[0]), action) for key, action in ranked],
        ksp_anchor,
        best_fit_anchor,
    )


def build_diverse_shadow_pool(
    env: PDSRMSAEnv,
    request: Request,
    pricer: DemandShadowPricer,
    *,
    top_n: int = 10,
) -> list[tuple[float, RMSAAction]]:
    """Keep shadow Top-1 while suppressing path/start redundancy."""
    if top_n <= 0:
        raise ValueError("top_n must be positive")
    ranked, ksp_anchor, best_fit_anchor = (
        _rank_demand_shadow_actions_with_anchors(
            env,
            request,
            pricer,
            max_feasible_paths=3,
            path_penalty=1.0,
        )
    )
    if len(ranked) <= 1:
        return ranked
    score_by_id = {action.action_id: score for score, action in ranked}
    action_by_id = {action.action_id: action for _, action in ranked}
    selected: list[tuple[float, RMSAAction]] = []
    selected_ids = set()

    def add(action: RMSAAction | BlockedAction | None) -> None:
        if not isinstance(action, RMSAAction):
            return
        if action.action_id not in action_by_id:
            return
        if action.action_id in selected_ids or len(selected) >= top_n:
            return
        selected.append((score_by_id[action.action_id], action_by_id[action.action_id]))
        selected_ids.add(action.action_id)

    add(ranked[0][1])
    add(ksp_anchor)
    add(best_fit_anchor)

    seen_paths = set()
    for _, action in ranked:
        if action.path_rank not in seen_paths:
            add(action)
            seen_paths.add(action.path_rank)

    def overlaps(a: RMSAAction, b: RMSAAction) -> bool:
        if a.path_rank != b.path_rank:
            return False
        return not (
            a.start_slot + a.required_fs <= b.start_slot
            or b.start_slot + b.required_fs <= a.start_slot
        )

    for _, action in ranked:
        if len(selected) >= top_n:
            break
        if all(not overlaps(action, existing) for _, existing in selected):
            add(action)
    for _, action in ranked:
        if len(selected) >= top_n:
            break
        add(action)
    return selected
