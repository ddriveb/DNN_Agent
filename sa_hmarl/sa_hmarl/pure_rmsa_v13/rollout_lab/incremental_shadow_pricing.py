"""Exact incremental price kernel for demand-shadow spectrum costs."""
from __future__ import annotations

import copy
import os

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import PDSRMSAEnv
from sa_hmarl.pds_rmsa.protocol import (
    BITRATE_CHOICES_GBPS,
    MODULATION_TABLE,
    required_fs,
)
from sa_hmarl.pure_rmsa_v13.rollout_lab.demand_shadow_pricing import (
    DemandShadowPricer,
    _free_runs,
    _window_count,
)

try:
    from sa_hmarl.pure_rmsa_v13.rollout_lab._direct_sketch_kernel import (
        build_shadow_price_row as _compiled_build_shadow_price_row,
    )
except ImportError:
    _compiled_build_shadow_price_row = None

_NATIVE_KERNEL_ENABLED = (
    os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL") == "1"
)
SHADOW_ROW_BACKEND = (
    "c_extension"
    if _NATIVE_KERNEL_ENABLED
    and _compiled_build_shadow_price_row is not None
    else "python"
)


def _build_price_row_python(
    arc_bitmap: np.ndarray,
    weights: np.ndarray,
    allocated_width: int,
) -> np.ndarray:
    """Reference implementation retained for exact kernel verification."""
    num_slots = len(arc_bitmap)
    starts = np.arange(num_slots, dtype=np.int16)
    run_left = np.full(num_slots, -1, dtype=np.int16)
    run_right = np.full(num_slots, -1, dtype=np.int16)
    runs = _free_runs(~arc_bitmap)
    for left, right in runs:
        last = right - allocated_width + 1
        if last > left:
            run_left[left:last] = left
            run_right[left:last] = right

    valid = run_left >= 0
    prices = np.zeros(num_slots, dtype=np.float64)
    for future_width in np.flatnonzero(weights):
        width = int(future_width)
        total = sum(
            _window_count(right - left, width)
            for left, right in runs
        )
        if total <= 0:
            continue
        run_total = np.maximum(
            0,
            run_right - run_left - width + 1,
        )
        left_surviving = np.maximum(
            0,
            starts - run_left - width + 1,
        )
        right_surviving = np.maximum(
            0,
            run_right
            - (starts + allocated_width)
            - width
            + 1,
        )
        destroyed = run_total - left_surviving - right_surviving
        destroyed[~valid] = 0
        prices += weights[future_width] * destroyed / total
    prices[~valid] = np.inf
    return prices


def _build_price_rows_python(
    arc_bitmap: np.ndarray,
    weights: np.ndarray,
    allocated_widths: np.ndarray,
) -> np.ndarray:
    """Build all requested widths while preserving per-row accumulation."""
    num_slots = len(arc_bitmap)
    widths = np.asarray(allocated_widths, dtype=np.int16)
    starts = np.arange(num_slots, dtype=np.int16)[None, :]
    run_left = np.full(
        (len(widths), num_slots),
        -1,
        dtype=np.int16,
    )
    run_right = np.full_like(run_left, -1)
    runs = _free_runs(~arc_bitmap)
    for row, allocated_width in enumerate(widths):
        for left, right in runs:
            last = right - int(allocated_width) + 1
            if last > left:
                run_left[row, left:last] = left
                run_right[row, left:last] = right

    valid = run_left >= 0
    prices = np.zeros(run_left.shape, dtype=np.float64)
    future_widths = np.flatnonzero(weights).astype(
        np.int16,
        copy=False,
    )
    if future_widths.size:
        future_axis = future_widths[:, None, None]
        run_left_3d = run_left[None, :, :]
        run_right_3d = run_right[None, :, :]
        run_total = np.maximum(
            0,
            run_right_3d - run_left_3d - future_axis + 1,
        )
        left_surviving = np.maximum(
            0,
            starts[None, :, :] - run_left_3d - future_axis + 1,
        )
        right_surviving = np.maximum(
            0,
            run_right_3d
            - (starts[None, :, :] + widths[None, :, None])
            - future_axis
            + 1,
        )
        destroyed = run_total - left_surviving - right_surviving
        destroyed[:, ~valid] = 0
        run_lengths = np.fromiter(
            (right - left for left, right in runs),
            dtype=np.int16,
            count=len(runs),
        )
        totals = np.maximum(
            0,
            run_lengths[:, None].astype(np.int64, copy=False)
            - future_widths[None, :]
            + 1,
        ).sum(axis=0)
        for index, (future_width, total) in enumerate(
            zip(future_widths, totals)
        ):
            if total > 0:
                prices += (
                    weights[int(future_width)]
                    * destroyed[index]
                    / total
                )
    prices[~valid] = np.inf
    return prices


def _build_price_geometry(
    arc_bitmap: np.ndarray,
    weights: np.ndarray,
) -> tuple[
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
]:
    """Build versioned integer run geometry shared by lazy width rows."""
    runs = tuple(_free_runs(~arc_bitmap))
    future_widths = np.flatnonzero(weights).astype(
        np.int16,
        copy=False,
    )
    run_lengths = np.fromiter(
        (right - left for left, right in runs),
        dtype=np.int16,
        count=len(runs),
    )
    totals = np.maximum(
        0,
        run_lengths[:, None].astype(np.int64, copy=False)
        - future_widths[None, :]
        + 1,
    ).sum(axis=0)
    return runs, future_widths, totals


def _free_runs_from_occupied_word(
    occupied_word: int,
    num_slots: int,
) -> tuple[tuple[int, int], ...]:
    """Return exact free runs from an incrementally maintained slot word."""
    full_word = (1 << num_slots) - 1
    free_word = ~int(occupied_word) & full_word
    runs: list[tuple[int, int]] = []
    while free_word:
        lowest = free_word & -free_word
        left = lowest.bit_length() - 1
        shifted = free_word >> left
        first_zero = (~shifted) & (shifted + 1)
        length = first_zero.bit_length() - 1
        right = min(num_slots, left + length)
        runs.append((left, right))
        free_word &= ~(((1 << length) - 1) << left)
    return tuple(runs)


def _build_price_geometry_from_word(
    occupied_word: int,
    num_slots: int,
    future_widths: np.ndarray,
) -> tuple[
    tuple[tuple[int, int], ...],
    np.ndarray,
    np.ndarray,
]:
    """Build shared run geometry without rescanning a NumPy bitmap."""
    runs = _free_runs_from_occupied_word(occupied_word, num_slots)
    run_lengths = np.fromiter(
        (right - left for left, right in runs),
        dtype=np.int16,
        count=len(runs),
    )
    totals = np.maximum(
        0,
        run_lengths[:, None].astype(np.int64, copy=False)
        - future_widths[None, :]
        + 1,
    ).sum(axis=0)
    return runs, future_widths, totals


def _build_price_row_from_geometry(
    num_slots: int,
    weights: np.ndarray,
    allocated_width: int,
    geometry: tuple[
        tuple[tuple[int, int], ...],
        np.ndarray,
        np.ndarray,
    ],
) -> np.ndarray:
    """Materialize one exact width row from shared integer geometry."""
    runs, future_widths, totals = geometry
    starts = np.arange(num_slots, dtype=np.int16)
    run_left = np.full(num_slots, -1, dtype=np.int16)
    run_right = np.full(num_slots, -1, dtype=np.int16)
    for left, right in runs:
        last = right - allocated_width + 1
        if last > left:
            run_left[left:last] = left
            run_right[left:last] = right

    valid = run_left >= 0
    prices = np.zeros(num_slots, dtype=np.float64)
    if future_widths.size:
        future_axis = future_widths[:, None]
        run_total = np.maximum(
            0,
            run_right[None, :] - run_left[None, :] - future_axis + 1,
        )
        left_surviving = np.maximum(
            0,
            starts[None, :] - run_left[None, :] - future_axis + 1,
        )
        right_surviving = np.maximum(
            0,
            run_right[None, :]
            - (starts[None, :] + allocated_width)
            - future_axis
            + 1,
        )
        destroyed = run_total - left_surviving - right_surviving
        destroyed[:, ~valid] = 0
        for index, (future_width, total) in enumerate(
            zip(future_widths, totals)
        ):
            if total > 0:
                prices += (
                    weights[int(future_width)]
                    * destroyed[index]
                    / total
                )
    prices[~valid] = np.inf
    return prices


class IncrementalDemandShadowPricer(DemandShadowPricer):
    """Cache exact per-link prices until the link bitmap changes."""

    def __init__(self, env: PDSRMSAEnv) -> None:
        super().__init__(env)
        self._num_slots = env.num_slots
        self._arc_bitmap = env.spectrum.as_bitmap().copy()
        self._arc_versions = np.zeros(
            len(self._arcs),
            dtype=np.int64,
        )
        # >64 slots use Python-int words (arbitrary precision); <=64
        # keeps the uint64 path bitwise unchanged.
        self._wide = env.num_slots > 64
        if self._wide:
            self._slot_bits = np.asarray(
                [1 << index for index in range(self._num_slots)],
                dtype=object,
            )
            self._arc_words = np.asarray(
                self._arc_bitmap.astype(object, copy=False) @ self._slot_bits,
                dtype=object,
            )
        else:
            self._slot_bits = np.asarray(
                [np.uint64(1 << index) for index in range(self._num_slots)],
                dtype=np.uint64,
            )
            self._arc_words = np.asarray(
                self._arc_bitmap.astype(np.uint64, copy=False) @ self._slot_bits,
                dtype=np.uint64,
            )
        self._arc_words_int = [
            int(value) for value in self._arc_words
        ]
        self._state_version = 0
        self._price_cache: dict[
            tuple[int, int], tuple[int, np.ndarray]
        ] = {}
        self._allocated_widths = np.asarray(
            sorted(
                {
                    required_fs(bitrate, spectral_efficiency)
                    for bitrate in BITRATE_CHOICES_GBPS
                    for _, _, spectral_efficiency in MODULATION_TABLE
                }
            ),
            dtype=np.int16,
        )
        self._allocated_width_index = {
            int(width): index
            for index, width in enumerate(self._allocated_widths)
        }
        self._future_widths_by_arc = tuple(
            np.flatnonzero(weights).astype(np.int16, copy=False)
            for weights in self._weights
        )
        self._price_matrix_cache: dict[
            int, tuple[int, np.ndarray]
        ] = {}
        self._geometry_cache: dict[
            int,
            tuple[
                int,
                tuple[
                    tuple[tuple[int, int], ...],
                    np.ndarray,
                    np.ndarray,
                ],
            ],
        ] = {}
        self._cache_checks = 0
        self._cache_hits = 0
        self._price_refreshes = 0
        self._arc_change_events = 0

    def fork_for_env(
        self,
        env: PDSRMSAEnv,
    ) -> "IncrementalDemandShadowPricer":
        """Fork dynamic price state while sharing immutable topology data."""
        if env.num_slots != self._num_slots:
            raise ValueError("cannot fork shadow pricer across slot counts")
        if tuple(env.spectrum.directed_link_ids()) != self._arcs:
            raise ValueError("cannot fork shadow pricer across topologies")

        fork = copy.copy(self)
        fork._arc_bitmap = self._arc_bitmap.copy()
        fork._arc_versions = self._arc_versions.copy()
        fork._arc_words = self._arc_words.copy()
        fork._arc_words_int = self._arc_words_int.copy()
        fork._price_cache = self._price_cache.copy()
        fork._price_matrix_cache = self._price_matrix_cache.copy()
        fork._geometry_cache = self._geometry_cache.copy()
        return fork

    @property
    def cache_stats(self) -> dict[str, int]:
        return {
            "cache_checks": self._cache_checks,
            "cache_hits": self._cache_hits,
            "price_refreshes": self._price_refreshes,
            "arc_change_events": self._arc_change_events,
            "cached_price_rows": (
                len(self._price_cache)
                + len(self._price_matrix_cache)
                * len(self._allocated_widths)
            ),
        }

    def _sync_bitmap(
        self,
        env: PDSRMSAEnv,
        bitmap: np.ndarray | None,
        changed: np.ndarray | None = None,
        occupied_words: np.ndarray | None = None,
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
        if occupied_words is None:
            self._arc_words[changed] = np.asarray(
                current[changed].astype(
                    object if self._wide else np.uint64, copy=False)
                @ self._slot_bits,
                dtype=object if self._wide else np.uint64,
            )
        else:
            words = np.asarray(occupied_words,
                              dtype=object if self._wide else np.uint64)
            if words.shape != self._arc_words.shape:
                raise ValueError("occupied word shape changed")
            self._arc_words[changed] = words[changed]
        for arc_index in changed:
            index = int(arc_index)
            self._arc_words_int[index] = int(self._arc_words[index])
        self._arc_change_events += int(changed.size)

    def sync_state(
        self,
        env: PDSRMSAEnv,
        bitmap: np.ndarray | None = None,
        changed: np.ndarray | None = None,
        occupied_words: np.ndarray | None = None,
    ) -> np.ndarray:
        """Synchronize once and return the exact bitmap used for scoring."""
        current = (
            env.spectrum.as_bitmap()
            if bitmap is None
            else np.asarray(bitmap, dtype=bool)
        )
        self._sync_bitmap(
            env,
            current,
            changed,
            occupied_words,
        )
        return current

    def _build_price_row(
        self,
        arc_index: int,
        allocated_width: int,
    ) -> np.ndarray:
        if (
            _NATIVE_KERNEL_ENABLED
            and _compiled_build_shadow_price_row is not None
        ):
            return _compiled_build_shadow_price_row(
                self._arc_bitmap[arc_index],
                self._weights[arc_index],
                allocated_width,
            )
        return _build_price_row_python(
            self._arc_bitmap[arc_index],
            self._weights[arc_index],
            allocated_width,
        )

    def _price_row(
        self,
        arc_index: int,
        allocated_width: int,
    ) -> np.ndarray:
        self._cache_checks += 1
        version = int(self._arc_versions[arc_index])
        if not _NATIVE_KERNEL_ENABLED:
            key = (arc_index, allocated_width)
            cached = self._price_cache.get(key)
            if cached is not None and cached[0] == version:
                self._cache_hits += 1
                return cached[1]
            cached_geometry = self._geometry_cache.get(arc_index)
            if (
                cached_geometry is None
                or cached_geometry[0] != version
            ):
                geometry = _build_price_geometry_from_word(
                    self._arc_words_int[arc_index],
                    self._num_slots,
                    self._future_widths_by_arc[arc_index],
                )
                self._geometry_cache[arc_index] = (version, geometry)
            else:
                geometry = cached_geometry[1]
            prices = _build_price_row_from_geometry(
                self._num_slots,
                self._weights[arc_index],
                allocated_width,
                geometry,
            )
            self._price_cache[key] = (version, prices)
            self._price_refreshes += 1
            return prices
        key = (arc_index, allocated_width)
        cached = self._price_cache.get(key)
        if cached is not None and cached[0] == version:
            self._cache_hits += 1
            return cached[1]
        prices = self._build_price_row(arc_index, allocated_width)
        self._price_cache[key] = (version, prices)
        self._price_refreshes += 1
        return prices

    def marginal_losses(
        self,
        env: PDSRMSAEnv,
        path: tuple[int, ...],
        start_slots: np.ndarray,
        allocated_width: int,
        *,
        bitmap: np.ndarray | None = None,
        assume_synced: bool = False,
    ) -> np.ndarray:
        """Return the same exact losses using incrementally cached rows."""
        starts = np.asarray(start_slots, dtype=np.int16)
        losses = np.zeros(starts.shape, dtype=np.float64)
        if starts.size == 0:
            return losses
        if np.any(starts < 0) or np.any(starts >= self._num_slots):
            raise ValueError("start slot outside spectrum")
        if not 0 < allocated_width <= self._num_slots:
            raise ValueError("allocated width outside spectrum")
        if not assume_synced:
            self._sync_bitmap(env, bitmap)
        for arc in zip(path[:-1], path[1:]):
            row = self._arc_index[arc]
            losses += self._price_row(row, allocated_width)[starts]
        return losses
