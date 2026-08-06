"""Action-identical native hot path for KSP-FF K=50 hops."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from sa_hmarl.pds_rmsa.baselines.ksp_ff import ksp_ff_action
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

try:
    from sa_hmarl.pure_rmsa_v13.rollout_lab._direct_sketch_kernel import (
        ksp_first_fit as _compiled_ksp_first_fit,
    )
except ImportError:
    _compiled_ksp_first_fit = None


KSP_FF_KERNEL_BACKEND = (
    "c_extension" if _compiled_ksp_first_fit is not None else "python"
)
PYTHON_KSP_FF_BACKEND = "python_bitparallel_early_exit"


@dataclass(frozen=True)
class _ODRoutes:
    specs: tuple[tuple[int, tuple[int, ...], str], ...]
    path_arcs: np.ndarray
    widths_by_bitrate: dict[int, np.ndarray]
    blocked_reason: str


def _ksp_first_fit_python(
    arc_words: np.ndarray,
    path_arcs: np.ndarray,
    widths: np.ndarray,
    full_word: int,
) -> tuple[int, int]:
    """Reference for the compiled first-feasible-path kernel."""
    for path_index, (arc_indices, width) in enumerate(
        zip(path_arcs, widths)
    ):
        occupied = 0
        for arc_index in arc_indices:
            occupied |= int(arc_words[int(arc_index)])
        available = ~occupied & int(full_word)
        starts = available
        for shift in range(1, int(width)):
            starts &= available >> shift
        if starts:
            first = (starts & -starts).bit_length() - 1
            return path_index, first
    return -1, -1


class CompiledKSPFFSelector:
    """KSP-FF K=50 hops with native path scan and First-Fit."""

    def __init__(self, env: PDSRMSAEnv) -> None:
        # >64 slots use Python-int words (arbitrary precision); <=64 keeps
        # the uint64 path bitwise unchanged (C kernel still used there).
        self._wide = env.num_slots > 64
        self._arcs = tuple(env.spectrum.directed_link_ids())
        self._arc_index = {
            arc: index for index, arc in enumerate(self._arcs)
        }
        self._sentinel_arc = len(self._arcs)
        self._full_word = (1 << env.num_slots) - 1
        if self._wide:
            self._slot_bits = np.asarray(
                [1 << slot for slot in range(env.num_slots)],
                dtype=object,
            )
            self._arc_bitmap = env.spectrum.as_bitmap().copy()
            self._arc_words = np.zeros(len(self._arcs) + 1, dtype=object)
        else:
            self._slot_bits = np.asarray(
                [np.uint64(1 << slot) for slot in range(env.num_slots)],
                dtype=np.uint64,
            )
            self._arc_bitmap = env.spectrum.as_bitmap().copy()
            self._arc_words = np.zeros(len(self._arcs) + 1, dtype=np.uint64)
        self._arc_words[:-1] = self._bitmap_to_words(self._arc_bitmap)
        self._routes: dict[tuple[int, int], _ODRoutes] = {}
        self._precompute_routes(env)

    def _bitmap_to_words(self, bitmap: np.ndarray) -> np.ndarray:
        if self._wide:
            # object matmul: Python-int slot bits, arbitrary precision
            return np.asarray(
                bitmap.astype(object, copy=False) @ self._slot_bits,
                dtype=object,
            )
        return np.asarray(
            bitmap.astype(np.uint64, copy=False) @ self._slot_bits,
            dtype=np.uint64,
        )

    def _precompute_routes(self, env: PDSRMSAEnv) -> None:
        for src in range(env.topology.num_nodes):
            for dst in range(env.topology.num_nodes):
                if src == dst:
                    continue
                raw_paths = env._cached_k_paths(src, dst, env.k_paths)
                specs: list[tuple[int, tuple[int, ...], str]] = []
                arc_rows: list[tuple[int, ...]] = []
                efficiencies: list[float] = []
                for rank, path in enumerate(raw_paths):
                    modulation = highest_modulation_for_distance(
                        env.topology.path_length_km(path)
                    )
                    if modulation is None:
                        continue
                    modulation_name, efficiency = modulation
                    specs.append((rank, path, modulation_name))
                    efficiencies.append(efficiency)
                    arc_rows.append(
                        tuple(
                            self._arc_index[arc]
                            for arc in zip(path[:-1], path[1:])
                        )
                    )
                max_hops = max((len(row) for row in arc_rows), default=1)
                path_arcs = np.full(
                    (len(arc_rows), max_hops),
                    self._sentinel_arc,
                    dtype=np.intp,
                )
                for row_index, row in enumerate(arc_rows):
                    path_arcs[row_index, : len(row)] = row
                widths_by_bitrate = {
                    bitrate: np.asarray(
                        [
                            required_fs(bitrate, efficiency)
                            for efficiency in efficiencies
                        ],
                        dtype=np.intp,
                    )
                    for bitrate in BITRATE_CHOICES_GBPS
                }
                if not raw_paths:
                    blocked_reason = "no_candidate_path"
                elif not specs:
                    blocked_reason = "no_reach_feasible_path_mod"
                else:
                    blocked_reason = "insufficient_spectrum"
                self._routes[(src, dst)] = _ODRoutes(
                    specs=tuple(specs),
                    path_arcs=path_arcs,
                    widths_by_bitrate=widths_by_bitrate,
                    blocked_reason=blocked_reason,
                )

    def _sync_words(self, env: PDSRMSAEnv) -> None:
        bitmap = env.spectrum.as_bitmap()
        changed = np.flatnonzero(
            np.any(bitmap != self._arc_bitmap, axis=1)
        )
        if changed.size == 0:
            return
        self._arc_bitmap[changed] = bitmap[changed]
        self._arc_words[changed] = self._bitmap_to_words(bitmap[changed])

    def select(
        self,
        env: PDSRMSAEnv,
        request: Request,
    ) -> RMSAAction | BlockedAction:
        if self._wide:
            # >64 slots: Python-int path (C kernel is uint64-bound)
            self._sync_words(env)
            routes = self._routes[(request.src_node, request.dst_node)]
            widths = routes.widths_by_bitrate[int(request.bitrate_gbps)]
            path_index, start = _ksp_first_fit_python(
                self._arc_words,
                routes.path_arcs,
                widths,
                self._full_word,
            )
        elif _compiled_ksp_first_fit is None:
            return ksp_ff_action(env, request)
        else:
            self._sync_words(env)
            routes = self._routes[(request.src_node, request.dst_node)]
            widths = routes.widths_by_bitrate[int(request.bitrate_gbps)]
            path_index, start = _compiled_ksp_first_fit(
                self._arc_words,
                routes.path_arcs,
                widths,
                self._full_word,
            )
        if path_index < 0:
            return BlockedAction(reason=routes.blocked_reason)
        rank, path, modulation = routes.specs[path_index]
        width = int(widths[path_index])
        return RMSAAction(
            action_id=(rank, modulation, start, width),
            path_rank=rank,
            path=path,
            modulation=modulation,
            start_slot=start,
            required_fs=width,
        )


class PythonKSPFFSelector(CompiledKSPFFSelector):
    """Pure-Python bit-parallel KSP-FF with first-feasible-path exit."""

    def select(
        self,
        env: PDSRMSAEnv,
        request: Request,
    ) -> RMSAAction | BlockedAction:
        self._sync_words(env)
        routes = self._routes[(request.src_node, request.dst_node)]
        widths = routes.widths_by_bitrate[int(request.bitrate_gbps)]
        path_index, start = _ksp_first_fit_python(
            self._arc_words,
            routes.path_arcs,
            widths,
            self._full_word,
        )
        if path_index < 0:
            return BlockedAction(reason=routes.blocked_reason)
        rank, path, modulation = routes.specs[path_index]
        width = int(widths[path_index])
        return RMSAAction(
            action_id=(rank, modulation, start, width),
            path_rank=rank,
            path=path,
            modulation=modulation,
            start_slot=start,
            required_fs=width,
        )
