"""N1-100 evaluation on the XLRON-compatible environment.

The N1-100 frozen checkpoint (210->16->100) was trained in the project
environment.  The XLRON environment shares the same NSFNET topology
(44 directed arcs, same edge SET, different arc ORDER), so the static
feature structure (conflict kernel, path meta) is rebuilt against the
XLRON arc order; features are permutation-equivalent to the project
order, and the frozen weights apply unchanged.

Candidate semantics mirror the N1-100 selector: K=50 hops-first pool,
first MAX_FEASIBLE_PATHS=3 paths with a feasible start, 8-tuple
candidates, mask + argmin(path_rank + price).
"""
from __future__ import annotations

from typing import List, Optional, Tuple

import numpy as np

from .env import RMSAEnv, highest_modulation, required_slots
from .traffic import Request

MAX_FEASIBLE_PATHS = 3
K_POOL = 50


class _SpectrumAdapter:
    def __init__(self, env: RMSAEnv):
        self._env = env

    def directed_link_ids(self):
        return tuple((u, v) for u, v, _ in self._env.edges)

    def occupied_bitmap_view(self) -> np.ndarray:
        return self._env.link_slot_array


class _TopologyAdapter:
    def __init__(self, env: RMSAEnv):
        self._env = env
        self.num_nodes = env.num_nodes

    def path_length_km(self, path) -> float:
        return self._env.path_km(list(path))


class _EnvAdapter:
    """Minimal env facade satisfying PathSlotFeatureExtractor."""

    def __init__(self, env: RMSAEnv):
        self.num_slots = env.num_slots
        self.k_paths = K_POOL
        self.topology = _TopologyAdapter(env)
        self.spectrum = _SpectrumAdapter(env)
        self._env = env

    def _cached_k_paths(self, src: int, dst: int, k: int):
        return [tuple(p) for p in self._env.ksp_paths(src, dst)]


def _to_word(bits: np.ndarray) -> int:
    w = 0
    for s in np.flatnonzero(bits):
        w |= 1 << int(s)
    return w


class N1XLRONEvaluator:
    """Frozen N1-100 policy on the XLRON environment."""

    def __init__(self, env: RMSAEnv, weights_path: str):
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.features import (
            PathSlotFeatureExtractor,
        )
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.n1_100.model import (
            TinyOpportunityWeights,
        )
        self.env = env
        self.arc_index = {(u, v): i for i, (u, v, _) in enumerate(env.edges)}
        self.adapter = _EnvAdapter(env)
        self.extractor = PathSlotFeatureExtractor(self.adapter,
                                                  self.arc_index)
        self.weights = TinyOpportunityWeights.load(weights_path)

    def _candidates(self, req: Request) -> List[tuple]:
        out = []
        for rank, path in enumerate(
                self.env.ksp_paths(req.src_node, req.dst_node)):
            if len(out) >= MAX_FEASIBLE_PATHS:
                break
            mod = highest_modulation(self.env.path_km(path))
            if mod is None:
                continue
            _, se = mod
            width = required_slots(req.bitrate_gbps, se)
            if width > self.env.num_slots:
                continue
            links = self.env.path_links(path)
            occupied = np.bitwise_or.reduce(
                self.env.link_slot_array[links], axis=0).astype(bool)
            free = ~occupied
            c = np.concatenate([[0], np.cumsum(free.astype(np.int64))])
            start = np.zeros(self.env.num_slots, dtype=bool)
            if width <= self.env.num_slots:
                start[:self.env.num_slots - width + 1] = \
                    (c[width:] - c[:-width]) == width
            if not start.any():
                continue
            arc_indices = [self.arc_index[(u, v)]
                           for u, v in zip(path[:-1], path[1:])]
            out.append((
                int(rank),
                tuple(path),
                mod[0],
                int(width),
                _to_word(start),
                _to_word(free),
                arc_indices,
                float(self.env.path_km(path)),
            ))
        return out

    def select(self, req: Request) -> Optional[Tuple[tuple, int, int]]:
        cands = self._candidates(req)
        if not cands:
            return None
        bitmap = self.env.link_slot_array.astype(np.float32)
        batch = self.extractor.build(req, bitmap, cands)
        prices = self.weights.forward(batch.features)          # (n, 100)
        scores = prices + np.arange(len(cands))[:, None]        # rank prior
        flat = np.where(batch.valid_starts.reshape(-1),
                        scores.reshape(-1), np.inf)
        if not np.isfinite(flat).any():
            return None
        best = int(np.argmin(flat))
        p, s = best // self.env.num_slots, best % self.env.num_slots
        cand = cands[p]
        return (cand[1], int(s), cand[3])


def run_n1_episode(env: RMSAEnv, weights_path: str,
                   warmup: int, eval_requests: int) -> dict:
    n1 = N1XLRONEvaluator(env, weights_path)
    blocked = 0
    served = 0
    for _ in range(warmup):
        req = env.next_request()
        if req is None:
            break
        chosen = n1.select(req)
        if chosen is not None:
            env.allocate(chosen[0], chosen[1], chosen[2], req.holding_time)
    for _ in range(eval_requests):
        req = env.next_request()
        if req is None:
            break
        chosen = n1.select(req)
        if chosen is not None:
            env.allocate(chosen[0], chosen[1], chosen[2], req.holding_time)
            served += 1
        else:
            blocked += 1
    total = served + blocked
    return {"blocked": int(blocked), "served": int(served),
            "total": int(total),
            "blocked_rate": blocked / total * 100.0 if total else 0.0}
