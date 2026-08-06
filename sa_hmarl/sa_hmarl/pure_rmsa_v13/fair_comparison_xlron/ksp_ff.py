"""XLRON-compatible KSP-FF baseline (pure NumPy).

Matches XLRON heuristics.ksp_ff / first_fit:
- first_fit: for each path, the FIRST feasible start slot (sentinel
  column guarantees occupied paths get link_resources).
- ksp_ff: the first path (in KSP order) with a feasible start.
"""
from __future__ import annotations

from typing import Optional

import numpy as np

from .env import RMSAEnv
from .traffic import Request


def first_fit_starts(env: RMSAEnv, path_actions) -> np.ndarray:
    """(n_paths,) first feasible start per path; num_slots = none."""
    starts = np.full(len(path_actions), env.num_slots, dtype=np.int64)
    for i, pa in enumerate(path_actions):
        m = env.path_mask(pa["path"], pa["required_slots"])
        idx = np.flatnonzero(m)
        if idx.size:
            starts[i] = idx[0]
    return starts


def ksp_ff(env: RMSAEnv, req: Request) -> Optional[tuple]:
    """Returns (path, start_slot, required_slots) or None if blocked."""
    path_actions = env.path_actions(req)
    if not path_actions:
        return None
    starts = first_fit_starts(env, path_actions)
    feasible = np.flatnonzero(starts < env.num_slots)
    if feasible.size == 0:
        return None
    i = int(feasible[0])
    pa = path_actions[i]
    return (pa["path"], int(starts[i]), pa["required_slots"])


def run_episode(env: RMSAEnv, warmup: int, eval_requests: int) -> dict:
    """Advance warmup + eval requests; SBP over the eval window."""
    blocked = 0
    served = 0
    for _ in range(warmup):
        req = env.next_request()
        if req is None:
            break
        chosen = ksp_ff(env, req)
        if chosen is not None:
            env.allocate(chosen[0], chosen[1], chosen[2], req.holding_time)
    for _ in range(eval_requests):
        req = env.next_request()
        if req is None:
            break
        chosen = ksp_ff(env, req)
        if chosen is not None:
            env.allocate(chosen[0], chosen[1], chosen[2], req.holding_time)
            served += 1
        else:
            blocked += 1
    total = served + blocked
    return {"blocked": int(blocked), "served": int(served),
            "total": int(total),
            "blocked_rate": blocked / total * 100.0 if total else 0.0}
