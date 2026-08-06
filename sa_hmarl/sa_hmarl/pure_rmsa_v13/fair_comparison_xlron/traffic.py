"""XLRON-compatible traffic generation (pure NumPy).

Matches XLRON's env_funcs._arrival_holding_from_uniforms:
- arrival_time = -log1p(-u) / arrival_rate,  arrival_rate = load / mean_holding
- holding_time: rejection sampling with 5 candidates; candidates >= 2*mean
  are zeroed; the FIRST nonzero candidate is kept; if all five are zeroed
  the 5th candidate is kept (forced truncation).
- source-dest pairs uniform over all ordered pairs; bandwidth uniform over
  {25, 26, ..., 100} Gbps (XLRON min_bw/max_bw/step_bw defaults).
- Deterministic via np.random.RandomState(seed).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List

import numpy as np

MEAN_HOLDING_TIME = 25.0        # XLRON default mean_service_holding_time
TRUNCATION_FACTOR = 2.0
N_HOLDING_CANDIDATES = 5
MIN_BW = 25
MAX_BW = 100
STEP_BW = 1


@dataclass(frozen=True)
class Request:
    request_id: int
    src_node: int
    dst_node: int
    bitrate_gbps: float
    arrival_time: float
    holding_time: float


@dataclass
class Trace:
    topology: str
    num_slots: int
    load_erlang: float
    seed: int
    requests: List[Request] = field(default_factory=list)


def _holding_rejection_sampling(rng: np.random.RandomState) -> float:
    """XLRON rejection sampling: 5 exponentials, zero those >= 2*mean,
    keep the first nonzero, else force the 5th."""
    us = rng.random_sample(N_HOLDING_CANDIDATES)
    holding_times = -np.log1p(-us) * MEAN_HOLDING_TIME
    cap = TRUNCATION_FACTOR * MEAN_HOLDING_TIME
    zeroed = np.where(holding_times < cap, holding_times, 0.0)
    nonzero = np.flatnonzero(zeroed > 0.0)
    if nonzero.size:
        return float(zeroed[nonzero[0]])
    return float(holding_times[N_HOLDING_CANDIDATES - 1])


def _holding_hard_truncation(rng: np.random.RandomState) -> float:
    """Hard truncation (project convention): resample until <= 2*mean."""
    cap = TRUNCATION_FACTOR * MEAN_HOLDING_TIME
    while True:
        h = -np.log1p(-rng.random_sample()) * MEAN_HOLDING_TIME
        if h <= cap:
            return float(h)


def generate_trace(
    topology: str,
    num_nodes: int,
    num_requests: int,
    load_erlang: float,
    seed: int,
    num_slots: int = 100,
    truncate_holding_time: bool = True,
    truncation_mode: str = "rejection",
) -> Trace:
    """Generate an XLRON-style trace.  arrival_rate = load / mean_holding.

    truncation_mode: "rejection" (XLRON 5-candidate) or "hard" (project
    convention, resample until <= 2*mean); ignored when
    truncate_holding_time is False."""
    if load_erlang <= 0:
        raise ValueError(f"load_erlang must be > 0, got {load_erlang}")
    rng = np.random.RandomState(int(seed))
    arrival_rate = float(load_erlang) / MEAN_HOLDING_TIME
    node_pairs = [(s, d) for s in range(num_nodes)
                  for d in range(num_nodes) if s != d]
    bw_values = np.arange(MIN_BW, MAX_BW + 1, STEP_BW, dtype=np.float64)
    requests: list[Request] = []
    arrival_time = 0.0
    for req_id in range(int(num_requests)):
        u_arrival = rng.random_sample()
        arrival_time += -np.log1p(-u_arrival) / arrival_rate
        src, dst = node_pairs[rng.randint(len(node_pairs))]
        bitrate = float(bw_values[rng.randint(len(bw_values))])
        if truncate_holding_time:
            if truncation_mode == "hard":
                holding = _holding_hard_truncation(rng)
            else:
                holding = _holding_rejection_sampling(rng)
        else:
            holding = -np.log1p(-rng.random_sample()) * MEAN_HOLDING_TIME
        requests.append(Request(
            request_id=req_id, src_node=int(src), dst_node=int(dst),
            bitrate_gbps=bitrate, arrival_time=float(arrival_time),
            holding_time=float(holding)))
    return Trace(topology=str(topology), num_slots=int(num_slots),
                 load_erlang=float(load_erlang), seed=int(seed),
                 requests=requests)
