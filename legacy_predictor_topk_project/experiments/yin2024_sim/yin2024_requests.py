"""Yin 2024 request generator + dynamic DNN model adapter.

Maps Yin 2024 parameters (subtasks, compute in Hz, data in bits)
into the existing DNNRequest / DNNModel framework.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
from dataclasses import dataclass
from typing import List

from dnn_models import DNNModel, SplitProfile, MODEL_REGISTRY
from traffic_generator import DNNRequest


# ------------------------------------------------------------------
# Register a placeholder model so DNNRequest.__post_init__ works
# The actual model will be overridden per-request.
# ------------------------------------------------------------------

if "Yin2024Dynamic" not in MODEL_REGISTRY:
    MODEL_REGISTRY["Yin2024Dynamic"] = DNNModel(
        model_id=99,
        name="Yin2024Dynamic",
        input_size_mb=1.0,
        deadline_ms=30.0,
        splits=[
            SplitProfile(0, 1, 0.75, 1.0),
            SplitProfile(1, 3, 0.45, 3.0),
            SplitProfile(2, 6, 0.20, 6.0),
        ],
    )


# ------------------------------------------------------------------
# Request generator
# ------------------------------------------------------------------

@dataclass
class Yin2024RequestSpec:
    """Raw Yin 2024 request parameters."""
    req_id: int
    source_server_node: int
    deadline_ms: float
    num_subtasks: int
    compute_hz: List[float]       # per subtask
    data_bits: List[float]        # per subtask (output of layer i)


def build_dynamic_model(spec: Yin2024RequestSpec) -> DNNModel:
    """Build a DNNModel with 3 splits from a Yin2024 request spec.

    Split semantics (aligned with existing framework):
      - split_id=0: remote compute ratio high (~80%), bandwidth low
      - split_id=1: remote compute ratio medium (~50%), bandwidth medium
      - split_id=2: remote compute ratio low (~20%), bandwidth high
    """
    n = spec.num_subtasks
    total_compute = sum(spec.compute_hz)
    if total_compute <= 0:
        total_compute = 1e-9

    # partition points (number of local subtasks)
    local_tasks = [
        max(0, min(n - 1, int(np.floor(n * 0.2)))),
        max(0, min(n - 1, int(np.floor(n * 0.5)))),
        max(0, min(n - 1, int(np.floor(n * 0.8)))),
    ]

    splits = []
    for split_id, k in enumerate(local_tasks):
        remote_compute = sum(spec.compute_hz[k:])
        compute_cost = min(0.95, remote_compute / total_compute)

        if k < n and k > 0:
            data_mb = spec.data_bits[k - 1] / 8e6
        elif k == 0:
            # All remote: transfer original input (largest)
            data_mb = spec.data_bits[0] / 8e6 if spec.data_bits else 1.0
        else:
            data_mb = 0.5

        data_mb = max(0.1, data_mb)
        # Map data_mb to bandwidth slots heuristically (range 1-8)
        bw = max(1, min(8, int(data_mb / 10.0) + 1))

        splits.append(SplitProfile(split_id, bw, compute_cost, data_mb))

    return DNNModel(
        model_id=100 + n,
        name="Yin2024Dynamic",
        input_size_mb=1.0,
        deadline_ms=spec.deadline_ms,
        splits=splits,
    )


def make_dnn_request(spec: Yin2024RequestSpec,
                       arrival_time: float = 0.0,
                       holding_time: float = 0.0) -> DNNRequest:
    """Convert a Yin2024RequestSpec into a DNNRequest compatible with env_wrapper."""
    req = DNNRequest(
        req_id=spec.req_id,
        model_name="Yin2024Dynamic",
        source_node=spec.source_server_node,
        deadline_ms=spec.deadline_ms,
        arrival_time=arrival_time,
        holding_time=holding_time,
    )
    # Override the model with dynamically built splits
    req.model = build_dynamic_model(spec)
    return req


def generate_requests(num_requests: int,
                      server_nodes: List[int],
                      seed: int = 42,
                      arrival_rate: float = 5.0,
                      avg_holding_time: float = 5.0) -> List[DNNRequest]:
    """Generate a batch of Yin-2024-style requests with event-driven timing.

    Args:
        num_requests: number of requests to generate
        server_nodes: list of nodes that have MEC servers
        seed: random seed
        arrival_rate: average requests per time unit (Poisson process)
        avg_holding_time: average lightpath holding time (exponential)
    """
    rng = np.random.RandomState(seed)
    requests = []
    current_time = 0.0
    for i in range(num_requests):
        # Poisson inter-arrival
        inter_arrival = rng.exponential(1.0 / arrival_rate)
        current_time += inter_arrival

        # Exponential holding time
        holding_time = rng.exponential(avg_holding_time)

        n_sub = rng.randint(1, 6)  # 1–5
        deadline_ms = rng.uniform(20.0, 40.0)
        source = int(rng.choice(server_nodes))

        # Compute magnitude raised to 1e10–1.5e10 to match server capacity 2e10–5e10
        compute = [rng.uniform(1e10, 1.5e10) for _ in range(n_sub)]
        data = [rng.uniform(8e7, 60e7) for _ in range(n_sub)]
        # Sort decreasing with depth (shallow layers produce more data)
        data = sorted(data, reverse=True)

        spec = Yin2024RequestSpec(
            req_id=i + 1,
            source_server_node=source,
            deadline_ms=deadline_ms,
            num_subtasks=n_sub,
            compute_hz=compute,
            data_bits=data,
        )
        requests.append(make_dnn_request(spec, arrival_time=current_time, holding_time=holding_time))
    return requests
