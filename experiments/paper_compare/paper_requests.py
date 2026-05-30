"""Paper-style dynamic request generation compatible with the current env."""
from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent / "agent_mvp"))
sys.path.insert(0, str(Path(__file__).parent.parent / "yin2024_sim"))

from traffic_generator import DNNRequest
from yin2024_requests import Yin2024RequestSpec, build_dynamic_model


def generate_request_stream(
    num_requests: int,
    server_nodes: list[int],
    *,
    seed: int = 42,
    arrival_rate: float = 5.0,
    avg_holding_time: float = 10.0,
):
    """Generate event-driven requests under the paper-style subtask ranges."""
    rng = np.random.RandomState(seed)
    requests = []
    current_time = 0.0

    for i in range(num_requests):
        inter_arrival = rng.exponential(1.0 / max(arrival_rate, 1e-6))
        current_time += inter_arrival
        holding_time = float(rng.exponential(avg_holding_time))

        num_subtasks = int(rng.randint(1, 6))
        deadline_ms = float(rng.uniform(20.0, 40.0))
        source_node = int(rng.choice(server_nodes))

        compute_hz = []
        data_bits = []
        for j in range(num_subtasks):
            depth = j / max(num_subtasks - 1, 1)
            compute = rng.uniform(10e9, 15e9) * (0.8 + 0.4 * depth)
            data = rng.uniform(8e6, 60e6) * (0.3 + 0.7 * (1.0 - depth))
            compute_hz.append(float(compute))
            data_bits.append(float(data * 8.0))

        spec = Yin2024RequestSpec(
            req_id=i + 1,
            source_server_node=source_node,
            deadline_ms=deadline_ms,
            num_subtasks=num_subtasks,
            compute_hz=compute_hz,
            data_bits=data_bits,
        )
        req = DNNRequest(
            req_id=spec.req_id,
            model_name="Yin2024Dynamic",
            source_node=spec.source_server_node,
            deadline_ms=spec.deadline_ms,
            arrival_time=current_time,
            holding_time=holding_time,
        )
        req.model = build_dynamic_model(spec)
        requests.append(req)

    return requests
