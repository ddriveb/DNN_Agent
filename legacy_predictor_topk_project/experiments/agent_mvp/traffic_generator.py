"""DNN inference request generator."""
import numpy as np
from dataclasses import dataclass
from typing import List, Optional
from dnn_models import MODEL_NAMES, get_model


@dataclass
class DNNRequest:
    """A single DNN inference request arriving at the network edge."""
    req_id: int
    model_name: str
    source_node: int        # Where the user/device is attached
    deadline_ms: float      # SLA deadline
    arrival_time: float     # Simulation time of arrival
    holding_time: float     # How long the lightpath is held
    # Derived at decision time
    model = None

    def __post_init__(self):
        self.model = get_model(self.model_name)


class TrafficGenerator:
    """Generate event-driven DNN inference requests."""

    def __init__(self, num_nodes: int,
                 arrival_rate: float = 5.0,      # requests per time unit
                 avg_holding_time: float = 5.0,  # average lightpath duration
                 seed: Optional[int] = None):
        self.num_nodes = num_nodes
        self.arrival_rate = arrival_rate
        self.avg_holding_time = avg_holding_time
        self.rng = np.random.RandomState(seed)
        self.req_counter = 0

    def reset(self):
        self.req_counter = 0

    def next_request(self, current_time: float) -> DNNRequest:
        """Generate the next request and its inter-arrival time."""
        # Exponential inter-arrival
        inter_arrival = self.rng.exponential(1.0 / self.arrival_rate)
        arrival_time = current_time + inter_arrival

        # Random source node
        source_node = int(self.rng.randint(0, self.num_nodes))

        # Random model type
        model_name = self.rng.choice(MODEL_NAMES)
        model = get_model(model_name)

        # Deadline with some jitter around the model's base deadline
        jitter = self.rng.uniform(0.8, 1.2)
        deadline_ms = model.deadline_ms * jitter

        # Holding time for the lightpath
        holding_time = self.rng.exponential(self.avg_holding_time)

        self.req_counter += 1
        return DNNRequest(
            req_id=self.req_counter,
            model_name=model_name,
            source_node=source_node,
            deadline_ms=deadline_ms,
            arrival_time=arrival_time,
            holding_time=holding_time,
        )

    def generate_batch(self, num_requests: int, start_time: float = 0.0) -> List[DNNRequest]:
        """Generate a batch of requests (for offline dataset creation)."""
        requests = []
        t = start_time
        for _ in range(num_requests):
            req = self.next_request(t)
            requests.append(req)
            t = req.arrival_time
        return requests
