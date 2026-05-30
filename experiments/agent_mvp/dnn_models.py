"""DNN model registry: split points, bandwidth, compute profiles."""
from dataclasses import dataclass
from typing import List, Dict


@dataclass
class SplitProfile:
    """Profile for one split point of a DNN model."""
    split_id: int           # 0,1,2 mapped to partition in predictor
    bandwidth_slots: int    # FS slots needed for intermediate data transfer
    compute_cost: float     # Normalized compute cost at target server (0-1)
    intermediate_size_mb: float  # Intermediate data size in MB


@dataclass
class DNNModel:
    """A DNN inference request profile."""
    model_id: int
    name: str
    input_size_mb: float
    deadline_ms: float      # Target deadline in ms
    splits: List[SplitProfile]


# ------------------------------------------------------------------
# Registry — simplified but grounded in real model behavior.
# Split 0: tiny intermediate (edge-friendly, high local compute)
# Split 1: medium intermediate (balanced)
# Split 2: large intermediate (cloud-friendly, low local compute)
# ------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, DNNModel] = {
    "ResNet18": DNNModel(
        model_id=0,
        name="ResNet18",
        input_size_mb=0.6,      # 224x224 RGB image
        deadline_ms=100.0,
        splits=[
            SplitProfile(split_id=0, bandwidth_slots=1,  compute_cost=0.85, intermediate_size_mb=0.05),
            SplitProfile(split_id=1, bandwidth_slots=4,  compute_cost=0.50, intermediate_size_mb=0.30),
            SplitProfile(split_id=2, bandwidth_slots=8,  compute_cost=0.15, intermediate_size_mb=1.20),
        ],
    ),
    "MobileNetV2": DNNModel(
        model_id=1,
        name="MobileNetV2",
        input_size_mb=0.6,
        deadline_ms=80.0,
        splits=[
            SplitProfile(split_id=0, bandwidth_slots=1,  compute_cost=0.80, intermediate_size_mb=0.04),
            SplitProfile(split_id=1, bandwidth_slots=4,  compute_cost=0.45, intermediate_size_mb=0.25),
            SplitProfile(split_id=2, bandwidth_slots=8,  compute_cost=0.10, intermediate_size_mb=0.90),
        ],
    ),
    "BERT-tiny": DNNModel(
        model_id=2,
        name="BERT-tiny",
        input_size_mb=2.0,      # text embedding
        deadline_ms=150.0,
        splits=[
            SplitProfile(split_id=0, bandwidth_slots=1,  compute_cost=0.90, intermediate_size_mb=0.10),
            SplitProfile(split_id=1, bandwidth_slots=4,  compute_cost=0.55, intermediate_size_mb=0.80),
            SplitProfile(split_id=2, bandwidth_slots=8,  compute_cost=0.20, intermediate_size_mb=3.00),
        ],
    ),
}

MODEL_NAMES = list(MODEL_REGISTRY.keys())


def get_model(name: str) -> DNNModel:
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {MODEL_NAMES}")
    return MODEL_REGISTRY[name]


def get_split_bandwidth_map() -> Dict[int, int]:
    """Return {partition_id: bandwidth_slots} used by predictor MVP."""
    return {0: 1, 1: 4, 2: 8}
