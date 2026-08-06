"""Locked protocol for Neural Opportunity v1."""
from __future__ import annotations

PROTOCOL_ID = "pure_rmsa_neural_opportunity_v1"
FEATURE_SCHEMA_ID = "neural_opportunity_conflict_pressure_map_v1"

K_PATHS = 50
MAX_FEASIBLE_PATHS = 3
NUM_SLOTS = 50
MAX_CANDIDATES = MAX_FEASIBLE_PATHS * NUM_SLOTS
HIDDEN_DIM = 16
ROUTE_DEPTH = 3

TOPOLOGY_CONFIG = {
    "xlron_nsfnet_deeprmsa": {
        "load_erlang": 100,
        "budget": 256,
        "opportunity_weight": 3000.0,
    },
    "xlron_usnet_gcnrmsa": {
        "load_erlang": 220,
        "budget": 256,
        "opportunity_weight": 8000.0,
    },
    "xlron_jpn48": {
        "load_erlang": 150,
        "budget": 64,
        "opportunity_weight": 5000.0,
    },
}

# These allocations are new and disjoint from existing pure_rmsa_v13 studies.
TRAIN_SEEDS = (60001, 60002, 60003)
VALIDATION_SEEDS = (60101,)
SMOKE_SEED = 60201
PILOT_SEEDS = (60301, 60302, 60303, 60304, 60305)
DAGGER_SEEDS = (60401, 60402)
CONFIRMATORY_SEEDS = tuple(range(23001, 23011))

WARMUP_REQUESTS = 1000
EVAL_REQUESTS = 10000

LATENCY_MEAN_MS_GATE = 0.060
OFFLINE_TOP1_RECALL_GATE = 0.80
OFFLINE_NORMALIZED_REGRET_GATE = 0.10
PILOT_MAX_TEACHER_DELTA_PP = 0.20
PILOT_MIN_KSP_GAIN_RETENTION = 0.85
MAX_DAGGER_ROUNDS = 2

STATIC_FEATURE_NAMES = (
    "path_rank",
    "feasible_ordinal",
    "path_hops",
    "path_km",
    "required_width",
    "request_bitrate",
    "common_free_fraction",
    "route_centrality_sum",
    "route_centrality_max",
    "candidate_overlap_fraction",
)
STATIC_FEATURE_DIM = len(STATIC_FEATURE_NAMES)
FEATURE_DIM = STATIC_FEATURE_DIM + 2 * NUM_SLOTS
OUTPUT_DIM = NUM_SLOTS
