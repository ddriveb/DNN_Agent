"""Core constants, dataclasses and helpers for the PDS-RMSA subproject."""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

SLOT_BW_HZ = 12.5e9
GUARD_BAND_FS = 1
MEAN_HOLDING_TIME = 10.0
BITRATE_CHOICES_GBPS = (25, 50, 75, 100)  # exact set, uniform
K_PATHS = 5
MAX_BLOCKS_PER_PATH = 8
PROBE_SEED = 2025
PROBE_OD_COUNT = 20
PROBE_BANDWIDTHS_GBPS = (25, 50, 75, 100)

MODULATION_TABLE = (
    ("BPSK", 10000.0, 1.0),
    ("QPSK", 2500.0, 2.0),
    ("8QAM", 1250.0, 3.0),
    ("16QAM", 625.0, 4.0),
)


@dataclass(frozen=True)
class Request:
    request_id: int
    src_node: int
    dst_node: int
    bitrate_gbps: int
    arrival_time: float
    holding_time: float


def required_fs(bitrate_gbps: int, spectral_efficiency: float) -> int:
    """Number of frequency slots required for a bitrate/modulation pair."""
    return math.ceil(bitrate_gbps * 1e9 / (SLOT_BW_HZ * spectral_efficiency)) + GUARD_BAND_FS


def highest_modulation_for_distance(
    path_length_km: float,
) -> Optional[Tuple[str, float]]:
    """Return (mod_name, spectral_efficiency) of the highest-SE reachable format."""
    chosen = None
    for name, reach, se in MODULATION_TABLE:
        if reach >= path_length_km and (chosen is None or se > chosen[1]):
            chosen = (name, se)
    return chosen
