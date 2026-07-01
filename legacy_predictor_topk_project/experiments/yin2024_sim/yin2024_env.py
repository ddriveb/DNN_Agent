"""Environment factory for Yin 2024 simulation.

Assembles DNNOpticalEnv from Yin2024OpticalNetwork + Yin2024MECCluster
so that existing agents (baselines, TopK2, CorrectionNet) can run
with minimal or zero modification.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np

from env_wrapper import DNNOpticalEnv
from encoder import Encoder
from mapper import KSPMapper

from yin2024_network import (
    Yin2024OpticalNetwork,
    Yin2024MECCluster,
    NET1_SERVERS,
    NET2_SERVERS,
    NET3_SERVERS,
)


def create_yin2024_env(topology_name: str,
                       frag_level: float = 0.2,
                       seed: int = 42):
    """Create a fully compatible env for Yin 2024 experiments.

    Args:
        topology_name: "net1", "net2", or "net3"
        frag_level: 0.2 or 0.5 (spectrum fragmentation)
        seed: random seed

    Returns:
        env: DNNOpticalEnv instance
        encoder: Encoder instance
        mec: Yin2024MECCluster instance
    """
    if topology_name == "net1":
        num_slots = 100
        server_nodes = NET1_SERVERS
    elif topology_name == "net2":
        num_slots = 150
        server_nodes = NET2_SERVERS
    elif topology_name == "net3":
        num_slots = 150
        server_nodes = NET3_SERVERS
    else:
        raise ValueError(f"Unknown topology: {topology_name}")

    rng = np.random.RandomState(seed)

    # --- Network ---
    net = Yin2024OpticalNetwork(
        topology=topology_name,
        num_slots=num_slots,
        seed=seed,
        init_load_factor=0.6,
        frag_level=frag_level,
    )

    # --- Mapper & Encoder ---
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b

    # --- MEC servers (Yin 2024 parameters) ---
    loads_hz = [rng.uniform(3e8, 13e8) for _ in server_nodes]
    caps_hz = [rng.uniform(2e10, 5e10) for _ in server_nodes]

    # Apply uneven distribution (Net-3 most uneven)
    if topology_name == "net1":
        factor_range = (0.9, 1.1)
    elif topology_name == "net2":
        factor_range = (0.7, 1.3)
    else:
        factor_range = (0.5, 1.5)

    for i in range(len(loads_hz)):
        factor = rng.uniform(*factor_range)
        loads_hz[i] = min(caps_hz[i] * 0.95, loads_hz[i] * factor)

    mec = Yin2024MECCluster(server_nodes, loads_hz, caps_hz)

    # --- Env wrapper ---
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version="v1")
    return env, encoder, mec
