"""Shared fixtures for PDS-RMSA tests."""
from __future__ import annotations

import pytest

from sa_hmarl.pds_rmsa.env.rmsa_env import PDSRMSAEnv
from sa_hmarl.pds_rmsa.env.topology import make_tiny_ring3
from sa_hmarl.pds_rmsa.env.traffic_trace import generate_trace, save_trace
from sa_hmarl.pds_rmsa.features.afterstate import build_probe_suite


@pytest.fixture
def tiny_env():
    env = PDSRMSAEnv(
        topology_name="tiny_ring3",
        num_slots=8,
        load_erlang=10.0,
        seed=0,
        trace=generate_trace("tiny_ring3", 3, 20, 10.0, 0, num_slots=8),
    )
    env.reset()
    return env


@pytest.fixture
def nsfnet_env():
    env = PDSRMSAEnv(
        topology_name="nsfnet",
        num_slots=50,
        load_erlang=100.0,
        seed=42,
        trace=generate_trace("nsfnet", 14, 100, 100.0, 42, num_slots=50),
    )
    env.reset()
    return env


@pytest.fixture
def tiny_trace(tmp_path):
    trace = generate_trace("tiny_ring3", 3, 50, 10.0, seed=7, num_slots=8)
    path = tmp_path / "tiny_trace.jsonl"
    save_trace(path, trace)
    return trace, str(path)


@pytest.fixture
def nsfnet_trace(tmp_path):
    trace = generate_trace("nsfnet", 14, 200, 100.0, seed=99, num_slots=50)
    path = tmp_path / "nsfnet_trace.jsonl"
    save_trace(path, trace)
    return trace, str(path)


@pytest.fixture
def tiny_probe_suite():
    return build_probe_suite(
        num_nodes=3,
        topology_name="tiny_ring3",
        num_od_pairs=3,
        bandwidths=(25, 50, 100),
    )
