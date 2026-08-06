"""env.py tests: topology, modulation, mask, allocation, edge cases."""
from __future__ import annotations

import numpy as np
import pytest

from sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.env import (
    RMSAEnv,
    highest_modulation,
    required_slots,
)
from sa_hmarl.pure_rmsa_v13.fair_comparison_xlron.traffic import generate_trace


def _env(k=5, seed=69301, n_req=2000):
    t = generate_trace("xlron_nsfnet_deeprmsa", 14, n_req, 250, seed,
                       num_slots=100)
    return RMSAEnv(t, k_paths=k, topology_name="xlron_nsfnet_deeprmsa")


def test_topology_size():
    env = _env()
    assert env.num_nodes == 14
    assert env.num_links == 44
    assert env.num_slots == 100


def test_modulation_table():
    assert highest_modulation(500.0) == ("16QAM", 4.0)
    assert highest_modulation(700.0) == ("8QAM", 3.0)
    assert highest_modulation(1300.0) == ("QPSK", 2.0)
    assert highest_modulation(2600.0) == ("BPSK", 1.0)
    assert highest_modulation(11000.0) is None


def test_required_slots_formula():
    # ceil(50 / (2 * 12.5)) + 1 = ceil(2) + 1 = 3
    assert required_slots(50.0, 2.0) == 3
    # ceil(100 / (4 * 12.5)) + 1 = 2 + 1 = 3
    assert required_slots(100.0, 4.0) == 3
    # ceil(25 / (1 * 12.5)) + 1 = 2 + 1 = 3
    assert required_slots(25.0, 1.0) == 3


def test_ksp_hops_first():
    env = _env()
    # path 0-1 (direct edge) has 1 hop; verify ordering by hops
    paths = env.ksp_paths(0, 1)
    hops = [len(p) - 1 for p in paths]
    assert hops == sorted(hops)
    assert paths[0] == [0, 1]  # direct link first


def test_path_mask_empty_state():
    env = _env()
    paths = env.ksp_paths(0, 1)
    m = env.path_mask(paths[0], 3)
    assert m.sum() == env.num_slots - 3 + 1  # all windows free


def test_allocate_and_release():
    env = _env()
    env.next_request()  # advance time to first request
    path = env.ksp_paths(0, 1)[0]
    assert env.allocate(path, 0, 3, holding_time=10.0)
    assert env.link_slot_array[env.path_links(path)[0], 0:3].all()
    # window overlapping the allocation is not free
    m = env.path_mask(path, 3)
    assert not m[0] and not m[1] and not m[2]
    # time advance releases
    env.advance_time(11.0)
    assert not env.link_slot_array[env.path_links(path)[0], 0:3].any()


def test_allocate_overlap_fails():
    env = _env()
    env.next_request()
    path = env.ksp_paths(0, 1)[0]
    assert env.allocate(path, 5, 3, holding_time=10.0)
    assert not env.allocate(path, 6, 3, holding_time=10.0)  # overlaps


def test_no_path_available_edge():
    """Fill all slots on the only path -> blocked."""
    env = _env()
    env.next_request()
    path = env.ksp_paths(0, 1)[0]
    links = env.path_links(path)
    env.link_slot_array[links] = 1  # saturate the direct path
    m = env.path_mask(path, 3)
    assert not m.any()
