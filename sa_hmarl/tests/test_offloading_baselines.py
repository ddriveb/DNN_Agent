"""Unit tests for C-side offloading baselines.

Focus: verify that fixed-split variants (df_fixed0, rf_fixed0) always return
split0 actions, respect agent_c_mask, and never fall back to other splits.
"""
import numpy as np
import pytest

from sa_hmarl.evaluation.offloading_baselines import (
    select_df,
    select_df_fixed0,
    select_rf,
    select_rf_fixed0,
)


class _DummyNet:
    """Minimal network stand-in for distance estimation.

    Mirrors the interface used by _estimate_distance_km:
      - env.net.path_distance_km(path)
      - env.net.G.has_edge(u, v)
      - env.net.G.edges[(u, v)]  -> length_km
    """

    def __init__(self, edge_lengths):
        self.edge_lengths = edge_lengths  # dict {(u,v): km}
        self.G = _DummyGraph(edge_lengths)

    def path_distance_km(self, path):
        dist = 0.0
        for u, v in zip(path[:-1], path[1:]):
            key = (min(u, v), max(u, v))
            dist += self.edge_lengths.get(key, 1.0)
        return dist


class _DummyGraph:
    def __init__(self, edge_lengths):
        # networkx EdgeView semantics: G.edges[(u,v)] returns edge data dict.
        self.edges = {k: {"length_km": v} for k, v in edge_lengths.items()}

    def has_edge(self, u, v):
        return (min(u, v), max(u, v)) in self.edges


class _DummyServer:
    def __init__(self, server_id, node_id):
        self.server_id = server_id
        self.node_id = node_id


class _DummyMEC:
    def __init__(self, servers):
        self.servers = servers


class _DummyEnv:
    def __init__(self, edge_lengths, servers):
        self.net = _DummyNet(edge_lengths)
        self.mec = _DummyMEC(servers)


def _make_obs(num_splits=3, num_servers=4, src_node=0):
    """Build a minimal obs_c dict consistent with split-major action encoding."""
    n_actions = num_splits * num_servers
    features = []
    for split_id in range(num_splits):
        for server_id in range(num_servers):
            # Higher split id -> lower edge compute delay in this fixture.
            features.append(
                {
                    "edge_compute_ms": float(100 - split_id * 10 + server_id),
                    "best_fs_estimate": float(split_id + server_id + 1),
                }
            )
    return {
        "candidate_features": features,
        "server_utilizations": np.array([0.2, 0.4, 0.1, 0.3], dtype=float),
        "request_features": {"src_node": src_node, "num_splits": num_splits},
    }


def _make_env():
    # Linear topology: 0 --10-- 1 --10-- 2 --10-- 3
    edge_lengths = {(0, 1): 10.0, (1, 2): 10.0, (2, 3): 10.0}
    servers = [
        _DummyServer(0, 0),
        _DummyServer(1, 1),
        _DummyServer(2, 2),
        _DummyServer(3, 3),
    ]
    return _DummyEnv(edge_lengths, servers)


def test_df_selects_any_split():
    env = _make_env()
    obs = _make_obs(src_node=0)
    mask = np.ones(12, dtype=bool)
    action = select_df(env, obs, mask)
    split_id, server_id = divmod(action, 4)
    # Nearest server from node 0 is server 0 (node 0). DF may pick any split
    # for that server because all are valid.
    assert server_id == 0


def test_df_fixed0_only_split0():
    env = _make_env()
    obs = _make_obs(src_node=0)
    mask = np.ones(12, dtype=bool)
    action = select_df_fixed0(env, obs, mask)
    split_id, server_id = divmod(action, 4)
    assert split_id == 0
    assert server_id == 0  # nearest server


def test_rf_fixed0_only_split0():
    env = _make_env()
    obs = _make_obs(src_node=0)
    mask = np.ones(12, dtype=bool)
    action = select_rf_fixed0(obs, mask)
    split_id, server_id = divmod(action, 4)
    assert split_id == 0
    assert server_id == 2  # lightest utilization


def test_fixed0_respects_mask_no_fallback():
    """If split0 is fully masked, fixed0 must return None even if split1/2 are legal."""
    env = _make_env()
    obs = _make_obs(src_node=0)
    mask = np.ones(12, dtype=bool)
    mask[0:4] = False  # mask out all split0 actions

    assert select_df_fixed0(env, obs, mask) is None
    assert select_rf_fixed0(obs, mask) is None

    # Sanity: adapted df/rf can still find an action in split1/2.
    assert select_df(env, obs, mask) is not None
    assert select_rf(obs, mask) is not None


def test_fixed0_ignores_other_splits_when_split0_partially_masked():
    """If some split0 servers are masked, fixed0 must not use a split1/2 action."""
    env = _make_env()
    obs = _make_obs(src_node=0)
    mask = np.ones(12, dtype=bool)
    mask[0] = False  # mask split0/server0 (the nearest server for DF)

    df_action = select_df_fixed0(env, obs, mask)
    assert df_action is not None
    split_id, server_id = divmod(df_action, 4)
    assert split_id == 0
    assert server_id == 1  # next nearest server from node 0

    # RF: mask split0/server2 (lightest).  Next lightest is server0.
    mask = np.ones(12, dtype=bool)
    mask[2] = False
    rf_action = select_rf_fixed0(obs, mask)
    assert rf_action is not None
    split_id, server_id = divmod(rf_action, 4)
    assert split_id == 0
    assert server_id == 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
