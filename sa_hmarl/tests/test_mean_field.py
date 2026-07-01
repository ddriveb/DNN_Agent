"""Unit tests for mean-field feature construction."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.mean_field import (
    classify_request_type,
    compute_global_mean_field,
    compute_typed_mean_field,
    global_mean_field_to_vector,
    typed_mean_field_to_vector,
    TYPED_MEAN_FIELD_TYPES,
)


def _make_env(num_servers=3):
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(
        num_nodes=net.NUM_NODES,
        num_servers=num_servers,
        seed=42,
        server_nodes=list(range(num_servers)),
    )
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    return SMDPEnv(net, mec, mod_reg, fs_calc, k=3, max_blocks=5)


def test_classify_request_type():
    """Request classification based on size/deadline score."""
    # light_urgent: small data, tight deadline but score still low
    req_light = DNNRequest(
        req_id=0, src_node=0, arrival_time=0.0, holding_time=5.0, deadline_ms=100.0,
        splits=[SplitProfile(0, 1.0, 0.5, 0.5)],
    )
    assert classify_request_type(req_light) == "light_urgent"

    # heavy_relax: large data, relaxed deadline but score high
    req_heavy = DNNRequest(
        req_id=1, src_node=0, arrival_time=0.0, holding_time=5.0, deadline_ms=50.0,
        splits=[SplitProfile(0, 10.0, 0.5, 0.5)],
    )
    assert classify_request_type(req_heavy) == "heavy_relax"

    print("test_classify_request_type PASSED")


def test_global_mean_field_empty():
    """Empty active connections yield a zero mean-field vector."""
    env = _make_env(num_servers=3)
    env.reset([])
    mf = compute_global_mean_field(env)
    assert mf["count"] == 0
    assert np.allclose(mf["srv"], np.zeros(3))
    assert mf["dem"] == 0.0
    assert mf["rel"] == 0.0

    vec = global_mean_field_to_vector(mf, 3, num_slots=32)
    assert vec.shape == (5,)
    assert np.allclose(vec, np.zeros(5))
    print("test_global_mean_field_empty PASSED")


def test_global_mean_field_nonempty():
    """Global mean field reflects active connection distribution."""
    env = _make_env(num_servers=3)
    env.time = 1.0
    env.active_connections = [
        {"server_id": 0, "num_slots": 4, "release_time": 5.0,
         "intermediate_size_mb": 1.0, "deadline_ms": 100.0},
        {"server_id": 0, "num_slots": 6, "release_time": 7.0,
         "intermediate_size_mb": 2.0, "deadline_ms": 100.0},
        {"server_id": 2, "num_slots": 8, "release_time": 6.0,
         "intermediate_size_mb": 5.0, "deadline_ms": 100.0},
    ]

    mf = compute_global_mean_field(env)
    assert mf["count"] == 3
    assert np.allclose(mf["srv"], np.array([2 / 3, 0.0, 1 / 3]))
    assert abs(mf["dem"] - (4 + 6 + 8) / 3) < 1e-6
    assert abs(mf["rel"] - ((5 - 1) + (7 - 1) + (6 - 1)) / 3) < 1e-6

    vec = global_mean_field_to_vector(mf, 3, num_slots=32)
    assert vec.shape == (5,)
    assert abs(vec[3] - mf["dem"] / 32) < 1e-6
    assert abs(vec[4] - mf["rel"] / 10.0) < 1e-6
    print("test_global_mean_field_nonempty PASSED")


def test_typed_mean_field():
    """Typed mean field aggregates per-type statistics."""
    env = _make_env(num_servers=3)
    env.time = 0.0
    # light_urgent on server 0
    env.active_connections = [
        {"server_id": 0, "num_slots": 4, "release_time": 5.0,
         "intermediate_size_mb": 1.0, "deadline_ms": 100.0},
        {"server_id": 0, "num_slots": 6, "release_time": 7.0,
         "intermediate_size_mb": 1.2, "deadline_ms": 100.0},
        # heavy_relax on server 2
        {"server_id": 2, "num_slots": 10, "release_time": 8.0,
         "intermediate_size_mb": 10.0, "deadline_ms": 50.0},
    ]

    mf = compute_typed_mean_field(env)
    for t in TYPED_MEAN_FIELD_TYPES:
        assert t in mf

    light = mf["light_urgent"]
    assert light["count"] == 2
    assert np.allclose(light["srv"], np.array([1.0, 0.0, 0.0]))
    assert abs(light["dem"] - 5.0) < 1e-6
    assert abs(light["rel"] - 6.0) < 1e-6

    heavy = mf["heavy_relax"]
    assert heavy["count"] == 1
    assert np.allclose(heavy["srv"], np.array([0.0, 0.0, 1.0]))
    assert heavy["dem"] == 10.0
    assert heavy["rel"] == 8.0

    vec = typed_mean_field_to_vector(mf, 3, num_slots=32)
    assert vec.shape == (3 * (3 + 2),)
    print("test_typed_mean_field PASSED")


def test_event_env_records_mean_field_metadata():
    """SMDPEnv records the extra metadata needed for mean-field typing."""
    env = _make_env(num_servers=2)
    requests = [
        DNNRequest(
            req_id=0, src_node=1, arrival_time=0.0, holding_time=5.0, deadline_ms=100.0,
            splits=[SplitProfile(0, 2.0, 0.3, 0.7)],
        ),
    ]
    env.reset(requests)
    obs, reward, done, info = env.step((0, 0), (0, 0, 0))
    assert info["success"]
    assert len(env.active_connections) == 1
    conn = env.active_connections[0]
    assert "intermediate_size_mb" in conn
    assert "deadline_ms" in conn
    assert "holding_time" in conn
    assert conn["intermediate_size_mb"] == 2.0
    assert conn["deadline_ms"] == 100.0
    print("test_event_env_records_mean_field_metadata PASSED")


def test_agent_c_typed_mean_field_feature_dim():
    """Agent-C typed_mean_field feature_mode produces the expected feature size."""
    from sa_hmarl.agents.c_agent import AgentC
    agent = AgentC(feature_mode="typed_mean_field", num_servers=3)
    assert agent.input_dim == 17 + 3 * (3 + 2)

    obs = {
        "agent_c_mask": np.array([True, False]),
        "candidate_features": [
            {
                "intermediate_size_mb": 1.0,
                "local_compute_ms": 1.0,
                "edge_compute_ms": 2.0,
                "server_utilization": 0.5,
                "best_fs_estimate": 3,
                "safe_fs_estimate": 4,
                "feasible_count": 5,
                "spectrum_summary": np.zeros(10, dtype=np.float32),
            },
            {
                "intermediate_size_mb": 2.0,
                "local_compute_ms": 2.0,
                "edge_compute_ms": 3.0,
                "server_utilization": 0.6,
                "best_fs_estimate": 4,
                "safe_fs_estimate": 5,
                "feasible_count": 6,
                "spectrum_summary": np.zeros(10, dtype=np.float32),
            },
        ],
        "typed_mean_field": np.arange(3 * (3 + 2), dtype=np.float32),
    }
    features, mask = agent.build_action_features(obs)
    assert features.shape == (2, agent.input_dim)
    assert mask.shape == (2,)
    print("test_agent_c_typed_mean_field_feature_dim PASSED")


def test_ppo_agent_c_typed_mean_field_dim():
    """PPOAgent-C typed_mean_field feature_mode computes the correct input dim."""
    from sa_hmarl.agents.ppo_agents import PPOAgentC
    agent = PPOAgentC(feature_mode="typed_mean_field", num_servers=4)
    assert agent.input_dim == 17 + 3 * (4 + 2)
    print("test_ppo_agent_c_typed_mean_field_dim PASSED")


def test_gated_typed_mean_field_feature_dim():
    """Agent-C gated_typed_mean_field feature_mode produces the expected feature size."""
    from sa_hmarl.agents.c_agent import AgentC
    agent = AgentC(feature_mode="gated_typed_mean_field", num_servers=3)
    assert agent.input_dim == 17 + 3 * (3 + 2)

    obs = {
        "agent_c_mask": np.array([True, False]),
        "candidate_features": [
            {
                "intermediate_size_mb": 1.0,
                "local_compute_ms": 1.0,
                "edge_compute_ms": 2.0,
                "server_utilization": 0.5,
                "best_fs_estimate": 3,
                "safe_fs_estimate": 4,
                "feasible_count": 5,
                "spectrum_summary": np.zeros(10, dtype=np.float32),
            },
            {
                "intermediate_size_mb": 2.0,
                "local_compute_ms": 2.0,
                "edge_compute_ms": 3.0,
                "server_utilization": 0.6,
                "best_fs_estimate": 4,
                "safe_fs_estimate": 5,
                "feasible_count": 6,
                "spectrum_summary": np.zeros(10, dtype=np.float32),
            },
        ],
        "typed_mean_field": np.arange(3 * (3 + 2), dtype=np.float32),
    }
    features, mask = agent.build_action_features(obs)
    assert features.shape == (2, agent.input_dim)
    assert mask.shape == (2,)
    print("test_gated_typed_mean_field_feature_dim PASSED")


def test_ppo_agent_c_gated_typed_mean_field():
    """PPOAgent-C gated_typed_mean_field uses a GatedMaskedPPOActorNetwork."""
    from sa_hmarl.agents.ppo_agents import PPOAgentC, GatedMaskedPPOActorNetwork
    agent = PPOAgentC(feature_mode="gated_typed_mean_field", num_servers=4)
    assert agent.input_dim == 17 + 3 * (4 + 2)
    assert isinstance(agent.policy_net, GatedMaskedPPOActorNetwork)
    assert agent.policy_net.base_dim == 17
    assert agent.policy_net.mf_dim == 3 * (4 + 2)
    print("test_ppo_agent_c_gated_typed_mean_field PASSED")


def test_gated_actor_forward_shape():
    """GatedMaskedPPOActorNetwork forward returns the correct logits shape."""
    import torch
    from sa_hmarl.agents.ppo_agents import GatedMaskedPPOActorNetwork
    net = GatedMaskedPPOActorNetwork(base_dim=17, mf_dim=18, hidden_dims=(128, 64))
    x = torch.randn(4, 7, 35)  # batch=4, actions=7, total_dim=35
    logits = net(x)
    assert logits.shape == (4, 7)
    print("test_gated_actor_forward_shape PASSED")


if __name__ == "__main__":
    test_classify_request_type()
    test_global_mean_field_empty()
    test_global_mean_field_nonempty()
    test_typed_mean_field()
    test_event_env_records_mean_field_metadata()
    test_agent_c_typed_mean_field_feature_dim()
    test_ppo_agent_c_typed_mean_field_dim()
    test_gated_typed_mean_field_feature_dim()
    test_ppo_agent_c_gated_typed_mean_field()
    test_gated_actor_forward_shape()
    print("All mean-field tests PASSED")
