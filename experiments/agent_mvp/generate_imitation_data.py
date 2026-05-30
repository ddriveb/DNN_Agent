"""Generate imitation learning dataset from AdaptiveRuleAgent.

Records (state_vector, best_action_id) pairs for behavior cloning.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import pickle
from typing import List, Tuple, Dict
from dataclasses import dataclass

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from predictor import Predictor
from dnn_models import get_split_bandwidth_map
from mec_servers import MECCluster
from env_wrapper import DNNOpticalEnv
from adaptive_rule_agent import AdaptiveRuleAgent
from fixed_trace import load_trace, generate_and_save_trace


@dataclass
class ImitationSample:
    """One (state, action) pair for imitation learning."""
    state: np.ndarray          # Concatenated state vector
    action_id: int             # split_id * num_servers + server_id
    model_name: str
    source_node: int
    deadline_ms: float
    chosen_strategy: str
    load_level: str
    split_id: int
    server_id: int


def load_predictor(path: str, max_servers: int = 128):
    predictor = Predictor(10, num_partitions=3, max_servers=max_servers, hidden_dim=64)
    state = torch.load(path, map_location="cpu", weights_only=True)
    predictor.load_state_dict(state)
    predictor.eval()
    return predictor


def create_env(topology, num_slots, num_servers, seed):
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=num_servers, seed=seed)
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version="v1")
    return env, encoder, mec


def build_state_vector(request, env: DNNOpticalEnv, chosen_server_id: int,
                       encoder, num_servers: int) -> np.ndarray:
    """Build a rich state vector for imitation learning.

    Components:
      - encoder z for (src, chosen_dst): 10-dim
      - request features: model_id, deadline_norm, source_node
      - global network state: frag_index, max_free_block
      - server load vector: utilizations
      - candidate count: how many (split, server) combos exist
    """
    model = request.model
    src = request.source_node
    srv = env.mec.get_server_by_id(chosen_server_id)
    dst = srv.node_id

    # 1. Network encoding for this src-dst pair
    z = encoder.encode(src, dst)

    # 2. Request features
    model_id = float(model.model_id)
    deadline_norm = request.deadline_ms / 200.0  # normalize to ~0-1
    src_norm = src / env.net.NUM_NODES

    # 3. Global network state
    frag = env._global_frag_index()
    # Compute max consecutive free block across all links
    max_free_vals = []
    for link, slots in env.net.link_states.items():
        free = ~slots
        if not np.any(free):
            continue
        max_len = curr = 0
        for v in free:
            if v:
                curr += 1
                max_len = max(max_len, curr)
            else:
                curr = 0
        max_free_vals.append(max_len)
    max_free = (max(max_free_vals) / env.net.num_slots) if max_free_vals else 0.0

    # 4. Server load vector
    server_utils = env.mec.server_load_vector(normalize=True)
    # Pad if fewer servers than expected
    target_dim = num_servers
    if len(server_utils) < target_dim:
        server_utils = np.pad(server_utils, (0, target_dim - len(server_utils)))
    elif len(server_utils) > target_dim:
        server_utils = server_utils[:target_dim]

    # 5. Candidate count (total possible split × server combos)
    num_splits = len(model.splits)
    num_srv = len(env.mec.servers)
    candidate_count = num_splits * num_srv

    state = np.concatenate([
        z,
        [model_id, deadline_norm, src_norm],
        [frag, max_free],
        server_utils,
        [candidate_count / 50.0],  # normalize
    ]).astype(np.float32)

    return state


def generate_dataset(
    scenarios: List[Tuple],
    predictor_configs: List[Tuple[str, str, int]],
    seed: int = 42,
    max_samples: int = 100000,
) -> List[ImitationSample]:
    """Generate imitation dataset across multiple scenarios."""
    all_samples = []

    for topology, num_slots, num_servers, num_requests, arr, ht, preload in scenarios:
        print(f"\nGenerating data: {topology} {num_slots}s arr={arr} ht={ht} seed={seed}")

        num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
        trace_dir = Path(__file__).parent / "traces"
        trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
        trace_path = trace_dir / trace_fname
        if not trace_path.exists():
            trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
        requests = load_trace(trace_path)

        # Load predictors and encoders
        predictors = {}
        encoders = {}
        for pname, ppath, max_srv in predictor_configs:
            if not Path(ppath).exists():
                continue
            predictors[pname] = load_predictor(ppath, max_servers=max_srv)
            env_tmp, enc_tmp, mec_tmp = create_env(topology, num_slots, num_servers, seed)
            encoders[pname] = enc_tmp

        # Create env and agent
        env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
        agent = AdaptiveRuleAgent(predictors, encoders, mec, history_window=100)

        # Run agent and record decisions
        env.reset()
        if hasattr(agent, 'reset'):
            agent.reset()
        # Share env state with sub-agents
        for sub_name, sub_agent in agent._sub_agents.items():
            sub_agent.mec = env.mec
            sub_agent.encoder = env.encoder

        rng = np.random.RandomState(seed)
        bw_map = get_split_bandwidth_map()
        for _ in range(preload):
            src = rng.randint(0, env.net.NUM_NODES)
            dst = rng.randint(0, env.net.NUM_NODES)
            while dst == src:
                dst = rng.randint(0, env.net.NUM_NODES)
            bw = bw_map[rng.randint(0, 3)]
            success, path, start_slot, _ = env.mapper.map(src, dst, bw)
            if success:
                env.active_connections.append((path, start_slot, bw, rng.exponential(10.0), -1, 0.0))

        for req in requests:
            env.advance_time(req.arrival_time)
            split_id, server_id, score, info = agent.decide(req, env)

            # Build state vector
            state = build_state_vector(req, env, server_id, env.encoder, num_servers)

            # Action encoding
            action_id = split_id * len(env.mec.servers) + server_id

            sample = ImitationSample(
                state=state,
                action_id=action_id,
                model_name=req.model_name,
                source_node=req.source_node,
                deadline_ms=req.deadline_ms,
                chosen_strategy=info.get("adaptive_strategy", "unknown"),
                load_level=info.get("load_level", "unknown"),
                split_id=split_id,
                server_id=server_id,
            )
            all_samples.append(sample)

            result = env.step(req, split_id, server_id)
            if hasattr(agent, 'update_history'):
                agent.update_history(result.success, result.info.get("blocking_reason"))

            if len(all_samples) >= max_samples:
                break

        if len(all_samples) >= max_samples:
            break

    print(f"\nTotal samples generated: {len(all_samples)}")
    return all_samples


def main():
    base = Path(__file__).parent.parent / "predictor_mvp"
    predictor_configs = [
        ("mixed-v2b", str(base / "pretrained_mixed_v2b.pt"), 128),
        ("nsfnet-v2b", str(base / "pretrained_nsfnet_v2b.pt"), 128),
    ]

    # Generate across diverse scenarios to get rich data
    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300),
        ("nsfnet", 32, 5, 2000, 8.0, 10.0, 300),
        ("nsfnet", 64, 5, 2000, 5.0, 10.0, 300),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300),
        ("usnet", 64, 5, 2000, 5.0, 10.0, 300),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300),
    ]

    seeds = [42, 123, 456]
    all_samples = []

    for seed in seeds:
        samples = generate_dataset(scenarios, predictor_configs, seed=seed, max_samples=50000)
        all_samples.extend(samples)
        if len(all_samples) >= 50000:
            all_samples = all_samples[:50000]
            break

    # Save dataset
    out_dir = Path(__file__).parent / "data"
    out_dir.mkdir(exist_ok=True)

    # Extract numpy arrays
    states = np.stack([s.state for s in all_samples])
    actions = np.array([s.action_id for s in all_samples], dtype=np.int64)
    metadata = [
        {
            "model_name": s.model_name,
            "source_node": s.source_node,
            "deadline_ms": s.deadline_ms,
            "strategy": s.chosen_strategy,
            "load_level": s.load_level,
            "split_id": s.split_id,
            "server_id": s.server_id,
        }
        for s in all_samples
    ]

    dataset = {
        "states": states,
        "actions": actions,
        "metadata": metadata,
        "state_dim": states.shape[1],
        "num_actions": int(actions.max()) + 1,
        "num_samples": len(all_samples),
    }

    with open(out_dir / "imitation_dataset.pkl", "wb") as f:
        pickle.dump(dataset, f)

    print(f"\nDataset saved: {out_dir / 'imitation_dataset.pkl'}")
    print(f"  Samples: {len(all_samples)}")
    print(f"  State dim: {states.shape[1]}")
    print(f"  Num actions: {dataset['num_actions']}")
    print(f"  Action distribution:")
    unique, counts = np.unique(actions, return_counts=True)
    for a, c in zip(unique[:10], counts[:10]):
        print(f"    action {a}: {c} ({c/len(actions)*100:.1f}%)")
    if len(unique) > 10:
        print(f"    ... and {len(unique)-10} more actions")

    return dataset


if __name__ == "__main__":
    main()
