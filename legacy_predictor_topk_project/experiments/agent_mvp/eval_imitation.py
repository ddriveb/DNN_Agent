"""Evaluate Imitation Agent in closed loop against AdaptiveRA and baselines."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
import json
from typing import Dict

from env import OpticalNetwork
from mapper import KSPMapper
from encoder import Encoder
from mec_servers import MECCluster
from env_wrapper import DNNOpticalEnv
from baselines import YinLikeAgent
from train_imitation import ImitationAgent
from fixed_trace import load_trace, generate_and_save_trace


def create_env(topology, num_slots, num_servers, seed):
    net = OpticalNetwork(topology=topology, num_slots=num_slots, seed=seed)
    mapper = KSPMapper(net, k=3)
    encoder = Encoder(net, k=3)
    encoder.encode = encoder.encode_v2b
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=num_servers, seed=seed)
    env = DNNOpticalEnv(net, encoder, mapper, mec, reward_version="v1")
    return env, encoder, mec


def run_agent(agent, requests, topology, num_slots, num_servers, preload, seed):
    env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
    env.reset()
    if hasattr(agent, 'mec'):
        agent.mec = env.mec
    if hasattr(agent, 'encoder'):
        agent.encoder = env.encoder

    rng = np.random.RandomState(seed)
    from dnn_models import get_split_bandwidth_map
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

    rewards = []
    action_match_count = 0
    teacher_action = None

    for req in requests:
        env.advance_time(req.arrival_time)
        split_id, server_id, score, info = agent.decide(req, env)
        result = env.step(req, split_id, server_id)
        rewards.append(result.reward)

        if "teacher_action" in info:
            teacher_action = info["teacher_action"]
            if split_id == teacher_action[0] and server_id == teacher_action[1]:
                action_match_count += 1

    metrics = env.get_metrics()
    metrics["avg_reward"] = float(np.mean(rewards))
    if teacher_action is not None:
        metrics["action_match_rate"] = action_match_count / len(requests)
    return metrics


def build_state_vector_for_imitation(req, env, encoder, num_servers):
    """Build the same state vector used during training."""
    import numpy as np
    model = req.model
    src = req.source_node

    # Use first server as default for state encoding (simplified)
    srv = env.mec.servers[0]
    dst = srv.node_id

    # 1. Network encoding
    z = encoder.encode(src, dst)

    # 2. Request features
    model_id = float(model.model_id)
    deadline_norm = req.deadline_ms / 200.0
    src_norm = src / env.net.NUM_NODES

    # 3. Global network state
    frag = env._global_frag_index()
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
    target_dim = num_servers
    if len(server_utils) < target_dim:
        server_utils = np.pad(server_utils, (0, target_dim - len(server_utils)))
    elif len(server_utils) > target_dim:
        server_utils = server_utils[:target_dim]

    # 5. Candidate count
    num_splits = len(model.splits)
    num_srv = len(env.mec.servers)
    candidate_count = num_splits * num_srv

    state = np.concatenate([
        z,
        [model_id, deadline_norm, src_norm],
        [frag, max_free],
        server_utils,
        [candidate_count / 50.0],
    ]).astype(np.float32)
    return state


class ImitationAgentWrapper:
    """Wrap trained ImitationAgent for env evaluation."""

    def __init__(self, checkpoint_path: str, num_servers: int = 5):
        self.num_servers = num_servers
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        self.model = ImitationAgent(
            checkpoint["state_dim"],
            checkpoint["num_actions"],
            hidden_dims=checkpoint.get("hidden_dims", (128, 128)),
        )
        self.model.load_state_dict(checkpoint["model_state"])
        self.model.eval()
        self.mec = None
        self.encoder = None
        self.val_acc = checkpoint.get("best_val_acc", 0.0)

    def decide(self, request, network_state):
        state = build_state_vector_for_imitation(request, network_state, self.encoder, self.num_servers)
        action_id = self.model.predict_action(state)

        num_srv = len(self.mec.servers)
        split_id = action_id // num_srv
        server_id = action_id % num_srv

        # Ensure split_id is valid
        split_id = min(split_id, len(request.model.splits) - 1)
        server_id = min(server_id, len(self.mec.servers) - 1)

        return split_id, server_id, 0.0, {"type": "imitation", "action_id": action_id}


def main():
    scenarios = [
        ("nsfnet", 32, 5, 2000, 5.0, 10.0, 300, "NSFNET standard"),
        ("nsfnet", 64, 5, 2000, 8.0, 12.0, 300, "NSFNET high load"),
        ("usnet", 64, 5, 2000, 8.0, 12.0, 300, "USNET cross-topo"),
    ]

    seeds = [42, 123, 456, 789, 2024]
    checkpoint_path = "checkpoints/imitation_agent.pt"

    all_results = {}

    for topology, num_slots, num_servers, num_requests, arr, ht, preload, label in scenarios:
        print(f"\n{'='*90}")
        print(f"IMITATION AGENT EVAL: {label}")
        print(f"{'='*90}")

        num_nodes = 14 if topology == "nsfnet" else (28 if topology == "usnet" else 50)
        trace_dir = Path(__file__).parent / "traces"

        for seed in seeds:
            trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
            trace_path = trace_dir / trace_fname
            if not trace_path.exists():
                trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
            requests = load_trace(trace_path)

            # Imitation Agent
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = ImitationAgentWrapper(checkpoint_path, num_servers=num_servers)
            agent.mec = env.mec
            agent.encoder = env.encoder
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "ImitationAgent"), []).append(metrics)

            # YinLike baseline
            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            agent = YinLikeAgent(mec)
            metrics = run_agent(agent, requests, topology, num_slots, num_servers, preload, seed)
            all_results.setdefault((label, "YinLike"), []).append(metrics)

    # Print summary
    print("\n" + "=" * 100)
    print("IMITATION AGENT CLOSED-LOOP RESULTS (Multi-Seed)")
    print("=" * 100)
    print(f"{'Scenario':<25} {'Agent':<20} {'Blocking%':>15} {'Accept%':>15} {'Reward':>15}")
    print("-" * 100)

    for label in sorted(set(l for l, _ in all_results.keys())):
        for agent_name in ["YinLike", "ImitationAgent"]:
            if (label, agent_name) not in all_results:
                continue
            runs = all_results[(label, agent_name)]
            br = [r["blocking_rate"] for r in runs]
            ar = [r["acceptance_rate"] for r in runs]
            rw = [r["avg_reward"] for r in runs]
            print(f"{label:<25} {agent_name:<20} {np.mean(br)*100:>7.2f}±{np.std(br)*100:<5.2f}% "
                  f"{np.mean(ar)*100:>7.2f}±{np.std(ar)*100:<5.2f}% {np.mean(rw):>8.3f}±{np.std(rw):<5.3f}")

    # Save
    out_dir = Path(__file__).parent / "results"
    out_dir.mkdir(exist_ok=True)
    with open(out_dir / "imitation_eval.json", "w") as f:
        serializable = {}
        for (label, agent), runs in all_results.items():
            serializable[f"{label}__{agent}"] = [{k: float(v) if isinstance(v, (np.floating, float)) else v
                                                   for k, v in r.items()} for r in runs]
        json.dump(serializable, f, indent=2)

    print(f"\nSaved to {out_dir / 'imitation_eval.json'}")
    return all_results


if __name__ == "__main__":
    main()
