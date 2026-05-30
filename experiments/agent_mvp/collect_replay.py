"""Collect replay transitions from the TopK2 policy for CorrectionNet training."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import pickle
import numpy as np
import torch
import argparse

from correction_net import ACTION_FEATURE_NAMES_V2
from eval_imitation import create_env
from eval_topk_selector import load_predictor
from fixed_trace import generate_and_save_trace, load_trace
from state_builder import ActionAwareStateBuilder
from topk_selector_agent import TopKSelectorAgent


def ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed):
    trace_fname = f"trace_n{num_nodes}_r{num_requests}_a{arr}_h{ht}_s{seed}.pkl"
    trace_path = trace_dir / trace_fname
    if not trace_path.exists():
        trace_path = generate_and_save_trace(num_nodes, num_requests, arr, ht, seed)
    return trace_path


def preload_env(env, preload, seed):
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


def collect(
    out_path="data/replay_buffer_topk2_fragfeat_v1_30k.pkl",
    num_requests=2000,
    preload=300,
    device="cpu",
):
    scenarios = [
        ("nsfnet", 32, 5, num_requests, 5.0, 10.0, preload, "NSFNET standard", 14),
        ("nsfnet", 64, 5, num_requests, 8.0, 12.0, preload, "NSFNET high load", 14),
        ("usnet", 64, 5, num_requests, 8.0, 12.0, preload, "USNET cross-topo", 28),
    ]
    seeds = [42, 123, 456, 789, 2024]

    predictor_path = str(Path(__file__).parent.parent / "predictor_mvp" / "pretrained_nsfnet_v2b.pt")
    predictor = load_predictor(predictor_path, max_servers=128, device=device)
    imitation_ckpt = "checkpoints/imitation_agent_enhanced.pt"

    transitions = []
    trace_dir = Path(__file__).parent / "traces"

    for topology, num_slots, num_servers, _, arr, ht, preload, label, num_nodes in scenarios:
        print(f"\nCollecting: {label}")
        for seed in seeds:
            trace_path = ensure_trace(trace_dir, num_nodes, num_requests, arr, ht, seed)
            requests = load_trace(trace_path)

            env, encoder, mec = create_env(topology, num_slots, num_servers, seed)
            env.reset()
            preload_env(env, preload, seed)

            agent = TopKSelectorAgent(
                imitation_ckpt,
                predictor,
                encoder,
                env.mec,
                num_servers=num_servers,
                top_k=2,
                alpha=2.0,
                beta=0.3,
                gamma=0.2,
                delta=0.05,
                use_enhanced_state=True,
                device=device,
            )
            builder = ActionAwareStateBuilder(
                encoder,
                predictor,
                env.mec,
                imitation_checkpoint_path=imitation_ckpt,
                num_servers=num_servers,
                use_enhanced_state=True,
                device=device,
            )

            for idx, req in enumerate(requests):
                env.advance_time(req.arrival_time)
                state_dict = builder.build(req, env, top_k=agent.top_k)
                split_id, server_id, _, _ = agent.decide(req, env)
                action_id = split_id * num_servers + server_id

                result = env.step(req, split_id, server_id)
                done = (idx == len(requests) - 1)

                if not done:
                    next_req = requests[idx + 1]
                    env.advance_time(next_req.arrival_time)
                    next_state_dict = builder.build(next_req, env, top_k=agent.top_k)
                else:
                    next_state_dict = state_dict

                transitions.append({
                    "state": state_dict["flat_state"],
                    "action": action_id,
                    "reward": float(result.reward),
                    "next_state": next_state_dict["flat_state"],
                    "done": done,
                    "valid_mask": state_dict["valid_mask"],
                    "topk_mask": state_dict["topk_mask"],
                    "next_valid_mask": next_state_dict["valid_mask"],
                    "next_topk_mask": next_state_dict["topk_mask"],
                    "success": bool(result.success),
                    "delay_ms": float(result.real_delay_ms),
                    "topology": topology,
                })

            print(f"  seed={seed} done, transitions={len(transitions)}")

    out = {
        "transitions": transitions,
        "state_dim": int(builder.flat_state_dim),
        "global_state_dim": 30,
        "action_feat_dim": int(state_dict["action_feat_dim"]),
        "action_feature_names": list(ACTION_FEATURE_NAMES_V2),
        "num_actions": int(state_dict["num_actions"]),
        "num_transitions": len(transitions),
        "collection_policy": "TopK2-30D",
        "reward_version": "v1",
        "num_requests_per_trace": int(num_requests),
        "num_scenarios": len(scenarios),
        "num_seeds": len(seeds),
        "preload": int(preload),
    }

    out_file = Path(__file__).parent / out_path
    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(out, f)

    print(f"\nSaved replay buffer to {out_file}")
    print(f"Transitions: {len(transitions)}")
    return out_file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_path", default="data/replay_buffer_topk2_fragfeat_v1_30k.pkl")
    parser.add_argument("--num_requests", type=int, default=2000)
    parser.add_argument("--preload", type=int, default=300)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    collect(
        out_path=args.out_path,
        num_requests=args.num_requests,
        preload=args.preload,
        device=device,
    )


if __name__ == "__main__":
    main()
