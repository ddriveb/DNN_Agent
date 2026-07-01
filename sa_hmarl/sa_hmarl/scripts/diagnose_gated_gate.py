"""Diagnose gated typed mean-field gate activations during evaluation.

Loads a gated Agent-C checkpoint and runs the same K=5 evaluation protocol,
recording per-step gate statistics (mean/min/max/std and chosen-action gate)
from the GatedMaskedPPOActorNetwork.

Usage::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.scripts.diagnose_gated_gate \
        --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_gated_tmf_indppo_s42_s20_r80_best.pt \
        --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
        --seeds 42,123,456,789,101112 --episodes 20 --output gate_diag_s42.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"),
    )
    num_servers = ckpt_args.get("num_servers")
    if num_servers is None:
        num_servers = ckpt.get("num_servers")
    agent_c_kwargs = dict(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    if feature_mode in ("mean_field", "typed_mean_field", "gated_typed_mean_field"):
        agent_c_kwargs["num_servers"] = num_servers
    agent = PPOAgentC(**agent_c_kwargs)
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=ckpt.get("agent_r_feature_mode", "default"),
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _gate_stats_for_obs(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[Optional[torch.Tensor], Optional[int], np.ndarray]:
    """Build features, apply risk mask, return gate values for all actions.

    Returns:
        gate_values: Tensor of shape (num_actions,) or None if no valid actions.
        chosen_action: Index chosen by deterministic policy.
        mask: Risk-masked boolean array.
    """
    features, mask = agent_c.build_action_features(obs_c)

    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )

    n_valid = int(mask.sum())
    if n_valid == 0:
        return None, None, mask

    x = torch.tensor(features, dtype=torch.float32, device=agent_c.device).unsqueeze(0)
    with torch.no_grad():
        logits, gate = agent_c.policy_net.forward_with_gate(x)

    logits = logits.squeeze(0).masked_fill(~torch.tensor(mask, device=agent_c.device), -1e9)
    chosen_action = int(torch.argmax(logits).item())
    gate_values = gate.squeeze(0).cpu()
    return gate_values, chosen_action, mask


def _eval_episode(
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    requests: List,
    env_proto,
    args,
    ep_idx: int,
) -> Dict[str, Any]:
    """Evaluate one episode and collect gate diagnostics."""
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities,
        k=args.k_paths,
    )
    env.reset(requests)

    total = blocked = success = 0
    gate_records: List[Dict[str, float]] = []

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        gate_values, chosen_action, c_mask = _gate_stats_for_obs(
            agent_c, obs_c, env.net.num_slots,
        )

        n_valid_c = int(c_mask.sum())
        if gate_values is not None:
            valid_gates = gate_values[c_mask].numpy()
            record = {
                "n_valid_c": n_valid_c,
                "gate_mean": float(valid_gates.mean()),
                "gate_min": float(valid_gates.min()),
                "gate_max": float(valid_gates.max()),
                "gate_std": float(valid_gates.std()),
                "gate_chosen": float(gate_values[chosen_action].item()),
                "gate_median": float(np.median(valid_gates)),
            }
            # Histogram bins: 0.0-0.1, 0.1-0.2, ..., 0.9-1.0
            hist, _ = np.histogram(valid_gates, bins=10, range=(0.0, 1.0))
            for i, count in enumerate(hist):
                record[f"gate_bin_{i}"] = int(count)
        else:
            record = {
                "n_valid_c": 0,
                "gate_mean": np.nan,
                "gate_min": np.nan,
                "gate_max": np.nan,
                "gate_std": np.nan,
                "gate_chosen": np.nan,
                "gate_median": np.nan,
            }
        gate_records.append(record)

        action_c = (
            (0, 0) if chosen_action is None
            else decode_agent_c_action(chosen_action, num_servers)
        )
        split_id, server_id = action_c

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        action_r_idx = agent_r.select_action(obs_r, deterministic=True)
        action_r_tuple = (
            (0, 0, 0) if action_r_idx is None
            else decode_agent_r_action(
                action_r_idx, len(obs_r["mod_names"]), env.max_blocks,
            )
        )

        _, _, _, info = env.step(action_c, action_r_tuple)
        total += 1
        if info.get("success", False):
            success += 1
        else:
            blocked += 1

    n = max(total, 1)
    result = {
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocked / n,
        "gate_records": gate_records,
    }
    return result


def _aggregate_gate_records(ep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate gate statistics across episodes."""
    all_records: List[Dict[str, float]] = []
    for r in ep_results:
        all_records.extend(r["gate_records"])

    valid_records = [r for r in all_records if not np.isnan(r["gate_mean"])]
    agg: Dict[str, Any] = {
        "n_steps_total": len(all_records),
        "n_steps_with_valid_c": len(valid_records),
    }
    if not valid_records:
        return agg

    for key in ["gate_mean", "gate_min", "gate_max", "gate_std", "gate_median", "gate_chosen"]:
        vals = [r[key] for r in valid_records]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    # Aggregate histogram bins
    for i in range(10):
        counts = [r.get(f"gate_bin_{i}", 0) for r in valid_records]
        total_valid_actions = sum(
            r.get(f"gate_bin_{j}", 0) for r in valid_records for j in range(10)
        )
        agg[f"gate_bin_{i}_count"] = int(sum(counts))
        agg[f"gate_bin_{i}_frac"] = (
            float(sum(counts) / total_valid_actions) if total_valid_actions > 0 else 0.0
        )

    return agg


def main():
    parser = argparse.ArgumentParser(description="Diagnose gated TMF gate activations")
    parser.add_argument("--agent_c_checkpoint", required=True)
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--seeds", default="42,123,456,789,101112")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--num_requests", type=int, default=80)
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default=None, help="JSON output path")
    parser.add_argument("--src_node", type=int, default=0)
    args = parser.parse_args()

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, device=args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, device=args.device)

    if not hasattr(agent_c.policy_net, "forward_with_gate"):
        raise ValueError(
            "Loaded policy does not support forward_with_gate. "
            "Only gated_typed_mean_field checkpoints are supported."
        )

    seeds = [int(s) for s in args.seeds.split(",")]

    server_nodes = list(range(args.num_servers))
    capacities = [1.0] * args.num_servers
    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities,
        k=args.k_paths,
    )

    per_seed_results: Dict[int, Any] = {}
    all_ep_results: List[Dict[str, Any]] = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        requests = generate_requests(
            env_proto, rng, src_node=args.src_node,
            num_requests=args.num_requests,
        )
        ep_results: List[Dict[str, Any]] = []
        for ep_idx in range(args.episodes):
            result = _eval_episode(
                agent_c, agent_r, requests, env_proto, args, ep_idx,
            )
            ep_results.append(result)
        agg = _aggregate_gate_records(ep_results)
        agg["blocking_rate"] = float(np.mean([r["blocking_rate"] for r in ep_results]))
        per_seed_results[seed] = agg
        all_ep_results.extend(ep_results)

    overall = _aggregate_gate_records(all_ep_results)
    overall["blocking_rate"] = float(np.mean([r["blocking_rate"] for r in all_ep_results]))

    output = {
        "checkpoint": args.agent_c_checkpoint,
        "seeds": seeds,
        "episodes": args.episodes,
        "per_seed": {str(k): v for k, v in per_seed_results.items()},
        "overall": overall,
    }

    print(json.dumps(output, indent=2))
    if args.output:
        Path(args.output).write_text(json.dumps(output, indent=2))
        print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()
