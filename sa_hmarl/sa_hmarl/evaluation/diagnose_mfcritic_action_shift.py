"""Same-state diagnosis of action shifts caused by request-MF critic training."""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path

import numpy as np

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
)
from sa_hmarl.evaluation.diagnose_candidate_resource_impact import (
    _compute_spectrum_field,
    _select_r_action,
)
from sa_hmarl.evaluation.eval_c_closed_loop import _load_ppo_c, _load_ppo_r, _select_c_action
from sa_hmarl.training.utils import generate_requests, make_env


def _selected_metrics(env, req, next_req, agent_c, agent_r, args):
    obs_c = build_agent_c_observation(env, req)
    action_idx, raw_mask = _select_c_action(agent_c, obs_c, env.net.num_slots)
    split_id, server_id = decode_agent_c_action(action_idx, args.num_servers)
    candidate_idx = split_id * args.num_servers + server_id
    feat = obs_c["candidate_features"][candidate_idx]
    selected_valid_r = int(obs_c["feasible_counts"][split_id][server_id])
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
    _, _, _, info = env.step((split_id, server_id), action_r)

    if next_req is None:
        k_c_next = k_r_next = 0
        phi_next = 0.0
    else:
        obs_next = build_agent_c_observation(env, next_req)
        k_c_next, k_r_next = _compute_spectrum_field(obs_next, 0.95)
        phi_next = float(np.log1p(k_c_next) + args.alpha * np.log1p(k_r_next))

    spectrum = np.asarray(feat.get("spectrum_summary", []), dtype=np.float64)
    lfb_max = float(spectrum[0]) if len(spectrum) else 0.0
    safe_fs = float(feat.get("safe_fs_estimate") or 0.0)
    return {
        "action_idx": int(action_idx),
        "split_id": int(split_id),
        "server_id": int(server_id),
        "raw_empty": bool(np.sum(raw_mask) == 0),
        "success": bool(info.get("success", False)),
        "reason": str(info.get("reason", "unknown")),
        "delay_ms": float(info.get("delay_ms", 0.0)) if info.get("success", False) else np.nan,
        "num_slots": float(info.get("num_slots", 0.0)) if info.get("success", False) else np.nan,
        "selected_valid_r": selected_valid_r,
        "server_util_before": float(obs_c["server_utilizations"][server_id]),
        "server_available_ratio": float(feat.get("server_available_compute", 0.0))
        / max(float(feat.get("server_capacity", 1.0)), 1e-8),
        "edge_compute_cost": float(feat.get("edge_compute_cost", 0.0)),
        "path_km": float(feat.get("server_min_path_km", 0.0)),
        "safe_fs": safe_fs,
        "lfb_max": lfb_max,
        "lfb_pressure": safe_fs / max(lfb_max, 1.0),
        "k_c_next": int(k_c_next),
        "k_r_next": int(k_r_next),
        "phi_next": phi_next,
        "collapse_next": bool(next_req is not None and k_c_next == 0),
    }


def _mean(records, key):
    values = np.asarray([record[key] for record in records], dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if len(values) else 0.0


def run(args):
    seeds = [int(seed) for seed in args.seeds.split(",")]
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        k=args.k_paths,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
    )
    base_c = _load_ppo_c(args.base_checkpoint)
    mf_c = _load_ppo_c(args.mf_checkpoint)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, env_proto.mod_reg)
    records = {"base": [], "mf": []}
    disagreements = []

    for seed in seeds:
        rng = np.random.RandomState(seed)
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            requests = generate_requests(
                env_proto,
                rng,
                src,
                num_requests=args.requests_per_episode,
                arrival_interval=0.25,
                holding_min=4.0,
                holding_max=10.0,
                deadline_min=30.0,
                deadline_max=100.0,
                size_min_mb=5.0,
                size_max_mb=30.0,
                edge_cost_min=0.5,
                edge_cost_max=15.0,
                num_splits=3,
                split_profile="default3",
                traffic_mode="markov_regime",
                regime_stay_prob=args.regime_stay_prob,
            )
            env = make_env(
                topology=args.topology,
                num_slots=args.num_slots,
                num_servers=args.num_servers,
                seed=42,
                k=args.k_paths,
                max_blocks=args.max_blocks,
                block_sort_strategy=args.block_sort_strategy,
            )
            env.reset(requests)
            for index, req in enumerate(requests):
                next_req = requests[index + 1] if index + 1 < len(requests) else None
                snapshot = copy.deepcopy(env)
                base = _selected_metrics(
                    copy.deepcopy(snapshot), req, next_req, base_c, agent_r, args
                )
                mf = _selected_metrics(
                    copy.deepcopy(snapshot), req, next_req, mf_c, agent_r, args
                )
                records["base"].append(base)
                records["mf"].append(mf)
                if base["action_idx"] != mf["action_idx"]:
                    disagreements.append((base, mf))
                # Advance the shared trajectory with the base policy only.
                _selected_metrics(env, req, next_req, base_c, agent_r, args)

    keys = (
        "success",
        "delay_ms",
        "num_slots",
        "selected_valid_r",
        "server_util_before",
        "server_available_ratio",
        "edge_compute_cost",
        "path_km",
        "safe_fs",
        "lfb_max",
        "lfb_pressure",
        "k_c_next",
        "k_r_next",
        "phi_next",
        "collapse_next",
    )
    summary = {
        policy: {
            **{key: _mean(policy_records, key) for key in keys},
            "failure_reasons": dict(Counter(r["reason"] for r in policy_records if not r["success"])),
            "split_distribution": dict(Counter(r["split_id"] for r in policy_records)),
            "server_distribution": dict(Counter(r["server_id"] for r in policy_records)),
        }
        for policy, policy_records in records.items()
    }
    disagreement_base = [pair[0] for pair in disagreements]
    disagreement_mf = [pair[1] for pair in disagreements]
    report = {
        "config": vars(args),
        "states": len(records["base"]),
        "action_disagreements": len(disagreements),
        "action_disagreement_rate": len(disagreements) / max(len(records["base"]), 1),
        "overall": summary,
        "disagreement_only": {
            "base": {key: _mean(disagreement_base, key) for key in keys},
            "mf": {key: _mean(disagreement_mf, key) for key in keys},
        },
    }
    return report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_checkpoint", required=True)
    parser.add_argument("--mf_checkpoint", required=True)
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", default="mixed")
    parser.add_argument("--regime_stay_prob", type=float, default=0.95)
    parser.add_argument("--seeds", default="42,123,456")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--alpha", type=float, default=0.3)
    parser.add_argument("--output", default="sa_hmarl/experiments/mfcritic_action_shift.json")
    args = parser.parse_args()
    report = run(args)
    Path(args.output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
