"""Latency audit v2 for corrected compact poststate ranker (COST239).

Profiles per-component online latency of the full corrected ranker:
- C-side observation and action selection
- R-side observation and feature extraction
- candidate enumeration
- compact v2 poststate feature construction per candidate
- normalization + model forward pass
- total decision time

Reports mean/P50/P95/max in ms.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
)
from sa_hmarl.evaluation.r_poststate_features_v2 import (
    POSTSTATE_COMPACT_V2_FEATURE_NAMES,
    build_poststate_compact_v2_feature,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")


def load_ranker(path: str, device: str = "cpu"):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"),
        int(ckpt["input_dim"]),
        tuple(ckpt["hidden_dims"]),
        float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)
    feature_names = ckpt.get("feature_names", POSTSTATE_COMPACT_V2_FEATURE_NAMES)
    return model, mean, std, list(feature_names)


def _profile(
    ranker_path: str,
    n_states: int = 30,
    seed: int = 7777,
    warmup: int = 1000,
    device: str = "cpu",
) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c("sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt", device)
    agent_r = _load_ppo_r("sa_hmarl/checkpoints/agent_r_mixed.pt", mod_reg, device)
    model, mean, std, feature_names = load_ranker(ranker_path, device)

    env = make_env(
        "xlron_cost239_ptrnet_real", 320, 4, seed,
        modulation_profile="default", max_blocks=10,
        block_sort_strategy="start_asc", path_sort_strategy="hops", k=50,
    )
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env.net.NUM_NODES))
    requests = generate_requests(
        env, rng, src, warmup + n_states + 50,
        0.0625, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2, 3, "default3",
    )
    env.reset(requests)

    times: Dict[str, List[float]] = {
        "obs_c": [],
        "select_c": [],
        "obs_r": [],
        "r_features": [],
        "candidate_selection": [],
        "poststate_v2_per_candidate": [],
        "normalization_forward": [],
        "total_per_decision": [],
    }

    collected = 0
    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        if request_index < warmup:
            env.k = 5
            obs_c = build_agent_c_observation(env, req)
            c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
            split_id, server_id = decode_agent_c_action(int(c_idx), 4)
            env.k = 50
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), r_action)
            continue
        if collected >= n_states:
            break

        t0 = time.perf_counter()
        env.k = 5
        t1 = time.perf_counter()
        obs_c = build_agent_c_observation(env, req)
        t2 = time.perf_counter()
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), 4)
        env.k = 50
        t3 = time.perf_counter()
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        t4 = time.perf_counter()
        r_features, r_mask = agent_r.build_action_features(obs_r)
        t5 = time.perf_counter()
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        candidates = legal[:30]
        t6 = time.perf_counter()
        if not candidates:
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), r_action)
            continue

        ps_times = []
        for action_idx in candidates:
            tps0 = time.perf_counter()
            _ = build_poststate_compact_v2_feature(
                env, req, obs_c, obs_r, r_features, int(action_idx),
                split_id, server_id, feature_names=feature_names,
            )
            tps1 = time.perf_counter()
            ps_times.append((tps1 - tps0) * 1000.0)

        t7 = time.perf_counter()
        X = np.stack([
            build_poststate_compact_v2_feature(
                env, req, obs_c, obs_r, r_features, int(a),
                split_id, server_id, feature_names=feature_names,
            )
            for a in candidates
        ])
        Xn = (X - mean) / std
        with torch.no_grad():
            scores = model(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy()
        t8 = time.perf_counter()
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), r_action)
        t9 = time.perf_counter()

        n = len(candidates)
        times["obs_c"].append((t2 - t1) * 1000.0)
        times["select_c"].append((t3 - t2) * 1000.0)
        times["obs_r"].append((t4 - t3) * 1000.0)
        times["r_features"].append((t5 - t4) * 1000.0)
        times["candidate_selection"].append((t6 - t5) * 1000.0)
        times["poststate_v2_per_candidate"].append(sum(ps_times) / n)
        times["normalization_forward"].append((t8 - t7) * 1000.0)
        times["total_per_decision"].append((t9 - t0) * 1000.0)
        collected += 1

    summary: Dict[str, Any] = {}
    for k, vals in times.items():
        arr = np.array(vals)
        summary[k] = {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }
    summary["n_states"] = n_states
    return summary


def main():
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ranker_checkpoint", required=True,
                        help="Path to a trained compact-v2 ranker checkpoint (ranking_model.pt).")
    parser.add_argument("--n_states", type=int, default=30)
    parser.add_argument("--seed", type=int, default=7777)
    parser.add_argument("--warmup", type=int, default=1000)
    parser.add_argument("--output_dir", default=str(OUTPUT_DIR))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summary = _profile(args.ranker_checkpoint, args.n_states, args.seed, args.warmup)
    (output_dir / "LATENCY_V2.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Online Latency Audit v2 (corrected compact poststate, COST239)",
        "",
        f"Ranker: `{args.ranker_checkpoint}`",
        f"States profiled: {summary['n_states']}",
        "",
        "| Component | Mean ms | P50 ms | P95 ms | Max ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, s in summary.items():
        if not isinstance(s, dict):
            continue
        lines.append(
            f"| {k} | {s['mean_ms']:.3f} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {s['max_ms']:.3f} |"
        )
    lines.append("")
    per_cand = summary.get("poststate_v2_per_candidate", {}).get("mean_ms", 0.0)
    approx_total = per_cand * 30
    lines.append(f"Approximate compact-v2 poststate cost for 30 candidates: **{approx_total:.2f} ms**")
    lines.append(f"Mean total per decision: **{summary['total_per_decision']['mean_ms']:.2f} ms**")
    (output_dir / "LATENCY_V2.md").write_text("\n".join(lines), encoding="utf-8")
    print("Latency v2 report:", output_dir / "LATENCY_V2.md")


if __name__ == "__main__":
    main()
