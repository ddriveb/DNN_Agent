"""Part 9: online latency breakdown of poststate vs v1 ranker."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch

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
from sa_hmarl.evaluation.r_poststate_features import build_poststate_v1_feature
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _profile(n_states: int = 30, seed: int = 7777, warmup: int = 1000) -> Dict[str, Any]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c("sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt", "cpu")
    agent_r = _load_ppo_r("sa_hmarl/checkpoints/agent_r_mixed.pt", mod_reg, "cpu")
    # Load poststate checkpoint
    ckpt = torch.load("sa_hmarl/experiments/v135_parallel_paired/ranker_poststate_v1_s456/ranking_model.pt", map_location="cpu", weights_only=False)
    from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"), int(ckpt["input_dim"]), tuple(ckpt["hidden_dims"]), float(ckpt.get("dropout", 0.0))
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    mean = np.asarray(ckpt["feature_mean"], dtype=np.float32)
    std = np.asarray(ckpt["feature_std"], dtype=np.float32)

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

    times = {
        "obs_c": [], "obs_r": [], "r_features": [], "candidate_selection": [],
        "v1_feature_per_candidate": [], "poststate_feature_per_candidate": [],
        "normalization_forward": [], "total_per_decision": [],
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
            # No feasible optical candidates; skip this decision.
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), r_action)
            continue
        v1_times = []
        ps_times = []
        for action_idx in candidates:
            tv0 = time.perf_counter()
            _ = r_features[int(action_idx)]
            tv1 = time.perf_counter()
            _ = build_poststate_v1_feature(env, req, obs_c, obs_r, r_features, int(action_idx), split_id, server_id)
            tv2 = time.perf_counter()
            v1_times.append((tv1 - tv0) * 1000.0)
            ps_times.append((tv2 - tv1) * 1000.0)
        t7 = time.perf_counter()
        X = np.stack([
            build_poststate_v1_feature(env, req, obs_c, obs_r, r_features, int(a), split_id, server_id)
            for a in candidates
        ])
        Xn = (X - mean) / std
        with torch.no_grad():
            scores = model(torch.tensor(Xn, dtype=torch.float32)).cpu().numpy()
        t8 = time.perf_counter()
        # Advance env
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), r_action)
        t9 = time.perf_counter()
        n = len(candidates)
        times["obs_c"].append((t2 - t1) * 1000.0)
        times["obs_r"].append((t4 - t3) * 1000.0)
        times["r_features"].append((t5 - t4) * 1000.0)
        times["candidate_selection"].append((t6 - t5) * 1000.0)
        times["v1_feature_per_candidate"].append(sum(v1_times) / n)
        times["poststate_feature_per_candidate"].append(sum(ps_times) / n)
        times["normalization_forward"].append((t8 - t7) * 1000.0)
        times["total_per_decision"].append((t8 - t0) * 1000.0)
        collected += 1

    summary = {}
    for k, vals in times.items():
        arr = np.array(vals)
        summary[k] = {
            "mean_ms": float(arr.mean()),
            "p50_ms": float(np.median(arr)),
            "p95_ms": float(np.percentile(arr, 95)),
            "max_ms": float(arr.max()),
        }
    return summary


def main():
    summary = _profile()
    (OUTPUT_DIR / "LATENCY_BREAKDOWN.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [
        "# Online Latency Breakdown",
        "",
        "| Component | Mean ms | P50 ms | P95 ms | Max ms |",
        "|---|---:|---:|---:|---:|",
    ]
    for k, s in summary.items():
        lines.append(f"| {k} | {s['mean_ms']:.3f} | {s['p50_ms']:.3f} | {s['p95_ms']:.3f} | {s['max_ms']:.3f} |")
    lines.append("")
    total = summary["total_per_decision"]["mean_ms"]
    ps_total = summary["poststate_feature_per_candidate"]["mean_ms"] * 30  # approximate for 30 candidates
    lines.append(f"Approximate analytical afterstate cost for 30 candidates: **{ps_total:.2f} ms**")
    lines.append(f"Total mean per decision: **{total:.2f} ms**")
    (OUTPUT_DIR / "LATENCY_BREAKDOWN.md").write_text("\n".join(lines), encoding="utf-8")
    print("Part 9 complete. Report written to", OUTPUT_DIR / "LATENCY_BREAKDOWN.md")


if __name__ == "__main__":
    main()
