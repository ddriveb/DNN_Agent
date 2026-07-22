"""Part 7: train/online distribution shift analysis."""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
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


DATASET_DIR = Path("sa_hmarl/datasets/v135_paired_feature_ablation_medium")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def _collect_online_states(n_online: int = 300, seed: int = 9999, warmup: int = 1000, eval_requests: int = 1500) -> Dict[str, np.ndarray]:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c("sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt", "cpu")
    agent_r = _load_ppo_r("sa_hmarl/checkpoints/agent_r_mixed.pt", mod_reg, "cpu")
    env = make_env(
        "xlron_cost239_ptrnet_real", 320, 4, seed,
        modulation_profile="default", max_blocks=10,
        block_sort_strategy="start_asc", path_sort_strategy="hops", k=50,
    )
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env.net.NUM_NODES))
    total = warmup + eval_requests
    requests = generate_requests(
        env, rng, src, total,
        0.0625, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2, 3, "default3",
    )
    env.reset(requests)
    online_features = []
    online_v1_features = []
    online_returns_proxy = []
    online_counts = []
    online_server_utils = []
    online_path_idx = []
    online_mod_idx = []
    online_block_idx = []
    collected = 0
    step = max(1, eval_requests // n_online)
    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        env.k = 5
        env.path_sort_strategy = "hops"
        env.block_sort_strategy = "start_asc"
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), 4)
        env.k = 50
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        if request_index >= warmup and (request_index - warmup) % step == 0 and len(legal) >= 2:
            # Build features for up to 30 legal candidates
            candidates = legal[:30]
            feats = []
            v1s = []
            proxies = []
            p_idx = []
            m_idx = []
            b_idx = []
            num_mods = len(obs_r["mod_names"])
            max_blocks = int(env.max_blocks)
            for action_idx in candidates:
                psf = build_poststate_v1_feature(env, req, obs_c, obs_r, r_features, int(action_idx), split_id, server_id)
                feats.append(psf)
                v1s.append(r_features[int(action_idx)])
                # Proxy return components
                path_km = r_features[int(action_idx)][0]
                req_fs = r_features[int(action_idx)][7]
                proxies.append(-0.05 * path_km - 0.05 * req_fs)
                p_idx.append(action_idx // (num_mods * max_blocks))
                rem = action_idx % (num_mods * max_blocks)
                m_idx.append(rem // max_blocks)
                b_idx.append(rem % max_blocks)
            online_features.append(np.stack(feats))
            online_v1_features.append(np.stack(v1s))
            online_returns_proxy.append(np.array(proxies))
            online_counts.append(len(candidates))
            online_server_utils.append(float(obs_c["server_utilizations"][server_id]))
            online_path_idx.append(p_idx)
            online_mod_idx.append(m_idx)
            online_block_idx.append(b_idx)
            collected += 1
            if collected >= n_online:
                break
        # Advance real env with PPO-R
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        from sa_hmarl.env.observation_builder import decode_agent_r_action
        r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), r_action)

    # Pad to same candidate count for array storage
    max_c = max(f.shape[0] for f in online_features)
    n = len(online_features)
    ps_pad = np.zeros((n, max_c, online_features[0].shape[1]), dtype=np.float32)
    v1_pad = np.zeros((n, max_c, online_v1_features[0].shape[1]), dtype=np.float32)
    proxy_pad = np.full((n, max_c), np.nan, dtype=np.float32)
    mask = np.zeros((n, max_c), dtype=bool)
    for i, (ps, v1, prox) in enumerate(zip(online_features, online_v1_features, online_returns_proxy)):
        ps_pad[i, :len(ps)] = ps
        v1_pad[i, :len(v1)] = v1
        proxy_pad[i, :len(prox)] = prox
        mask[i, :len(ps)] = True
    return {
        "features_poststate_v1": ps_pad,
        "features_v1": v1_pad,
        "proxy_returns": proxy_pad,
        "mask": mask,
        "candidate_counts": np.array(online_counts),
        "server_utils": np.array(online_server_utils),
        "path_idx": online_path_idx,
        "mod_idx": online_mod_idx,
        "block_idx": online_block_idx,
    }


def _smd(train_vals: np.ndarray, online_vals: np.ndarray) -> float:
    m1, s1 = train_vals.mean(), train_vals.std() + 1e-8
    m2, s2 = online_vals.mean(), online_vals.std() + 1e-8
    return float((m2 - m1) / np.sqrt((s1 ** 2 + s2 ** 2) / 2))


def _psi(train_vals: np.ndarray, online_vals: np.ndarray, bins: int = 10) -> float:
    # Use train quantile bins
    edges = np.quantile(train_vals, np.linspace(0, 1, bins + 1))
    edges[0] -= 1e-6
    edges[-1] += 1e-6
    train_counts, _ = np.histogram(train_vals, edges)
    online_counts, _ = np.histogram(online_vals, edges)
    train_prop = train_counts / max(train_counts.sum(), 1)
    online_prop = online_counts / max(online_counts.sum(), 1)
    psi = 0.0
    for p1, p2 in zip(train_prop, online_prop):
        if p1 > 0 and p2 > 0:
            psi += (p2 - p1) * np.log(p2 / p1)
    return float(psi)


def main():
    print("[part7] Collecting online states...")
    online = _collect_online_states(n_online=300)
    print(f"[part7] Collected {online['features_poststate_v1'].shape[0]} online groups")

    meta = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    names = meta["feature_names_poststate_v1"]

    out = {"per_feature": {}, "summary": {}}
    for split in ["train", "val", "test"]:
        d = np.load(str(DATASET_DIR / f"{split}.npz"), allow_pickle=True)
        mask = d["mask"].astype(bool)
        train_ps = d["features_poststate_v1"][mask]
        online_ps = online["features_poststate_v1"][online["mask"]]
        per_feat = []
        for j, name in enumerate(names):
            smd = _smd(train_ps[:, j], online_ps[:, j])
            psi = _psi(train_ps[:, j], online_ps[:, j])
            train_min, train_max = float(train_ps[:, j].min()), float(train_ps[:, j].max())
            out_of_range = float((online_ps[:, j] < train_min).mean() + (online_ps[:, j] > train_max).mean())
            per_feat.append({"feature": name, "smd": smd, "psi": psi, "out_of_range_rate": out_of_range})
        out["per_feature"][split] = per_feat

    # Candidate count and server util comparison
    for split in ["train", "val", "test"]:
        d = np.load(str(DATASET_DIR / f"{split}.npz"), allow_pickle=True)
        counts = d["mask"].sum(axis=1)
        out["summary"][f"{split}_candidate_count_mean"] = float(counts.mean())
    out["summary"]["online_candidate_count_mean"] = float(online["candidate_counts"].mean())
    out["summary"]["online_server_util_mean"] = float(online["server_utils"].mean())

    (OUTPUT_DIR / "TRAIN_ONLINE_SHIFT.json").write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")

    lines = [
        "# Train / Online Distribution Shift",
        "",
        f"Online states collected: {online['features_poststate_v1'].shape[0]} (after {1000} warmup)",
        "",
        "## Candidate count and server utilization",
        "",
        "| Split | Mean candidates | Server util mean |",
        "|---|---:|---:|",
        f"| train | {out['summary']['train_candidate_count_mean']:.1f} | - |",
        f"| val | {out['summary']['val_candidate_count_mean']:.1f} | - |",
        f"| test | {out['summary']['test_candidate_count_mean']:.1f} | - |",
        f"| online | {out['summary']['online_candidate_count_mean']:.1f} | {out['summary']['online_server_util_mean']:.3f} |",
        "",
        "## Top features by standardized mean difference (train vs online)",
        "",
        "| Feature | SMD | PSI | Out-of-train-range rate |",
        "|---|---:|---:|---:|",
    ]
    # Show top 15 by absolute SMD for train split
    feats = out["per_feature"]["train"]
    feats_sorted = sorted(feats, key=lambda x: abs(x["smd"]), reverse=True)[:15]
    for f in feats_sorted:
        lines.append(f"| {f['feature']} | {f['smd']:.3f} | {f['psi']:.3f} | {f['out_of_range_rate']:.2%} |")
    lines.append("")
    (OUTPUT_DIR / "TRAIN_ONLINE_SHIFT.md").write_text("\n".join(lines), encoding="utf-8")
    print("Part 7 complete. Report written to", OUTPUT_DIR / "TRAIN_ONLINE_SHIFT.md")


if __name__ == "__main__":
    main()
