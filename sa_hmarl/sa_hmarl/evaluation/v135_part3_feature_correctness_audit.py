"""Part 3: feature correctness audit for poststate_v1 features."""
from __future__ import annotations

import copy
import json
import os
import random
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats

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
from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    _global_spectrum_stats,
    _path_conflict_count,
    _path_to_edges,
    _per_edge_stats_after,
    _summarize_availability,
    build_poststate_v1_feature,
    compute_optical_afterstate,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


DATASET_DIR = Path("sa_hmarl/datasets/v135_paired_feature_ablation_medium")
OUTPUT_DIR = Path("sa_hmarl/experiments/v135_root_cause_diagnosis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

NEW_FEATURE_NAMES = POSTSTATE_V1_FEATURE_NAMES[25:]


def _true_optical_afterstate(env, path_idx: int, mod_idx: int, block_idx: int, obs_r: Dict[str, Any],
                             split_id: int, server_id: int) -> Tuple[Dict[str, float], bool]:
    """Execute the action in a copy and measure true afterstate statistics."""
    branch = copy.deepcopy(env)
    num_total_slots = int(branch.net.num_slots)
    path = obs_r["candidate_paths"][path_idx]
    path_edges = _path_to_edges(path)
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    start_slot = int(blocks[block_idx][0])
    fs_req = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    num_slots = int(fs_req) if fs_req is not None else int(blocks[block_idx][1])
    r_action = decode_agent_r_action(path_idx * len(obs_r["mod_names"]) * branch.max_blocks + mod_idx * branch.max_blocks + block_idx,
                                     len(obs_r["mod_names"]), branch.max_blocks)
    _, _, _, info = branch.step((split_id, server_id), r_action)
    if not info.get("success", False):
        return {}, False

    avail_after = branch.net.get_available_slots(path)
    stats_after_path = _summarize_availability(avail_after, num_total_slots)
    stats_after_global = _global_spectrum_stats(branch.net.link_states, num_total_slots)
    edge_stats = _per_edge_stats_after(branch.net.link_states, path_edges, num_total_slots)
    # Path conflict excludes the connection we just added.
    raw_conflict = _path_conflict_count(branch, set(path_edges))
    # The newly added connection always shares the path edges, so subtract it.
    conflict_after = max(raw_conflict - 1, 0)
    return {
        "path_lfb_after": float(stats_after_path["max_free_slots"]),
        "path_free_ratio_after": float(stats_after_path["free_ratio"]),
        "global_lfb_after": float(stats_after_global["largest_free_block_ratio"]) * num_total_slots,
        "global_lfb_ratio_after": float(stats_after_global["largest_free_block_ratio"]),
        "frag_after": float(stats_after_global["avg_frag_index"]),
        "free_block_count_after": float(stats_after_global["avg_free_block_count"]),
        "min_edge_lfb_margin_after": float(edge_stats["min_edge_lfb_margin"]),
        "min_edge_free_ratio_after": float(edge_stats["min_edge_free_ratio"]),
        "bottleneck_edge_util_after": float(edge_stats["bottleneck_edge_util"]),
        "occupied_slot_hops": float(max(len(path_edges), 1) * num_slots),
        "path_conflict_after": float(conflict_after),
    }, True


def _dataset_feature_stats() -> Dict[str, Any]:
    meta = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    v1_names = meta["feature_names_v1"]
    ps_names = meta["feature_names_poststate_v1"]
    out = {}
    for split in ["train", "val", "test"]:
        d = np.load(str(DATASET_DIR / f"{split}.npz"), allow_pickle=True)
        mask = d["mask"].astype(bool)
        v1 = d["features_v1"][mask]
        ps = d["features_poststate_v1"][mask]
        returns_valid = d["returns"][mask]
        returns_full = d["returns"]
        # Oracle top-1 indicator per group (full 2D)
        oracle = np.zeros_like(returns_full, dtype=bool)
        regret = np.zeros_like(returns_full)
        for i in range(returns_full.shape[0]):
            m = mask[i]
            if m.sum() == 0:
                continue
            group = returns_full[i, m]
            best = group.max()
            oracle[i, m] = group == best
            regret[i, m] = best - group
        oracle_valid = oracle[mask]
        regret_valid = regret[mask]

        stats = []
        for j, name in enumerate(ps_names):
            col = ps[:, j]
            const_rate = float((np.abs(col - col.mean()) < 1e-6).mean())
            # group-wise variance
            group_vars = []
            uniq_per_group = []
            for i in range(d["returns"].shape[0]):
                m = mask[i]
                vals = d["features_poststate_v1"][i, m, j]
                if len(vals) > 1:
                    group_vars.append(float(vals.var()))
                uniq_per_group.append(int(len(np.unique(vals))))
            group_var_mean = float(np.mean(group_vars)) if group_vars else 0.0
            corr_return = float(np.corrcoef(col, returns_valid)[0, 1]) if col.std() > 1e-8 else 0.0
            corr_oracle = float(np.corrcoef(col, oracle_valid.astype(float))[0, 1]) if col.std() > 1e-8 else 0.0
            corr_regret = float(np.corrcoef(col, regret_valid)[0, 1]) if col.std() > 1e-8 else 0.0
            stats.append({
                "name": name,
                "mean": float(col.mean()),
                "std": float(col.std()),
                "min": float(col.min()),
                "max": float(col.max()),
                "constant_rate": const_rate,
                "group_variance_mean": group_var_mean,
                "unique_values": int(len(np.unique(col))),
                "pearson_return": corr_return,
                "pearson_oracle": corr_oracle,
                "pearson_regret": corr_regret,
            })
        out[split] = stats
    return {"per_split_stats": out, "feature_names": ps_names, "v1_names": v1_names}


def _cross_feature_correlations() -> Dict[str, Any]:
    meta = json.loads((DATASET_DIR / "metadata.json").read_text(encoding="utf-8"))
    ps_names = meta["feature_names_poststate_v1"]
    d = np.load(str(DATASET_DIR / "train.npz"), allow_pickle=True)
    mask = d["mask"].astype(bool)
    ps = d["features_poststate_v1"][mask]
    corr = np.corrcoef(ps, rowvar=False)
    # condition number on standardized matrix
    X = (ps - ps.mean(axis=0)) / (ps.std(axis=0) + 1e-8)
    cond = float(np.linalg.cond(X))
    # find near-duplicates
    duplicates = []
    for i in range(len(ps_names)):
        for j in range(i + 1, len(ps_names)):
            if abs(corr[i, j]) > 0.99:
                duplicates.append({"f1": ps_names[i], "f2": ps_names[j], "corr": float(corr[i, j])})
    # new vs old max correlation
    new_vs_old = {}
    for j_new, name_new in enumerate(ps_names[25:], start=25):
        max_corr = 0.0
        max_old = ""
        for j_old, name_old in enumerate(ps_names[:25]):
            c = abs(corr[j_new, j_old])
            if c > max_corr:
                max_corr = c
                max_old = name_old
        new_vs_old[name_new] = {"max_old_feature": max_old, "abs_corr": float(max_corr)}
    return {
        "condition_number": cond,
        "duplicate_pairs": duplicates,
        "new_vs_old_max_corr": new_vs_old,
    }


def _analytical_vs_true_check(n_states: int = 10, samples_per_state: int = 3, seed: int = 4000) -> Dict[str, Any]:
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
    total = 200
    requests = generate_requests(
        env, rng, src, total,
        0.0625, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2, 3, "default3",
    )
    env.reset(requests)
    decision_states = []
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
        if len(legal) >= 2 and request_index >= 50:
            decision_states.append((copy.deepcopy(env), req, obs_c, obs_r, r_features, legal, split_id, server_id))
        if len(decision_states) >= n_states:
            break
        # advance real env with PPO-R action
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), r_action)

    comparisons = []
    for env_state, req, obs_c, obs_r, r_features, legal, split_id, server_id in decision_states:
        sampled = random.sample(legal, min(samples_per_state, len(legal)))
        num_mods = len(obs_r["mod_names"])
        max_blocks = int(env_state.max_blocks)
        for action_idx in sampled:
            path_idx = action_idx // (num_mods * max_blocks)
            rem = action_idx % (num_mods * max_blocks)
            mod_idx = rem // max_blocks
            block_idx = rem % max_blocks
            ana = compute_optical_afterstate(env_state, path_idx, mod_idx, block_idx, obs_r)
            true, ok = _true_optical_afterstate(env_state, path_idx, mod_idx, block_idx, obs_r, split_id, server_id)
            if not ok:
                continue
            comparisons.append({"action_idx": int(action_idx), "analytical": ana, "true": true})

    # Only compare optical afterstate keys present in both analytical and true dicts.
    compare_keys = [k for k in NEW_FEATURE_NAMES if k not in ("server_util_context", "server_margin_context", "holding_time_norm")]
    errors = {name: [] for name in compare_keys}
    rel_errors = {name: [] for name in compare_keys}
    for comp in comparisons:
        for name in compare_keys:
            if name in ("delta_frag", "free_block_count_delta"):
                # Deltas require before-state; skip direct true comparison in this simplified check.
                continue
            a = comp["analytical"][name]
            t = comp["true"].get(name, np.nan)
            if np.isfinite(a) and np.isfinite(t):
                errors[name].append(abs(a - t))
                denom = max(abs(t), 1e-8)
                rel_errors[name].append(abs(a - t) / denom)

    summary = {}
    for name in compare_keys:
        if not errors[name]:
            continue
        summary[name] = {
            "abs_error_mean": float(np.mean(errors[name])),
            "abs_error_max": float(np.max(errors[name])),
            "rel_error_mean": float(np.mean(rel_errors[name])),
            "rel_error_max": float(np.max(rel_errors[name])),
            "samples": len(errors[name]),
        }
    return {"n_comparisons": len(comparisons), "feature_error_summary": summary}


def _write_md(stats: Dict[str, Any], corr: Dict[str, Any], correctness: Dict[str, Any], out: Path):
    lines = ["# Feature Correctness Audit", ""]

    lines += ["## Condition number and duplicates", "", f"- Condition number (standardized 41-d train matrix): **{corr['condition_number']:.2e}**", ""]
    if corr["duplicate_pairs"]:
        lines += ["### Near-duplicate feature pairs (|r| > 0.99)", "", "| Feature 1 | Feature 2 | Correlation |", "|---|---|---:|"]
        for p in corr["duplicate_pairs"]:
            lines.append(f"| {p['f1']} | {p['f2']} | {p['corr']:.4f} |")
    else:
        lines += ["No feature pairs with |r| > 0.99.", ""]

    lines += ["", "## New 16-d vs old 25-d max correlation", "", "| New feature | Most correlated old feature | |r| |", "|---|---|---:|"]
    for name, info in corr["new_vs_old_max_corr"].items():
        lines.append(f"| {name} | {info['max_old_feature']} | {info['abs_corr']:.3f} |")
    lines.append("")

    lines += ["## Per-feature statistics (train split)", "", "| Feature | Mean | Std | Min | Max | Constant rate | Group var mean | Unique vals | Pearson(return) | Pearson(oracle) | Pearson(regret) |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for s in stats["per_split_stats"]["train"]:
        lines.append(
            f"| {s['name']} | {s['mean']:.3f} | {s['std']:.3f} | {s['min']:.3f} | {s['max']:.3f} | "
            f"{s['constant_rate']:.2%} | {s['group_variance_mean']:.4f} | {s['unique_values']} | "
            f"{s['pearson_return']:.3f} | {s['pearson_oracle']:.3f} | {s['pearson_regret']:.3f} |"
        )
    lines.append("")

    lines += ["## Analytical vs true afterstate error", "", f"Comparisons: {correctness['n_comparisons']}", "", "| Feature | Mean abs error | Max abs error | Mean rel error | Max rel error |", "|---|---:|---:|---:|---:|"]
    for name, e in correctness["feature_error_summary"].items():
        lines.append(
            f"| {name} | {e['abs_error_mean']:.4f} | {e['abs_error_max']:.4f} | "
            f"{e['rel_error_mean']:.4f} | {e['rel_error_max']:.4f} |"
        )
    lines.append("")

    out.write_text("\n".join(lines), encoding="utf-8")


def main():
    print("[part3] Computing dataset feature stats...")
    stats = _dataset_feature_stats()
    print("[part3] Computing correlations...")
    corr = _cross_feature_correlations()
    print("[part3] Running analytical vs true afterstate check (this may take a few minutes)...")
    correctness = _analytical_vs_true_check()

    out_json = {
        "feature_stats": stats,
        "correlations": corr,
        "correctness_check": correctness,
    }
    (OUTPUT_DIR / "FEATURE_CORRECTNESS_AUDIT.json").write_text(json.dumps(out_json, indent=2, default=str), encoding="utf-8")
    _write_md(stats, corr, correctness, OUTPUT_DIR / "FEATURE_CORRECTNESS_AUDIT.md")
    print("[part3] Report written to", OUTPUT_DIR / "FEATURE_CORRECTNESS_AUDIT.md")


if __name__ == "__main__":
    main()
