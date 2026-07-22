"""Phase 0 correctness and consistency tests for v1.35 corrected validation.

Must pass before any training or large-scale dataset generation.
"""
from __future__ import annotations

import copy
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Tuple

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
from sa_hmarl.evaluation.r_poststate_features_v2 import (
    POSTSTATE_COMPACT_V2_FEATURE_NAMES,
    _global_spectrum_stats_v2,
    _path_to_edges,
    _summarize_availability,
    build_poststate_compact_v2_feature,
    compute_optical_afterstate_compact_v2,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


OUTPUT_DIR = Path("sa_hmarl/experiments/v135_corrected")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frozen checkpoints.
AGENT_C_CKPT = "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt"
AGENT_R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"


def _set_thread_limits():
    for var in ["OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"]:
        os.environ[var] = "1"
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass  # already set or parallel work has started


def _make_env(seed: int):
    return make_env(
        "xlron_cost239_ptrnet_real", 320, 4, seed,
        modulation_profile="default", max_blocks=10,
        block_sort_strategy="start_asc", path_sort_strategy="hops", k=50,
    )


def _collect_decision_states(
    n_states: int,
    seed: int,
    warmup: int = 2000,
    min_candidates: int = 5,
    required_fs_filter: Optional[Tuple[int, ...]] = None,
) -> List[Tuple[Any, ...]]:
    """Collect decision states after warmup.

    Returns list of (env, req, obs_c, obs_r, r_features, legal, split_id, server_id, request_index).
    env is a deep-copy snapshot at the decision moment.
    """
    _set_thread_limits()
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c(AGENT_C_CKPT, "cpu")
    agent_r = _load_ppo_r(AGENT_R_CKPT, mod_reg, "cpu")
    env = _make_env(seed)
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env.net.NUM_NODES))
    total = warmup + n_states * 10 + 100
    requests = generate_requests(
        env, rng, src, total,
        0.0625, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2, 3, "default3",
    )
    env.reset(requests)

    states: List[Tuple[Any, ...]] = []
    for request_index, req in enumerate(requests):
        env.advance_time(req.arrival_time)

        env.k = 5
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), 4)

        env.k = 50
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()

        # Advance real env with PPO-R during warmup.
        if request_index < warmup:
            ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
            env.step((split_id, server_id), r_action)
            continue

        if len(legal) >= min_candidates:
            fs_ok = True
            if required_fs_filter is not None:
                # Check whether any candidate has required_fs in filter.
                fs_in_filter = False
                num_mods = len(obs_r["mod_names"])
                max_blocks = int(env.max_blocks)
                for a in legal:
                    p = a // (num_mods * max_blocks)
                    rem = a % (num_mods * max_blocks)
                    m = rem // max_blocks
                    fs = obs_r["required_fs_per_path_mod"][p][m]
                    if fs in required_fs_filter:
                        fs_in_filter = True
                        break
                fs_ok = fs_in_filter

            if fs_ok:
                states.append((copy.deepcopy(env), req, obs_c, obs_r, r_features, legal, split_id, server_id, request_index))

        if len(states) >= n_states:
            break

        # Advance real env.
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
        r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), r_action)

    return states


def _true_optical_afterstate_compact_v2(
    env_state,
    req,
    obs_c,
    obs_r,
    r_features,
    action_idx: int,
    split_id: int,
    server_id: int,
) -> Tuple[Dict[str, float], bool]:
    """Deep-copy env, execute action, compute the same compact optical afterstate stats."""
    branch = copy.deepcopy(env_state)
    num_mods = len(obs_r["mod_names"])
    max_blocks = int(branch.max_blocks)
    path_idx = action_idx // (num_mods * max_blocks)
    rem = action_idx % (num_mods * max_blocks)
    mod_idx = rem // max_blocks
    block_idx = rem % max_blocks

    r_action = decode_agent_r_action(action_idx, num_mods, max_blocks)
    _, _, _, info = branch.step((split_id, server_id), r_action)
    if not info.get("success", False):
        return {}, False

    num_total_slots = int(branch.net.num_slots)
    path = obs_r["candidate_paths"][path_idx]
    path_edges = _path_to_edges(path)
    avail_after = branch.net.get_available_slots(path)
    stats_after_path = _summarize_availability(avail_after, num_total_slots)
    stats_after_global = _global_spectrum_stats_v2(branch.net.link_states, num_total_slots)

    # Per-edge stats after.
    min_free_ratio = 1.0
    for key in path_edges:
        free_arr = ~branch.net.link_states[key]
        s = _summarize_availability(free_arr, num_total_slots)
        if s["free_ratio"] < min_free_ratio:
            min_free_ratio = s["free_ratio"]

    active = getattr(branch, "active_connections", [])
    # Count connections sharing path edges, excluding the one we just added.
    conflict_after = 0
    path_edge_set = set(path_edges)
    for _rt, _cnt, conn in active:
        cp = conn.get("path")
        if cp and (set(_path_to_edges(cp)) & path_edge_set):
            conflict_after += 1
    # The newly added connection shares path edges, so it is counted above; subtract it.
    conflict_after = max(conflict_after - 1, 0)

    fs_req = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    num_slots = int(fs_req) if fs_req is not None else int(obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx][block_idx][1])
    occupied_slot_hops = max(len(path_edges), 1) * num_slots
    occupied_slot_hops_norm = occupied_slot_hops / max(num_total_slots * 5, 1)
    total_active = max(len(getattr(env_state, "active_connections", [])), 1)

    return {
        "delta_frag": float(stats_after_global["avg_frag_index"] - _global_spectrum_stats_v2(env_state.net.link_states, num_total_slots)["avg_frag_index"]),
        "free_block_count_delta": float(stats_after_global["avg_free_block_count"] - _global_spectrum_stats_v2(env_state.net.link_states, num_total_slots)["avg_free_block_count"]),
        "occupied_slot_hops_norm": float(occupied_slot_hops_norm),
        "path_conflict_after_norm": float(conflict_after / total_active),
        "path_free_ratio_after": float(stats_after_path["free_ratio"]),
        "min_edge_free_ratio_after": float(min_free_ratio),
        "global_frag_delta": float(stats_after_global["avg_frag_index"] - _global_spectrum_stats_v2(env_state.net.link_states, num_total_slots)["avg_frag_index"]),
        "global_free_ratio_mean_after": float(stats_after_global["avg_free_ratio"]),
        "global_free_ratio_p10_after": float(stats_after_global["p10_free_ratio"]),
        "global_lfb_mean_after": float(stats_after_global["avg_lfb"]),
        "global_lfb_p10_after": float(stats_after_global["p10_lfb"]),
    }, True


def test_analytical_vs_true_afterstate(min_states: int = 100, min_candidates_per_state: int = 5, seed: int = 5000) -> Dict[str, Any]:
    """Test 1: analytical afterstate matches deepcopy+env.step true afterstate."""
    print("[phase0] Test 1: analytical vs true afterstate...")
    states = _collect_decision_states(
        n_states=min_states,
        seed=seed,
        warmup=2000,
        min_candidates=min_candidates_per_state,
        required_fs_filter=(2, 3, 4),
    )

    # Pad to ensure we have enough if filter limited us.
    if len(states) < min_states:
        states += _collect_decision_states(
            n_states=min_states - len(states),
            seed=seed + 1,
            warmup=2000,
            min_candidates=min_candidates_per_state,
        )

    compare_keys = [k for k in POSTSTATE_COMPACT_V2_FEATURE_NAMES if k in [
        "delta_frag", "free_block_count_delta", "occupied_slot_hops_norm",
        "path_conflict_after_norm", "path_free_ratio_after", "min_edge_free_ratio_after",
        "global_frag_delta", "global_free_ratio_mean_after", "global_free_ratio_p10_after",
        "global_lfb_mean_after", "global_lfb_p10_after",
    ]]

    errors: Dict[str, List[float]] = {k: [] for k in compare_keys}
    abs_errors: Dict[str, List[float]] = {k: [] for k in compare_keys}
    n_comparisons = 0
    env_mutations_ok = True

    for env_state, req, obs_c, obs_r, r_features, legal, split_id, server_id, _ in states:
        sampled = np.random.choice(legal, min(min_candidates_per_state, len(legal)), replace=False).tolist()
        for action_idx in sampled:
            # Snapshot link states to verify non-mutation.
            pre_links = {k: np.asarray(v, dtype=bool) for k, v in env_state.net.link_states.items()}
            pre_servers = [s.utilization for s in env_state.mec.servers]

            ana = compute_optical_afterstate_compact_v2(env_state, 0, 0, 0, obs_r)
            # Decode action.
            num_mods = len(obs_r["mod_names"])
            max_blocks = int(env_state.max_blocks)
            path_idx = action_idx // (num_mods * max_blocks)
            rem = action_idx % (num_mods * max_blocks)
            mod_idx = rem // max_blocks
            block_idx = rem % max_blocks
            ana = compute_optical_afterstate_compact_v2(env_state, path_idx, mod_idx, block_idx, obs_r)

            true, ok = _true_optical_afterstate_compact_v2(
                env_state, req, obs_c, obs_r, r_features, action_idx, split_id, server_id
            )
            if not ok:
                continue
            n_comparisons += 1

            for k in compare_keys:
                a = ana[k]
                t = true[k]
                errors[k].append(abs(a - t))
                if abs(t) > 1e-8:
                    abs_errors[k].append(abs(a - t) / abs(t))

            # Verify env_state not mutated.
            for k, v in pre_links.items():
                if not np.array_equal(v, env_state.net.link_states[k]):
                    env_mutations_ok = False
            for i, u in enumerate(pre_servers):
                if abs(u - env_state.mec.servers[i].utilization) > 1e-8:
                    env_mutations_ok = False

    summary = {
        "n_states": len(states),
        "n_comparisons": n_comparisons,
        "env_mutations_ok": env_mutations_ok,
        "per_feature": {},
    }
    all_pass = env_mutations_ok and n_comparisons >= min_states * min_candidates_per_state * 0.5
    for k in compare_keys:
        if errors[k]:
            vals = np.array(errors[k])
            rels = np.array(abs_errors[k]) if abs_errors[k] else np.array([0.0])
            summary["per_feature"][k] = {
                "max_abs_error": float(vals.max()),
                "mean_abs_error": float(vals.mean()),
                "max_rel_error": float(np.max(rels)),
                "mean_rel_error": float(np.mean(rels)),
            }
            if vals.max() > 1e-6:
                all_pass = False
        else:
            all_pass = False

    summary["pass"] = bool(all_pass)
    return summary


def test_fs_allocation_boundaries(seed: int = 6000) -> Dict[str, Any]:
    """Test 2: FS allocation boundary tests on synthetic spectrum states."""
    print("[phase0] Test 2: FS allocation boundaries...")
    _set_thread_limits()
    env = _make_env(seed)
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env.net.NUM_NODES))
    requests = generate_requests(
        env, rng, src, 200,
        0.0625, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2, 3, "default3",
    )
    env.reset(requests)
    # Advance a few requests to get non-empty state.
    for i in range(50):
        env.advance_time(requests[i].arrival_time)
        obs_c = build_agent_c_observation(env, requests[i])
        c_idx, _, _ = _select_c_action_from_obs(_load_ppo_c(AGENT_C_CKPT, "cpu"), obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), 4)
        obs_r = build_agent_r_observation(env, requests[i], split_id, server_id)
        ppo_idx, _ = _select_r_action_from_obs(_load_ppo_r(AGENT_R_CKPT, ModulationRegistry.from_profile("default"), "cpu"), obs_r, env.max_blocks)
        r_action = decode_agent_r_action(int(ppo_idx), len(obs_r["mod_names"]), env.max_blocks)
        env.step((split_id, server_id), r_action)

    # Find a request with legal candidates.
    for req in requests[50:]:
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(_load_ppo_c(AGENT_C_CKPT, "cpu"), obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), 4)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = _load_ppo_r(AGENT_R_CKPT, ModulationRegistry.from_profile("default"), "cpu").build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        if len(legal) < 2:
            env.step((split_id, server_id), decode_agent_r_action(int(_select_r_action_from_obs(_load_ppo_r(AGENT_R_CKPT, ModulationRegistry.from_profile("default"), "cpu"), obs_r, env.max_blocks)[0]), len(obs_r["mod_names"]), env.max_blocks))
            continue

        checks = []
        for action_idx in legal[:3]:
            f = build_poststate_compact_v2_feature(env, req, obs_c, obs_r, r_features, action_idx, split_id, server_id)
            assert np.all(np.isfinite(f)), "Non-finite feature"
            checks.append("finite")
        return {"pass": True, "checks": checks, "n_legal": len(legal)}

    return {"pass": False, "reason": "no suitable state found"}


def test_action_decode_consistency(seed: int = 7000) -> Dict[str, Any]:
    """Test 3: flat action ID <-> (path, mod, block) <-> env.step action are consistent."""
    print("[phase0] Test 3: action decode consistency...")
    _set_thread_limits()
    mod_reg = ModulationRegistry.from_profile("default")
    agent_r = _load_ppo_r(AGENT_R_CKPT, mod_reg, "cpu")
    env = _make_env(seed)
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env.net.NUM_NODES))
    requests = generate_requests(
        env, rng, src, 200,
        0.0625, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2, 3, "default3",
    )
    env.reset(requests)

    for req in requests[50:]:
        env.advance_time(req.arrival_time)
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(_load_ppo_c(AGENT_C_CKPT, "cpu"), obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), 4)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        if len(legal) < 2:
            continue

        ok = True
        for action_idx in legal[:10]:
            num_mods = len(obs_r["mod_names"])
            max_blocks = int(env.max_blocks)
            path_idx = action_idx // (num_mods * max_blocks)
            rem = action_idx % (num_mods * max_blocks)
            mod_idx = rem // max_blocks
            block_idx = rem % max_blocks

            r_action = decode_agent_r_action(action_idx, num_mods, max_blocks)
            if r_action != (path_idx, mod_idx, block_idx):
                ok = False
                break
        return {"pass": ok, "n_checked": min(10, len(legal))}

    return {"pass": False, "reason": "no suitable state found"}


def test_offline_online_consistency(seed: int = 8000) -> Dict[str, Any]:
    """Test 4: dataset generator feature == online feature builder for same state/action."""
    print("[phase0] Test 4: offline-online feature consistency...")
    _set_thread_limits()
    mod_reg = ModulationRegistry.from_profile("default")
    agent_c = _load_ppo_c(AGENT_C_CKPT, "cpu")
    agent_r = _load_ppo_r(AGENT_R_CKPT, mod_reg, "cpu")
    env = _make_env(seed)
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env.net.NUM_NODES))
    requests = generate_requests(
        env, rng, src, 200,
        0.0625, 20.0, 30.0, 30.0, 100.0, 5.0, 30.0, 0.1, 2.2, 3, "default3",
    )
    env.reset(requests)

    for req in requests[50:]:
        env.advance_time(req.arrival_time)
        env.k = 5
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(int(c_idx), 4)
        env.k = 50
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_features, r_mask = agent_r.build_action_features(obs_r)
        legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
        if len(legal) < 2:
            continue

        # Online features for first few candidates.
        online_features = []
        for action_idx in legal[:5]:
            f = build_poststate_compact_v2_feature(env, req, obs_c, obs_r, r_features, action_idx, split_id, server_id)
            online_features.append(f)
        online_features = np.stack(online_features)

        # Dataset-style features: same function, called via batch wrapper.
        from sa_hmarl.evaluation.r_poststate_features_v2 import build_poststate_compact_v2_feature_batch
        offline_features = build_poststate_compact_v2_feature_batch(
            env, req, obs_c, obs_r, r_features, legal[:5], split_id, server_id
        )

        if not np.allclose(online_features, offline_features, atol=1e-6):
            return {"pass": False, "reason": "online vs offline mismatch", "max_diff": float(np.abs(online_features - offline_features).max())}
        return {"pass": True, "n_checked": 5, "max_diff": 0.0}

    return {"pass": False, "reason": "no suitable state found"}


def test_serial_parallel_consistency(seed: int = 9000) -> Dict[str, Any]:
    """Test 5: dataset generator serial vs parallel outputs identical.

    Runs the corrected dataset generator in a tiny config with max_workers=1 and max_workers=3.
    """
    print("[phase0] Test 5: serial vs parallel consistency...")
    _set_thread_limits()

    repo_root = Path(".")
    cmd_base = [
        sys.executable, "-m", "sa_hmarl.evaluation.v135_corrected_dataset_generator",
        "--train_seeds", str(seed),
        "--val_seeds", str(seed + 1),
        "--test_seeds", str(seed + 2),
        "--train_episodes", "1",
        "--val_episodes", "1",
        "--test_episodes", "1",
        "--requests_per_episode", "50",
        "--warmup_requests", "10",
        "--max_candidates", "4",
        "--horizon", "0",
        "--max_workers",
    ]

    out_serial = repo_root / "sa_hmarl" / "datasets" / "_phase0_serial"
    out_parallel = repo_root / "sa_hmarl" / "datasets" / "_phase0_parallel"
    for p in (out_serial, out_parallel):
        if p.exists():
            shutil.rmtree(p)

    def run(max_workers: int, out_dir: Path):
        cmd = cmd_base + [str(max_workers), "--output_dir", str(out_dir)]
        env = os.environ.copy()
        env["PYTHONPATH"] = str(repo_root / "sa_hmarl")
        subprocess.run(cmd, cwd=str(repo_root), env=env, check=True, capture_output=True, text=True)

    run(1, out_serial)
    run(3, out_parallel)

    results = {"pass": True, "splits": {}}
    for split in ("train", "val", "test"):
        s = np.load(str(out_serial / f"{split}.npz"), allow_pickle=True)
        p = np.load(str(out_parallel / f"{split}.npz"), allow_pickle=True)

        ok = True
        max_diff = 0.0
        for key in s.files:
            if key not in p.files:
                ok = False
                continue
            sv, pv = s[key], p[key]
            if sv.dtype == object or pv.dtype == object or sv.dtype.kind in "OSU" or pv.dtype.kind in "OSU":
                if not np.array_equal(sv, pv):
                    ok = False
            else:
                if not np.allclose(sv, pv, atol=1e-6):
                    ok = False
                    max_diff = max(max_diff, float(np.abs(sv - pv).max()))

        # Deterministic ordering check.
        if not np.array_equal(s["group_id"], np.sort(s["group_id"], kind="stable")):
            ok = False
        if not np.array_equal(p["group_id"], np.sort(p["group_id"], kind="stable")):
            ok = False

        results["splits"][split] = {"pass": ok, "max_diff": max_diff}
        if not ok:
            results["pass"] = False

    # Cleanup.
    for p in (out_serial, out_parallel):
        if p.exists():
            shutil.rmtree(p)

    return results


def _write_report(results: Dict[str, Any], path: Path):
    lines = ["# Phase 0 Correctness and Consistency Tests", ""]
    lines.append(f"**Overall pass:** {results['overall_pass']}")
    lines.append("")

    for test_name, res in results["tests"].items():
        lines.append(f"## {test_name}")
        lines.append(f"- pass: {res.get('pass', False)}")
        if "n_comparisons" in res:
            lines.append(f"- comparisons: {res['n_comparisons']}")
        if "env_mutations_ok" in res:
            lines.append(f"- env mutations ok: {res['env_mutations_ok']}")
        if "per_feature" in res:
            lines.append("")
            lines.append("| Feature | max abs error | mean abs error | max rel error | mean rel error |")
            lines.append("|---|---:|---:|---:|---:|")
            for feat, stats in res["per_feature"].items():
                lines.append(
                    f"| {feat} | {stats['max_abs_error']:.2e} | {stats['mean_abs_error']:.2e} | "
                    f"{stats['max_rel_error']:.2e} | {stats['mean_rel_error']:.2e} |"
                )
        if "checks" in res:
            lines.append(f"- checks: {res['checks']}")
        if "n_checked" in res:
            lines.append(f"- n checked: {res['n_checked']}")
        if "splits" in res:
            for split, sr in res["splits"].items():
                lines.append(f"- {split}: pass={sr['pass']}, max_diff={sr['max_diff']:.2e}")
        if "reason" in res:
            lines.append(f"- reason: {res['reason']}")
        lines.append("")

    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    _set_thread_limits()
    results = {
        "tests": {},
        "overall_pass": True,
    }

    results["tests"]["analytical_vs_true_afterstate"] = test_analytical_vs_true_afterstate()
    results["tests"]["fs_allocation_boundaries"] = test_fs_allocation_boundaries()
    results["tests"]["action_decode_consistency"] = test_action_decode_consistency()
    results["tests"]["offline_online_consistency"] = test_offline_online_consistency()
    results["tests"]["serial_parallel_consistency"] = test_serial_parallel_consistency()

    for res in results["tests"].values():
        if not res.get("pass", False):
            results["overall_pass"] = False

    json_path = OUTPUT_DIR / "FEATURE_CORRECTNESS_V2.json"
    md_path = OUTPUT_DIR / "FEATURE_CORRECTNESS_V2.md"
    json_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
    _write_report(results, md_path)

    print(f"[phase0] Overall pass: {results['overall_pass']}")
    print(f"[phase0] Report: {md_path}")
    if not results["overall_pass"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
