"""COST239 compute-vs-spectrum bottleneck quick diagnostic (2×2).

Resource-condition intervention experiment.  Does not train models or overwrite
historical checkpoints.  Compares PPO-R, KSP-FF heuristics, and the historical
v1.3 ranker under four capacity/spectrum combinations.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
import os
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action, ksp_ff_highest_mod_action
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
from sa_hmarl.evaluation.r_ranker_features import build_r_ranker_feature_batch
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


# ---------------------------------------------------------------------------
# Fixed protocol
# ---------------------------------------------------------------------------
TOPOLOGY = "xlron_cost239_ptrnet_real"
NUM_SERVERS = 4
K_C = 5
K_R = 50
PATH_SORT = "hops"
BLOCK_SORT = "start_asc"
EDGE_COST_MIN = 0.1
EDGE_COST_MAX = 2.2
ARRIVAL_INTERVAL = 0.0625
HOLDING_MIN = 20.0
HOLDING_MAX = 30.0
DEADLINE_MIN = 30.0
DEADLINE_MAX = 100.0
SIZE_MIN_MB = 5.0
SIZE_MAX_MB = 30.0
NUM_SPLITS = 3
SPLIT_PROFILE = "default3"
MAX_BLOCKS = 10
MODULATION_PROFILE = "default"

CHECKPOINT_C = Path("sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
CHECKPOINT_R = Path("sa_hmarl/checkpoints/agent_r_mixed.pt")
CHECKPOINT_V13 = Path(
    "sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt"
)


# ---------------------------------------------------------------------------
# Worker globals
# ---------------------------------------------------------------------------
_worker_agent_c: Optional[PPOAgentC] = None
_worker_agent_r: Optional[PPOAgentR] = None
_worker_v13: Optional[Dict[str, Any]] = None
_worker_device: str = "cpu"
_worker_log_dir: Optional[Path] = None


def _set_worker_threads() -> None:
    """Limit BLAS/Torch threading inside each worker."""
    for var in (
        "OMP_NUM_THREADS",
        "MKL_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "NUMEXPR_NUM_THREADS",
    ):
        os.environ[var] = "1"
    try:
        torch.set_num_threads(1)
    except Exception:
        pass


def _init_worker(device: str, log_dir: str) -> None:
    _set_worker_threads()
    global _worker_agent_c, _worker_agent_r, _worker_v13, _worker_device, _worker_log_dir
    _worker_device = device
    _worker_log_dir = Path(log_dir)
    mod_reg = ModulationRegistry.from_profile(MODULATION_PROFILE)
    _worker_agent_c = _load_ppo_c(str(CHECKPOINT_C), device)
    _worker_agent_r = _load_ppo_r(str(CHECKPOINT_R), mod_reg, device)
    _worker_v13 = _load_v13_ranker(str(CHECKPOINT_V13), device)


# ---------------------------------------------------------------------------
# Checkpoint loading
# ---------------------------------------------------------------------------
def _load_v13_ranker(ckpt_path: str, device: str = "cpu") -> Dict[str, Any]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"),
        int(ckpt["input_dim"]),
        tuple(ckpt["hidden_dims"]),
        float(ckpt.get("dropout", 0.0)),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad = False
    return {
        "model": model,
        "mean": np.asarray(ckpt["feature_mean"], dtype=np.float32),
        "std": np.asarray(ckpt["feature_std"], dtype=np.float32),
        "feature_names": list(ckpt.get("feature_names", [])),
        "candidate_mode": ckpt.get("candidate_mode", "ppo_r_topk_only"),
        "max_candidates": int(ckpt.get("max_candidates", 30)),
        "ppo_top_k": int(ckpt.get("ppo_top_k", 50)),
        "num_random_candidates": int(ckpt.get("num_random_candidates", 5)),
        "min_candidates": int(ckpt.get("min_candidates", 15)),
        "candidate_seed": int(ckpt.get("candidate_seed", 12345)),
    }


# ---------------------------------------------------------------------------
# Trace generation and hashing
# ---------------------------------------------------------------------------
def _requests_to_bytes(requests: List[Any]) -> bytes:
    """Deterministic serialization for trace hashing."""
    rows = []
    for req in requests:
        splits = []
        for sp in req.splits:
            splits.append(
                (
                    int(sp.split_id),
                    float(sp.intermediate_size_mb),
                    float(sp.local_compute_cost),
                    float(sp.edge_compute_cost),
                )
            )
        rows.append(
            (
                int(req.req_id),
                int(req.src_node),
                float(req.arrival_time),
                float(req.holding_time),
                float(req.deadline_ms),
                tuple(splits),
            )
        )
    return json.dumps(tuple(rows), ensure_ascii=True, sort_keys=True).encode("utf-8")


def _trace_hash(requests: List[Any]) -> str:
    return hashlib.sha256(_requests_to_bytes(requests)).hexdigest()[:32]


def _generate_trace(
    env_proto: Any,
    seed: int,
    n_requests: int,
) -> Tuple[List[Any], str]:
    rng = np.random.RandomState(seed)
    src = int(rng.randint(0, env_proto.net.NUM_NODES))
    requests = generate_requests(
        env_proto,
        rng,
        src,
        n_requests,
        arrival_interval=ARRIVAL_INTERVAL,
        holding_min=HOLDING_MIN,
        holding_max=HOLDING_MAX,
        deadline_min=DEADLINE_MIN,
        deadline_max=DEADLINE_MAX,
        size_min_mb=SIZE_MIN_MB,
        size_max_mb=SIZE_MAX_MB,
        edge_cost_min=EDGE_COST_MIN,
        edge_cost_max=EDGE_COST_MAX,
        num_splits=NUM_SPLITS,
        split_profile=SPLIT_PROFILE,
    )
    return requests, _trace_hash(requests)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _ppo_logits(agent_r: PPOAgentR, obs_r: Dict[str, Any]) -> np.ndarray:
    features, mask = agent_r.build_action_features(obs_r)
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    return logits


def _decode_action(action_idx: int, obs_r: Dict[str, Any]) -> Tuple[int, int, int]:
    num_mods = len(obs_r["mod_names"])
    max_blocks = int(obs_r["agent_r_mask"].shape[0]) // (
        len(obs_r["candidate_paths"]) * num_mods
    )
    return decode_agent_r_action(int(action_idx), num_mods, max_blocks)


def _action_attributes(
    action_idx: int, obs_r: Dict[str, Any], mod_reg: ModulationRegistry
) -> Dict[str, Any]:
    path_idx, mod_idx, block_idx = _decode_action(action_idx, obs_r)
    path_feat = obs_r["path_features"][path_idx]
    mod_name = obs_r["mod_names"][mod_idx]
    mod = mod_reg.by_name(mod_name)
    se = float(mod.spectral_efficiency) if mod else float("nan")
    req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
    blocks = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
    block_start = int(blocks[block_idx][0]) if block_idx < len(blocks) else None
    return {
        "path_idx": path_idx,
        "mod_idx": mod_idx,
        "block_idx": block_idx,
        "path_length_km": float(path_feat["path_length_km"]),
        "hop_count": int(path_feat["hop_count"]),
        "modulation": mod_name,
        "spectral_efficiency": se,
        "required_fs": int(req_fs) if req_fs is not None else None,
        "block_start": block_start,
    }


# ---------------------------------------------------------------------------
# Per-request decision loop
# ---------------------------------------------------------------------------
@dataclass
class _PerRequestMetrics:
    blocked: int = 0
    server_overload: int = 0
    no_suitable_block: int = 0
    allocation_failed: int = 0
    other_block: int = 0
    delay_ms: List[float] = field(default_factory=list)
    decision_ms: List[float] = field(default_factory=list)
    required_fs: List[float] = field(default_factory=list)
    path_length_km: List[float] = field(default_factory=list)
    se: List[float] = field(default_factory=list)
    k50_pos_k5_neg: int = 0
    k50_pos: int = 0


def _run_method_on_trace(
    env: Any,
    requests: List[Any],
    method: str,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    v13: Optional[Dict[str, Any]],
    mod_reg: ModulationRegistry,
) -> Dict[str, Any]:
    """Run one method on a fresh env already reset to the trace."""
    m = _PerRequestMetrics()
    num_servers = len(env.mec.servers)

    for req in requests:
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        # K_C observation and action
        env.k = K_C
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_c = build_agent_c_observation(env, req)
        c_idx, raw_c_mask_k5, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)

        # K_R counterfactual C-mask for audit
        env.k = K_R
        obs_c50 = build_agent_c_observation(env, req)
        raw_c_mask_k50 = np.asarray(obs_c50["agent_c_mask"], dtype=bool)
        k5_mask = np.asarray(raw_c_mask_k5, dtype=bool)
        m.k50_pos += int(raw_c_mask_k50.sum())
        m.k50_pos_k5_neg += int((raw_c_mask_k50 & ~k5_mask).sum())

        # Restore K_C for downstream consistency
        env.k = K_C
        split_id, server_id = decode_agent_c_action(int(c_idx), num_servers)

        # K_R R observation
        env.k = K_R
        env.path_sort_strategy = PATH_SORT
        env.block_sort_strategy = BLOCK_SORT
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        legal = np.flatnonzero(r_mask).tolist()

        chosen_r_idx: Optional[int] = None
        if not legal:
            # No legal R action -> block
            chosen_r_idx = 0
        else:
            if method == "ppo_r":
                chosen_r_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)
            elif method == "ksp_ff_plain":
                chosen_r_idx = ksp_ff_action(obs_r)
                if chosen_r_idx is None:
                    chosen_r_idx = 0
            elif method == "ksp_ff_highmod":
                chosen_r_idx = ksp_ff_highest_mod_action(obs_r)
                if chosen_r_idx is None:
                    chosen_r_idx = 0
            elif method == "v1.3_ranker":
                chosen_r_idx = _v13_select_action(
                    env, req, obs_c, obs_r, agent_r, v13, split_id, server_id
                )
            else:
                raise ValueError(f"Unknown method: {method}")

        chosen_r_idx = int(chosen_r_idx)
        r_action = decode_agent_r_action(chosen_r_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)
        t1 = time.perf_counter()

        m.decision_ms.append((t1 - t0) * 1000.0)
        m.delay_ms.append(float(info.get("delay_ms", 0.0)))

        if not info.get("success", False):
            m.blocked += 1
            reason = info.get("reason", "")
            if reason == "server_overload":
                m.server_overload += 1
            elif reason == "no_suitable_block":
                m.no_suitable_block += 1
            elif reason == "allocation_failed":
                m.allocation_failed += 1
            else:
                m.other_block += 1
            continue

        # Record chosen action attributes only on success
        attrs = _action_attributes(chosen_r_idx, obs_r, mod_reg)
        if attrs["required_fs"] is not None:
            m.required_fs.append(float(attrs["required_fs"]))
        m.path_length_km.append(attrs["path_length_km"])
        m.se.append(attrs["spectral_efficiency"])

    total = max(len(requests), 1)
    blocked = max(m.blocked, 1)
    delays = np.asarray(m.delay_ms, dtype=np.float32)
    decs = np.asarray(m.decision_ms, dtype=np.float32)
    return {
        "total": len(requests),
        "blocking_rate": m.blocked / total,
        "server_overload_rate": m.server_overload / total,
        "no_suitable_block_rate": m.no_suitable_block / total,
        "allocation_failed_rate": m.allocation_failed / total,
        "other_block_rate": m.other_block / total,
        "optical_block_rate": (m.no_suitable_block + m.allocation_failed + m.other_block) / total,
        "overload_share_of_blocking": m.server_overload / blocked,
        "optical_share_of_blocking": (m.no_suitable_block + m.allocation_failed + m.other_block) / blocked,
        "mean_delay_ms": float(delays.mean()),
        "p95_delay_ms": float(np.percentile(delays, 95)),
        "mean_decision_time_ms": float(decs.mean()),
        "p95_decision_time_ms": float(np.percentile(decs, 95)),
        "mean_required_fs": float(np.mean(m.required_fs)) if m.required_fs else None,
        "mean_path_length_km": float(np.mean(m.path_length_km)) if m.path_length_km else None,
        "mean_spectral_efficiency": float(np.mean(m.se)) if m.se else None,
        "k50_pos": m.k50_pos,
        "k50_pos_k5_neg": m.k50_pos_k5_neg,
        "c_mask_fn_rate": float(m.k50_pos_k5_neg / max(m.k50_pos, 1)),
    }


def _v13_select_action(
    env: Any,
    req: Any,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r: PPOAgentR,
    v13: Dict[str, Any],
    split_id: int,
    server_id: int,
) -> int:
    """v1.3 ranker inference matching its checkpoint candidate protocol."""
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).tolist()
    if not legal:
        return 0
    if len(legal) == 1:
        return int(legal[0])

    # ppo_r_topk_only: top-K by PPO-R logits, truncated to max_candidates.
    logits = _ppo_logits(agent_r, obs_r)
    ppo_top_k = min(v13["ppo_top_k"], len(legal))
    top_local = np.argsort(-logits[legal], kind="stable")[:ppo_top_k]
    candidates = [int(legal[i]) for i in top_local]
    candidates = candidates[: v13["max_candidates"]]

    if not candidates:
        return int(legal[0])

    X = build_r_ranker_feature_batch(
        env, req, obs_c, obs_r, r_features, candidates, split_id, server_id,
        feature_names=v13["feature_names"],
    )
    mean = v13["mean"]
    std = v13["std"]
    Xn = (X - mean) / std
    device = next(v13["model"].parameters()).device
    with torch.no_grad():
        scores = v13["model"](
            torch.as_tensor(Xn, dtype=torch.float32, device=device)
        ).cpu().numpy().ravel()
    best_local = int(np.argmax(scores))
    return int(candidates[best_local])


# ---------------------------------------------------------------------------
# Environment factory
# ---------------------------------------------------------------------------
def _make_env_for_cell(
    num_slots: int,
    capacities: List[float],
    seed: int = 42,
) -> Any:
    return make_env(
        TOPOLOGY,
        num_slots=num_slots,
        num_servers=NUM_SERVERS,
        seed=seed,
        capacities=capacities,
        modulation_profile=MODULATION_PROFILE,
        max_blocks=MAX_BLOCKS,
        block_sort_strategy=BLOCK_SORT,
        path_sort_strategy=PATH_SORT,
        k=K_R,
    )


# ---------------------------------------------------------------------------
# Calibration task
# ---------------------------------------------------------------------------
def _calibration_task(
    cap_idx: int, slot_idx: int, capacity: int, num_slots: int, seed: int
) -> Dict[str, Any]:
    """Worker task: (capacity_idx, slots_idx, capacity, slots, seed)."""
    try:
        env = _make_env_for_cell(num_slots, [float(capacity)] * NUM_SERVERS)
        requests, trace_hash = _generate_trace(env, seed, 500)
        env.reset(requests)

        metrics = _run_method_on_trace(
            env,
            requests,
            "ppo_r",
            _worker_agent_c,
            _worker_agent_r,
            None,
            ModulationRegistry.from_profile(MODULATION_PROFILE),
        )
        return {
            "status": "ok",
            "cap_idx": cap_idx,
            "slot_idx": slot_idx,
            "capacity": capacity,
            "num_slots": num_slots,
            "seed": seed,
            "trace_hash": trace_hash,
            "metrics": metrics,
        }
    except Exception as exc:
        return {
            "status": "error",
            "cap_idx": cap_idx,
            "slot_idx": slot_idx,
            "capacity": capacity,
            "num_slots": num_slots,
            "seed": seed,
            "error": repr(exc),
        }


# ---------------------------------------------------------------------------
# 2×2 evaluation task
# ---------------------------------------------------------------------------
_METHODS = ["ppo_r", "ksp_ff_plain", "ksp_ff_highmod", "v1.3_ranker"]


def _eval_task(
    cell_label: str, num_slots: int, capacity_int: int, capacities: List[float], seed: int
) -> Dict[str, Any]:
    """Worker task: (cell_label, num_slots, capacity_value_as_int, capacities, seed)."""
    capacity = float(capacity_int)
    try:
        env_proto = _make_env_for_cell(num_slots, capacities)
        requests, trace_hash = _generate_trace(env_proto, seed, 1000)

        mod_reg = ModulationRegistry.from_profile(MODULATION_PROFILE)
        per_method: Dict[str, Any] = {}
        for method in _METHODS:
            env = _make_env_for_cell(num_slots, capacities)
            env.reset(requests)
            per_method[method] = _run_method_on_trace(
                env,
                requests,
                method,
                _worker_agent_c,
                _worker_agent_r,
                _worker_v13,
                mod_reg,
            )
            per_method[method]["trace_hash"] = trace_hash

        return {
            "status": "ok",
            "cell": cell_label,
            "num_slots": num_slots,
            "capacity": capacity,
            "seed": seed,
            "trace_hash": trace_hash,
            "methods": per_method,
        }
    except Exception as exc:
        return {
            "status": "error",
            "cell": cell_label,
            "num_slots": num_slots,
            "capacity": capacity,
            "seed": seed,
            "error": repr(exc),
        }


# ---------------------------------------------------------------------------
# Aggregation and cell selection
# ---------------------------------------------------------------------------
def _aggregate_seeds(
    records: List[Dict[str, Any]],
    key_path: List[str],
) -> Tuple[float, float]:
    vals = []
    for rec in records:
        obj = rec
        for k in key_path:
            obj = obj[k]
        vals.append(float(obj))
    arr = np.asarray(vals, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0))


def _calibration_table(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    table: Dict[Tuple[int, int], Dict[str, Any]] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        key = (r["capacity"], r["num_slots"])
        table.setdefault(key, {"capacity": r["capacity"], "num_slots": r["num_slots"], "seeds": []})
        table[key]["seeds"].append({"seed": r["seed"], "metrics": r["metrics"]})

    cells = []
    for key, cell in sorted(table.items()):
        seeds = cell["seeds"]
        blocking_mean, blocking_std = _aggregate_seeds(seeds, ["metrics", "blocking_rate"])
        overload_mean, _ = _aggregate_seeds(seeds, ["metrics", "overload_share_of_blocking"])
        optical_mean, _ = _aggregate_seeds(seeds, ["metrics", "optical_share_of_blocking"])
        cells.append(
            {
                "capacity": cell["capacity"],
                "num_slots": cell["num_slots"],
                "blocking_mean": blocking_mean,
                "blocking_std": blocking_std,
                "overload_share_mean": overload_mean,
                "optical_share_mean": optical_mean,
                "n_seeds": len(seeds),
                "seeds": seeds,
            }
        )
    return cells


def _select_2x2_cells(cells: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Select four cells following the protocol rules.

    Returns None if spectrum-tight optical gate cannot be satisfied.
    """
    max_slots = max(c["num_slots"] for c in cells)
    compute_rows_max_slots = [c for c in cells if c["num_slots"] == max_slots]
    if not compute_rows_max_slots:
        return None

    # compute-ample: highest capacity whose overload share is negligible at max slots.
    ample_candidates = [c for c in compute_rows_max_slots if c["overload_share_mean"] <= 0.05]
    if ample_candidates:
        compute_ample = max(ample_candidates, key=lambda c: c["capacity"])
    else:
        compute_ample = min(compute_rows_max_slots, key=lambda c: c["overload_share_mean"])

    # spectrum-ample: at compute-ample capacity, highest slots with near-zero optical blocking.
    spectrum_rows = [c for c in cells if c["capacity"] == compute_ample["capacity"]]
    spectrum_rows_sorted = sorted(spectrum_rows, key=lambda c: -c["num_slots"])
    spectrum_ample_candidates = [
        c for c in spectrum_rows_sorted
        if c["optical_share_mean"] <= 0.05 and c["blocking_mean"] <= 0.05
    ]
    spectrum_ample = spectrum_ample_candidates[0] if spectrum_ample_candidates else spectrum_rows_sorted[0]

    # compute-tight: at spectrum-ample slots, capacity with 5-20% blocking and >=80% overload.
    compute_tight_rows = [c for c in cells if c["num_slots"] == spectrum_ample["num_slots"]]
    compute_tight_candidates = [
        c for c in compute_tight_rows
        if 0.05 <= c["blocking_mean"] <= 0.20 and c["overload_share_mean"] >= 0.80
    ]
    if compute_tight_candidates:
        compute_tight = min(compute_tight_candidates, key=lambda c: abs(c["blocking_mean"] - 0.10))
    else:
        fallback = [c for c in compute_tight_rows if c["blocking_mean"] >= 0.05]
        if not fallback:
            return None
        compute_tight = max(fallback, key=lambda c: c["overload_share_mean"])

    # spectrum-tight: at compute-ample capacity, slots with 5-20% blocking and >=50% optical.
    spectrum_tight_candidates = [
        c for c in spectrum_rows
        if 0.05 <= c["blocking_mean"] <= 0.20 and c["optical_share_mean"] >= 0.50
    ]
    if not spectrum_tight_candidates:
        return None
    spectrum_tight = min(
        spectrum_tight_candidates, key=lambda c: abs(c["blocking_mean"] - 0.10)
    )

    return {
        "A": {"capacity": compute_ample["capacity"], "num_slots": spectrum_ample["num_slots"]},
        "B": {"capacity": compute_tight["capacity"], "num_slots": spectrum_ample["num_slots"]},
        "C": {"capacity": compute_ample["capacity"], "num_slots": spectrum_tight["num_slots"]},
        "D": {"capacity": compute_tight["capacity"], "num_slots": spectrum_tight["num_slots"]},
        "selection_notes": {
            "compute_ample": compute_ample,
            "compute_tight": compute_tight,
            "spectrum_ample": spectrum_ample,
            "spectrum_tight": spectrum_tight,
        },
    }


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------
def _smoke_test(output_dir: Path, device: str) -> Dict[str, Any]:
    """100-request order-invariance smoke with capacity=50, slots=320, seed=3030."""
    _set_worker_threads()
    mod_reg = ModulationRegistry.from_profile(MODULATION_PROFILE)
    agent_c = _load_ppo_c(str(CHECKPOINT_C), device)
    agent_r = _load_ppo_r(str(CHECKPOINT_R), mod_reg, device)
    v13 = _load_v13_ranker(str(CHECKPOINT_V13), device)

    env_proto = _make_env_for_cell(320, [50.0] * NUM_SERVERS)
    requests, trace_hash = _generate_trace(env_proto, 3030, 100)

    order_a = _METHODS
    order_b = list(reversed(_METHODS))

    def run_order(order):
        out = {}
        for method in order:
            env = _make_env_for_cell(320, [50.0] * NUM_SERVERS)
            env.reset(requests)
            out[method] = _run_method_on_trace(
                env, requests, method, agent_c, agent_r, v13, mod_reg
            )
        return out

    res_a = run_order(order_a)
    res_b = run_order(order_b)

    passed = True
    mismatches = []
    for method in _METHODS:
        keys = ["blocking_rate", "server_overload_rate", "no_suitable_block_rate"]
        for k in keys:
            if not math.isclose(res_a[method][k], res_b[method][k], rel_tol=1e-9, abs_tol=1e-9):
                passed = False
                mismatches.append((method, k, res_a[method][k], res_b[method][k]))

    report = {
        "status": "passed" if passed else "failed",
        "trace_hash": trace_hash,
        "order_a": order_a,
        "order_b": order_b,
        "results_a": res_a,
        "results_b": res_b,
        "mismatches": mismatches,
    }
    (output_dir / "SMOKE_TEST.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    return report


# ---------------------------------------------------------------------------
# Report writing
# ---------------------------------------------------------------------------
def _write_calibration_report(output_dir: Path, cells: List[Dict[str, Any]]) -> None:
    json_path = output_dir / "CALIBRATION.json"
    json_path.write_text(json.dumps({"cells": cells}, indent=2, default=str), encoding="utf-8")

    lines = [
        "# COST239 Compute-vs-Spectrum Calibration",
        "",
        f"Topology: {TOPOLOGY}, K_C={K_C}, K_R={K_R}, path_sort={PATH_SORT}, block_sort={BLOCK_SORT}",
        f"Traffic: arrival_interval={ARRIVAL_INTERVAL}, holding={HOLDING_MIN}-{HOLDING_MAX}, "
        f"size={SIZE_MIN_MB}-{SIZE_MAX_MB} MB, edge_cost={EDGE_COST_MIN}-{EDGE_COST_MAX}",
        "",
        "| Capacity | Slots | Blocking | Overload share | Optical share | N seeds |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for c in cells:
        lines.append(
            f"| {c['capacity']} | {c['num_slots']} | "
            f"{c['blocking_mean']:.2%} ± {c['blocking_std']:.2%} | "
            f"{c['overload_share_mean']:.2%} | "
            f"{c['optical_share_mean']:.2%} | {c['n_seeds']} |"
        )
    lines.append("")
    (output_dir / "CALIBRATION.md").write_text("\n".join(lines), encoding="utf-8")


def _write_protocol(output_dir: Path, selected_cells: Optional[Dict[str, Any]]) -> None:
    protocol = {
        "topology": TOPOLOGY,
        "num_servers": NUM_SERVERS,
        "K_C": K_C,
        "K_R": K_R,
        "path_sort_strategy": PATH_SORT,
        "block_sort_strategy": BLOCK_SORT,
        "edge_cost_min": EDGE_COST_MIN,
        "edge_cost_max": EDGE_COST_MAX,
        "arrival_interval": ARRIVAL_INTERVAL,
        "holding_min": HOLDING_MIN,
        "holding_max": HOLDING_MAX,
        "deadline_min": DEADLINE_MIN,
        "deadline_max": DEADLINE_MAX,
        "size_min_mb": SIZE_MIN_MB,
        "size_max_mb": SIZE_MAX_MB,
        "num_splits": NUM_SPLITS,
        "split_profile": SPLIT_PROFILE,
        "max_blocks": MAX_BLOCKS,
        "modulation_profile": MODULATION_PROFILE,
        "checkpoints": {
            "ppo_c": str(CHECKPOINT_C),
            "ppo_r": str(CHECKPOINT_R),
            "v1.3_ranker": str(CHECKPOINT_V13),
        },
        "seeds": [3030, 4040],
        "warmup": 300,
        "evaluation_requests": 1000,
        "selected_cells": selected_cells,
    }
    (output_dir / "PROTOCOL.json").write_text(json.dumps(protocol, indent=2, default=str), encoding="utf-8")


def _write_2x2_report(output_dir: Path, results: List[Dict[str, Any]]) -> None:
    json_path = output_dir / "QUICK_2X2_RESULTS.json"
    json_path.write_text(json.dumps({"results": results}, indent=2, default=str), encoding="utf-8")

    # Aggregate per cell/method.
    table: Dict[str, Dict[str, Any]] = {}
    for r in results:
        if r["status"] != "ok":
            continue
        cell = r["cell"]
        table.setdefault(cell, {"capacity": r["capacity"], "num_slots": r["num_slots"], "seeds": []})
        table[cell]["seeds"].append(r)

    lines = [
        "# COST239 2×2 Bottleneck Quick Diagnostic Results",
        "",
        f"Topology: {TOPOLOGY}, K_C={K_C}, K_R={K_R}",
        "",
        "| Cell | Compute | Spectrum | Method | Blocking | Overload share | Optical share | "
        "Delay mean/P95 | Decision ms mean/P95 | Mean FS | Mean SE |",
        "|---|---:|---:|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    v1_3_deltas: Dict[str, Dict[str, float]] = {}
    for cell_label in ["A", "B", "C", "D"]:
        if cell_label not in table:
            continue
        cell = table[cell_label]
        cap = cell["capacity"]
        slots = cell["num_slots"]
        methods = _METHODS
        blocking_by_method: Dict[str, float] = {}
        for method in methods:
            seeds = [s["methods"][method] for s in cell["seeds"]]
            blocking_mean, _ = _aggregate(cell["seeds"], ["methods", method, "blocking_rate"])
            overload_mean, _ = _aggregate(cell["seeds"], ["methods", method, "overload_share_of_blocking"])
            optical_mean, _ = _aggregate(cell["seeds"], ["methods", method, "optical_share_of_blocking"])
            delay_mean, _ = _aggregate(cell["seeds"], ["methods", method, "mean_delay_ms"])
            delay_p95, _ = _aggregate(cell["seeds"], ["methods", method, "p95_delay_ms"])
            dec_mean, _ = _aggregate(cell["seeds"], ["methods", method, "mean_decision_time_ms"])
            dec_p95, _ = _aggregate(cell["seeds"], ["methods", method, "p95_decision_time_ms"])
            fs_mean, _ = _aggregate(cell["seeds"], ["methods", method, "mean_required_fs"])
            se_mean, _ = _aggregate(cell["seeds"], ["methods", method, "mean_spectral_efficiency"])
            lines.append(
                f"| {cell_label} | {cap} | {slots} | {method} | "
                f"{blocking_mean:.2%} | {overload_mean:.2%} | {optical_mean:.2%} | "
                f"{delay_mean:.2f}/{delay_p95:.2f} | {dec_mean:.2f}/{dec_p95:.2f} | "
                f"{fs_mean if fs_mean is not None else '-':.2f} | {se_mean if se_mean is not None else '-':.2f} |"
            )
            blocking_by_method[method] = blocking_mean
        lines.append("|")
        v1 = blocking_by_method.get("v1.3_ranker", float("nan"))
        v1_3_deltas[cell_label] = {
            m: (v1 - blocking_by_method[m]) for m in ["ppo_r", "ksp_ff_plain", "ksp_ff_highmod"]
        }

    lines.extend(["", "## v1.3 blocking delta (v1.3 - baseline)", ""])
    lines.append("| Cell | v1.3 - PPO-R | v1.3 - KSP-FF plain | v1.3 - KSP-FF high-mod |")
    lines.append("|---|---:|---:|---:|")
    for cell_label in ["A", "B", "C", "D"]:
        if cell_label not in v1_3_deltas:
            continue
        d = v1_3_deltas[cell_label]
        lines.append(
            f"| {cell_label} | "
            f"{d['ppo_r']*100:+.2f}pp | {d['ksp_ff_plain']*100:+.2f}pp | {d['ksp_ff_highmod']*100:+.2f}pp |"
        )

    lines.append("")
    (output_dir / "QUICK_2X2_RESULTS.md").write_text("\n".join(lines), encoding="utf-8")


def _aggregate(records: List[Dict[str, Any]], key_path: List[str]) -> Tuple[float, float]:
    vals = []
    for rec in records:
        obj = rec
        for k in key_path:
            obj = obj[k]
        if obj is None:
            continue
        vals.append(float(obj))
    if not vals:
        return None, 0.0
    arr = np.asarray(vals, dtype=np.float32)
    return float(arr.mean()), float(arr.std(ddof=0))


def _write_final_diagnosis(
    output_dir: Path,
    selected_cells: Optional[Dict[str, Any]],
    results: List[Dict[str, Any]],
    smoke_passed: bool,
) -> None:
    if not smoke_passed:
        conclusion = (
            "Smoke test failed: per-method metrics changed when the execution order was swapped. "
            "This indicates trace or environment-state contamination and invalidates the 2×2 results."
        )
    elif selected_cells is None:
        conclusion = (
            "当前 COST239 流量模型和资源配置无法形成稳定的 R-side optical-discrimination regime。"
        )
    else:
        # Compute optical-tight v1.3 advantage.
        cell_c_v1_adv = None
        cell_d_v1_adv = None
        cell_c_optical = None
        cell_b_overload = None
        for r in results:
            if r["status"] != "ok" or r["cell"] not in ("C", "D", "B"):
                continue
            methods = r["methods"]
            v1 = methods["v1.3_ranker"]["blocking_rate"]
            if r["cell"] == "C":
                cell_c_v1_adv = v1 - methods["ppo_r"]["blocking_rate"]
                cell_c_optical = methods["ppo_r"]["optical_share_of_blocking"]
            elif r["cell"] == "D":
                cell_d_v1_adv = v1 - methods["ppo_r"]["blocking_rate"]
            elif r["cell"] == "B":
                cell_b_overload = methods["ppo_r"]["overload_share_of_blocking"]

        # Heuristic diagnosis text.
        if cell_c_v1_adv is not None and cell_c_v1_adv < -0.005 and (
            cell_c_optical is None or cell_c_optical >= 0.50
        ):
            conclusion = (
                "v1.3 仅在 optical-tight cell (C) 明显领先，支持「当前 overload-dominated 协议掩盖 "
                "R-side 光层排序价值」的解释。"
            )
        elif cell_c_v1_adv is not None and abs(cell_c_v1_adv) <= 0.005 and (
            cell_c_optical is None or cell_c_optical >= 0.50
        ):
            conclusion = (
                "optical-tight cell 仍存在较大 optical blocking，但 v1.3 未显著优于 PPO-R。"
                "更可能是标签、特征或候选策略本身不足，而不是 C-side ceiling。"
            )
        elif cell_d_v1_adv is not None and cell_d_v1_adv < -0.005 and (
            cell_b_overload is not None and cell_b_overload >= 0.80
        ):
            conclusion = (
                "v1.3 在 overload-dominated cell 领先，不得把增益归因于长期光谱机会成本，"
                "应进一步检查 C/R 耦合、动作偏好或隐式资源正则化。"
            )
        else:
            conclusion = (
                "2×2 结果未显示 v1.3 在任何资源瓶颈组合下有稳定优势。"
                "当前实验不能检验后决策光层价值，需要换拓扑、业务分布或光谱资源模型。"
            )

    (output_dir / "FINAL_DIAGNOSIS.md").write_text(
        "# Final Diagnosis\n\n" + conclusion + "\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------
def _worker_with_log(args):
    """Top-level worker wrapper; dispatches calibration or eval tasks."""
    tag = args[0]
    inner = args[1:]
    if tag == "cal":
        result = _calibration_task(*inner)
    elif tag == "eval":
        result = _eval_task(*inner)
    else:
        raise ValueError(f"Unknown task tag: {tag}")
    label = (
        f"cell_{result.get('cell', 'cal')}_cap{result.get('capacity', 0)}_slots{result.get('num_slots', 0)}_seed{result.get('seed', 0)}"
    )
    log_dir = _worker_log_dir
    if log_dir is not None:
        log_dir.mkdir(parents=True, exist_ok=True)
        tmp = log_dir / f"{label}.json.tmp"
        final = log_dir / f"{label}.json"
        tmp.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        tmp.replace(final)
    return result


def _run_pool(tasks: List[Any], max_workers: int, device: str) -> List[Dict[str, Any]]:
    output_dir = Path("sa_hmarl/experiments/v13_cost239_2x2_quick")
    log_dir = output_dir / "task_logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    start = time.time()
    completed = 0
    results: List[Dict[str, Any]] = []
    with mp.Pool(
        processes=max_workers,
        initializer=_init_worker,
        initargs=(device, str(log_dir)),
    ) as pool:
        for result in pool.imap_unordered(_worker_with_log, tasks):
            results.append(result)
            completed += 1
            elapsed = time.time() - start
            avg = elapsed / completed if completed else 0.0
            eta = avg * (len(tasks) - completed)
            print(
                f"[{completed}/{len(tasks)}] completed in {elapsed:.1f}s, "
                f"ETA {eta:.1f}s, last={result.get('cell') or result.get('capacity')}/{result.get('num_slots')}/seed{result.get('seed')}"
            )
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--max_workers", type=int, default=5)
    parser.add_argument("--skip_calibration", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments/v13_cost239_2x2_quick")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    task_logs_dir = output_dir / "task_logs"
    task_logs_dir.mkdir(parents=True, exist_ok=True)

    # Smoke test
    if not args.skip_smoke:
        smoke = _smoke_test(output_dir, args.device)
        print("Smoke test:", smoke["status"])
        if smoke["status"] != "passed":
            print("Smoke mismatches:", smoke["mismatches"])
            _write_final_diagnosis(output_dir, None, [], False)
            return
    else:
        smoke = {"status": "skipped"}

    # Calibration
    if not args.skip_calibration:
        capacities = [50, 75, 100, 150]
        num_slots = [64, 80, 100, 160, 320]
        seeds = [3030, 4040]
        cal_tasks = [
            ("cal", ci, si, cap, slots, seed)
            for ci, cap in enumerate(capacities)
            for si, slots in enumerate(num_slots)
            for seed in seeds
        ]
        print(f"Running calibration grid: {len(cal_tasks)} tasks")
        cal_results = _run_pool(cal_tasks, args.max_workers, args.device)
        cells = _calibration_table(cal_results)
        _write_calibration_report(output_dir, cells)
        selected = _select_2x2_cells(cells)
    else:
        # Load previously selected cells from protocol if available.
        proto_path = output_dir / "PROTOCOL.json"
        if proto_path.exists():
            selected = json.loads(proto_path.read_text(encoding="utf-8")).get("selected_cells")
        else:
            selected = None
        cells = []

    _write_protocol(output_dir, selected)

    if selected is None:
        print("No stable optical-discrimination regime found. Stopping.")
        _write_final_diagnosis(output_dir, selected, [], smoke["status"] == "passed")
        return

    print("Selected cells:", json.dumps(selected, indent=2, default=str))

    # 2×2 evaluation
    eval_tasks = []
    for label, cfg in selected.items():
        if label not in ("A", "B", "C", "D"):
            continue
        for seed in [3030, 4040]:
            eval_tasks.append(
                (
                    "eval",
                    label,
                    cfg["num_slots"],
                    int(cfg["capacity"]),
                    [float(cfg["capacity"])] * NUM_SERVERS,
                    seed,
                )
            )

    print(f"Running 2×2 evaluation: {len(eval_tasks)} tasks")
    eval_results = _run_pool(eval_tasks, args.max_workers, args.device)
    _write_2x2_report(output_dir, eval_results)
    _write_final_diagnosis(output_dir, selected, eval_results, smoke["status"] == "passed")
    print("Done. Reports in", output_dir)


if __name__ == "__main__":
    main()
