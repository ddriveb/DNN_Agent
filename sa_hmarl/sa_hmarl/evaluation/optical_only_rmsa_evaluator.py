"""Evaluator for the optical-only R-side discriminative audit.

This module runs micro/calibration/smoke/pilot phases without any C-side
machinery.  All R methods act on identical optical-demand request traces.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
from sa_hmarl.agents.ppo_agents import PPOAgentR
from sa_hmarl.agents.r_agent import AgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.evaluation.optical_only_rmsa_env import (
    DEFAULT_BITRATE_MAX_GBPS,
    DEFAULT_BITRATE_MIN_GBPS,
    DEFAULT_GUARD_BAND_FS,
    DEFAULT_MEAN_HOLDING_TIME,
    DEFAULT_SLOT_BW_HZ,
    ODRequest,
    OpticalOnlyRMSAEnv,
    generate_od_requests,
)
from sa_hmarl.evaluation.r_poststate_features import (
    POSTSTATE_V1_FEATURE_NAMES,
    compute_optical_afterstate,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.utils.checkpoint import load_checkpoint


R_MODES = ("ksp_ff_highest", "ppo_r_top1", "strict_v13", "v135_afterstate", "v135_afterstate_explicit")
RANKER_CKPTS = {
    "strict_v13": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/strict_v13/seed_42/ranking_model.pt",
    "v135_afterstate": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate/seed_42/ranking_model.pt",
    "v135_afterstate_explicit": "sa_hmarl/experiments/v135_afterstate_diagnostic_pilot/checkpoints/v135_afterstate_explicit/seed_42/ranking_model.pt",
}
PPO_R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"
BASE_DIR = Path("sa_hmarl/experiments/v135_optical_only_rmsa_audit")

# Feature names that are C-only in the locked checkpoints.  They are replaced by
# the checkpoint training mean so that normalization yields zero.
C_ONLY_FEATURE_NAMES = {
    "split_norm", "server_norm", "deadline_norm", "intermediate_size_norm",
    "server_utilization", "k_c_valid_ratio", "k_r_total_ratio", "phi_spec_norm",
    "server_util_context", "server_margin_context",
}

# Afterstate feature keys produced by compute_optical_afterstate.
OPTICAL_AFTERSTATE_NAMES = [
    "path_lfb_after", "path_free_ratio_after", "global_lfb_after",
    "global_lfb_ratio_after", "frag_after", "delta_frag",
    "free_block_count_after", "free_block_count_delta",
    "min_edge_lfb_margin_after", "min_edge_free_ratio_after",
    "bottleneck_edge_util_after", "occupied_slot_hops", "path_conflict_after",
]


def _set_thread_env():
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("TORCH_NUM_THREADS", "1")


# ---------------------------------------------------------------------------
# Checksum helpers
# ---------------------------------------------------------------------------
def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    try:
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return "N/A"


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_requests(requests: List[ODRequest]) -> str:
    payload = [
        {
            "req_id": r.req_id,
            "src": r.src_node,
            "dst": r.dst_node,
            "bitrate_gbps": r.bitrate_gbps,
            "arrival": r.arrival_time,
            "holding": r.holding_time,
        }
        for r in requests
    ]
    return _hash_text(json.dumps(payload, sort_keys=True, default=str))


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
def load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    for p in agent.policy_net.parameters():
        p.requires_grad = False
    return agent


def load_ranker(ckpt_path: str, device: str = "cpu") -> Dict[str, Any]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    input_dim = int(ckpt["input_dim"])
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"),
        input_dim,
        tuple(ckpt.get("hidden_dims", [128, 64])),
        float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    feature_mean = np.asarray(ckpt.get("feature_mean", 0.0), dtype=np.float32)
    feature_std = np.maximum(
        np.asarray(ckpt.get("feature_std", 1.0), dtype=np.float32), 1e-6
    )
    if feature_mean.ndim == 0:
        feature_mean = np.zeros(input_dim, dtype=np.float32)
    if feature_std.ndim == 0:
        feature_std = np.ones(input_dim, dtype=np.float32)
    feature_names = list(ckpt.get("feature_names", []))
    if not feature_names:
        # Fallback to the locked schema names when the checkpoint omits them.
        if input_dim == 25:
            feature_names = POSTSTATE_V1_FEATURE_NAMES[:25]
        elif input_dim == 41:
            feature_names = POSTSTATE_V1_FEATURE_NAMES[:]
        elif input_dim == 16:
            feature_names = POSTSTATE_V1_FEATURE_NAMES[25:]
        else:
            feature_names = [f"f{i}" for i in range(input_dim)]
        # Realign mean/std to the inferred names if they are scalar placeholders.
        if feature_mean.size == 1 and input_dim > 1:
            feature_mean = np.zeros(input_dim, dtype=np.float32)
        if feature_std.size == 1 and input_dim > 1:
            feature_std = np.ones(input_dim, dtype=np.float32)

    mean_by_name = {name: float(feature_mean[i]) for i, name in enumerate(feature_names)}
    std_by_name = {name: float(feature_std[i]) for i, name in enumerate(feature_names)}

    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_names": feature_names,
        "input_dim": input_dim,
        "mean_by_name": mean_by_name,
        "std_by_name": std_by_name,
    }


# ---------------------------------------------------------------------------
# Feature adapter
# ---------------------------------------------------------------------------
def _build_base_features(obs: Dict[str, Any], mod_reg: ModulationRegistry) -> Tuple[np.ndarray, np.ndarray]:
    helper = AgentR(input_dim=11, mod_registry=mod_reg, feature_mode="default")
    return helper.build_action_features(obs)


def _ppo_r_topk_actions(agent_r: PPOAgentR, obs: Dict[str, Any], top_k: int = 30) -> List[int]:
    features, mask = agent_r.build_action_features(obs)
    legal = np.flatnonzero(np.asarray(mask, dtype=bool))
    if legal.size == 0:
        return []
    with torch.no_grad():
        x = torch.as_tensor(features, dtype=torch.float32, device=agent_r.device).unsqueeze(0)
        logits = agent_r.policy_net(x).squeeze(0).cpu().numpy()
    logits[~np.asarray(mask, dtype=bool)] = -np.inf
    top_k = min(top_k, legal.size)
    top_local = np.argsort(-logits[legal], kind="stable")[:top_k]
    return [int(legal[i]) for i in top_local]


def _decode_action(action_idx: int, obs: Dict[str, Any], max_blocks: int) -> Tuple[int, int, int]:
    num_mods = len(obs["mod_names"])
    return decode_agent_r_action(action_idx, num_mods, max_blocks)


def build_transfer_features(
    obs: Dict[str, Any],
    env: OpticalOnlyRMSAEnv,
    req: ODRequest,
    action_indices: List[int],
    ranker: Dict[str, Any],
) -> np.ndarray:
    """Build a normalized feature matrix for candidate actions using the locked
    checkpoint's feature names.  C-only names are replaced by the checkpoint
    training mean so that they normalize to zero.
    """
    base_features, mask = _build_base_features(obs, env.mod_reg)
    num_paths = max(len(obs["candidate_paths"]), 1)
    num_mods = max(len(obs["mod_names"]), 1)
    max_blocks = max(env.max_blocks, 1)

    raw_r_valid_ratio = float(mask.sum()) / max(len(mask), 1)
    selected_valid_r_ratio = float(mask.sum()) / max(num_paths * num_mods * max_blocks, 1)
    holding_norm = float(req.holding_time) / 10.0
    holding_time_norm = float(req.holding_time) / 100.0

    mean_by_name = ranker["mean_by_name"]
    std_by_name = ranker["std_by_name"]

    rows = []
    for action_idx in action_indices:
        path_idx, mod_idx, block_idx = _decode_action(action_idx, obs, env.max_blocks)

        base = base_features[action_idx]
        optical_after = compute_optical_afterstate(env, path_idx, mod_idx, block_idx, obs)

        feature_values: Dict[str, float] = {
            "path_length_km": float(base[0]),
            "hop_count": float(base[1]),
            "lfb": float(base[2]),
            "free_ratio": float(base[3]),
            "frag_index": float(base[4]),
            "spectral_efficiency": float(base[5]),
            "reach_km": float(base[6]),
            "required_fs": float(base[7]),
            "block_size": float(base[8]),
            "block_waste": float(base[9]),
            "path_mod_feasible": float(base[10]),
            "path_idx_norm": float(path_idx) / max(num_paths - 1, 1),
            "mod_idx_norm": float(mod_idx) / max(num_mods - 1, 1),
            "block_idx_norm": float(block_idx) / max(max_blocks - 1, 1),
            "raw_r_valid_ratio": raw_r_valid_ratio,
            "selected_valid_r_ratio": selected_valid_r_ratio,
            "holding_norm": holding_norm,
            "holding_time_norm": holding_time_norm,
        }
        feature_values.update(optical_after)

        # Start with C-only and missing names set to their training mean.
        vector = []
        for name in ranker["feature_names"]:
            if name in C_ONLY_FEATURE_NAMES:
                raw_value = mean_by_name.get(name, 0.0)
            else:
                raw_value = feature_values.get(name, mean_by_name.get(name, 0.0))
            vector.append(raw_value)
        vector = np.asarray(vector, dtype=np.float32)
        if not np.all(np.isfinite(vector)):
            raise ValueError(f"Non-finite transfer feature for action {action_idx}")
        rows.append(vector)

    return np.stack(rows, axis=0).astype(np.float32)


def normalize_transfer_features(x: np.ndarray, ranker: Dict[str, Any]) -> np.ndarray:
    """Apply checkpoint mean/std normalization to raw transfer features."""
    x_norm = (x - ranker["feature_mean"]) / ranker["feature_std"]
    x_norm[~np.isfinite(x_norm)] = 0.0
    return x_norm


# ---------------------------------------------------------------------------
# R method dispatch
# ---------------------------------------------------------------------------
def select_r_action(
    r_mode: str,
    obs: Dict[str, Any],
    env: OpticalOnlyRMSAEnv,
    req: ODRequest,
    agent_r: Optional[PPOAgentR],
    rankers: Dict[str, Dict[str, Any]],
) -> Tuple[Optional[int], bool, Dict[str, Any]]:
    """Return (action_idx, is_valid, info)."""
    info: Dict[str, Any] = {
        "candidate_count": 0,
        "score_margin": 0.0,
        "feature_finite": True,
        "nan_inf_feature_count": 0,
        "nan_inf_score_count": 0,
        "ranker_top1_in_candidates": False,
    }
    mask = np.asarray(obs.get("agent_r_mask", []), dtype=bool)
    if not mask.any():
        return None, False, info

    if r_mode == "ksp_ff_highest":
        a = ksp_ff_highest_mod_action(obs)
        return (a, a is not None, info)

    if r_mode == "ppo_r_top1":
        a = agent_r.select_action(obs, deterministic=True)
        return (a, a is not None, info)

    # Ranker-based methods: use PPO-R top-30 as the candidate pool.
    candidates = _ppo_r_topk_actions(agent_r, obs, top_k=30)
    info["candidate_count"] = len(candidates)
    if not candidates:
        return None, False, info

    ranker = rankers[r_mode]
    try:
        x = build_transfer_features(obs, env, req, candidates, ranker)
        x_norm = normalize_transfer_features(x, ranker)
        finite_all = np.isfinite(x_norm).all()
        info["feature_finite"] = bool(finite_all)
        if not finite_all:
            info["nan_inf_feature_count"] = int(np.sum(~np.isfinite(x_norm)))
        with torch.no_grad():
            scores = (
                ranker["model"](
                    torch.as_tensor(x_norm, dtype=torch.float32, device=ranker["device"]).unsqueeze(0)
                )
                .squeeze(0)
                .cpu()
                .numpy()
            )
        if not np.all(np.isfinite(scores)):
            info["nan_inf_score_count"] = int(np.sum(~np.isfinite(scores)))
        best = int(np.argmax(scores))
        sorted_scores = np.sort(scores)[::-1]
        info["score_margin"] = float(sorted_scores[0] - sorted_scores[1]) if len(sorted_scores) >= 2 else 0.0
        return candidates[best], True, info
    except Exception as exc:
        info["exception"] = str(exc)
        return None, False, info


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------
@dataclass
class EpisodeResult:
    seed: int
    r_mode: str
    evaluated_requests: int = 0
    admitted: int = 0
    blocked: int = 0
    r_no_valid_action: int = 0

    fs_sum: float = 0.0
    slot_hop_sum: float = 0.0
    path_length_sum: float = 0.0
    hop_count_sum: float = 0.0

    mod_name_list: List[str] = field(default_factory=list)
    path_idx_list: List[int] = field(default_factory=list)
    block_start_list: List[int] = field(default_factory=list)
    required_fs_list: List[int] = field(default_factory=list)
    hop_count_list: List[float] = field(default_factory=list)
    path_length_list: List[float] = field(default_factory=list)

    candidate_counts: List[int] = field(default_factory=list)
    score_margins: List[float] = field(default_factory=list)
    ppo_agreement: int = 0
    ppo_decisions: int = 0

    selected_actions: List[Dict[str, Any]] = field(default_factory=list)

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.evaluated_requests, 1))

    def avg_fs(self) -> float:
        return float(self.fs_sum / max(self.admitted, 1))

    def avg_slot_hops(self) -> float:
        return float(self.slot_hop_sum / max(self.admitted, 1))

    def avg_path_length_km(self) -> float:
        return float(self.path_length_sum / max(self.admitted, 1))

    def avg_hop_count(self) -> float:
        return float(self.hop_count_sum / max(self.admitted, 1))

    def ppo_agreement_rate(self) -> Optional[float]:
        if self.ppo_decisions == 0:
            return None
        return float(self.ppo_agreement / self.ppo_decisions)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------
def run_episode(
    seed: int,
    requests: List[ODRequest],
    r_mode: str,
    agent_r: Optional[PPOAgentR],
    rankers: Dict[str, Dict[str, Any]],
    config: Dict[str, Any],
    warmup_requests: int = 0,
    store_actions: bool = True,
) -> Dict[str, Any]:
    env = OpticalOnlyRMSAEnv(
        topology=config["topology"],
        num_slots=config["num_slots"],
        k_paths=config["k_paths"],
        max_blocks=config["max_blocks"],
        path_sort_strategy=config["path_sort_strategy"],
        block_sort_strategy=config["block_sort_strategy"],
        mod_registry=ModulationRegistry.from_profile(config["modulation_profile"]),
        slot_bw_hz=config.get("slot_bw_hz", DEFAULT_SLOT_BW_HZ),
        guard_band_fs=config.get("guard_band_fs", DEFAULT_GUARD_BAND_FS),
        seed=seed,
    )
    env.reset(requests)

    result = EpisodeResult(seed=seed, r_mode=r_mode)

    for step_idx, req in enumerate(requests):
        env.advance_time(req.arrival_time)
        obs = env.build_observation(req)

        is_warmup = step_idx < warmup_requests

        action_idx, valid, info = select_r_action(r_mode, obs, env, req, agent_r, rankers)

        if not valid:
            if not is_warmup:
                result.evaluated_requests += 1
                result.blocked += 1
                result.r_no_valid_action += 1
                if store_actions:
                    result.selected_actions.append({
                        "req_id": req.req_id,
                        "action_idx": None,
                        "success": False,
                        "reason": "r_no_valid_action",
                    })
            continue

        step_info = env.step(action_idx, obs, req.holding_time)

        if not is_warmup:
            result.evaluated_requests += 1
            if step_info["success"]:
                result.admitted += 1
                result.fs_sum += step_info["required_fs"]
                result.slot_hop_sum += step_info["required_fs"] * step_info["hop_count"]
                result.path_length_sum += step_info["path_length_km"]
                result.hop_count_sum += step_info["hop_count"]

                result.mod_name_list.append(obs["mod_names"][_decode_action(action_idx, obs, env.max_blocks)[1]])
                result.path_idx_list.append(_decode_action(action_idx, obs, env.max_blocks)[0])
                result.block_start_list.append(step_info["start_slot"])
                result.required_fs_list.append(step_info["required_fs"])
                result.hop_count_list.append(step_info["hop_count"])
                result.path_length_list.append(step_info["path_length_km"])

                if store_actions:
                    result.selected_actions.append({
                        "req_id": req.req_id,
                        "action_idx": int(action_idx),
                        "success": True,
                        "path_idx": _decode_action(action_idx, obs, env.max_blocks)[0],
                        "mod_idx": _decode_action(action_idx, obs, env.max_blocks)[1],
                        "block_idx": _decode_action(action_idx, obs, env.max_blocks)[2],
                        "start_slot": step_info["start_slot"],
                        "required_fs": step_info["required_fs"],
                        "path_length_km": step_info["path_length_km"],
                        "hop_count": step_info["hop_count"],
                    })
            else:
                result.blocked += 1
                if store_actions:
                    result.selected_actions.append({
                        "req_id": req.req_id,
                        "action_idx": int(action_idx),
                        "success": False,
                        "reason": step_info.get("reason", "unknown"),
                    })

            if r_mode in ("strict_v13", "v135_afterstate", "v135_afterstate_explicit"):
                result.candidate_counts.append(info.get("candidate_count", 0))
                result.score_margins.append(info.get("score_margin", 0.0))

            # PPO agreement: compare selected action to PPO-R top-1.
            if agent_r is not None:
                ppo_action = agent_r.select_action(obs, deterministic=True)
                result.ppo_decisions += 1
                if ppo_action is not None and int(ppo_action) == int(action_idx):
                    result.ppo_agreement += 1

    # Drain remaining lightpaths for conservation checks.
    if requests:
        max_release = max(r.arrival_time + r.holding_time for r in requests)
        env.advance_time(max_release + 1.0)

    return _result_to_dict(result, env, requests, config, store_actions=store_actions)


def _count(items: List[Any]) -> Dict[str, int]:
    c: Dict[str, int] = {}
    for x in items:
        k = str(x)
        c[k] = c.get(k, 0) + 1
    return c


def _normalized_dist(counts: Dict[str, int], denom: int) -> Dict[str, Any]:
    if denom <= 0:
        return {"_reason": "denominator_zero", "normalized": {}}
    total = sum(counts.values())
    if total == 0:
        return {"_reason": "count_zero", "normalized": {}}
    return {
        "raw_counts": counts,
        "normalized": {k: v / denom for k, v in counts.items()},
        "sum_normalized": sum(v / denom for v in counts.values()),
    }


def _result_to_dict(
    result: EpisodeResult,
    env: OpticalOnlyRMSAEnv,
    requests: List[ODRequest],
    config: Dict[str, Any],
    store_actions: bool = True,
) -> Dict[str, Any]:
    ad = result.admitted
    ev = result.evaluated_requests

    mod_counts = _count(result.mod_name_list)
    path_idx_counts = _count(result.path_idx_list)
    block_start_counts = _count(result.block_start_list)
    required_fs_counts = _count(result.required_fs_list)
    hop_count_counts = _count([round(h, 2) for h in result.hop_count_list])
    path_length_counts = _count([round(pl, 2) for pl in result.path_length_list])

    return {
        "schema_version": "optical-only-rmsa-audit-v1.0",
        "r_mode": result.r_mode,
        "seed": result.seed,
        "evaluated_requests": ev,
        "admitted": ad,
        "blocked": result.blocked,
        "blocking_rate": result.blocking_rate(),
        "r_no_valid_action": result.r_no_valid_action,
        "avg_fs": result.avg_fs(),
        "avg_slot_hops": result.avg_slot_hops(),
        "avg_path_length_km": result.avg_path_length_km(),
        "avg_hop_count": result.avg_hop_count(),
        "ppo_agreement_rate": result.ppo_agreement_rate(),
        "ppo_agreement_numerator": result.ppo_agreement,
        "ppo_agreement_denominator": result.ppo_decisions,
        "avg_candidate_count": float(np.mean(result.candidate_counts)) if result.candidate_counts else 0.0,
        "candidate_count_p95": float(np.percentile(result.candidate_counts, 95)) if result.candidate_counts else 0.0,
        "avg_score_margin": float(np.mean(result.score_margins)) if result.score_margins else 0.0,
        "score_margin_p95": float(np.percentile(result.score_margins, 95)) if result.score_margins else 0.0,
        "mod_distribution": _normalized_dist(mod_counts, ad),
        "path_idx_distribution": _normalized_dist(path_idx_counts, ad),
        "block_start_distribution": _normalized_dist(block_start_counts, ad),
        "required_fs_distribution": _normalized_dist(required_fs_counts, ad),
        "hop_count_distribution": _normalized_dist(hop_count_counts, ad),
        "path_length_distribution": _normalized_dist(path_length_counts, ad),
        "request_trace_hash": _hash_requests(requests),
        "config_hash": _hash_text(json.dumps(config, sort_keys=True, default=str)),
        "evaluator_code_hash": _sha256_file(__file__),
        "final_active_connections": len(env.active_connections),
        "final_utilization": env.get_utilization(),
        "selected_actions": result.selected_actions if store_actions else [],
        "selected_actions_stored": store_actions,
    }


# ---------------------------------------------------------------------------
# Phase runners
# ---------------------------------------------------------------------------
def _make_config(args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "topology": args.topology,
        "num_slots": args.num_slots,
        "k_paths": args.k_paths,
        "max_blocks": args.max_blocks,
        "path_sort_strategy": args.path_sort_strategy,
        "block_sort_strategy": args.block_sort_strategy,
        "modulation_profile": args.modulation_profile,
        "slot_bw_hz": args.slot_bw_hz,
        "guard_band_fs": args.guard_band_fs,
        "mean_holding_time": args.mean_holding_time,
        "bitrate_min_gbps": args.bitrate_min_gbps,
        "bitrate_max_gbps": args.bitrate_max_gbps,
    }


def _generate_trace(seed: int, arrival_interval: float, warmup: int, evaluated: int, config: Dict[str, Any]) -> List[ODRequest]:
    num_nodes = OpticalNetwork(config["topology"], config["num_slots"], seed=seed).NUM_NODES
    rng = np.random.RandomState(seed)
    return generate_od_requests(
        num_nodes=num_nodes,
        rng=rng,
        num_requests=warmup + evaluated,
        arrival_interval=arrival_interval,
        mean_holding_time=config["mean_holding_time"],
        bitrate_min_gbps=config["bitrate_min_gbps"],
        bitrate_max_gbps=config["bitrate_max_gbps"],
        poisson_arrivals=True,
        exponential_holding=True,
    )


def run_micro(args: argparse.Namespace) -> Dict[str, Any]:
    config = _make_config(args)
    requests = _generate_trace(args.seed, args.arrival_interval, args.warmup_requests, args.requests_per_episode, config)
    agent_r = load_ppo_r(PPO_R_CKPT, ModulationRegistry.from_profile(config["modulation_profile"]), args.device)
    rankers = {m: load_ranker(RANKER_CKPTS[m], args.device) for m in ("strict_v13", "v135_afterstate", "v135_afterstate_explicit")}

    results = {}
    for r_mode in R_MODES:
        results[r_mode] = run_episode(args.seed, requests, r_mode, agent_r, rankers, config, warmup_requests=args.warmup_requests, store_actions=args.store_actions)
    return {"phase": "micro", "config": config, "results": results}


def run_calibration(args: argparse.Namespace) -> Dict[str, Any]:
    config = _make_config(args)
    agent_r = load_ppo_r(PPO_R_CKPT, ModulationRegistry.from_profile(config["modulation_profile"]), args.device)
    rankers = {m: load_ranker(RANKER_CKPTS[m], args.device) for m in ("strict_v13", "v135_afterstate", "v135_afterstate_explicit")}

    calibration_results = []
    for rho in args.rho_list:
        arrival_interval = config["mean_holding_time"] / rho
        warmup = max(3000, int(math.ceil(5 * rho)))
        evaluated = max(5000, args.calib_evaluated)
        requests = _generate_trace(args.seed, arrival_interval, warmup, evaluated, config)
        result = run_episode(args.seed, requests, "ksp_ff_highest", agent_r, rankers, config, warmup_requests=warmup)
        calibration_results.append({
            "rho": rho,
            "arrival_interval": arrival_interval,
            "warmup_requests": warmup,
            "evaluated_requests": evaluated,
            "ksp_blocking_rate": result["blocking_rate"],
            "final_utilization": result["final_utilization"],
            "final_active_connections": result["final_active_connections"],
        })
        print(f"[calib] rho={rho:5.0f} arr={arrival_interval:.5f} blocking={result['blocking_rate']:.4%} util={result['final_utilization']:.4f}")

    return {"phase": "calibration", "config": config, "calibration": calibration_results}


def run_smoke(args: argparse.Namespace, arrival_interval: Optional[float] = None) -> Dict[str, Any]:
    config = _make_config(args)
    if arrival_interval is None:
        arrival_interval = args.arrival_interval
    requests = _generate_trace(args.seed, arrival_interval, args.warmup_requests, args.requests_per_episode, config)
    agent_r = load_ppo_r(PPO_R_CKPT, ModulationRegistry.from_profile(config["modulation_profile"]), args.device)
    rankers = {m: load_ranker(RANKER_CKPTS[m], args.device) for m in ("strict_v13", "v135_afterstate", "v135_afterstate_explicit")}

    results = {}
    for r_mode in R_MODES:
        results[r_mode] = run_episode(args.seed, requests, r_mode, agent_r, rankers, config, warmup_requests=args.warmup_requests, store_actions=args.store_actions)
    return {"phase": "smoke", "config": config, "arrival_interval": arrival_interval, "results": results}


def run_pilot(args: argparse.Namespace, arrival_interval: Optional[float] = None) -> Dict[str, Any]:
    config = _make_config(args)
    if arrival_interval is None:
        arrival_interval = args.arrival_interval
    agent_r = load_ppo_r(PPO_R_CKPT, ModulationRegistry.from_profile(config["modulation_profile"]), args.device)
    rankers = {m: load_ranker(RANKER_CKPTS[m], args.device) for m in ("strict_v13", "v135_afterstate", "v135_afterstate_explicit")}

    seeds = [int(s) for s in args.seeds.split(",")]
    results: Dict[str, Dict[int, Dict[str, Any]]] = {}
    for r_mode in R_MODES:
        results[r_mode] = {}
        for seed in seeds:
            requests = _generate_trace(seed, arrival_interval, args.warmup_requests, args.requests_per_episode, config)
            results[r_mode][seed] = run_episode(seed, requests, r_mode, agent_r, rankers, config, warmup_requests=args.warmup_requests, store_actions=args.store_actions)
    return {"phase": "pilot", "config": config, "arrival_interval": arrival_interval, "results": results}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", required=True, choices=["micro", "calibration", "smoke", "pilot"])
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=100)
    parser.add_argument("--k_paths", type=int, default=50)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--path_sort_strategy", default="hops")
    parser.add_argument("--block_sort_strategy", default="start_asc")
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=DEFAULT_SLOT_BW_HZ)
    parser.add_argument("--guard_band_fs", type=int, default=DEFAULT_GUARD_BAND_FS)
    parser.add_argument("--mean_holding_time", type=float, default=DEFAULT_MEAN_HOLDING_TIME)
    parser.add_argument("--bitrate_min_gbps", type=int, default=DEFAULT_BITRATE_MIN_GBPS)
    parser.add_argument("--bitrate_max_gbps", type=int, default=DEFAULT_BITRATE_MAX_GBPS)
    parser.add_argument("--arrival_interval", type=float, default=0.02)
    parser.add_argument("--warmup_requests", type=int, default=1000)
    parser.add_argument("--requests_per_episode", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=3030)
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--rho_list", type=float, nargs="+", default=[50, 100, 200, 400, 800, 1200, 1600])
    parser.add_argument("--calib_evaluated", type=int, default=5000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--store_actions", action="store_true", help="Store per-request selected actions (default false for smoke/pilot)")
    parser.add_argument("--output_dir", default=str(BASE_DIR))
    return parser


def main() -> int:
    _set_thread_env()
    parser = build_parser()
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    if args.phase == "micro":
        payload = run_micro(args)
    elif args.phase == "calibration":
        payload = run_calibration(args)
    elif args.phase == "smoke":
        payload = run_smoke(args)
    elif args.phase == "pilot":
        payload = run_pilot(args)
    else:
        raise ValueError(f"Unknown phase: {args.phase}")

    payload["elapsed_seconds"] = time.perf_counter() - t0
    payload["checkpoint_sha256s"] = {
        "ppo_r": _sha256_file(PPO_R_CKPT),
        **{k: _sha256_file(v) for k, v in RANKER_CKPTS.items()},
    }

    out_path = out_dir / f"{args.phase}.json"
    out_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"[optical-only] {args.phase} completed -> {out_path} ({payload['elapsed_seconds']:.1f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
