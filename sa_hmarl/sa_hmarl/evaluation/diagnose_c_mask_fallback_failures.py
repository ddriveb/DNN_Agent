"""Diagnose how much C-mask / fallback behavior explains blocking/server_overload.

This script distinguishes four failure paths:
  1. raw C-mask is empty → forced fallback
  2. raw C-mask non-empty, risk mask empty → policy fallback
  3. raw C-mask non-empty, policy selects a candidate, candidate succeeds
  4. raw C-mask non-empty, policy selects a candidate, candidate fails

For path 4 it further checks whether the selected candidate was MR-safe and
whether a safer alternative existed in the raw mask.

Outputs:
    * JSON: ``experiments/c_mask_fallback_diagnostic.json``
    * Markdown: ``experiments/c_mask_fallback_diagnostic.md``
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


MR_DIM_NAMES = [
    "server_available_ratio",
    "server_margin_after",
    "server_util_after",
    "deadline_margin",
    "spectrum_fit_margin",
    "multi_resource_safe_indicator",
]


@dataclass
class RequestRecord:
    """One row per request."""
    seed: int
    episode: int
    step: int
    req_id: int

    raw_mask_valid: int
    risk_mask_valid: int
    policy_selected: bool
    fallback: bool

    selected_split: int
    selected_server: int
    selected_idx: int

    selected_mr_features: np.ndarray = field(repr=False)
    selected_server_margin_raw: float
    selected_deadline_margin_raw: float
    selected_valid_r_actions: int
    selected_safe_indicator: float

    any_raw_safe_candidate: bool
    num_raw_safe_candidates: int
    best_raw_safe_server_margin: float

    success: bool
    reason: str
    blocked: bool
    server_overload: bool
    server_saturated: bool
    no_suitable_block: bool
    deadline_infeasible: bool
    delay_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "seed": self.seed,
            "episode": self.episode,
            "step": self.step,
            "req_id": self.req_id,
            "raw_mask_valid": self.raw_mask_valid,
            "risk_mask_valid": self.risk_mask_valid,
            "policy_selected": self.policy_selected,
            "fallback": self.fallback,
            "selected_split": self.selected_split,
            "selected_server": self.selected_server,
            "selected_idx": self.selected_idx,
            "selected_server_margin_raw": self.selected_server_margin_raw,
            "selected_deadline_margin_raw": self.selected_deadline_margin_raw,
            "selected_valid_r_actions": self.selected_valid_r_actions,
            "selected_safe_indicator": self.selected_safe_indicator,
            "any_raw_safe_candidate": self.any_raw_safe_candidate,
            "num_raw_safe_candidates": self.num_raw_safe_candidates,
            "best_raw_safe_server_margin": self.best_raw_safe_server_margin,
            "success": self.success,
            "reason": self.reason,
            "blocked": self.blocked,
            "server_overload": self.server_overload,
            "server_saturated": self.server_saturated,
            "no_suitable_block": self.no_suitable_block,
            "deadline_infeasible": self.deadline_infeasible,
            "delay_ms": self.delay_ms,
        }
        for i, name in enumerate(MR_DIM_NAMES):
            d[f"selected_{name}"] = float(self.selected_mr_features[i])
        return d


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
    if feature_mode in (
        "mean_field", "typed_mean_field", "gated_typed_mean_field",
        "fixed_blend_typed_mean_field", "candidate_mean_field",
        "candidate_mean_field_count_only",
    ):
        agent_c_kwargs["num_servers"] = num_servers
    if feature_mode == "fixed_blend_typed_mean_field":
        agent_c_kwargs["fixed_blend_alpha"] = ckpt_args.get("fixed_blend_alpha", 0.5)
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


def _mr_features_for_candidate(obs_c: Dict[str, Any], cand_idx: int) -> Tuple[np.ndarray, float, float, int]:
    """Return MR-rule features and raw margins for a single candidate."""
    feat_dict = obs_c["candidate_features"][cand_idx]

    from sa_hmarl.env.fs_demand import DEFAULT_PROP_SPEED_KM_S, DEFAULT_SETUP_TIME_S

    available = float(feat_dict.get("server_available_compute", 0.0))
    capacity = max(float(feat_dict.get("server_capacity", 1.0)), 1e-6)
    edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
    margin_raw = (available - edge_cost) / capacity

    deadline_ms = float(obs_c["request_features"]["deadline_ms"])
    local_ms = float(feat_dict.get("local_compute_ms", 0.0))
    edge_ms = float(feat_dict.get("edge_compute_ms", 0.0))
    min_path_km = float(feat_dict.get("server_min_path_km", 0.0))
    path_delay_est = (min_path_km / DEFAULT_PROP_SPEED_KM_S) * 1000.0 + DEFAULT_SETUP_TIME_S * 1000.0
    deadline_raw = (deadline_ms - (local_ms + edge_ms + path_delay_est)) / max(deadline_ms, 1e-6)

    diag = AgentC.compute_r_feasibility_diagnostics(obs_c, feat_dict)
    valid_r_actions = int(diag.get("valid_r_actions", 0))
    min_required_fs = diag.get("min_required_fs")
    best_block_size = diag.get("best_block_size", 0.0)

    server_available_ratio = float(np.clip(available / capacity, 0.0, 1.0))
    server_margin_after = float(0.5 * (np.clip(margin_raw, -1.0, 1.0) + 1.0))
    server_util_after = float(np.clip((capacity - (available - edge_cost)) / capacity, 0.0, 1.0))
    deadline_margin = float(0.5 * (np.clip(deadline_raw, -1.0, 1.0) + 1.0))

    if valid_r_actions > 0 and min_required_fs is not None and min_required_fs > 0:
        fit_pressure = min_required_fs / max(best_block_size, 1.0)
        spectrum_fit_margin = float(np.clip(1.0 - fit_pressure, 0.0, 1.0))
    else:
        spectrum_fit_margin = 0.0

    safe = (margin_raw >= -1e-6) and (deadline_raw >= -1e-6) and (valid_r_actions > 0)
    safe_indicator = 1.0 if safe else 0.0

    features = np.array([
        server_available_ratio, server_margin_after, server_util_after,
        deadline_margin, spectrum_fit_margin, safe_indicator,
    ], dtype=np.float32)
    features = np.clip(np.nan_to_num(features, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0)
    return features, margin_raw, deadline_raw, valid_r_actions


def _select_c_action_with_masks(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[Optional[int], np.ndarray, np.ndarray, int, int]:
    """Select action and return raw/risk masks and valid counts."""
    raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    raw_valid = int(raw_mask.sum())

    features, _ = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    risk_mask = raw_mask.copy()
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        risk_mask = apply_agent_c_risk_mask(
            obs_c, risk_mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    risk_valid = int(risk_mask.sum())

    action, _, _ = agent_c.select_from_features(features, risk_mask, deterministic=True)
    return action, raw_mask, risk_mask, raw_valid, risk_valid


def _select_r_action(agent_r: PPOAgentR, obs_r: Dict[str, Any], max_blocks: int) -> Tuple[int, int, int]:
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return 0, 0, 0
    return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), max_blocks)


def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg

    print("Loading frozen R backend...")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    print(f"Loading C policy for rollout: {args.rollout_c_checkpoint}")
    agent_c = _load_ppo_c(args.rollout_c_checkpoint, args.device)

    all_episodes: Dict[int, List[List[Any]]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps

    records: List[RequestRecord] = []
    t_start = time.time()

    for seed in seeds:
        for ep_idx, requests in enumerate(all_episodes[seed]):
            env = make_env(
                topology=args.topology, num_slots=args.num_slots,
                num_servers=args.num_servers, seed=42,
                slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)

            for step_idx, req in enumerate(requests):
                obs_c = build_agent_c_observation(env, req)
                action_idx_c, raw_mask, risk_mask, raw_valid, risk_valid = _select_c_action_with_masks(
                    agent_c, obs_c, env.net.num_slots
                )

                policy_selected = action_idx_c is not None
                fallback = not policy_selected

                # Compute MR features for all raw-mask-valid candidates
                safe_candidates = []
                best_safe_margin = -999.0
                for idx, allowed in enumerate(raw_mask):
                    if not allowed:
                        continue
                    features, margin_raw, _, _ = _mr_features_for_candidate(obs_c, idx)
                    if features[-1] > 0.5:
                        safe_candidates.append(idx)
                        if margin_raw > best_safe_margin:
                            best_safe_margin = margin_raw

                if policy_selected:
                    action_c = decode_agent_c_action(action_idx_c, args.num_servers)
                    selected_idx = action_idx_c
                    features, margin_raw, deadline_raw, valid_r = _mr_features_for_candidate(obs_c, selected_idx)
                else:
                    action_c = (0, 0)
                    selected_idx = -1
                    # Fallback candidate MR features
                    features = np.zeros(6, dtype=np.float32)
                    margin_raw = deadline_raw = 0.0
                    valid_r = 0

                split_id, server_id = action_c
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
                _, _, _, info = env.step(action_c, action_r)

                success = bool(info.get("success", False))
                reason = info.get("reason", "unknown")
                blocked = not success
                server_overload = reason == "server_overload"
                server_saturated = reason == "server_saturated"
                no_suitable_block = reason == "no_suitable_block"
                deadline_infeasible = reason == "deadline_infeasible"
                delay_ms = float(info.get("delay_ms", 0.0)) if success else 0.0

                records.append(RequestRecord(
                    seed=seed, episode=ep_idx, step=step_idx,
                    req_id=int(req.req_id),
                    raw_mask_valid=raw_valid,
                    risk_mask_valid=risk_valid,
                    policy_selected=policy_selected,
                    fallback=fallback,
                    selected_split=action_c[0],
                    selected_server=action_c[1],
                    selected_idx=selected_idx if selected_idx is not None else -1,
                    selected_mr_features=features,
                    selected_server_margin_raw=float(margin_raw),
                    selected_deadline_margin_raw=float(deadline_raw),
                    selected_valid_r_actions=valid_r,
                    selected_safe_indicator=float(features[-1]),
                    any_raw_safe_candidate=len(safe_candidates) > 0,
                    num_raw_safe_candidates=len(safe_candidates),
                    best_raw_safe_server_margin=float(best_safe_margin) if safe_candidates else 0.0,
                    success=success, reason=reason, blocked=blocked,
                    server_overload=server_overload,
                    server_saturated=server_saturated,
                    no_suitable_block=no_suitable_block,
                    deadline_infeasible=deadline_infeasible,
                    delay_ms=delay_ms,
                ))

    df = pd.DataFrame([r.to_dict() for r in records])
    n = max(len(df), 1)

    # Path decomposition
    path_counter = Counter()
    path_counter["total"] = len(df)
    path_counter["raw_mask_empty"] = int((df["raw_mask_valid"] == 0).sum())
    path_counter["raw_mask_nonempty_risk_empty"] = int(
        ((df["raw_mask_valid"] > 0) & (df["risk_mask_valid"] == 0)).sum()
    )
    path_counter["policy_selected"] = int(df["policy_selected"].sum())
    path_counter["fallback"] = int(df["fallback"].sum())

    # Outcomes by path
    outcome_by_path: Dict[str, Dict[str, Any]] = {}
    for name, mask in [
        ("raw_mask_empty", df["raw_mask_valid"] == 0),
        ("risk_mask_empty_fallback", (df["raw_mask_valid"] > 0) & (df["risk_mask_valid"] == 0)),
        ("policy_selected_success", df["policy_selected"] & df["success"]),
        ("policy_selected_fail", df["policy_selected"] & df["blocked"]),
    ]:
        sub = df[mask]
        m = max(len(sub), 1)
        outcome_by_path[name] = {
            "count": len(sub),
            "blocking_rate": float(sub["blocked"].mean()) if len(sub) else 0.0,
            "success_rate": float(sub["success"].mean()) if len(sub) else 0.0,
            "server_overload_rate": float(sub["server_overload"].mean()) if len(sub) else 0.0,
            "server_saturated_rate": float(sub["server_saturated"].mean()) if len(sub) else 0.0,
            "no_suitable_block_rate": float(sub["no_suitable_block"].mean()) if len(sub) else 0.0,
            "deadline_infeasible_rate": float(sub["deadline_infeasible"].mean()) if len(sub) else 0.0,
        }

    # Failure decomposition: how much of total blocking comes from each path
    total_blocked = int(df["blocked"].sum())
    blocked_breakdown = {
        "raw_mask_empty": int(((df["raw_mask_valid"] == 0) & df["blocked"]).sum()),
        "risk_mask_empty_fallback": int(
            (((df["raw_mask_valid"] > 0) & (df["risk_mask_valid"] == 0)) & df["blocked"]).sum()
        ),
        "policy_selected_then_failed": int((df["policy_selected"] & df["blocked"]).sum()),
    }

    # MR-rule signal on policy-selected failures
    df_fail = df[df["policy_selected"] & df["blocked"]].copy()
    mr_signal = {}
    mr_signal["policy_fail_selected_safe"] = int((df_fail["selected_multi_resource_safe_indicator"] > 0.5).sum())
    mr_signal["policy_fail_selected_unsafe"] = int((df_fail["selected_multi_resource_safe_indicator"] <= 0.5).sum())
    mr_signal["policy_fail_had_safer_alternative"] = int(
        (df_fail["policy_selected"] & df_fail["any_raw_safe_candidate"]).sum()
    )
    mr_signal["policy_fail_no_safer_alternative"] = int(
        (df_fail["policy_selected"] & ~df_fail["any_raw_safe_candidate"]).sum()
    )

    # Success vs blocked by selected safe indicator
    safe_indicator_stats = {}
    for val in [0, 1]:
        sub = df[df["policy_selected"] & (df["selected_multi_resource_safe_indicator"] == val)]
        safe_indicator_stats[str(val)] = {
            "count": len(sub),
            "success_rate": float(sub["success"].mean()) if len(sub) else 0.0,
            "blocking_rate": float(sub["blocked"].mean()) if len(sub) else 0.0,
            "server_overload_rate": float(sub["server_overload"].mean()) if len(sub) else 0.0,
        }

    # Bin analysis on policy-selected requests
    def _qbin(series: pd.Series, n: int = 5) -> pd.Series:
        labels = [f"bin_{i}" for i in range(n)]
        try:
            return pd.qcut(series.rank(method="first"), q=n, labels=labels)
        except ValueError:
            return pd.qcut(series, q=n, labels=labels, duplicates="drop")

    def _bin(df_sub: pd.DataFrame, x: str, y: str, n: int = 5) -> List[Dict[str, Any]]:
        if df_sub.empty:
            return []
        df_sub = df_sub.copy()
        df_sub["_bin"] = _qbin(df_sub[x], n)
        out = df_sub.groupby("_bin", observed=False).agg(
            count=(x, "size"),
            mean_x=(x, "mean"),
            mean_y=(y, "mean"),
        ).reset_index().rename(columns={"_bin": "bin"}).to_dict(orient="records")
        df_sub.drop(columns=["_bin"], inplace=True, errors="ignore")
        return out

    df_sel = df[df["policy_selected"]].copy()
    bin_analysis = {
        "selected_server_margin_after_vs_server_overload": _bin(df_sel, "selected_server_margin_after", "server_overload"),
        "selected_server_margin_after_vs_blocked": _bin(df_sel, "selected_server_margin_after", "blocked"),
        "selected_deadline_margin_vs_deadline_infeasible": _bin(df_sel, "selected_deadline_margin", "deadline_infeasible"),
        "selected_spectrum_fit_margin_vs_no_suitable_block": _bin(df_sel, "selected_spectrum_fit_margin", "no_suitable_block"),
    }

    # Distribution of raw_mask_valid count
    mask_dist = df["raw_mask_valid"].value_counts().sort_index().to_dict()
    mask_dist = {int(k): int(v) for k, v in mask_dist.items()}

    report = {
        "config": {
            "topology": args.topology,
            "num_slots": args.num_slots,
            "num_servers": args.num_servers,
            "k_paths": args.k_paths,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "split_profile": args.split_profile,
            "seeds": seeds,
            "episodes": args.episodes,
            "requests_per_episode": args.requests_per_episode,
            "rollout_c_checkpoint": args.rollout_c_checkpoint,
            "agent_r_checkpoint": args.agent_r_checkpoint,
        },
        "total_requests": len(df),
        "total_blocked": total_blocked,
        "path_counts": dict(path_counter),
        "outcome_by_path": outcome_by_path,
        "blocked_breakdown": blocked_breakdown,
        "mr_rule_signal_on_policy_failures": mr_signal,
        "safe_indicator_stats": safe_indicator_stats,
        "bin_analysis": bin_analysis,
        "raw_mask_valid_distribution": mask_dist,
        "elapsed_seconds": time.time() - t_start,
    }
    return report


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# C-Mask / Fallback Failure Diagnostic")
    lines.append("")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    cfg = report["config"]
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Topology:** {cfg['topology']}")
    lines.append(f"- **Slots:** {cfg['num_slots']}, **Servers:** {cfg['num_servers']}, **k_paths:** {cfg['k_paths']}")
    lines.append(f"- **max_blocks:** {cfg['max_blocks']}, **block_sort:** {cfg['block_sort_strategy']}")
    lines.append(f"- **split_profile:** {cfg['split_profile']}")
    lines.append(f"- **Seeds:** {cfg['seeds']}, **Episodes/seed:** {cfg['episodes']}, **Requests/episode:** {cfg['requests_per_episode']}")
    lines.append(f"- **Rollout C checkpoint:** `{cfg['rollout_c_checkpoint']}`")
    lines.append("")

    lines.append("## Path Decomposition")
    lines.append("")
    lines.append(f"- Total requests: **{report['total_requests']}**")
    lines.append(f"- Total blocked: **{report['total_blocked']}**")
    lines.append("")
    lines.append("| Path | Count | % of total |")
    lines.append("|---|---:|---:|")
    for k, v in report["path_counts"].items():
        if k == "total":
            continue
        pct = 100.0 * v / max(report['total_requests'], 1)
        lines.append(f"| {k} | {v} | {pct:.2f}% |")
    lines.append("")

    lines.append("## Outcome by Path")
    lines.append("")
    lines.append("| Path | Count | Blocking | Success | SrvOvld | SrvSat | NoBlock | DeadlineFail |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for name, s in report["outcome_by_path"].items():
        lines.append(
            f"| {name} | {s['count']} | {s['blocking_rate']:.4f} | {s['success_rate']:.4f} | "
            f"{s['server_overload_rate']:.4f} | {s['server_saturated_rate']:.4f} | "
            f"{s['no_suitable_block_rate']:.4f} | {s['deadline_infeasible_rate']:.4f} |"
        )
    lines.append("")

    lines.append("## Blocking Breakdown")
    lines.append("")
    lines.append("| Source | Blocked | % of total blocked |")
    lines.append("|---|---:|---:|")
    for k, v in report["blocked_breakdown"].items():
        pct = 100.0 * v / max(report['total_blocked'], 1)
        lines.append(f"| {k} | {v} | {pct:.2f}% |")
    lines.append("")

    lines.append("## MR-Rule Signal on Policy-Selected Failures")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|---|---:|")
    for k, v in report["mr_rule_signal_on_policy_failures"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Safe-Indicator Discrimination (policy-selected only)")
    lines.append("")
    lines.append("| safe_indicator | Count | Success | Blocking | Server Overload |")
    lines.append("|---:|---:|---:|---:|---:|")
    for val, s in report["safe_indicator_stats"].items():
        lines.append(
            f"| {val} | {s['count']} | {s['success_rate']:.4f} | {s['blocking_rate']:.4f} | "
            f"{s['server_overload_rate']:.4f} |"
        )
    lines.append("")

    lines.append("## Bin Analyses (policy-selected requests)")
    lines.append("")
    for title, rows in report["bin_analysis"].items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Bin | Count | Mean Predictor | Mean Outcome |")
        lines.append("|-----|------:|---------------:|-------------:|")
        for row in rows:
            lines.append(f"| {row['bin']} | {row['count']} | {row['mean_x']:.4f} | {row['mean_y']:.4f} |")
        lines.append("")

    lines.append("## Raw C-Mask Valid Count Distribution")
    lines.append("")
    lines.append("| #valid_candidates | #requests |")
    lines.append("|---:|---:|")
    for k, v in sorted(report["raw_mask_valid_distribution"].items()):
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Elapsed: {report['elapsed_seconds']:.1f}s*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rollout_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str, default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="sa_hmarl/experiments")
    args = parser.parse_args()

    report = _run_diagnostic(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "c_mask_fallback_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    md_path = output_dir / "c_mask_fallback_diagnostic.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))
    print(f"Markdown saved: {md_path}")

    print(f"\nElapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
