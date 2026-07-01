"""Diagnose whether a spectrum feasibility field predicts raw_mask_empty / blocking.

Compares multiple Agent-C checkpoints by computing, *before* each checkpoint selects
an action, the same request-level spectrum-feasibility statistics on a shared env state:
  K_C_valid  = # of (split, server) candidates that are C-mask feasible
  K_R_total  = sum of valid R actions over all candidates
  NR_hist    = histogram of per-candidate valid-R-action counts
  Phi_spec   = log(1 + K_C_valid) + alpha * log(1 + K_R_total)
  raw_mask_empty / raw_mask_valid = from the raw obs_c mask before any checkpoint

To make checkpoints comparable, the first checkpoint in the list is used as a reference
rollout; its env snapshots before each action are deep-copied and replayed for the other
checkpoints, so all checkpoints see the identical pre-decision state.

Outputs:
    * JSON: ``experiments/spectrum_feasibility_field_diagnostic.json``
    * Markdown: ``sa_hmarl/experiments/spectrum_feasibility_field_diagnostic.md``
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
from scipy import stats

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


@dataclass
class RequestRecord:
    seed: int
    episode: int
    step: int
    req_id: int

    checkpoint: str

    k_c_valid_before: int
    k_r_total_before: int
    phi_spec_before: float

    raw_mask_empty_before: bool
    raw_mask_valid_before: int
    total_candidates: int

    nr_0: int
    nr_1_2: int
    nr_3_5: int
    nr_gt5: int

    success: bool
    blocked: bool
    reason: str
    server_overload: bool
    no_suitable_block: bool
    deadline_infeasible: bool
    delay_ms: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "step": self.step,
            "req_id": self.req_id,
            "checkpoint": self.checkpoint,
            "k_c_valid_before": self.k_c_valid_before,
            "k_r_total_before": self.k_r_total_before,
            "phi_spec_before": self.phi_spec_before,
            "raw_mask_empty_before": self.raw_mask_empty_before,
            "raw_mask_valid_before": self.raw_mask_valid_before,
            "total_candidates": self.total_candidates,
            "nr_0": self.nr_0,
            "nr_1_2": self.nr_1_2,
            "nr_3_5": self.nr_3_5,
            "nr_gt5": self.nr_gt5,
            "success": self.success,
            "blocked": self.blocked,
            "reason": self.reason,
            "server_overload": self.server_overload,
            "no_suitable_block": self.no_suitable_block,
            "deadline_infeasible": self.deadline_infeasible,
            "delay_ms": self.delay_ms,
        }


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


def _select_c_action(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    features, mask = agent_c.build_action_features(obs_c)
    raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
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
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    if action is None:
        action = 0
    return action, raw_mask, mask


def _select_r_action(agent_r: PPOAgentR, obs_r: Dict[str, Any], max_blocks: int) -> Tuple[int, int, int]:
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return 0, 0, 0
    return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), max_blocks)


def _compute_spectrum_field(obs_c: Dict[str, Any], util_threshold: float = 0.95) -> Tuple[int, int, Dict[str, int]]:
    """Compute K_C_valid, K_R_total, and NR_hist for one request observation."""
    num_servers = len(obs_c["server_utilizations"])
    feasible_counts = obs_c["feasible_counts"]

    k_c_valid = 0
    k_r_total = 0
    nr_0 = nr_1_2 = nr_3_5 = nr_gt5 = 0

    for split_id, server_list in enumerate(feasible_counts):
        for server_id, count in enumerate(server_list):
            cand_idx = split_id * num_servers + server_id
            feat_dict = obs_c["candidate_features"][cand_idx]

            # Compute feasibility
            available = float(feat_dict.get("server_available_compute", 0.0))
            edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
            edge_ms = feat_dict.get("edge_compute_ms", 0.0)
            compute_ok = (edge_cost <= available) and np.isfinite(edge_ms) and edge_ms != float('inf')

            # Utilization feasibility
            util = float(obs_c["server_utilizations"][server_id])
            util_ok = util <= util_threshold

            # Valid R actions
            diag = AgentC.compute_r_feasibility_diagnostics(obs_c, feat_dict)
            n_r = int(diag.get("valid_r_actions", 0))

            if compute_ok and util_ok and n_r > 0:
                k_c_valid += 1
            k_r_total += n_r

            if n_r == 0:
                nr_0 += 1
            elif n_r <= 2:
                nr_1_2 += 1
            elif n_r <= 5:
                nr_3_5 += 1
            else:
                nr_gt5 += 1

    return k_c_valid, k_r_total, {"0": nr_0, "1-2": nr_1_2, "3-5": nr_3_5, ">5": nr_gt5}


def _run_reference_rollout(
    ckpt_label: str,
    ckpt_path: str,
    agent_r: PPOAgentR,
    all_episodes: Dict[int, List[List[Any]]],
    args: argparse.Namespace,
) -> Tuple[List[RequestRecord], Dict[int, List[List[Any]]]]:
    """Run the reference checkpoint rollout and return records plus env snapshots."""
    print(f"\n[Reference rollout] Loading C policy: {ckpt_label} -> {ckpt_path}")
    agent_c = _load_ppo_c(ckpt_path, args.device)

    records: List[RequestRecord] = []
    snapshots: Dict[int, List[List[Any]]] = {}

    for seed in sorted(all_episodes.keys()):
        snapshots[seed] = []
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

            ep_snapshots = []
            for step_idx, req in enumerate(requests):
                # Snapshot state BEFORE any checkpoint acts
                ep_snapshots.append(copy.deepcopy(env))

                obs_c = build_agent_c_observation(env, req)

                k_c_valid, k_r_total, nr_hist = _compute_spectrum_field(
                    obs_c, util_threshold=args.util_threshold
                )
                phi_spec = np.log1p(k_c_valid) + args.alpha * np.log1p(k_r_total)

                raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                raw_valid = int(raw_mask.sum())
                raw_empty = raw_valid == 0

                action_idx_c, _, _ = _select_c_action(agent_c, obs_c, env.net.num_slots)
                action_c = decode_agent_c_action(action_idx_c, args.num_servers)
                split_id, server_id = action_c

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
                _, _, _, info = env.step(action_c, action_r)

                success = bool(info.get("success", False))
                reason = info.get("reason", "unknown")
                blocked = not success

                records.append(RequestRecord(
                    seed=seed, episode=ep_idx, step=step_idx,
                    req_id=int(req.req_id),
                    checkpoint=ckpt_label,
                    k_c_valid_before=k_c_valid,
                    k_r_total_before=k_r_total,
                    phi_spec_before=float(phi_spec),
                    raw_mask_empty_before=raw_empty,
                    raw_mask_valid_before=raw_valid,
                    total_candidates=len(obs_c["candidate_features"]),
                    nr_0=nr_hist["0"],
                    nr_1_2=nr_hist["1-2"],
                    nr_3_5=nr_hist["3-5"],
                    nr_gt5=nr_hist[">5"],
                    success=success,
                    blocked=blocked,
                    reason=reason,
                    server_overload=(reason == "server_overload"),
                    no_suitable_block=(reason == "no_suitable_block"),
                    deadline_infeasible=(reason == "deadline_infeasible"),
                    delay_ms=float(info.get("delay_ms", 0.0)) if success else 0.0,
                ))
            snapshots[seed].append(ep_snapshots)

    return records, snapshots


def _run_replay_rollout(
    ckpt_label: str,
    ckpt_path: str,
    agent_r: PPOAgentR,
    all_episodes: Dict[int, List[List[Any]]],
    reference_snapshots: Dict[int, List[List[Any]]],
    before_metrics: Dict[Tuple[int, int, int], Dict[str, Any]],
    args: argparse.Namespace,
) -> List[RequestRecord]:
    """Replay a checkpoint on the reference env snapshots and record outcomes."""
    print(f"\n[Replay rollout] Loading C policy: {ckpt_label} -> {ckpt_path}")
    agent_c = _load_ppo_c(ckpt_path, args.device)

    records: List[RequestRecord] = []

    for seed in sorted(all_episodes.keys()):
        for ep_idx, requests in enumerate(all_episodes[seed]):
            ep_snapshots = reference_snapshots[seed][ep_idx]
            for step_idx, req in enumerate(requests):
                env = ep_snapshots[step_idx]
                key = (seed, ep_idx, step_idx)
                bm = before_metrics[key]

                obs_c = build_agent_c_observation(env, req)

                action_idx_c, _, _ = _select_c_action(agent_c, obs_c, env.net.num_slots)
                action_c = decode_agent_c_action(action_idx_c, args.num_servers)
                split_id, server_id = action_c

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
                _, _, _, info = env.step(action_c, action_r)

                success = bool(info.get("success", False))
                reason = info.get("reason", "unknown")

                records.append(RequestRecord(
                    seed=seed, episode=ep_idx, step=step_idx,
                    req_id=int(req.req_id),
                    checkpoint=ckpt_label,
                    k_c_valid_before=bm["k_c_valid_before"],
                    k_r_total_before=bm["k_r_total_before"],
                    phi_spec_before=bm["phi_spec_before"],
                    raw_mask_empty_before=bm["raw_mask_empty_before"],
                    raw_mask_valid_before=bm["raw_mask_valid_before"],
                    total_candidates=bm["total_candidates"],
                    nr_0=bm["nr_0"],
                    nr_1_2=bm["nr_1_2"],
                    nr_3_5=bm["nr_3_5"],
                    nr_gt5=bm["nr_gt5"],
                    success=success,
                    blocked=not success,
                    reason=reason,
                    server_overload=(reason == "server_overload"),
                    no_suitable_block=(reason == "no_suitable_block"),
                    deadline_infeasible=(reason == "deadline_infeasible"),
                    delay_ms=float(info.get("delay_ms", 0.0)) if success else 0.0,
                ))
    return records


def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    ckpt_paths = [p.strip() for p in args.rollout_c_checkpoints.split(",")]
    ckpt_labels = [l.strip() for l in args.checkpoint_labels.split(",")]
    if len(ckpt_labels) != len(ckpt_paths):
        raise ValueError(
            f"Number of checkpoint labels ({len(ckpt_labels)}) must match "
            f"number of checkpoint paths ({len(ckpt_paths)})."
        )
    if len(ckpt_paths) == 0:
        raise ValueError("At least one checkpoint must be provided.")

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

    # Generate requests once; reuse across checkpoints for fair comparison
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

    t_start = time.time()
    all_records: List[RequestRecord] = []

    # Reference rollout: also collects env snapshots and before-metrics
    ref_label, ref_path = ckpt_labels[0], ckpt_paths[0]
    ref_records, ref_snapshots = _run_reference_rollout(
        ref_label, ref_path, agent_r, all_episodes, args
    )
    all_records.extend(ref_records)

    # Extract before-metrics from reference records for replay rollouts
    before_metrics = {}
    for r in ref_records:
        before_metrics[(r.seed, r.episode, r.step)] = {
            "k_c_valid_before": r.k_c_valid_before,
            "k_r_total_before": r.k_r_total_before,
            "phi_spec_before": r.phi_spec_before,
            "raw_mask_empty_before": r.raw_mask_empty_before,
            "raw_mask_valid_before": r.raw_mask_valid_before,
            "total_candidates": r.total_candidates,
            "nr_0": r.nr_0,
            "nr_1_2": r.nr_1_2,
            "nr_3_5": r.nr_3_5,
            "nr_gt5": r.nr_gt5,
        }

    # Replay remaining checkpoints on the reference snapshots
    for label, path in zip(ckpt_labels[1:], ckpt_paths[1:]):
        all_records.extend(_run_replay_rollout(
            label, path, agent_r, all_episodes, ref_snapshots, before_metrics, args
        ))

    df = pd.DataFrame([r.to_dict() for r in all_records])

    # Sanity: per-request before-metrics should now be identical across checkpoints
    before_consistent = True
    if len(ckpt_paths) > 1:
        grouped = df.groupby(["seed", "episode", "step"])
        for col in ["k_c_valid_before", "k_r_total_before", "phi_spec_before",
                    "raw_mask_empty_before", "raw_mask_valid_before"]:
            if not grouped[col].nunique().eq(1).all():
                before_consistent = False
                print(f"WARNING: before-metric '{col}' differs across checkpoints!")

    # Overall per-checkpoint stats
    overall = []
    for label in ckpt_labels:
        sub = df[df["checkpoint"] == label]
        overall.append({
            "checkpoint": label,
            "requests": len(sub),
            "raw_mask_empty": int(sub["raw_mask_empty_before"].sum()),
            "raw_mask_empty_rate": float(sub["raw_mask_empty_before"].mean()),
            "blocked": int(sub["blocked"].sum()),
            "blocking_rate": float(sub["blocked"].mean()),
            "server_overload_rate": float(sub["server_overload"].mean()),
            "no_suitable_block_rate": float(sub["no_suitable_block"].mean()),
            "deadline_infeasible_rate": float(sub["deadline_infeasible"].mean()),
            "mean_delay_ms": float(sub["delay_ms"].mean()),
        })

    # Correlations (before-metrics are identical, so any checkpoint works)
    df_first = df[df["checkpoint"] == ckpt_labels[0]].copy()

    def _corr(x: pd.Series, y: pd.Series) -> Tuple[float, float]:
        try:
            with np.errstate(invalid="ignore"):
                r, p = stats.pearsonr(x, y)
                return float(r), float(p)
        except Exception:
            return float("nan"), float("nan")

    correlations = {
        "phi_spec_vs_raw_empty": _corr(df_first["phi_spec_before"], df_first["raw_mask_empty_before"].astype(float)),
        "phi_spec_vs_blocked": _corr(df_first["phi_spec_before"], df_first["blocked"].astype(float)),
        "k_c_valid_vs_raw_empty": _corr(df_first["k_c_valid_before"], df_first["raw_mask_empty_before"].astype(float)),
        "k_r_total_vs_raw_empty": _corr(df_first["k_r_total_before"], df_first["raw_mask_empty_before"].astype(float)),
    }

    # Bin analysis for Phi_spec (per-checkpoint outcomes)
    def _qbin(series: pd.Series, n: int = 5) -> pd.Series:
        labels = [f"bin_{i}" for i in range(n)]
        try:
            return pd.qcut(series.rank(method="first"), q=n, labels=labels)
        except ValueError:
            return pd.qcut(series, q=n, labels=labels, duplicates="drop")

    df_first["_phi_bin"] = _qbin(df_first["phi_spec_before"], n=args.n_bins)
    bin_map = df_first.set_index(["seed", "episode", "step"])["_phi_bin"].to_dict()
    df["_phi_bin"] = df.set_index(["seed", "episode", "step"]).index.map(bin_map.get)

    phi_bin_rows = []
    for bin_name in sorted(df_first["_phi_bin"].unique(), key=lambda x: str(x)):
        sub_first = df_first[df_first["_phi_bin"] == bin_name]
        base_info = {
            "count": len(sub_first),
            "mean_phi": float(sub_first["phi_spec_before"].mean()),
            "mean_k_c_valid": float(sub_first["k_c_valid_before"].mean()),
            "mean_k_r_total": float(sub_first["k_r_total_before"].mean()),
            "raw_empty_rate": float(sub_first["raw_mask_empty_before"].mean()),
        }
        per_ckpt = {}
        for label in ckpt_labels:
            sub = df[(df["checkpoint"] == label) & (df["_phi_bin"] == bin_name)]
            per_ckpt[label] = {
                "blocking_rate": float(sub["blocked"].mean()) if len(sub) else None,
                "server_overload_rate": float(sub["server_overload"].mean()) if len(sub) else None,
                "no_suitable_block_rate": float(sub["no_suitable_block"].mean()) if len(sub) else None,
            }
        phi_bin_rows.append({"bin": str(bin_name), **base_info, "per_checkpoint": per_ckpt})

    df_first.drop(columns=["_phi_bin"], inplace=True, errors="ignore")
    df.drop(columns=["_phi_bin"], inplace=True, errors="ignore")

    # K_C_valid analysis per checkpoint
    k_c_analysis = []
    for k_c_val in sorted(df_first["k_c_valid_before"].unique()):
        sub_first = df_first[df_first["k_c_valid_before"] == k_c_val]
        row = {
            "k_c_valid_before": int(k_c_val),
            "count": len(sub_first),
            "raw_empty_rate": float(sub_first["raw_mask_empty_before"].mean()),
        }
        per_ckpt = {}
        for label in ckpt_labels:
            sub = df[(df["checkpoint"] == label) & (df["k_c_valid_before"] == k_c_val)]
            per_ckpt[label] = {
                "blocking_rate": float(sub["blocked"].mean()) if len(sub) else None,
                "server_overload_rate": float(sub["server_overload"].mean()) if len(sub) else None,
                "no_suitable_block_rate": float(sub["no_suitable_block"].mean()) if len(sub) else None,
            }
        k_c_analysis.append({**row, "per_checkpoint": per_ckpt})

    # Aggregate NR histogram (same across checkpoints)
    total_nr_hist = {
        "0": int(df_first["nr_0"].sum()),
        "1-2": int(df_first["nr_1_2"].sum()),
        "3-5": int(df_first["nr_3_5"].sum()),
        ">5": int(df_first["nr_gt5"].sum()),
    }

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
            "rollout_c_checkpoints": ckpt_paths,
            "checkpoint_labels": ckpt_labels,
            "reference_checkpoint": ref_label,
            "agent_r_checkpoint": args.agent_r_checkpoint,
            "alpha": args.alpha,
            "util_threshold": args.util_threshold,
            "n_bins": args.n_bins,
        },
        "before_metrics_consistent_across_checkpoints": before_consistent,
        "total_requests_per_checkpoint": len(df_first),
        "overall_per_checkpoint": overall,
        "correlations": {k: {"r": v[0], "pvalue": v[1]} for k, v in correlations.items()},
        "phi_spec_bin_analysis": phi_bin_rows,
        "k_c_valid_analysis": k_c_analysis,
        "total_nr_histogram": total_nr_hist,
        "elapsed_seconds": time.time() - t_start,
    }
    return report


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    labels = report["config"]["checkpoint_labels"]

    lines.append("# Spectrum Feasibility Field Diagnostic (Multi-Checkpoint, Shared State)")
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
    lines.append(f"- **Alpha:** {cfg['alpha']}")
    lines.append(f"- **Reference checkpoint (state provider):** {cfg['reference_checkpoint']}")
    lines.append(f"- **Before-metrics consistent across checkpoints:** {report['before_metrics_consistent_across_checkpoints']}")
    lines.append("")
    lines.append("### Checkpoints")
    for lbl, pth in zip(labels, cfg["rollout_c_checkpoints"]):
        lines.append(f"- **{lbl}:** `{pth}`")
    lines.append("")

    lines.append("## Overall Per-Checkpoint")
    lines.append("")
    header = "| Checkpoint | Requests | raw_empty | raw_empty_rate | blocked | blocking_rate | srv_ovld | no_block | deadline | mean_delay_ms |"
    lines.append(header)
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")
    for row in report["overall_per_checkpoint"]:
        lines.append(
            f"| {row['checkpoint']} | {row['requests']} | {row['raw_mask_empty']} | "
            f"{row['raw_mask_empty_rate']:.4f} | {row['blocked']} | {row['blocking_rate']:.4f} | "
            f"{row['server_overload_rate']:.4f} | {row['no_suitable_block_rate']:.4f} | "
            f"{row['deadline_infeasible_rate']:.4f} | {row['mean_delay_ms']:.4f} |"
        )
    lines.append("")

    lines.append("## Correlations (before-metrics vs outcomes)")
    lines.append("")
    lines.append("| Pair | r | p-value |")
    lines.append("|---|---:|---:|")
    for name, vals in report["correlations"].items():
        r = vals['r'] if vals['r'] is not None else float('nan')
        p = vals['pvalue'] if vals['pvalue'] is not None else float('nan')
        lines.append(f"| {name} | {r:.4f} | {p:.2e} |")
    lines.append("")

    lines.append("## Phi_spec Bin Analysis")
    lines.append("")
    ckpt_header = " | ".join([f"{lbl}_block" for lbl in labels])
    lines.append(f"| Bin | Count | Mean Phi | Mean K_C | Mean K_R | RawEmpty | {ckpt_header} |")
    sep_parts = ["---:"] * (5 + len(labels))
    lines.append("|" + "|".join([""] + sep_parts) + "|")
    for row in report["phi_spec_bin_analysis"]:
        per = row["per_checkpoint"]
        ckpt_vals = " | ".join([f"{per[lbl]['blocking_rate']:.4f}" for lbl in labels])
        lines.append(
            f"| {row['bin']} | {row['count']} | {row['mean_phi']:.4f} | "
            f"{row['mean_k_c_valid']:.2f} | {row['mean_k_r_total']:.2f} | "
            f"{row['raw_empty_rate']:.4f} | {ckpt_vals} |"
        )
    lines.append("")

    lines.append("## K_C_valid_before Analysis")
    lines.append("")
    ckpt_header2 = " | ".join([f"{lbl}_block" for lbl in labels])
    lines.append(f"| K_C_valid | Count | RawEmpty | {ckpt_header2} |")
    lines.append("|---:|---:|---:|" + "|".join(["---:" for _ in labels]) + "|")
    for row in report["k_c_valid_analysis"]:
        per = row["per_checkpoint"]
        ckpt_vals = " | ".join([f"{per[lbl]['blocking_rate']:.4f}" for lbl in labels])
        lines.append(
            f"| {row['k_c_valid_before']} | {row['count']} | {row['raw_empty_rate']:.4f} | {ckpt_vals} |"
        )
    lines.append("")

    lines.append("## Total NR Histogram (all candidates across all requests)")
    lines.append("")
    lines.append("| NR bucket | Count |")
    lines.append("|---|---:|")
    for k, v in report["total_nr_histogram"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Elapsed: {report['elapsed_seconds']:.1f}s*")
    return "\n".join(lines)


def _clean_for_json(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _clean_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_clean_for_json(v) for v in obj]
    if isinstance(obj, float):
        if np.isnan(obj) or np.isinf(obj):
            return None
    return obj


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rollout_c_checkpoints", type=str,
        default=(
            "sa_hmarl/checkpoints/agent_c_default_s123_s20_r80_best.pt,"
            "sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt,"
            "sa_hmarl/checkpoints/agent_c_r_feasibility_safe_s123_s20_r80_best.pt"
        ),
    )
    parser.add_argument(
        "--checkpoint_labels", type=str,
        default="default,r_feasibility,r_feasibility_safe",
    )
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
    parser.add_argument("--alpha", type=float, default=0.3,
                        help="Weight for K_R_total term in Phi_spec.")
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="sa_hmarl/experiments")
    args = parser.parse_args()

    report = _run_diagnostic(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "spectrum_feasibility_field_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(_clean_for_json(report), f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    md_path = output_dir / "spectrum_feasibility_field_diagnostic.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))
    print(f"Markdown saved: {md_path}")

    print(f"\nElapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
