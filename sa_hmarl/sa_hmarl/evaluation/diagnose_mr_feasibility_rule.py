"""Diagnostic for the `mr_feasibility_rule` Agent-C feature mode.

Does NOT train. It:
  1. Builds default / r_feasibility / mr_feasibility_rule features for every candidate.
  2. Checks shape, finiteness, and [0,1] range of the new 6 rule dims.
  3. Rolls out with a frozen r_feasibility Agent-C + frozen PPO-R.
  4. Reports relationships between the rule features and request outcomes.

Outputs:
    * JSON: ``experiments/mr_feasibility_rule_diagnostic.json``
    * Markdown: ``experiments/mr_feasibility_rule_diagnostic.md``
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.fs_demand import DEFAULT_PROP_SPEED_KM_S, DEFAULT_SETUP_TIME_S
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
class CandidateRecord:
    """One row per (request, candidate)."""
    seed: int
    episode: int
    step: int
    req_id: int
    candidate_idx: int
    split_id: int
    server_id: int
    selected: bool
    mr_features: np.ndarray = field(repr=False)
    # Raw values for diagnostics
    server_margin_raw: float
    deadline_margin_raw: float
    valid_r_actions: int
    success: bool
    reason: str
    blocked: bool
    server_overload: bool
    no_suitable_block: bool
    deadline_infeasible: bool
    delay_ms: float

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "seed": self.seed,
            "episode": self.episode,
            "step": self.step,
            "req_id": self.req_id,
            "candidate_idx": self.candidate_idx,
            "split_id": self.split_id,
            "server_id": self.server_id,
            "selected": self.selected,
            "server_margin_raw": self.server_margin_raw,
            "deadline_margin_raw": self.deadline_margin_raw,
            "valid_r_actions": self.valid_r_actions,
            "success": self.success,
            "reason": self.reason,
            "blocked": self.blocked,
            "server_overload": self.server_overload,
            "no_suitable_block": self.no_suitable_block,
            "deadline_infeasible": self.deadline_infeasible,
            "delay_ms": self.delay_ms,
        }
        for i, name in enumerate(MR_DIM_NAMES):
            d[name] = float(self.mr_features[i])
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


def _select_c_action(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[Optional[int], int]:
    features, mask = agent_c.build_action_features(obs_c)
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
    n_valid = int(mask.sum())
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action, n_valid


def _select_r_action(agent_r: PPOAgentR, obs_r: Dict[str, Any], max_blocks: int) -> Tuple[int, int, int]:
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return 0, 0, 0
    return decode_agent_r_action(action_idx, len(obs_r["mod_names"]), max_blocks)


def _bin_by_quantile(series: pd.Series, n_bins: int = 5) -> pd.Series:
    labels = [f"bin_{i}" for i in range(n_bins)]
    try:
        return pd.qcut(series.rank(method="first"), q=n_bins, labels=labels)
    except ValueError:
        return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")


def _bin_analysis(df: pd.DataFrame, x_col: str, outcome: str, n_bins: int = 5) -> List[Dict[str, Any]]:
    df["_bin"] = _bin_by_quantile(df[x_col], n_bins=n_bins)
    grouped = df.groupby("_bin", observed=False).agg(
        count=(x_col, "size"),
        mean_x=(x_col, "mean"),
        mean_outcome=(outcome, "mean"),
    ).reset_index().rename(columns={"_bin": "bin"})
    df.drop(columns=["_bin"], inplace=True, errors="ignore")
    return grouped.to_dict(orient="records")


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
    agent_c_rollout = _load_ppo_c(args.rollout_c_checkpoint, args.device)

    # Feature-building agents (no weights needed; only feature shapes matter).
    print("Instantiating feature-mode agents for shape checks...")
    agent_default = PPOAgentC(input_dim=17, hidden_dims=(8, 8), device="cpu",
                              feature_mode="default", num_servers=args.num_servers)
    agent_r_feas = PPOAgentC(input_dim=17, hidden_dims=(8, 8), device="cpu",
                             feature_mode="r_feasibility", num_servers=args.num_servers)
    agent_mr = PPOAgentC(input_dim=17, hidden_dims=(8, 8), device="cpu",
                         feature_mode="mr_feasibility_rule", num_servers=args.num_servers)

    # Pre-generate episodes
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

    records: List[CandidateRecord] = []
    total_candidates = 0
    invalid_feature_count = 0
    safe_indicator_violations = 0

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

                # Build features for all three modes
                default_feat, _ = agent_default.build_action_features(obs_c)
                rfeat_feat, _ = agent_r_feas.build_action_features(obs_c)
                mr_feat, _ = agent_mr.build_action_features(obs_c)

                num_candidates = mr_feat.shape[0]
                total_candidates += num_candidates

                # Checks
                if mr_feat.shape[1] != 33:
                    invalid_feature_count += 1
                if not np.isfinite(mr_feat[:, -6:]).all():
                    invalid_feature_count += 1
                if not ((mr_feat[:, -6:] >= -1e-6) & (mr_feat[:, -6:] <= 1 + 1e-6)).all():
                    invalid_feature_count += 1

                # Per-candidate raw values for diagnostics
                available = np.array([f["server_available_compute"] for f in obs_c["candidate_features"]])
                capacity = np.array([f["server_capacity"] for f in obs_c["candidate_features"]])
                edge_cost = np.array([f["edge_compute_cost"] for f in obs_c["candidate_features"]])
                capacity_safe = np.maximum(capacity, 1e-6)
                margin_raw = (available - edge_cost) / capacity_safe

                local_ms = np.array([f["local_compute_ms"] for f in obs_c["candidate_features"]])
                edge_ms = np.array([f["edge_compute_ms"] for f in obs_c["candidate_features"]])
                min_path_km = np.array([f["server_min_path_km"] for f in obs_c["candidate_features"]])
                path_delay_est = (min_path_km / DEFAULT_PROP_SPEED_KM_S) * 1000.0 + DEFAULT_SETUP_TIME_S * 1000.0
                deadline_ms = req.deadline_ms
                deadline_raw = (deadline_ms - (local_ms + edge_ms + path_delay_est)) / max(deadline_ms, 1e-6)

                # valid_r_actions per candidate from r_feasibility diagnostics
                from sa_hmarl.agents.c_agent import AgentC
                valid_r_actions_list = []
                for cand_idx, feat_dict in enumerate(obs_c["candidate_features"]):
                    diag = AgentC.compute_r_feasibility_diagnostics(obs_c, feat_dict)
                    valid_r_actions_list.append(int(diag.get("valid_r_actions", 0)))

                # Selected action and outcome
                action_idx_c, _ = _select_c_action(agent_c_rollout, obs_c, env.net.num_slots)
                if action_idx_c is None:
                    action_c = (0, 0)
                else:
                    action_c = decode_agent_c_action(action_idx_c, args.num_servers)
                split_id, server_id = action_c

                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                action_r = _select_r_action(agent_r, obs_r, env.max_blocks)
                _, _, _, info = env.step(action_c, action_r)

                success = bool(info.get("success", False))
                reason = info.get("reason", "unknown")
                blocked = not success
                server_overload = reason in ("server_overload", "server_saturated")
                no_suitable_block = reason == "no_suitable_block"
                deadline_infeasible = reason == "deadline_infeasible"
                delay_ms = float(info.get("delay_ms", 0.0)) if success else 0.0

                # Record each candidate
                for cand_idx in range(num_candidates):
                    mr_vec = mr_feat[cand_idx, -6:]
                    safe_indicator = float(mr_vec[5])

                    # Violation checks
                    if valid_r_actions_list[cand_idx] == 0 and safe_indicator > 0.5:
                        safe_indicator_violations += 1
                    if margin_raw[cand_idx] < -1e-6 and safe_indicator > 0.5:
                        safe_indicator_violations += 1

                    records.append(CandidateRecord(
                        seed=seed, episode=ep_idx, step=step_idx,
                        req_id=int(req.req_id), candidate_idx=cand_idx,
                        split_id=cand_idx // args.num_servers,
                        server_id=cand_idx % args.num_servers,
                        selected=(cand_idx == (action_idx_c if action_idx_c is not None else -1)),
                        mr_features=mr_vec,
                        server_margin_raw=float(margin_raw[cand_idx]),
                        deadline_margin_raw=float(deadline_raw[cand_idx]),
                        valid_r_actions=valid_r_actions_list[cand_idx],
                        success=success, reason=reason, blocked=blocked,
                        server_overload=server_overload,
                        no_suitable_block=no_suitable_block,
                        deadline_infeasible=deadline_infeasible,
                        delay_ms=delay_ms,
                    ))

    df_all = pd.DataFrame([r.to_dict() for r in records])
    df_selected = df_all[df_all["selected"]].copy()

    # Selected-candidate statistics by safe indicator
    safe_stats = {}
    for val in [0, 1]:
        sub = df_selected[df_selected["multi_resource_safe_indicator"] == val]
        n = max(len(sub), 1)
        safe_stats[str(val)] = {
            "count": len(sub),
            "success_rate": float(sub["success"].mean()) if len(sub) else 0.0,
            "blocking_rate": float(sub["blocked"].mean()) if len(sub) else 0.0,
            "server_overload_rate": float(sub["server_overload"].mean()) if len(sub) else 0.0,
            "no_suitable_block_rate": float(sub["no_suitable_block"].mean()) if len(sub) else 0.0,
            "avg_delay_ms": float(sub["delay_ms"].mean()) if len(sub) else 0.0,
        }

    # Bin analyses on selected candidates
    bin_results = {}
    for x_col, outcome in [
        ("server_margin_after", "server_overload"),
        ("server_margin_after", "blocked"),
        ("deadline_margin", "deadline_infeasible"),
        ("deadline_margin", "blocked"),
        ("spectrum_fit_margin", "no_suitable_block"),
        ("spectrum_fit_margin", "blocked"),
    ]:
        bin_results[f"{x_col}_vs_{outcome}"] = _bin_analysis(df_selected, x_col, outcome, n_bins=args.n_bins)

    # All-candidate violation counts
    all_violations = {
        "valid_r_actions_zero_but_safe_one": int(
            ((df_all["valid_r_actions"] == 0) & (df_all["multi_resource_safe_indicator"] > 0.5)).sum()
        ),
        "margin_negative_but_safe_one": int(
            ((df_all["server_margin_raw"] < -1e-6) & (df_all["multi_resource_safe_indicator"] > 0.5)).sum()
        ),
        "deadline_negative_but_safe_one": int(
            ((df_all["deadline_margin_raw"] < -1e-6) & (df_all["multi_resource_safe_indicator"] > 0.5)).sum()
        ),
    }

    # Mismatch examples
    mismatch_mask = (
        (df_all["valid_r_actions"] == 0) | (df_all["server_margin_raw"] < -1e-6)
    ) & (df_all["multi_resource_safe_indicator"] > 0.5)
    mismatches = df_all[mismatch_mask].head(20).to_dict(orient="records")

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
        "total_candidates": total_candidates,
        "invalid_feature_count": invalid_feature_count,
        "safe_indicator_violations": safe_indicator_violations,
        "all_candidate_violations": all_violations,
        "selected_candidate_count": len(df_selected),
        "safe_indicator_stats": safe_stats,
        "bin_analysis": bin_results,
        "mismatch_examples": mismatches,
        "elapsed_seconds": time.time() - t_start,
    }
    return report


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# MR-feasibility-rule Diagnostic Report")
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
    lines.append(f"- **Frozen R checkpoint:** `{cfg['agent_r_checkpoint']}`")
    lines.append("")

    lines.append("## Sanity Checks")
    lines.append("")
    lines.append(f"- Total candidates inspected: **{report['total_candidates']}**")
    lines.append(f"- Invalid feature vectors: **{report['invalid_feature_count']}**")
    lines.append(f"- Safe-indicator violations: **{report['safe_indicator_violations']}**")
    lines.append("")
    lines.append("| Violation type | Count |")
    lines.append("|---|---:|")
    for k, v in report["all_candidate_violations"].items():
        lines.append(f"| {k} | {v} |")
    lines.append("")

    lines.append("## Safe-Indicator Discrimination (selected candidate)")
    lines.append("")
    lines.append("| safe_indicator | Count | Success | Blocking | Server Overload | No Suitable Block | Avg Delay |")
    lines.append("|---:|---:|---:|---:|---:|---:|---:|")
    for val, s in report["safe_indicator_stats"].items():
        lines.append(
            f"| {val} | {s['count']} | {s['success_rate']:.4f} | {s['blocking_rate']:.4f} | "
            f"{s['server_overload_rate']:.4f} | {s['no_suitable_block_rate']:.4f} | {s['avg_delay_ms']:.2f} |"
        )
    lines.append("")

    lines.append("## Bin Analyses (selected candidate)")
    lines.append("")
    for title, rows in report["bin_analysis"].items():
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Bin | Count | Mean Predictor | Mean Outcome |")
        lines.append("|-----|------:|---------------:|-------------:|")
        for row in rows:
            lines.append(f"| {row['bin']} | {row['count']} | {row['mean_x']:.4f} | {row['mean_outcome']:.4f} |")
        lines.append("")

    lines.append("## Mismatch Examples (first 20)")
    lines.append("")
    if report["mismatch_examples"]:
        lines.append("Selected fields:")
        lines.append("")
        lines.append("| req_id | cand | selected | server_margin_raw | deadline_margin_raw | valid_r_actions | safe_indicator | reason |")
        lines.append("|---:|---:|:---:|---:|---:|---:|---:|:---|")
        for row in report["mismatch_examples"]:
            lines.append(
                f"| {row['req_id']} | {row['candidate_idx']} | {row['selected']} | "
                f"{row['server_margin_raw']:.4f} | {row['deadline_margin_raw']:.4f} | "
                f"{row['valid_r_actions']} | {row['multi_resource_safe_indicator']:.0f} | {row['reason']} |"
            )
    else:
        lines.append("No violations found.")
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
    parser.add_argument("--episodes", type=int, default=5)
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
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="sa_hmarl/experiments")
    args = parser.parse_args()

    report = _run_diagnostic(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "mr_feasibility_rule_diagnostic.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    md_path = output_dir / "mr_feasibility_rule_diagnostic.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))
    print(f"Markdown saved: {md_path}")

    print(f"\nElapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
