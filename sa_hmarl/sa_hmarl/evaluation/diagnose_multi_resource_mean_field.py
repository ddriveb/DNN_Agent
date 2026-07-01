"""Step-2 diagnostic for the multi-resource mean-field future-work branch.

Rolls out two frozen Agent-C policies (``default`` and ``r_feasibility``) with a
frozen PPO-R backend, records the 15-dimensional μ_res vector at every request
arrival, and performs bin-wise correlation analysis between μ_res dimensions and
request outcomes.

Outputs:
    * JSON: ``experiments/multi_resource_mean_field_correlation_report.json``
    * Markdown: ``experiments/multi_resource_mean_field_correlation_report.md``

Usage (from repo root with PYTHONPATH=sa_hmarl):
    python -m sa_hmarl.evaluation.diagnose_multi_resource_mean_field
"""
from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.multi_resource_mean_field import (
    compute_multi_resource_mean_field,
    multi_resource_mean_field_to_vector,
)
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.training.utils import generate_requests, make_env, SPLIT_PROFILES
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MU_DIM_NAMES = [
    # spectrum (0-5)
    "small_block_ratio",
    "medium_block_ratio",
    "large_block_ratio",
    "mean_largest_free_block_norm",
    "mean_fragmentation",
    "mean_free_slot_ratio",
    # compute (6-10)
    "idle_server_ratio",
    "normal_server_ratio",
    "busy_server_ratio",
    "overload_risk_server_ratio",
    "mean_available_compute_ratio",
    # queue (11-14)
    "mean_queue_delay_norm",
    "max_queue_delay_norm",
    "short_queue_ratio",
    "long_queue_ratio",
]

POLICY_LABELS = {
    "default": "Agent-C (default)",
    "r_feasibility": "Agent-C (r_feasibility)",
}

DEFAULT_CHECKPOINTS = {
    "default": "sa_hmarl/checkpoints/agent_c_default_indppo_s123_s20_r80_best.pt",
    "r_feasibility": "sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt",
}

DEFAULT_R_CKPT = "sa_hmarl/checkpoints/agent_r_mixed.pt"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StepRecord:
    """One row of the rollout dataset."""
    policy: str
    seed: int
    episode: int
    step: int
    req_id: int
    mu_res: np.ndarray = field(repr=False)
    action_c_idx: int
    action_c_split: int
    action_c_server: int
    n_valid_c: int
    n_valid_r_selected: int
    success: bool
    reason: str
    delay_ms: float
    blocked: bool
    server_overload: bool
    no_suitable_block: bool

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        d["mu_res"] = self.mu_res.astype(float).tolist()
        d["blocked"] = not self.success
        return d


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

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


# ---------------------------------------------------------------------------
# Action selection
# ---------------------------------------------------------------------------

def _select_c_action(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[Optional[int], int]:
    """Select Agent-C action and return (flat_idx, n_valid_after_mask)."""
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


def _select_r_action(
    agent_r: PPOAgentR, obs_r: Dict[str, Any], max_blocks: int
) -> Tuple[int, int, int]:
    """Select Agent-R action and return (path_idx, mod_idx, block_idx)."""
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return 0, 0, 0
    return decode_agent_r_action(
        action_idx, len(obs_r["mod_names"]), max_blocks
    )


# ---------------------------------------------------------------------------
# Rollout
# ---------------------------------------------------------------------------

def _rollout_policy(
    policy_name: str,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    env_proto: Any,
    episodes: List[List[Any]],
    seed: int,
    modulation_profile: str,
) -> List[StepRecord]:
    """Evaluate one policy on a list of pre-generated episodes."""
    records: List[StepRecord] = []
    num_servers = len(env_proto.mec.servers)

    for ep_idx, requests in enumerate(episodes):
        env = make_env(
            topology=env_proto.net.topology,
            num_slots=env_proto.net.num_slots,
            num_servers=num_servers,
            seed=42,
            slot_bw_hz=env_proto.fs_calc.slot_bw_hz,
            guard_band_fs=env_proto.fs_calc.guard_band_fs,
            modulation_profile=modulation_profile,
            max_blocks=env_proto.max_blocks,
            block_sort_strategy=env_proto.block_sort_strategy,
            k=env_proto.k,
        )
        env.reset(requests)

        for step_idx, req in enumerate(requests):
            # μ_res at request arrival (same observation point as the agent)
            mf = compute_multi_resource_mean_field(env)
            mu_vec = multi_resource_mean_field_to_vector(mf)

            # Agent-C observation and action
            obs_c = build_agent_c_observation(env, req)
            action_idx_c, n_valid_c = _select_c_action(agent_c, obs_c, env.net.num_slots)
            if action_idx_c is None:
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, num_servers)
            split_id, server_id = action_c

            # Agent-R observation and action for the chosen (split, server)
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            n_valid_r_selected = int(obs_r["agent_r_mask"].sum())
            action_r = _select_r_action(agent_r, obs_r, env.max_blocks)

            # Execute
            _, _, _, info = env.step(action_c, action_r)

            success = bool(info.get("success", False))
            reason = info.get("reason", "unknown")
            delay_ms = float(info.get("delay_ms", 0.0)) if success else 0.0

            records.append(StepRecord(
                policy=policy_name,
                seed=seed,
                episode=ep_idx,
                step=step_idx,
                req_id=int(req.req_id),
                mu_res=mu_vec,
                action_c_idx=int(action_idx_c if action_idx_c is not None else -1),
                action_c_split=split_id,
                action_c_server=server_id,
                n_valid_c=n_valid_c,
                n_valid_r_selected=n_valid_r_selected,
                success=success,
                reason=reason,
                delay_ms=delay_ms,
                blocked=not success,
                server_overload=(reason in ("server_overload", "server_saturated")),
                no_suitable_block=(reason == "no_suitable_block"),
            ))

    return records


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def _records_to_dataframe(records: List[StepRecord]) -> pd.DataFrame:
    """Convert step records to a pandas DataFrame."""
    rows = []
    for r in records:
        row = r.to_dict()
        for i, name in enumerate(MU_DIM_NAMES):
            row[name] = float(r.mu_res[i])
        # Delay is only defined for successful requests.
        row["delay_ms_success_only"] = r.delay_ms if r.success else float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def _bin_by_quantile(series: pd.Series, n_bins: int = 5) -> pd.Series:
    """Return quantile-based bins (labels are bin indices)."""
    # Use rank-based binning to avoid duplicate edge issues on discrete data.
    labels = [f"bin_{i}" for i in range(n_bins)]
    try:
        return pd.qcut(series.rank(method="first"), q=n_bins, labels=labels)
    except ValueError:
        # Fallback to simple quantile cut
        return pd.qcut(series, q=n_bins, labels=labels, duplicates="drop")


def _bin_analysis(df: pd.DataFrame, n_bins: int = 5) -> Dict[str, Any]:
    """Bin each μ_res dimension and compute outcome rates per bin."""
    result: Dict[str, Any] = {}
    for dim in MU_DIM_NAMES:
        df["_bin"] = _bin_by_quantile(df[dim], n_bins=n_bins)
        grouped = df.groupby("_bin", observed=False).agg(
            count=(dim, "size"),
            mean_dim=(dim, "mean"),
            blocking_rate=("blocked", "mean"),
            server_overload_rate=("server_overload", "mean"),
            no_suitable_block_rate=("no_suitable_block", "mean"),
            success_rate=("success", "mean"),
            mean_valid_r=("n_valid_r_selected", "mean"),
            mean_delay_ms=("delay_ms_success_only", "mean"),
        ).reset_index().rename(columns={"_bin": "bin"})
        result[dim] = grouped.to_dict(orient="records")
    df.drop(columns=["_bin"], inplace=True, errors="ignore")
    return result


def _joint_bin_analysis(
    df: pd.DataFrame,
    x_dim: str,
    outcome: str,
    n_bins: int = 5,
) -> List[Dict[str, Any]]:
    """Bin a single dimension and report one outcome metric per bin."""
    df["_bin"] = _bin_by_quantile(df[x_dim], n_bins=n_bins)
    grouped = df.groupby("_bin", observed=False).agg(
        count=(x_dim, "size"),
        mean_x=(x_dim, "mean"),
        mean_outcome=(outcome, "mean"),
    ).reset_index().rename(columns={"_bin": "bin"})
    df.drop(columns=["_bin"], inplace=True, errors="ignore")
    return grouped.to_dict(orient="records")


def _compute_correlations(df: pd.DataFrame) -> Dict[str, Any]:
    """Pearson and Spearman correlations between μ_res dims and outcomes."""
    outcomes = {
        "blocked": "blocked",
        "server_overload": "server_overload",
        "no_suitable_block": "no_suitable_block",
        "success": "success",
        "selected_r_valid_actions": "n_valid_r_selected",
        "delay_ms": "delay_ms_success_only",
    }
    pearson: Dict[str, Dict[str, float]] = defaultdict(dict)
    spearman: Dict[str, Dict[str, float]] = defaultdict(dict)

    for dim in MU_DIM_NAMES:
        x_all = df[dim].values
        for out_label, col in outcomes.items():
            y_all = df[col].values
            # Delay is undefined for blocked requests; drop NaNs for that outcome.
            mask = np.isfinite(x_all) & np.isfinite(y_all)
            x = x_all[mask]
            y = y_all[mask]
            if len(x) < 2 or np.std(x) == 0 or np.std(y) == 0:
                r_p = p_p = r_s = p_s = float("nan")
            else:
                r_p, p_p = stats.pearsonr(x, y)
                r_s, p_s = stats.spearmanr(x, y)
            pearson[dim][out_label] = float(r_p)
            pearson[dim][f"{out_label}_pvalue"] = float(p_p)
            spearman[dim][out_label] = float(r_s)
            spearman[dim][f"{out_label}_pvalue"] = float(p_s)

    return {
        "pearson": dict(pearson),
        "spearman": dict(spearman),
    }


def _aggregate_policy_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """High-level outcome rates per policy."""
    summary = df.groupby("policy").agg(
        total=("policy", "size"),
        blocked_count=("blocked", "sum"),
        blocking_rate=("blocked", "mean"),
        success_rate=("success", "mean"),
        server_overload_rate=("server_overload", "mean"),
        no_suitable_block_rate=("no_suitable_block", "mean"),
        mean_valid_r_actions=("n_valid_r_selected", "mean"),
        mean_delay_ms=("delay_ms", "mean"),
    ).reset_index()
    return summary.to_dict(orient="records")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _build_markdown_report(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Multi-Resource Mean-Field Step 2 — Correlation Diagnostic")
    lines.append("")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    # Setup
    lines.append("## Experimental Setup")
    lines.append("")
    cfg = report["config"]
    lines.append(f"- **Topology:** {cfg['topology']}")
    lines.append(f"- **Slots:** {cfg['num_slots']}, **Servers:** {cfg['num_servers']}, **k_paths:** {cfg['k_paths']}")
    lines.append(f"- **max_blocks:** {cfg['max_blocks']}, **block_sort:** {cfg['block_sort_strategy']}")
    lines.append(f"- **split_profile:** {cfg['split_profile']}")
    lines.append(f"- **Seeds:** {cfg['seeds']}, **Episodes/seed:** {cfg['episodes']}, **Requests/episode:** {cfg['requests_per_episode']}")
    lines.append(f"- **Policies:** {', '.join(cfg['policies'])}")
    lines.append(f"- **Frozen R backend:** `{cfg['agent_r_checkpoint']}`")
    lines.append("")

    # Summary
    lines.append("## Policy-Level Summary")
    lines.append("")
    lines.append("| Policy | Total | Blocking | Success | Server Overload | No Suitable Block | Avg Valid-R | Mean Delay (ms) |")
    lines.append("|--------|------:|---------:|--------:|----------------:|------------------:|------------:|----------------:|")
    for row in report["policy_summary"]:
        lines.append(
            f"| {row['policy']} | {row['total']} | "
            f"{row['blocking_rate']:.4f} | {row['success_rate']:.4f} | "
            f"{row['server_overload_rate']:.4f} | {row['no_suitable_block_rate']:.4f} | "
            f"{row['mean_valid_r_actions']:.3f} | {row['mean_delay_ms']:.2f} |"
        )
    lines.append("")

    # Joint highlights
    lines.append("## Key Bin-Wise Relationships")
    lines.append("")

    for title, key, x_dim, out_col in [
        ("Fragmentation vs Blocking", "fragmentation_vs_blocking", "mean_fragmentation", "blocked"),
        ("Overload Risk vs Server Overload", "overload_risk_vs_server_overload", "overload_risk_server_ratio", "server_overload"),
        ("Available Compute vs Success", "available_compute_vs_success", "mean_available_compute_ratio", "success"),
        ("Queue Pressure vs Delay", "queue_pressure_vs_delay", "mean_queue_delay_norm", "delay_ms"),
    ]:
        lines.append(f"### {title}")
        lines.append("")
        lines.append("| Bin | Count | Mean Predictor | Mean Outcome |")
        lines.append("|-----|------:|---------------:|-------------:|")
        for row in report["joint_analysis"][key]:
            lines.append(
                f"| {row['bin']} | {row['count']} | "
                f"{row['mean_x']:.4f} | {row['mean_outcome']:.4f} |"
            )
        lines.append("")

    # Correlation highlights
    lines.append("## Strongest Correlations (|r| > 0.05)")
    lines.append("")
    lines.append("### Pearson")
    lines.append("")
    lines.append("| Dimension | Outcome | r | p-value |")
    lines.append("|-----------|---------|---:|--------:|")
    for dim, out, r, p in _top_correlations(report["correlations"]["pearson"], n=20):
        lines.append(f"| {dim} | {out} | {r:.4f} | {p:.2e} |")
    lines.append("")

    lines.append("### Spearman")
    lines.append("")
    lines.append("| Dimension | Outcome | r | p-value |")
    lines.append("|-----------|---------|---:|--------:|")
    for dim, out, r, p in _top_correlations(report["correlations"]["spearman"], n=20):
        lines.append(f"| {dim} | {out} | {r:.4f} | {p:.2e} |")
    lines.append("")

    # Per-dimension bin tables
    lines.append("## Per-Dimension Bin Analysis")
    lines.append("")
    for dim in MU_DIM_NAMES:
        lines.append(f"### {dim}")
        lines.append("")
        lines.append(
            "| Bin | Count | Mean Dim | Blocking | Overload | No Block | Success | Avg Valid-R | Avg Delay (succ) |"
        )
        lines.append(
            "|-----|------:|---------:|---------:|---------:|---------:|--------:|------------:|-----------------:|")
        for row in report["bin_analysis"][dim]:
            lines.append(
                f"| {row['bin']} | {row['count']} | {row['mean_dim']:.4f} | "
                f"{row['blocking_rate']:.4f} | {row['server_overload_rate']:.4f} | "
                f"{row['no_suitable_block_rate']:.4f} | {row['success_rate']:.4f} | "
                f"{row['mean_valid_r']:.3f} | {row['mean_delay_ms']:.2f} |"
            )
        lines.append("")

    lines.append("---")
    lines.append("*Report generated by `sa_hmarl/evaluation/diagnose_multi_resource_mean_field.py`*")
    return "\n".join(lines)


def _top_correlations(corr_dict: Dict[str, Dict[str, float]], n: int = 20):
    """Return the top-N correlations by absolute value (excluding p-value keys)."""
    items = []
    for dim, vals in corr_dict.items():
        for k, v in vals.items():
            if k.endswith("_pvalue"):
                continue
            items.append((dim, k, abs(v), v, vals.get(f"{k}_pvalue", float("nan"))))
    items.sort(key=lambda x: x[2], reverse=True)
    return [(dim, out, r, p) for dim, out, _, r, p in items[:n]]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _parse_seeds(seeds_str: str) -> List[int]:
    return [int(s.strip()) for s in seeds_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Multi-resource mean-field Step-2 correlation diagnostic",
    )
    parser.add_argument("--agent_c_default", type=str,
                        default=DEFAULT_CHECKPOINTS["default"])
    parser.add_argument("--agent_c_r_feasibility", type=str,
                        default=DEFAULT_CHECKPOINTS["r_feasibility"])
    parser.add_argument("--agent_r_checkpoint", type=str, default=DEFAULT_R_CKPT)
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
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
    parser.add_argument("--output_dir", type=str,
                        default="sa_hmarl/experiments")
    parser.add_argument("--n_bins", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--json_only", action="store_true",
                        help="Skip markdown report generation.")
    args = parser.parse_args()

    t_start = time.time()
    seeds = _parse_seeds(args.seeds)
    policies = ["default", "r_feasibility"]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prototype environment (used for request generation and topology defaults)
    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg

    print("Loading frozen R backend...")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    print("Loading Agent-C policies...")
    agents = {
        "default": _load_ppo_c(args.agent_c_default, args.device),
        "r_feasibility": _load_ppo_c(args.agent_c_r_feasibility, args.device),
    }

    # Pre-generate identical request episodes for both policies
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
                num_splits=len(SPLIT_PROFILES.get(args.split_profile, SPLIT_PROFILES["default3"])["size_multipliers"]),
                split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps

    # Collect rollouts
    all_records: List[StepRecord] = []
    for policy in policies:
        print(f"\nRolling out {POLICY_LABELS[policy]} ...")
        t0 = time.time()
        for seed in seeds:
            records = _rollout_policy(
                policy, agents[policy], agent_r, env_proto,
                all_episodes[seed], seed,
                modulation_profile=args.modulation_profile,
            )
            all_records.extend(records)
            print(f"  seed {seed}: {len(records)} requests")
        print(f"  done in {time.time() - t0:.1f}s")

    df = _records_to_dataframe(all_records)
    print(f"\nTotal records: {len(df)}")

    # Analysis
    print("Running bin analysis...")
    bin_result = _bin_analysis(df, n_bins=args.n_bins)

    print("Running correlation analysis...")
    corr_result = _compute_correlations(df)

    print("Running joint analysis...")
    joint_analysis = {
        "fragmentation_vs_blocking": _joint_bin_analysis(
            df, "mean_fragmentation", "blocked", n_bins=args.n_bins
        ),
        "overload_risk_vs_server_overload": _joint_bin_analysis(
            df, "overload_risk_server_ratio", "server_overload", n_bins=args.n_bins
        ),
        "available_compute_vs_success": _joint_bin_analysis(
            df, "mean_available_compute_ratio", "success", n_bins=args.n_bins
        ),
        "queue_pressure_vs_delay": _joint_bin_analysis(
            df, "mean_queue_delay_norm", "delay_ms_success_only", n_bins=args.n_bins
        ),
    }

    policy_summary = _aggregate_policy_summary(df)

    report = {
        "config": {
            "policies": policies,
            "agent_c_default": args.agent_c_default,
            "agent_c_r_feasibility": args.agent_c_r_feasibility,
            "agent_r_checkpoint": args.agent_r_checkpoint,
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
            "n_bins": args.n_bins,
            "device": args.device,
        },
        "policy_summary": policy_summary,
        "bin_analysis": bin_result,
        "joint_analysis": joint_analysis,
        "correlations": corr_result,
        "num_records": len(df),
        "elapsed_seconds": time.time() - t_start,
    }

    # JSON output
    json_path = output_dir / "multi_resource_mean_field_correlation_report.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON report saved: {json_path}")

    # Markdown output
    if not args.json_only:
        md_path = output_dir / "multi_resource_mean_field_correlation_report.md"
        md = _build_markdown_report(report)
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(md)
        print(f"Markdown report saved: {md_path}")

    print(f"Total elapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
