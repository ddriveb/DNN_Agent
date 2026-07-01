"""C Action-Space Feasibility Sweep.

Sweeps scenario parameters to discover configurations where the C action
space offers meaningful room for improvement beyond the current Agent-C.

Evaluates 5 C-action methods per scenario:
  Agent-C (PPO), DF-C, IWD-C, RF-C, WO-C

All paired with the SAME frozen BC-PPO-R backend.

Usage::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.sweep_c_feasibility_space
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from itertools import product
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.c_action_risk import apply_agent_c_risk_mask, risk_kwargs_from_args
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWEEP_AXES = {
    "num_slots": [16, 18, 20, 24],
    "requests_per_episode": [40, 60, 80],
    "arrival_interval": [0.20, 0.25, 0.30],
    "edge_cost_max": [10, 15, 20],
    "size_max_mb": [20, 30],
}

METHOD_ORDER = ["agent", "df", "rf", "wo", "iwd"]

METHOD_LABELS = {
    "agent": "Agent-C",
    "df": "DF-C",
    "rf": "RF-C",
    "wo": "WO-C",
    "iwd": "IWD-C",
}


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"),
    )
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=feature_mode,
    )
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
# Agent-C selection + mask stats
# ---------------------------------------------------------------------------

def _select_agent_c_with_stats(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Tuple[Optional[int], int]:
    """Select Agent-C action and return (action_idx, num_valid_actions)."""
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


# ---------------------------------------------------------------------------
# Per-method evaluation (single episode)
# ---------------------------------------------------------------------------

def _eval_method_on_episode(
    method: str,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    requests: List,
    env_proto,
    args,
    ep_idx: int,
) -> Dict[str, Any]:
    """Evaluate one method on one episode.  Returns per-episode metrics."""
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities,
    )
    env.reset(requests)

    server_selected_count = np.zeros(num_servers, dtype=int)
    seed_val = int(getattr(args, "seeds", "42").split(",")[0]) if hasattr(args, "seeds") else getattr(args, "seed", 42)
    rng = np.random.RandomState(seed_val + ep_idx)

    total = blocked = success = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    mod_counter = Counter()
    server_overload_counter = Counter()

    # mask stats (collected only during Agent-C to avoid method-dependent state)
    mask_valid_counts: List[int] = []
    mask_zero_count = 0

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        raw_mask = obs_c["agent_c_mask"]

        if method == "agent":
            action_idx_c, n_valid = _select_agent_c_with_stats(
                agent_c, obs_c, env.net.num_slots,
            )
            mask_valid_counts.append(n_valid)
            if n_valid == 0:
                mask_zero_count += 1
            action_c = (
                (0, 0) if action_idx_c is None
                else decode_agent_c_action(action_idx_c, num_servers)
            )

        else:
            action_idx_c = select_offloading_action(
                method, env, req, obs_c, raw_mask,
                rng=rng, server_selected_count=server_selected_count,
            )
            action_c = (
                (0, 0) if action_idx_c is None
                else decode_agent_c_action(action_idx_c, num_servers)
            )

        split_id, server_id = action_c
        split_counter[f"split{split_id}"] += 1
        server_counter[f"s{server_id}"] += 1
        server_selected_count[server_id] += 1

        # Agent-R selection (same frozen R for all methods)
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        action_idx_r = agent_r.select_action(obs_r, deterministic=True)
        action_r = (0, 0, 0) if action_idx_r is None else (
            decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), env.max_blocks)
        )

        _, _, _, info = env.step(action_c, action_r)
        total += 1

        server_obj = env.mec.servers[server_id]
        reward = compute_agent_c_reward(
            info, req.deadline_ms, args.waste_coef, server_obj.utilization,
        )
        total_reward += reward

        if info.get("success", False):
            success += 1
            delay = float(info.get("delay_ms", 0.0))
            total_delay += delay
            total_fs += float(info.get("num_slots", 0))
            total_waste += float(info.get("block_waste", 0.0))
            total_path += float(info.get("path_dist_km", 0.0))
            mod_counter[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1
            reason = info.get("reason", "unknown")
            reason_counter[reason] += 1
            if reason in ("server_overload", "server_saturated"):
                server_overload_counter[f"s{server_id}"] += 1

    n = max(total, 1)
    s = max(success, 1)

    result = {
        "total": total, "blocked": blocked, "success": success,
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "reason_counter": dict(reason_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "mod_counter": dict(mod_counter),
    }

    if method == "agent":
        result["mask_valid_counts"] = mask_valid_counts
        result["mask_zero_count"] = mask_zero_count
        result["no_valid_c_rate_ep"] = mask_zero_count / n

    return result


# ---------------------------------------------------------------------------
# Evaluate one scenario (all methods)
# ---------------------------------------------------------------------------

def evaluate_scenario(
    scenario_key: str,
    scenario_args,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    env_proto,
    all_episodes: Dict[int, List[List]],
    base_args,
) -> Dict[str, Any]:
    """Evaluate all 5 methods for one scenario config."""
    seeds = [int(s) for s in base_args.seeds.split(",")]
    scenario_result: Dict[str, Any] = {"scenario": scenario_key}

    for method in METHOD_ORDER:
        ep_results = []
        for seed in seeds:
            for ep_idx, requests in enumerate(all_episodes[seed]):
                res = _eval_method_on_episode(
                    method, agent_c, agent_r, requests, env_proto, scenario_args, ep_idx,
                )
                ep_results.append(res)

        # Aggregate across seeds × episodes
        scalar_keys = [
            "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
            "avg_fs", "avg_waste", "avg_path_len_km",
        ]
        agg: Dict[str, Any] = {}
        for key in scalar_keys:
            vals = [float(r[key]) for r in ep_results]
            agg[key] = float(np.mean(vals))
        for key in ["reason_counter", "split_counter", "server_counter", "mod_counter"]:
            c = Counter()
            for r in ep_results:
                c.update(r.get(key, {}))
            agg[key] = dict(c)

        # no_suitable_block ratio
        reason = agg.get("reason_counter", {})
        total = sum(reason.values())
        nsb = reason.get("no_suitable_block", 0)
        agg["no_suitable_block_ratio"] = nsb / max(total, 1)

        # server_overload ratio
        so = reason.get("server_overload", 0) + reason.get("server_saturated", 0)
        agg["server_overload_ratio"] = so / max(total, 1)

        # mask stats (from Agent-C trajectory only)
        if method == "agent":
            all_counts = []
            zero_total = 0
            req_total = 0
            for r in ep_results:
                if "mask_valid_counts" in r:
                    all_counts.extend(r["mask_valid_counts"])
                    zero_total += r.get("mask_zero_count", 0)
                    req_total += r["total"]
            agg["no_valid_c_rate"] = zero_total / max(req_total, 1)
            agg["avg_valid_c_actions"] = float(np.mean(all_counts)) if all_counts else 0.0
            agg["p25_valid_c_actions"] = float(np.percentile(all_counts, 25)) if all_counts else 0.0
            agg["p50_valid_c_actions"] = float(np.percentile(all_counts, 50)) if all_counts else 0.0

        scenario_result[method] = agg

    return scenario_result


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _scenario_label(params: Dict[str, Any]) -> str:
    return (
        f"S{params['num_slots']}_R{params['requests_per_episode']}"
        f"_AI{params['arrival_interval']}_EC{params['edge_cost_max']}"
        f"_SZ{params['size_max_mb']}"
    )


def _fmt_pct(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    items = sorted(c.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v / total:.1%}" for k, v in items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/c_feasibility_space_sweep.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/c_feasibility_space_sweep.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Load models once
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    # Build sweep grid
    axes = list(SWEEP_AXES.keys())
    values = list(SWEEP_AXES.values())
    grid = list(product(*values))
    total_scenarios = len(grid)
    print(f"\nC Action-Space Feasibility Sweep")
    print(f"Axes: {axes}")
    print(f"Total scenarios: {total_scenarios}  Seeds: {seeds}  Eps/seed: {args.episodes}")
    print(f"Methods: {METHOD_ORDER}")
    print(f"={ '=' * 60}")

    # ------------------------------------------------------------------
    # Run sweep
    # ------------------------------------------------------------------
    all_scenario_results: List[Dict[str, Any]] = []
    t_start = time.time()

    for idx, combo in enumerate(grid):
        params = dict(zip(axes, combo))
        label = _scenario_label(params)

        # Build scenario-specific args
        scenario_args = argparse.Namespace(**{
            **vars(args),
            "num_slots": params["num_slots"],
            "requests_per_episode": params["requests_per_episode"],
            "arrival_interval": params["arrival_interval"],
            "edge_cost_max": params["edge_cost_max"],
            "size_max_mb": params["size_max_mb"],
        })

        # Build env prototype for this scenario (for generate_requests)
        env_proto = make_env(
            topology=args.topology, num_slots=params["num_slots"],
            num_servers=args.num_servers, seed=42,
            slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        )

        # Pre-generate episodes (same for all methods)
        all_episodes: Dict[int, List[List]] = {}
        for seed in seeds:
            rng = np.random.RandomState(seed)
            eps = []
            for _ in range(args.episodes):
                src = rng.randint(0, env_proto.net.NUM_NODES)
                eps.append(generate_requests(
                    env_proto, rng, src,
                    num_requests=params["requests_per_episode"],
                    arrival_interval=params["arrival_interval"],
                    holding_min=args.holding_min, holding_max=args.holding_max,
                    deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                    size_min_mb=args.size_min_mb, size_max_mb=params["size_max_mb"],
                    edge_cost_min=args.edge_cost_min, edge_cost_max=params["edge_cost_max"],
                    num_splits=args.num_splits, split_profile=args.split_profile,
                ))
            all_episodes[seed] = eps

        # Evaluate all methods
        scenario_result = evaluate_scenario(
            label, scenario_args, agent_c, agent_r, env_proto, all_episodes, args,
        )
        all_scenario_results.append(scenario_result)

        # Progress
        agent_blk = scenario_result["agent"]["blocking_rate"]
        noc = scenario_result["agent"]["no_valid_c_rate"]
        n_actions = scenario_result["agent"]["avg_valid_c_actions"]
        elapsed = time.time() - t_start
        avg_per = elapsed / (idx + 1)
        eta = avg_per * (total_scenarios - idx - 1)
        print(f"[{idx + 1:3d}/{total_scenarios}] {label}  "
              f"blk={agent_blk:.3f}  noC={noc:.2f}  avgActs={n_actions:.1f}  "
              f"elapsed={elapsed:.0f}s  ETA={eta:.0f}s")

    total_elapsed = time.time() - t_start
    print(f"\nDone. Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Find candidate scenarios
    # ------------------------------------------------------------------
    candidates = []
    for sr in all_scenario_results:
        agent = sr["agent"]
        noc = agent["no_valid_c_rate"]
        agent_blk = agent["blocking_rate"]
        agent_delay = agent["avg_delay_ms"]

        # Find best baseline
        best_blk = 1.0
        best_method = None
        best_delay = float("inf")
        for method in ["df", "rf", "wo", "iwd"]:
            m = sr[method]
            if m["blocking_rate"] < best_blk:
                best_blk = m["blocking_rate"]
                best_method = method
                best_delay = m["avg_delay_ms"]

        delta_blk = agent_blk - best_blk
        delay_ratio = agent_delay / max(best_delay, 0.001)

        sr["_best_baseline"] = best_method
        sr["_best_baseline_blk"] = best_blk
        sr["_delta_blk_vs_best"] = delta_blk
        sr["_delay_ratio"] = delay_ratio

        if (noc <= 0.10
                and 0.15 <= agent_blk <= 0.35
                and delta_blk >= 0.03
                and delay_ratio <= 1.3):
            candidates.append(sr)

    # Sort candidates by delta_blk descending
    candidates.sort(key=lambda x: -x["_delta_blk_vs_best"])
    top10 = candidates[:10]

    # ------------------------------------------------------------------
    # JSON output
    # ------------------------------------------------------------------
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    json_output = {
        "args": vars(args),
        "sweep_axes": SWEEP_AXES,
        "total_scenarios": total_scenarios,
        "total_time_s": total_elapsed,
        "num_candidates": len(candidates),
        "top10_candidates": _clean(top10),
        "all_results": _clean(all_scenario_results),
    }
    out_json.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown output
    # ------------------------------------------------------------------
    md: List[str] = []
    md.append("# C Action-Space Feasibility Sweep\n\n")
    md.append(f"**Topology:** {args.topology}  **Servers:** {args.num_servers}  "
              f"**MaxBlocks:** {args.max_blocks}\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}\n\n")
    md.append(f"**Total scenarios:** {total_scenarios}  "
              f"**Total time:** {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)\n\n")

    md.append("## Sweep Axes\n\n")
    md.append("| Axis | Values |\n|---|---|\n")
    for axis, vals in SWEEP_AXES.items():
        md.append(f"| {axis} | {vals} |\n")

    # Candidate filter
    md.append("\n## Candidate Criteria\n\n")
    md.append("- `no_valid_c_rate` ≤ 0.10\n")
    md.append("- Agent-C blocking ∈ [0.15, 0.35]\n")
    md.append("- Best-baseline Δblocking ≥ 0.03 (3pp)\n")
    md.append("- Agent-C delay ≤ 1.3× best-baseline delay\n\n")

    md.append(f"**Passed:** {len(candidates)} / {total_scenarios}\n\n")

    # Top 10
    md.append("## Top 10 Candidate Scenarios\n\n")
    md.append("| # | Scenario | AgentBlk | noC% | avgActs | "
              "BestBase | BestBlk | ΔBlk | AgentDelay | BestDelay | "
              "DFblk | RFblk | WOblk | IWDblk |\n")
    md.append("|---|----------|----------|------|---------|"
              "----------|---------|------|------------|-----------|"
              "-------|-------|-------|--------|\n")
    for rank, sr in enumerate(top10, 1):
        agent = sr["agent"]
        best = sr["_best_baseline"]
        md.append(
            f"| {rank} | {sr['scenario']} "
            f"| {agent['blocking_rate']:.4f} "
            f"| {agent['no_valid_c_rate']:.2f} "
            f"| {agent['avg_valid_c_actions']:.1f} "
            f"| {best} "
            f"| {sr['_best_baseline_blk']:.4f} "
            f"| {sr['_delta_blk_vs_best']:+.4f} "
            f"| {agent['avg_delay_ms']:.1f} "
            f"| {sr.get('_best_baseline_delay', sr[best]['avg_delay_ms']):.1f} "
            f"| {sr['df']['blocking_rate']:.4f} "
            f"| {sr['rf']['blocking_rate']:.4f} "
            f"| {sr['wo']['blocking_rate']:.4f} "
            f"| {sr['iwd']['blocking_rate']:.4f} |\n"
        )

    # Full results table
    md.append("\n## All Scenarios\n\n")
    md.append("| Scenario | Slots | Req | ArrI | ECMax | SzMax | "
              "noC% | avgActs | p25Acts | "
              "AgentBlk | DFblk | RFblk | WOblk | IWDblk | "
              "AgentDelay | BestΔ |\n")
    md.append("|----------|-------|-----|------|-------|-------|"
              "------|--------|--------|"
              "----------|-------|-------|-------|--------|"
              "------------|-------|\n")
    for sr in all_scenario_results:
        agent = sr["agent"]
        best_delta = sr["_delta_blk_vs_best"]
        md.append(
            f"| {sr['scenario']} "
            f"| {sr['scenario'].split('_')[0].replace('S','')} "
            f"| {sr['scenario'].split('_')[1].replace('R','')} "
            f"| {sr['scenario'].split('_')[2].replace('AI','')} "
            f"| {sr['scenario'].split('_')[3].replace('EC','')} "
            f"| {sr['scenario'].split('_')[4].replace('SZ','')} "
            f"| {agent['no_valid_c_rate']:.3f} "
            f"| {agent['avg_valid_c_actions']:.1f} "
            f"| {agent['p25_valid_c_actions']:.1f} "
            f"| {agent['blocking_rate']:.4f} "
            f"| {sr['df']['blocking_rate']:.4f} "
            f"| {sr['rf']['blocking_rate']:.4f} "
            f"| {sr['wo']['blocking_rate']:.4f} "
            f"| {sr['iwd']['blocking_rate']:.4f} "
            f"| {agent['avg_delay_ms']:.1f} "
            f"| {best_delta:+.4f} |\n"
        )

    # Failure reason summary for candidates
    if top10:
        md.append("\n## Top Candidate Failure Reasons\n\n")
        for sr in top10:
            md.append(f"### {sr['scenario']}\n\n")
            for method in METHOD_ORDER:
                m = sr[method]
                md.append(f"- **{METHOD_LABELS[method]}**: "
                          f"blk={m['blocking_rate']:.4f} "
                          f"nsb={m.get('no_suitable_block_ratio', 0):.3f} "
                          f"so={m.get('server_overload_ratio', 0):.3f} "
                          f"delay={m['avg_delay_ms']:.1f}ms "
                          f"fs={m['avg_fs']:.2f}\n")

    out_md = Path(args.out_md)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print(f"SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total scenarios: {total_scenarios}")
    print(f"Passed filter: {len(candidates)}")
    if not candidates:
        print("No candidates found!")
    else:
        print(f"Top {min(10, len(candidates))} candidates:")
        for rank, sr in enumerate(top10, 1):
            agent = sr["agent"]
            print(f"  {rank}. {sr['scenario']}  "
                  f"AgentBlk={agent['blocking_rate']:.4f}  "
                  f"ΔBlk={sr['_delta_blk_vs_best']:+.4f} vs {sr['_best_baseline']}  "
                  f"noC={agent['no_valid_c_rate']:.2f}  "
                  f"acts={agent['avg_valid_c_actions']:.1f}")


if __name__ == "__main__":
    main()
