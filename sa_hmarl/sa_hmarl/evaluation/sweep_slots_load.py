"""Num-Slots × Load Sweep — Find viable paper-experiment pressure range.

Sweeps num_slots and requests_per_episode to identify scenarios where:
  - Agent-C blocking ∈ [0.10, 0.30]
  - no_valid_c_rate ≤ 0.20
  - Agent-C beats best heuristic by ≥ 3pp
  - Agent-C delay ≤ 1.3× best heuristic delay
  - Per-seed blocking std ≤ 0.15

Six C-action methods, all paired with BC-PPO-R (k=5).

Usage::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.sweep_slots_load
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
    "num_slots": [16, 20, 24, 28, 32],
    "requests_per_episode": [40, 60, 80],
}

METHOD_ORDER = ["agent", "df", "rf", "wo", "iwd", "greedy"]

METHOD_LABELS = {
    "agent":  "Agent-C + PPO-R",
    "df":     "DF-C + PPO-R",
    "rf":     "RF-C + PPO-R",
    "wo":     "WO-C + PPO-R",
    "iwd":    "IWD-C + PPO-R",
    "greedy": "Greedy-C + PPO-R",
}


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    fm = ckpt_args.get("agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"))
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=fm,
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
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask, num_slots_total=num_slots,
            min_valid_after_mask=min_valid, **risk_kwargs,
        )
    n_valid_c = int(mask.sum())
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action, n_valid_c


# ---------------------------------------------------------------------------
# Per-episode evaluation
# ---------------------------------------------------------------------------

def _eval_episode(
    method: str, agent_c: PPOAgentC, agent_r: PPOAgentR,
    requests: List, env_proto, args, ep_idx: int, seed: int,
) -> Dict[str, Any]:
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    env = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        server_nodes=server_nodes, capacities=capacities, k=args.k_paths,
    )
    env.reset(requests)

    server_selected_count = np.zeros(num_servers, dtype=int)
    rng = np.random.RandomState(seed + ep_idx)

    total = blocked = success = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    reason_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    mod_counter = Counter()

    mask_valid_c_counts: List[int] = []
    mask_zero_c_count = 0
    mask_valid_r_counts: List[int] = []

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        raw_mask = obs_c["agent_c_mask"]

        if method == "agent":
            action_idx_c, n_valid_c = _select_agent_c_with_stats(
                agent_c, obs_c, env.net.num_slots,
            )
            mask_valid_c_counts.append(n_valid_c)
            if n_valid_c == 0:
                mask_zero_c_count += 1
            action_c = (
                (0, 0) if action_idx_c is None
                else decode_agent_c_action(action_idx_c, num_servers)
            )
        else:
            action_idx_c = select_offloading_action(
                method, env, req, obs_c, raw_mask,
                rng=rng, server_selected_count=server_selected_count,
            )
            n_valid_c = int(raw_mask.sum())
            mask_valid_c_counts.append(n_valid_c)
            if n_valid_c == 0:
                mask_zero_c_count += 1
            action_c = (
                (0, 0) if action_idx_c is None
                else decode_agent_c_action(action_idx_c, num_servers)
            )

        split_id, server_id = action_c
        split_counter[f"split{split_id}"] += 1
        server_counter[f"s{server_id}"] += 1
        server_selected_count[server_id] += 1

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        mask_valid_r_counts.append(int(obs_r["agent_r_mask"].sum()))

        action_r_idx = agent_r.select_action(obs_r, deterministic=True)
        action_r_tuple = (0, 0, 0) if action_r_idx is None else (
            decode_agent_r_action(action_r_idx, len(obs_r["mod_names"]), env.max_blocks)
        )

        _, _, _, info = env.step(action_c, action_r_tuple)
        total += 1

        server_obj = env.mec.servers[server_id]
        reward = compute_agent_c_reward(
            info, req.deadline_ms, args.waste_coef, server_obj.utilization,
        )
        total_reward += reward

        if info.get("success", False):
            success += 1
            total_delay += float(info.get("delay_ms", 0.0))
            total_fs += float(info.get("num_slots", 0))
            total_waste += float(info.get("block_waste", 0.0))
            total_path += float(info.get("path_dist_km", 0.0))
            mod_counter[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1
            reason_counter[info.get("reason", "unknown")] += 1

    n = max(total, 1); s = max(success, 1)
    return {
        "total": total, "blocked": blocked, "success": success,
        "blocking_rate": blocked / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s, "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "reason_counter": dict(reason_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "mod_counter": dict(mod_counter),
        "mask_valid_c_counts": mask_valid_c_counts,
        "mask_valid_r_counts": mask_valid_r_counts,
        "no_valid_c_rate_ep": mask_zero_c_count / n,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_scenario(ep_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate across all seeds × episodes."""
    scalar_keys = [
        "blocking_rate", "avg_reward", "avg_delay_ms",
        "avg_fs", "avg_waste", "avg_path_len_km",
    ]
    agg: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(r[key]) for r in ep_results]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))

    for key in ["reason_counter", "split_counter", "server_counter", "mod_counter"]:
        c = Counter()
        for r in ep_results:
            c.update(r.get(key, {}))
        agg[key] = dict(c)

    all_c = []; zero_c = 0; req_c = 0; all_r = []
    for r in ep_results:
        if "mask_valid_c_counts" in r:
            all_c.extend(r["mask_valid_c_counts"])
            zero_c += r.get("no_valid_c_rate_ep", 0) * r["total"]
            req_c += r["total"]
        if "mask_valid_r_counts" in r:
            all_r.extend(r["mask_valid_r_counts"])
    agg["no_valid_c_rate"] = zero_c / max(req_c, 1)
    agg["avg_valid_c_actions"] = float(np.mean(all_c)) if all_c else 0.0
    agg["avg_valid_r_actions"] = float(np.mean(all_r)) if all_r else 0.0

    total_reasons = sum(agg.get("reason_counter", {}).values())
    nsb = agg.get("reason_counter", {}).get("no_suitable_block", 0)
    so = (agg.get("reason_counter", {}).get("server_overload", 0) +
          agg.get("reason_counter", {}).get("server_saturated", 0))
    agg["no_suitable_block_ratio"] = nsb / max(total_reasons, 1)
    agg["server_overload_ratio"] = so / max(total_reasons, 1)

    # Per-seed blocking std
    per_seed_blk: Dict[int, List[float]] = {}
    for r in ep_results:
        seed = r.get("_seed", 0)
        per_seed_blk.setdefault(seed, []).append(r["blocking_rate"])
    seed_blk_means = [float(np.mean(v)) for v in per_seed_blk.values()]
    agg["per_seed_std"] = float(np.std(seed_blk_means)) if len(seed_blk_means) > 1 else 0.0

    return agg


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(counter: Dict[str, int]) -> str:
    c = Counter(counter); total = sum(c.values())
    if total <= 0: return "none"
    items = sorted(c.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v / total:.1%}" for k, v in items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--requests_per_episode", type=int, default=80,
                        help="Default requests per episode (overridden per scenario).")
    parser.add_argument("--arrival_interval", type=float, default=0.20)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/slots_load_sweep.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/slots_load_sweep.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]
    axes = list(SWEEP_AXES.keys()); values = list(SWEEP_AXES.values())
    grid = list(product(*values))
    total_scenarios = len(grid)

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    print(f"\nNum-Slots × Load Sweep")
    print(f"Axes: {axes}")
    print(f"Total scenarios: {total_scenarios}  Seeds: {seeds}  Eps/seed: {args.episodes}")
    print(f"Methods: {METHOD_ORDER}")
    print("=" * 70)

    # Pre-generate shared episodes (per seed, independent of num_slots)
    env_proto_base = make_env(
        topology=args.topology, num_slots=16, num_servers=args.num_servers,
        seed=42, slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy, k=3,
    )
    all_episodes: Dict[int, List[List]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto_base.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto_base, rng, src,
                num_requests=args.requests_per_episode,  # dummy, will be varied
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps

    t_start = time.time()
    all_results: List[Dict[str, Any]] = []

    for idx, combo in enumerate(grid):
        params = dict(zip(axes, combo))
        label = f"S{params['num_slots']}_R{params['requests_per_episode']}"

        scenario_args = argparse.Namespace(**{
            **vars(args),
            "num_slots": params["num_slots"],
            "requests_per_episode": params["requests_per_episode"],
        })

        # Re-generate episodes with correct requests_per_episode for this scenario
        scenario_episodes: Dict[int, List[List]] = {}
        for seed in seeds:
            rng = np.random.RandomState(seed)
            eps = []
            for _ in range(args.episodes):
                src = rng.randint(0, env_proto_base.net.NUM_NODES)
                eps.append(generate_requests(
                    env_proto_base, rng, src,
                    num_requests=params["requests_per_episode"],
                    arrival_interval=args.arrival_interval,
                    holding_min=args.holding_min, holding_max=args.holding_max,
                    deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                    size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                    edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                    num_splits=args.num_splits, split_profile=args.split_profile,
                ))
            scenario_episodes[seed] = eps

        scenario_result: Dict[str, Any] = {"scenario": label, "params": params}

        for method in METHOD_ORDER:
            ep_results = []
            for seed in seeds:
                for ep_idx, requests in enumerate(scenario_episodes[seed]):
                    res = _eval_episode(
                        method, agent_c, agent_r, requests, env_proto_base,
                        scenario_args, ep_idx, seed,
                    )
                    res["_seed"] = seed
                    ep_results.append(res)

            agg = _agg_scenario(ep_results)
            scenario_result[method] = agg

        all_results.append(scenario_result)

        # Progress
        a = scenario_result["agent"]
        elapsed = time.time() - t_start
        eta = elapsed / (idx + 1) * (total_scenarios - idx - 1)
        print(f"[{idx + 1:2d}/{total_scenarios}] {label}  "
              f"blk={a['blocking_rate']:.3f}±{a['per_seed_std']:.2f}  "
              f"noC={a['no_valid_c_rate']:.2f}  avgCa={a['avg_valid_c_actions']:.1f}  "
              f"avgRa={a['avg_valid_r_actions']:.1f}  "
              f"delay={a['avg_delay_ms']:.1f}ms  elap={elapsed:.0f}s  ETA={eta:.0f}s")

    total_elapsed = time.time() - t_start
    print(f"\nDone. Total time: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Candidate filtering
    # ------------------------------------------------------------------
    candidates = []
    for sr in all_results:
        a = sr["agent"]
        noc = a["no_valid_c_rate"]
        ablk = a["blocking_rate"]
        adelay = a["avg_delay_ms"]
        astd = a["per_seed_std"]

        # Find best heuristic
        best_heu = min(
            [m for m in METHOD_ORDER if m != "agent"],
            key=lambda m: sr[m]["blocking_rate"],
        )
        best_blk = sr[best_heu]["blocking_rate"]
        best_delay = sr[best_heu]["avg_delay_ms"]
        delta = ablk - best_blk
        delay_ratio = adelay / max(best_delay, 0.001)

        sr["_best_heuristic"] = best_heu
        sr["_best_heuristic_blk"] = best_blk
        sr["_delta_vs_best"] = delta
        sr["_delay_ratio"] = delay_ratio

        if (0.10 <= ablk <= 0.30
                and noc <= 0.20
                and delta <= -0.03  # Agent-C is at least 3pp BETTER (more negative = better)
                and delay_ratio <= 1.3
                and astd <= 0.15):
            candidates.append(sr)

    candidates.sort(key=lambda x: abs(x["agent"]["blocking_rate"] - 0.20))

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    def _clean(obj):
        if isinstance(obj, dict): return {str(k): _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        elif isinstance(obj, list): return [_clean(v) for v in obj]
        return obj

    json_output = {
        "args": vars(args), "sweep_axes": SWEEP_AXES,
        "total_scenarios": total_scenarios, "total_time_s": total_elapsed,
        "num_candidates": len(candidates),
        "top_candidates": _clean(candidates[:5]),
        "all_results": _clean(all_results),
    }
    out_json.write_text(json.dumps(json_output, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    out_md = Path(args.out_md)
    md: List[str] = []
    md.append("# Num-Slots × Load Sweep\n\n")
    md.append(f"**Topology:** {args.topology}  **Servers:** {args.num_servers}  "
              f"**k_paths:** {args.k_paths}  **MaxBlocks:** {args.max_blocks}\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}\n\n")
    md.append(f"**Total scenarios:** {total_scenarios}  "
              f"**Total time:** {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)\n\n")

    md.append("## Sweep Axes\n\n")
    for axis, vals in SWEEP_AXES.items():
        md.append(f"- **{axis}**: {vals}\n")

    # Candidate criteria
    md.append("\n## Candidate Criteria\n\n")
    md.append("- Agent-C blocking ∈ [0.10, 0.30]\n")
    md.append("- no_valid_c_rate ≤ 0.20\n")
    md.append("- Agent-C beats best heuristic by ≥ 3pp\n")
    md.append("- Agent-C delay ≤ 1.3× best heuristic delay\n")
    md.append("- Per-seed blocking std ≤ 0.15\n\n")
    md.append(f"**Passed:** {len(candidates)} / {total_scenarios}\n\n")

    # All scenarios table
    md.append("## All Scenarios\n\n")
    hdr = ("| Scenario | Slots | Req | AgentBlk | ±Std | noC% | avgCa | avgRa | "
           "DFblk | RFblk | WOblk | IWDblk | GreedyBlk | BestHeu | ΔBlk | Delay | FS |\n")
    md.append(hdr)
    md.append("|" + "---|" * 17 + "\n")
    for sr in all_results:
        a = sr["agent"]
        md.append(
            f"| {sr['scenario']} | {sr['params']['num_slots']} | {sr['params']['requests_per_episode']} "
            f"| {a['blocking_rate']:.4f} | ±{a['per_seed_std']:.3f} "
            f"| {a['no_valid_c_rate']:.2f} | {a['avg_valid_c_actions']:.1f} | {a['avg_valid_r_actions']:.1f} "
            f"| {sr['df']['blocking_rate']:.4f} | {sr['rf']['blocking_rate']:.4f} "
            f"| {sr['wo']['blocking_rate']:.4f} | {sr['iwd']['blocking_rate']:.4f} "
            f"| {sr['greedy']['blocking_rate']:.4f} "
            f"| {sr['_best_heuristic']} | {sr['_delta_vs_best']:+.4f} "
            f"| {a['avg_delay_ms']:.1f} | {a['avg_fs']:.2f} |\n"
        )

    # Top candidates
    if candidates:
        md.append(f"\n## Top Candidates (sorted by distance from 20% center)\n\n")
        for rank, sr in enumerate(candidates[:5], 1):
            a = sr["agent"]
            md.append(f"### #{rank}: {sr['scenario']}  "
                      f"(blk={a['blocking_rate']:.4f}  noC={a['no_valid_c_rate']:.2f}  "
                      f"std={a['per_seed_std']:.3f})\n\n")
            md.append(f"- Δ vs {sr['_best_heuristic']}: {sr['_delta_vs_best']:+.4f}\n")
            md.append(f"- Delay: {a['avg_delay_ms']:.1f}ms  FS: {a['avg_fs']:.2f}\n")
            for method in METHOD_ORDER:
                m = sr[method]
                md.append(f"- **{METHOD_LABELS[method]}**: blk={m['blocking_rate']:.4f}  "
                          f"failures: {_fmt_pct(m.get('reason_counter', {}))}\n")
            md.append(f"- Split dist: {_fmt_pct(a.get('split_counter', {}))}\n")
            md.append(f"- Server dist: {_fmt_pct(a.get('server_counter', {}))}\n\n")
    else:
        md.append("\n## ⚠ No candidates passed all criteria\n\n")
        md.append("snap24_gnutella_reach + complex5_v2_lite may be unsuitable. "
                  "Consider changing topology or split_profile.\n\n")

        # Show closest misses
        closest = sorted(all_results, key=lambda sr: (
            abs(sr["agent"]["blocking_rate"] - 0.20) / 0.25 +
            max(0, sr["agent"]["no_valid_c_rate"] - 0.20) / 0.10 +
            max(0, sr["agent"]["per_seed_std"] - 0.15) / 0.10 +
            max(0, sr["_delta_vs_best"]) / 0.03
        ))[:5]
        md.append("### Closest Misses\n\n")
        for sr in closest:
            a = sr["agent"]
            fails = []
            if a["blocking_rate"] < 0.10: fails.append(f"blk={a['blocking_rate']:.3f} too low")
            elif a["blocking_rate"] > 0.30: fails.append(f"blk={a['blocking_rate']:.3f} too high")
            if a["no_valid_c_rate"] > 0.20: fails.append(f"noC={a['no_valid_c_rate']:.2f} too high")
            if a["per_seed_std"] > 0.15: fails.append(f"std={a['per_seed_std']:.3f} too high")
            if sr["_delta_vs_best"] > -0.03: fails.append(f"Δ={sr['_delta_vs_best']:+.4f} insufficient")
            if sr["_delay_ratio"] > 1.3: fails.append(f"delayRatio={sr['_delay_ratio']:.2f}")
            md.append(f"- **{sr['scenario']}**: blk={a['blocking_rate']:.4f}  "
                      f"noC={a['no_valid_c_rate']:.2f}  std={a['per_seed_std']:.3f}  "
                      f"Δ={sr['_delta_vs_best']:+.4f}  "
                      f"→ fails: {', '.join(fails)}\n")

    # Recommended setting
    md.append("\n## Recommendation\n\n")
    if candidates:
        best = candidates[0]
        md.append(f"**Main experiment setting: {best['scenario']}**\n\n")
        md.append(f"- num_slots={best['params']['num_slots']}  "
                  f"requests_per_episode={best['params']['requests_per_episode']}\n")
        md.append(f"- Agent-C blocking: {best['agent']['blocking_rate']:.4f}\n")
    else:
        md.append("No viable setting found in current sweep space.\n\n")
        md.append("**Options:**\n")
        md.append("1. Try different topology (e.g., metro24_c_sensitive)\n")
        md.append("2. Try different split_profile (e.g., complex5_v2 instead of lite)\n")
        md.append("3. Try num_slots beyond [16,32] range\n")
        md.append("4. Try different arrival_interval (e.g., 0.15 or 0.25)\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Total scenarios: {total_scenarios}  Candidates: {len(candidates)}")
    if candidates:
        for rank, sr in enumerate(candidates[:5], 1):
            a = sr["agent"]
            print(f"  #{rank}: {sr['scenario']}  blk={a['blocking_rate']:.4f}  "
                  f"noC={a['no_valid_c_rate']:.2f}  std={a['per_seed_std']:.3f}  "
                  f"Δ={sr['_delta_vs_best']:+.4f}")
        print(f"\nRecommended: {candidates[0]['scenario']}")
    else:
        print("No candidates! Seeing closest misses...")
        for sr in sorted(all_results, key=lambda s: s["agent"]["blocking_rate"])[:3]:
            a = sr["agent"]
            print(f"  {sr['scenario']}: blk={a['blocking_rate']:.4f}  noC={a['no_valid_c_rate']:.2f}  "
                  f"std={a['per_seed_std']:.3f}  Δ={sr['_delta_vs_best']:+.4f}")


if __name__ == "__main__":
    main()
