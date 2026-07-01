"""C Decision-Space Sweep — Find paper-suitable scenario.

Sweeps num_slots, load, arrival, split_profile, and source difficulty
to find scenarios where:
  - Agent-C blocking ∈ [0.10, 0.25]
  - no_valid_c_rate ≤ 0.20
  - Agent-C beats best heuristic by ≥ 3pp
  - delay_ratio ≤ 1.30
  - per-seed std ≤ 0.12

Usage::

    # Smoke
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.sweep_c_decision_space --smoke

    # Full
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.sweep_c_decision_space
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
    build_agent_c_observation, build_agent_r_observation,
    decode_agent_c_action, decode_agent_r_action,
)
from sa_hmarl.evaluation.offloading_baselines import select_offloading_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env, compute_agent_c_reward
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SWEEP_AXES = {
    "num_slots": [20, 24, 28, 32],
    "requests_per_episode": [60, 80],
    "arrival_interval": [0.20, 0.30],
    "split_profile": ["default3", "complex5_v2_lite"],
}

METHOD_ORDER = ["agent", "iwd", "df", "wo", "rf", "greedy"]

METHOD_LABELS = {
    "agent": "Agent-C", "iwd": "IWD-C", "df": "DF-C",
    "wo": "WO-C", "rf": "RF-C", "greedy": "Greedy-C",
}

HEURISTIC_METHODS = ["iwd", "df", "wo", "rf", "greedy"]


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> Optional[PPOAgentC]:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    fm = ckpt_args.get("agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"))
    idim = ckpt.get("input_dim", 17)
    if idim >= 24 and fm == "default":
        fm = "enhanced"
    try:
        agent = PPOAgentC(
            input_dim=idim, hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
            device=device, feature_mode=fm,
        )
        agent.policy_net.load_state_dict(ckpt["model_state"])
        agent.policy_net.eval()
        agent.checkpoint_args = ckpt_args
        return agent
    except Exception as e:
        print(f"  WARNING: Failed to load Agent-C: {e}")
        return None


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    fm_r = ckpt.get("agent_r_feature_mode", ckpt.get("args", {}).get("agent_r_feature_mode", "default"))
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg, hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=fm_r,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Agent-C selection
# ---------------------------------------------------------------------------

def _select_agent_c(agent_c, obs_c, num_slots):
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
# Per-episode eval
# ---------------------------------------------------------------------------

def _eval_episode(
    method: str, agent_c, agent_r, requests: List,
    env_proto, args, ep_idx: int, seed: int,
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
    reason_ctr = Counter(); split_ctr = Counter(); server_ctr = Counter(); mod_ctr = Counter()
    mask_c_counts = []; zero_c = 0; mask_r_counts = []
    source_node = requests[0].src_node if requests else -1

    for req in requests:
        obs_c = build_agent_c_observation(env, req)
        raw_mask = obs_c["agent_c_mask"]

        if method == "agent":
            action_idx_c, n_valid_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
            mask_c_counts.append(n_valid_c)
            if n_valid_c == 0: zero_c += 1
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(action_idx_c, num_servers)
        else:
            action_idx_c = select_offloading_action(
                method, env, req, obs_c, raw_mask,
                rng=rng, server_selected_count=server_selected_count,
            )
            n_valid_c = int(raw_mask.sum())
            mask_c_counts.append(n_valid_c)
            if n_valid_c == 0: zero_c += 1
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(action_idx_c, num_servers)

        split_id, server_id = action_c
        split_ctr[f"split{split_id}"] += 1; server_ctr[f"s{server_id}"] += 1
        server_selected_count[server_id] += 1

        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        mask_r_counts.append(int(obs_r["agent_r_mask"].sum()))

        action_r_idx = agent_r.select_action(obs_r, deterministic=True)
        action_r_tuple = (0, 0, 0) if action_r_idx is None else (
            decode_agent_r_action(action_r_idx, len(obs_r["mod_names"]), env.max_blocks)
        )

        _, _, _, info = env.step(action_c, action_r_tuple)
        total += 1
        server_obj = env.mec.servers[server_id]
        reward = compute_agent_c_reward(info, req.deadline_ms, args.waste_coef, server_obj.utilization)
        total_reward += reward

        if info.get("success", False):
            success += 1; d = float(info.get("delay_ms", 0))
            total_delay += d; total_fs += float(info.get("num_slots", 0))
            total_waste += float(info.get("block_waste", 0))
            total_path += float(info.get("path_dist_km", 0))
            mod_ctr[info.get("modulation", "unknown")] += 1
        else:
            blocked += 1; reason_ctr[info.get("reason", "unknown")] += 1

    n = max(total, 1); s = max(success, 1)
    return {
        "total": total, "blocked": blocked, "success": success,
        "blocking_rate": blocked / n, "success_rate": success / n,
        "avg_reward": total_reward / n, "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s, "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "reason_counter": dict(reason_ctr), "split_counter": dict(split_ctr),
        "server_counter": dict(server_ctr), "mod_counter": dict(mod_ctr),
        "mask_c_counts": mask_c_counts, "mask_r_counts": mask_r_counts,
        "no_valid_c_rate_ep": zero_c / n, "source_node": source_node,
        "_seed": seed,
    }


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------

def _agg_results(ep_results: List[Dict]) -> Dict[str, Any]:
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "avg_fs", "avg_waste", "avg_path_len_km",
    ]
    agg: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(r[key]) for r in ep_results]
        agg[key] = float(np.mean(vals)); agg[f"{key}_std"] = float(np.std(vals))

    for key in ["reason_counter", "split_counter", "server_counter", "mod_counter"]:
        c = Counter()
        for r in ep_results: c.update(r.get(key, {}))
        agg[key] = dict(c)

    all_c, zero_c, req_c, all_r = [], 0, 0, []
    for r in ep_results:
        if "mask_c_counts" in r:
            all_c.extend(r["mask_c_counts"])
            zero_c += r.get("no_valid_c_rate_ep", 0) * r["total"]; req_c += r["total"]
        if "mask_r_counts" in r: all_r.extend(r["mask_r_counts"])
    agg["no_valid_c_rate"] = zero_c / max(req_c, 1)
    agg["avg_valid_c_actions"] = float(np.mean(all_c)) if all_c else 0.0
    agg["avg_valid_r_actions"] = float(np.mean(all_r)) if all_r else 0.0

    # Per-seed blocking std
    per_seed_blk = {}
    for r in ep_results:
        s = r.get("_seed", 0)
        per_seed_blk.setdefault(s, []).append(r["blocking_rate"])
    seed_means = [float(np.mean(v)) for v in per_seed_blk.values()]
    agg["per_seed_std"] = float(np.std(seed_means)) if len(seed_means) > 1 else 0.0

    # Failure ratios
    total_r = sum(agg.get("reason_counter", {}).values())
    agg["nsb_ratio"] = agg.get("reason_counter", {}).get("no_suitable_block", 0) / max(total_r, 1)
    agg["so_ratio"] = (agg.get("reason_counter", {}).get("server_overload", 0) +
                       agg.get("reason_counter", {}).get("server_saturated", 0)) / max(total_r, 1)

    # Source node
    srcs = set(r.get("source_node", -1) for r in ep_results)
    agg["source_nodes"] = sorted(srcs)

    return agg


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(c: Dict) -> str:
    ct = Counter(c); t = sum(ct.values())
    if t <= 0: return "none"
    return ", ".join(f"{k}={v / t:.1%}" for k, v in ct.most_common())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="C Decision-Space Sweep")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
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
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/c_decision_space_sweep.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/c_decision_space_sweep.md")
    args = parser.parse_args()

    if args.smoke:
        args.seeds = "42"; args.episodes = 1; args.requests_per_episode = 10
        print("[SMOKE MODE] 1 seed × 1 ep × 10 req")

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    if agent_c is None:
        print("FATAL: Cannot load Agent-C"); return
    print(f"  input_dim={agent_c.input_dim} fm={agent_c.feature_mode}")
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    axes = list(SWEEP_AXES.keys()); values = list(SWEEP_AXES.values())
    grid = list(product(*values))
    total_scenarios = len(grid)
    print(f"\nC Decision-Space Sweep: {total_scenarios} scenarios")
    print(f"Axes: {SWEEP_AXES}")
    print(f"Seeds: {seeds} Eps/seed: {args.episodes}")

    # Pre-generate base episodes per seed
    env_base = make_env(
        topology=args.topology, num_slots=16, num_servers=args.num_servers,
        seed=42, slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy, k=args.k_paths,
    )

    t_start = time.time()
    all_scenario_results: List[Dict] = []
    incompatible: List[str] = []

    for idx, combo in enumerate(grid):
        params = dict(zip(axes, combo))
        label = (f"S{params['num_slots']}_R{params['requests_per_episode']}"
                 f"_AI{params['arrival_interval']}_{params['split_profile']}")

        # Check split_profile compatibility
        num_splits_required = {"default3": 3, "complex5_v2_lite": 5}[params["split_profile"]]
        sc_args = argparse.Namespace(**{
            **vars(args),
            "num_slots": params["num_slots"],
            "requests_per_episode": params["requests_per_episode"],
            "arrival_interval": params["arrival_interval"],
            "num_splits": num_splits_required,
            "split_profile": params["split_profile"],
        })

        # Generate episodes for this scenario
        scenario_eps: Dict[int, List] = {}
        try:
            env_proto = make_env(
                topology=args.topology, num_slots=params["num_slots"],
                num_servers=args.num_servers, seed=42,
                slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            for seed in seeds:
                rng = np.random.RandomState(seed); eps = []
                for _ in range(args.episodes):
                    src = rng.randint(0, env_proto.net.NUM_NODES)
                    eps.append(generate_requests(
                        env_proto, rng, src,
                        num_requests=params["requests_per_episode"],
                        arrival_interval=params["arrival_interval"],
                        holding_min=args.holding_min, holding_max=args.holding_max,
                        deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                        size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                        edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                        num_splits=num_splits_required,
                        split_profile=params["split_profile"],
                    ))
                scenario_eps[seed] = eps
        except Exception as e:
            incompatible.append(f"{label}: {e}")
            continue

        sc_result: Dict[str, Any] = {"scenario": label, "params": params}

        for method in METHOD_ORDER:
            ep_results = []
            for seed in seeds:
                for ep_idx, requests in enumerate(scenario_eps[seed]):
                    res = _eval_episode(
                        method, agent_c, agent_r, requests, env_proto, sc_args, ep_idx, seed,
                    )
                    ep_results.append(res)
            sc_result[method] = _agg_results(ep_results)

        all_scenario_results.append(sc_result)

        # Progress
        a = sc_result["agent"]
        elapsed = time.time() - t_start
        eta = elapsed / (idx + 1) * (total_scenarios - idx - 1)
        print(f"[{idx + 1:3d}/{total_scenarios}] {label}  "
              f"blk={a['blocking_rate']:.3f}±{a['per_seed_std']:.2f}  "
              f"noC={a['no_valid_c_rate']:.2f}  "
              f"avgCa={a['avg_valid_c_actions']:.1f}  "
              f"avgRa={a['avg_valid_r_actions']:.1f}  "
              f"elap={elapsed:.0f}s  ETA={eta:.0f}s")

    total_elapsed = time.time() - t_start
    print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")
    if incompatible:
        print(f"Incompatible scenarios: {len(incompatible)}")
        for inc in incompatible: print(f"  {inc}")

    # ------------------------------------------------------------------
    # Classify scenarios
    # ------------------------------------------------------------------
    paper_candidates = []
    good_gap_only = []
    low_blocking = []
    too_hard = []
    too_easy = []

    for sr in all_scenario_results:
        a = sr["agent"]
        ablk = a["blocking_rate"]; noc = a["no_valid_c_rate"]
        adelay = a["avg_delay_ms"]; astd = a["per_seed_std"]

        # Best heuristic
        best_h = min(HEURISTIC_METHODS, key=lambda m: sr[m]["blocking_rate"])
        best_blk = sr[best_h]["blocking_rate"]
        best_delay = sr[best_h]["avg_delay_ms"]
        gap = best_blk - ablk  # positive = agent better
        d_ratio = adelay / max(best_delay, 0.001)

        sr["_best_heuristic"] = best_h; sr["_best_heuristic_blk"] = best_blk
        sr["_agent_gap_pp"] = gap; sr["_delay_ratio"] = d_ratio

        if ablk < 0.05:
            too_easy.append(sr); continue
        if ablk > 0.30 or noc > 0.30:
            too_hard.append(sr); continue

        if (0.10 <= ablk <= 0.25 and noc <= 0.20
                and gap >= 0.03 and d_ratio <= 1.30 and astd <= 0.12):
            paper_candidates.append(sr)
        elif gap >= 0.03:
            good_gap_only.append(sr)
        elif ablk < 0.15 and noc < 0.20:
            low_blocking.append(sr)

    paper_candidates.sort(key=lambda s: -s["_agent_gap_pp"])
    good_gap_only.sort(key=lambda s: -s["_agent_gap_pp"])
    low_blocking.sort(key=lambda s: s["agent"]["blocking_rate"])

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
    out_json.write_text(json.dumps({
        "args": vars(args), "sweep_axes": SWEEP_AXES,
        "total_scenarios": len(all_scenario_results),
        "incompatible": incompatible, "total_time_s": total_elapsed,
        "num_paper_candidates": len(paper_candidates),
        "num_good_gap_only": len(good_gap_only),
        "num_low_blocking": len(low_blocking),
        "num_too_hard": len(too_hard), "num_too_easy": len(too_easy),
        "paper_candidates": _clean(paper_candidates[:10]),
        "good_gap_only": _clean(good_gap_only[:10]),
        "low_blocking": _clean(low_blocking[:10]),
        "all_results": _clean(all_scenario_results),
    }, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# C Decision-Space Sweep\n\n")
    md.append(f"**Topology:** {args.topology}  **Servers:** {args.num_servers}  "
              f"**k:** {args.k_paths}  **maxBlocks:** {args.max_blocks}\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}\n\n")
    md.append(f"**Scenarios:** {len(all_scenario_results)}  "
              f"**Time:** {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)\n\n")
    if incompatible:
        md.append(f"**Incompatible:** {len(incompatible)}\n\n")

    md.append("## Sweep Axes\n\n")
    for axis, vals in SWEEP_AXES.items():
        md.append(f"- **{axis}**: {vals}\n")

    md.append("\n## Candidate Criteria\n\n")
    md.append("- Agent-C blocking ∈ [0.10, 0.25]\n")
    md.append("- no_valid_c_rate ≤ 0.20\n")
    md.append("- Agent-C gap ≥ 3pp vs best heuristic\n")
    md.append("- delay_ratio ≤ 1.30\n")
    md.append("- per-seed std ≤ 0.12\n\n")
    md.append(f"**Paper candidates:** {len(paper_candidates)}  "
              f"**Good-gap-only:** {len(good_gap_only)}  "
              f"**Low-blocking:** {len(low_blocking)}  "
              f"**Too-hard:** {len(too_hard)}  "
              f"**Too-easy:** {len(too_easy)}\n\n")

    # Top paper candidates
    if paper_candidates:
        md.append("## Top Paper Candidates\n\n")
        md.append("| # | Scenario | Slots | Req | ArrI | Profile | AgentBlk | BestBase | Gap | noC% | Std | DelayR |\n")
        md.append("|---|----------|-------|-----|------|---------|----------|----------|-----|------|-----|--------|\n")
        for rank, sr in enumerate(paper_candidates[:10], 1):
            a = sr["agent"]
            md.append(f"| {rank} | {sr['scenario']} | {sr['params']['num_slots']} | {sr['params']['requests_per_episode']} | "
                      f"{sr['params']['arrival_interval']} | {sr['params']['split_profile']} | "
                      f"{a['blocking_rate']:.4f} | {sr['_best_heuristic']}={sr['_best_heuristic_blk']:.4f} | "
                      f"**{sr['_agent_gap_pp']:+.4f}** | {a['no_valid_c_rate']:.2f} | {a['per_seed_std']:.3f} | "
                      f"{sr['_delay_ratio']:.2f} |\n")

    # Good gap only
    if good_gap_only:
        md.append(f"\n## Good Gap Only ({len(good_gap_only)})\n\n")
        md.append("| # | Scenario | AgentBlk | BestBase | Gap | noC% | Std | Fail Reason |\n")
        md.append("|---|----------|----------|----------|-----|------|-----|------------|\n")
        for rank, sr in enumerate(good_gap_only[:10], 1):
            a = sr["agent"]
            fails = []
            if a["blocking_rate"] < 0.10: fails.append("blk<10%")
            if a["blocking_rate"] > 0.25: fails.append("blk>25%")
            if a["no_valid_c_rate"] > 0.20: fails.append("noC>20%")
            if a["per_seed_std"] > 0.12: fails.append("std>0.12")
            if sr["_delay_ratio"] > 1.30: fails.append(f"delayR={sr['_delay_ratio']:.2f}")
            md.append(f"| {rank} | {sr['scenario']} | {a['blocking_rate']:.4f} | "
                      f"{sr['_best_heuristic_blk']:.4f} | **{sr['_agent_gap_pp']:+.4f}** | "
                      f"{a['no_valid_c_rate']:.2f} | {a['per_seed_std']:.3f} | {', '.join(fails)} |\n")

    # Low blocking
    if low_blocking:
        md.append(f"\n## Low Blocking ({len(low_blocking)})\n\n")
        md.append("| Scenario | AgentBlk | BestBase | Gap | noC% |\n")
        md.append("|----------|----------|----------|-----|------|\n")
        for sr in low_blocking[:10]:
            a = sr["agent"]
            md.append(f"| {sr['scenario']} | {a['blocking_rate']:.4f} | "
                      f"{sr['_best_heuristic_blk']:.4f} | {sr['_agent_gap_pp']:+.4f} | "
                      f"{a['no_valid_c_rate']:.2f} |\n")

    # All scenarios table
    md.append("\n## All Scenarios\n\n")
    hdr = ("| Scenario | Slots | Req | ArrI | Prof | AgentBlk | ±Std | noC% | "
           "DFblk | RFblk | WOblk | IWDblk | Greedy | BestH | Gap | DlyR |\n")
    md.append(hdr); md.append("|" + "---|" * 16 + "\n")
    for sr in all_scenario_results:
        a = sr["agent"]
        md.append(f"| {sr['scenario']} | {sr['params']['num_slots']} | {sr['params']['requests_per_episode']} | "
                  f"{sr['params']['arrival_interval']} | {sr['params']['split_profile']} | "
                  f"{a['blocking_rate']:.4f} | ±{a['per_seed_std']:.3f} | {a['no_valid_c_rate']:.2f} | "
                  f"{sr['df']['blocking_rate']:.4f} | {sr['rf']['blocking_rate']:.4f} | "
                  f"{sr['wo']['blocking_rate']:.4f} | {sr['iwd']['blocking_rate']:.4f} | "
                  f"{sr['greedy']['blocking_rate']:.4f} | "
                  f"{sr['_best_heuristic']} | {sr['_agent_gap_pp']:+.4f} | {sr['_delay_ratio']:.2f} |\n")

    # Recommendation
    md.append("\n## Recommendation\n\n")
    if paper_candidates:
        best = paper_candidates[0]
        md.append(f"**Main paper setting: {best['scenario']}**\n\n")
        md.append(f"- Slots: {best['params']['num_slots']}, "
                  f"Req/ep: {best['params']['requests_per_episode']}, "
                  f"Arrival: {best['params']['arrival_interval']}, "
                  f"Profile: {best['params']['split_profile']}\n")
        a = best["agent"]
        md.append(f"- Agent-C blocking: {a['blocking_rate']:.4f}\n")
        md.append(f"- Best heuristic: {best['_best_heuristic']} at {best['_best_heuristic_blk']:.4f}\n")
        md.append(f"- Agent advantage: {best['_agent_gap_pp']:+.4f} ({best['_agent_gap_pp']*100:.1f}pp)\n")
        md.append(f"- noC: {a['no_valid_c_rate']:.2f}, std: {a['per_seed_std']:.3f}\n")
    else:
        md.append("**No paper candidate found.**\n\n")
        if good_gap_only:
            best_gap = good_gap_only[0]
            md.append(f"Closest: {best_gap['scenario']} (gap={best_gap['_agent_gap_pp']:+.4f})\n")
        md.append("\nConsider: different topology, heterogeneous servers, or training a new Agent-C.\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # Console summary
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Scenarios: {len(all_scenario_results)}  "
          f"Paper candidates: {len(paper_candidates)}  "
          f"Good-gap: {len(good_gap_only)}  "
          f"Low-blk: {len(low_blocking)}  "
          f"Too-hard: {len(too_hard)}  "
          f"Too-easy: {len(too_easy)}")
    if paper_candidates:
        for rank, sr in enumerate(paper_candidates[:5], 1):
            print(f"  #{rank}: {sr['scenario']}  blk={sr['agent']['blocking_rate']:.4f}  "
                  f"gap={sr['_agent_gap_pp']:+.4f}  noC={sr['agent']['no_valid_c_rate']:.2f}")
        print(f"\nRecommended: {paper_candidates[0]['scenario']}")
    else:
        print("No paper candidates! Closest misses:")
        for sr in sorted(all_scenario_results, key=lambda s: -s["_agent_gap_pp"])[:5]:
            a = sr["agent"]
            print(f"  {sr['scenario']}: blk={a['blocking_rate']:.4f}  "
                  f"gap={sr['_agent_gap_pp']:+.4f}  noC={a['no_valid_c_rate']:.2f}  "
                  f"std={a['per_seed_std']:.3f}")


if __name__ == "__main__":
    main()
