"""RefKL-11D-PPO-R vs BC-PPO-R — Formal S20 Evaluation.

Compares two R checkpoints under the SAME generated episodes across
6 C-action methods.  Fixed topology / slots / traffic.

Usage::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_refkl_r_s20
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
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
# Config
# ---------------------------------------------------------------------------

C_METHODS = ["agent", "iwd", "df", "wo", "rf", "greedy"]

C_LABELS = {
    "agent":  "Agent-C",
    "iwd":    "IWD-C",
    "df":     "DF-C",
    "wo":     "WO-C",
    "rf":     "RF-C",
    "greedy": "Greedy-C",
}

R_CHECKPOINTS = {
    "BC-PPO-R (old)":         "sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt",
    "RefKL-11D-PPO-R (new)":  "sa_hmarl/checkpoints/ppo_r_refkl_11d_s20_best.pt",
}

FIXED = dict(
    topology="snap24_gnutella_reach", num_slots=20, num_servers=4,
    k_paths=5, max_blocks=10, block_sort_strategy="mixed",
    modulation_profile="default",
    split_profile="complex5_v2_lite", num_splits=5,
    slot_bw_hz=1.25e9, guard_band_fs=1,
    requests_per_episode=80, arrival_interval=0.25,
    holding_min=4.0, holding_max=10.0,
    deadline_min=30.0, deadline_max=100.0,
    size_min_mb=5.0, size_max_mb=30.0,
    edge_cost_min=0.5, edge_cost_max=15.0,
    waste_coef=0.8,
    seeds="42,123,456,789,2024", episodes_per_seed=5,
    agent_c_checkpoint="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt",
)

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


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu", label: str = "") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    fm_r = ckpt.get("agent_r_feature_mode", ckpt.get("args", {}).get("agent_r_feature_mode", "default"))
    idim = ckpt.get("input_dim", 11)
    print(f"  Loaded {label}: input_dim={idim} feature_mode={fm_r}")
    agent = PPOAgentR(
        input_dim=idim, mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device, feature_mode=fm_r,
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
    c_method: str, agent_c: PPOAgentC, agent_r: PPOAgentR,
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

        if c_method == "agent":
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
                c_method, env, req, obs_c, raw_mask,
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
        "blocking_rate": blocked / n, "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s, "avg_fs": total_fs / s,
        "avg_waste": total_waste / s, "avg_path_len_km": total_path / s,
        "reason_counter": dict(reason_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "mod_counter": dict(mod_counter),
        "mask_valid_c_counts": mask_valid_c_counts,
        "mask_valid_r_counts": mask_valid_r_counts,
        "no_valid_c_rate_ep": mask_zero_c_count / n,
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

def _fmt_pct(counter: Dict[str, int]) -> str:
    c = Counter(counter); total = sum(c.values())
    if total <= 0: return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in sorted(c.items(), key=lambda x: -x[1]))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="RefKL-11D-PPO-R vs BC-PPO-R — S20 Evaluation")
    for k, v in FIXED.items():
        t = type(v); flag = f"--{k}"
        if t == bool:
            parser.add_argument(flag, default=v, action="store_true" if not v else "store_false")
        else:
            parser.add_argument(flag, type=t, default=v)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--out_json", type=str, default="sa_hmarl/experiments/refkl_r_s20_eval.json")
    parser.add_argument("--out_md", type=str, default="sa_hmarl/experiments/refkl_r_s20_eval.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    # Load C agent once
    mod_reg = ModulationRegistry.from_profile("default")
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"  input_dim={agent_c.input_dim} feature_mode={agent_c.feature_mode}")

    # Pre-generate episodes (shared across R checkpoints)
    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=1.25e9, guard_band_fs=1,
        modulation_profile="default",
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    all_episodes: Dict[int, List[List]] = {}
    source_nodes: Dict[int, int] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed); eps = []
        for _ in range(args.episodes_per_seed):
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
        source_nodes[seed] = eps[0][0].src_node if eps and eps[0] else -1

    print(f"\n{'=' * 80}")
    print("RefKL-11D-PPO-R vs BC-PPO-R — S20 Formal Evaluation")
    print(f"Seeds: {seeds}  Eps/seed: {args.episodes_per_seed}  Req/ep: {args.requests_per_episode}")
    print(f"Source nodes: {source_nodes}")
    print(f"C methods: {C_METHODS}")
    print(f"{'=' * 80}")

    t_start = time.time()

    # Results: {r_label: {c_label: agg_dict}}
    all_data: Dict[str, Dict[str, Any]] = {}

    for r_label, r_ckpt in R_CHECKPOINTS.items():
        print(f"\n{'#' * 70}")
        print(f"# R: {r_label}")
        print(f"{'#' * 70}")

        agent_r = _load_ppo_r(r_ckpt, mod_reg, args.device, r_label)

        r_data: Dict[str, Any] = {}

        for c_method in C_METHODS:
            c_label = C_LABELS[c_method]
            t0 = time.time()

            ep_results = []
            for seed in seeds:
                for ep_idx, requests in enumerate(all_episodes[seed]):
                    res = _eval_episode(
                        c_method, agent_c, agent_r, requests, env_proto, args, ep_idx, seed,
                    )
                    ep_results.append(res)

            # Aggregate
            scalar_keys = [
                "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
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

            # Mask stats
            all_c, zero_c, req_c = [], 0, 0
            all_r = []
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

            # Failure ratios
            total_r = sum(agg.get("reason_counter", {}).values())
            nsb = agg.get("reason_counter", {}).get("no_suitable_block", 0)
            so = (agg.get("reason_counter", {}).get("server_overload", 0) +
                  agg.get("reason_counter", {}).get("server_saturated", 0))
            agg["no_suitable_block_ratio"] = nsb / max(total_r, 1) if total_r > 0 else 0.0
            agg["server_overload_ratio"] = so / max(total_r, 1) if total_r > 0 else 0.0

            r_data[c_label] = agg
            elapsed = time.time() - t0
            print(
                f"  {c_label:10s} blk={agg['blocking_rate']:.4f}±{agg['blocking_rate_std']:.3f}  "
                f"delay={agg['avg_delay_ms']:.1f}ms  fs={agg['avg_fs']:.2f}  "
                f"noC={agg['no_valid_c_rate']:.1%}  avgRa={agg['avg_valid_r_actions']:.0f}  "
                f"NSB={agg['no_suitable_block_ratio']:.1%}  "
                f"t={elapsed:.0f}s"
            )

        all_data[r_label] = r_data

    total_elapsed = time.time() - t_start
    print(f"\nTotal: {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)")

    # ------------------------------------------------------------------
    # Console comparison table
    # ------------------------------------------------------------------
    print(f"\n{'=' * 90}")
    print("CORE COMPARISON")
    print(f"{'=' * 90}")
    hdr = (f"{'C Method':12s} | {'Old R blk':>9s} | {'New R blk':>9s} | {'ΔBlk':>7s} | "
           f"{'Old del':>7s} | {'New del':>7s} | {'Old FS':>6s} | {'New FS':>6s}")
    print(hdr); print("-" * 90)
    for c_method in C_METHODS:
        c_label = C_LABELS[c_method]
        old = all_data["BC-PPO-R (old)"][c_label]
        new = all_data["RefKL-11D-PPO-R (new)"][c_label]
        d_blk = old["blocking_rate"] - new["blocking_rate"]
        print(f"{c_label:12s} | {old['blocking_rate']:9.4f} | {new['blocking_rate']:9.4f} | "
              f"{d_blk:+7.4f} | {old['avg_delay_ms']:6.1f} | {new['avg_delay_ms']:6.1f} | "
              f"{old['avg_fs']:6.2f} | {new['avg_fs']:6.2f}")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    def _clean(obj):
        if isinstance(obj, dict): return {str(k): _clean(v) for k, v in obj.items()}
        elif isinstance(obj, (np.floating, np.integer)): return float(obj)
        elif isinstance(obj, np.ndarray): return obj.tolist()
        return obj
    out_json.write_text(json.dumps({
        "args": vars(args), "r_checkpoints": R_CHECKPOINTS, "c_methods": C_METHODS,
        "source_nodes": source_nodes, "total_time_s": total_elapsed,
        "results": _clean(all_data),
    }, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    out_md = Path(args.out_md); out_md.parent.mkdir(parents=True, exist_ok=True)
    md: List[str] = []
    md.append("# RefKL-11D-PPO-R vs BC-PPO-R — S20 Formal Evaluation\n\n")
    md.append(f"**Topology:** {args.topology}  **Slots:** {args.num_slots}  "
              f"**Servers:** {args.num_servers}  **k_paths:** {args.k_paths}\n\n")
    md.append(f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes_per_seed}  "
              f"**Req/ep:** {args.requests_per_episode}  "
              f"**Arrival:** {args.arrival_interval}s\n\n")
    md.append(f"**Agent-C:** `{args.agent_c_checkpoint}`\n\n")
    md.append(f"**Source nodes per seed:** {source_nodes}\n\n")
    md.append(f"**Total time:** {total_elapsed:.0f}s ({total_elapsed / 60:.1f} min)\n\n")

    # Core comparison
    md.append("## Core Comparison: Old BC-PPO-R vs New RefKL-11D-PPO-R\n\n")
    md.append("| C Method | Old Blk | New Blk | ΔBlk | Old Delay | New Delay | Old FS | New FS | Old NSB | New NSB |\n")
    md.append("|----------|---------|---------|------|-----------|-----------|--------|--------|---------|--------|\n")
    for c_method in C_METHODS:
        c_label = C_LABELS[c_method]
        old = all_data["BC-PPO-R (old)"][c_label]
        new = all_data["RefKL-11D-PPO-R (new)"][c_label]
        d_blk = old["blocking_rate"] - new["blocking_rate"]
        md.append(
            f"| {c_label} | {old['blocking_rate']:.4f} | {new['blocking_rate']:.4f} | "
            f"**{d_blk:+.4f}** | {old['avg_delay_ms']:.1f} | {new['avg_delay_ms']:.1f} | "
            f"{old['avg_fs']:.2f} | {new['avg_fs']:.2f} | "
            f"{old['no_suitable_block_ratio']:.1%} | {new['no_suitable_block_ratio']:.1%} |\n"
        )

    # Full detail per R checkpoint
    for r_label in R_CHECKPOINTS:
        md.append(f"\n## Full Results: {r_label}\n\n")
        md.append("| C Method | Blocking | Success | Reward | Delay | FS | Waste | PathKm | noC% | avgRa | NSB% | SvOv% |\n")
        md.append("|----------|----------|---------|--------|-------|----|-------|--------|------|-------|------|-------|\n")
        for c_method in C_METHODS:
            c_label = C_LABELS[c_method]
            r = all_data[r_label][c_label]
            md.append(
                f"| {c_label} | {r['blocking_rate']:.4f}±{r['blocking_rate_std']:.3f} | "
                f"{r['success_rate']:.4f} | {r['avg_reward']:+.3f} | "
                f"{r['avg_delay_ms']:.1f} | {r['avg_fs']:.2f} | {r['avg_waste']:.3f} | "
                f"{r['avg_path_len_km']:.1f} | {r['no_valid_c_rate']:.1%} | "
                f"{r['avg_valid_r_actions']:.0f} | {r['no_suitable_block_ratio']:.1%} | "
                f"{r['server_overload_ratio']:.1%} |\n"
            )

        # Failure reasons
        md.append(f"\n### Failure Reasons — {r_label}\n\n")
        for c_method in C_METHODS:
            c_label = C_LABELS[c_method]
            r = all_data[r_label][c_label]
            md.append(f"- **{c_label}**: {_fmt_pct(r.get('reason_counter', {}))}\n")

        # Split distribution
        md.append(f"\n### Split Distribution — {r_label}\n\n")
        for c_method in C_METHODS:
            c_label = C_LABELS[c_method]
            r = all_data[r_label][c_label]
            md.append(f"- **{c_label}**: {_fmt_pct(r.get('split_counter', {}))}\n")

        # Server distribution
        md.append(f"\n### Server Distribution — {r_label}\n\n")
        for c_method in C_METHODS:
            c_label = C_LABELS[c_method]
            r = all_data[r_label][c_label]
            md.append(f"- **{c_label}**: {_fmt_pct(r.get('server_counter', {}))}\n")

        # Modulation
        md.append(f"\n### Modulation — {r_label}\n\n")
        for c_method in C_METHODS:
            c_label = C_LABELS[c_method]
            r = all_data[r_label][c_label]
            md.append(f"- **{c_label}**: {_fmt_pct(r.get('mod_counter', {}))}\n")

    # Verdict
    md.append("\n## Verdict\n\n")
    agent_old = all_data["BC-PPO-R (old)"][C_LABELS["agent"]]
    agent_new = all_data["RefKL-11D-PPO-R (new)"][C_LABELS["agent"]]
    d_blk = agent_old["blocking_rate"] - agent_new["blocking_rate"]
    d_delay = agent_new["avg_delay_ms"] - agent_old["avg_delay_ms"]
    d_fs = agent_new["avg_fs"] - agent_old["avg_fs"]

    md.append(f"- **Agent-C Δblocking (old → new): {d_blk:+.4f}**\n")
    md.append(f"- Agent-C Δdelay: {d_delay:+.1f} ms\n")
    md.append(f"- Agent-C ΔFS: {d_fs:+.2f}\n\n")

    if d_blk >= 0.02:
        md.append("### ✓ RefKL-11D-PPO-R UPGRADE RECOMMENDED\n\n")
        md.append(f"New R checkpoint reduces Agent-C blocking by {d_blk:.1%} (≥ 2pp).\n")
        md.append(f"Delay change {d_delay:+.1f}ms, FS change {d_fs:+.2f}.\n\n")
        md.append("**Upgrade main method to: Enhanced Agent-C + RefKL-11D-PPO-R**\n")
    elif d_blk > 0.005:
        md.append("### ~ RefKL-11D-PPO-R marginal improvement\n\n")
        md.append(f"New R reduces blocking by {d_blk:.1%} (< 2pp threshold). "
                  "Consider further R training or retain old BC-PPO-R.\n")
    else:
        md.append("### ✗ RefKL-11D-PPO-R does NOT improve over BC-PPO-R\n\n")
        md.append(f"Δblocking = {d_blk:+.4f} — no improvement or regression.\n")
        md.append("**Retain original: Enhanced Agent-C + BC-PPO-R**\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # Console verdict
    print(f"\n{'=' * 60}")
    print("VERDICT")
    print(f"{'=' * 60}")
    print(f"Agent-C + old R:   blk={agent_old['blocking_rate']:.4f}  delay={agent_old['avg_delay_ms']:.1f}ms  fs={agent_old['avg_fs']:.2f}")
    print(f"Agent-C + new R:   blk={agent_new['blocking_rate']:.4f}  delay={agent_new['avg_delay_ms']:.1f}ms  fs={agent_new['avg_fs']:.2f}")
    print(f"Δblocking = {d_blk:+.4f}  Δdelay = {d_delay:+.1f}ms  ΔFS = {d_fs:+.2f}")
    if d_blk >= 0.02:
        print("✓ UPGRADE to RefKL-11D-PPO-R")
    elif d_blk > 0.005:
        print("~ Marginal — consider further R training")
    else:
        print("✗ RETAIN BC-PPO-R — no improvement")


if __name__ == "__main__":
    main()
