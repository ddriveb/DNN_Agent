"""Lookahead-C Oracle score-weight ablation sweep.

Tests 5 score-weight configurations (A–E) under a fixed small-formal scenario.
Agent-C baseline is run once and shared across all configs.

Usage::

    # Quick mode (1 seed × 1 ep):
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.sweep_c_lookahead_score --quick

    # Full small-formal:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.sweep_c_lookahead_score
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint
from sa_hmarl.evaluation.eval_c_lookahead_oracle import evaluate_one


# ---------------------------------------------------------------------------
# Configs
# ---------------------------------------------------------------------------

CONFIGS = {
    "A_baseline": dict(
        lookahead_horizon=5,
        current_block_penalty=5.0, future_block_penalty=2.0,
        no_block_penalty=1.0, delay_coef=0.1,
        fs_coef=0.2, waste_coef_score=0.2, server_overload_coef=1.0,
    ),
    "B_block_heavy": dict(
        lookahead_horizon=5,
        current_block_penalty=10.0, future_block_penalty=8.0,
        no_block_penalty=5.0, delay_coef=0.05,
        fs_coef=0.2, waste_coef_score=0.1, server_overload_coef=1.0,
    ),
    "C_long_horizon": dict(
        lookahead_horizon=10,
        current_block_penalty=10.0, future_block_penalty=8.0,
        no_block_penalty=5.0, delay_coef=0.05,
        fs_coef=0.2, waste_coef_score=0.1, server_overload_coef=1.0,
    ),
    "D_spectrum_pressure": dict(
        lookahead_horizon=5,
        current_block_penalty=10.0, future_block_penalty=6.0,
        no_block_penalty=4.0, delay_coef=0.05,
        fs_coef=0.5, waste_coef_score=0.3, server_overload_coef=1.0,
    ),
    "E_aggressive_blocking": dict(
        lookahead_horizon=10,
        current_block_penalty=20.0, future_block_penalty=12.0,
        no_block_penalty=8.0, delay_coef=0.02,
        fs_coef=0.5, waste_coef_score=0.2, server_overload_coef=1.0,
    ),
}


# ---------------------------------------------------------------------------
# Helpers
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


def _aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "avg_fs", "avg_waste", "avg_path_len_km",
        "avg_lookahead_score_finite", "inf_score_rate", "no_valid_c_rate",
    ]
    agg: Dict[str, Any] = {}
    for key in scalar_keys:
        vals = [float(r.get(key, 0.0)) for r in results]
        agg[key] = float(np.mean(vals))
        agg[f"{key}_std"] = float(np.std(vals))
    for key in ["split_counter", "server_counter", "reason_counter", "mod_counter"]:
        c = Counter()
        for r in results:
            c.update(r.get(key, {}))
        agg[key] = dict(c)
    agg["total_oracle_time_s"] = float(sum(
        r.get("oracle_time_s", 0.0) for r in results
    ))
    return agg


def _fmt_pct(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    items = sorted(c.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v / total:.1%}" for k, v in items)


def _fmt_cnt(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    items = sorted(c.items(), key=lambda x: -x[1])
    return ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in items)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--episodes", type=int, default=2)
    parser.add_argument("--requests_per_episode", type=int, default=40)
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/lookahead_score_sweep.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/lookahead_score_sweep.md")
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    if args.quick:
        args.seeds = "42"
        args.episodes = 1
        args.requests_per_episode = 40
        print("[QUICK MODE] 1 seed × 1 episode × 40 req")

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Load models
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R: {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
    )

    # Pre-generate episodes
    all_episodes: Dict[int, List[List]] = {}
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

    print("=" * 90)
    print("Lookahead-C Oracle Score-Weight Ablation")
    print(f"Seeds: {seeds}  Eps/seed: {args.episodes}  Req/ep: {args.requests_per_episode}")
    print(f"Configs: {list(CONFIGS.keys())}")
    print("=" * 90)

    # ------------------------------------------------------------------
    # Agent-C baseline (once)
    # ------------------------------------------------------------------
    print("\n>>> Agent-C baseline (shared)")
    agent_results = [evaluate_one("agent", agent_c, agent_r, all_episodes[s], env_proto, args) for s in seeds]
    for seed, res in zip(seeds, agent_results):
        print(f"  seed={seed}: blk={res['blocking_rate']:.4f} delay={res['avg_delay_ms']:.1f}ms")
    agent_agg = _aggregate(agent_results)
    agent_blk = agent_agg["blocking_rate"]
    agent_delay = agent_agg["avg_delay_ms"]
    print(f"  => blk={agent_blk:.4f} delay={agent_delay:.1f}ms fs={agent_agg['avg_fs']:.2f}")

    # ------------------------------------------------------------------
    # Lookahead Oracle per config
    # ------------------------------------------------------------------
    all_results: Dict[str, Dict[str, Any]] = {"Agent-C_baseline": agent_agg}

    for cfg_name, cfg_params in CONFIGS.items():
        print(f"\n{'─' * 70}")
        print(f">>> {cfg_name}  {cfg_params}")
        print(f"{'─' * 70}")

        cfg_args = argparse.Namespace(**{**vars(args), **cfg_params})
        cfg_results = []
        for seed in seeds:
            t0 = time.time()
            res = evaluate_one("lookahead_oracle", agent_c, agent_r, all_episodes[seed], env_proto, cfg_args)
            elapsed = time.time() - t0
            cfg_results.append(res)
            print(
                f"  seed={seed}: blk={res['blocking_rate']:.4f} "
                f"delay={res['avg_delay_ms']:.1f}ms "
                f"score={res.get('avg_lookahead_score_finite', 0):.2f} "
                f"inf%={res.get('inf_score_rate', 0):.1%} "
                f"noC%={res.get('no_valid_c_rate', 0):.1%} "
                f"({elapsed:.0f}s)"
            )

        agg = _aggregate(cfg_results)
        blk = agg["blocking_rate"]; delay = agg["avg_delay_ms"]
        d_blk = agent_blk - blk; d_delay = delay - agent_delay
        agg["delta_blocking"] = d_blk; agg["delta_delay"] = d_delay
        all_results[cfg_name] = agg

        tag = "✓ ≥3pp" if d_blk >= 0.03 else f"{d_blk:+.4f}"
        print(f"  => blk={blk:.4f} Δblk={d_blk:+.4f} delay={delay:.1f}ms Δdelay={d_delay:+.1f}ms [{tag}]")

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------
    print("\n" + "=" * 90)
    print("SWEEP RESULTS")
    print("=" * 90)
    hdr = (f"{'Config':28s} {'AgentBlk':>8s} {'OracleBlk':>9s} {'ΔBlock':>8s} "
           f"{'Delay':>7s} {'ΔDelay':>7s} {'Score':>7s} {'Inf%':>6s} {'NoC%':>6s} {'Time':>6s}")
    print(hdr); print("-" * 90)
    for cfg_name in ["Agent-C_baseline"] + list(CONFIGS.keys()):
        r = all_results[cfg_name]
        if cfg_name == "Agent-C_baseline":
            print(f"{cfg_name:28s} {r['blocking_rate']:8.4f} {'—':>9s} {'—':>8s} {r['avg_delay_ms']:7.1f} {'—':>7s}")
        else:
            d_blk = r.get("delta_blocking", 0); d_delay = r.get("delta_delay", 0)
            sc = r.get("avg_lookahead_score_finite", 0); inf_r = r.get("inf_score_rate", 0)
            noc_r = r.get("no_valid_c_rate", 0); t = r.get("total_oracle_time_s", 0)
            print(f"{cfg_name:28s} {agent_blk:8.4f} {r['blocking_rate']:9.4f} {d_blk:+8.4f} "
                  f"{r['avg_delay_ms']:7.1f} {d_delay:+7.1f} {sc:7.2f} {inf_r:5.1%} {noc_r:5.1%} {t:5.0f}s")

    print("\n" + "-" * 90 + "\nFAILURE REASONS\n" + "-" * 90)
    for cfg_name in ["Agent-C_baseline"] + list(CONFIGS.keys()):
        r = all_results[cfg_name]
        print(f"  {cfg_name:28s}  {_fmt_cnt(r.get('reason_counter', {}))}")

    print("\n" + "-" * 90 + "\nSPLIT DISTRIBUTION\n" + "-" * 90)
    for cfg_name in ["Agent-C_baseline"] + list(CONFIGS.keys()):
        r = all_results[cfg_name]
        print(f"  {cfg_name:28s}  {_fmt_pct(r.get('split_counter', {}))}")

    # ------------------------------------------------------------------
    # JSON + MD
    # ------------------------------------------------------------------
    out_json = Path(args.out_json); out_json.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for name, agg in all_results.items():
        clean[name] = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in agg.items()}
    out_json.write_text(json.dumps({"args": vars(args), "configs": CONFIGS, "results": clean}, indent=2), encoding="utf-8")
    print(f"\nJSON → {out_json}")

    out_md = Path(args.out_md)
    md = [
        "# Lookahead-C Oracle Score-Weight Ablation\n\n",
        f"**Seeds:** {args.seeds}  **Eps/seed:** {args.episodes}  **Req/ep:** {args.requests_per_episode}\n\n",
        "## Configs\n\n",
    ]
    for n, p in CONFIGS.items():
        md.append(f"### {n}\n```\n{json.dumps(p, indent=2)}\n```\n\n")
    md.append("## Results\n\n| Config | AgentBlk | OracleBlk | ΔBlock | Delay | ΔDelay | Score | Inf% | NoC% | Time |\n")
    md.append("|--------|----------|-----------|--------|-------|--------|-------|------|------|------|\n")
    for cfg_name in ["Agent-C_baseline"] + list(CONFIGS.keys()):
        r = all_results[cfg_name]
        if cfg_name == "Agent-C_baseline":
            md.append(f"| {cfg_name} | {r['blocking_rate']:.4f} | — | — | {r['avg_delay_ms']:.1f} | — | — | — | — | — |\n")
        else:
            d_blk = r.get("delta_blocking", 0); d_delay = r.get("delta_delay", 0)
            sc = r.get("avg_lookahead_score_finite", 0); inf_r = r.get("inf_score_rate", 0)
            noc_r = r.get("no_valid_c_rate", 0); t = r.get("total_oracle_time_s", 0)
            md.append(f"| {cfg_name} | {agent_blk:.4f} | {r['blocking_rate']:.4f} | {d_blk:+.4f} | "
                      f"{r['avg_delay_ms']:.1f} | {d_delay:+.1f} | {sc:.2f} | {inf_r:.1%} | {noc_r:.1%} | {t:.0f}s |\n")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")

    # ------------------------------------------------------------------
    # Verdict
    # ------------------------------------------------------------------
    best_d_blk = max((all_results[n].get("delta_blocking", -999) for n in CONFIGS), default=-999)
    best_cfg = max(CONFIGS, key=lambda n: all_results[n].get("delta_blocking", -999))
    print(f"\n{'✓' if best_d_blk >= 0.03 else '✗'} BEST: {best_cfg} Δblocking={best_d_blk:+.4f} "
          f"({'≥3pp → expand' if best_d_blk >= 0.03 else '<3pp → stop'})")


if __name__ == "__main__":
    main()
