"""Minimal Lookahead-C Oracle evaluation.

Compares two C-action selection methods under the SAME fixed BC-PPO-R:

1. **Agent-C** — learned PPO policy (baseline)
2. **Lookahead-C Oracle** — enumerates all valid C actions, scores each with
   a forward-looking simulation over the next H requests, picks the best.

Usage
-----
Smoke::

    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_c_lookahead_oracle \\
        --seeds 42 --episodes 1 --requests_per_episode 5 \\
        --methods agent,lookahead_oracle --lookahead_horizon 2
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
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.c_lookahead_oracle import select_lookahead_c_action
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_agent_c_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


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
        device=device,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry,
                device: str = "cpu") -> PPOAgentR:
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
# Helpers
# ---------------------------------------------------------------------------

def _select_agent_c(
    agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int,
) -> Optional[int]:
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
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


def _select_agent_r(
    agent_r: PPOAgentR, env, req, split_id: int, server_id: int,
) -> Tuple[Tuple[int, int, int], Optional[int]]:
    from sa_hmarl.env.observation_builder import build_agent_r_observation
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    action_idx = agent_r.select_action(obs_r, deterministic=True)
    if action_idx is None:
        return (0, 0, 0), None
    return (
        decode_agent_r_action(action_idx, len(obs_r["mod_names"]), env.max_blocks),
        action_idx,
    )


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_one(
    method: str,
    agent_c: PPOAgentC,
    agent_r: PPOAgentR,
    episodes_data: List[List],
    env_proto,
    args,
) -> Dict[str, Any]:
    num_servers = len(env_proto.mec.servers)
    server_nodes = [s.node_id for s in env_proto.mec.servers]
    capacities = [s.compute_capacity for s in env_proto.mec.servers]

    total = blocked = success = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    split_counter = Counter()
    server_counter = Counter()
    reason_counter = Counter()
    mod_counter = Counter()
    total_oracle_time = 0.0
    all_lookahead_scores: List[float] = []
    no_valid_c_count = 0
    inf_score_count = 0

    for requests in episodes_data:
        env = make_env(
            topology=args.topology,
            num_slots=args.num_slots,
            num_servers=args.num_servers,
            seed=42,
            slot_bw_hz=args.slot_bw_hz,
            guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
            server_nodes=server_nodes,
            capacities=capacities,
            k=args.k_paths,
        )
        env.reset(requests)

        for i, req in enumerate(requests):
            obs_c = build_agent_c_observation(env, req)

            if method == "agent":
                action_idx_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
                action_c = (
                    (0, 0) if action_idx_c is None
                    else decode_agent_c_action(action_idx_c, num_servers)
                )
                action_r, _ = _select_agent_r(
                    agent_r, env, req, action_c[0], action_c[1],
                )

            elif method == "lookahead_oracle":
                future_reqs = requests[i + 1:]  # remaining requests in episode
                t0 = time.time()
                action_idx_c, diag = select_lookahead_c_action(
                    agent_c, agent_r, env, req, future_reqs, args,
                )
                total_oracle_time += time.time() - t0
                best_score = diag.get("best_score", float("inf"))
                all_lookahead_scores.append(best_score)
                if not np.isfinite(best_score):
                    inf_score_count += 1
                if diag.get("reason") == "no_valid_c_actions":
                    no_valid_c_count += 1

                action_c = (
                    (0, 0) if action_idx_c is None
                    else decode_agent_c_action(action_idx_c, num_servers)
                )
                action_r, _ = _select_agent_r(
                    agent_r, env, req, action_c[0], action_c[1],
                )

            else:
                raise ValueError(f"Unknown method: {method}")

            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1

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
                reason_counter[info.get("reason", "unknown")] += 1

    n = max(total, 1)
    s = max(success, 1)
    return {
        "total": total,
        "blocked": blocked,
        "success": success,
        "blocking_rate": blocked / n,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": total_delay / s,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "reason_counter": dict(reason_counter),
        "mod_counter": dict(mod_counter),
        "oracle_time_s": total_oracle_time,
        "avg_lookahead_score": (
            float(np.mean(all_lookahead_scores)) if all_lookahead_scores else 0.0
        ),
        "avg_lookahead_score_finite": (
            float(np.mean([s for s in all_lookahead_scores if np.isfinite(s)]))
            if any(np.isfinite(s) for s in all_lookahead_scores) else 0.0
        ),
        "inf_score_count": inf_score_count,
        "inf_score_rate": inf_score_count / max(total, 1),
        "no_valid_c_count": no_valid_c_count,
        "no_valid_c_rate": no_valid_c_count / max(total, 1),
    }


# ---------------------------------------------------------------------------
# Formatting
# ---------------------------------------------------------------------------

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
    parser = argparse.ArgumentParser(
        description="Minimal Lookahead-C Oracle vs Agent-C comparison"
    )
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--k_paths", type=int, default=3,
                        help="Number of shortest paths for R action space (k).")
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="42")
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--requests_per_episode", type=int, default=5)
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
    parser.add_argument("--methods", type=str, default="agent,lookahead_oracle")
    parser.add_argument("--lookahead_horizon", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--out_json", type=str,
                        default="sa_hmarl/experiments/c_lookahead_oracle_eval.json")
    parser.add_argument("--out_md", type=str,
                        default="sa_hmarl/experiments/c_lookahead_oracle_eval.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]

    METHOD_LABELS = {
        "agent": "Agent-C + BC-PPO-R",
        "lookahead_oracle": f"Lookahead-C Oracle (H={args.lookahead_horizon}) + BC-PPO-R",
    }

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
        k=args.k_paths,
    )

    print("=" * 80)
    print("Lookahead-C Oracle vs Agent-C")
    print(f"Topology: {args.topology}  Slots: {args.num_slots}  "
          f"Servers: {args.num_servers}")
    print(f"Methods: {methods}  Horizon: {args.lookahead_horizon}")
    print(f"Seeds: {seeds}  Episodes/seed: {args.episodes}  "
          f"Requests/ep: {args.requests_per_episode}")
    print("=" * 80)

    # Pre-generate episodes (same for both methods)
    all_episodes: Dict[int, List[List]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(
                generate_requests(
                    env_proto, rng, src,
                    num_requests=args.requests_per_episode,
                    arrival_interval=args.arrival_interval,
                    holding_min=args.holding_min,
                    holding_max=args.holding_max,
                    deadline_min=args.deadline_min,
                    deadline_max=args.deadline_max,
                    size_min_mb=args.size_min_mb,
                    size_max_mb=args.size_max_mb,
                    edge_cost_min=args.edge_cost_min,
                    edge_cost_max=args.edge_cost_max,
                    num_splits=args.num_splits,
                    split_profile=args.split_profile,
                )
            )
        all_episodes[seed] = eps

    # Evaluate
    results: Dict[str, Dict[str, Any]] = {}
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        print(f"\n{'─' * 60}")
        print(f"Evaluating: {label}")
        print(f"{'─' * 60}")

        seed_results = []
        for seed in seeds:
            res = evaluate_one(
                method, agent_c, agent_r, all_episodes[seed], env_proto, args,
            )
            seed_results.append(res)
            extra = ""
            if res.get("oracle_time_s", 0) > 0:
                extra = f" oracle_time={res['oracle_time_s']:.1f}s"
            print(
                f"  seed={seed:4d}: blk={res['blocking_rate']:.4f} "
                f"reward={res['avg_reward']:+.3f} "
                f"delay={res['avg_delay_ms']:.1f}ms "
                f"fs={res['avg_fs']:.2f}{extra}"
            )

        # Aggregate
        scalar_keys = [
            "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
            "avg_fs", "avg_waste", "avg_path_len_km",
        ]
        agg: Dict[str, Any] = {}
        for key in scalar_keys:
            vals = [float(r[key]) for r in seed_results]
            agg[key] = float(np.mean(vals))
            agg[f"{key}_std"] = float(np.std(vals))
        for key in ["split_counter", "server_counter", "reason_counter", "mod_counter"]:
            c = Counter()
            for r in seed_results:
                c.update(r.get(key, {}))
            agg[key] = dict(c)
        agg["oracle_time_s"] = float(sum(
            r.get("oracle_time_s", 0.0) for r in seed_results
        ))
        agg["avg_lookahead_score"] = float(np.mean([
            r.get("avg_lookahead_score", 0.0) for r in seed_results
        ]))
        results[label] = agg

        print(
            f"  => AVG: blk={agg['blocking_rate']:.4f} "
            f"delay={agg['avg_delay_ms']:.1f}ms fs={agg['avg_fs']:.2f}"
        )

    # ------------------------------------------------------------------
    # Console report
    # ------------------------------------------------------------------
    print()
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)

    header = (
        f"{'Method':45s} {'Blocking':>9s} {'Reward':>8s} {'Delay':>8s} "
        f"{'AvgFS':>7s} {'Waste':>7s}"
    )
    print(header)
    print("-" * 80)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        r = results[label]
        print(
            f"{label:45s} {r['blocking_rate']:9.4f} {r['avg_reward']:8.3f} "
            f"{r['avg_delay_ms']:8.1f} {r['avg_fs']:7.2f} {r['avg_waste']:7.3f}"
        )

    print()
    print("-" * 80)
    print("FAILURE REASONS")
    print("-" * 80)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        r = results[label]
        print(f"  {label:45s}  {_fmt_cnt(r.get('reason_counter', {}))}")

    print()
    print("-" * 80)
    print("SPLIT DISTRIBUTION")
    print("-" * 80)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        r = results[label]
        print(f"  {label:45s}  {_fmt_pct(r.get('split_counter', {}))}")

    print()
    print("-" * 80)
    print("SERVER DISTRIBUTION")
    print("-" * 80)
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        r = results[label]
        print(f"  {label:45s}  {_fmt_pct(r.get('server_counter', {}))}")

    # ------------------------------------------------------------------
    # JSON
    # ------------------------------------------------------------------
    out_json = Path(args.out_json)
    out_json.parent.mkdir(parents=True, exist_ok=True)
    clean = {}
    for label, agg in results.items():
        clean[label] = {
            k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
            for k, v in agg.items()
        }
    out_json.write_text(
        json.dumps({"args": vars(args), "results": clean}, indent=2),
        encoding="utf-8",
    )
    print(f"\nJSON → {out_json}")

    # ------------------------------------------------------------------
    # Markdown
    # ------------------------------------------------------------------
    out_md = Path(args.out_md)
    md = [
        "# Lookahead-C Oracle vs Agent-C\n\n",
        f"**Topology:** {args.topology}  "
        f"**Slots:** {args.num_slots}  **Servers:** {args.num_servers}  "
        f"**MaxBlocks:** {args.max_blocks}\n\n",
        f"**Agent-C:** `{args.agent_c_checkpoint}`  \n",
        f"**Agent-R:** `{args.agent_r_checkpoint}`  \n",
        f"**Lookahead horizon:** {args.lookahead_horizon}  \n",
        f"**Seeds:** {args.seeds}  "
        f"**Episodes/seed:** {args.episodes}  "
        f"**Requests/ep:** {args.requests_per_episode}\n\n",
        "## Results\n\n",
        "| Method | Blocking | Reward | Delay(ms) | AvgFS | Waste |\n",
        "|--------|----------|--------|-----------|-------|-------|\n",
    ]
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        r = results[label]
        md.append(
            f"| {label} | {r['blocking_rate']:.4f} | {r['avg_reward']:+.3f} | "
            f"{r['avg_delay_ms']:.1f} | {r['avg_fs']:.2f} | {r['avg_waste']:.3f} |\n"
        )
    md.append("\n## Failure Reasons\n\n")
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        r = results[label]
        md.append(f"- **{label}**: {_fmt_cnt(r.get('reason_counter', {}))}\n")
    md.append("\n## Split Distribution\n\n")
    for method in methods:
        label = METHOD_LABELS.get(method, method)
        r = results[label]
        md.append(f"- **{label}**: {_fmt_pct(r.get('split_counter', {}))}\n")

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("".join(md), encoding="utf-8")
    print(f"Markdown → {out_md}")


if __name__ == "__main__":
    main()
