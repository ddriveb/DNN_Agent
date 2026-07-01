"""Exhaustive Oracle R Analysis — per-request R action enumeration.

Fixes Agent-C action, deep-copies env per request, enumerates ALL valid R
actions, finds the oracle (best immediate outcome), and quantifies R-side
improvement headroom.

Key metrics:
  - ppo_fail_oracle_success_rate: PPO-R fails but some other R action succeeds
  - oracle_blocking_upper_bound: min blocking if oracle R per request
  - oracle_gain_pp: ppo_blocking - oracle_blocking (pp)

Stop criteria:
  oracle_gain < 0.5pp  → R-side done
  0.5pp ≤ oracle_gain < 2pp → small R improvements only
  oracle_gain ≥ 2pp → worth systematic R rework

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_oracle_r_exhaustive \
        --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_act_tanh_s42_s20_r80_best.pt \
        --agent_r_checkpoint sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt \
        --topology snap24_gnutella_reach --num_slots 20 \
        --requests_per_episode 80 --k_paths 5 --max_blocks 10 \
        --block_sort_strategy mixed --seeds 42,123,456,789,2024 --episodes 20
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import copy
import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    activation = (
        ckpt_args.get("agent_c_activation", "tanh")
        if isinstance(ckpt_args, dict) else "tanh"
    )
    feature_mode = (
        ckpt_args.get("agent_c_feature_mode", "default")
        if isinstance(ckpt_args, dict) else "default"
    )
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        activation=activation,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    feature_mode = ckpt.get("agent_r_feature_mode", "default")
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Oracle selection
# ---------------------------------------------------------------------------

def _select_oracle(results: List[Dict]) -> Optional[Dict]:
    """Select best R action: success > fewer FS > lower delay > lower waste > shorter path."""
    if not results:
        return None
    successful = [r for r in results if r.get("success", False)]
    candidates = successful if successful else results

    def _key(r):
        success = 0 if r.get("success", False) else 1
        fs = r.get("num_slots", 999)
        delay = r.get("delay_ms", 1e9)
        waste = r.get("block_waste", 1.0)
        path = r.get("path_dist_km", 1e9)
        return (success, fs, delay, waste, path)

    return min(candidates, key=_key)


# ---------------------------------------------------------------------------
# Per-request oracle trial
# ---------------------------------------------------------------------------

def _try_all_r_actions(env, action_c, obs_r, r_mask, max_blocks):
    """Try all valid R actions on independent env deep-copies."""
    t0 = time.perf_counter()
    num_mods = len(obs_r["mod_names"])
    valid_indices = np.where(r_mask)[0]

    results = []
    for r_idx in valid_indices:
        env_try = copy.deepcopy(env)
        action_r = decode_agent_r_action(int(r_idx), num_mods, max_blocks)
        _, _, _, info = env_try.step(action_c, action_r)
        results.append({
            "action_idx": int(r_idx),
            "success": info.get("success", False),
            "reason": info.get("reason", ""),
            "num_slots": info.get("num_slots", 0),
            "delay_ms": info.get("delay_ms", 0.0),
            "block_waste": info.get("block_waste", 0.0),
            "path_dist_km": info.get("path_dist_km", 0.0),
            "modulation": info.get("modulation", ""),
        })
    elapsed = time.perf_counter() - t0
    return results, elapsed


# ---------------------------------------------------------------------------
# Episode evaluation
# ---------------------------------------------------------------------------

def _evaluate_episode(env_factory, requests, agent_c, agent_r, max_blocks):
    env = env_factory()
    env.reset(requests)
    num_servers = len(env.mec.servers)

    total = 0
    ppo_blocked = 0
    ppo_success = 0
    ppo_total_fs = 0.0
    ppo_total_delay = 0.0

    oracle_blocked = 0
    oracle_total_fs = 0.0
    oracle_total_delay = 0.0

    ppo_fail_oracle_success = 0
    ppo_success_oracle_better = 0
    ppo_success_oracle_same = 0

    total_copy_time = 0.0
    valid_r_counts = []
    reasons_ppo = Counter()
    reasons_oracle = Counter()

    for req in requests:
        # --- C action ---
        obs_c = build_agent_c_observation(env, req)
        c_features, c_mask = agent_c.build_action_features(obs_c)
        action_idx_c, _, _ = agent_c.select_from_features(c_features, c_mask, deterministic=True)
        if action_idx_c is None:
            action_c = (0, 0)
        else:
            action_c = decode_agent_c_action(action_idx_c, num_servers)
        split_id, server_id = action_c

        # --- R observation ---
        obs_r = build_agent_r_observation(env, req, split_id, server_id)
        r_mask = obs_r["agent_r_mask"]
        valid_r_count = int(np.sum(r_mask))
        valid_r_counts.append(valid_r_count)
        num_mods = len(obs_r["mod_names"])

        if valid_r_count == 0:
            env.step(action_c, (0, 0, 0))
            total += 1
            ppo_blocked += 1
            oracle_blocked += 1
            reasons_ppo["no_valid_r"] += 1
            reasons_oracle["no_valid_r"] += 1
            continue

        # --- PPO-R action ---
        action_idx_r_ppo = agent_r.select_action(obs_r, deterministic=True)
        if action_idx_r_ppo is None:
            action_idx_r_ppo = 0

        # --- Oracle: try all valid R actions on copies ---
        trial_results, copy_time = _try_all_r_actions(env, action_c, obs_r, r_mask, max_blocks)
        total_copy_time += copy_time

        oracle_action = _select_oracle(trial_results)

        # --- Execute PPO-R on real env ---
        action_r_ppo = decode_agent_r_action(action_idx_r_ppo, num_mods, max_blocks)
        _, _, _, info_ppo = env.step(action_c, action_r_ppo)
        total += 1

        # --- Record PPO ---
        ppo_ok = info_ppo.get("success", False)
        if ppo_ok:
            ppo_success += 1
            ppo_total_fs += info_ppo.get("num_slots", 0)
            ppo_total_delay += info_ppo.get("delay_ms", 0.0)
        else:
            ppo_blocked += 1
            reasons_ppo[info_ppo.get("reason", "unknown")] += 1

        # --- Record oracle ---
        oracle_ok = oracle_action.get("success", False) if oracle_action else False
        if oracle_ok:
            oracle_total_fs += oracle_action.get("num_slots", 0)
            oracle_total_delay += oracle_action.get("delay_ms", 0.0)
        else:
            oracle_blocked += 1
            reasons_oracle[oracle_action.get("reason", "unknown") if oracle_action else "no_oracle"] += 1

        # --- Comparison ---
        if not ppo_ok and oracle_ok:
            ppo_fail_oracle_success += 1
        elif ppo_ok and oracle_ok:
            ppo_fs = info_ppo.get("num_slots", 0)
            oracle_fs = oracle_action.get("num_slots", 0)
            ppo_delay = info_ppo.get("delay_ms", 0.0)
            oracle_delay = oracle_action.get("delay_ms", 0.0)

            if oracle_fs < ppo_fs or (oracle_fs == ppo_fs and oracle_delay < ppo_delay - 0.5):
                ppo_success_oracle_better += 1
            elif action_idx_r_ppo == oracle_action["action_idx"]:
                ppo_success_oracle_same += 1

    n = max(total, 1)
    s_ppo = max(ppo_success, 1)
    s_ora = max(total - oracle_blocked, 1)

    return {
        "total": total,
        "ppo_blocking": ppo_blocked / n,
        "ppo_avg_delay_ms": ppo_total_delay / s_ppo,
        "ppo_avg_fs": ppo_total_fs / s_ppo,
        "oracle_blocking": oracle_blocked / n,
        "oracle_avg_delay_ms": oracle_total_delay / s_ora,
        "oracle_avg_fs": oracle_total_fs / s_ora,
        "oracle_gain_pp": (ppo_blocked - oracle_blocked) / n,
        "ppo_fail_oracle_success_rate": ppo_fail_oracle_success / n,
        "ppo_success_oracle_better_rate": ppo_success_oracle_better / n,
        "ppo_success_oracle_same_rate": ppo_success_oracle_same / n,
        "mean_valid_r_actions": float(np.mean(valid_r_counts)) if valid_r_counts else 0.0,
        "total_copy_time_s": total_copy_time,
        "reasons_ppo": dict(reasons_ppo),
        "reasons_oracle": dict(reasons_oracle),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Exhaustive Oracle R Analysis")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_act_tanh_s42_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--output_json", type=str,
                        default="sa_hmarl/experiments/oracle_r_exhaustive.json")
    parser.add_argument("--output_md", type=str,
                        default="sa_hmarl/experiments/oracle_r_exhaustive.md")
    args = parser.parse_args()

    eval_seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    print(f"Loading Agent-C from {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading Agent-R from {args.agent_r_checkpoint}")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    def make_env_factory():
        return make_env(
            topology=args.topology, num_slots=args.num_slots,
            num_servers=args.num_servers, seed=42,
            slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
            modulation_profile=args.modulation_profile,
            k=args.k_paths, max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy,
        )

    env_proto = make_env_factory()

    print("=" * 90)
    print("EXHAUSTIVE ORACLE R ANALYSIS")
    print(f"  Agent-C: {args.agent_c_checkpoint}")
    print(f"  Agent-R: {args.agent_r_checkpoint}")
    print(f"  Topology: {args.topology}  Slots: {args.num_slots}  "
          f"k={args.k_paths}  max_blocks={args.max_blocks}  sort={args.block_sort_strategy}")
    print(f"  Eval seeds: {eval_seeds}  Episodes/seed: {args.episodes}  "
          f"Requests/episode: {args.requests_per_episode}")
    print("=" * 90)

    all_ep_results = []
    for seed in eval_seeds:
        print(f"\n--- Seed {seed} ---")
        rng = np.random.RandomState(seed)
        for ep in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            requests = generate_requests(
                env_proto, rng, src, args.requests_per_episode,
                arrival_interval=0.25, holding_min=4.0, holding_max=10.0,
                deadline_min=30.0, deadline_max=100.0,
                size_min_mb=5.0, size_max_mb=30.0,
                edge_cost_min=0.5, edge_cost_max=15.0,
                num_splits=args.num_splits,
            )
            ep_res = _evaluate_episode(make_env_factory, requests, agent_c, agent_r, args.max_blocks)
            all_ep_results.append(ep_res)
            if (ep + 1) % 5 == 0:
                print(f"  ep {ep+1:3d}: ppo_blk={ep_res['ppo_blocking']:.4f} "
                      f"oracle_blk={ep_res['oracle_blocking']:.4f} "
                      f"gain={ep_res['oracle_gain_pp']:.4f} "
                      f"fail->succ={ep_res['ppo_fail_oracle_success_rate']:.4f} "
                      f"copy={ep_res['total_copy_time_s']:.1f}s")

    # Aggregate
    scalar_keys = [
        "ppo_blocking", "oracle_blocking", "oracle_gain_pp",
        "ppo_avg_delay_ms", "oracle_avg_delay_ms",
        "ppo_avg_fs", "oracle_avg_fs",
        "ppo_fail_oracle_success_rate", "ppo_success_oracle_better_rate",
        "ppo_success_oracle_same_rate", "mean_valid_r_actions",
    ]
    agg = {}
    for k in scalar_keys:
        vals = [ep[k] for ep in all_ep_results]
        agg[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}

    reasons_ppo = Counter()
    reasons_oracle = Counter()
    for ep in all_ep_results:
        reasons_ppo.update(ep.get("reasons_ppo", {}))
        reasons_oracle.update(ep.get("reasons_oracle", {}))
    agg["reasons_ppo"] = dict(reasons_ppo)
    agg["reasons_oracle"] = dict(reasons_oracle)
    agg["total_copy_time_s"] = sum(ep["total_copy_time_s"] for ep in all_ep_results)
    agg["num_episodes"] = len(all_ep_results)
    agg["total_requests"] = sum(ep["total"] for ep in all_ep_results)

    # Print
    print("\n" + "=" * 90)
    print("ORACLE R ANALYSIS — AGGREGATED RESULTS")
    print("=" * 90)

    labels = {
        "ppo_blocking": "PPO-R Blocking Rate",
        "oracle_blocking": "Oracle R Blocking (upper bound)",
        "oracle_gain_pp": "Oracle Gain (pp)",
        "ppo_avg_delay_ms": "PPO-R Avg Delay (ms)",
        "oracle_avg_delay_ms": "Oracle Avg Delay (ms)",
        "ppo_avg_fs": "PPO-R Avg FS",
        "oracle_avg_fs": "Oracle Avg FS",
        "ppo_fail_oracle_success_rate": "PPO fail -> Oracle success rate",
        "ppo_success_oracle_better_rate": "PPO success -> Oracle better rate",
        "ppo_success_oracle_same_rate": "PPO action IS oracle rate",
        "mean_valid_r_actions": "Mean valid R actions/request",
    }
    for k in scalar_keys:
        print(f"  {labels.get(k, k):40s} {agg[k]['mean']:8.4f} ± {agg[k]['std']:.4f}")
    print(f"  {'Total copy time (s)':40s} {agg['total_copy_time_s']:8.1f}")
    print(f"  {'Total requests':40s} {agg['total_requests']:8d}")

    # Verdict
    oracle_gain = agg["oracle_gain_pp"]["mean"]
    print("\n" + "-" * 65)
    print("STOP-CRITERIA VERDICT")
    print("-" * 65)
    print(f"Oracle gain: {oracle_gain:.4f} pp  (= {oracle_gain*100:.2f}%)")
    if oracle_gain < 0.005:
        verdict = "R-SIDE DONE — oracle gain < 0.5pp. R is near-optimal. Stop R work."
    elif oracle_gain < 0.02:
        verdict = "SMALL HEADROOM — 0.5-2pp oracle gain. Minor R improvements only."
    else:
        verdict = "WORTH REWORK — oracle gain >= 2pp. Systematic R-side improvement justified."
    print(f"Verdict: {verdict}")
    print(f"\nPPO-R failures: {dict(reasons_ppo)}")
    print(f"Oracle failures: {dict(reasons_oracle)}")

    # JSON
    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(agg, open(str(out_path), "w"), indent=2)
        print(f"\nJSON -> {out_path}")

    # Markdown
    if args.output_md:
        md_path = Path(args.output_md)
        md_path.parent.mkdir(parents=True, exist_ok=True)
        lines = []
        def w(s=""): lines.append(s + "\n")
        w("# Exhaustive Oracle R Analysis")
        w()
        w("## Configuration")
        w()
        w("| Parameter | Value |")
        w("|-----------|-------|")
        w(f"| Agent-C | `{args.agent_c_checkpoint}` |")
        w(f"| Agent-R | `{args.agent_r_checkpoint}` |")
        w(f"| Topology | `{args.topology}` |")
        w(f"| Slots | `{args.num_slots}` |")
        w(f"| k_paths | `{args.k_paths}` |")
        w(f"| max_blocks | `{args.max_blocks}` |")
        w(f"| block_sort | `{args.block_sort_strategy}` |")
        w(f"| Eval seeds | `{args.seeds}` |")
        w(f"| Episodes/seed | `{args.episodes}` |")
        w(f"| Requests/episode | `{args.requests_per_episode}` |")
        w(f"| Total requests | `{agg['total_requests']}` |")
        w()
        w("## Results")
        w()
        w("| Metric | Value |")
        w("|--------|-------|")
        w(f"| PPO-R Blocking | {agg['ppo_blocking']['mean']:.4f} ± {agg['ppo_blocking']['std']:.4f} |")
        w(f"| Oracle Blocking (upper bound) | {agg['oracle_blocking']['mean']:.4f} ± {agg['oracle_blocking']['std']:.4f} |")
        w(f"| **Oracle Gain** | **{oracle_gain:.4f} pp** |")
        w(f"| PPO fail → Oracle success | {agg['ppo_fail_oracle_success_rate']['mean']:.4f} |")
        w(f"| PPO success → Oracle better | {agg['ppo_success_oracle_better_rate']['mean']:.4f} |")
        w(f"| PPO action IS oracle | {agg['ppo_success_oracle_same_rate']['mean']:.4f} |")
        w(f"| PPO-R Avg Delay | {agg['ppo_avg_delay_ms']['mean']:.1f} ms |")
        w(f"| Oracle Avg Delay | {agg['oracle_avg_delay_ms']['mean']:.1f} ms |")
        w(f"| PPO-R Avg FS | {agg['ppo_avg_fs']['mean']:.2f} |")
        w(f"| Oracle Avg FS | {agg['oracle_avg_fs']['mean']:.2f} |")
        w(f"| Mean valid R actions | {agg['mean_valid_r_actions']['mean']:.1f} |")
        w()
        w("## Verdict")
        w()
        w(f"**{verdict}**")
        w()
        w("### Failure Reasons")
        w()
        w(f"- PPO-R: {dict(reasons_ppo)}")
        w(f"- Oracle: {dict(reasons_oracle)}")
        w()
        w("### Methodology")
        w()
        w(
            "For each request: (1) Agent-C selects (split, server) once. "
            "(2) Environment is deep-copied before the step. "
            "(3) ALL valid R actions (from `agent_r_mask`) are enumerated and "
            "individually tried on independent env copies. "
            "(4) Oracle = success > fewer FS > lower delay > lower block_waste > "
            "shorter path. "
            "(5) PPO-R action is executed on the real env to continue the episode."
        )

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("".join(lines))
        print(f"Markdown -> {md_path}")


if __name__ == "__main__":
    main()
