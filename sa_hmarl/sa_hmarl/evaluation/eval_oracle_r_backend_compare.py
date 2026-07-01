"""Oracle-R gap comparison for multiple R backends.

This diagnostic fixes Agent-C decisions and asks, for each R backend state:

    If the backend fails, did any valid R action succeed?

The answer distinguishes "R policy made a bad action choice" from
"Agent-C selected a split/server pair with no feasible R action".

Backends supported:
  - independent PPO-R
  - DeepRMSA
  - KSP-BF

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_oracle_r_backend_compare
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from sa_hmarl.agents.deep_rmsa_agent import DeepRMSAAgent
from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.baselines.rmsa_baselines import ksp_bf_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {})
    activation = ckpt_args.get("agent_c_activation", "tanh") if isinstance(ckpt_args, dict) else "tanh"
    feature_mode = ckpt_args.get("agent_c_feature_mode", "default") if isinstance(ckpt_args, dict) else "default"
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


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    feature_mode = ckpt.get("agent_r_feature_mode", ckpt.get("args", {}).get("agent_r_feature_mode", "default"))
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


def _load_deep_rmsa(
    ckpt_path: str,
    env,
    mod_reg: ModulationRegistry,
    device: str = "cpu",
) -> DeepRMSAAgent:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_nodes = ckpt.get("num_nodes")
    if ckpt_nodes is not None and ckpt_nodes != env.net.NUM_NODES:
        raise ValueError(
            f"DeepRMSA topology mismatch: checkpoint num_nodes={ckpt_nodes}, "
            f"env NUM_NODES={env.net.NUM_NODES}"
        )
    ckpt_slots = ckpt.get("num_slots")
    if ckpt_slots is not None and ckpt_slots != env.net.num_slots:
        raise ValueError(
            f"DeepRMSA slot mismatch: checkpoint num_slots={ckpt_slots}, "
            f"env num_slots={env.net.num_slots}"
        )
    agent = DeepRMSAAgent(
        num_nodes=env.net.NUM_NODES,
        num_slots=env.net.num_slots,
        k_path=ckpt.get("k_path", env.k),
        m_blocks=ckpt.get("m_blocks", env.max_blocks),
        mod_registry=mod_reg,
        gamma=ckpt.get("gamma", 0.95),
        device=device,
    )
    agent.load_state_dict(ckpt)
    agent.eval()
    return agent


def _select_oracle(results: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not results:
        return None
    successful = [r for r in results if r.get("success", False)]
    candidates = successful if successful else results

    def key(r: Dict[str, Any]):
        success_rank = 0 if r.get("success", False) else 1
        return (
            success_rank,
            r.get("num_slots", 999),
            r.get("delay_ms", 1e9),
            r.get("block_waste", 1.0),
            r.get("path_dist_km", 1e9),
        )

    return min(candidates, key=key)


def _try_all_r_actions(env, action_c, obs_r, r_mask, max_blocks):
    num_mods = len(obs_r["mod_names"])
    results = []
    t0 = time.perf_counter()
    for r_idx in np.where(r_mask)[0]:
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
    return results, time.perf_counter() - t0


def _select_backend_action(backend: Dict[str, Any], obs_r) -> Optional[int]:
    if backend["type"] == "ppo":
        return backend["agent"].select_action(obs_r, deterministic=True)
    if backend["type"] == "deep_rmsa":
        return backend["agent"].select_action(obs_r)
    if backend["type"] == "ksp_bf":
        return ksp_bf_action(obs_r)
    raise ValueError(f"Unknown backend type: {backend['type']}")


def _new_env_from_proto(env_proto, args):
    return make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        k=args.k_paths,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
        server_nodes=[s.node_id for s in env_proto.mec.servers],
        capacities=[s.compute_capacity for s in env_proto.mec.servers],
    )


def evaluate_episode(env_proto, args, requests, agent_c, backends):
    envs = []
    for _ in backends:
        env = _new_env_from_proto(env_proto, args)
        env.reset(requests)
        envs.append(env)

    ref_env = envs[0]
    num_servers = len(env_proto.mec.servers)
    per_backend = {
        b["name"]: {
            "total": 0,
            "backend_blocked": 0,
            "backend_success": 0,
            "oracle_blocked": 0,
            "fail_oracle_success": 0,
            "success_oracle_better": 0,
            "success_oracle_same": 0,
            "backend_delay": 0.0,
            "backend_fs": 0.0,
            "oracle_delay": 0.0,
            "oracle_fs": 0.0,
            "valid_r_counts": [],
            "backend_reasons": Counter(),
            "oracle_reasons": Counter(),
            "copy_time": 0.0,
        }
        for b in backends
    }

    for req in requests:
        obs_c = build_agent_c_observation(ref_env, req)
        c_features, c_mask = agent_c.build_action_features(obs_c)
        action_idx_c, _, _ = agent_c.select_from_features(c_features, c_mask, deterministic=True)
        action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(action_idx_c, num_servers)
        split_id, server_id = action_c

        for env, backend in zip(envs, backends):
            name = backend["name"]
            m = per_backend[name]
            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            r_mask = obs_r["agent_r_mask"]
            valid_r_count = int(np.sum(r_mask))
            m["valid_r_counts"].append(valid_r_count)
            m["total"] += 1

            if valid_r_count == 0:
                env.step(action_c, (0, 0, 0))
                m["backend_blocked"] += 1
                m["oracle_blocked"] += 1
                m["backend_reasons"]["no_valid_r"] += 1
                m["oracle_reasons"]["no_valid_r"] += 1
                continue

            trial_results, copy_time = _try_all_r_actions(env, action_c, obs_r, r_mask, args.max_blocks)
            oracle_action = _select_oracle(trial_results)
            m["copy_time"] += copy_time

            action_idx_r = _select_backend_action(backend, obs_r)
            if action_idx_r is None:
                action_idx_r = 0
            action_r = decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), args.max_blocks)
            _, _, _, info_backend = env.step(action_c, action_r)

            backend_ok = info_backend.get("success", False)
            oracle_ok = oracle_action.get("success", False) if oracle_action else False

            if backend_ok:
                m["backend_success"] += 1
                m["backend_delay"] += info_backend.get("delay_ms", 0.0)
                m["backend_fs"] += info_backend.get("num_slots", 0)
            else:
                m["backend_blocked"] += 1
                m["backend_reasons"][info_backend.get("reason", "unknown")] += 1

            if oracle_ok:
                m["oracle_delay"] += oracle_action.get("delay_ms", 0.0)
                m["oracle_fs"] += oracle_action.get("num_slots", 0)
            else:
                m["oracle_blocked"] += 1
                m["oracle_reasons"][oracle_action.get("reason", "unknown") if oracle_action else "no_oracle"] += 1

            if not backend_ok and oracle_ok:
                m["fail_oracle_success"] += 1
            elif backend_ok and oracle_ok:
                backend_fs = info_backend.get("num_slots", 0)
                oracle_fs = oracle_action.get("num_slots", 0)
                backend_delay = info_backend.get("delay_ms", 0.0)
                oracle_delay = oracle_action.get("delay_ms", 0.0)
                if oracle_fs < backend_fs or (
                    oracle_fs == backend_fs and oracle_delay < backend_delay - 0.5
                ):
                    m["success_oracle_better"] += 1
                elif action_idx_r == oracle_action["action_idx"]:
                    m["success_oracle_same"] += 1

    out = {}
    for name, m in per_backend.items():
        n = max(m["total"], 1)
        s_backend = max(m["backend_success"], 1)
        s_oracle = max(m["total"] - m["oracle_blocked"], 1)
        out[name] = {
            "total": m["total"],
            "backend_blocking": m["backend_blocked"] / n,
            "oracle_blocking": m["oracle_blocked"] / n,
            "oracle_gain_pp": (m["backend_blocked"] - m["oracle_blocked"]) / n,
            "fail_oracle_success_rate": m["fail_oracle_success"] / n,
            "success_oracle_better_rate": m["success_oracle_better"] / n,
            "success_oracle_same_rate": m["success_oracle_same"] / n,
            "backend_avg_delay_ms": m["backend_delay"] / s_backend,
            "oracle_avg_delay_ms": m["oracle_delay"] / s_oracle,
            "backend_avg_fs": m["backend_fs"] / s_backend,
            "oracle_avg_fs": m["oracle_fs"] / s_oracle,
            "mean_valid_r_actions": float(np.mean(m["valid_r_counts"])) if m["valid_r_counts"] else 0.0,
            "copy_time_s": m["copy_time"],
            "backend_reasons": dict(m["backend_reasons"]),
            "oracle_reasons": dict(m["oracle_reasons"]),
        }
    return out


def _aggregate(ep_results: List[Dict[str, Dict[str, Any]]], backend_names: List[str]):
    scalar_keys = [
        "backend_blocking",
        "oracle_blocking",
        "oracle_gain_pp",
        "fail_oracle_success_rate",
        "success_oracle_better_rate",
        "success_oracle_same_rate",
        "backend_avg_delay_ms",
        "oracle_avg_delay_ms",
        "backend_avg_fs",
        "oracle_avg_fs",
        "mean_valid_r_actions",
    ]
    agg = {}
    for name in backend_names:
        out = {}
        for k in scalar_keys:
            vals = [ep[name][k] for ep in ep_results]
            out[k] = {"mean": float(np.mean(vals)), "std": float(np.std(vals))}
        backend_reasons = Counter()
        oracle_reasons = Counter()
        for ep in ep_results:
            backend_reasons.update(ep[name].get("backend_reasons", {}))
            oracle_reasons.update(ep[name].get("oracle_reasons", {}))
        out["backend_reasons"] = dict(backend_reasons)
        out["oracle_reasons"] = dict(oracle_reasons)
        out["total_requests"] = int(sum(ep[name]["total"] for ep in ep_results))
        out["total_copy_time_s"] = float(sum(ep[name]["copy_time_s"] for ep in ep_results))
        agg[name] = out
    return agg


def _write_markdown(path: Path, args, agg: Dict[str, Any], backend_names: List[str]):
    lines = []

    def w(s=""):
        lines.append(s + "\n")

    w("# Oracle-R Backend Gap Comparison")
    w()
    w("> Fixed Agent-C action; for each backend state, enumerate all valid R actions.")
    w()
    w("## Configuration")
    w()
    w("| Parameter | Value |")
    w("|---|---|")
    w(f"| Agent-C | `{args.agent_c_checkpoint}` |")
    w(f"| PPO-R | `{args.ppo_r_checkpoint}` |")
    w(f"| DeepRMSA | `{args.deep_rmsa_checkpoint}` |")
    w(f"| Topology | `{args.topology}` |")
    w(f"| Slots | `{args.num_slots}` |")
    w(f"| Requests/episode | `{args.requests_per_episode}` |")
    w(f"| Seeds | `{args.seeds}` |")
    w(f"| Episodes/seed | `{args.episodes}` |")
    w(f"| k_paths | `{args.k_paths}` |")
    w(f"| max_blocks | `{args.max_blocks}` |")
    w(f"| block_sort | `{args.block_sort_strategy}` |")
    w()
    w("## Main Results")
    w()
    w("| Backend | Backend Blocking | Oracle Blocking | Oracle Gain | Fail -> Oracle Success | Success -> Oracle Better | AvgFS Backend/Oracle | Delay Backend/Oracle | Mean Valid R Actions |")
    w("|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for name in backend_names:
        r = agg[name]
        w(
            f"| {name} | {r['backend_blocking']['mean']:.4f} | "
            f"{r['oracle_blocking']['mean']:.4f} | "
            f"{r['oracle_gain_pp']['mean']:.4f} | "
            f"{r['fail_oracle_success_rate']['mean']:.4f} | "
            f"{r['success_oracle_better_rate']['mean']:.4f} | "
            f"{r['backend_avg_fs']['mean']:.2f}/{r['oracle_avg_fs']['mean']:.2f} | "
            f"{r['backend_avg_delay_ms']['mean']:.1f}/{r['oracle_avg_delay_ms']['mean']:.1f} | "
            f"{r['mean_valid_r_actions']['mean']:.1f} |"
        )
    w()
    w("## Failure Reasons")
    w()
    for name in backend_names:
        w(f"- **{name} backend failures:** {agg[name]['backend_reasons']}")
        w(f"- **{name} oracle failures:** {agg[name]['oracle_reasons']}")
    w()
    w("## Interpretation")
    w()
    w("- `Oracle Gain = 0` means this backend has no blocking headroom under the fixed C decision and current R action space.")
    w("- `Fail -> Oracle Success > 0` means the backend failed even though another valid R action could have succeeded.")
    w("- If PPO-R and DeepRMSA both have zero gain, residual blocking is not an R-executor issue.")
    w("- KSP-BF is useful as a sanity check: if it has positive gain, Oracle-R can detect weak R policies.")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))


def main():
    parser = argparse.ArgumentParser(description="Oracle-R gap comparison across R backends")
    parser.add_argument("--agent_c_checkpoint", type=str, default="sa_hmarl/checkpoints/agent_c_act_tanh_s42_s20_r80_best.pt")
    parser.add_argument("--ppo_r_checkpoint", type=str, default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--deep_rmsa_checkpoint", type=str, default="sa_hmarl/checkpoints/deep_rmsa_snap24_reach_mixed.pt")
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
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--output_json", type=str, default="sa_hmarl/experiments/oracle_r_backend_compare.json")
    parser.add_argument("--output_md", type=str, default="sa_hmarl/experiments/oracle_r_backend_compare.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)

    env_proto = make_env(
        topology=args.topology,
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=42,
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        k=args.k_paths,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
    )

    print(f"Loading Agent-C: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    print(f"Loading independent PPO-R: {args.ppo_r_checkpoint}")
    ppo_r = _load_ppo_r(args.ppo_r_checkpoint, mod_reg, args.device)
    print(f"Loading DeepRMSA: {args.deep_rmsa_checkpoint}")
    deep_rmsa = _load_deep_rmsa(args.deep_rmsa_checkpoint, env_proto, mod_reg, args.device)

    backends = [
        {"name": "Independent PPO-R", "type": "ppo", "agent": ppo_r},
        {"name": "DeepRMSA", "type": "deep_rmsa", "agent": deep_rmsa},
        {"name": "KSP-BF", "type": "ksp_bf", "agent": None},
    ]
    backend_names = [b["name"] for b in backends]

    print("=" * 96)
    print("ORACLE-R BACKEND GAP COMPARISON")
    print(f"Topology={args.topology} slots={args.num_slots} k={args.k_paths} "
          f"max_blocks={args.max_blocks} sort={args.block_sort_strategy}")
    print(f"Seeds={seeds} episodes/seed={args.episodes} requests/episode={args.requests_per_episode}")
    print("=" * 96)

    all_ep_results = []
    for seed in seeds:
        rng = np.random.RandomState(seed)
        print(f"\n--- Seed {seed} ---")
        for ep in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            requests = generate_requests(
                env_proto,
                rng,
                src,
                args.requests_per_episode,
                arrival_interval=0.25,
                holding_min=4.0,
                holding_max=10.0,
                deadline_min=30.0,
                deadline_max=100.0,
                size_min_mb=5.0,
                size_max_mb=30.0,
                edge_cost_min=0.5,
                edge_cost_max=15.0,
                num_splits=args.num_splits,
            )
            ep_res = evaluate_episode(env_proto, args, requests, agent_c, backends)
            all_ep_results.append(ep_res)
            print(
                f"  ep {ep + 1:02d}: "
                + "  ".join(
                    f"{name} blk={ep_res[name]['backend_blocking']:.3f} "
                    f"gain={ep_res[name]['oracle_gain_pp']:.3f}"
                    for name in backend_names
                )
            )

    agg = _aggregate(all_ep_results, backend_names)

    print("\n" + "=" * 96)
    print("AGGREGATED")
    print("=" * 96)
    for name in backend_names:
        r = agg[name]
        print(
            f"{name:18s} backend_blk={r['backend_blocking']['mean']:.4f} "
            f"oracle_blk={r['oracle_blocking']['mean']:.4f} "
            f"gain={r['oracle_gain_pp']['mean']:.4f} "
            f"fail->oracle={r['fail_oracle_success_rate']['mean']:.4f} "
            f"better={r['success_oracle_better_rate']['mean']:.4f}"
        )
        print(f"  backend failures: {r['backend_reasons']}")
        print(f"  oracle failures : {r['oracle_reasons']}")

    if args.output_json:
        out_path = Path(args.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(agg, indent=2))
        print(f"\nJSON -> {out_path}")

    if args.output_md:
        out_path = Path(args.output_md)
        _write_markdown(out_path, args, agg, backend_names)
        print(f"Markdown -> {out_path}")


if __name__ == "__main__":
    main()

