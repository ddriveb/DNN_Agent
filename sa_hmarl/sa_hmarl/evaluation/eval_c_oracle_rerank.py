"""[DEPRECATED] Use :mod:`eval_c_oracle_and_rerank` instead.

This file is kept for reference only.  ``eval_c_oracle_and_rerank.py`` is the
canonical evaluation script with the same methods plus:
- proper risk-mask handling for TopK-rerank
- one-step myopic Oracle-C using the original (unclipped) C mask
- integrated ``c_candidate_pressure`` / ``c_topk_rerank`` modules
- richer per-server overload and distribution reporting

---

Diagnose Agent-C headroom with top-K reranking and Oracle-C.

The purpose of this script is not to introduce a new training algorithm.  It
answers two diagnostic questions:

1. If Agent-C's top-K candidates are reranked by lightweight spectrum pressure,
   does blocking decrease?
2. If C selection were oracle-perfect for the current request and current R,
   how much lower could blocking go?

Oracle-C deep-copies the current environment, tries every mask-valid
(split, server) candidate with the current Agent-R, and then executes the
one-step best candidate on the real environment.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

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


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode",
        ckpt.get("agent_c_feature_mode", "default"),
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


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> PPOAgentR:
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


def _agent_c_features_and_mask(agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int):
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c,
            mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    return features, mask


def _agent_c_logits(agent_c: PPOAgentC, features: np.ndarray, mask: np.ndarray) -> np.ndarray:
    if len(mask) == 0 or features.size == 0:
        return np.array([], dtype=np.float32)
    with torch.no_grad():
        x = torch.tensor(features, dtype=torch.float32, device=agent_c.device).unsqueeze(0)
        logits = agent_c.policy_net(x).squeeze(0).cpu().numpy()
    logits = np.array(logits, dtype=np.float32, copy=True)
    logits[~mask] = -np.inf
    return logits


def _select_agent_c(agent_c: PPOAgentC, obs_c: Dict[str, Any], num_slots: int) -> Optional[int]:
    features, mask = _agent_c_features_and_mask(agent_c, obs_c, num_slots)
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


def _candidate_pressure_score(obs_c: Dict[str, Any], action_idx: int, num_slots: int) -> float:
    feat = obs_c["candidate_features"][action_idx]
    spec = feat.get("spectrum_summary", [])
    if isinstance(spec, np.ndarray):
        spec = spec.tolist()

    feasible_count = float(feat.get("feasible_count", 0))
    best_fs = feat.get("best_fs_estimate")
    safe_fs = feat.get("safe_fs_estimate")
    best_fs = float(best_fs) if best_fs is not None else float(num_slots)
    safe_fs = float(safe_fs) if safe_fs is not None else best_fs

    lfb_max = max(float(spec[0]) if len(spec) > 0 else float(num_slots), 1.0)
    lfb_p25 = max(float(spec[3]) if len(spec) > 3 else lfb_max, 1.0)
    free_mean = float(spec[6]) if len(spec) > 6 else 0.0
    frag_mean = float(spec[4]) if len(spec) > 4 else 1.0
    min_delay_s = float(spec[9]) if len(spec) > 9 else 0.0

    deadline_ms = max(float(obs_c["request_features"].get("deadline_ms", 100.0)), 1.0)
    local_ms = float(feat.get("local_compute_ms", 0.0))
    edge_ms = float(feat.get("edge_compute_ms", 0.0))
    if not np.isfinite(edge_ms):
        edge_ms = 1e6

    best_pressure = best_fs / lfb_max
    safe_pressure = safe_fs / lfb_p25
    delay_norm = (local_ms + edge_ms + min_delay_s * 1000.0) / deadline_ms
    scarcity = 1.0 / (1.0 + max(feasible_count, 0.0))

    return (
        1.2 * scarcity
        + 0.8 * best_pressure
        + 0.8 * safe_pressure
        + 0.4 * frag_mean
        + 0.2 * (1.0 - free_mean)
        + 0.5 * delay_norm
    )


def _select_topk_rerank(
    agent_c: PPOAgentC,
    obs_c: Dict[str, Any],
    num_slots: int,
    top_k: int,
) -> Optional[int]:
    features, mask = _agent_c_features_and_mask(agent_c, obs_c, num_slots)
    logits = _agent_c_logits(agent_c, features, mask)
    valid = np.where(np.isfinite(logits))[0]
    if len(valid) == 0:
        return None
    ranked = valid[np.argsort(logits[valid])[::-1]]
    top = ranked[: max(1, min(top_k, len(ranked)))]
    return int(min(top, key=lambda idx: _candidate_pressure_score(obs_c, int(idx), num_slots)))


def _select_r(agent_r: PPOAgentR, env, req, split_id: int, server_id: int):
    obs_r = build_agent_r_observation(env, req, split_id, server_id)
    action_idx_r = agent_r.select_action(obs_r, deterministic=True)
    if action_idx_r is None:
        return (0, 0, 0), None
    return decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), env.max_blocks), action_idx_r


def _oracle_score(info: Dict[str, Any], req_deadline_ms: float, waste_coef: float) -> Tuple:
    if info.get("success", False):
        delay = float(info.get("delay_ms", 1e9))
        waste = float(info.get("block_waste", 1.0))
        fs = float(info.get("num_slots", 1e9))
        path = float(info.get("path_dist_km", 1e9))
        deadline_violation = 1.0 if delay > req_deadline_ms else 0.0
        return (0, deadline_violation, delay / max(req_deadline_ms, 1.0), waste_coef * waste, fs, path)
    reason = info.get("reason", "unknown")
    reason_rank = {
        "server_overload": 1,
        "server_saturated": 1,
        "no_suitable_block": 2,
        "fs_too_large": 3,
        "deadline_infeasible": 4,
    }.get(reason, 5)
    return (1, reason_rank, 1e9, 1e9, 1e9, 1e9)


def _select_oracle_c(
    agent_r: PPOAgentR,
    env,
    req,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    waste_coef: float,
) -> Tuple[Tuple[int, int], Tuple[int, int, int], Dict[str, Any]]:
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return (0, 0), (0, 0, 0), {"success": False, "reason": "no_valid_c_oracle"}

    best = None
    num_servers = len(env.mec.servers)
    for action_idx_c in valid:
        action_c = decode_agent_c_action(int(action_idx_c), num_servers)
        split_id, server_id = action_c
        action_r, _ = _select_r(agent_r, env, req, split_id, server_id)
        env_try = copy.deepcopy(env)
        _, _, _, info_try = env_try.step(action_c, action_r)
        score = _oracle_score(info_try, req.deadline_ms, waste_coef)
        if best is None or score < best[0]:
            best = (score, action_c, action_r, info_try)

    return best[1], best[2], best[3]


def evaluate_method(
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

    total = blocked = success = deadline_met = 0
    total_reward = total_delay = total_fs = total_waste = total_path = 0.0
    split_counter = Counter()
    server_counter = Counter()
    reason_counter = Counter()
    mod_counter = Counter()

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
        )
        env.reset(requests)
        server_selected_count = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            features, mask = _agent_c_features_and_mask(agent_c, obs_c, env.net.num_slots)

            if method == "agent":
                action_idx_c = _select_agent_c(agent_c, obs_c, env.net.num_slots)
                action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(action_idx_c, num_servers)
                action_r, _ = _select_r(agent_r, env, req, action_c[0], action_c[1])
            elif method == "topk_rerank":
                action_idx_c = _select_topk_rerank(agent_c, obs_c, env.net.num_slots, args.top_k)
                action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(action_idx_c, num_servers)
                action_r, _ = _select_r(agent_r, env, req, action_c[0], action_c[1])
            elif method == "oracle":
                action_c, action_r, _ = _select_oracle_c(agent_r, env, req, obs_c, mask, args.waste_coef)
            else:
                action_idx_c = select_offloading_action(
                    method,
                    env,
                    req,
                    obs_c,
                    obs_c["agent_c_mask"],
                    rng=np.random.RandomState(args.seed),
                    server_selected_count=server_selected_count,
                )
                action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(action_idx_c, num_servers)
                action_r, _ = _select_r(agent_r, env, req, action_c[0], action_c[1])

            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count[server_id] += 1

            _, _, _, info = env.step(action_c, action_r)
            total += 1
            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(info, req.deadline_ms, args.waste_coef, server.utilization)
            total_reward += reward

            if info.get("success", False):
                success += 1
                delay = float(info.get("delay_ms", 0.0))
                total_delay += delay
                total_fs += float(info.get("num_slots", 0))
                total_waste += float(info.get("block_waste", 0.0))
                total_path += float(info.get("path_dist_km", 0.0))
                if delay <= req.deadline_ms:
                    deadline_met += 1
                mod_counter[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reason_counter[info.get("reason", "unknown")] += 1

    n = max(total, 1)
    s = max(success, 1)
    avg_delay = total_delay / s
    blocking = blocked / n
    return {
        "blocking_rate": blocking,
        "success_rate": success / n,
        "avg_reward": total_reward / n,
        "avg_delay_ms": avg_delay,
        "deadline_sat_rate": deadline_met / n,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "objective": blocking + args.delay_objective_coef * (avg_delay / max(args.deadline_max, 1.0)),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "reason_counter": dict(reason_counter),
        "mod_counter": dict(mod_counter),
    }


def aggregate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km", "objective",
    ]
    out = {}
    for key in keys:
        vals = [float(r[key]) for r in results]
        out[key] = float(np.mean(vals))
        out[f"{key}_std"] = float(np.std(vals))
    for key in ["split_counter", "server_counter", "reason_counter", "mod_counter"]:
        c = Counter()
        for r in results:
            c.update(r[key])
        out[key] = dict(c)
    return out


def _fmt_counter(counter: Dict[str, int]) -> str:
    c = Counter(counter)
    total = sum(c.values())
    if total <= 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in sorted(c.items()))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--max_blocks", type=int, default=5)
    parser.add_argument("--block_sort_strategy", type=str, default="size_desc",
                        choices=["size_desc", "waste_asc", "start_asc", "center_asc", "mixed"])
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.20)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--delay_objective_coef", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--methods", type=str, default="agent,topk_rerank,oracle,iwd,df,greedy")
    parser.add_argument("--out_json", type=str, default="sa_hmarl/experiments/c_oracle_rerank_eval.json")
    parser.add_argument("--out_md", type=str, default="sa_hmarl/experiments/c_oracle_rerank_eval.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    methods = [m.strip() for m in args.methods.split(",") if m.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
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
    )

    all_results = {}
    for method in methods:
        print(f"\nEvaluating {method}...")
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            episodes_data = []
            for _ in range(args.episodes):
                src = rng.randint(0, env_proto.net.NUM_NODES)
                episodes_data.append(
                    generate_requests(
                        env_proto,
                        rng,
                        src,
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
            res = evaluate_method(method, agent_c, agent_r, episodes_data, env_proto, args)
            seed_results.append(res)
            print(
                f"  seed={seed}: blk={res['blocking_rate']:.4f} "
                f"delay={res['avg_delay_ms']:.2f} obj={res['objective']:.4f}"
            )
        all_results[method] = aggregate(seed_results)
        r = all_results[method]
        print(
            f"=> {method}: blk={r['blocking_rate']:.4f} "
            f"delay={r['avg_delay_ms']:.2f} obj={r['objective']:.4f}"
        )

    Path(args.out_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_json).write_text(json.dumps({"args": vars(args), "results": all_results}, indent=2), encoding="utf-8")

    lines = []
    lines.append("# Agent-C Oracle and Top-K Rerank Diagnostic")
    lines.append("")
    lines.append(
        f"Scenario: `{args.topology}`, slots={args.num_slots}, "
        f"requests/episode={args.requests_per_episode}, split_profile={args.split_profile}"
    )
    lines.append("")
    lines.append("| Method | Blocking | AvgReward | Delay(ms) | Objective | AvgFS | Waste | PathKm |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|")
    for method in methods:
        r = all_results[method]
        lines.append(
            f"| {method} | {r['blocking_rate']:.4f} | {r['avg_reward']:+.4f} | "
            f"{r['avg_delay_ms']:.2f} | {r['objective']:.4f} | {r['avg_fs']:.2f} | "
            f"{r['avg_waste']:.3f} | {r['avg_path_len_km']:.1f} |"
        )
    lines.append("")
    lines.append("## Split Distribution")
    lines.append("")
    for method in methods:
        lines.append(f"- {method}: {_fmt_counter(all_results[method]['split_counter'])}")
    lines.append("")
    lines.append("## Server Distribution")
    lines.append("")
    for method in methods:
        lines.append(f"- {method}: {_fmt_counter(all_results[method]['server_counter'])}")
    lines.append("")
    lines.append("## Failure Reasons")
    lines.append("")
    for method in methods:
        lines.append(f"- {method}: {_fmt_counter(all_results[method]['reason_counter'])}")

    Path(args.out_md).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nSaved JSON: {args.out_json}")
    print(f"Saved report: {args.out_md}")
    print("\n" + "\n".join(lines))


if __name__ == "__main__":
    main()
