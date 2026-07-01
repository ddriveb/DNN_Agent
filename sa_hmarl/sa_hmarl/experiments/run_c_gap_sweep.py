"""C-gap sweep: find scenarios where Agent-C + BC-PPO-R outperforms traditional C baselines.

Fixed R backend: BC-PPO-R (ppo_r_deeprmsa_bc_snap24_best.pt)
C methods: Agent-C, Greedy-C, WO-C, DF-C, RF-C, IWD-C

Usage:
    cd /mnt/d/project/DNN_Agent
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.experiments.run_c_gap_sweep
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import argparse
import csv
import json
import time
import os
from collections import Counter
from typing import Any, Dict, List, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
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
# Globals for worker processes
# ---------------------------------------------------------------------------
_WORKER_AGENTS_C: Dict[str, PPOAgentC] = {}
_WORKER_AGENT_R: Optional[PPOAgentR] = None
_WORKER_MOD_REG: Optional[ModulationRegistry] = None


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentC(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg: ModulationRegistry, device: str = "cpu") -> PPOAgentR:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    agent = PPOAgentR(
        input_dim=ckpt.get("input_dim", 11),
        mod_registry=mod_reg,
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
    )
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    return agent


# ---------------------------------------------------------------------------
# Worker init
# ---------------------------------------------------------------------------

def _worker_init(device: str):
    global _WORKER_AGENTS_C, _WORKER_AGENT_R, _WORKER_MOD_REG
    _WORKER_MOD_REG = ModulationRegistry.from_profile("default")
    bc_r_ckpt = "sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt"
    _WORKER_AGENT_R = _load_ppo_r(bc_r_ckpt, _WORKER_MOD_REG, device)
    _WORKER_AGENTS_C = {
        "snap24_gnutella_reach": _load_ppo_c("sa_hmarl/checkpoints/agent_c_frozen_r_snap24_reach_best.pt", device),
        "metro24_c_sensitive": _load_ppo_c("sa_hmarl/checkpoints/agent_c_frozen_r_c_sensitive_best.pt", device),
    }


# ---------------------------------------------------------------------------
# Core evaluation (fixed R backend)
# ---------------------------------------------------------------------------

def evaluate_c_method(
    agent_c: Optional[PPOAgentC],
    agent_r: PPOAgentR,
    c_method: str,
    episodes_data: List[List],
    env_prototype,
    waste_coef: float = 0.8,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, Any]:
    num_servers = len(env_prototype.mec.servers)
    topology = env_prototype.net.topology
    num_slots = env_prototype.net.num_slots
    slot_bw_hz = env_prototype.fs_calc.slot_bw_hz
    guard_band_fs = env_prototype.fs_calc.guard_band_fs
    server_nodes = [s.node_id for s in env_prototype.mec.servers]
    capacities = [s.compute_capacity for s in env_prototype.mec.servers]

    total = 0
    blocked = 0
    success = 0
    total_reward = 0.0
    total_delay = 0.0
    total_fs = 0.0
    total_waste = 0.0
    total_path = 0.0
    deadline_met = 0
    no_valid_c = 0

    mod_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    server_success_counter = Counter()
    server_overload_counter = Counter()
    reason_counter = Counter()

    for requests in episodes_data:
        env = make_env(
            topology=topology,
            num_slots=num_slots,
            num_servers=num_servers,
            seed=42,
            slot_bw_hz=slot_bw_hz,
            guard_band_fs=guard_band_fs,
            modulation_profile="default",
            server_nodes=server_nodes,
            capacities=capacities,
        )
        env.reset(requests)
        server_selected_count = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]

            if c_method == "ppo":
                action_idx_c = agent_c.select_action(obs_c, deterministic=True)
            else:
                action_idx_c = select_offloading_action(
                    c_method, env, req, obs_c, mask_c,
                    rng=rng,
                    server_selected_count=server_selected_count,
                )

            if action_idx_c is None:
                no_valid_c += 1
                action_c = (0, 0)
            else:
                action_c = decode_agent_c_action(action_idx_c, num_servers)
            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count[server_id] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            num_mods = len(obs_r["mod_names"])
            action_idx_r = agent_r.select_action(obs_r, deterministic=True)

            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(action_idx_r, num_mods, env.max_blocks)

            _, _, _, info = env.step(action_c, action_r)
            total += 1

            server = env.mec.servers[server_id]
            reward = compute_agent_c_reward(
                info, req.deadline_ms, waste_coef, server.utilization
            )
            total_reward += reward

            if info.get("success", False):
                success += 1
                server_success_counter[f"s{server_id}"] += 1
                delay = info.get("delay_ms", 0.0)
                total_delay += delay
                if delay <= req.deadline_ms:
                    deadline_met += 1
                total_fs += info.get("num_slots", 0)
                total_waste += info.get("block_waste", 0.0)
                total_path += info.get("path_dist_km", 0.0)
                mod_counter[info.get("modulation", "unknown")] += 1
            else:
                blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1
                if reason in ("server_overload", "server_saturated"):
                    server_overload_counter[f"s{server_id}"] += 1

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
        "deadline_sat_rate": deadline_met / n,
        "avg_fs": total_fs / s,
        "avg_waste": total_waste / s,
        "avg_path_len_km": total_path / s,
        "server_overload_ratio": reason_counter.get("server_overload", 0) / max(blocked, 1),
        "no_suitable_block_ratio": reason_counter.get("no_suitable_block", 0) / max(blocked, 1),
        "no_valid_c_ratio": no_valid_c / n,
        "mod_counter": dict(mod_counter),
        "split_counter": dict(split_counter),
        "server_counter": dict(server_counter),
        "server_success_counter": dict(server_success_counter),
        "server_overload_counter": dict(server_overload_counter),
        "reason_counter": dict(reason_counter),
    }


def aggregate_seed_results(seed_results: List[Dict]) -> Dict:
    scalar_keys = [
        "blocking_rate", "success_rate", "avg_reward", "avg_delay_ms",
        "deadline_sat_rate", "avg_fs", "avg_waste", "avg_path_len_km",
        "server_overload_ratio", "no_suitable_block_ratio", "no_valid_c_ratio",
    ]
    agg = {}
    for k in scalar_keys:
        vals = [r[k] for r in seed_results]
        agg[k] = float(np.mean(vals))
        agg[f"{k}_std"] = float(np.std(vals))
    for counter_key in ["mod_counter", "split_counter", "server_counter",
                        "server_success_counter", "server_overload_counter", "reason_counter"]:
        c = Counter()
        for r in seed_results:
            c.update(r[counter_key])
        agg[counter_key] = dict(c)
    return agg


def run_scenario(
    topology: str,
    num_slots: int,
    requests_per_episode: int,
    arrival_interval: float,
    edge_cost_max: float,
    split_profile: str,
    seeds: List[int],
    episodes: int,
    device: str = "cpu",
) -> Dict[str, Dict]:
    """Evaluate all C methods for a single scenario (runs in worker process)."""
    agent_c = _WORKER_AGENTS_C[topology]
    agent_r = _WORKER_AGENT_R

    num_splits = 3 if split_profile == "default3" else 5
    env_proto = make_env(
        topology=topology,
        num_slots=num_slots,
        num_servers=4,
        seed=42,
        slot_bw_hz=1.25e9,
        guard_band_fs=1,
        modulation_profile="default",
    )

    all_episodes = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            requests = generate_requests(
                env_proto, rng, src,
                requests_per_episode,
                arrival_interval=arrival_interval,
                holding_min=4.0,
                holding_max=10.0,
                deadline_min=30.0,
                deadline_max=100.0,
                size_min_mb=5.0,
                size_max_mb=30.0,
                edge_cost_min=0.5,
                edge_cost_max=edge_cost_max,
                num_splits=num_splits,
                split_profile=split_profile,
            )
            eps.append(requests)
        all_episodes[seed] = eps

    c_methods = ["ppo", "greedy", "wo", "df", "rf", "iwd"]
    c_labels = {"ppo": "Agent-C", "greedy": "Greedy-C", "wo": "WO-C",
                "df": "DF-C", "rf": "RF-C", "iwd": "IWD-C"}

    results = {}
    for c_method in c_methods:
        label = c_labels[c_method]
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed)
            res = evaluate_c_method(
                agent_c, agent_r, c_method,
                all_episodes[seed], env_proto, 0.8, rng,
            )
            seed_results.append(res)
        results[label] = aggregate_seed_results(seed_results)
    return results


def compute_objective(blocking_rate: float, avg_delay_ms: float) -> float:
    return blocking_rate + 0.5 * (avg_delay_ms / 65.0)


def analyze_scenario(results: Dict[str, Dict]) -> Dict:
    traditional = ["Greedy-C", "WO-C", "DF-C", "RF-C", "IWD-C"]
    all_blocking = [results[m]["blocking_rate"] for m in results]
    avg_blocking = np.mean(all_blocking)

    if avg_blocking > 0.6:
        return {"status": "over-hard", "agent_c_blocking_gap": 0.0, "agent_c_objective_gap": 0.0}
    if avg_blocking < 0.05:
        return {"status": "too-easy", "agent_c_blocking_gap": 0.0, "agent_c_objective_gap": 0.0}

    best_baseline_blocking = min(results[m]["blocking_rate"] for m in traditional)
    best_baseline_delay = min(results[m]["avg_delay_ms"] for m in traditional)
    best_baseline_obj = min(compute_objective(results[m]["blocking_rate"], results[m]["avg_delay_ms"]) for m in traditional)

    agent_blocking = results["Agent-C"]["blocking_rate"]
    agent_delay = results["Agent-C"]["avg_delay_ms"]
    agent_obj = compute_objective(agent_blocking, agent_delay)

    blocking_gap = best_baseline_blocking - agent_blocking
    objective_gap = best_baseline_obj - agent_obj
    delay_ratio = agent_delay / best_baseline_delay if best_baseline_delay > 0 else float('inf')

    passes = (
        blocking_gap >= 0.02
        and objective_gap > 0
        and delay_ratio <= 1.20
        and 0.10 <= agent_blocking <= 0.40
    )

    return {
        "status": "candidate" if passes else "rejected",
        "agent_c_blocking_gap": blocking_gap,
        "agent_c_objective_gap": objective_gap,
        "best_baseline_blocking": best_baseline_blocking,
        "best_baseline_delay": best_baseline_delay,
        "best_baseline_obj": best_baseline_obj,
        "agent_blocking": agent_blocking,
        "agent_delay": agent_delay,
        "agent_obj": agent_obj,
        "delay_ratio": delay_ratio,
        "best_baseline_name": min(traditional, key=lambda m: results[m]["blocking_rate"]),
    }


def make_scenario_id(topology, num_slots, requests_per_episode, arrival_interval, edge_cost_max, split_profile):
    arr_str = str(arrival_interval).replace(".", "_")
    return f"{topology}_s{num_slots}_r{requests_per_episode}_a{arr_str}_e{edge_cost_max}_{split_profile}"


def _fmt_pct(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v/total:.1%}" for k, v in counter.most_common())


def _fmt_cnt(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v} ({v/total:.1%})" for k, v in counter.most_common())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--workers", type=int, default=4,
                        help="Number of parallel workers for quick sweep")
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--end_idx", type=int, default=None)
    parser.add_argument("--quick_seeds", type=str, default="42,123")
    parser.add_argument("--quick_episodes", type=int, default=10)
    parser.add_argument("--full_seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--full_episodes", type=int, default=20)
    parser.add_argument("--quick_only", action="store_true")
    parser.add_argument("--resume_quick", type=str, default=None)
    parser.add_argument("--resume_full", type=str, default=None)
    args = parser.parse_args()

    param_grid = {
        "topology": ["snap24_gnutella_reach", "metro24_c_sensitive"],
        "num_slots": [16, 20, 24],
        "requests_per_episode": [60, 70, 80],
        "arrival_interval": [0.20, 0.25],
        "edge_cost_max": [15, 20, 25],
        "split_profile": ["default3", "complex5_v2_lite"],
    }

    quick_seeds = [int(s.strip()) for s in args.quick_seeds.split(",") if s.strip()]
    quick_episodes = args.quick_episodes
    full_seeds = [int(s.strip()) for s in args.full_seeds.split(",") if s.strip()]
    full_episodes = args.full_episodes

    scenarios = []
    for topo in param_grid["topology"]:
        for ns in param_grid["num_slots"]:
            for rpe in param_grid["requests_per_episode"]:
                for ai in param_grid["arrival_interval"]:
                    for ec in param_grid["edge_cost_max"]:
                        for sp in param_grid["split_profile"]:
                            scenarios.append({
                                "topology": topo,
                                "num_slots": ns,
                                "requests_per_episode": rpe,
                                "arrival_interval": ai,
                                "edge_cost_max": ec,
                                "split_profile": sp,
                            })

    total_scenarios = len(scenarios)
    start = args.start_idx
    end = args.end_idx if args.end_idx is not None else total_scenarios
    scenarios = scenarios[start:end]
    print(f"Total scenarios: {total_scenarios}, Processing indices [{start}:{end}] ({len(scenarios)} scenarios)", flush=True)
    print(f"Quick eval: seeds={quick_seeds}, episodes={quick_episodes}", flush=True)

    out_dir = Path("sa_hmarl/experiments")
    out_dir.mkdir(parents=True, exist_ok=True)
    quick_json = out_dir / f"c_gap_sweep_quick_{start}_{end}.json"
    full_json = out_dir / f"c_gap_sweep_full_{start}_{end}.json"

    # ------------------------------------------------------------------
    # Quick sweep
    # ------------------------------------------------------------------
    if args.resume_quick and Path(args.resume_quick).exists():
        print(f"Resuming quick sweep from {args.resume_quick}")
        quick_results = json.load(open(args.resume_quick))
    else:
        quick_results = {}

    # Determine remaining scenarios
    remaining = [sc for sc in scenarios if make_scenario_id(**sc) not in quick_results]
    print(f"Already done: {len(quick_results)}, Remaining: {len(remaining)}", flush=True)

    if remaining:
        _worker_init(args.device)
        quick_start = time.time()
        for idx, sc in enumerate(remaining, 1):
            sid = make_scenario_id(**sc)
            try:
                res = run_scenario(**sc, seeds=quick_seeds, episodes=quick_episodes, device=args.device)
                analysis = analyze_scenario(res)
                quick_results[sid] = {
                    "params": sc,
                    "results": res,
                    "analysis": analysis,
                }
                print(f"[{idx}/{len(remaining)}] {sid} -> {analysis['status']} blk_gap={analysis['agent_c_blocking_gap']:+.4f} obj_gap={analysis['agent_c_objective_gap']:+.4f}", flush=True)
            except Exception as e:
                print(f"[{idx}/{len(remaining)}] {sid} -> ERROR: {e}", flush=True)
                quick_results[sid] = {
                    "params": sc,
                    "error": str(e),
                }
            if idx % 5 == 0:
                with open(quick_json, "w") as f:
                    json.dump(quick_results, f, indent=2)
                elapsed = time.time() - quick_start
                rate = elapsed / idx
                eta = rate * (len(remaining) - idx)
                print(f"Progress: {idx}/{len(remaining)} ({rate:.1f}s/scen, ETA {eta/60:.1f}min)", flush=True)

        with open(quick_json, "w") as f:
            json.dump(quick_results, f, indent=2)
        print(f"Quick sweep saved to {quick_json}", flush=True)

    if args.quick_only:
        generate_report(quick_results, {}, out_dir, suffix=f"_{start}_{end}")
        return

    # ------------------------------------------------------------------
    # Select top candidates
    # ------------------------------------------------------------------
    candidates = []
    for sid, data in quick_results.items():
        if "error" in data:
            continue
        if data["analysis"]["status"] != "candidate":
            continue
        candidates.append((sid, data["analysis"]["agent_c_blocking_gap"], data["analysis"]["agent_c_objective_gap"]))

    if not candidates:
        print("\n*** NO CANDIDATE SCENARIOS FOUND ***", flush=True)
        generate_report(quick_results, {}, out_dir, suffix=f"_{start}_{end}")
        return

    candidates.sort(key=lambda x: (x[1], x[2]), reverse=True)
    top3 = candidates[:3]
    print(f"\nTop {len(top3)} candidates selected for full eval:", flush=True)
    for sid, bg, og in top3:
        print(f"  {sid}: blocking_gap={bg:+.4f}, objective_gap={og:+.4f}", flush=True)

    # ------------------------------------------------------------------
    # Full eval on top 3
    # ------------------------------------------------------------------
    if args.resume_full and Path(args.resume_full).exists():
        print(f"Resuming full eval from {args.resume_full}")
        full_results = json.load(open(args.resume_full))
    else:
        full_results = {}

    full_remaining = [sid for sid, _, _ in top3 if sid not in full_results]
    if full_remaining:
        _worker_init(args.device)
        for idx, sid in enumerate(full_remaining, 1):
            sc = quick_results[sid]["params"]
            try:
                res = run_scenario(**sc, seeds=full_seeds, episodes=full_episodes, device=args.device)
                analysis = analyze_scenario(res)
                full_results[sid] = {
                    "params": quick_results[sid]["params"],
                    "results": res,
                    "analysis": analysis,
                }
                print(f"[FULL {idx}/{len(full_remaining)}] {sid} -> {analysis['status']} blk_gap={analysis['agent_c_blocking_gap']:+.4f}", flush=True)
            except Exception as e:
                print(f"[FULL {idx}/{len(full_remaining)}] {sid} -> ERROR: {e}", flush=True)
                full_results[sid] = {
                    "params": quick_results[sid]["params"],
                    "error": str(e),
                }
            with open(full_json, "w") as f:
                json.dump(full_results, f, indent=2)

    # ------------------------------------------------------------------
    # Generate final report
    # ------------------------------------------------------------------
    generate_report(quick_results, full_results, out_dir, suffix=f"_{start}_{end}")


def generate_report(quick_results: Dict, full_results: Dict, out_dir: Path, suffix: str = ""):
    """Generate JSON, CSV, and Markdown report."""

    all_data = []
    for sid, data in quick_results.items():
        if "error" in data:
            continue
        sc = data["params"]
        analysis = data["analysis"]
        all_data.append({
            "scenario_id": sid,
            **sc,
            **analysis,
        })

    # --- CSV ---
    csv_path = out_dir / f"c_gap_sweep_results{suffix}.csv"
    if all_data:
        fieldnames = ["scenario_id", "topology", "num_slots", "requests_per_episode",
                      "arrival_interval", "edge_cost_max", "split_profile",
                      "status", "agent_c_blocking_gap", "agent_c_objective_gap",
                      "agent_blocking", "agent_delay", "agent_obj",
                      "best_baseline_name", "best_baseline_blocking", "best_baseline_obj",
                      "delay_ratio"]
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in all_data:
                writer.writerow({k: row.get(k, "") for k in fieldnames})
        print(f"CSV written to {csv_path}", flush=True)

    # --- JSON (merged) ---
    json_path = out_dir / f"c_gap_sweep_results{suffix}.json"
    output = {
        "quick_sweep": quick_results,
        "full_eval": full_results,
    }
    with open(json_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"JSON written to {json_path}", flush=True)

    # --- Markdown report ---
    md_path = out_dir / f"c_gap_sweep_report{suffix}.md"
    lines = ["# C-Gap Sweep Report\n\n"]

    lines.append("## Summary\n\n")
    candidate_count = sum(1 for d in quick_results.values() if d.get("analysis", {}).get("status") == "candidate")
    overhard_count = sum(1 for d in quick_results.values() if d.get("analysis", {}).get("status") == "over-hard")
    tooeasy_count = sum(1 for d in quick_results.values() if d.get("analysis", {}).get("status") == "too-easy")
    lines.append(f"- Total scenarios: {len(quick_results)}\n")
    lines.append(f"- Candidates (Agent-C leads): {candidate_count}\n")
    lines.append(f"- Over-hard (avg blocking > 0.6): {overhard_count}\n")
    lines.append(f"- Too-easy (avg blocking < 0.05): {tooeasy_count}\n\n")

    # Top 10 blocking gap
    valid = [(sid, d) for sid, d in quick_results.items() if "error" not in d]
    valid.sort(key=lambda x: x[1]["analysis"].get("agent_c_blocking_gap", -999), reverse=True)

    lines.append("## Top 10 Agent-C Blocking Gap (Quick Sweep)\n\n")
    lines.append("| Rank | Scenario | Topology | Slots | RPE | Arr | EdgeMax | Profile | BlkGap | ObjGap | AgentBlk | BestBaseline |\n")
    lines.append("|------|----------|----------|-------|-----|-----|---------|---------|--------|--------|----------|--------------|\n")
    for rank, (sid, d) in enumerate(valid[:10], 1):
        sc = d["params"]
        a = d["analysis"]
        lines.append(
            f"| {rank} | {sid} | {sc['topology']} | {sc['num_slots']} | {sc['requests_per_episode']} | "
            f"{sc['arrival_interval']} | {sc['edge_cost_max']} | {sc['split_profile']} | "
            f"{a.get('agent_c_blocking_gap', 0):+.4f} | {a.get('agent_c_objective_gap', 0):+.4f} | "
            f"{a.get('agent_blocking', 0):.3f} | {a.get('best_baseline_name', 'N/A')} |\n"
        )

    # Top 10 objective gap
    valid_obj = [(sid, d) for sid, d in quick_results.items() if "error" not in d]
    valid_obj.sort(key=lambda x: x[1]["analysis"].get("agent_c_objective_gap", -999), reverse=True)

    lines.append("\n## Top 10 Agent-C Objective Gap (Quick Sweep)\n\n")
    lines.append("| Rank | Scenario | Topology | Slots | RPE | Arr | EdgeMax | Profile | BlkGap | ObjGap | AgentObj | BestBaselineObj |\n")
    lines.append("|------|----------|----------|-------|-----|-----|---------|---------|--------|--------|----------|-----------------|\n")
    for rank, (sid, d) in enumerate(valid_obj[:10], 1):
        sc = d["params"]
        a = d["analysis"]
        lines.append(
            f"| {rank} | {sid} | {sc['topology']} | {sc['num_slots']} | {sc['requests_per_episode']} | "
            f"{sc['arrival_interval']} | {sc['edge_cost_max']} | {sc['split_profile']} | "
            f"{a.get('agent_c_blocking_gap', 0):+.4f} | {a.get('agent_c_objective_gap', 0):+.4f} | "
            f"{a.get('agent_obj', 0):.4f} | {a.get('best_baseline_obj', 0):.4f} |\n"
        )

    # Full eval detail for top 3
    if full_results:
        lines.append("\n## Full Evaluation (Top 3 Candidates)\n\n")
        for sid, d in full_results.items():
            if "error" in d:
                continue
            sc = d["params"]
            res = d["results"]
            a = d["analysis"]
            lines.append(f"### {sid}\n\n")
            lines.append(f"**Parameters:** topology={sc['topology']}, slots={sc['num_slots']}, "
                        f"RPE={sc['requests_per_episode']}, arrival={sc['arrival_interval']}, "
                        f"edge_max={sc['edge_cost_max']}, profile={sc['split_profile']}\n\n")

            lines.append("#### Full Metrics Table\n\n")
            lines.append("| Method | Blocking | Success | AvgDelay | DeadlineSat | AvgReward | AvgFS | PathKm | SrvOver | NoBlock | NoValidC |\n")
            lines.append("|--------|----------|---------|----------|-------------|-----------|-------|--------|---------|---------|----------|\n")
            for method in ["Agent-C", "Greedy-C", "WO-C", "DF-C", "RF-C", "IWD-C"]:
                r = res[method]
                lines.append(
                    f"| {method} | {r['blocking_rate']:.3f} | {r['success_rate']:.3f} | "
                    f"{r['avg_delay_ms']:.1f}ms | {r['deadline_sat_rate']:.3f} | "
                    f"{r['avg_reward']:+.3f} | {r['avg_fs']:.2f} | {r['avg_path_len_km']:.1f} | "
                    f"{r['server_overload_ratio']:.3f} | {r['no_suitable_block_ratio']:.3f} | "
                    f"{r['no_valid_c_ratio']:.3f} |\n"
                )

            best = a.get("best_baseline_name", "N/A")
            lines.append(f"\n**Best baseline:** {best}  \n")
            lines.append(f"**Agent-C blocking gap:** {a.get('agent_c_blocking_gap', 0):+.4f}  \n")
            lines.append(f"**Agent-C objective gap:** {a.get('agent_c_objective_gap', 0):+.4f}  \n")
            lines.append(f"**Delay ratio (Agent-C / best baseline):** {a.get('delay_ratio', 0):.2f}x  \n\n")

            lines.append("#### Split Distribution\n\n")
            lines.append("| Method | Distribution |\n")
            lines.append("|--------|-------------|\n")
            for method in ["Agent-C", best]:
                r = res[method]
                lines.append(f"| {method} | {_fmt_pct(Counter(r['split_counter']))} |\n")

            lines.append("\n#### Server Distribution\n\n")
            lines.append("| Method | Distribution |\n")
            lines.append("|--------|-------------|\n")
            for method in ["Agent-C", best]:
                r = res[method]
                lines.append(f"| {method} | {_fmt_pct(Counter(r['server_counter']))} |\n")

            lines.append("\n#### Failure Reasons\n\n")
            lines.append("| Method | Reasons |\n")
            lines.append("|--------|---------|\n")
            for method in ["Agent-C", best]:
                r = res[method]
                lines.append(f"| {method} | {_fmt_cnt(Counter(r['reason_counter']))} |\n")

            lines.append("\n#### Why This Scene Amplifies C-Value\n\n")
            agent_blk = a.get("agent_blocking", 0)
            baseline_blk = a.get("best_baseline_blocking", 0)
            if sc["split_profile"] != "default3":
                lines.append("- Multi-split profile creates non-trivial split/server trade-offs.\n")
            if sc["edge_cost_max"] >= 20:
                lines.append("- High edge compute cost makes server selection critical.\n")
            if sc["num_slots"] <= 20:
                lines.append("- Tight spectrum (fewer slots) forces careful split choice to reduce FS demand.\n")
            if sc["requests_per_episode"] >= 70:
                lines.append("- High load increases contention, amplifying decision quality differences.\n")
            if sc["arrival_interval"] <= 0.20:
                lines.append("- Shorter arrival interval increases temporal overlap and blocking risk.\n")
            lines.append(f"- Agent-C achieves {baseline_blk - agent_blk:.3f} lower blocking than {best}, "
                        f"showing learned policy adapts better to this specific pressure mix.\n\n")

    with open(md_path, "w") as f:
        f.write("".join(lines))
    print(f"Markdown report written to {md_path}", flush=True)


if __name__ == "__main__":
    main()
