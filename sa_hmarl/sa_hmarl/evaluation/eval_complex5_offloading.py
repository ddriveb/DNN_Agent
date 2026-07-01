"""Evaluate Complex5 Agent-C variants and baselines.

Compares Agent-C trained on complex5 split profile against rule-based
baselines (WO, DF, RF, IWD, Greedy) on identical 5-split request sets.

Usage:
    cd sa_hmarl
    python -m sa_hmarl.evaluation.eval_complex5_offloading \
        --default_ckpt checkpoints/agent_c_complex5_default_best.pt \
        --delayaware_ckpt checkpoints/agent_c_complex5_delayaware_best.pt
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from collections import Counter
from typing import Any, Dict, List, Optional

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
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


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


def evaluate_method(
    method_name: str,
    agent_c: Optional[PPOAgentC],
    agent_r: PPOAgentR,
    episodes_data: List[List],
    env_prototype,
    args,
    rng: Optional[np.random.RandomState] = None,
) -> Dict[str, Any]:
    """Evaluate a method on pre-generated episodes."""
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
    total_delay = 0.0
    total_fs = 0.0
    total_waste = 0.0
    total_path = 0.0
    deadline_met = 0

    mod_counter = Counter()
    split_counter = Counter()
    server_counter = Counter()
    server_success_counter = Counter()
    reason_counter = Counter()

    server_selected_count = np.zeros(num_servers, dtype=int)

    for requests in episodes_data:
        env = make_env(
            topology=topology,
            num_slots=num_slots,
            num_servers=num_servers,
            seed=42,
            slot_bw_hz=slot_bw_hz,
            guard_band_fs=guard_band_fs,
            modulation_profile=args.modulation_profile,
            server_nodes=server_nodes,
            capacities=capacities,
        )
        env.reset(requests)
        server_selected_count_ep = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]

            if method_name == "ppo":
                action_idx_c = agent_c.select_action(obs_c, deterministic=True)
            else:
                action_idx_c = select_offloading_action(
                    method_name, env, req, obs_c, mask_c,
                    rng=rng,
                    server_selected_count=server_selected_count_ep,
                )

            if action_idx_c is None:
                total += 1
                blocked += 1
                reason_counter["no_valid_c_action"] += 1
                continue

            action_c = decode_agent_c_action(action_idx_c, num_servers)
            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count_ep[server_id] += 1
            server_selected_count[server_id] += 1

            obs_r = build_agent_r_observation(env, req, split_id, server_id)
            action_idx_r = agent_r.select_action(obs_r, deterministic=True)
            if action_idx_r is None:
                action_r = (0, 0, 0)
            else:
                action_r = decode_agent_r_action(action_idx_r, len(obs_r["mod_names"]), env.max_blocks)

            _, _, _, info = env.step(action_c, action_r)
            total += 1

            if info.get("success") is False:
                blocked += 1
                reason = info.get("reason", "unknown")
                reason_counter[reason] += 1
                if server_id >= 0:
                    server_counter[f"s{server_id}_attempted"] += 1
            else:
                success += 1
                delay = info.get("delay_ms", 0.0)
                total_delay += delay
                fs = info.get("num_slots", 0)
                total_fs += fs
                waste = info.get("block_waste", 0.0)
                total_waste += waste
                path_len = info.get("path_dist_km", 0.0)
                total_path += path_len
                # Successful allocations have already passed the deadline
                # feasibility check in SMDPEnv.step(). Older env info dicts do
                # not include an explicit deadline_met field, so default True.
                if info.get("deadline_met", True):
                    deadline_met += 1
                mod_name = info.get("modulation", "unknown")
                mod_counter[mod_name] += 1
                server_success_counter[f"s{server_id}"] += 1

    blocking_rate = blocked / total if total > 0 else 0.0
    avg_delay = total_delay / success if success > 0 else 0.0
    avg_fs = total_fs / success if success > 0 else 0.0
    avg_waste = total_waste / success if success > 0 else 0.0
    avg_path = total_path / success if success > 0 else 0.0
    deadline_rate = deadline_met / success if success > 0 else 0.0

    # Objective score: BlockingRate + 0.5 * (AvgDelay_ms / 65.0)
    objective_score = blocking_rate + 0.5 * (avg_delay / 65.0)

    # Normalize split distribution
    split_dist = {}
    if total > 0:
        for k, v in sorted(split_counter.items()):
            split_dist[k] = v / total

    return {
        "method": method_name,
        "total": total,
        "blocked": blocked,
        "blocking_rate": blocking_rate,
        "success": success,
        "avg_delay_ms": avg_delay,
        "avg_fs": avg_fs,
        "avg_waste_fs": avg_waste,
        "avg_path_km": avg_path,
        "deadline_met_rate": deadline_rate,
        "objective_score": objective_score,
        "split_dist": split_dist,
        "server_dist": dict(server_counter),
        "server_success": dict(server_success_counter),
        "mod_dist": dict(mod_counter),
        "reason_dist": dict(reason_counter),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=60)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=40.0)
    parser.add_argument("--deadline_max", type=float, default=120.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=24)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5",
                        choices=["default3", "complex5", "complex5_v2"])
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--bc_r_ckpt", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--default_ckpt", type=str,
                        default="sa_hmarl/checkpoints/agent_c_complex5_default_best.pt")
    parser.add_argument("--delayaware_ckpt", type=str,
                        default="sa_hmarl/checkpoints/agent_c_complex5_delayaware_best.pt")
    parser.add_argument("--output_json", type=str,
                        default="experiments/complex5_offloading_results.json")
    parser.add_argument("--output_md", type=str,
                        default="experiments/complex5_offloading_results.md")
    args = parser.parse_args()

    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]

    # Build environment prototype
    env_proto = make_env(
        topology="snap24_gnutella_reach",
        num_slots=args.num_slots,
        num_servers=args.num_servers,
        seed=seeds[0],
        slot_bw_hz=args.slot_bw_hz,
        guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
    )

    # Load Agent-R
    mod_reg = ModulationRegistry()
    if args.modulation_profile == "extended":
        mod_reg.register_extended()
    bc_r = _load_ppo_r(args.bc_r_ckpt, mod_reg, device=args.device)

    # Load Agent-C models
    agent_c_models = {}
    if Path(args.default_ckpt).exists():
        agent_c_models["Complex5-Default"] = ("ppo", _load_ppo_c(args.default_ckpt, args.device))
    if Path(args.delayaware_ckpt).exists():
        agent_c_models["Complex5-DelayAware"] = ("ppo", _load_ppo_c(args.delayaware_ckpt, args.device))

    # Pre-generate episodes
    all_episodes = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            requests = generate_requests(
                env_proto, rng, src,
                args.requests_per_episode,
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
            eps.append(requests)
        all_episodes[seed] = eps

    # Evaluate all methods
    all_results = {}

    for label, (method, agent_c) in agent_c_models.items():
        print(f"\n--- {label} ---")
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed + 5000)
            res = evaluate_method(
                method, agent_c, bc_r,
                all_episodes[seed], env_proto, args, rng=rng,
            )
            seed_results.append(res)
            print(f"  seed {seed}: blk={res['blocking_rate']:.3f} delay={res['avg_delay_ms']:.1f}ms "
                  f"obj={res['objective_score']:.4f}")
        # Aggregate
        agg = _aggregate_results(seed_results)
        all_results[label] = agg
        print(f"  >> AVG: blk={agg['blocking_rate_mean']:.3f}±{agg['blocking_rate_std']:.3f} "
              f"delay={agg['avg_delay_ms_mean']:.1f}±{agg['avg_delay_ms_std']:.1f}ms "
              f"obj={agg['objective_score_mean']:.4f}±{agg['objective_score_std']:.4f}")
        print(f"     Splits: {_fmt_split_dist(agg['split_dist_mean'])}")

    # Baselines
    baseline_methods = ["greedy", "df", "rf", "wo", "iwd"]
    for method in baseline_methods:
        label = method.upper()
        print(f"\n--- {label} ---")
        seed_results = []
        for seed in seeds:
            rng = np.random.RandomState(seed + 5000)
            res = evaluate_method(
                method, None, bc_r,
                all_episodes[seed], env_proto, args, rng=rng,
            )
            seed_results.append(res)
            print(f"  seed {seed}: blk={res['blocking_rate']:.3f} delay={res['avg_delay_ms']:.1f}ms "
                  f"obj={res['objective_score']:.4f}")
        agg = _aggregate_results(seed_results)
        all_results[label] = agg
        print(f"  >> AVG: blk={agg['blocking_rate_mean']:.3f}±{agg['blocking_rate_std']:.3f} "
              f"delay={agg['avg_delay_ms_mean']:.1f}±{agg['avg_delay_ms_std']:.1f}ms "
              f"obj={agg['objective_score_mean']:.4f}±{agg['objective_score_std']:.4f}")
        print(f"     Splits: {_fmt_split_dist(agg['split_dist_mean'])}")

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON: {args.output_json}")

    # Write Markdown report
    _write_markdown(args.output_md, all_results, args)
    print(f"Saved Markdown: {args.output_md}")


def _aggregate_results(results: List[Dict]) -> Dict:
    """Aggregate results across seeds."""
    import numpy as np
    metrics = ["blocking_rate", "avg_delay_ms", "avg_fs", "avg_waste_fs",
               "avg_path_km", "deadline_met_rate", "objective_score"]
    agg = {}
    for m in metrics:
        vals = [r[m] for r in results]
        agg[f"{m}_mean"] = float(np.mean(vals))
        agg[f"{m}_std"] = float(np.std(vals))

    # Split distribution: average across seeds
    all_splits = set()
    for r in results:
        all_splits.update(r["split_dist"].keys())
    split_means = {}
    for s in sorted(all_splits):
        vals = [r["split_dist"].get(s, 0.0) for r in results]
        split_means[s] = float(np.mean(vals))
    agg["split_dist_mean"] = split_means
    return agg


def _fmt_split_dist(dist: Dict[str, float]) -> str:
    parts = []
    for k in sorted(dist.keys()):
        parts.append(f"{k}={dist[k]:.1%}")
    return ", ".join(parts)


def _write_markdown(path: str, results: Dict, args):
    lines = [
        "# Complex5 Offloading Evaluation Results",
        "",
        f"**Profile:** {args.split_profile} | **Splits:** {args.num_splits} | **Servers:** {args.num_servers} | **Slots:** {args.num_slots}",
        f"**Seeds:** {args.seeds} | **Episodes/seed:** {args.episodes} | **Requests/episode:** {args.requests_per_episode}",
        "",
        "## Summary",
        "",
        "| Method | Blocking | Delay (ms) | ObjScore | Split Distribution |",
        "|--------|----------|------------|----------|-------------------|",
    ]
    order = ["Complex5-Default", "Complex5-DelayAware", "GREEDY", "DF", "RF", "WO", "IWD"]
    for name in order:
        if name not in results:
            continue
        r = results[name]
        blk = f"{r['blocking_rate_mean']:.3f}±{r['blocking_rate_std']:.3f}"
        delay = f"{r['avg_delay_ms_mean']:.1f}±{r['avg_delay_ms_std']:.1f}"
        obj = f"{r['objective_score_mean']:.4f}±{r['objective_score_std']:.4f}"
        splits = _fmt_split_dist(r['split_dist_mean'])
        lines.append(f"| {name} | {blk} | {delay} | {obj} | {splits} |")

    lines.extend(["", "## Detailed Metrics", ""])
    for name in order:
        if name not in results:
            continue
        r = results[name]
        lines.extend([
            f"### {name}",
            f"- Blocking: {r['blocking_rate_mean']:.3f} ± {r['blocking_rate_std']:.3f}",
            f"- Avg Delay: {r['avg_delay_ms_mean']:.1f} ± {r['avg_delay_ms_std']:.1f} ms",
            f"- Avg FS: {r['avg_fs_mean']:.1f} ± {r['avg_fs_std']:.1f}",
            f"- Avg Waste: {r['avg_waste_fs_mean']:.1f} ± {r['avg_waste_fs_std']:.1f}",
            f"- Avg Path: {r['avg_path_km_mean']:.1f} ± {r['avg_path_km_std']:.1f} km",
            f"- Deadline Met: {r['deadline_met_rate_mean']:.1%} ± {r['deadline_met_rate_std']:.1%}",
            f"- Objective Score: {r['objective_score_mean']:.4f} ± {r['objective_score_std']:.4f}",
            f"- Splits: {_fmt_split_dist(r['split_dist_mean'])}",
            "",
        ])

    Path(path).write_text("\n".join(lines))


if __name__ == "__main__":
    main()
