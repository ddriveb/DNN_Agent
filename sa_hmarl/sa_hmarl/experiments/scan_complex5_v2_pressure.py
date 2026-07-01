"""Pressure scan for complex5_v2 split profile.

Scans a grid of (num_slots, requests_per_episode, arrival_interval) to find
a challenging but tractable scenario where baselines show non-trivial blocking
(10%-25% ideally) and diverse failure reasons.

Usage:
    PYTHONPATH=sa_hmarl python -m sa_hmarl.experiments.scan_complex5_v2_pressure
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
from collections import Counter
from typing import Any, Dict, List

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentR
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


def evaluate_baseline(
    method_name: str,
    agent_r: PPOAgentR,
    episodes_data: List[List],
    env_prototype,
    num_slots: int,
) -> Dict[str, Any]:
    """Evaluate a single baseline method on pre-generated episodes."""
    num_servers = len(env_prototype.mec.servers)
    topology = env_prototype.net.topology
    slot_bw_hz = env_prototype.fs_calc.slot_bw_hz
    guard_band_fs = env_prototype.fs_calc.guard_band_fs
    server_nodes = [s.node_id for s in env_prototype.mec.servers]
    capacities = [s.compute_capacity for s in env_prototype.mec.servers]

    total = 0
    blocked = 0
    success = 0
    no_valid_c = 0
    total_delay = 0.0
    total_fs = 0.0
    deadline_met = 0
    valid_action_counts = []

    split_counter = Counter()
    reason_counter = Counter()
    server_counter = Counter()

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
        server_selected_count_ep = np.zeros(num_servers, dtype=int)

        for req in requests:
            obs_c = build_agent_c_observation(env, req)
            mask_c = obs_c["agent_c_mask"]
            valid_action_counts.append(int(np.sum(mask_c)))
            action_idx_c = select_offloading_action(
                method_name, env, req, obs_c, mask_c,
                rng=np.random.RandomState(0),
                server_selected_count=server_selected_count_ep,
            )
            if action_idx_c is None:
                total += 1
                blocked += 1
                no_valid_c += 1
                reason_counter["no_valid_c_action"] += 1
                continue

            action_c = decode_agent_c_action(action_idx_c, num_servers)
            split_id, server_id = action_c
            split_counter[f"split{split_id}"] += 1
            server_counter[f"s{server_id}"] += 1
            server_selected_count_ep[server_id] += 1

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
                reason_counter[info.get("reason", "unknown")] += 1
            else:
                success += 1
                total_delay += info.get("delay_ms", 0.0)
                total_fs += info.get("num_slots", 0)
                if info.get("deadline_met", True):
                    deadline_met += 1

    blocking_rate = blocked / total if total > 0 else 0.0
    avg_delay = total_delay / success if success > 0 else 0.0
    avg_fs = total_fs / success if success > 0 else 0.0
    deadline_rate = deadline_met / success if success > 0 else 0.0
    objective_score = blocking_rate + 0.5 * (avg_delay / 65.0)

    split_dist = {k: v / total for k, v in sorted(split_counter.items())} if total > 0 else {}

    return {
        "method": method_name,
        "total": total,
        "blocked": blocked,
        "blocking_rate": blocking_rate,
        "avg_delay_ms": avg_delay,
        "avg_fs": avg_fs,
        "deadline_met_rate": deadline_rate,
        "objective_score": objective_score,
        "split_dist": split_dist,
        "reason_dist": dict(reason_counter),
        "server_dist": dict(server_counter),
        "no_valid_c": no_valid_c,
        "valid_action_mean": float(np.mean(valid_action_counts)) if valid_action_counts else 0.0,
        "valid_action_min": int(np.min(valid_action_counts)) if valid_action_counts else 0,
        "valid_action_max": int(np.max(valid_action_counts)) if valid_action_counts else 0,
    }


def scan_combo(
    num_slots: int,
    requests_per_episode: int,
    arrival_interval: float,
    seeds: List[int],
    episodes: int,
    agent_r: PPOAgentR,
    bc_r_ckpt: str,
    device: str,
    edge_cost_max: float,
) -> Dict[str, Any]:
    """Scan one parameter combination."""
    env_proto = make_env(
        topology="snap24_gnutella_reach",
        num_slots=num_slots,
        num_servers=4,
        seed=seeds[0],
        slot_bw_hz=1.25e9,
        guard_band_fs=1,
        modulation_profile="default",
    )

    # Pre-generate episodes
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
                num_splits=5,
                split_profile="complex5_v2",
            )
            eps.append(requests)
        all_episodes[seed] = eps

    methods = ["greedy", "df", "rf", "iwd"]
    combo_results = {}
    for method in methods:
        seed_results = []
        for seed in seeds:
            res = evaluate_baseline(
                method, agent_r, all_episodes[seed],
                env_proto, num_slots,
            )
            seed_results.append(res)

        # Aggregate across seeds
        blocking_rates = [r["blocking_rate"] for r in seed_results]
        avg_delays = [r["avg_delay_ms"] for r in seed_results]
        obj_scores = [r["objective_score"] for r in seed_results]
        avg_fss = [r["avg_fs"] for r in seed_results]
        no_valids = [r["no_valid_c"] / r["total"] if r["total"] > 0 else 0.0 for r in seed_results]
        valid_means = [r["valid_action_mean"] for r in seed_results]

        # Aggregate split distribution
        all_splits = set()
        for r in seed_results:
            all_splits.update(r["split_dist"].keys())
        split_means = {}
        for s in sorted(all_splits):
            split_means[s] = float(np.mean([r["split_dist"].get(s, 0.0) for r in seed_results]))

        # Aggregate reasons
        all_reasons = Counter()
        for r in seed_results:
            for reason, count in r["reason_dist"].items():
                all_reasons[reason] += count

        combo_results[method] = {
            "blocking_mean": float(np.mean(blocking_rates)),
            "blocking_std": float(np.std(blocking_rates)),
            "avg_delay_mean": float(np.mean(avg_delays)),
            "avg_delay_std": float(np.std(avg_delays)),
            "objective_score_mean": float(np.mean(obj_scores)),
            "objective_score_std": float(np.std(obj_scores)),
            "avg_fs_mean": float(np.mean(avg_fss)),
            "avg_fs_std": float(np.std(avg_fss)),
            "no_valid_c_rate_mean": float(np.mean(no_valids)),
            "no_valid_c_rate_std": float(np.std(no_valids)),
            "valid_action_mean": float(np.mean(valid_means)),
            "split_dist": split_means,
            "reason_dist": dict(all_reasons),
        }

    return combo_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--edge_cost_max_values", type=str, default="30,40,50,60")
    parser.add_argument("--num_slots_values", type=str, default="24,20,16,12")
    parser.add_argument("--requests_values", type=str, default="60,90,120,150")
    parser.add_argument("--arrival_values", type=str, default="0.25,0.20")
    parser.add_argument("--seeds", type=str, default="42,123")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--bc_r_ckpt", type=str,
                        default="sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    parser.add_argument("--output_json", type=str,
                        default="sa_hmarl/experiments/complex5_v2_pressure_scan.json")
    parser.add_argument("--output_md", type=str,
                        default="sa_hmarl/experiments/complex5_v2_pressure_scan.md")
    args = parser.parse_args()

    # Grid
    num_slots_list = [int(x.strip()) for x in args.num_slots_values.split(",") if x.strip()]
    requests_per_episode_list = [int(x.strip()) for x in args.requests_values.split(",") if x.strip()]
    arrival_interval_list = [float(x.strip()) for x in args.arrival_values.split(",") if x.strip()]
    edge_cost_max_list = [float(x.strip()) for x in args.edge_cost_max_values.split(",") if x.strip()]
    seeds = [int(x.strip()) for x in args.seeds.split(",") if x.strip()]
    episodes = args.episodes

    print("=" * 80)
    print("Complex5_v2 Pressure Scan")
    print(f"Seeds: {seeds}, Episodes/seed: {episodes}")
    print(
        f"Grid: slots={num_slots_list}, reqs={requests_per_episode_list}, "
        f"interval={arrival_interval_list}, edge_cost_max={edge_cost_max_list}"
    )
    print("=" * 80)

    mod_reg = ModulationRegistry()
    agent_r = _load_ppo_r(args.bc_r_ckpt, mod_reg, device=args.device)

    all_results = []
    total_combos = (
        len(num_slots_list)
        * len(requests_per_episode_list)
        * len(arrival_interval_list)
        * len(edge_cost_max_list)
    )
    combo_idx = 0

    for edge_cost_max in edge_cost_max_list:
        for num_slots in num_slots_list:
            for requests_per_episode in requests_per_episode_list:
                for arrival_interval in arrival_interval_list:
                    combo_idx += 1
                    label = (
                        f"edge={edge_cost_max:.0f} slots={num_slots:2d} "
                        f"reqs={requests_per_episode:3d} interval={arrival_interval:.2f}"
                    )
                    print(f"\n[{combo_idx}/{total_combos}] {label}")

                    combo_res = scan_combo(
                        num_slots, requests_per_episode, arrival_interval,
                        seeds, episodes, agent_r, args.bc_r_ckpt, args.device,
                        edge_cost_max=edge_cost_max,
                    )

                    # Print summary
                    for method in ["greedy", "df", "rf", "iwd"]:
                        r = combo_res[method]
                        blk = r["blocking_mean"]
                        delay = r["avg_delay_mean"]
                        obj = r["objective_score_mean"]
                        no_valid = r["no_valid_c_rate_mean"]
                        valid_mean = r["valid_action_mean"]
                        splits = ", ".join([f"{k}={v:.1%}" for k, v in sorted(r["split_dist"].items())])
                        reasons = ", ".join([f"{k}={v}" for k, v in r["reason_dist"].items()])
                        print(
                            f"  {method.upper():6s}: blk={blk:.3f} delay={delay:.1f}ms "
                            f"obj={obj:.4f} no_valid={no_valid:.1%} valid={valid_mean:.1f} | {splits}"
                        )
                        if reasons:
                            print(f"          reasons: {reasons}")

                    all_results.append({
                        "edge_cost_max": edge_cost_max,
                        "num_slots": num_slots,
                        "requests_per_episode": requests_per_episode,
                        "arrival_interval": arrival_interval,
                        "methods": combo_res,
                    })

    # Save JSON
    with open(args.output_json, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nSaved JSON: {args.output_json}")

    # Write Markdown
    _write_markdown(args.output_md, all_results)
    print(f"Saved Markdown: {args.output_md}")

    # Recommend best combo
    _print_recommendations(all_results)


def _write_markdown(path: str, results: List[Dict]):
    lines = [
        "# Complex5_v2 Pressure Scan Results",
        "",
        "**Profile:** complex5_v2 | **Splits:** 5 | **Servers:** 4",
        f"**Seeds:** 42, 123 | **Episodes/seed:** 5",
        "",
        "## Grid",
        "",
        "| Slots | Reqs/Ep | Interval | Greedy Blk | DF Blk | RF Blk | IWD Blk | Best Obj |",
        "|-------|---------|----------|------------|--------|--------|---------|----------|",
    ]
    for combo in results:
        slots = combo["num_slots"]
        reqs = combo["requests_per_episode"]
        interval = combo["arrival_interval"]
        methods = combo["methods"]
        gre_blk = methods["greedy"]["blocking_mean"]
        df_blk = methods["df"]["blocking_mean"]
        rf_blk = methods["rf"]["blocking_mean"]
        iwd_blk = methods["iwd"]["blocking_mean"]
        best_obj = min(m["objective_score_mean"] for m in methods.values())
        lines.append(
            f"| {slots} | {reqs} | {interval} | {gre_blk:.3f} | {df_blk:.3f} | {rf_blk:.3f} | {iwd_blk:.3f} | {best_obj:.4f} |"
        )

    lines.extend(["", "## Detailed Results", ""])
    for combo in results:
        slots = combo["num_slots"]
        reqs = combo["requests_per_episode"]
        interval = combo["arrival_interval"]
        lines.append(f"### slots={slots}, reqs={reqs}, interval={interval}")
        for method in ["greedy", "df", "rf", "iwd"]:
            r = combo["methods"][method]
            lines.append(
                f"- **{method.upper()}**: blk={r['blocking_mean']:.3f}±{r['blocking_std']:.3f}, "
                f"delay={r['avg_delay_mean']:.1f}±{r['avg_delay_std']:.1f}ms, "
                f"obj={r['objective_score_mean']:.4f}±{r['objective_score_std']:.4f}, "
                f"fs={r['avg_fs_mean']:.1f}±{r['avg_fs_std']:.1f}"
            )
            splits = ", ".join([f"{k}={v:.1%}" for k, v in sorted(r["split_dist"].items())])
            lines.append(f"  - Splits: {splits}")
            if r["reason_dist"]:
                reasons = ", ".join([f"{k}={v}" for k, v in r["reason_dist"].items()])
                lines.append(f"  - Reasons: {reasons}")
        lines.append("")

    Path(path).write_text("\n".join(lines))


def _print_recommendations(results: List[Dict]):
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    scored = []
    for combo in results:
        slots = combo["num_slots"]
        reqs = combo["requests_per_episode"]
        interval = combo["arrival_interval"]
        methods = combo["methods"]

        best_blk = min(m["blocking_mean"] for m in methods.values())
        worst_blk = max(m["blocking_mean"] for m in methods.values())
        avg_blk = np.mean([m["blocking_mean"] for m in methods.values()])

        # Criteria scoring
        score = 0.0
        reasons = []

        # 1. Best baseline not 0%
        if best_blk > 0.0:
            score += 2.0
        else:
            reasons.append("best_baseline_blocking=0")

        # 2. Best baseline in 10%-25%
        if 0.10 <= best_blk <= 0.25:
            score += 3.0
        elif 0.05 <= best_blk < 0.10:
            score += 1.5
        elif 0.25 < best_blk <= 0.35:
            score += 1.0
        else:
            reasons.append(f"best_baseline_blocking={best_blk:.3f} out of sweet spot")

        # 3. Not all >50%
        if worst_blk < 0.50:
            score += 1.0
        else:
            reasons.append("some_baseline_blocking>0.50")

        # 4. Diverse failure reasons
        all_reasons = set()
        for m in methods.values():
            all_reasons.update(m["reason_dist"].keys())
        if "no_suitable_block" in all_reasons and "server_overload" in all_reasons:
            score += 2.0
        elif len(all_reasons) >= 2:
            score += 1.0
        else:
            reasons.append("failure_reasons_not_diverse")

        # 5. Split distribution not collapsed
        split_scores = []
        for m in methods.values():
            dist = m["split_dist"]
            if dist:
                max_split = max(dist.values())
                split_scores.append(max_split)
        if split_scores and max(split_scores) < 0.90:
            score += 1.0
        else:
            reasons.append("split_distribution_collapsed")

        scored.append({
            "combo": combo,
            "score": score,
            "reasons": reasons,
            "best_blk": best_blk,
            "worst_blk": worst_blk,
            "avg_blk": avg_blk,
        })

    scored.sort(key=lambda x: x["score"], reverse=True)

    print(f"{'Rank':>4} {'Slots':>5} {'Reqs':>5} {'Interval':>8} {'Score':>5} {'BestBlk':>8} {'WorstBlk':>8} {'Notes'}")
    for i, item in enumerate(scored[:5], 1):
        c = item["combo"]
        notes = "OK" if not item["reasons"] else "; ".join(item["reasons"])
        print(f"{i:4d} {c['num_slots']:5d} {c['requests_per_episode']:5d} {c['arrival_interval']:8.2f} "
              f"{item['score']:5.1f} {item['best_blk']:8.3f} {item['worst_blk']:8.3f} {notes}")

    best = scored[0]
    print(f"\n>>> RECOMMENDED: slots={best['combo']['num_slots']}, "
          f"reqs={best['combo']['requests_per_episode']}, "
          f"interval={best['combo']['arrival_interval']}")
    if best["reasons"]:
        print(f"    Caveats: {'; '.join(best['reasons'])}")


if __name__ == "__main__":
    main()
