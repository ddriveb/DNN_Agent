"""Oracle-R Evaluation: Future-Aware Action Selection Upper Bound.

Evaluates the Oracle-R selector (future-aware exhaustive simulation over all
candidate R actions) against baseline R policies under a fixed Agent-C.

The Oracle-R simulates every valid (path, modulation, block) candidate,
computes future feasibility degradation, and picks the action that minimises
future damage while succeeding immediately.  This establishes an upper bound
for what a future-aware R policy could achieve.

Usage:
    PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_oracle_r
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import argparse
import json
import time
from collections import Counter
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.agents.r_future_aware_selector import oracle_r_select
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import compute_reward, generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


# ---------------------------------------------------------------------------
# Model loaders
# ---------------------------------------------------------------------------

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
# Helpers
# ---------------------------------------------------------------------------

def _select_c_agent(agent_c: PPOAgentC, obs_c: Dict[str, Any]) -> Optional[int]:
    features, mask = agent_c.build_action_features(obs_c)
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        from sa_hmarl.env.c_action_risk import (
            apply_agent_c_risk_mask,
            risk_kwargs_from_args,
        )
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        mask = apply_agent_c_risk_mask(
            obs_c, mask,
            num_slots_total=int(ckpt_args.get("num_slots", 24)),
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, mask, deterministic=True)
    return action


def _fmt_pct(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v / total:.1%}" for k, v in counter.most_common())


def _fmt_cnt(counter: Counter) -> str:
    total = sum(counter.values())
    if total == 0:
        return "none"
    return ", ".join(f"{k}={v} ({v / total:.1%})" for k, v in counter.most_common())


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def evaluate_methods(
    agent_c: PPOAgentC,
    methods: List[Dict[str, Any]],
    episodes_data: List[List],
    env_prototype,
    args,
) -> Dict[str, Dict]:
    """Evaluate multiple R methods under the same fixed C."""
    num_methods = len(methods)
    num_servers = len(env_prototype.mec.servers)
    topology = env_prototype.net.topology
    num_slots = env_prototype.net.num_slots
    slot_bw_hz = getattr(env_prototype.fs_calc, "slot_bw_hz", 1.25e9)
    guard_band_fs = getattr(env_prototype.fs_calc, "guard_band_fs", 1)
    server_nodes = [s.node_id for s in env_prototype.mec.servers]
    capacities = [s.compute_capacity for s in env_prototype.mec.servers]

    # Per-method accumulators
    totals = [0] * num_methods
    blockeds = [0] * num_methods
    successes = [0] * num_methods
    total_delays = [0.0] * num_methods
    total_fs = [0.0] * num_methods
    total_waste = [0.0] * num_methods
    total_path = [0.0] * num_methods
    total_oracle_time = [0.0] * num_methods
    mod_counters = [Counter() for _ in range(num_methods)]
    split_counters = [Counter() for _ in range(num_methods)]
    reason_counters = [Counter() for _ in range(num_methods)]

    for ep_idx, requests in enumerate(episodes_data):
        print(f"  Episode {ep_idx + 1}/{len(episodes_data)}: {len(requests)} requests")

        # Independent env copies
        envs = []
        for _ in range(num_methods):
            env = make_env(
                topology=topology, num_slots=num_slots, num_servers=num_servers,
                seed=42, slot_bw_hz=slot_bw_hz, guard_band_fs=guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=getattr(args, "max_blocks", 10),
                block_sort_strategy=getattr(args, "block_sort_strategy", "mixed"),
                server_nodes=server_nodes, capacities=capacities,
            )
            env.reset(requests)
            envs.append(env)

        ref_env = envs[0]

        for req_idx, req in enumerate(requests):
            # Agent-C decision (once, on reference env)
            obs_c = build_agent_c_observation(ref_env, req)
            action_idx_c = _select_c_agent(agent_c, obs_c)
            action_c = (0, 0) if action_idx_c is None else decode_agent_c_action(
                action_idx_c, num_servers
            )
            split_id, server_id = action_c

            # Each R method on its own env copy
            for i, (env, method) in enumerate(zip(envs, methods)):
                split_counters[i][f"split{split_id}"] += 1
                mtype = method["type"]

                if mtype == "oracle_r":
                    t0 = time.time()
                    oracle_action, oracle_info, obs_r_before, decoded_action_r = oracle_r_select(
                        env, req, action_c, agent_c,
                        future_feas_coef=method.get("future_feas_coef", 1.0),
                        future_feas_bonus_coef=method.get("future_feas_bonus_coef", 0.1),
                        future_feas_horizon=method.get("future_feas_horizon", 5),
                        future_feas_norm=method.get("future_feas_norm", 60.0),
                        waste_coef=method.get("waste_coef", args.waste_coef),
                        c_policy="agent",
                        verbose=(req_idx == 0 and ep_idx == 0),
                    )
                    total_oracle_time[i] += time.time() - t0

                    # Use decoded action from oracle (correct indexing into obs_r_before)
                    if decoded_action_r is not None:
                        action_r = decoded_action_r
                    elif oracle_action is not None and obs_r_before is not None:
                        action_r = decode_agent_r_action(
                            oracle_action, len(obs_r_before["mod_names"]),
                            getattr(env, "max_blocks", 10),
                        )
                    else:
                        action_r = (0, 0, 0)

                elif mtype == "ppo_r":
                    obs_r = build_agent_r_observation(env, req, split_id, server_id)
                    action_idx_r = method["agent"].select_action(obs_r, deterministic=True)
                    if action_idx_r is None:
                        action_r = (0, 0, 0)
                    else:
                        action_r = decode_agent_r_action(
                            action_idx_r, len(obs_r["mod_names"]),
                            getattr(env, "max_blocks", 10),
                        )

                else:
                    raise ValueError(f"Unknown method type: {mtype}")

                _, _, _, info = env.step(action_c, action_r)
                totals[i] += 1

                if info.get("success", False):
                    successes[i] += 1
                    total_delays[i] += float(info.get("delay_ms", 0.0))
                    total_fs[i] += float(info.get("num_slots", 0))
                    total_waste[i] += float(info.get("block_waste", 0.0))
                    total_path[i] += float(info.get("path_dist_km", 0.0))
                    mod_counters[i][info.get("modulation", "unknown")] += 1
                else:
                    blockeds[i] += 1
                    reason_counters[i][info.get("reason", "unknown")] += 1

    # Assemble results
    results = {}
    for i, method in enumerate(methods):
        name = method["name"]
        n = max(totals[i], 1)
        s = max(successes[i], 1)
        results[name] = {
            "total": totals[i],
            "blocked": blockeds[i],
            "success": successes[i],
            "blocking_rate": blockeds[i] / n,
            "success_rate": successes[i] / n,
            "avg_delay_ms": total_delays[i] / s,
            "avg_fs": total_fs[i] / s,
            "avg_waste": total_waste[i] / s,
            "avg_path_len_km": total_path[i] / s,
            "no_suitable_block_ratio": (
                reason_counters[i].get("no_suitable_block", 0) / max(blockeds[i], 1)
            ),
            "server_overload_ratio": (
                reason_counters[i].get("server_overload", 0) / max(blockeds[i], 1)
            ),
            "mod_counter": dict(mod_counters[i]),
            "split_counter": dict(split_counters[i]),
            "reason_counter": dict(reason_counters[i]),
            "total_oracle_time_s": total_oracle_time[i],
        }
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Oracle-R Future-Aware Action Selection Evaluation"
    )
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--seeds", type=str, default="42,123,456,789,2024")
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--requests_per_episode", type=int, default=60)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--waste_coef", type=float, default=0.8)
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--num_slots", type=int, default=16)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--num_splits", type=int, default=5)
    parser.add_argument("--split_profile", type=str, default="complex5_v2_lite")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--modulation_profile", type=str, default="default",
                        choices=["default", "extended"])
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="checkpoints/agent_c_meanpath_cgap_msval_best.pt")
    parser.add_argument("--ppo_r_checkpoint", type=str,
                        default="checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt")
    # Oracle-R parameters
    parser.add_argument("--future_feas_coef", type=float, default=1.0,
                        help="Penalty for future feasibility drop")
    parser.add_argument("--future_feas_bonus_coef", type=float, default=0.1,
                        help="Bonus for future feasibility gain")
    parser.add_argument("--future_feas_horizon", type=int, default=5,
                        help="Number of future requests to consider")
    parser.add_argument("--future_feas_norm", type=float, default=60.0,
                        help="Normalisation for feasibility counts")
    # Sweep mode: evaluate multiple oracle coef values
    parser.add_argument("--sweep_coefs", type=str, default=None,
                        help="Comma-separated future_feas_coef values to sweep, e.g. '0.5,1.0,2.0'")
    parser.add_argument("--output_json", type=str,
                        default="experiments/oracle_r_eval/oracle_r_eval.json")
    parser.add_argument("--output_md", type=str,
                        default="experiments/oracle_r_eval/oracle_r_eval.md")
    args = parser.parse_args()

    print("=" * 90)
    print("Oracle-R Future-Aware Action Selection Evaluation")
    print(f"Topology: {args.topology}  Slots: {args.num_slots}  Servers: {args.num_servers}")
    print(f"Agent-C: {args.agent_c_checkpoint}")
    print(f"Baseline PPO-R: {args.ppo_r_checkpoint}")
    print(f"Future Feas: coef={args.future_feas_coef} bonus={args.future_feas_bonus_coef} "
          f"horizon={args.future_feas_horizon} norm={args.future_feas_norm}")
    print("=" * 90)

    # Build methods list
    methods: List[Dict[str, Any]] = []

    # 1. Baseline PPO-R
    print("Loading baseline PPO-R...")
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    ppo_r = _load_ppo_r(args.ppo_r_checkpoint, mod_reg, args.device)
    methods.append({"name": "PPO-R (BC baseline)", "type": "ppo_r", "agent": ppo_r})

    # 2. Oracle-R (main config)
    methods.append({
        "name": f"Oracle-R (coef={args.future_feas_coef})",
        "type": "oracle_r",
        "agent": None,
        "future_feas_coef": args.future_feas_coef,
        "future_feas_bonus_coef": args.future_feas_bonus_coef,
        "future_feas_horizon": args.future_feas_horizon,
        "future_feas_norm": args.future_feas_norm,
        "waste_coef": args.waste_coef,
    })

    # 3-5. Sweep additional coef values if requested
    if args.sweep_coefs:
        for coef_str in args.sweep_coefs.split(","):
            coef = float(coef_str.strip())
            if abs(coef - args.future_feas_coef) < 1e-6:
                continue  # already added
            methods.append({
                "name": f"Oracle-R (coef={coef})",
                "type": "oracle_r",
                "agent": None,
                "future_feas_coef": coef,
                "future_feas_bonus_coef": args.future_feas_bonus_coef,
                "future_feas_horizon": args.future_feas_horizon,
                "future_feas_norm": args.future_feas_norm,
                "waste_coef": args.waste_coef,
            })

    # Load fixed Agent-C
    print(f"Loading fixed Agent-C from {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)

    # Prototype env
    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks,
        block_sort_strategy=args.block_sort_strategy,
    )

    # Generate episodes
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    all_episodes: Dict[int, List[List]] = {}
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

    # Evaluate per seed
    seed_results: Dict[int, Dict[str, Dict]] = {}
    for seed in seeds:
        print(f"\n--- Seed {seed} ---")
        res = evaluate_methods(agent_c, methods, all_episodes[seed], env_proto, args)
        seed_results[seed] = res
        for name, metrics in res.items():
            print(f"  {name:35s} blk={metrics['blocking_rate']:.4f} "
                  f"delay={metrics['avg_delay_ms']:.1f}ms fs={metrics['avg_fs']:.2f}")

    # Aggregate across seeds
    method_names = [m["name"] for m in methods]
    agg_results: Dict[str, Dict] = {}
    for name in method_names:
        scalar_keys = [
            "blocking_rate", "success_rate", "avg_delay_ms", "avg_fs",
            "avg_waste", "avg_path_len_km",
            "no_suitable_block_ratio", "server_overload_ratio",
        ]
        agg = {}
        for k in scalar_keys:
            vals = [seed_results[s][name][k] for s in seeds]
            agg[k] = float(np.mean(vals))
            agg[f"{k}_std"] = float(np.std(vals))

        for counter_key in ["mod_counter", "split_counter", "reason_counter"]:
            c = Counter()
            for s in seeds:
                c.update(seed_results[s][name][counter_key])
            agg[counter_key] = dict(c)

        # Sum oracle time
        agg["total_oracle_time_s"] = float(sum(
            seed_results[s][name].get("total_oracle_time_s", 0.0) for s in seeds
        ))
        agg_results[name] = agg

    # ------------------------------------------------------------------
    # Print report
    # ------------------------------------------------------------------
    report_lines: List[str] = []
    def _print(line: str = ""):
        print(line)
        report_lines.append(line)

    _print()
    _print("=" * 90)
    _print("AGGREGATED RESULTS (mean across seeds)")
    _print("=" * 90)

    header = (
        f"{'Method':35s} {'Blocking':>9s} {'Delay':>8s} {'AvgFS':>7s} "
        f"{'Waste':>7s} {'NoBlock':>8s} {'SrvOver':>8s}"
    )
    _print(header)
    _print("-" * 90)

    for name, agg in agg_results.items():
        _print(
            f"{name:35s} "
            f"{agg['blocking_rate']:9.4f} "
            f"{agg['avg_delay_ms']:8.1f} "
            f"{agg['avg_fs']:7.2f} "
            f"{agg['avg_waste']:7.3f} "
            f"{agg['no_suitable_block_ratio']:8.3f} "
            f"{agg['server_overload_ratio']:8.3f}"
        )

    _print()
    _print("-" * 90)
    _print("FAILURE REASONS")
    _print("-" * 90)
    for name, agg in agg_results.items():
        _print(f"  {name:35s}  {_fmt_cnt(Counter(agg['reason_counter']))}")

    _print()
    _print("-" * 90)
    _print("SPLIT DISTRIBUTION")
    _print("-" * 90)
    for name, agg in agg_results.items():
        _print(f"  {name:35s}  {_fmt_pct(Counter(agg['split_counter']))}")

    _print()
    _print("-" * 90)
    _print("TIMING")
    _print("-" * 90)
    for name, agg in agg_results.items():
        t = agg.get("total_oracle_time_s", 0.0)
        if t > 0:
            _print(f"  {name:35s}  total={t:.1f}s")

    # ------------------------------------------------------------------
    # JSON export
    # ------------------------------------------------------------------
    if args.output_json:
        out_path = Path(args.output_json)
        clean = {}
        for name, agg in agg_results.items():
            clean[name] = {
                k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
                for k, v in agg.items()
            }
        out_path.parent.mkdir(parents=True, exist_ok=True)
        json.dump(clean, open(str(out_path), "w"), indent=2)
        _print(f"\nResults written to {out_path}")

    # ------------------------------------------------------------------
    # Markdown export
    # ------------------------------------------------------------------
    if args.output_md:
        md_path = Path(args.output_md)
        md_lines = [
            "# Oracle-R Future-Aware Action Selection Evaluation\n\n",
            f"**Topology:** {args.topology}  \n",
            f"**Servers:** {args.num_servers}  **Slots:** {args.num_slots}  \n",
            f"**Fixed Agent-C:** `{args.agent_c_checkpoint}`  \n",
            f"**Baseline PPO-R:** `{args.ppo_r_checkpoint}`  \n",
            f"**Seeds:** {args.seeds}  \n",
            f"**Episodes/seed:** {args.episodes}  \n",
            f"**Requests/episode:** {args.requests_per_episode}  \n",
            f"**Future Feas Horizon:** {args.future_feas_horizon}  \n",
            f"**Future Feas Norm:** {args.future_feas_norm}  \n\n",
        ]

        md_lines.append("## Results\n\n")
        md_lines.append(
            "| Method | Blocking | Delay(ms) | AvgFS | Waste | NoBlock% | SrvOver% |\n"
        )
        md_lines.append(
            "|--------|----------|-----------|-------|-------|----------|----------|\n"
        )
        for name, agg in agg_results.items():
            md_lines.append(
                f"| {name} | {agg['blocking_rate']:.4f} | {agg['avg_delay_ms']:.1f} | "
                f"{agg['avg_fs']:.2f} | {agg['avg_waste']:.3f} | "
                f"{agg['no_suitable_block_ratio']:.1%} | "
                f"{agg['server_overload_ratio']:.1%} |\n"
            )

        md_lines.append("\n## Failure Reasons\n\n")
        for name, agg in agg_results.items():
            md_lines.append(
                f"- **{name}**: {_fmt_cnt(Counter(agg['reason_counter']))}\n"
            )

        md_lines.append("\n## Split Distribution\n\n")
        for name, agg in agg_results.items():
            md_lines.append(
                f"- **{name}**: {_fmt_pct(Counter(agg['split_counter']))}\n"
            )

        md_path.parent.mkdir(parents=True, exist_ok=True)
        md_path.write_text("".join(md_lines))
        _print(f"Markdown report written to {md_path}")


if __name__ == "__main__":
    main()
