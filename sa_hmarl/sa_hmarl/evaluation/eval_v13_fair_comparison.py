"""Fair closed-loop comparison of v1.3 ranker variants on COST239.

Modes:
- ppo_r: frozen PPO-R baseline.
- ksp_ff_plain: KSP-FF K=50 (first-fit over sorted paths).
- ksp_ff_highest: KSP-FF K=50 with highest feasible modulation.
- ranker_<ckpt>_<candidate_mode>: distilled ranker using the policy wrapper.

Outputs per-seed JSON and an aggregated Markdown table.
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action, ksp_ff_highest_mod_action
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _load_ppo_c,
    _load_ppo_r,
    _select_c_action_from_obs,
    _select_r_action_from_obs,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset import (
    _load_ranker_policy_from_checkpoint,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


@dataclass
class MethodResult:
    seed: int
    mode: str
    total: int = 0
    blocked: int = 0
    server_overload: int = 0
    no_suitable_block: int = 0
    other_failure: int = 0
    delay_sum: float = 0.0
    fs_sum: float = 0.0
    admitted: int = 0
    same_as_ppo: int = 0
    decision_ms_sum: float = 0.0

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.total, 1))

    def overload_rate(self) -> float:
        return float(self.server_overload / max(self.total, 1))

    def nsb_rate(self) -> float:
        return float(self.no_suitable_block / max(self.total, 1))

    def avg_delay_ms(self) -> float:
        return float(self.delay_sum / max(self.admitted, 1))

    def avg_fs(self) -> float:
        return float(self.fs_sum / max(self.admitted, 1))

    def avg_decision_ms(self) -> float:
        return float(self.decision_ms_sum / max(self.total, 1))

    def ppo_agreement(self) -> float:
        return float(self.same_as_ppo / max(self.total, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "mode": self.mode,
            "total": self.total,
            "blocked": self.blocked,
            "blocking_rate": self.blocking_rate(),
            "server_overload": self.server_overload,
            "overload_rate": self.overload_rate(),
            "no_suitable_block": self.no_suitable_block,
            "nsb_rate": self.nsb_rate(),
            "other_failure": self.other_failure,
            "avg_delay_ms": self.avg_delay_ms(),
            "avg_fs": self.avg_fs(),
            "avg_decision_ms": self.avg_decision_ms(),
            "ppo_agreement": self.ppo_agreement(),
        }


def _run_episode(
    env,
    requests,
    agent_c,
    agent_r,
    mode: str,
    ranker_policy,
    args,
    warmup_requests: int = 0,
) -> MethodResult:
    result = MethodResult(seed=args._current_seed, mode=mode)
    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)
        # C-side uses the training-time path support (usually K=5).
        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)
        # R-side uses the wider path support under test (usually K=50).
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)

        legal = np.flatnonzero(np.asarray(obs_r["agent_r_mask"], dtype=bool)).tolist()
        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        if mode == "ppo_r":
            r_idx = int(ppo_idx)
        elif mode == "ksp_ff_plain":
            a = ksp_ff_action(obs_r)
            r_idx = int(a) if a is not None else int(ppo_idx)
        elif mode == "ksp_ff_highest":
            a = ksp_ff_highest_mod_action(obs_r)
            r_idx = int(a) if a is not None else int(ppo_idx)
        elif mode.startswith("ranker_"):
            a = ranker_policy.select_action(env, req, obs_c, obs_r, agent_r, split_id, server_id)
            r_idx = int(a) if a is not None else int(ppo_idx)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        r_action = decode_agent_r_action(r_idx, len(obs_r["mod_names"]), env.max_blocks)
        _, _, _, info = env.step((split_id, server_id), r_action)

        if not is_warmup:
            result.decision_ms_sum += (time.perf_counter() - t0) * 1000.0
            result.total += 1
            if info.get("success", False):
                result.admitted += 1
                result.delay_sum += float(info.get("delay_ms", 0.0))
                result.fs_sum += float(info.get("num_slots", 0))
            else:
                result.blocked += 1
                reason = info.get("reason", "")
                if reason == "server_overload":
                    result.server_overload += 1
                elif reason == "no_suitable_block":
                    result.no_suitable_block += 1
                else:
                    result.other_failure += 1
            if int(r_idx) == int(ppo_idx):
                result.same_as_ppo += 1
    return result


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    ranker_policies: Dict[str, Any] = {}
    for spec in args.ranker_specs:
        # spec format: name=path[:candidate_mode]
        if "=" not in spec:
            raise ValueError(f"Ranker spec must be name=path[:candidate_mode], got {spec}")
        name, rest = spec.split("=", 1)
        parts = rest.split(":")
        path = parts[0]
        cand_mode = parts[1] if len(parts) > 1 else None
        policy = _load_ranker_policy_from_checkpoint(path, args.device)
        if cand_mode is not None:
            policy.candidate_mode = cand_mode
        ranker_policies[f"ranker_{name}"] = policy
        print(f"Loaded ranker '{name}' from {path} (candidate_mode={policy.candidate_mode}, feature_names={len(policy.feature_names)})", flush=True)

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results: List[MethodResult] = []

    for seed in seeds:
        args._current_seed = seed
        env = make_env(
            args.topology, args.num_slots, args.num_servers, seed,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy_r,
            path_sort_strategy=args.path_sort_strategy_r, k=args.k_paths_r,
        )
        rng = np.random.RandomState(seed)
        src = int(rng.randint(0, env.net.NUM_NODES))
        total_requests = args.warmup_requests + args.requests_per_episode
        requests = generate_requests(
            env, rng, src, total_requests,
            args.arrival_interval, args.holding_min, args.holding_max,
            args.deadline_min, args.deadline_max, args.size_min_mb, args.size_max_mb,
            args.edge_cost_min, args.edge_cost_max, args.num_splits, args.split_profile,
        )
        for mode in modes:
            ranker_policy = ranker_policies.get(mode)
            env.reset(requests)
            result = _run_episode(env, requests, agent_c, agent_r, mode, ranker_policy, args, warmup_requests=args.warmup_requests)
            all_results.append(result)
            print(
                f"[seed={seed}][mode={mode}] blocking={result.blocking_rate():.4%} "
                f"overload={result.overload_rate():.4%} nsb={result.nsb_rate():.4%} "
                f"decision_ms={result.avg_decision_ms():.2f}",
                flush=True,
            )

    # Aggregate per mode.
    summary: Dict[str, Any] = {}
    for mode in modes:
        rows = [r.to_dict() for r in all_results if r.mode == mode]
        if not rows:
            continue
        summary[mode] = {
            "seeds": [r["seed"] for r in rows],
            "blocking_rate_mean": float(np.mean([r["blocking_rate"] for r in rows])),
            "blocking_rate_std": float(np.std([r["blocking_rate"] for r in rows])),
            "overload_rate_mean": float(np.mean([r["overload_rate"] for r in rows])),
            "nsb_rate_mean": float(np.mean([r["nsb_rate"] for r in rows])),
            "avg_delay_ms_mean": float(np.mean([r["avg_delay_ms"] for r in rows])),
            "avg_fs_mean": float(np.mean([r["avg_fs"] for r in rows])),
            "avg_decision_ms_mean": float(np.mean([r["avg_decision_ms"] for r in rows])),
            "ppo_agreement_mean": float(np.mean([r["ppo_agreement"] for r in rows])),
            "per_seed": rows,
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_json = out / args.output_json
    out_md = out / args.output_md
    out_json.write_text(json.dumps({
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "summary": summary,
        "results": [r.to_dict() for r in all_results],
    }, indent=2), encoding="utf-8")
    _write_markdown(out_md, summary, args)
    print(f"Saved {out_json} and {out_md}")
    return summary


def _write_markdown(path: Path, summary: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Fair Comparison: v1.3 Post-Decision Ranker vs Baselines",
        "",
        f"- Topology: `{args.topology}`, slots={args.num_slots}, servers={args.num_servers}, "
        f"C-K={args.k_paths_c}/{args.path_sort_strategy_c}, R-K={args.k_paths_r}/{args.path_sort_strategy_r}",
        f"- Seeds: {args.seeds}",
        f"- Total requests/seed: {args.warmup_requests + args.requests_per_episode} "
        f"(warmup={args.warmup_requests}, evaluated={args.requests_per_episode})",
        "",
        "| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, s in summary.items():
        lines.append(
            f"| {mode} | {s['blocking_rate_mean']:.2%} ± {s['blocking_rate_std']:.2%} | "
            f"{s['overload_rate_mean']:.2%} | {s['nsb_rate_mean']:.2%} | "
            f"{s['avg_delay_ms_mean']:.2f} | {s['avg_fs_mean']:.2f} | "
            f"{s['avg_decision_ms_mean']:.2f} | {s['ppo_agreement_mean']:.2%} |"
        )
    lines.append("")
    lines.append("## Per-seed details")
    lines.append("")
    lines.append("| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for mode, s in summary.items():
        for row in s["per_seed"]:
            lines.append(
                f"| {row['seed']} | {mode} | {row['blocking_rate']:.2%} | "
                f"{row['overload_rate']:.2%} | {row['nsb_rate']:.2%} | {row['avg_decision_ms']:.2f} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent_c_checkpoint", default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt")
    parser.add_argument("--agent_r_checkpoint", default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--ranker_specs", nargs="*", default=[],
                        help="List of name=path[:candidate_mode] ranker checkpoints to evaluate.")
    parser.add_argument("--modes", default="ppo_r,ksp_ff_plain,ksp_ff_highest",
                        help="Comma-separated list of modes to run. Ranker modes are added automatically from --ranker_specs unless explicitly listed.")
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=50, help="Legacy alias for --k_paths_r.")
    parser.add_argument("--k_paths_c", type=int, default=5, help="Path support for the frozen C-side policy.")
    parser.add_argument("--k_paths_r", type=int, default=50, help="Path support for R-side selectors.")
    parser.add_argument("--path_sort_strategy", default="hops", choices=["km", "hops"], help="Legacy alias for R-side path sort.")
    parser.add_argument("--path_sort_strategy_c", default="hops", choices=["km", "hops"])
    parser.add_argument("--path_sort_strategy_r", default="hops", choices=["km", "hops"])
    parser.add_argument("--block_sort_strategy", default="start_asc", help="Legacy alias for R-side block sort.")
    parser.add_argument("--block_sort_strategy_c", default="start_asc")
    parser.add_argument("--block_sort_strategy_r", default="start_asc")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", default="3030,4040,5050,6060,7070")
    parser.add_argument("--requests_per_episode", type=int, default=10000)
    parser.add_argument("--warmup_requests", type=int, default=2000)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--output_dir", default="sa_hmarl/experiments")
    parser.add_argument("--output_json", default="v13_fair_comparison.json")
    parser.add_argument("--output_md", default="v13_fair_comparison.md")
    parser.add_argument("--device", default="cpu")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    # Auto-append ranker modes if not present.
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for name in args.ranker_specs:
        mode_name = f"ranker_{name.split('=', 1)[0]}"
        if mode_name not in modes:
            modes.append(mode_name)
    args.modes = ",".join(modes)
    evaluate(args)
