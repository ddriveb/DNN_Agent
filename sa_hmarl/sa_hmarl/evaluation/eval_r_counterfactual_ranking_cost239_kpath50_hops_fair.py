"""Fair closed-loop comparison for SA-HMARL v1.3 strict counterfactual reranking.

All ranker-based methods use the *same* candidate support: the top-K_prop legal
flat actions proposed by the frozen PPO-R under K_path=50.  This removes any
advantage from a larger or mixed candidate pool and isolates the ranking
supervision signal.

Methods:
- ppo_r_top1: frozen PPO-R greedy action.
- ksp_ff_plain: first-fit over the K_path=50 legal action list.
- ksp_ff_highest: KSP-FF with highest feasible modulation.
- ranker_old_v13: legacy planner-distilled ranker scored over PPO-R Top-30.
- ranker_new_v13: newly trained ranker scored over PPO-R Top-30 (optional).

The C-side always uses K_C=5; the R-side observation and step use K_path=50.
"""
from __future__ import annotations

import argparse
import copy
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch

from sa_hmarl.agents.counterfactual_r_ranker import build_counterfactual_r_ranker
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
from sa_hmarl.evaluation.diagnose_r_action_horizon_oracle import (
    _snapshot_before_r_decision,
)
from sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5 import (
    _depth_stratum,
    _ppo_r_topk_actions,
)
from sa_hmarl.evaluation.generate_r_post_decision_dataset import (
    FEATURE_NAMES,
    _r_feature_vector,
)
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env


@dataclass
class SubgroupCounters:
    """Per-request counters for the E=0 / E=1 subgroups."""

    total: int = 0
    blocked: int = 0
    server_overload: int = 0
    no_suitable_block: int = 0
    other_failure: int = 0
    delay_sum: float = 0.0
    fs_sum: float = 0.0
    admitted: int = 0
    same_as_ppo: int = 0
    ranker_changes: int = 0
    ranker_better: int = 0
    ranker_worse: int = 0
    ranker_equal: int = 0

    def blocking_rate(self) -> float:
        return float(self.blocked / max(self.total, 1))

    def overload_rate(self) -> float:
        return float(self.server_overload / max(self.total, 1))

    def nsb_rate(self) -> float:
        return float(self.no_suitable_block / max(self.total, 1))

    def other_rate(self) -> float:
        return float(self.other_failure / max(self.total, 1))

    def avg_delay_ms(self) -> float:
        return float(self.delay_sum / max(self.admitted, 1))

    def avg_fs(self) -> float:
        return float(self.fs_sum / max(self.admitted, 1))

    def ppo_agreement(self) -> float:
        return float(self.same_as_ppo / max(self.total, 1))

    def change_rate(self) -> float:
        return float(self.ranker_changes / max(self.total, 1))

    def better_rate(self) -> float:
        return float(self.ranker_better / max(self.ranker_changes, 1))

    def worse_rate(self) -> float:
        return float(self.ranker_worse / max(self.ranker_changes, 1))

    def to_dict(self) -> Dict[str, Any]:
        return {
            "total": self.total,
            "blocked": self.blocked,
            "blocking_rate": self.blocking_rate(),
            "server_overload": self.server_overload,
            "overload_rate": self.overload_rate(),
            "no_suitable_block": self.no_suitable_block,
            "nsb_rate": self.nsb_rate(),
            "other_failure": self.other_failure,
            "other_rate": self.other_rate(),
            "avg_delay_ms": self.avg_delay_ms(),
            "avg_fs": self.avg_fs(),
            "ppo_agreement": self.ppo_agreement(),
            "ranker_changes": self.ranker_changes,
            "change_rate": self.change_rate(),
            "ranker_better": self.ranker_better,
            "better_rate": self.better_rate(),
            "ranker_worse": self.ranker_worse,
            "worse_rate": self.worse_rate(),
            "ranker_equal": self.ranker_equal,
        }


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
    ranker_top1_in_candidates: int = 0
    e0: SubgroupCounters = field(default_factory=SubgroupCounters)
    e1: SubgroupCounters = field(default_factory=SubgroupCounters)

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

    def ranker_top1_in_candidates_rate(self) -> float:
        return float(self.ranker_top1_in_candidates / max(self.total, 1))

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
            "ranker_top1_in_candidates": self.ranker_top1_in_candidates,
            "ranker_top1_in_candidates_rate": self.ranker_top1_in_candidates_rate(),
            "e0": self.e0.to_dict(),
            "e1": self.e1.to_dict(),
        }


def _load_ranker_model(ckpt_path: str, device: str):
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    model = build_counterfactual_r_ranker(
        ckpt.get("model_type", "mlp"),
        int(ckpt["input_dim"]),
        tuple(ckpt.get("hidden_dims", [128, 64])),
        float(ckpt.get("dropout", 0.0)),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    feature_mean = np.asarray(ckpt.get("feature_mean", 0.0), dtype=np.float32)
    feature_std = np.maximum(
        np.asarray(ckpt.get("feature_std", 1.0), dtype=np.float32), 1e-6
    )
    if feature_mean.ndim == 0:
        feature_mean = np.zeros(int(ckpt["input_dim"]), dtype=np.float32)
    if feature_std.ndim == 0:
        feature_std = np.ones(int(ckpt["input_dim"]), dtype=np.float32)

    feature_names = list(ckpt.get("feature_names", FEATURE_NAMES))
    return {
        "model": model,
        "device": device,
        "feature_mean": feature_mean,
        "feature_std": feature_std,
        "feature_names": feature_names,
        "input_dim": int(ckpt["input_dim"]),
    }


def _ranker_select_action(
    ranker: Dict[str, Any],
    env,
    req,
    obs_c: Dict[str, Any],
    obs_r: Dict[str, Any],
    agent_r,
    split_id: int,
    server_id: int,
    candidate_actions: List[int],
) -> Tuple[int, bool]:
    """Score the provided candidates and return the highest-scoring action.

    Returns (action_id, was_in_candidate_list).  Falls back to the first
    candidate if scoring fails.
    """
    if not candidate_actions:
        return 0, False

    r_features, _ = agent_r.build_action_features(obs_r)
    feats = []
    for a in candidate_actions:
        feat = _r_feature_vector(
            env,
            req,
            obs_c,
            obs_r,
            r_features,
            int(a),
            split_id,
            server_id,
            feature_names=ranker["feature_names"],
        )
        feats.append(feat)
    x = np.stack(feats, axis=0).astype(np.float32)
    x = (x - ranker["feature_mean"]) / ranker["feature_std"]
    with torch.no_grad():
        scores = (
            ranker["model"](
                torch.as_tensor(x, dtype=torch.float32, device=ranker["device"])
            )
            .cpu()
            .numpy()
        )
    best_local = int(np.argmax(scores))
    return int(candidate_actions[best_local]), True


def _action_to_path_idx(action_id: int, num_mods: int, max_blocks: int) -> int:
    return action_id // (num_mods * max_blocks)


def _e_gate_from_candidates(
    candidate_actions: List[int], obs_r: Dict[str, Any], max_blocks: int
) -> Tuple[bool, int, int]:
    """Return (is_deep_path/E=1, max_path_idx, depth_stratum)."""
    if not candidate_actions:
        return False, -1, -1
    num_mods = len(obs_r["mod_names"])
    path_indices = [_action_to_path_idx(a, num_mods, max_blocks) for a in candidate_actions]
    max_path_idx = int(max(path_indices))
    return max_path_idx >= 5, max_path_idx, _depth_stratum(max_path_idx)


def _ppo_r_top1_outcome(
    env, req, split_id: int, server_id: int, ppo_idx: int, obs_r: Dict[str, Any]
) -> Dict[str, Any]:
    """Counterfactual immediate outcome of taking PPO-R top-1 from the current state."""
    branch = copy.deepcopy(env)
    r_action = decode_agent_r_action(ppo_idx, len(obs_r["mod_names"]), env.max_blocks)
    _, _, _, info = branch.step((split_id, server_id), r_action)
    return info


def _update_subgroup(
    sub: SubgroupCounters,
    info: Dict[str, Any],
    same_as_ppo: bool,
    ranker_changed: bool,
    ppo_info: Optional[Dict[str, Any]],
) -> None:
    """Update per-E-gate counters; ppo_info is required only when ranker_changed."""
    sub.total += 1
    if info.get("success", False):
        sub.admitted += 1
        sub.delay_sum += float(info.get("delay_ms", 0.0))
        sub.fs_sum += float(info.get("num_slots", 0))
    else:
        sub.blocked += 1
        reason = info.get("reason", "")
        if reason == "server_overload":
            sub.server_overload += 1
        elif reason == "no_suitable_block":
            sub.no_suitable_block += 1
        else:
            sub.other_failure += 1
    if same_as_ppo:
        sub.same_as_ppo += 1
    if ranker_changed and ppo_info is not None:
        sub.ranker_changes += 1
        ranker_success = info.get("success", False)
        ppo_success = ppo_info.get("success", False)
        if ranker_success and not ppo_success:
            sub.ranker_better += 1
        elif not ranker_success and ppo_success:
            sub.ranker_worse += 1
        else:
            sub.ranker_equal += 1


def _run_episode(
    env,
    requests,
    agent_c,
    agent_r,
    mode: str,
    ranker: Optional[Dict[str, Any]],
    args,
    warmup_requests: int = 0,
) -> MethodResult:
    result = MethodResult(seed=args._current_seed, mode=mode)
    is_ranker_mode = mode.startswith("ranker_")
    for step_idx, req in enumerate(requests):
        is_warmup = step_idx < warmup_requests
        t0 = time.perf_counter()
        env.advance_time(req.arrival_time)

        # C-side under K_C.
        env.k = args.k_paths_c
        env.path_sort_strategy = args.path_sort_strategy_c
        env.block_sort_strategy = args.block_sort_strategy_c
        obs_c = build_agent_c_observation(env, req)
        c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
        split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)

        # R-side under K_path.
        env.k = args.k_paths_r
        env.path_sort_strategy = args.path_sort_strategy_r
        env.block_sort_strategy = args.block_sort_strategy_r
        obs_r = build_agent_r_observation(env, req, split_id, server_id)

        ppo_idx, _ = _select_r_action_from_obs(agent_r, obs_r, env.max_blocks)

        same_as_ppo_flag = False
        ranker_in_candidates_flag = False
        e1 = False
        max_path_idx = -1
        depth_stratum = -1
        ppo_info: Optional[Dict[str, Any]] = None

        if mode == "ppo_r_top1":
            r_idx = int(ppo_idx)
        elif mode == "ksp_ff_plain":
            a = ksp_ff_action(obs_r)
            r_idx = int(a) if a is not None else int(ppo_idx)
        elif mode == "ksp_ff_highest":
            a = ksp_ff_highest_mod_action(obs_r)
            r_idx = int(a) if a is not None else int(ppo_idx)
        elif is_ranker_mode:
            # Strict PPO-R Top-K_prop candidate set.
            candidates = _ppo_r_topk_actions(agent_r, obs_r, args.k_prop)
            e1, max_path_idx, depth_stratum = _e_gate_from_candidates(
                candidates, obs_r, env.max_blocks
            )

            if not candidates:
                # No legal candidate: fall back to PPO-R top-1 (existing env fallback).
                r_idx = int(ppo_idx)
                same_as_ppo_flag = True
            elif args.ranker_gate == "deep_path_only" and not e1:
                # E=0 states bypass the ranker and use PPO-R top-1 directly.
                r_idx = int(ppo_idx)
                same_as_ppo_flag = True
            else:
                a, in_list = _ranker_select_action(
                    ranker, env, req, obs_c, obs_r, agent_r, split_id, server_id, candidates
                )
                same_as_ppo_flag = int(a) == int(ppo_idx)
                ranker_in_candidates_flag = in_list
                r_idx = int(a) if a is not None else int(ppo_idx)

            # Counterfactual PPO-R top-1 outcome, only when the ranker actually changes
            # the action (used to compute improvement/deterioration rates).
            if is_ranker_mode and not is_warmup and not same_as_ppo_flag:
                ppo_info = _ppo_r_top1_outcome(
                    env, req, split_id, server_id, int(ppo_idx), obs_r
                )
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
            if ranker_in_candidates_flag:
                result.ranker_top1_in_candidates += 1

            # Per-E-gate subgroup reporting (only meaningful for ranker modes).
            if is_ranker_mode and max_path_idx >= 0:
                sub = result.e1 if e1 else result.e0
                _update_subgroup(
                    sub,
                    info,
                    same_as_ppo_flag,
                    ranker_changed=not same_as_ppo_flag,
                    ppo_info=ppo_info,
                )
    return result


def evaluate(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",") if s.strip()]
    mod_reg = ModulationRegistry.from_profile(args.modulation_profile)
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)

    rankers: Dict[str, Dict[str, Any]] = {}
    for spec in args.ranker_specs:
        if "=" not in spec:
            raise ValueError(f"Ranker spec must be name=path, got {spec}")
        name, path = spec.split("=", 1)
        rankers[f"ranker_{name}"] = _load_ranker_model(path, args.device)
        print(f"[fair] Loaded ranker '{name}' from {path}")

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    all_results: List[MethodResult] = []

    for seed in seeds:
        args._current_seed = seed
        env = make_env(
            args.topology,
            args.num_slots,
            args.num_servers,
            seed,
            modulation_profile=args.modulation_profile,
            max_blocks=args.max_blocks,
            block_sort_strategy=args.block_sort_strategy_r,
            path_sort_strategy=args.path_sort_strategy_r,
            k=args.k_paths_r,
        )
        rng = np.random.RandomState(seed)
        src = int(rng.randint(0, env.net.NUM_NODES))
        total_requests = args.warmup_requests + args.requests_per_episode
        requests = generate_requests(
            env,
            rng,
            src,
            total_requests,
            args.arrival_interval,
            args.holding_min,
            args.holding_max,
            args.deadline_min,
            args.deadline_max,
            args.size_min_mb,
            args.size_max_mb,
            args.edge_cost_min,
            args.edge_cost_max,
            args.num_splits,
            args.split_profile,
        )
        for mode in modes:
            ranker = rankers.get(mode)
            env.reset(requests)
            result = _run_episode(
                env,
                requests,
                agent_c,
                agent_r,
                mode,
                ranker,
                args,
                warmup_requests=args.warmup_requests,
            )
            all_results.append(result)
            print(
                f"[fair][seed={seed}][mode={mode}] "
                f"blocking={result.blocking_rate():.4%} "
                f"overload={result.overload_rate():.4%} "
                f"nsb={result.nsb_rate():.4%} "
                f"decision_ms={result.avg_decision_ms():.2f} "
                f"ppo_agree={result.ppo_agreement():.2%}",
                flush=True,
            )

    summary: Dict[str, Any] = {}
    for mode in modes:
        rows = [r.to_dict() for r in all_results if r.mode == mode]
        if not rows:
            continue
        def _subgroup_mean(key: str, subgroup: str):
            vals = [r[subgroup].get(key, 0.0) for r in rows if subgroup in r]
            return float(np.mean(vals)) if vals else 0.0

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
            "ranker_top1_in_candidates_mean": float(
                np.mean([r["ranker_top1_in_candidates_rate"] for r in rows])
            ),
            "e0": {
                "total_mean": _subgroup_mean("total", "e0"),
                "blocking_rate_mean": _subgroup_mean("blocking_rate", "e0"),
                "ppo_agreement_mean": _subgroup_mean("ppo_agreement", "e0"),
                "change_rate_mean": _subgroup_mean("change_rate", "e0"),
                "better_rate_mean": _subgroup_mean("better_rate", "e0"),
                "worse_rate_mean": _subgroup_mean("worse_rate", "e0"),
            },
            "e1": {
                "total_mean": _subgroup_mean("total", "e1"),
                "blocking_rate_mean": _subgroup_mean("blocking_rate", "e1"),
                "ppo_agreement_mean": _subgroup_mean("ppo_agreement", "e1"),
                "change_rate_mean": _subgroup_mean("change_rate", "e1"),
                "better_rate_mean": _subgroup_mean("better_rate", "e1"),
                "worse_rate_mean": _subgroup_mean("worse_rate", "e1"),
            },
            "per_seed": rows,
        }

    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    out_json = out / args.output_json
    out_md = out / args.output_md
    payload = {
        "config": {k: v for k, v in vars(args).items() if not k.startswith("_")},
        "summary": summary,
        "results": [r.to_dict() for r in all_results],
    }
    out_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_markdown(out_md, summary, args)
    print(f"[fair] Saved {out_json} and {out_md}")
    return summary


def _write_markdown(path: Path, summary: Dict[str, Any], args: argparse.Namespace) -> None:
    lines = [
        "# Fair Comparison: v1.3 PPO-R Proposal-Supported Counterfactual Reranking",
        "",
        f"- Topology: `{args.topology}`, slots={args.num_slots}, servers={args.num_servers}",
        f"- C-side: K={args.k_paths_c}, sort={args.path_sort_strategy_c}/{args.block_sort_strategy_c}",
        f"- R-side: K={args.k_paths_r}, sort={args.path_sort_strategy_r}/{args.block_sort_strategy_r}",
        f"- Ranker candidate budget: K_prop={args.k_prop}",
        f"- Seeds: {args.seeds}",
        f"- Requests per seed: {args.warmup_requests + args.requests_per_episode} "
        f"(warmup={args.warmup_requests}, evaluated={args.requests_per_episode})",
        "",
        "| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree | Top-1 in cand |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode, s in summary.items():
        lines.append(
            f"| {mode} | {s['blocking_rate_mean']:.2%} ± {s['blocking_rate_std']:.2%} | "
            f"{s['overload_rate_mean']:.2%} | {s['nsb_rate_mean']:.2%} | "
            f"{s['avg_delay_ms_mean']:.2f} | {s['avg_fs_mean']:.2f} | "
            f"{s['avg_decision_ms_mean']:.2f} | {s['ppo_agreement_mean']:.2%} | "
            f"{s.get('ranker_top1_in_candidates_mean', 0):.2%} |"
        )
    lines.extend(["", "## Per-seed details", "", "| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms | PPO agree |", "|---|---|---|---:|---:|---:|---:|"])
    for mode, s in summary.items():
        for row in s["per_seed"]:
            lines.append(
                f"| {row['seed']} | {mode} | {row['blocking_rate']:.2%} | "
                f"{row['overload_rate']:.2%} | {row['nsb_rate']:.2%} | "
                f"{row['avg_decision_ms']:.2f} | {row['ppo_agreement']:.2%} |"
            )
    path.write_text("\n".join(lines), encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--agent_c_checkpoint",
        default="sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt",
    )
    parser.add_argument(
        "--agent_r_checkpoint",
        default="sa_hmarl/checkpoints/agent_r_mixed.pt",
    )
    parser.add_argument(
        "--ranker_specs",
        action="append",
        default=[],
        help="name=path ranker checkpoints to evaluate (e.g. old_v13=path/to/ranking_model.pt). May be repeated.",
    )
    parser.add_argument(
        "--modes",
        default="ppo_r_top1,ksp_ff_plain,ksp_ff_highest",
        help="Comma-separated modes. Ranker modes are added automatically from --ranker_specs.",
    )
    parser.add_argument("--topology", default="xlron_cost239_ptrnet_real")
    parser.add_argument("--num_slots", type=int, default=320)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths_c", type=int, default=5)
    parser.add_argument("--k_paths_r", type=int, default=50)
    parser.add_argument("--k_prop", type=int, default=30)
    parser.add_argument("--path_sort_strategy_c", default="hops")
    parser.add_argument("--path_sort_strategy_r", default="hops")
    parser.add_argument("--block_sort_strategy_c", default="start_asc")
    parser.add_argument("--block_sort_strategy_r", default="start_asc")
    parser.add_argument("--modulation_profile", default="default")
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--split_profile", default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--arrival_interval", type=float, default=0.0625)
    parser.add_argument("--holding_min", type=float, default=20.0)
    parser.add_argument("--holding_max", type=float, default=30.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.1)
    parser.add_argument("--edge_cost_max", type=float, default=2.2)
    parser.add_argument("--seeds", default="4001,4002")
    parser.add_argument("--requests_per_episode", type=int, default=2000)
    parser.add_argument("--warmup_requests", type=int, default=500)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output_dir",
        default="sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking",
    )
    parser.add_argument("--output_json", default="fair_comparison.json")
    parser.add_argument("--output_md", default="FAIR_COMPARISON_QUICK.md")
    parser.add_argument(
        "--ranker_gate",
        choices=["all", "deep_path_only"],
        default="all",
        help=(
            "Online ranker gate. 'all' invokes the ranker on every non-empty candidate "
            "state (v1.3 fix). 'deep_path_only' bypasses the ranker for E=0 states and "
            "uses PPO-R top-1 instead."
        ),
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()

    # Automatically add ranker modes if ranker specs are provided.
    base_modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    auto_ranker_modes = [f"ranker_{spec.split('=', 1)[0]}" for spec in args.ranker_specs]
    all_modes = list(dict.fromkeys(base_modes + auto_ranker_modes))
    args.modes = ",".join(all_modes)

    evaluate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
