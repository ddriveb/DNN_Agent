"""Decompose the causes of raw Agent-C mask emptiness.

For each request we inspect every (split, server) candidate and classify why it is
masked:
  - compute infeasible: edge_compute_cost > available_compute OR edge_compute_ms inf
  - spectrum infeasible: compute OK but no valid (path, mod, block)
  - utilization masked: server_utilization > 0.95

Then we classify each raw_mask_empty request by its dominant bottleneck.

Outputs:
    * JSON: ``experiments/raw_mask_empty_causes.json``
    * Markdown: ``experiments/raw_mask_empty_causes.md``
"""
from __future__ import annotations

import argparse
import json
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sa_hmarl.agents.ppo_agents import PPOAgentC, PPOAgentR
from sa_hmarl.env.observation_builder import build_agent_c_observation
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.training.utils import generate_requests, make_env
from sa_hmarl.utils.checkpoint import load_checkpoint


@dataclass
class RequestRecord:
    seed: int
    episode: int
    step: int
    req_id: int

    raw_mask_empty: bool
    raw_mask_valid: int

    total_candidates: int
    compute_feasible_candidates: int
    spectrum_feasible_candidates: int
    util_passed_candidates: int
    raw_mask_passed_candidates: int

    dominant_cause: str  # compute | spectrum | utilization | mixed | none

    fallback_success: bool
    fallback_reason: str

    avg_server_utilization: float
    max_server_utilization: float
    min_server_available_ratio: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            "seed": self.seed,
            "episode": self.episode,
            "step": self.step,
            "req_id": self.req_id,
            "raw_mask_empty": self.raw_mask_empty,
            "raw_mask_valid": self.raw_mask_valid,
            "total_candidates": self.total_candidates,
            "compute_feasible_candidates": self.compute_feasible_candidates,
            "spectrum_feasible_candidates": self.spectrum_feasible_candidates,
            "util_passed_candidates": self.util_passed_candidates,
            "raw_mask_passed_candidates": self.raw_mask_passed_candidates,
            "dominant_cause": self.dominant_cause,
            "fallback_success": self.fallback_success,
            "fallback_reason": self.fallback_reason,
            "avg_server_utilization": self.avg_server_utilization,
            "max_server_utilization": self.max_server_utilization,
            "min_server_available_ratio": self.min_server_available_ratio,
        }


def _load_ppo_c(ckpt_path: str, device: str = "cpu") -> PPOAgentC:
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    ckpt_args = ckpt.get("args", {}) or {}
    feature_mode = ckpt_args.get(
        "agent_c_feature_mode", ckpt.get("agent_c_feature_mode", "default"),
    )
    num_servers = ckpt_args.get("num_servers")
    if num_servers is None:
        num_servers = ckpt.get("num_servers")
    agent_c_kwargs = dict(
        input_dim=ckpt.get("input_dim", 17),
        hidden_dims=tuple(ckpt.get("hidden_dims", (128, 64))),
        device=device,
        feature_mode=feature_mode,
    )
    if feature_mode in (
        "mean_field", "typed_mean_field", "gated_typed_mean_field",
        "fixed_blend_typed_mean_field", "candidate_mean_field",
        "candidate_mean_field_count_only",
    ):
        agent_c_kwargs["num_servers"] = num_servers
    if feature_mode == "fixed_blend_typed_mean_field":
        agent_c_kwargs["fixed_blend_alpha"] = ckpt_args.get("fixed_blend_alpha", 0.5)
    agent = PPOAgentC(**agent_c_kwargs)
    agent.policy_net.load_state_dict(ckpt["model_state"])
    agent.policy_net.eval()
    agent.checkpoint_args = ckpt_args
    return agent


def _load_ppo_r(ckpt_path: str, mod_reg, device: str = "cpu") -> PPOAgentR:
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


def _classify_candidates(obs_c: Dict[str, Any], util_threshold: float = 0.95) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return boolean arrays (compute_ok, spectrum_ok, util_ok) per candidate."""
    num_servers = len(obs_c["server_utilizations"])
    num_candidates = len(obs_c["candidate_features"])

    compute_ok = np.zeros(num_candidates, dtype=bool)
    spectrum_ok = np.zeros(num_candidates, dtype=bool)
    util_ok = np.zeros(num_candidates, dtype=bool)

    for idx, feat_dict in enumerate(obs_c["candidate_features"]):
        available = float(feat_dict.get("server_available_compute", 0.0))
        capacity = max(float(feat_dict.get("server_capacity", 1.0)), 1e-6)
        edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
        edge_ms = feat_dict.get("edge_compute_ms", 0.0)

        # Compute feasible if edge_cost fits and edge_ms is finite
        cand_compute_ok = (edge_cost <= available) and np.isfinite(edge_ms) and edge_ms != float('inf')
        compute_ok[idx] = cand_compute_ok

        # Utilization pass if server util <= threshold
        server_id = idx % num_servers
        util = float(obs_c["server_utilizations"][server_id])
        util_ok[idx] = util <= util_threshold

        # Spectrum feasible if feasible_count > 0
        split_id = idx // num_servers
        spectrum_ok[idx] = (obs_c["feasible_counts"][split_id][server_id] > 0)

    return compute_ok, spectrum_ok, util_ok


def _dominant_cause(compute_ok: np.ndarray, spectrum_ok: np.ndarray, util_ok: np.ndarray) -> str:
    """Hierarchical classification of why raw mask is empty."""
    n = len(compute_ok)
    if n == 0:
        return "none"

    any_compute = compute_ok.any()
    if not any_compute:
        return "compute"

    # Among compute-feasible candidates, any spectrum feasible?
    compute_feasible_set = np.where(compute_ok)[0]
    if not spectrum_ok[compute_feasible_set].any():
        return "spectrum"

    # Among compute+spectrum feasible candidates, any passes util?
    cs_feasible = compute_ok & spectrum_ok
    if not (cs_feasible & util_ok).any():
        return "utilization"

    return "mixed"


def _run_diagnostic(args: argparse.Namespace) -> Dict[str, Any]:
    seeds = [int(s.strip()) for s in args.seeds.split(",")]

    env_proto = make_env(
        topology=args.topology, num_slots=args.num_slots,
        num_servers=args.num_servers, seed=42,
        slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
        modulation_profile=args.modulation_profile,
        max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
        k=args.k_paths,
    )
    mod_reg = env_proto.mod_reg

    print("Loading frozen R backend...")
    agent_r = _load_ppo_r(args.agent_r_checkpoint, mod_reg, args.device)
    print(f"Loading C policy for rollout: {args.agent_c_checkpoint}")
    agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)

    # We need agent_c only to roll forward; here we just use env.step with fallback.
    # Load is still useful to keep the script consistent with prior diagnostics.
    _ = agent_c

    all_episodes: Dict[int, List[List[Any]]] = {}
    for seed in seeds:
        rng = np.random.RandomState(seed)
        eps = []
        for _ in range(args.episodes):
            src = rng.randint(0, env_proto.net.NUM_NODES)
            eps.append(generate_requests(
                env_proto, rng, src,
                num_requests=args.requests_per_episode,
                arrival_interval=args.arrival_interval,
                holding_min=args.holding_min, holding_max=args.holding_max,
                deadline_min=args.deadline_min, deadline_max=args.deadline_max,
                size_min_mb=args.size_min_mb, size_max_mb=args.size_max_mb,
                edge_cost_min=args.edge_cost_min, edge_cost_max=args.edge_cost_max,
                num_splits=args.num_splits, split_profile=args.split_profile,
            ))
        all_episodes[seed] = eps

    records: List[RequestRecord] = []
    t_start = time.time()

    for seed in seeds:
        for ep_idx, requests in enumerate(all_episodes[seed]):
            env = make_env(
                topology=args.topology, num_slots=args.num_slots,
                num_servers=args.num_servers, seed=42,
                slot_bw_hz=args.slot_bw_hz, guard_band_fs=args.guard_band_fs,
                modulation_profile=args.modulation_profile,
                max_blocks=args.max_blocks, block_sort_strategy=args.block_sort_strategy,
                k=args.k_paths,
            )
            env.reset(requests)

            for step_idx, req in enumerate(requests):
                obs_c = build_agent_c_observation(env, req)
                raw_mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
                raw_valid = int(raw_mask.sum())

                compute_ok, spectrum_ok, util_ok = _classify_candidates(
                    obs_c, util_threshold=args.util_threshold
                )

                cause = "none"
                if raw_valid == 0:
                    cause = _dominant_cause(compute_ok, spectrum_ok, util_ok)

                # Roll forward with fallback (split=0, server=0) only when mask empty
                if raw_valid == 0:
                    from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
                    obs_r = build_agent_r_observation(env, req, 0, 0)
                    action_idx_r = agent_r.select_action(obs_r, deterministic=True)
                    if action_idx_r is not None:
                        action_r = decode_agent_r_action(
                            action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                        )
                    else:
                        action_r = (0, 0, 0)
                    _, _, _, info = env.step((0, 0), action_r)
                    fallback_success = bool(info.get("success", False))
                    fallback_reason = info.get("reason", "unknown")
                else:
                    # Still need to advance env; pick first valid candidate deterministically
                    valid_idx = int(np.where(raw_mask)[0][0])
                    split_id = valid_idx // args.num_servers
                    server_id = valid_idx % args.num_servers
                    from sa_hmarl.env.observation_builder import build_agent_r_observation, decode_agent_r_action
                    obs_r = build_agent_r_observation(env, req, split_id, server_id)
                    action_idx_r = agent_r.select_action(obs_r, deterministic=True)
                    if action_idx_r is not None:
                        action_r = decode_agent_r_action(
                            action_idx_r, len(obs_r["mod_names"]), env.max_blocks
                        )
                    else:
                        action_r = (0, 0, 0)
                    _, _, _, info = env.step((split_id, server_id), action_r)
                    fallback_success = bool(info.get("success", False))
                    fallback_reason = info.get("reason", "unknown")

                utils = np.array(obs_c["server_utilizations"])
                avail_ratios = np.array([
                    f["server_available_compute"] / max(f["server_capacity"], 1e-6)
                    for f in obs_c["candidate_features"]
                ])

                records.append(RequestRecord(
                    seed=seed, episode=ep_idx, step=step_idx,
                    req_id=int(req.req_id),
                    raw_mask_empty=(raw_valid == 0),
                    raw_mask_valid=raw_valid,
                    total_candidates=len(compute_ok),
                    compute_feasible_candidates=int(compute_ok.sum()),
                    spectrum_feasible_candidates=int(spectrum_ok.sum()),
                    util_passed_candidates=int(util_ok.sum()),
                    raw_mask_passed_candidates=raw_valid,
                    dominant_cause=cause,
                    fallback_success=fallback_success,
                    fallback_reason=fallback_reason,
                    avg_server_utilization=float(utils.mean()),
                    max_server_utilization=float(utils.max()),
                    min_server_available_ratio=float(avail_ratios.min()),
                ))

    # Aggregate
    total = len(records)
    raw_empty_records = [r for r in records if r.raw_mask_empty]
    n_empty = len(raw_empty_records)

    cause_counter = Counter(r.dominant_cause for r in raw_empty_records)

    # Candidate-level averages for empty requests
    avg_compute_feasible = float(np.mean([r.compute_feasible_candidates for r in raw_empty_records])) if n_empty else 0.0
    avg_spectrum_feasible = float(np.mean([r.spectrum_feasible_candidates for r in raw_empty_records])) if n_empty else 0.0
    avg_util_passed = float(np.mean([r.util_passed_candidates for r in raw_empty_records])) if n_empty else 0.0

    # Fallback outcomes for empty requests
    empty_success = sum(1 for r in raw_empty_records if r.fallback_success)
    reason_counter = Counter(r.fallback_reason for r in raw_empty_records if not r.fallback_success)

    # Non-empty requests stats
    non_empty = [r for r in records if not r.raw_mask_empty]
    non_empty_success = sum(1 for r in non_empty if r.fallback_success)

    report = {
        "config": {
            "topology": args.topology,
            "num_slots": args.num_slots,
            "num_servers": args.num_servers,
            "k_paths": args.k_paths,
            "max_blocks": args.max_blocks,
            "block_sort_strategy": args.block_sort_strategy,
            "split_profile": args.split_profile,
            "seeds": seeds,
            "episodes": args.episodes,
            "requests_per_episode": args.requests_per_episode,
            "agent_c_checkpoint": args.agent_c_checkpoint,
            "agent_r_checkpoint": args.agent_r_checkpoint,
            "util_threshold": args.util_threshold,
        },
        "total_requests": total,
        "raw_mask_empty_count": n_empty,
        "raw_mask_empty_rate": n_empty / max(total, 1),
        "dominant_cause_counts": dict(cause_counter),
        "dominant_cause_rates": {
            k: v / max(n_empty, 1) for k, v in cause_counter.items()
        },
        "empty_request_avg_feasible_candidates": {
            "compute": avg_compute_feasible,
            "spectrum": avg_spectrum_feasible,
            "util_passed": avg_util_passed,
        },
        "empty_request_fallback": {
            "success": empty_success,
            "blocked": n_empty - empty_success,
            "success_rate": empty_success / max(n_empty, 1),
            "failure_reasons": dict(reason_counter),
        },
        "non_empty_request_fallback": {
            "count": len(non_empty),
            "success": non_empty_success,
            "success_rate": non_empty_success / max(len(non_empty), 1),
        },
        "empty_avg_server_utilization": float(np.mean([r.avg_server_utilization for r in raw_empty_records])) if n_empty else 0.0,
        "empty_max_server_utilization": float(np.mean([r.max_server_utilization for r in raw_empty_records])) if n_empty else 0.0,
        "non_empty_avg_server_utilization": float(np.mean([r.avg_server_utilization for r in non_empty])) if non_empty else 0.0,
        "non_empty_max_server_utilization": float(np.mean([r.max_server_utilization for r in non_empty])) if non_empty else 0.0,
        "elapsed_seconds": time.time() - t_start,
    }
    return report


def _build_markdown(report: Dict[str, Any]) -> str:
    lines: List[str] = []
    lines.append("# Raw C-Mask Empty — Cause Decomposition")
    lines.append("")
    lines.append(f"*Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}*")
    lines.append("")

    cfg = report["config"]
    lines.append("## Setup")
    lines.append("")
    lines.append(f"- **Topology:** {cfg['topology']}")
    lines.append(f"- **Slots:** {cfg['num_slots']}, **Servers:** {cfg['num_servers']}, **k_paths:** {cfg['k_paths']}")
    lines.append(f"- **max_blocks:** {cfg['max_blocks']}, **block_sort:** {cfg['block_sort_strategy']}")
    lines.append(f"- **split_profile:** {cfg['split_profile']}")
    lines.append(f"- **Seeds:** {cfg['seeds']}, **Episodes/seed:** {cfg['episodes']}, **Requests/episode:** {cfg['requests_per_episode']}")
    lines.append(f"- **Utilization threshold:** {cfg['util_threshold']}")
    lines.append(f"- **C checkpoint:** `{cfg['agent_c_checkpoint']}`")
    lines.append("")

    lines.append("## Overall")
    lines.append("")
    lines.append(f"- Total requests: **{report['total_requests']}**")
    lines.append(f"- raw_mask_empty: **{report['raw_mask_empty_count']}** ({report['raw_mask_empty_rate']:.2%})")
    lines.append("")

    lines.append("## Dominant Cause of raw_mask_empty")
    lines.append("")
    lines.append("| Cause | Count | Rate within empty |")
    lines.append("|---|---:|---:|")
    for cause in ["compute", "spectrum", "utilization", "mixed", "none"]:
        count = report["dominant_cause_counts"].get(cause, 0)
        rate = report["dominant_cause_rates"].get(cause, 0.0)
        lines.append(f"| {cause} | {count} | {rate:.2%} |")
    lines.append("")

    lines.append("## Average Feasible Candidates per empty request")
    lines.append("")
    lines.append("| Type | Avg count |")
    lines.append("|---|---:|")
    for k, v in report["empty_request_avg_feasible_candidates"].items():
        lines.append(f"| {k} | {v:.2f} |")
    lines.append("")

    lines.append("## Fallback Outcomes")
    lines.append("")
    fe = report["empty_request_fallback"]
    lines.append(f"- raw_mask_empty requests: {report['raw_mask_empty_count']}")
    lines.append(f"- Fallback success: {fe['success']} ({fe['success_rate']:.2%})")
    lines.append(f"- Fallback blocked: {fe['blocked']}")
    lines.append("")
    lines.append("Failure reasons for empty-mask fallback:")
    lines.append("")
    lines.append("| Reason | Count |")
    lines.append("|---|---:|")
    for reason, count in fe["failure_reasons"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    ne = report["non_empty_request_fallback"]
    lines.append(f"- Non-empty requests: {ne['count']}, success: {ne['success']} ({ne['success_rate']:.2%})")
    lines.append("")

    lines.append("## Server Utilization Context")
    lines.append("")
    lines.append("| Group | Avg util | Avg max util |")
    lines.append("|---|---:|---:|")
    lines.append(f"| raw_mask_empty | {report['empty_avg_server_utilization']:.4f} | {report['empty_max_server_utilization']:.4f} |")
    lines.append(f"| raw_mask_non_empty | {report['non_empty_avg_server_utilization']:.4f} | {report['non_empty_max_server_utilization']:.4f} |")
    lines.append("")

    lines.append("---")
    lines.append(f"*Elapsed: {report['elapsed_seconds']:.1f}s*")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent_c_checkpoint", type=str,
                        default="sa_hmarl/checkpoints/agent_c_r_feasibility_s123_s20_r80_best.pt")
    parser.add_argument("--agent_r_checkpoint", type=str, default="sa_hmarl/checkpoints/agent_r_mixed.pt")
    parser.add_argument("--topology", type=str, default="snap24_gnutella_reach")
    parser.add_argument("--num_slots", type=int, default=20)
    parser.add_argument("--num_servers", type=int, default=4)
    parser.add_argument("--k_paths", type=int, default=5)
    parser.add_argument("--max_blocks", type=int, default=10)
    parser.add_argument("--block_sort_strategy", type=str, default="mixed")
    parser.add_argument("--split_profile", type=str, default="default3")
    parser.add_argument("--num_splits", type=int, default=3)
    parser.add_argument("--seeds", type=str, default="42,123,456")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--requests_per_episode", type=int, default=80)
    parser.add_argument("--arrival_interval", type=float, default=0.25)
    parser.add_argument("--holding_min", type=float, default=4.0)
    parser.add_argument("--holding_max", type=float, default=10.0)
    parser.add_argument("--deadline_min", type=float, default=30.0)
    parser.add_argument("--deadline_max", type=float, default=100.0)
    parser.add_argument("--size_min_mb", type=float, default=5.0)
    parser.add_argument("--size_max_mb", type=float, default=30.0)
    parser.add_argument("--edge_cost_min", type=float, default=0.5)
    parser.add_argument("--edge_cost_max", type=float, default=15.0)
    parser.add_argument("--modulation_profile", type=str, default="default")
    parser.add_argument("--slot_bw_hz", type=float, default=1.25e9)
    parser.add_argument("--guard_band_fs", type=int, default=1)
    parser.add_argument("--util_threshold", type=float, default=0.95)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output_dir", type=str, default="sa_hmarl/experiments")
    args = parser.parse_args()

    report = _run_diagnostic(args)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    json_path = output_dir / "raw_mask_empty_causes.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nJSON saved: {json_path}")

    md_path = output_dir / "raw_mask_empty_causes.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_build_markdown(report))
    print(f"Markdown saved: {md_path}")

    print(f"\nElapsed: {report['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()
