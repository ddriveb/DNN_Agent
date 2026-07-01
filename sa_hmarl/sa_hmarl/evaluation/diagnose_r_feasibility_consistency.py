"""Consistency audit: existing r_feasibility feature vs real Agent-R mask.

This script verifies that the valid-R-action count implied by Agent-C's
existing ``feature_mode="r_feasibility"`` features matches the true number of
valid actions in the Agent-R mask built for the same (split, server).

It does NOT train anything and does NOT introduce new feature modes.
"""
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# Allow running from either repo root or sa_hmarl/.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "sa_hmarl"))
sys.path.insert(0, str(ROOT))

from sa_hmarl.agents.c_agent import AgentC
from sa_hmarl.env.observation_builder import (
    build_agent_c_observation,
    build_agent_r_observation,
    decode_agent_c_action,
    decode_agent_r_action,
)
from sa_hmarl.training.utils import generate_requests, make_env


# ------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------
CONFIG = {
    "topology": "snap24_gnutella_reach",
    "num_slots": 20,
    "num_servers": 4,
    "requests_per_episode": 80,
    "k_paths": 5,
    "max_blocks": 10,
    "block_sort_strategy": "mixed",
    "split_profile": "default3",
    "num_splits": 3,
    "episodes": 5,
    "seeds": [42, 123, 456],
    # Request generation defaults aligned with train_agent_c_with_frozen_r
    "arrival_interval": 0.2,
    "holding_min": 4.0,
    "holding_max": 10.0,
    "deadline_min": 30.0,
    "deadline_max": 100.0,
    "size_min_mb": 5.0,
    "size_max_mb": 30.0,
    "edge_cost_min": 0.5,
    "edge_cost_max": 15.0,
}


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
def _classify_mismatch(
    feat_dict: Dict[str, Any],
    obs_r: Dict[str, Any],
    feature_count: int,
    real_count: int,
) -> str:
    """Classify why feature_count != real_count.

    The candidate feature dict already contains the raw path/mod/block info
    used by the r_feasibility builder.  The real R observation is built from
    the same (split, server).  We compare the components to localize the
    inconsistency.
    """
    feasible_pm = feat_dict.get("_feasible_mask_per_path_mod", [])
    required_pm = feat_dict.get("_required_fs_per_path_mod", [])
    blocks_pm = feat_dict.get("_candidate_blocks_per_path_mod", [])

    real_feasible_pm = obs_r.get("feasible_mask_per_path_mod", [])
    real_required_pm = obs_r.get("required_fs_per_path_mod", [])
    real_blocks_pm = obs_r.get("candidate_blocks_per_path_mod", [])

    # 1. Path count mismatch
    if len(feasible_pm) != len(real_feasible_pm):
        return "path_mismatch"

    num_paths = len(feasible_pm)
    num_mods = len(feasible_pm[0]) if num_paths > 0 else 0

    # 2. Per (path, mod) component comparison
    has_modulation_diff = False
    has_required_fs_diff = False
    has_block_list_diff = False
    has_block_count_diff = False

    for p_idx in range(num_paths):
        if len(feasible_pm[p_idx]) != len(real_feasible_pm[p_idx]):
            return "modulation_reach"  # modulation dimension differs

        for m_idx in range(num_mods):
            feat_feasible = feasible_pm[p_idx][m_idx]
            real_feasible = real_feasible_pm[p_idx][m_idx]
            if feat_feasible != real_feasible:
                has_modulation_diff = True
                continue

            # If both infeasible, nothing more to compare
            if not feat_feasible and not real_feasible:
                continue

            feat_req = required_pm[p_idx][m_idx] if p_idx < len(required_pm) and m_idx < len(required_pm[p_idx]) else None
            real_req = real_required_pm[p_idx][m_idx] if p_idx < len(real_required_pm) and m_idx < len(real_required_pm[p_idx]) else None
            if feat_req != real_req:
                has_required_fs_diff = True
                continue

            feat_blocks = blocks_pm[p_idx][m_idx] if p_idx < len(blocks_pm) and m_idx < len(blocks_pm[p_idx]) else []
            real_blocks = real_blocks_pm[p_idx][m_idx] if p_idx < len(real_blocks_pm) and m_idx < len(real_blocks_pm[p_idx]) else []

            # Compare block lists (start, size) ignoring order differences that
            # may come from sorting strategy
            feat_set = set((int(b[0]), int(b[1])) for b in feat_blocks)
            real_set = set((int(b[0]), int(b[1])) for b in real_blocks)
            if feat_set != real_set:
                has_block_list_diff = True

            # Count valid blocks with current req_fs under both lists
            def _count_valid(blocks, req_fs):
                if req_fs is None or req_fs <= 0:
                    return 0
                return sum(1 for b in blocks if int(b[1]) >= req_fs)

            if _count_valid(feat_blocks, feat_req) != _count_valid(real_blocks, real_req):
                has_block_count_diff = True

    if feature_count != real_count:
        # Distinguish between block-list content vs just capping/sorting
        if has_block_list_diff or has_block_count_diff:
            return "block_count"
        if has_required_fs_diff:
            return "required_fs"
        if has_modulation_diff:
            return "modulation_reach"
        # Same raw info but different count -> almost certainly max_blocks capping
        return "max_blocks"

    return "unknown"


def _step_with_first_valid(env, obs_c, obs_r):
    """Step environment using the first valid (C, R) actions.

    Returns the same info dict as env.step().
    """
    c_mask = obs_c["agent_c_mask"]
    c_valid = np.where(c_mask)[0]
    if len(c_valid) == 0:
        action_c = (0, 0)
    else:
        action_c = decode_agent_c_action(c_valid[0], len(obs_c["server_utilizations"]))

    r_mask = obs_r["agent_r_mask"]
    r_valid = np.where(r_mask)[0]
    if len(r_valid) == 0:
        action_r = (0, 0, 0)
    else:
        action_r = decode_agent_r_action(r_valid[0], len(obs_r["mod_names"]), env.max_blocks)

    _, _, _, info = env.step(action_c, action_r)
    return info


# ------------------------------------------------------------------
# Main diagnostic
# ------------------------------------------------------------------
def run_diagnosis(config: Dict[str, Any]) -> Dict[str, Any]:
    agent = AgentC(input_dim=17, device="cpu", feature_mode="r_feasibility")

    all_mismatches: List[Dict[str, Any]] = []
    total_candidates = 0
    total_exact = 0
    abs_errors: List[int] = []
    reason_counter = Counter()

    for seed in config["seeds"]:
        rng = np.random.RandomState(seed)
        env = make_env(
            topology=config["topology"],
            num_slots=config["num_slots"],
            num_servers=config["num_servers"],
            seed=seed,
            k=config["k_paths"],
            max_blocks=config["max_blocks"],
            block_sort_strategy=config["block_sort_strategy"],
        )

        for ep_idx in range(config["episodes"]):
            src = rng.randint(0, env.net.NUM_NODES)
            requests = generate_requests(
                env,
                rng,
                src,
                config["requests_per_episode"],
                arrival_interval=config["arrival_interval"],
                holding_min=config["holding_min"],
                holding_max=config["holding_max"],
                deadline_min=config["deadline_min"],
                deadline_max=config["deadline_max"],
                size_min_mb=config["size_min_mb"],
                size_max_mb=config["size_max_mb"],
                edge_cost_min=config["edge_cost_min"],
                edge_cost_max=config["edge_cost_max"],
                num_splits=config["num_splits"],
                split_profile=config["split_profile"],
            )
            env.reset(requests)

            for req in requests:
                obs_c = build_agent_c_observation(env, req)
                c_features, _ = agent.build_action_features(obs_c)

                num_servers = len(obs_c["server_utilizations"])
                for cand_idx, feat_dict in enumerate(obs_c["candidate_features"]):
                    split_id = cand_idx // num_servers
                    server_id = cand_idx % num_servers

                    # Raw counts from the existing r_feasibility feature logic
                    diag = AgentC.compute_r_feasibility_diagnostics(obs_c, feat_dict)
                    feature_count = diag["valid_r_actions"]

                    # True Agent-R mask for this (split, server)
                    obs_r = build_agent_r_observation(env, req, split_id, server_id)
                    real_count = int(obs_r["agent_r_mask"].sum())

                    total_candidates += 1
                    abs_err = abs(feature_count - real_count)
                    abs_errors.append(abs_err)

                    if feature_count == real_count:
                        total_exact += 1
                    else:
                        reason = _classify_mismatch(feat_dict, obs_r, feature_count, real_count)
                        reason_counter[reason] += 1
                        if len(all_mismatches) < 20:
                            all_mismatches.append({
                                "seed": seed,
                                "episode": ep_idx,
                                "request_id": req.req_id,
                                "candidate_idx": cand_idx,
                                "split_id": split_id,
                                "server_id": server_id,
                                "feature_count": feature_count,
                                "real_count": real_count,
                                "abs_error": abs_err,
                                "reason": reason,
                                "feature_max_blocks": diag["max_blocks_used"],
                                "env_max_blocks": env.max_blocks,
                                "feature_k_paths": diag["k_paths"],
                                "real_num_paths": len(obs_r["candidate_paths"]),
                                "feature_num_paths": len(feat_dict.get("_feasible_mask_per_path_mod", [])),
                            })

                # Advance environment with first valid actions
                split_id = 0
                server_id = 0
                first_valid_c = int(obs_c["agent_c_mask"].argmax())
                if obs_c["agent_c_mask"][first_valid_c]:
                    split_id, server_id = decode_agent_c_action(first_valid_c, num_servers)
                obs_r = build_agent_r_observation(env, req, split_id, server_id)
                _step_with_first_valid(env, obs_c, obs_r)

    abs_errors_arr = np.array(abs_errors, dtype=int)
    report = {
        "config": config,
        "total_candidates": total_candidates,
        "exact_match_count": total_exact,
        "exact_match_rate": total_exact / max(total_candidates, 1),
        "mismatch_count": total_candidates - total_exact,
        "mean_abs_error": float(abs_errors_arr.mean()),
        "max_abs_error": int(abs_errors_arr.max()),
        "reason_counts": dict(reason_counter),
        "mismatch_examples": all_mismatches,
    }
    return report


# ------------------------------------------------------------------
# Reporting
# ------------------------------------------------------------------
def _fmt_report(report: Dict[str, Any]) -> str:
    lines = [
        "# R-Feasibility Consistency Audit Report\n",
        "## Configuration",
        "",
        "```json",
        json.dumps(report["config"], indent=2),
        "```",
        "",
        "## Summary",
        "",
        f"- **Total candidates audited**: {report['total_candidates']}",
        f"- **Exact matches**: {report['exact_match_count']}",
        f"- **Exact match rate**: {report['exact_match_rate']:.6%}",
        f"- **Mismatches**: {report['mismatch_count']}",
        f"- **Mean absolute error**: {report['mean_abs_error']:.6f}",
        f"- **Max absolute error**: {report['max_abs_error']}",
        "",
        "## Mismatch Reasons",
        "",
        "| Reason | Count |",
        "|--------|-------|",
    ]
    for reason, count in report["reason_counts"].items():
        lines.append(f"| {reason} | {count} |")
    lines.append("")

    lines.extend([
        "## Mismatch Examples (first 20)",
        "",
        "| seed | ep | req | cand | split | server | feature | real | error | reason |",
        "|------|----|-----|------|-------|--------|---------|------|-------|--------|",
    ])
    for ex in report["mismatch_examples"]:
        lines.append(
            f"| {ex['seed']} | {ex['episode']} | {ex['request_id']} | {ex['candidate_idx']} | "
            f"{ex['split_id']} | {ex['server_id']} | {ex['feature_count']} | {ex['real_count']} | "
            f"{ex['abs_error']} | {ex['reason']} |"
        )
    lines.append("")

    lines.extend([
        "## Conclusion",
        "",
    ])
    if report["exact_match_rate"] >= 0.9999 and report["max_abs_error"] == 0:
        lines.append(
            "- Existing `r_feasibility` feature count is **consistent** with the real Agent-R mask."
        )
        lines.append("- OK to proceed to K=5 feature-only training/evaluation.")
    else:
        lines.append(
            "- Existing `r_feasibility` feature count is **NOT fully consistent** with the real Agent-R mask."
        )
        lines.append("- Do NOT train with this feature until the inconsistency is fixed.")
        lines.append(
            f"- Dominant mismatch reason: {max(report['reason_counts'], key=report['reason_counts'].get) if report['reason_counts'] else 'N/A'}."
        )
    lines.append("")
    return "\n".join(lines)


def main():
    report = run_diagnosis(CONFIG)

    out_dir = Path(__file__).resolve().parents[2] / "experiments"
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / "r_feasibility_consistency_report.json"
    md_path = out_dir / "r_feasibility_consistency_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(_fmt_report(report))

    print(f"Report saved to:\n  {json_path}\n  {md_path}")
    print(f"Exact match rate: {report['exact_match_rate']:.6%}")
    print(f"Max abs error: {report['max_abs_error']}")
    print(f"Reason counts: {report['reason_counts']}")


if __name__ == "__main__":
    main()
