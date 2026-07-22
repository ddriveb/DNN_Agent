"""Correctness gates for the Doherty paper-parity pure RMSA adapter."""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import fields
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action
from sa_hmarl.evaluation.pure_rmsa_paper_env import (
    PureRMSAPaperEnv,
    generate_paper_requests,
)


def _check(name: str, passed: bool, evidence: Dict[str, Any]) -> Dict[str, Any]:
    return {"name": name, "pass": bool(passed), "evidence": evidence}


def _independent_mask(env: PureRMSAPaperEnv, obs: Dict[str, Any], req: Any) -> np.ndarray:
    num_paths = len(obs["candidate_paths"])
    num_mods = env.mod_reg.num_formats
    expected = np.zeros(num_paths * num_mods * env.max_blocks, dtype=bool)
    for path_idx, path in enumerate(obs["candidate_paths"]):
        distance = env.net.path_length_km(path)
        available = env.net.get_available_slots(path)
        for mod_idx in range(num_mods):
            mod = env.mod_reg[mod_idx]
            if distance > mod.reach_km:
                continue
            req_fs = req.required_fs(mod.spectral_efficiency)
            cursor = 0
            blocks = []
            while cursor < len(available):
                if not available[cursor]:
                    cursor += 1
                    continue
                start = cursor
                while cursor < len(available) and available[cursor]:
                    cursor += 1
                size = cursor - start
                if size >= req_fs:
                    blocks.append((start, size))
            for block_idx, _ in enumerate(blocks[:env.max_blocks]):
                expected[path_idx * num_mods * env.max_blocks + mod_idx * env.max_blocks + block_idx] = True
    return expected


def run_gates(states: int) -> Dict[str, Any]:
    checks: List[Dict[str, Any]] = []
    parity_dir = Path("sa_hmarl/experiments/deeprmsa_vs_ksp_ff_paper_parity").resolve()
    sys.path.insert(0, str(parity_dir))
    from paper_rmsa_core import (  # type: ignore
        PaperRMSAEnv,
        generate_paper_requests as canonical_generate,
        ksp_ff_decide,
    )

    topology = "cost239_deeprmsa"
    seed = 5001
    load = 600.0
    canonical = canonical_generate(11, seed, states, 10.0 / load)
    adapted = generate_paper_requests(11, seed, states, load)
    canonical_rows = [
        (x.req_id, x.src_node, x.dst_node, x.bitrate_gbps, x.arrival_time, x.holding_time)
        for x in canonical
    ]
    adapted_rows = [
        (x.req_id, x.src_node, x.dst_node, x.bitrate_gbps, x.arrival_time, x.holding_time)
        for x in adapted
    ]
    checks.append(_check("request_trace_exact_parity", canonical_rows == adapted_rows, {"states": states}))

    canonical_env = PaperRMSAEnv(topology)
    env = PureRMSAPaperEnv(topology)
    mask_mismatches = 0
    ksp_mismatches = 0
    legal_execution_failures = 0
    action_space_sizes = set()
    repeated_hash_material = []
    for old_req, req in zip(canonical, adapted):
        canonical_env.advance_time(old_req.arrival_time)
        env.advance_time(req.arrival_time)
        view = canonical_env.build_view(old_req, 50, "hops")
        old_decision = ksp_ff_decide(view)
        canonical_action = None
        if old_decision is not None:
            old_path = view["paths"][old_decision[0]]
            canonical_action = (
                int(old_decision[0]), int(old_path["best_mod_idx"]),
                int(old_decision[1]), int(old_path["required_fs"]),
            )
        obs = env.build_observation(req)
        action_space_sizes.add(len(obs["agent_r_mask"]))
        expected = _independent_mask(env, obs, req)
        if not np.array_equal(expected, np.asarray(obs["agent_r_mask"], dtype=bool)):
            mask_mismatches += 1
        action = ksp_ff_highest_mod_action(obs)
        adapted_decision = None
        if action is not None:
            path_idx = action // 40
            rem = action % 40
            mod_idx, block_idx = divmod(rem, 10)
            start = obs["candidate_blocks_per_path_mod"][path_idx][mod_idx][block_idx][0]
            adapted_decision = (
                int(path_idx), int(mod_idx), int(start),
                int(obs["required_fs_per_path_mod"][path_idx][mod_idx]),
            )
        if canonical_action != adapted_decision:
            ksp_mismatches += 1
        repeated_hash_material.append((req.req_id, action, int(np.asarray(obs["agent_r_mask"]).sum())))
        if old_decision is not None:
            canonical_env.commit(old_req, view["paths"][old_decision[0]], old_decision[1])
        if action is not None:
            result = env.step(action, obs, req)
            if not result["success"]:
                legal_execution_failures += 1

    checks.append(_check("mask_vs_independent_enumerator", mask_mismatches == 0, {
        "states": states, "mismatches": mask_mismatches,
    }))
    checks.append(_check("ksp_vs_canonical_paper_core", ksp_mismatches == 0, {
        "states": states, "mismatches": ksp_mismatches,
    }))
    checks.append(_check("legal_action_execution", legal_execution_failures == 0, {
        "failures": legal_execution_failures,
    }))
    checks.append(_check("action_space_2000", action_space_sizes == {2000}, {
        "observed_sizes": sorted(action_space_sizes),
    }))

    repeated = generate_paper_requests(11, seed, states, load)
    repeated_rows = [
        (x.req_id, x.src_node, x.dst_node, x.bitrate_gbps, x.arrival_time, x.holding_time)
        for x in repeated
    ]
    checks.append(_check("fixed_seed_reproducibility", adapted_rows == repeated_rows, {"states": states}))
    checks.append(_check("holding_truncation", all(0.0 < x.holding_time < 20.0 for x in adapted), {
        "min": min(x.holding_time for x in adapted),
        "max": max(x.holding_time for x in adapted),
        "mean": float(np.mean([x.holding_time for x in adapted])),
    }))

    low = generate_paper_requests(11, seed, states, 100.0)
    high = generate_paper_requests(11, seed, states, 600.0)
    low_inter = float(np.mean(np.diff([0.0] + [x.arrival_time for x in low])))
    high_inter = float(np.mean(np.diff([0.0] + [x.arrival_time for x in high])))
    checks.append(_check("load_changes_arrival_trace", high_inter < low_inter / 4.0, {
        "load100_mean_interarrival": low_inter,
        "load600_mean_interarrival": high_inter,
    }))

    env_api = set(vars(PureRMSAPaperEnv(topology)).keys())
    forbidden = sorted(env_api & {"mec", "servers", "deadline", "agent_c", "split_id", "server_id"})
    checks.append(_check("no_c_mec_state", not forbidden, {"forbidden_attributes": forbidden}))
    request_fields = sorted(field.name for field in fields(adapted[0]))
    checks.append(_check("pure_request_schema", request_fields == [
        "arrival_time", "bitrate_gbps", "dst_node", "holding_time", "req_id", "src_node"
    ], {"fields": request_fields}))
    fs_examples = {
        "25Gbps_BPSK": adapted[0].__class__(0, 0, 1, 25.0, 0.0, 1.0).required_fs(1.0),
        "100Gbps_16QAM": adapted[0].__class__(0, 0, 1, 100.0, 0.0, 1.0).required_fs(4.0),
    }
    checks.append(_check("required_fs_known_values", fs_examples == {
        "25Gbps_BPSK": 3, "100Gbps_16QAM": 3,
    }, fs_examples))

    passed = all(item["pass"] for item in checks)
    return {"status": "PASS" if passed else "FAIL", "states": states, "checks": checks}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--states", type=int, default=1000)
    parser.add_argument("--output-dir", default="sa_hmarl/experiments/pure_rmsa_strict_v13_vs_ksp_ff_ffksp_k50_hops")
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    payload = run_gates(args.states)
    (output_dir / "PURE_RMSA_MASK_AUDIT.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    lines = ["# Pure RMSA Correctness Gates", "", f"Overall: **{payload['status']}**", "", "| Gate | Status | Evidence |", "|---|---|---|"]
    for check in payload["checks"]:
        lines.append(f"| {check['name']} | {'PASS' if check['pass'] else 'FAIL'} | `{json.dumps(check['evidence'])}` |")
    (output_dir / "PURE_RMSA_MASK_AUDIT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    request_names = {
        "request_trace_exact_parity", "fixed_seed_reproducibility", "holding_truncation",
        "load_changes_arrival_trace", "pure_request_schema", "required_fs_known_values",
    }
    action_names = {
        "mask_vs_independent_enumerator", "ksp_vs_canonical_paper_core",
        "legal_action_execution", "action_space_2000", "no_c_mec_state",
    }
    for stem, names in (("REQUEST_TRACE_PARITY", request_names), ("ACTION_EXECUTION_PARITY", action_names)):
        subset = [check for check in payload["checks"] if check["name"] in names]
        document = {"status": "PASS" if all(x["pass"] for x in subset) else "FAIL", "checks": subset}
        (output_dir / f"{stem}.json").write_text(json.dumps(document, indent=2), encoding="utf-8")
        md = [f"# {stem.replace('_', ' ').title()}", "", f"Overall: **{document['status']}**", ""]
        md.extend(f"- `{item['name']}`: {'PASS' if item['pass'] else 'FAIL'}; `{json.dumps(item['evidence'])}`" for item in subset)
        (output_dir / f"{stem}.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0 if payload["status"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
