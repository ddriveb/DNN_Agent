"""Generate v2 audit/deliverable reports after micro and smoke."""
from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List

BASE = Path("sa_hmarl/experiments/v13_strict_multitopology_cside_verified_v2")
ROOT = Path(__file__).resolve().parents[2]


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _load_results(phase: str):
    rows = []
    phase_dir = BASE / phase
    if not phase_dir.exists():
        return rows
    for topo_dir in sorted(phase_dir.glob("xlron_*")):
        for seed_dir in sorted(topo_dir.glob("seed_*")):
            json_path = seed_dir / "results.json"
            if not json_path.exists():
                continue
            payload = json.loads(json_path.read_text())
            rows.extend(payload.get("results", []))
    return rows


def _implementation_fix_audit() -> Dict[str, Any]:
    files = [
        ROOT / "sa_hmarl/sa_hmarl/env/event_env.py",
        ROOT / "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_verified.py",
        ROOT / "sa_hmarl/sa_hmarl/evaluation/run_strict_v13_multitopology_cside_verified.py",
        ROOT / "sa_hmarl/sa_hmarl/evaluation/analyze_strict_v13_multitopology_cside_verified.py",
    ]
    return {
        "fixes": [
            {"item": "SMDPEnv.reject_next_request(expected_req_id, reason)", "file": str(files[0]), "sha256": _sha256(files[0])},
            {"item": "Per-loop queue-head req_id/arrival_time assertions and queue-length conservation", "file": str(files[1]), "sha256": _sha256(files[1])},
            {"item": "R-mask checked before PPO-R/KSP-FF/Strict ranker", "file": str(files[1]), "sha256": _sha256(files[1])},
            {"item": "Hard validation of selected R actions against mask, required_fs, modulation reach, block size", "file": str(files[1]), "sha256": _sha256(files[1])},
            {"item": "E=0/E=1 computed from PPO-R Top-30 legal candidates", "file": str(files[1]), "sha256": _sha256(files[1])},
            {"item": "Phase-isolated output directories (micro/smoke/full)", "file": str(files[2]), "sha256": _sha256(files[2])},
            {"item": "Smoke marker enrichment and full-phase gate (no --skip_smoke_check)", "file": str(files[2]), "sha256": _sha256(files[2])},
            {"item": "Delta direction unified to baseline - strict and failure decomposition split", "file": str(files[3]), "sha256": _sha256(files[3])},
        ]
    }


def _request_sync_audit(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "total_rows": len(rows),
        "request_id_mismatch_count_sum": sum(r["request_id_mismatch_count"] for r in rows),
        "final_event_queue_length_sum": sum(r["final_event_queue_length"] for r in rows),
        "selected_mask_false_count_sum": sum(r["selected_mask_false_count"] for r in rows),
        "invalid_path_after_legal_action_sum": sum(r["invalid_path_after_legal_action_count"] for r in rows),
        "modulation_reach_after_legal_action_sum": sum(r["modulation_reach_after_legal_action_count"] for r in rows),
        "no_suitable_block_after_legal_action_sum": sum(r["no_suitable_block_after_legal_action_count"] for r in rows),
        "action_observation_consistency_error_sum": sum(r["action_observation_consistency_error_count"] for r in rows),
        "all_gates_passed": all(
            r["request_id_mismatch_count"] == 0
            and r["final_event_queue_length"] == 0
            and r["action_observation_consistency_error_count"] == 0
            for r in rows
        ),
    }


def _test_report() -> Dict[str, Any]:
    test_files = [
        "sa_hmarl/tests/test_v13_multitopology_cside_verified.py",
        "sa_hmarl/tests/test_v13_multitopology_request_sync.py",
        "sa_hmarl/tests/test_ksp_ff_implementation_audit.py",
        "sa_hmarl/tests/test_v13_full_state_group_filter.py",
        "sa_hmarl/tests/test_regret_metrics.py",
        "sa_hmarl/tests/test_r_ranker_feature_batch_consistency.py",
    ]
    cmd = [
        sys.executable, "-m", "pytest", *test_files, "-q"
    ]
    env = {"PYTHONPATH": "sa_hmarl"}
    proc = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    return {
        "command": " ".join(cmd),
        "returncode": proc.returncode,
        "passed": "passed" in proc.stdout.lower() and proc.returncode == 0,
        "stdout_tail": proc.stdout.splitlines()[-10:],
        "stderr_tail": proc.stderr.splitlines()[-10:],
        "test_files": test_files,
    }


def _phase_report(phase: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    return {
        "phase": phase,
        "cells": len(rows),
        "request_sync_audit": _request_sync_audit(rows),
        "per_method": {
            row["method_name"]: {
                "topology": row["topology"],
                "seed": row["seed"],
                "evaluated_requests": row["evaluated_requests"],
                "blocked": row["blocked"],
                "blocking_rate": row["blocking_rate"],
                "c_no_valid_action": row["c_no_valid_action"],
                "r_no_valid_action": row["r_no_valid_action"],
                "no_suitable_block": row["no_suitable_block"],
                "server_overload": row["server_overload"],
                "deadline_failure": row["deadline_failure"],
                "other_failure": row["other_failure"],
            }
            for row in rows
        },
    }


def _write_json_md(name: str, payload: Dict[str, Any]):
    (BASE / f"{name}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    lines = [f"# {name.replace('_', ' ').title()}", "", "```json"]
    lines.append(json.dumps(payload, indent=2, default=str))
    lines.append("```")
    (BASE / f"{name}.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    BASE.mkdir(parents=True, exist_ok=True)
    impl = _implementation_fix_audit()
    _write_json_md("IMPLEMENTATION_FIX_AUDIT", impl)

    micro_rows = _load_results("micro")
    smoke_rows = _load_results("smoke")
    all_rows = micro_rows + smoke_rows
    req_sync = _request_sync_audit(all_rows)
    _write_json_md("REQUEST_SYNC_AUDIT", req_sync)

    test_rep = _test_report()
    _write_json_md("TEST_REPORT", test_rep)

    micro_rep = _phase_report("micro", micro_rows)
    _write_json_md("MICRO_E2E_REPORT", micro_rep)

    smoke_rep = _phase_report("smoke", smoke_rows)
    _write_json_md("SMOKE_VALIDATION", smoke_rep)

    manifest = {
        "schema_version": "v1.3-verified-2026-07-14",
        "base_dir": str(BASE),
        "files": {str(p.relative_to(BASE)): _sha256(p) for p in BASE.rglob("*") if p.is_file()},
    }
    (BASE / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, default=str), encoding="utf-8")

    print(f"[audit] Wrote reports to {BASE}")
    print(f"[audit] request sync gates passed: {req_sync['all_gates_passed']}")
    print(f"[audit] tests passed: {test_rep['passed']}")
    return 0 if req_sync["all_gates_passed"] and test_rep["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
