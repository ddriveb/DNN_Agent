#!/usr/bin/env python3
"""Audit existing df_c full results against the unified main-table protocol."""
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

ROOT = Path(__file__).parent

UNIFIED_PROTOCOL = {
    "block_sort_strategy": "start_asc",
    "path_sort_strategy": "hops",
    "ksp_ff_k50_hops_k_paths": 50,
    "requests_per_episode": 10000,
    "warmup_requests": 2000,
    "seeds": "3030,4040,5050,6060,7070",
    "episodes": 1,
    "poisson_arrivals": True,
    "exponential_holding": True,
    "num_slots": 320,
    "num_servers": 4,
    "split_profile": "default3",
    "num_splits": 3,
}

TOPOLOGY_PARAMS = {
    "xlron_cost239_ptrnet_real": {"arrival_interval": 0.0625, "edge_cost_max": 2.2},
    "xlron_german17": {"arrival_interval": 0.07142857142857142, "edge_cost_max": 3.0},
    "xlron_nsfnet_deeprmsa": {"arrival_interval": 0.07692307692307693, "edge_cost_max": 4.0},
    "xlron_jpn48": {"arrival_interval": 0.1, "edge_cost_max": 5.2},
}

V13_CONFIG = {
    "ranker_candidate_mode": "legalctx48",
    "ranker_max_candidates": 48,
    "ranker_ensure_ksp": True,
}

REQUIRED_METHODS = ["df_c+ksp_ff_k50_hops", "df_c+v12_k50_hops"]


def load_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def audit_topology(topo_key: str) -> Dict[str, Any]:
    file_path = ROOT / f"{topo_key}_df_full.json"
    result = {
        "topology": topo_key,
        "file": str(file_path.relative_to(ROOT.parent.parent.parent)),
        "exists": file_path.exists(),
        "compliant": False,
        "mismatches": [],
        "methods_present": [],
        "methods_missing": [],
    }
    if not file_path.exists():
        result["mismatches"].append("file not found")
        return result

    data = load_json(file_path)
    config = data.get("config", {})

    # Check unified protocol fields
    for key, expected in UNIFIED_PROTOCOL.items():
        actual = config.get(key)
        if actual != expected:
            result["mismatches"].append(f"{key}: expected {expected!r}, got {actual!r}")

    # Check topology-specific params
    topo_expected = TOPOLOGY_PARAMS.get(topo_key, {})
    for key, expected in topo_expected.items():
        actual = config.get(key)
        if abs(float(actual) - float(expected)) > 1e-9:
            result["mismatches"].append(f"{key}: expected {expected!r}, got {actual!r}")

    # Check v1.3 config for v12_k50_hops method
    for key, expected in V13_CONFIG.items():
        actual = config.get(key)
        if actual != expected:
            result["mismatches"].append(f"{key}: expected {expected!r}, got {actual!r}")

    # Check methods present
    methods = data.get("methods", {})
    for method in REQUIRED_METHODS:
        if method in methods:
            result["methods_present"].append(method)
        else:
            result["methods_missing"].append(method)

    # Check per-seed keys
    for method in result["methods_present"]:
        per_seed = methods[method].get("per_seed", {})
        expected_seeds = [str(s) for s in UNIFIED_PROTOCOL["seeds"].split(",")]
        missing_seeds = [s for s in expected_seeds if s not in per_seed]
        if missing_seeds:
            result["mismatches"].append(f"{method} missing seeds {missing_seeds}")

    result["compliant"] = (
        len(result["mismatches"]) == 0 and len(result["methods_missing"]) == 0
    )
    return result


def main():
    audits = [audit_topology(topo) for topo in TOPOLOGY_PARAMS.keys()]

    report_lines = [
        "# df_c Existing Results Audit Against Unified Protocol",
        "",
        "## Unified Protocol",
        "",
        "| Parameter | Value |",
        "|---|---|",
    ]
    for key, value in UNIFIED_PROTOCOL.items():
        report_lines.append(f"| {key} | {value} |")
    report_lines.append("| v1.3 ranker_candidate_mode | legalctx48 |")
    report_lines.append("| v1.3 ranker_max_candidates | 48 |")
    report_lines.append("| v1.3 ranker_ensure_ksp | true |")
    report_lines.append("")
    report_lines.append("| Topology | arrival_interval | edge_cost_max |")
    report_lines.append("|---|---|---|")
    for topo, params in TOPOLOGY_PARAMS.items():
        report_lines.append(
            f"| {topo} | {params['arrival_interval']} | {params['edge_cost_max']} |"
        )
    report_lines.append("")
    report_lines.append("## Audit Results")
    report_lines.append("")
    report_lines.append("| Topology | Compliant | Mismatches | Present Methods | Action |")
    report_lines.append("|---|---|---|---|---|")

    for audit in audits:
        mismatches_str = "; ".join(audit["mismatches"]) if audit["mismatches"] else "none"
        methods_str = ", ".join(audit["methods_present"])
        action = "reuse" if audit["compliant"] else "re-run"
        report_lines.append(
            f"| {audit['topology']} | {audit['compliant']} | {mismatches_str} | {methods_str} | {action} |"
        )

    report_lines.append("")

    out_md = ROOT / "DF_C_AUDIT_REPORT.md"
    out_json = ROOT / "DF_C_AUDIT_REPORT.json"
    out_md.write_text("\n".join(report_lines), encoding="utf-8")
    out_json.write_text(json.dumps(audits, indent=2), encoding="utf-8")
    print(f"Wrote {out_md}")
    print(f"Wrote {out_json}")

    # Print summary to stdout
    print("\nSummary:")
    for audit in audits:
        status = "REUSE" if audit["compliant"] else "RE-RUN"
        print(f"  {audit['topology']}: {status}")
        for m in audit["mismatches"]:
            print(f"    - {m}")


if __name__ == "__main__":
    main()
