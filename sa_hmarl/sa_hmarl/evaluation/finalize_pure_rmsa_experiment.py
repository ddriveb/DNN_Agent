"""Build final audits and statistical reports for the pure-RMSA comparison."""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

from sa_hmarl.evaluation.eval_pure_rmsa_strict_v13_three_way import (
    PPO_R_CHECKPOINT,
    STRICT_RANKER_CHECKPOINT,
    TOPOLOGIES,
)
from sa_hmarl.evaluation.pure_rmsa_paper_env import generate_paper_requests
from sa_hmarl.network.topology_data import get_topology_edges


EQUIVALENCE_MARGIN = 0.001  # +/-0.10 percentage points.
METHOD_LABELS = {
    "strict_v13": "Frozen Strict v1.3 pure-RMSA transfer",
    "ksp_ff_k50_hops": "KSP-FF K=50 hops",
    "ff_ksp_k50_hops": "FF-KSP K=50 hops",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _bootstrap_ci(values: Sequence[float], seed: int = 20260720, draws: int = 20000) -> Tuple[float, float]:
    array = np.asarray(values, dtype=float)
    rng = np.random.RandomState(seed)
    samples = rng.choice(array, size=(draws, len(array)), replace=True).mean(axis=1)
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def _verdict(ci: Tuple[float, float]) -> str:
    if ci[1] < 0:
        return "Strict significantly lower"
    if ci[0] > 0:
        return "Strict significantly higher"
    if ci[0] >= -EQUIVALENCE_MARGIN and ci[1] <= EQUIVALENCE_MARGIN:
        return "Supports equivalence within +/-0.10 pp"
    return "Insufficient evidence"


def _statistics(runs: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    by_key = {(r["topology"], r["load_erlang"], r["seed"], r["method"]): r for r in runs}
    conditions = sorted({(r["topology"], r["load_erlang"]) for r in runs})
    rows = []
    for topology, load in conditions:
        seeds = sorted({r["seed"] for r in runs if r["topology"] == topology and r["load_erlang"] == load})
        base = [by_key[(topology, load, seed, "strict_v13")]["blocking_rate"] for seed in seeds]
        for comparator in ("ksp_ff_k50_hops", "ff_ksp_k50_hops"):
            other = [by_key[(topology, load, seed, comparator)]["blocking_rate"] for seed in seeds]
            delta = np.asarray(base) - np.asarray(other)
            ci = _bootstrap_ci(delta)
            rows.append({
                "topology": topology, "load_erlang": load, "comparator": comparator,
                "n_seeds": len(seeds), "strict_mean": float(np.mean(base)),
                "comparator_mean": float(np.mean(other)), "paired_delta": float(delta.mean()),
                "ci95_low": ci[0], "ci95_high": ci[1], "verdict": _verdict(ci),
            })
    return rows


def _write_static_audits(output: Path, calibration: Dict[str, Any], checkpoint: Dict[str, Any]) -> None:
    topology_rows = []
    for name, spec in TOPOLOGIES.items():
        edges = get_topology_edges(spec["key"])
        nodes = sorted({int(node) for edge in edges for node in edge[:2]})
        topology_rows.append({
            "label": name, "topology_key": spec["key"], "nodes": len(nodes),
            "undirected_links": len(edges), "directed_arcs": 2 * len(edges),
            "slots_per_directed_arc": 100,
            "role": "paper parity" if name in {"cost239", "nsfnet"} else "protocol transfer",
            "source": "sa_hmarl.network.topology_data", "edge_sha256": hashlib.sha256(
                json.dumps(edges).encode()
            ).hexdigest(),
        })
    (output / "TOPOLOGY_PROVENANCE.json").write_text(json.dumps(topology_rows, indent=2), encoding="utf-8")
    lines = ["# Topology Provenance", "", "| Label | Key | Nodes | Links | Directed arcs | Role |", "|---|---|---:|---:|---:|---|"]
    for row in topology_rows:
        lines.append(f"| {row['label']} | `{row['topology_key']}` | {row['nodes']} | {row['undirected_links']} | {row['directed_arcs']} | {row['role']} |")
    (output / "TOPOLOGY_PROVENANCE.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    checkpoint_lines = [
        "# Checkpoint Compatibility Audit", "", f"Status: **{checkpoint['status']}**", "",
        f"- PPO-R: `{checkpoint['ppo_r_checkpoint']}`", f"- SHA256: `{checkpoint['ppo_r_sha256']}`",
        f"- Ranker: `{checkpoint['ranker_checkpoint']}`", f"- SHA256: `{checkpoint['ranker_sha256']}`",
        f"- Ranker input: {checkpoint['ranker_input_dim']} dimensions in stored feature-name order.",
        "- Action space: 50 paths x 4 modulations x 10 blocks = 2000.",
        "- C-only fields are neutralized to checkpoint means; this is frozen cross-domain transfer.", "",
        "## Disclosed Shifts", "",
    ] + [f"- {item}" for item in checkpoint["known_distribution_shifts"]]
    (output / "CHECKPOINT_COMPATIBILITY_AUDIT.md").write_text("\n".join(checkpoint_lines) + "\n", encoding="utf-8")

    (output / "KSP_FF_IMPLEMENTATION_AUDIT.md").write_text(
        "# KSP-FF Implementation Audit\n\n"
        "Canonical function: `ksp_ff_highest_mod_action`. Loop order is path first, paths are K=50 "
        "in hops/km order, modulation minimizes required FS on that path, and the legal block with "
        "the smallest physical start slot is selected. It is not flat BPSK-first and not K=5.\n",
        encoding="utf-8",
    )
    (output / "FF_KSP_IMPLEMENTATION_AUDIT.md").write_text(
        "# FF-KSP Implementation Audit\n\n"
        "Canonical function: `ff_ksp_highest_mod_action`. It first selects the highest feasible "
        "modulation per path, then sorts legal candidates by physical start slot, path order and "
        "action index. Thus it is spectrum-start-first, unlike path-first KSP-FF.\n",
        encoding="utf-8",
    )
    (output / "PATH_ORDERING_AUDIT.md").write_text(
        "# Path Ordering Audit\n\nAll three methods receive the same K=50 paths sorted by hop count, "
        "then km, then the canonical deterministic path tie-break. Blocks are `start_asc`; flat "
        "encoding is `p*(4*10)+m*10+b`.\n",
        encoding="utf-8",
    )

    locked = {"cost239": [600.0], "nsfnet": [250.0], **calibration["locked_loads"]}
    first = generate_paper_requests(11, 5001, 10, 600.0)
    trace = []
    previous = 0.0
    for request in first:
        trace.append({
            "req_id": request.req_id, "src": request.src_node, "dst": request.dst_node,
            "arrival": request.arrival_time, "interarrival": request.arrival_time - previous,
            "holding": request.holding_time, "bitrate_gbps": request.bitrate_gbps,
        })
        previous = request.arrival_time
    protocol = [
        "# Pure RMSA Protocol Lock", "", "- Traffic and physical core: Doherty 2025 paper-parity implementation.",
        "- Requests contain only src, dst, bitrate, arrival and holding time.",
        "- Interarrival: exponential with mean `10/load`; bitrate: integer uniform 25-100 Gbps.",
        "- Holding: exponential mean 10, redrawn unless `0 < holding < 20` (observed mean about 6.9).",
        "- 100 slots per directed arc, 12.5 GHz slots, one guard slot.",
        "- Modulations in order: BPSK/QPSK/8QAM/16QAM; reach 10000/2500/1250/625 km.",
        "- K=50 paths, hops then km; 10 blocks per path/mod, `start_asc`.",
        "- Strict is always labelled `Frozen Strict v1.3 pure-RMSA transfer`.",
        "- No C-side, MEC, deadline, compute delay, server allocation or queue state exists.", "",
        "## Locked Loads", "",
    ]
    protocol.extend(f"- `{key}`: {value}" for key, value in locked.items())
    protocol.extend(["", "## First Ten COST239 Requests", "", "```json", json.dumps(trace, indent=2), "```"])
    (output / "PURE_RMSA_PROTOCOL_LOCK.md").write_text("\n".join(protocol) + "\n", encoding="utf-8")


def _write_action_summary(output: Path, runs: Sequence[Dict[str, Any]]) -> None:
    totals = Counter()
    total_requests = 0
    for run in runs:
        if run["method"] != "strict_v13":
            continue
        total_requests += run["evaluated"]
        totals.update(run.get("same_state_action_differences", {}))
    lines = [
        "# Same-State Action Difference Summary", "",
        "These are shadow comparisons on the Strict trajectory; heuristic actions are computed but not executed.", "",
        "| Comparison/category | Count | Share of decisions |", "|---|---:|---:|",
    ]
    for key, count in sorted(totals.items()):
        lines.append(f"| {key} | {count} | {count / max(total_requests, 1):.2%} |")
    pilot_runs = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in (output / "shards" / "pilot_paper_exact").glob("*.json")
    ]
    grouped: Dict[Tuple[str, float, int], Dict[str, Dict[int, Dict[str, Any]]]] = defaultdict(dict)
    for run in pilot_runs:
        grouped[(run["topology"], run["load_erlang"], run["seed"])][run["method"]] = {
            int(row["request_index"]): row for row in run.get("trace_rows", [])
        }
    outcomes = Counter()
    first_divergences = []
    for key, methods in grouped.items():
        if set(methods) != set(METHOD_LABELS):
            continue
        request_indices = sorted(set.intersection(*(set(rows) for rows in methods.values())))
        first = None
        for request_index in request_indices:
            rows = {method: methods[method][request_index] for method in METHOD_LABELS}
            successes = tuple(method for method, row in rows.items() if row["success"])
            outcomes["+".join(successes) if successes else "all_block"] += 1
            if first is None and len({rows[method]["action"] for method in METHOD_LABELS}) > 1:
                first = request_index
        first_divergences.append({
            "topology": key[0], "load_erlang": key[1], "seed": key[2],
            "first_action_divergence_request_index": first,
        })
    lines.extend(["", "## Independent Closed-Loop Outcomes (Exact 5-Seed Pilot)", "",
                  "These outcomes align request identity after trajectories naturally diverge.", "",
                  "| Successful method set | Requests |", "|---|---:|"])
    for name, count in sorted(outcomes.items()):
        lines.append(f"| {name} | {count} |")
    lines.extend(["", "## First Action Divergence", "", "```json", json.dumps(first_divergences, indent=2), "```"])
    (output / "ACTION_DIFFERENCE_SUMMARY.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def _blocking_auc(aggregate: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for topology in ("usnet", "jpn48"):
        for method in METHOD_LABELS:
            points = sorted(
                (row["load_erlang"], row["mean_blocking_rate"])
                for row in aggregate if row["topology"] == topology and row["method"] == method
            )
            loads = np.asarray([point[0] for point in points], dtype=float)
            rates = np.asarray([point[1] for point in points], dtype=float)
            rows.append({
                "topology": topology, "method": method,
                "raw_auc": float(np.trapezoid(rates, loads)),
                "load_normalized_auc": float(np.trapezoid(rates, loads) / (loads[-1] - loads[0])),
            })
    return rows


def _plot_curves(output: Path, aggregate: Sequence[Dict[str, Any]]) -> None:
    from PIL import Image, ImageDraw

    image = Image.new("RGB", (1200, 500), "white")
    draw = ImageDraw.Draw(image)
    colors = {"strict_v13": "#c43d3d", "ksp_ff_k50_hops": "#2769a3", "ff_ksp_k50_hops": "#2f855a"}
    for panel, topology in enumerate(("usnet", "jpn48")):
        left, top, right, bottom = 70 + panel * 590, 55, 560 + panel * 590, 420
        draw.rectangle((left, top, right, bottom), outline="#333333", width=2)
        draw.text((left, 20), topology.upper(), fill="#111111")
        all_rows = [row for row in aggregate if row["topology"] == topology]
        x_min = min(row["load_erlang"] for row in all_rows)
        x_max = max(row["load_erlang"] for row in all_rows)
        y_max = max(row["mean_blocking_rate"] for row in all_rows) * 1.1
        for tick in range(5):
            y = bottom - tick * (bottom - top) / 4
            draw.line((left, y, right, y), fill="#dddddd")
            draw.text((left - 52, y - 7), f"{100*y_max*tick/4:.0f}%", fill="#333333")
        for method in METHOD_LABELS:
            rows = sorted(
                (row for row in all_rows if row["method"] == method),
                key=lambda row: row["load_erlang"],
            )
            points = []
            for row in rows:
                x = left + (row["load_erlang"] - x_min) / (x_max - x_min) * (right - left)
                y = bottom - row["mean_blocking_rate"] / y_max * (bottom - top)
                points.append((x, y))
                draw.ellipse((x - 4, y - 4, x + 4, y + 4), fill=colors[method])
                draw.text((x - 12, bottom + 8), f"{row['load_erlang']:.0f}", fill="#333333")
            draw.line(points, fill=colors[method], width=3)
        draw.text((left + 165, 460), "Offered load (Erlang)", fill="#111111")
    for index, method in enumerate(METHOD_LABELS):
        x = 380 + index * 260
        draw.line((x, 485, x + 24, 485), fill=colors[method], width=4)
        draw.text((x + 30, 477), METHOD_LABELS[method], fill="#111111")
    image.save(output / "BLOCKING_LOAD_CURVES.png")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    output = Path(args.output_dir)
    formal_path = output / "FORMAL_RESULTS.json"
    if not formal_path.exists():
        raise SystemExit("FORMAL_RESULTS.json is required")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    runs = formal["runs"]
    statistics = _statistics(runs)
    formal["paired_statistics"] = statistics
    formal["equivalence_margin"] = EQUIVALENCE_MARGIN
    formal["blocking_load_auc"] = _blocking_auc(formal["aggregate"])
    formal_path.write_text(json.dumps(formal, indent=2), encoding="utf-8")

    lines = [
        "# Formal Pure RMSA Results", "",
        "Positive delta means Frozen Strict v1.3 pure-RMSA transfer blocks more.", "",
        "| Topology | Load | Comparator | Strict | Comparator | Delta (pp) | 95% CI (pp) | Verdict |",
        "|---|---:|---|---:|---:|---:|---:|---|",
    ]
    for row in statistics:
        lines.append(
            f"| {row['topology']} | {row['load_erlang']:.0f} | {METHOD_LABELS[row['comparator']]} | "
            f"{row['strict_mean']:.3%} | {row['comparator_mean']:.3%} | {100*row['paired_delta']:+.3f} | "
            f"[{100*row['ci95_low']:+.3f}, {100*row['ci95_high']:+.3f}] | {row['verdict']} |"
        )
    lines.extend(["", "## Blocking-Load AUC", "", "| Topology | Method | Load-normalized AUC |", "|---|---|---:|"])
    for row in formal["blocking_load_auc"]:
        lines.append(f"| {row['topology']} | {METHOD_LABELS[row['method']]} | {row['load_normalized_auc']:.5f} |")
    (output / "FORMAL_RESULTS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    _write_static_audits(
        output,
        json.loads((output / "LOAD_CALIBRATION.json").read_text(encoding="utf-8")),
        json.loads((output / "CHECKPOINT_COMPATIBILITY_AUDIT.json").read_text(encoding="utf-8")),
    )
    _write_action_summary(output, runs)
    _plot_curves(output, formal["aggregate"])
    shutil.copyfile(output / "formal_per_seed_per_load_summary.csv", output / "per_seed_per_load_summary.csv")
    pilot_trace = output / "pilot_action_trace.jsonl.gz"
    if pilot_trace.exists():
        shutil.copyfile(pilot_trace, output / "paired_action_trace.jsonl.gz")
    old = json.loads((output / "OLD_MASK_CAUSE_DECOMPOSITION.json").read_text(encoding="utf-8"))
    with gzip.open(output / "mask_empty_cause_trace.jsonl.gz", "wt", encoding="utf-8") as handle:
        for run in old["runs"]:
            for example in run["examples"]:
                handle.write(json.dumps({"seed": run["seed"], **example}) + "\n")

    source_paths = [
        Path("sa_hmarl/sa_hmarl/evaluation/pure_rmsa_paper_env.py"),
        Path("sa_hmarl/sa_hmarl/evaluation/eval_pure_rmsa_strict_v13_three_way.py"),
        Path("sa_hmarl/sa_hmarl/evaluation/audit_pure_rmsa_gates.py"),
        Path("sa_hmarl/sa_hmarl/evaluation/diagnose_old_fixed_c_mask_causes.py"),
        Path("sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py"),
    ]
    manifest = {
        "status": "complete", "seeds": list(range(5001, 5021)), "warmup": 500,
        "evaluated_per_seed": 6000, "methods": METHOD_LABELS,
        "worker_peak_mb_probe": 260.1, "formal_max_workers": 8,
        "source_sha256": {str(path): _sha256(path) for path in source_paths},
        "checkpoint_sha256": {
            PPO_R_CHECKPOINT: _sha256(Path(PPO_R_CHECKPOINT)),
            STRICT_RANKER_CHECKPOINT: _sha256(Path(STRICT_RANKER_CHECKPOINT)),
        },
        "request_hash_parity": len({
            (r["topology"], r["load_erlang"], r["seed"], r["request_trace_hash"])
            for r in runs
        }) == len(runs) // 3,
    }
    (output / "EXPERIMENT_MANIFEST.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    all_worse = all(row["verdict"] == "Strict significantly higher" for row in statistics)
    decision = [
        "# Next Step Decision", "",
        "## Decision", "",
        "Stop using the frozen SA-HMARL Strict v1.3 checkpoint as a claimed pure-RMSA improvement."
        if all_worse else "Do not claim equivalence until all paired intervals satisfy the locked criterion.", "",
        "The result is a cross-domain transfer result, not native pure-RMSA capability. The strongest next step "
        "is either native pure-RMSA retraining with matched traffic/modulation/slot semantics, or a separate "
        "reconfiguration/defragmentation study. Do not tune load or redefine the heuristics after seeing Strict results.",
    ]
    (output / "NEXT_STEP_DECISION.md").write_text("\n".join(decision) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
