#!/usr/bin/env python3
"""Build the authoritative checkpoint inventory.

The version mapping is intentionally explicit. A newly added checkpoint must be
classified here before the manifest can be regenerated.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
CSV_PATH = PROJECT_ROOT / "checkpoints" / "CHECKPOINT_MANIFEST.csv"
MD_PATH = PROJECT_ROOT / "checkpoints" / "CHECKPOINT_MANIFEST.md"


@dataclass(frozen=True)
class Classification:
    family: str
    version: str
    role: str
    status: str
    canonical: str
    notes: str


@dataclass(frozen=True)
class Record:
    checkpoint_id: str
    path: str
    family: str
    version: str
    role: str
    status: str
    canonical: str
    topology: str
    seed: str
    snapshot: str
    architecture: str
    protocol_id: str
    size_bytes: int
    sha256: str
    notes: str


def _classify(relative_path: str) -> Classification:
    p = relative_path.replace("\\", "/")

    # PPO-C COST239 staged studies.
    match = re.fullmatch(
        r"checkpoints/agent_c_cost239_(default|overload_aware|r_feasibility_safe)"
        r"(?:_(step2_20260704|step3_mixedstress_20260704|step3_test))?_(best|last)\.pt",
        p,
    )
    if match:
        feature, stage, _snapshot = match.groups()
        stage_version = {
            None: "v1-base",
            "step2_20260704": "v2-step2",
            "step3_mixedstress_20260704": "v3-mixedstress",
            "step3_test": "v3-test",
        }[stage]
        version = f"PPO-C/COST239-{feature.replace('_', '-')}/{stage_version}"
        canonical = "support" if p.endswith("agent_c_cost239_r_feasibility_safe_last.pt") else "no"
        status = "active-protocol-support" if canonical == "support" else "historical-candidate"
        return Classification(
            "PPO-C",
            version,
            "C-side split/server policy",
            status,
            canonical,
            "Not used by fixed-C experiments; the r-feasibility-safe last snapshot is retained for native SA-HMARL protocol support.",
        )

    match = re.fullmatch(r"checkpoints/agent_c_delayaware(_v2)?_snap24_(best|last)\.pt", p)
    if match:
        revision = "v2" if match.group(1) else "v1"
        return Classification(
            "PPO-C",
            f"PPO-C/SNAP24-delay-aware/{revision}",
            "C-side split/server policy",
            "historical-candidate",
            "no",
            "Legacy SNAP24 delay-aware C-policy study.",
        )

    if re.fullmatch(r"checkpoints/agent_c_frozen_r_snap24_reach_(best|last)\.pt", p):
        return Classification(
            "PPO-C",
            "PPO-C/SNAP24-frozen-R-reach/v1",
            "C-side split/server policy",
            "historical-candidate",
            "no",
            "Legacy SNAP24 training with a frozen R-side policy.",
        )

    if re.fullmatch(r"checkpoints/agent_c_overload_aware_smoke_(best|last)\.pt", p):
        return Classification(
            "PPO-C",
            "PPO-C/COST239-overload-aware/smoke-v1",
            "C-side smoke checkpoint",
            "smoke-only",
            "no",
            "Smoke configuration (24 slots, K=3, five blocks); never a formal model.",
        )

    # R proposer and transfer checkpoints.
    if p == "checkpoints/agent_r_mixed.pt":
        return Classification(
            "PPO-R proposer",
            "PPO-R/independent-mixed/v1",
            "Frozen legal-action proposer for Strict v1.3",
            "active-core",
            "yes",
            "Canonical Strict v1.3 proposer checkpoint; Strict v1.3 still requires the canonical Ranker checkpoint below.",
        )

    if p == "checkpoints/agent_r_multitopo.pt":
        return Classification(
            "PPO-R proposer",
            "PPO-R/multitopology/v1",
            "Historical multi-topology R policy",
            "historical-candidate",
            "no",
            "Not the Strict v1.3 proposer.",
        )

    if re.fullmatch(r"checkpoints/ppo_r_deeprmsa_bc_snap24_(best|last)\.pt", p):
        return Classification(
            "PPO-R transfer",
            "BC-PPO-R/DeepRMSA-SNAP24/v1",
            "DeepRMSA behavior-cloning warm start",
            "preserved-deeprmsa",
            "no",
            "Teacher-transfer baseline; not the independent PPO-R used by Strict v1.3.",
        )

    # Root DeepRMSA studies.
    root_deeprmsa = {
        "checkpoints/deep_rmsa_c_sensitive_mixed.pt": (
            "DeepRMSA/SA-HMARL-C-sensitive-mixed/v1",
            "Legacy C-sensitive DeepRMSA baseline (K=3, M=1, 24 slots).",
        ),
        "checkpoints/deep_rmsa_snap24_reach_mixed.pt": (
            "DeepRMSA/SNAP24-reach-mixed/v1",
            "Legacy reach-aware DeepRMSA baseline (K=3, M=1, 24 slots).",
        ),
        "checkpoints/deep_rmsa_snap24_k5m10_ai015_h4_10_s5_30.pt": (
            "DeepRMSA/SNAP24-K5-M10-S24/v1",
            "Legacy SNAP24 K=5, M=10 checkpoint.",
        ),
        "checkpoints/deep_rmsa_snap24_s80_k5m10_ai015_h4_10_s5_30.pt": (
            "DeepRMSA/SNAP24-K5-M10-S80/v1",
            "SNAP24 K=5, M=10, 80-slot baseline.",
        ),
        "checkpoints/deep_rmsa_snap24_s100_k5m10_ai015_h4_10_s5_30.pt": (
            "DeepRMSA/SNAP24-K5-M10-S100/v1",
            "SNAP24 K=5, M=10, 100-slot baseline.",
        ),
    }
    if p in root_deeprmsa:
        version, note = root_deeprmsa[p]
        return Classification(
            "DeepRMSA",
            version,
            "DeepRMSA baseline",
            "preserved-deeprmsa",
            "no",
            note,
        )

    # Strict v1.3 development and formal Ranker families.
    match = re.fullmatch(
        r"checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/"
        r"(1k3|3k|5k|full)_seed(42|43|44)/ranking_model\.pt",
        p,
    )
    if match:
        sample_size, _seed = match.groups()
        return Classification(
            "Counterfactual R Ranker",
            f"v1.3-development/medium-{sample_size}/top1-selected",
            "25-D PPO-R Top-30 counterfactual Ranker",
            "historical-development",
            "no",
            "Checkpoint selection used validation Top-1, so this is not the canonical Strict v1.3 definition.",
        )

    match = re.fullmatch(
        r"checkpoints/v13_kpath50_hops_medium_regret_selected/seed_(42|123|456)/ranking_model\.pt",
        p,
    )
    if match:
        return Classification(
            "Counterfactual R Ranker",
            "v1.3-development/medium-full/regret-selected",
            "25-D PPO-R Top-30 counterfactual Ranker",
            "historical-development",
            "no",
            "Regret-selected medium-dataset precursor; superseded by the full-state Strict-loss rerun.",
        )

    match = re.fullmatch(
        r"experiments/v13_strict_fixed/checkpoints/(full_state_uniform|full_state_stratified_depth)/"
        r"seed_(42|43|44)/ranking_model\.pt",
        p,
    )
    if match:
        sampling, _seed = match.groups()
        sampling_label = "uniform" if sampling.endswith("uniform") else "stratified-depth"
        return Classification(
            "Counterfactual R Ranker",
            f"v1.3-full-state/{sampling_label}/pre-strict-loss",
            "25-D PPO-R Top-30 counterfactual Ranker",
            "historical-development",
            "no",
            "Full-state candidate trained before the locked Strict-loss rerun; do not cite as current Strict v1.3.",
        )

    match = re.fullmatch(
        r"experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/"
        r"(full_state_uniform_strict|full_state_stratified_depth_strict)/"
        r"seed_(42|43|44)/ranking_model\.pt",
        p,
    )
    if match:
        sampling, seed = match.groups()
        is_uniform = sampling.startswith("full_state_uniform")
        if is_uniform and seed == "42":
            status, canonical = "active-core", "yes"
        elif is_uniform:
            status, canonical = "reproducibility-replica", "replica"
        else:
            status, canonical = "strict-ablation", "no"
        sampling_label = "uniform" if is_uniform else "stratified-depth"
        return Classification(
            "Strict v1.3 Ranker",
            f"Strict-v1.3/full-state-{sampling_label}/strict-loss",
            "25-D PPO-R Top-30 common-future counterfactual Ranker",
            status,
            canonical,
            "Canonical online Ranker is the uniform seed-42 file; seeds 43/44 are replicas and stratified-depth is an ablation.",
        )

    # DeepRMSA K=50 adapters and paper-parity implementations.
    match = re.fullmatch(
        r"experiments/deeprmsa_adapted_paper_primary_k50/checkpoints/(m1|m10)/seed_(42)/best\.pt",
        p,
    )
    if match:
        blocks, _seed = match.groups()
        return Classification(
            "DeepRMSA",
            f"DeepRMSA/adapted-paper-primary-K50-{blocks.upper()}/v1",
            "Masked A2C DeepRMSA-style adapter",
            "preserved-deeprmsa",
            "no",
            "COST239, K=50 hops adapter; this is not upstream source-semantic DeepRMSA.",
        )

    match = re.fullmatch(
        r"experiments/deeprmsa_source_semantic_paper_primary_k50/(checkpoints|checkpoints_smoke)/"
        r"seed_(42|123|456|99)/best\.pt",
        p,
    )
    if match:
        area, _seed = match.groups()
        smoke = area == "checkpoints_smoke"
        return Classification(
            "DeepRMSA",
            "DeepRMSA/source-semantic-paper-primary-K50/v1",
            "Unmasked source-semantic DeepRMSA port",
            "smoke-only" if smoke else "preserved-deeprmsa",
            "no",
            "Smoke checkpoint." if smoke else "Formal source-semantic K=50 training replica.",
        )

    match = re.fullmatch(
        r"experiments/deeprmsa_vs_ksp_ff_paper_parity/checkpoints/(cost239|nsfnet)/"
        r"seed_(42|43|44)/(best|final)\.pt",
        p,
    )
    if match:
        topology, _seed, _snapshot = match.groups()
        return Classification(
            "DeepRMSA",
            f"DeepRMSA/paper-parity-v1/{topology.upper()}-K5-M1",
            "Paper-semantic DeepRMSA baseline",
            "preserved-deeprmsa",
            "no",
            "Protocol DEEPRMSA_VS_KSP_FF_PAPER_PARITY_V1; best is validation-selected and final is the terminal snapshot.",
        )

    if p == (
        "experiments/strict_v13_vs_ksp_ff_vs_deeprmsa_cost239_fixed_c_all_od/"
        "deep_rmsa_adapted/seed_42/best.pt"
    ):
        return Classification(
            "DeepRMSA",
            "DeepRMSA/topology-matched-COST239-K50-M10/v1",
            "Topology-matched adapted DeepRMSA baseline",
            "preserved-deeprmsa",
            "no",
            "Fixed-C/all-OD, 320 slots, 50 paths, four modulations, ten blocks, 2000 actions; epoch-1 best checkpoint.",
        )

    raise ValueError(f"Unclassified checkpoint: {relative_path}")


def _iter_checkpoint_paths() -> list[Path]:
    paths: list[Path] = []
    for start in (PROJECT_ROOT / "checkpoints", PROJECT_ROOT / "experiments"):
        for base, _dirs, files in os.walk(start, onerror=lambda _error: None):
            for name in files:
                if name.endswith(".pt"):
                    paths.append(Path(base) / name)
    return sorted(paths, key=lambda item: item.as_posix())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _small_metadata(path: Path) -> dict[str, Any]:
    obj = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(obj, dict):
        return {}
    args = obj.get("args") if isinstance(obj.get("args"), dict) else {}
    protocol_config = obj.get("protocol_config") if isinstance(obj.get("protocol_config"), dict) else {}
    return {
        "input_dim": obj.get("input_dim"),
        "hidden_dims": obj.get("hidden_dims"),
        "state_dim": obj.get("state_dim"),
        "n_actions": obj.get("n_actions"),
        "k_path": obj.get("k_path", args.get("k_paths")),
        "m_blocks": obj.get("m_blocks", args.get("max_blocks")),
        "num_slots": obj.get("num_slots", args.get("num_slots")),
        "protocol_id": obj.get("protocol_id", ""),
        "topology": obj.get("topology", protocol_config.get("topology", args.get("topologies", ""))),
        "seed": obj.get("training_seed", obj.get("seed", args.get("seed", ""))),
    }


def _path_seed(path: str) -> str:
    match = re.search(r"seed_?(\d+)", path)
    return match.group(1) if match else ""


def _snapshot(path: str) -> str:
    name = Path(path).name
    if name == "best.pt" or name.endswith("_best.pt"):
        return "best"
    if name == "final.pt":
        return "final"
    if name.endswith("_last.pt"):
        return "last"
    if name == "ranking_model.pt":
        return "metric-selected"
    return "single"


def _architecture(meta: dict[str, Any], family: str) -> str:
    if family in {"Strict v1.3 Ranker", "Counterfactual R Ranker"}:
        return "MLP 25-128-64-1"
    hidden = meta.get("hidden_dims")
    if family == "PPO-C":
        return f"actor-critic input={meta.get('input_dim')}, hidden={hidden}"
    if family in {"PPO-R proposer", "PPO-R transfer"}:
        return f"R-policy input={meta.get('input_dim')}, hidden={hidden}"
    if family == "DeepRMSA":
        return (
            f"actor-critic state={meta.get('state_dim')}, actions={meta.get('n_actions')}, "
            f"K={meta.get('k_path')}, M={meta.get('m_blocks')}"
        )
    return ""


def _build_records() -> list[Record]:
    records: list[Record] = []
    for index, path in enumerate(_iter_checkpoint_paths(), start=1):
        relative = path.relative_to(PROJECT_ROOT).as_posix()
        classification = _classify(relative)
        meta = _small_metadata(path)
        seed = _path_seed(relative) or str(meta.get("seed") or "")
        records.append(
            Record(
                checkpoint_id=f"CKPT-{index:03d}",
                path=relative,
                family=classification.family,
                version=classification.version,
                role=classification.role,
                status=classification.status,
                canonical=classification.canonical,
                topology=str(meta.get("topology") or ""),
                seed=seed,
                snapshot=_snapshot(relative),
                architecture=_architecture(meta, classification.family),
                protocol_id=str(meta.get("protocol_id") or ""),
                size_bytes=path.stat().st_size,
                sha256=_sha256(path),
                notes=classification.notes,
            )
        )
    return records


def _csv_text(records: list[Record]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(asdict(records[0]).keys()), lineterminator="\n")
    writer.writeheader()
    writer.writerows(asdict(record) for record in records)
    return output.getvalue()


def _md_text(records: list[Record]) -> str:
    family_counts = Counter(record.family for record in records)
    status_counts = Counter(record.status for record in records)
    lines = [
        "# Checkpoint Manifest",
        "",
        "This is the authoritative version map for every live `.pt` file under `checkpoints/` and `experiments/`.",
        "It is generated by `scripts/build_checkpoint_manifest.py`; do not hand-edit the table.",
        "",
        "## Version lock",
        "",
        "Strict v1.3 online inference uses exactly these two active-core files:",
        "",
        "1. `checkpoints/agent_r_mixed.pt` (frozen PPO-R legal Top-30 proposer).",
        "2. `experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt` (canonical 25-D Ranker).",
        "",
        "The retained native SA-HMARL C-side support checkpoint is",
        "`checkpoints/agent_c_cost239_r_feasibility_safe_last.pt`; fixed-C experiments do not invoke it.",
        "A `best`, `last`, or `final` suffix describes snapshot timing, not an algorithm version.",
        "",
        "## Counts",
        "",
        f"- Total checkpoints: {len(records)}",
        f"- Total bytes: {sum(record.size_bytes for record in records)}",
        "- Families: " + ", ".join(f"{key}={value}" for key, value in sorted(family_counts.items())),
        "- Statuses: " + ", ".join(f"{key}={value}" for key, value in sorted(status_counts.items())),
        "",
        "## Per-checkpoint versions",
        "",
        "| ID | Version | Snapshot | Seed | Status | Canonical | Path | SHA-256 (12) |",
        "|---|---|---|---:|---|---|---|---|",
    ]
    for record in records:
        lines.append(
            f"| {record.checkpoint_id} | `{record.version}` | {record.snapshot} | {record.seed or '-'} | "
            f"{record.status} | {record.canonical} | `{record.path}` | `{record.sha256[:12]}` |"
        )
    lines.extend(
        [
            "",
            "## Field definitions",
            "",
            "- `active-core`: required by the current Strict v1.3 online R-side method.",
            "- `active-protocol-support`: retained for native SA-HMARL protocol support but not used in fixed-C runs.",
            "- `reproducibility-replica`: same locked method family with a different training seed.",
            "- `strict-ablation`: Strict-loss ablation, not the canonical uniform checkpoint.",
            "- `historical-development` / `historical-candidate`: superseded research artifact.",
            "- `preserved-deeprmsa`: retained because DeepRMSA is the explicit baseline exception.",
            "- `smoke-only`: pipeline check, never formal evidence.",
            "",
            "Use `CHECKPOINT_MANIFEST.csv` for the full role, topology, architecture, protocol ID, byte size, hash, and notes.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true", help="Fail if generated manifests differ from disk.")
    args = parser.parse_args()

    records = _build_records()
    if not records:
        raise RuntimeError("No checkpoints found")
    csv_text = _csv_text(records)
    md_text = _md_text(records)

    if args.check:
        mismatches = []
        for path, expected in ((CSV_PATH, csv_text), (MD_PATH, md_text)):
            actual = path.read_text(encoding="utf-8") if path.exists() else None
            if actual != expected:
                mismatches.append(str(path))
        if mismatches:
            raise SystemExit("Stale checkpoint manifest: " + ", ".join(mismatches))
        print(f"Checkpoint manifest is current: {len(records)} files")
        return

    CSV_PATH.write_text(csv_text, encoding="utf-8", newline="")
    MD_PATH.write_text(md_text, encoding="utf-8", newline="")
    print(f"Wrote {CSV_PATH} and {MD_PATH} for {len(records)} checkpoints")


if __name__ == "__main__":
    main()
