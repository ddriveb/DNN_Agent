"""Create CHECKPOINT_PROVENANCE.md/.json for the v1.3 verified experiment."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "sa_hmarl/experiments/v13_strict_multitopology_cside_verified"

CHECKPOINTS = {
    "agent_c_cost239_native": ROOT / "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt",
    "agent_c_delayaware_transfer": ROOT / "sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt",
    "agent_r_mixed": ROOT / "sa_hmarl/checkpoints/agent_r_mixed.pt",
    "strict_v13_seed42": ROOT / "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt",
    "strict_v13_seed43": ROOT / "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_43/ranking_model.pt",
    "strict_v13_seed44": ROOT / "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_44/ranking_model.pt",
    "legacy_old_v13": ROOT / "sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt",
}


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


def _inspect(path: Path) -> Dict[str, Any]:
    info: Dict[str, Any] = {"path": str(path.relative_to(ROOT)), "sha256": _sha256(path)}
    try:
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        info["keys"] = sorted(ckpt.keys())
        args = ckpt.get("args") or ckpt.get("checkpoint_args") or ckpt.get("training_args")
        if args is not None:
            if hasattr(args, "__dict__"):
                args = {k: v for k, v in vars(args).items() if not k.startswith("_")}
            info["training_args"] = args
        info["model_type"] = ckpt.get("model_type")
        info["input_dim"] = ckpt.get("input_dim")
        info["hidden_dims"] = ckpt.get("hidden_dims")
    except Exception as exc:
        info["load_error"] = str(exc)
    return info


def _provenance_table() -> List[Dict[str, Any]]:
    rows = []
    for name, path in CHECKPOINTS.items():
        info = _inspect(path)
        info["name"] = name
        rows.append(info)
    return rows


def _native_or_transfer(name: str) -> str:
    if "cost239" in name:
        return "native for xlron_cost239_ptrnet_real PPO-C"
    if "delayaware" in name:
        return "transfer/zero-shot PPO-C for German17/NSFNET/JPN48"
    if "agent_r_mixed" in name:
        return "mixed/multi-topology PPO-R (deployed transfer; verify training args)"
    if "strict_v13_seed" in name:
        return "Strict v1.3 ranker trained on COST239; source-matched for COST239, zero-shot transfer to others"
    if "legacy" in name:
        return "legacy/old v1.3 diagnostic reference"
    return "unknown"


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = _provenance_table()
    for row in rows:
        row["deployment_label"] = _native_or_transfer(row["name"])

    (OUT_DIR / "CHECKPOINT_PROVENANCE.json").write_text(
        json.dumps(rows, indent=2, default=str), encoding="utf-8"
    )

    lines = ["# Checkpoint Provenance", ""]
    lines.append("| Name | Path | SHA-256 | Deployment label |")
    lines.append("|---|---|---|---|")
    for row in rows:
        lines.append(
            f"| {row['name']} | {row['path']} | `{row['sha256'][:16]}...` | {row['deployment_label']} |"
        )
    lines.append("")
    lines.append("## Notes")
    lines.append("- COST239 PPO-C is source-matched (native).")
    lines.append("- German17/NSFNET/JPN48 PPO-C uses the delay-aware v2 checkpoint in zero-shot/transfer mode.")
    lines.append("- Strict v1.3 ranker is trained on COST239; COST239 is source-matched, other topologies are zero-shot.")
    lines.append("- PPO-R (agent_r_mixed.pt) is a mixed/multi-topology checkpoint; deployment topology/protocol may differ.")
    (OUT_DIR / "CHECKPOINT_PROVENANCE.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[provenance] Wrote {OUT_DIR / 'CHECKPOINT_PROVENANCE.*'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
