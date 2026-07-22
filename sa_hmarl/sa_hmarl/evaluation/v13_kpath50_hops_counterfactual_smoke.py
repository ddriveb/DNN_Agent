"""Smoke test for SA-HMARL v1.3 strict counterfactual dataset generator.

Runs the generator on a tiny one-shard episode and validates the nine protocol
invariants.  Writes a short markdown report to the experiments directory.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np


# Invariant constants
EXPECTED_K_C = 5
EXPECTED_K_PATH = 50
EXPECTED_K_PROP = 30
EXPECTED_H = 5
EXPECTED_FEATURE_DIM = 25
MAX_FLAT_ACTIONS = 50 * 4 * 10  # K_path * mods * max_blocks (upper bound)


def _run_generator(output_dir: Path, root: Path) -> subprocess.CompletedProcess:
    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5",
        "--output_dir", str(output_dir),
        "--n_train_shards", "1",
        "--n_val_shards", "0",
        "--n_test_shards", "0",
        "--requests_per_episode", "200",
        "--warmup", "50",
        "--max_workers", "1",
        "--seeds", "9999,0,0",
        "--device", "cpu",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    return subprocess.run(
        cmd,
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )


def _load_shard(output_dir: Path):
    meta_path = output_dir / "metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata.json missing at {meta_path}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))

    shards = meta.get("shards", [])
    if not shards:
        raise ValueError("No shards in metadata")
    shard = shards[0]
    npz_path = output_dir / shard["npz_path"]
    data = np.load(str(npz_path), allow_pickle=True)
    return meta, data


def _validate_protocol(meta: dict, data: dict) -> dict:
    checks: dict = {}
    failures: list = []

    def ok(name: str, cond: bool, detail: str = ""):
        checks[name] = {"pass": bool(cond), "detail": detail}
        if not cond:
            failures.append(f"{name}: {detail}")

    # 1. Metadata protocol constants
    ok(
        "metadata_K_constants",
        meta.get("K_C") == EXPECTED_K_C
        and meta.get("K_path") == EXPECTED_K_PATH
        and meta.get("K_prop") == EXPECTED_K_PROP
        and meta.get("H") == EXPECTED_H,
        f"K_C={meta.get('K_C')} K_path={meta.get('K_path')} K_prop={meta.get('K_prop')} H={meta.get('H')}",
    )

    # 2. Feature schema
    feature_names = meta.get("feature_names", [])
    ok(
        "feature_dim",
        len(feature_names) == EXPECTED_FEATURE_DIM
        and meta.get("feature_dim") == EXPECTED_FEATURE_DIM,
        f"feature_dim={meta.get('feature_dim')} names={len(feature_names)}",
    )

    # 3. At least one group produced
    mask = np.asarray(data["mask"], dtype=bool)
    n_groups = mask.shape[0]
    n_nonempty = int(mask.any(axis=1).sum())
    ok("nonempty_groups", n_nonempty > 0, f"groups={n_groups} nonempty={n_nonempty}")

    # 4. Mask length equals the candidate budget K_prop (the dataset stores
    #     listwise groups of up to K_prop candidates, not the full flat mask).
    action_dim = mask.shape[1]
    ok("mask_length", action_dim == EXPECTED_K_PROP, f"mask_len={action_dim}")

    # 5. Per-group candidate budget
    cand_counts = mask.sum(axis=1)
    ok(
        "candidate_budget",
        int((cand_counts >= 1).sum()) == n_nonempty
        and int((cand_counts > EXPECTED_K_PROP).sum()) == 0,
        f"min={int(cand_counts.min())} max={int(cand_counts.max())}",
    )

    # 6. Action IDs inside the K_path flat space
    action_ids = np.asarray(data["action_ids"], dtype=np.int64)
    legal_action_ids = action_ids[mask]
    ok(
        "action_id_range",
        legal_action_ids.size > 0
        and int(legal_action_ids.min()) >= 0
        and int(legal_action_ids.max()) < MAX_FLAT_ACTIONS,
        f"min={int(legal_action_ids.min()) if legal_action_ids.size else 'n/a'} "
        f"max={int(legal_action_ids.max()) if legal_action_ids.size else 'n/a'}",
    )

    # 7. Path diversity: at least one group reaches path_idx >= 5 and action_id >= 200
    path_indices = np.asarray(data["path_indices"], dtype=np.int64)
    legal_paths = path_indices[mask]
    max_action_id = int(action_ids[mask].max()) if legal_action_ids.size else -1
    max_path_idx = int(legal_paths.max()) if legal_paths.size else -1
    ok(
        "path_diversity",
        max_path_idx >= 5 and max_action_id >= 200,
        f"max_path_idx={max_path_idx} max_action_id={max_action_id}",
    )

    # 8. Provenance strictly PPO-R Top-K
    provenance = np.asarray(data["provenance"], dtype=np.uint8)
    legal_prov = provenance[mask]
    ok(
        "provenance_ppo_r_topk",
        legal_prov.size > 0 and np.all(legal_prov == 1),
        f"unique={np.unique(legal_prov).tolist()}",
    )

    # 9. Numeric sanity
    features = np.asarray(data["features"], dtype=np.float32)
    returns = np.asarray(data["returns"], dtype=np.float32)
    ok(
        "finite_features",
        np.all(np.isfinite(features[mask])),
        "non-finite feature values detected",
    )
    ok(
        "finite_returns",
        np.all(np.isfinite(returns[mask])),
        "non-finite return values detected",
    )

    # 10. Trace-hash consistency within groups
    trace_hashes = np.asarray(data["trace_hash"], dtype=object)
    ok(
        "trace_hash_present",
        len(trace_hashes) == n_groups
        and all(isinstance(h, str) and len(h) == 16 for h in trace_hashes),
        f"trace_hashes={len(trace_hashes)}",
    )

    # 11. Group IDs are unique and large enough to encode split/seed/episode/req
    group_ids = np.asarray(data["group_id"], dtype=np.int64)
    ok(
        "unique_group_ids",
        len(np.unique(group_ids)) == n_groups,
        f"unique={len(np.unique(group_ids))} groups={n_groups}",
    )

    checks["_failures"] = failures
    checks["_summary"] = {
        "groups": n_groups,
        "nonempty_groups": n_nonempty,
        "max_action_id": max_action_id,
        "max_path_idx": max_path_idx,
        "cand_counts": {
            "min": int(cand_counts.min()),
            "max": int(cand_counts.max()),
            "mean": round(float(cand_counts.mean()), 2),
        },
    }
    return checks


def _write_report(report_path: Path, passed: bool, checks: dict, elapsed: float, stdout: str, stderr: str):
    lines = [
        "# SA-HMARL v1.3 Counterfactual Dataset Smoke Test",
        "",
        f"**Result:** {'PASS' if passed else 'FAIL'}",
        f"**Elapsed:** {elapsed:.1f}s",
        "",
        "## Summary",
        "",
        "```json",
        json.dumps(checks.get("_summary", {}), indent=2),
        "```",
        "",
        "## Invariants",
        "",
        "| Invariant | Pass | Detail |",
        "|---|---|---|",
    ]
    for name, val in checks.items():
        if name.startswith("_"):
            continue
        lines.append(f"| {name} | {'PASS' if val['pass'] else 'FAIL'} | {val.get('detail', '')} |")

    if checks.get("_failures"):
        lines.extend(["", "## Failures", ""])
        for f in checks["_failures"]:
            lines.append(f"- {f}")

    lines.extend(["", "## Generator stdout (last 80 lines)", "", "```text"])
    lines.extend(stdout.splitlines()[-80:])
    lines.extend(["```", "", "## Generator stderr (last 40 lines)", "", "```text"])
    lines.extend(stderr.splitlines()[-40:])
    lines.append("```")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    root = Path(__file__).resolve().parents[3]  # project root
    exp_dir = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking"
    report_path = exp_dir / "SMOKE_TEST.md"

    with tempfile.TemporaryDirectory(prefix="v13_kpath50_smoke_") as tmp:
        output_dir = Path(tmp)
        t0 = time.perf_counter()
        proc = _run_generator(output_dir, root)
        elapsed = time.perf_counter() - t0

        if proc.returncode != 0:
            _write_report(report_path, False, {"_failures": ["generator crashed"], "_summary": {}}, elapsed, proc.stdout, proc.stderr)
            print(f"[smoke] FAIL: generator crashed (see {report_path})")
            return 1

        try:
            meta, data = _load_shard(output_dir)
            checks = _validate_protocol(meta, data)
            passed = len(checks["_failures"]) == 0
        except Exception as exc:
            _write_report(report_path, False, {"_failures": [f"validation exception: {exc}"], "_summary": {}}, elapsed, proc.stdout, proc.stderr)
            print(f"[smoke] FAIL: validation exception (see {report_path})")
            return 1

        _write_report(report_path, passed, checks, elapsed, proc.stdout, proc.stderr)
        print(f"[smoke] {'PASS' if passed else 'FAIL'}: {report_path}")
        return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
