"""Run only the legacy old_v13 ranker over the same 5 seeds used in MEDIUM_CLOSED_LOOP.

This is used to back-fill the old_v13 row that was dropped because the fair
evaluator's --ranker_specs parser only kept the last spec (now fixed).
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict

SEEDS = [3030, 4040, 5050, 6060, 7070]
WARMUP = 1000
REQUESTS = 5000
MAX_WORKERS = 5
OLD_RANKER = "sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt"


def _run_seed(seed: int, root: Path, out_dir: Path) -> Path:
    seed_out = out_dir / f"seed_{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    json_path = seed_out / "old_v13_results.json"
    log_path = seed_out / "old_v13_eval.log"
    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.eval_r_counterfactual_ranking_cost239_kpath50_hops_fair",
        "--seeds", str(seed),
        "--warmup_requests", str(WARMUP),
        "--requests_per_episode", str(REQUESTS),
        "--modes", "ranker_old_v13",
        "--ranker_specs", f"old_v13={OLD_RANKER}",
        "--k_paths_c", "5",
        "--k_paths_r", "50",
        "--k_prop", "30",
        "--path_sort_strategy_c", "hops",
        "--path_sort_strategy_r", "hops",
        "--block_sort_strategy_c", "start_asc",
        "--block_sort_strategy_r", "start_asc",
        "--output_dir", str(seed_out),
        "--output_json", str(json_path.name),
        "--output_md", "old_v13_report.md",
    ]
    env = os.environ.copy()
    env["PYTHONPATH"] = "sa_hmarl"
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"
    env["NUMEXPR_NUM_THREADS"] = "1"
    env["TORCH_NUM_THREADS"] = "1"
    with log_path.open("w", encoding="utf-8") as logf:
        proc = subprocess.run(cmd, cwd=root, env=env, stdout=logf, stderr=subprocess.STDOUT, timeout=14400)
    if proc.returncode != 0:
        raise RuntimeError(f"Seed {seed} old_v13 evaluator failed (see {log_path})")
    return json_path


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    out_dir = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking" / "medium_closed_loop_per_seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    results: Dict[int, Dict[str, Any]] = {}
    print(f"[old_v13] Running {len(SEEDS)} seeds (max_workers={MAX_WORKERS})")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_seed, seed, root, out_dir): seed for seed in SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                json_path = fut.result()
                data = json.loads(json_path.read_text(encoding="utf-8"))
                results[seed] = data
                summary = data.get("summary", {}).get("ranker_old_v13", {})
                print(f"[old_v13] seed={seed} done: blocking={summary.get('blocking_rate_mean', -1):.2%}", flush=True)
            except Exception as exc:
                print(f"[old_v13] seed={seed} FAILED: {exc}", flush=True)
                return 1

    merged: Dict[int, Any] = {}
    for seed, data in results.items():
        row = data.get("results", [])
        if row:
            merged[seed] = row[0].to_dict() if hasattr(row[0], "to_dict") else row[0]
    out_file = out_dir / "old_v13_merged.json"
    out_file.write_text(json.dumps(merged, indent=2, default=str), encoding="utf-8")
    print(f"[old_v13] Saved merged results to {out_file}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
