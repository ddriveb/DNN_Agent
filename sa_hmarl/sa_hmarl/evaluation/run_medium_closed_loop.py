"""Run paired 5-seed closed-loop fair evaluation for the corrected v1.3 ranker.

Each seed runs in a separate process and evaluates all methods on the exact same
request trace.  Results are merged after all seeds finish, and paired statistical
comparisons are reported.

Outputs:
    sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_CLOSED_LOOP.md
    sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking/MEDIUM_CLOSED_LOOP.json
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from scipy import stats


SEEDS = [3030, 4040, 5050, 6060, 7070]
WARMUP = 1000
REQUESTS = 5000
MAX_WORKERS = 5
OLD_RANKER = "sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt"
SELECTED_RANKER = "sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt"


def _get_selected_ranker(root: Path) -> str:
    """Return the regret-selected checkpoint path, falling back if needed."""
    path = root / SELECTED_RANKER
    if path.exists():
        return str(path)
    report = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking" / "MEDIUM_TRAINING_REPORT_CORRECTED.json"
    if report.exists():
        data = json.loads(report.read_text(encoding="utf-8"))
        ckpt = data.get("selected_checkpoint")
        if ckpt and Path(ckpt).exists():
            return ckpt
    return str(path)


def _run_seed(seed: int, root: Path, out_dir: Path, new_ranker: str) -> Path:
    seed_out = out_dir / f"seed_{seed}"
    seed_out.mkdir(parents=True, exist_ok=True)
    json_path = seed_out / "results.json"
    log_path = seed_out / "eval.log"
    cmd = [
        sys.executable,
        "-m",
        "sa_hmarl.evaluation.eval_r_counterfactual_ranking_cost239_kpath50_hops_fair",
        "--seeds", str(seed),
        "--warmup_requests", str(WARMUP),
        "--requests_per_episode", str(REQUESTS),
        "--modes", "ppo_r_top1,ksp_ff_plain,ksp_ff_highest",
        "--ranker_specs", f"old_v13={OLD_RANKER}",
        "--ranker_specs", f"new_v13={new_ranker}",
        "--k_paths_c", "5",
        "--k_paths_r", "50",
        "--k_prop", "30",
        "--path_sort_strategy_c", "hops",
        "--path_sort_strategy_r", "hops",
        "--block_sort_strategy_c", "start_asc",
        "--block_sort_strategy_r", "start_asc",
        "--output_dir", str(seed_out),
        "--output_json", str(json_path.name),
        "--output_md", "seed_report.md",
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
        raise RuntimeError(f"Seed {seed} evaluator failed (see {log_path})")
    return json_path


def _paired_bootstrap_ci(
    a: np.ndarray, b: np.ndarray, n_boot: int = 2000, seed: int = 12345
) -> Tuple[float, float, float]:
    rng = np.random.RandomState(seed)
    n = len(a)
    diffs = []
    for _ in range(n_boot):
        idx = rng.randint(0, n, size=n)
        diffs.append(a[idx].mean() - b[idx].mean())
    diffs = np.asarray(diffs)
    return float(diffs.mean()), float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def _wilcoxon(a: np.ndarray, b: np.ndarray) -> Tuple[float, float]:
    diff = a - b
    diff = diff[diff != 0]
    if len(diff) == 0:
        return 0.0, 1.0
    stat, p = stats.wilcoxon(diff)
    return float(stat), float(p)


def main() -> int:
    root = Path(__file__).resolve().parents[3]
    exp_dir = root / "sa_hmarl" / "experiments" / "v13_kpath50_hops_counterfactual_reranking"
    exp_dir.mkdir(parents=True, exist_ok=True)
    out_dir = exp_dir / "medium_closed_loop_per_seed"
    out_dir.mkdir(parents=True, exist_ok=True)

    new_ranker = _get_selected_ranker(root)
    print(f"[closed_loop] Using strict v1.3 ranker: {new_ranker}")

    seed_results: Dict[int, Dict[str, Any]] = {}
    print(f"[closed_loop] Running {len(SEEDS)} seeds (max_workers={MAX_WORKERS})")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as pool:
        futures = {pool.submit(_run_seed, seed, root, out_dir, new_ranker): seed for seed in SEEDS}
        for fut in as_completed(futures):
            seed = futures[fut]
            try:
                json_path = fut.result()
                data = json.loads(json_path.read_text(encoding="utf-8"))
                seed_results[seed] = data
                summary = data.get("summary", {})
                print(
                    f"[closed_loop] seed={seed} done: "
                    f"ppo={summary.get('ppo_r_top1', {}).get('blocking_rate_mean', -1):.2%} "
                    f"new_v13={summary.get('ranker_new_v13', {}).get('blocking_rate_mean', -1):.2%}",
                    flush=True,
                )
            except Exception as exc:
                print(f"[closed_loop] seed={seed} FAILED: {exc}", flush=True)
                return 1

    # Aggregate per-seed results.
    all_results: List[Dict[str, Any]] = []
    for data in seed_results.values():
        all_results.extend(data.get("results", []))

    modes = ["ppo_r_top1", "ksp_ff_plain", "ksp_ff_highest", "ranker_old_v13", "ranker_new_v13"]
    blocking_by_seed: Dict[str, List[float]] = {m: [] for m in modes}
    per_mode_rows: Dict[str, List[Dict[str, Any]]] = {m: [] for m in modes}
    for row in all_results:
        mode = row["mode"]
        if mode not in modes:
            continue
        blocking_by_seed[mode].append(row["blocking_rate"])
        per_mode_rows[mode].append(row)

    summary: Dict[str, Any] = {}
    for mode in modes:
        rows = per_mode_rows[mode]
        if not rows:
            continue
        summary[mode] = {
            "seeds": [r["seed"] for r in rows],
            "blocking_rate_mean": float(np.mean(blocking_by_seed[mode])),
            "blocking_rate_std": float(np.std(blocking_by_seed[mode])),
            "overload_rate_mean": float(np.mean([r["overload_rate"] for r in rows])),
            "nsb_rate_mean": float(np.mean([r["nsb_rate"] for r in rows])),
            "avg_delay_ms_mean": float(np.mean([r["avg_delay_ms"] for r in rows])),
            "avg_delay_ms_p95": float(np.percentile([r["avg_delay_ms"] for r in rows], 95)),
            "avg_fs_mean": float(np.mean([r["avg_fs"] for r in rows])),
            "avg_decision_ms_mean": float(np.mean([r["avg_decision_ms"] for r in rows])),
            "avg_decision_ms_p95": float(np.percentile([r["avg_decision_ms"] for r in rows], 95)),
            "ppo_agreement_mean": float(np.mean([r["ppo_agreement"] for r in rows])),
            "ranker_top1_in_candidates_mean": float(np.mean([r.get("ranker_top1_in_candidates_rate", 0.0) for r in rows])),
            "per_seed": rows,
        }

    # Paired comparisons.
    new_v13 = np.asarray(blocking_by_seed.get("ranker_new_v13", []))
    ppo = np.asarray(blocking_by_seed.get("ppo_r_top1", []))
    ksp_plain = np.asarray(blocking_by_seed.get("ksp_ff_plain", []))
    ksp_high = np.asarray(blocking_by_seed.get("ksp_ff_highest", []))

    comparisons: Dict[str, Any] = {}
    if len(new_v13) == len(ppo) and len(new_v13) > 0:
        mean_diff, lo, hi = _paired_bootstrap_ci(ppo, new_v13)  # PPO - new_v13
        stat, p = _wilcoxon(ppo, new_v13)
        comparisons["new_v13_minus_ppo"] = {
            "mean_difference": float(new_v13.mean() - ppo.mean()),
            "bootstrap_mean_ppo_minus_new": mean_diff,
            "bootstrap_95ci_lower": lo,
            "bootstrap_95ci_upper": hi,
            "wilcoxon_statistic": stat,
            "wilcoxon_pvalue": p,
            "n_seeds": len(new_v13),
            "wins_vs_ppo": int((new_v13 < ppo).sum()),
            "ties_vs_ppo": int((new_v13 == ppo).sum()),
            "losses_vs_ppo": int((new_v13 > ppo).sum()),
        }
    if len(new_v13) == len(ksp_plain) and len(new_v13) > 0:
        mean_diff, lo, hi = _paired_bootstrap_ci(ksp_plain, new_v13)
        stat, p = _wilcoxon(ksp_plain, new_v13)
        comparisons["new_v13_minus_ksp_ff_plain"] = {
            "mean_difference": float(new_v13.mean() - ksp_plain.mean()),
            "bootstrap_mean_ksp_minus_new": mean_diff,
            "bootstrap_95ci_lower": lo,
            "bootstrap_95ci_upper": hi,
            "wilcoxon_statistic": stat,
            "wilcoxon_pvalue": p,
            "n_seeds": len(new_v13),
        }
    if len(new_v13) == len(ksp_high) and len(new_v13) > 0:
        mean_diff, lo, hi = _paired_bootstrap_ci(ksp_high, new_v13)
        stat, p = _wilcoxon(ksp_high, new_v13)
        comparisons["new_v13_minus_ksp_ff_highest"] = {
            "mean_difference": float(new_v13.mean() - ksp_high.mean()),
            "bootstrap_mean_ksp_minus_new": mean_diff,
            "bootstrap_95ci_lower": lo,
            "bootstrap_95ci_upper": hi,
            "wilcoxon_statistic": stat,
            "wilcoxon_pvalue": p,
            "n_seeds": len(new_v13),
        }

    payload = {
        "config": {
            "seeds": SEEDS,
            "warmup": WARMUP,
            "requests_per_episode": REQUESTS,
            "topology": "xlron_cost239_ptrnet_real",
            "K_C": 5,
            "K_path": 50,
            "K_prop": 30,
            "strict_v13_ranker": new_ranker,
        },
        "summary": summary,
        "comparisons": comparisons,
        "results": all_results,
    }
    json_path = exp_dir / "MEDIUM_CLOSED_LOOP.json"
    json_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    # Markdown report.
    lines = [
        "# SA-HMARL v1.3 Medium Closed-Loop Fair Comparison",
        "",
        f"- Topology: COST239, slots=320, servers=4",
        f"- C-side: K_C=5, sort=hops/start_asc",
        f"- R-side: K_path=50, sort=hops/start_asc",
        f"- K_prop=30, candidate_mode=ppo_r_topk_only",
        f"- Seeds: {SEEDS}",
        f"- Requests per seed: {WARMUP + REQUESTS} (warmup={WARMUP}, evaluated={REQUESTS})",
        f"- Strict v1.3 ranker: `{new_ranker}`",
        "",
        "## Summary",
        "",
        "| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Delay P95 | Avg FS | Decision ms | Decision P95 | PPO agree | Top-1 in cand |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in modes:
        s = summary.get(mode, {})
        lines.append(
            f"| {mode} | {s.get('blocking_rate_mean', 0):.2%} ± {s.get('blocking_rate_std', 0):.2%} | "
            f"{s.get('overload_rate_mean', 0):.2%} | {s.get('nsb_rate_mean', 0):.2%} | "
            f"{s.get('avg_delay_ms_mean', 0):.2f} | {s.get('avg_delay_ms_p95', 0):.2f} | "
            f"{s.get('avg_fs_mean', 0):.2f} | {s.get('avg_decision_ms_mean', 0):.2f} | "
            f"{s.get('avg_decision_ms_p95', 0):.2f} | {s.get('ppo_agreement_mean', 0):.2%} | "
            f"{s.get('ranker_top1_in_candidates_mean', 0):.2%} |"
        )

    lines.extend(["", "## Per-seed blocking", "", "| Seed | ppo_r_top1 | ksp_ff_plain | ksp_ff_highest | ranker_old_v13 | ranker_new_v13 |", "|---|---:|---:|---:|---:|---:|"])
    for seed in SEEDS:
        row = [str(seed)]
        for mode in modes:
            val = next((r["blocking_rate"] for r in per_mode_rows.get(mode, []) if r["seed"] == seed), float("nan"))
            row.append(f"{val:.2%}")
        lines.append("| " + " | ".join(row) + " |")

    if "new_v13_minus_ppo" in comparisons:
        c = comparisons["new_v13_minus_ppo"]
        lines.extend(["", "## strict v1.3 vs PPO-R Top-1", "", f"- Mean blocking difference (new - ppo): {c['mean_difference']:.4f}", f"- Wins / ties / losses: {c['wins_vs_ppo']} / {c['ties_vs_ppo']} / {c['losses_vs_ppo']}", f"- Paired bootstrap 95% CI for (PPO - new): [{c['bootstrap_95ci_lower']:.4f}, {c['bootstrap_95ci_upper']:.4f}]", f"- Wilcoxon signed-rank p-value: {c['wilcoxon_pvalue']:.4f}"])
    if "new_v13_minus_ksp_ff_plain" in comparisons:
        c = comparisons["new_v13_minus_ksp_ff_plain"]
        lines.extend(["", "## strict v1.3 vs KSP-FF plain", "", f"- Mean blocking difference (new - plain): {c['mean_difference']:.4f}", f"- Paired bootstrap 95% CI for (plain - new): [{c['bootstrap_95ci_lower']:.4f}, {c['bootstrap_95ci_upper']:.4f}]", f"- Wilcoxon signed-rank p-value: {c['wilcoxon_pvalue']:.4f}"])
    if "new_v13_minus_ksp_ff_highest" in comparisons:
        c = comparisons["new_v13_minus_ksp_ff_highest"]
        lines.extend(["", "## strict v1.3 vs KSP-FF highest-mod", "", f"- Mean blocking difference (new - highest): {c['mean_difference']:.4f}", f"- Paired bootstrap 95% CI for (highest - new): [{c['bootstrap_95ci_lower']:.4f}, {c['bootstrap_95ci_upper']:.4f}]", f"- Wilcoxon signed-rank p-value: {c['wilcoxon_pvalue']:.4f}"])

    (exp_dir / "MEDIUM_CLOSED_LOOP.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"[closed_loop] Saved {json_path} and {exp_dir / 'MEDIUM_CLOSED_LOOP.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
