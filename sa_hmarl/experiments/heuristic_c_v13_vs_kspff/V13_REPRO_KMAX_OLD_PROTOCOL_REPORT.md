# V1.3 Kmax Reproducibility Report — Old Long-Horizon Protocol

**Date:** 2026-07-07  
**Topology:** `xlron_cost239_ptrnet_real`  
**Protocol:** old long-horizon benchmark (5 seeds, `edge_cost_min=0.1`, `edge_cost_max=2.2`)

---

## Summary

The previous Kmax sweep produced **v1.3 K=48 blocking ≈15.84 %** because the sweep script did not pass `--edge_cost_min` and defaulted to `0.5`. Re-running the same code under the **old protocol (`edge_cost_min=0.1`)** reproduces the old long-horizon benchmark numbers exactly:

- **KSP-FF:** 10.37 % (old benchmark 10.37 %)
- **v1.3 K=48:** 7.68 % (old benchmark 7.68 %)

Therefore, the drift is **purely a protocol mismatch**, not a ranker, evaluator, or checkpoint regression.

---

## Results

| Method | K | edge_cost_min | Blocking mean ± std | Raw mask empty | Server overload | Delay mean / P95 (ms) | Decision mean / P95 (ms) |
|---|---:|---:|---:|---:|---:|---:|---:|
| KSP-FF | KSP-50 | 0.1 | **10.37 % ± 1.29 %** | 12.99 % | 10.37 % | 8.80 / 16.78 | 6.74 / 9.97 |
| v1.3 | 16 | 0.1 | **8.17 % ± 1.30 %** | 10.34 % | 8.17 % | 9.26 / 16.38 | 9.94 / 14.23 |
| v1.3 | 24 | 0.1 | **7.38 % ± 1.27 %** | 9.46 % | 7.38 % | 9.38 / 16.66 | 9.82 / 13.92 |
| v1.3 | 32 | 0.1 | **8.20 % ± 1.73 %** | 10.44 % | 8.20 % | 9.50 / 16.96 | 9.93 / 14.10 |
| v1.3 | 48 | 0.1 | **7.68 % ± 0.92 %** | 9.71 % | 7.68 % | 9.54 / 17.27 | 9.72 / 13.59 |

All values are averaged over seeds `3030,4040,5050,6060,7070` with `requests_per_episode=10000`, `warmup_requests=2000`, `arrival_interval=0.0625`.

---

## Comparison with Old Long-Horizon Benchmark

| Method | Old benchmark | Current repro (this report) | Δ |
|---|---:|---:|---:|
| KSP-FF | 10.37 % | 10.37 % | 0.00 pp |
| v1.3 K=48 | 7.68 % | 7.68 % | 0.00 pp |

The old long-horizon benchmark is **fully reproduced** by the current code when the same protocol is used.

---

## Comparison with New Kmax Sweep (edge_cost_min=0.5)

| Method | edge_cost_min=0.1 (this report) | edge_cost_min=0.5 (previous sweep) | Δ |
|---|---:|---:|---:|
| KSP-FF | 10.37 % | ~14.89 % (seed 3030) | +4–5 pp |
| v1.3 K=48 | 7.68 % | 15.84 % | +8.16 pp |

The `edge_cost_min` difference alone explains the entire drift.

---

## Interpretation of Kmax Budget

- K=48 matches the old benchmark (7.68 %).
- K=24 gives the lowest mean blocking (7.38 %) but is well within one standard deviation of K=16, K=32, and K=48.
- K=16, K=32, and K=48 blocking rates are all within ~1.3 pp and overlap within their standard deviations.

**Conclusion on candidate budget:** On COST239 under the old protocol, reducing the ranker candidate budget to **K=32 or K=16 does not cause a statistically significant blocking degradation** relative to K=48. However, this observation is based on a single topology; confirm on `german17`, `jpn48`, and `nsfnet_deeprmsa` before drawing a general conclusion.

---

## Output Files

All files use `edge_cost_min=0.1` and do not overwrite the previous `edge_cost_min=0.5` sweep.

| File | Description |
|---|---|
| `xlron_cost239_ptrnet_real_repro_kspff.json` / `.md` | KSP-FF baseline under old protocol |
| `xlron_cost239_ptrnet_real_repro_v13_K16.json` / `.md` | v1.3, K=16 |
| `xlron_cost239_ptrnet_real_repro_v13_K24.json` / `.md` | v1.3, K=24 (supplementary) |
| `xlron_cost239_ptrnet_real_repro_v13_K32.json` / `.md` | v1.3, K=32 |
| `xlron_cost239_ptrnet_real_repro_v13_K48.json` / `.md` | v1.3, K=48 |

The previous `xlron_cost239_ptrnet_real_v13_K*.json` files remain untouched; they represent the `edge_cost_min=0.5` high-load protocol.

---

## Repro Metadata

- Evaluator: `sa_hmarl/evaluation/eval_long_horizon_system_comparison.py`
- Agent-C checkpoint: `agent_c_cost239_r_feasibility_safe_last.pt`
- Ranking checkpoint: `r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`
- `ranker_candidate_mode`: `legalctx48`
- `ranker_ensure_ksp`: `true`
- Seeds: `3030,4040,5050,6060,7070`

---

## Conclusion

The 15.84 % blocking observed in the new Kmax sweep was caused by running under `edge_cost_min=0.5` instead of the old benchmark's `edge_cost_min=0.1`. With the correct protocol, the current code reproduces the old benchmark to four decimal places (KSP-FF 10.37 %, v1.3 K=48 7.68 %). No ranker code change, evaluator rollback, or checkpoint replacement is required.
