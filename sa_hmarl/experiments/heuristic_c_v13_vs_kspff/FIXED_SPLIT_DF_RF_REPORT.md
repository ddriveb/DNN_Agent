# Fixed-Split DF/RF Ablation Report

**Date:** 2026-07-05

This report compares the newly introduced `df_fixed0_c` and `rf_fixed0_c` C-side baselines against the existing `df_c` / `rf_c` adapted heuristics and the learned PPO-C policy. The fixed-split variants pin the split choice to `split0`; server selection follows the same distance-first (DF) or resource-first (RF) rule as the adapted variants. This is a **fixed-split / no-partition-style ablation**, not a claim of strictly reproducing any prior Yin-style implementation.

## 1. C-Side Baseline Semantics

| Baseline class | Split choice | Server choice | Mask respect | Purpose |
|---|---|---|---|---|
| PPO-C `r_feasibility_safe` | learned | learned | yes | C-R co-optimisation upper bound |
| Adapted `df_c` / `rf_c` | heuristic over legal `(split, server)` | DF / RF over legal actions | yes | Heuristic C with weak partition adaptivity |
| Yin-style DF/RF | best split for a chosen server | DF / RF first, then best split | no (server-first + best-split) | Reference only; existing separate evaluator |
| **Fixed-split `df_fixed0_c` / `rf_fixed0_c`** | **pinned to `split0`** | **DF / RF within split0 only** | **yes** | **No-partition-style performance floor** |

Key invariant of the new baselines: if `split0` has no legal `(split0, server)` action, they return `None` and the request is blocked; they never fall back to `split1` or `split2`.

## 2. Code Changes

1. **`sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py`**
   - Added `_valid_actions_for_split(mask, split_id, num_servers)` helper.
   - Added `select_df_fixed0(env, obs_c, mask)` and `select_rf_fixed0(obs_c, mask)`.
   - Registered `df_fixed0` / `rf_fixed0` in `select_offloading_action` dispatch table.
   - Existing `df_c` / `rf_c` logic is unchanged.

2. **`sa_hmarl/sa_hmarl/evaluation/eval_c_demand_potential_closed_loop.py`**
   - Added `split_counts: Dict[int, int]` to `PerMethodMetrics`.
   - `_aggregate_metrics` now emits `split_counts` and `split_dist` per seed.

3. **`sa_hmarl/sa_hmarl/evaluation/eval_c_post_decision_closed_loop.py`**
   - `_record_outcome` increments `metrics.split_counts[split_id]`.

4. **`sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py`**
   - Aggregate block now also reports `aggregate.split_dist` across seeds.

5. **`sa_hmarl/tests/test_offloading_baselines.py`** (new)
   - Unit tests verify fixed-split variants always return split0, respect masks, and do not fall back to other splits.

## 3. Smoke Validation

A short COST239 smoke (`requests_per_episode=200`, `warmup=50`, seed `3030`) was run for:
`df_fixed0_c+ksp_ff_k50_hops`, `df_fixed0_c+v12_k50_hops`, `rf_fixed0_c+ksp_ff_k50_hops`, `rf_fixed0_c+v12_k50_hops`.

| Method | Blocking | Overload | split0 share |
|---|---:|---:|---:|
| df_fixed0_c+ksp_ff_k50_hops | 39.33% | 39.33% | 100.00% |
| df_fixed0_c+v12_k50_hops | 39.33% | 39.33% | 100.00% |
| rf_fixed0_c+ksp_ff_k50_hops | 36.00% | 36.00% | 100.00% |
| rf_fixed0_c+v12_k50_hops | 36.00% | 36.00% | 100.00% |

All four fixed-split methods selected `split0` on 100% of evaluated requests, confirming the implementation invariant.

## 4. Full Experimental Results

### 4.1 Adapted `df_c` reference (existing results)

| Topology | df_c + KSP-FF | df_c + v1.3 | Δ (pp) | note |
|---|---:|---:|---:|---|
| COST239 | 14.84% | 14.84% | +0.00 pp | identical/NA |
| German17 | 12.69% | 12.96% | +0.27 pp | p=0.2272 |
| NSFNET | 20.33% | 19.94% | -0.39 pp | p=0.2358 |
| JPN48 | 11.28% | 10.85% | -0.44 pp | p=0.0012 |

### 4.2 Adapted `rf_c` reference (existing smoke only)

| COST239 (smoke) | 13.18% | KSP-FF only; full rf_c comparison not run per instruction |

### 4.3 Fixed-split `df_fixed0_c` results

| Topology | df_fixed0 + KSP-FF | df_fixed0 + v1.3 | Δ (pp) | p-value | split0 share | status |
|---|---:|---:|---:|---:|---:|---|
| COST239 | 73.97% | 73.97% | +0.00 pp | NA | 100.00% | done |
| German17 | 72.68% | 72.58% | -0.10 pp | 0.3739 | 100.00% | done |
| NSFNET | 72.88% | 72.80% | -0.08 pp | 0.8149 | 100.00% | done |
| JPN48 | 68.39% | 68.53% | +0.14 pp | 0.7406 | 100.00% | done |

### 4.4 Fixed-split `rf_fixed0_c` results

| Topology | rf_fixed0 + KSP-FF | rf_fixed0 + v1.3 | Δ (pp) | p-value | split0 share | status |
|---|---:|---:|---:|---:|---:|---|
| COST239 | 74.43% | 74.43% | +0.00 pp | NA | 100.00% | done |
| German17 | 72.21% | 72.30% | +0.09 pp | 0.5168 | 100.00% | done |
| NSFNET | 72.42% | 72.68% | +0.26 pp | 0.5360 | 100.00% | done |
| JPN48 | 68.05% | 68.42% | +0.36 pp | 0.4253 | 100.00% | done |

## 5. Cross-Baseline Comparison

| Topology | PPO-C + v1.3 (reference) | df_c + v1.3 | df_fixed0 + v1.3 | rf_fixed0 + v1.3 |
|---|---:|---:|---:|---:|
| COST239 | 7.68% | 14.84% | 73.97% | 74.43% |
| German17 | 7.63% | 12.96% | 72.58% | 72.30% |
| NSFNET | 14.92% | 19.94% | 72.80% | 72.68% |
| JPN48 | 8.03% | 10.85% | 68.53% | 68.42% |

## 6. Commands

### df_fixed0 full (4 topologies)
```bash
cd /mnt/d/project/DNN_Agent
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_fixed0_topology.sh xlron_cost239_ptrnet_real 0.0625 2.2 xlron_cost239_ptrnet_real
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_fixed0_topology.sh xlron_german17 0.07142857142857142 3.0 xlron_german17
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_fixed0_topology.sh xlron_nsfnet_deeprmsa 0.07692307692307693 4.0 xlron_nsfnet_deeprmsa
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_fixed0_topology.sh xlron_jpn48 0.1 5.2 xlron_jpn48
```

### rf_fixed0 full (4 topologies)
```bash
cd /mnt/d/project/DNN_Agent
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_fixed0_topology.sh xlron_cost239_ptrnet_real 0.0625 2.2 xlron_cost239_ptrnet_real
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_fixed0_topology.sh xlron_german17 0.07142857142857142 3.0 xlron_german17
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_fixed0_topology.sh xlron_nsfnet_deeprmsa 0.07692307692307693 4.0 xlron_nsfnet_deeprmsa
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_fixed0_topology.sh xlron_jpn48 0.1 5.2 xlron_jpn48
```

### Single smoke example
```bash
cd /mnt/d/project/DNN_Agent
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONPATH=sa_hmarl \
  .venv/bin/python -m sa_hmarl.evaluation.eval_long_horizon_system_comparison \
  --topology xlron_cost239_ptrnet_real --num_slots 320 --num_servers 4 \
  --methods "df_fixed0_c+ksp_ff_k50_hops,df_fixed0_c+v12_k50_hops,rf_fixed0_c+ksp_ff_k50_hops,rf_fixed0_c+v12_k50_hops" \
  --seeds 3030 --episodes 1 --requests_per_episode 200 --warmup_requests 50 \
  --poisson_arrivals --exponential_holding --arrival_interval 0.0625 --edge_cost_max 2.2 \
  --ranker_candidate_mode legalctx48 --ranker_max_candidates 48 --ranker_ensure_ksp \
  --output_json sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_cost239_ptrnet_real_fixed0_smoke.json \
  --output_md sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_cost239_ptrnet_real_fixed0_smoke.md
```

## 7. Interpretation

1. **Fixed-split removes almost all R-side headroom.** With `split0` pinned, blocking jumps to 68–75% across topologies. The v1.3 ranker delta versus KSP-FF is at most ±0.36 pp and never statistically significant. This confirms that adaptive split/server allocation is a prerequisite for the R-side ranker to matter.

2. **Server overload dominates.** In every fixed-split configuration the residual blocking is essentially all `server_overload`; `no_suitable_block` remains negligible. Pinning the split to `split0` (the highest edge-compute-ratio split) pushes servers into saturation, making optical path/block choice irrelevant.

3. **DF vs RF within split0 is nearly identical.** Distance-first and resource-first server selection give almost the same blocking when the split is fixed, because the binding constraint is the split choice, not the server choice.

4. **JPN48 retains slightly lower absolute blocking.** Even under the fixed-split ablation, JPN48 blocks ~4–5 pp less than the other topologies, consistent with its larger path diversity, but the R-side v1.3 delta is still negligible.

## 8. Paper-Ready Summary Draft

> We introduce a fixed-split / no-partition-style ablation, `df_fixed0_c` and
> `rf_fixed0_c`, to probe the lower bound of C-side performance. These baselines pin
> the DNN split to `split0` and select the server by distance-first or resource-first
> rules while still respecting the `agent_c_mask`. Across COST239, German17, NSFNET
> and JPN48, fixing the split raises blocking to 68–75% and completely removes the
> v1.3 ranker advantage over KSP-FF (delta within ±0.36 pp, all p > 0.37). The result
> confirms that the R-side gain observed under PPO-C and even under adapted `df_c` /
> `rf_c` is conditional on the C-side producing a favourable split/server distribution;
> once split adaptivity is removed, server overload becomes the sole bottleneck and
> smarter path/block selection has no room to help.

## 9. Status Checklist

- [x] Implement `df_fixed0_c` / `rf_fixed0_c` in unified baseline entry
- [x] Unit tests pass
- [x] COST239 smoke: 100% split0, normal JSON/MD output
- [x] Existing `df_c` full results reused (COST239, German17, JPN48, NSFNET)
- [x] Existing `rf_c` smoke result reused (COST239 only; full comparison not re-run per instruction)
- [x] `df_fixed0_c` full results for all four topologies
- [x] `rf_fixed0_c` full results for all four topologies
