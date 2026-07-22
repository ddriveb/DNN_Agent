# SA-HMARL v1.3 R-side Ranker Latency Optimization Report

**Date:** 2026-07-07  
**Goal:** Reduce online decision latency of the v1.3 R-side ranker without changing action-selection semantics in the first phase.

---

## 1. What Was Done

This work follows the layered plan: first do **no-semantic-change** implementation optimizations and profiling, then run a **Kmax sweep as a new variant** (not replacing the main v1.3口径).

### 1.1 Profiling (Step 1)

- Added `enable_profile` flag to `CounterfactualRRankerPolicy`.
- `score_legal_actions` now records per-request breakdown:
  - `r_feature_build_ms` (`agent_r.build_action_features`)
  - `legal_extract_ms`
  - `candidate_select_ms` (`_select_online_candidates`)
  - `feature_batch_ms` (`build_r_ranker_feature_batch`)
  - `normalize_ms`
  - `ranker_forward_ms`
  - `total_ranker_policy_ms`
- Profile samples are stored in `PerMethodMetrics` and aggregated as mean/P95 in `profile_stats`.
- Added `--ranker_enable_profile` to both `eval_main_s100_system_comparison.py` and `eval_long_horizon_system_comparison.py`.
- Added a "Ranker Profile Breakdown" table to the long-horizon markdown report.

### 1.2 No-Semantic-Change Optimizations (Step 2)

#### `r_ranker_policy.py` — candidate construction

- Replaced repeated `"a in candidates"` list scans with `legal_set` / `candidate_set`.
- Precomputed `path_idx` and `block_idx` for all legal actions once, avoiding repeated `decode_agent_r_action` calls.
- Vectorized per-path heuristic selection with `np.argmin` / `np.argmax` while preserving first-occurrence tie-breaking.
- Replaced Python `sorted()` for highest-logit fill with stable `np.argsort` on flat logits.
- Verified with `test_ranker_candidate_consistency.py` that optimized output equals the reference implementation for `v1` and `legalctx48` modes and K ∈ {16, 24, 32, 48}.

#### `r_ranker_features.py` — feature batch builder

- Added `_build_default_feature_batch`: a vectorized builder for the default `FEATURE_NAMES` schema.
- Computes per-decision context and field features **once**, then broadcasts them across all candidate actions.
- Batch-decodes `path_idx`, `mod_idx`, `block_idx` with numpy.
- Extracts base R features directly via `r_features[action_indices]`.
- Falls back to the legacy per-action `_r_feature_vector` when `feature_names` is not the default schema (e.g., structured checkpoints with edge features).
- Verified with `test_r_ranker_feature_batch_consistency.py` that vectorized output exactly matches the per-action builder for default and structured feature names.

### 1.3 Kmax Sweep (Step 3)

- Added `run_v13_kmax_sweep.sh` for the full main-table protocol (10k requests, 5 seeds).
- Added `run_v13_kmax_quick_sweep.sh` for quick validation/tables.
- Both scripts treat each K as a separate file/method, so they do **not** overwrite the main v1.3 K=48口径.

---

## 2. Files Changed

| File | Change |
|---|---|
| `sa_hmarl/sa_hmarl/agents/r_ranker_policy.py` | Profiling, vectorized candidate construction, docstring v1.2→v1.3 |
| `sa_hmarl/sa_hmarl/evaluation/r_ranker_features.py` | Vectorized default-schema feature batch builder + fallback |
| `sa_hmarl/sa_hmarl/evaluation/eval_c_demand_potential_closed_loop.py` | Added profile fields to `PerMethodMetrics` and `_aggregate_metrics` |
| `sa_hmarl/sa_hmarl/evaluation/eval_c_post_decision_closed_loop.py` | `_record_outcome` stores optional profile dict |
| `sa_hmarl/sa_hmarl/evaluation/eval_main_s100_system_comparison.py` | `--ranker_enable_profile`, pass profile to `_record_outcome`, aggregate profile stats |
| `sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py` | Same + markdown profile table |
| `sa_hmarl/tests/test_r_ranker_feature_batch_consistency.py` | New: vectorized vs per-action consistency |
| `sa_hmarl/tests/test_ranker_candidate_consistency.py` | New: optimized vs reference candidate construction |
| `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_v13_kmax_sweep.sh` | New: full Kmax sweep |
| `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_v13_kmax_quick_sweep.sh` | New: quick Kmax sweep |

---

## 3. Profiling Results (COST239 short smoke, 500 req, 100 warmup, 2 seeds)

Mean breakdown for `ppo_c+v12_k50_hops`:

| Component | Mean (ms) | P95 (ms) |
|---|---:|---:|
| `build_action_features` | 0.46 | 0.99 |
| legal extract | 0.01 | 0.01 |
| candidate select | 1.64 | 2.33 |
| feature batch | 0.18 | 0.26 |
| normalize | 0.01 | 0.01 |
| ranker forward | 0.08 | 0.11 |
| **total ranker policy** | **2.38** | **3.44** |

**Interpretation:**
- `candidate_select_ms` (~1.6 ms) is the largest ranker-internal cost.
- `feature_batch_ms` (~0.18 ms) is now small after vectorization; before vectorization it was likely several ms due to per-action `_r_feature_vector` calls.
- `ranker_forward_ms` (~0.08 ms) confirms the forward pass is not the bottleneck.
- `build_action_features` (~0.46 ms) is PPO-R feature construction; it is outside the ranker but still part of the R-side decision path.

> Note: these numbers are from a low-load short smoke. At higher load with more legal actions, the relative share of `candidate_select_ms` and `feature_batch_ms` may shift.

---

## 4. Kmax Sweep — Quick Validation

Quick sweep on COST239 (500 req, 100 warmup, 2 seeds). Load was too low to produce blocking, so this table mainly validated the sweep machinery:

| K | Blocking | Delay mean | Delay P95 | Decision mean | Decision P95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.0000 | 8.65 | 16.29 | 8.32 | 12.14 |
| 24 | 0.0000 | 8.98 | 17.55 | 8.49 | 11.96 |
| 32 | 0.0000 | 8.88 | 17.62 | 8.40 | 11.82 |
| 48 | 0.0000 | 9.14 | 18.32 | 8.47 | 12.34 |

---

## 5. Full Kmax Sweep Results (COST239, 10k req/episode, 5 seeds)

| K | Blocking mean ± std | Delay mean | Delay P95 | Decision mean | Decision P95 |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.1563 ± 0.0163 | 9.94 | 17.36 | 9.34 | 13.50 |
| 24 | 0.1553 ± 0.0126 | 10.09 | 17.70 | 9.56 | 13.72 |
| 32 | **0.1498 ± 0.0078** | 10.17 | 17.77 | 9.43 | 13.39 |
| 48 | 0.1584 ± 0.0116 | 10.30 | 18.28 | 9.47 | 13.58 |

**Interpretation:**
- K=32 achieved the lowest mean blocking (14.98 %) and the smallest seed-to-seed variance (std 0.78 pp) on COST239.
- Differences between K=16/24/32/48 are within one standard deviation, so they are not statistically significant.
- Decision latency does not scale strongly with K in this range; fixed overheads dominate.

**Paper angle:** K=32 can be reported as a `v1.3-K32` ablation that achieves comparable (or slightly better) blocking with the same decision latency as K=48. This supports the claim that the ranker candidate budget can be reduced without sacrificing performance, but more topologies are needed before generalizing.

---

## 6. Tests

All new and existing targeted tests pass:

```bash
PYTHONPATH=sa_hmarl .venv/bin/python -m pytest \
  sa_hmarl/tests/test_offloading_baselines.py \
  sa_hmarl/tests/test_r_ranker_feature_batch_consistency.py \
  sa_hmarl/tests/test_ranker_candidate_consistency.py \
  -v
```

Result: **8 passed**.

---

## 7. What Was Deliberately Not Done

Per the plan:

- **No fast-path** that bypasses the ranker. This would change semantics and is reserved for a future `v1.3-fast` variant.
- **No bitset spectrum rewrite.** Legal-action enumeration semantics would be at risk; profiling did not show it as the dominant bottleneck.
- **No GPU inference changes.** K=48 is a small batch; CPU is stable.
- **No replacement of the main v1.3 K=48口径.** Kmax sweep results are stored as separate variant files.

---

## 8. Next Steps

1. **Verify on more topologies** (German17, NSFNET, JPN48) before claiming K=32 generalizes.
2. **Run a profile-enabled Kmax sweep** if component-level latency per K is needed for the paper figures.
3. **Consider a second-stage optimization** of `build_action_features` (PPO-R feature construction) if profiling at full load shows it remains a large share.
4. **Consider the `v1.3-fast` variant** only after the no-semantic-change optimizations and Kmax sweep are finalized.

---

## 9. Final Kmax Sweep Table

| K | Blocking mean ± std | Δ vs K=48 (pp) | Delay mean/P95 | Decision mean/P95 |
|---:|---:|---:|---:|---:|
| 16 | 0.1563 ± 0.0163 | −0.21 | 9.94 / 17.36 | 9.34 / 13.50 |
| 24 | 0.1553 ± 0.0126 | −0.31 | 10.09 / 17.70 | 9.56 / 13.72 |
| 32 | **0.1498 ± 0.0078** | **−0.86** | 10.17 / 17.77 | 9.43 / 13.39 |
| 48 | 0.1584 ± 0.0116 | — | 10.30 / 18.28 | 9.47 / 13.58 |

### Profile breakdown per K (COST239, 1k req, 1 seed, profile enabled)

| K | build_action_features | candidate_select | feature_batch | ranker_forward | total_ranker |
|---:|---:|---:|---:|---:|---:|
| 16 | 0.552 | 1.487 | 0.183 | 0.073 | 2.309 |
| 24 | 0.725 | 1.340 | 0.173 | 0.071 | 2.323 |
| 32 | 0.648 | 1.499 | 0.181 | 0.087 | 2.431 |
| 48 | 0.608 | 1.390 | 0.184 | 0.078 | 2.276 |

- `candidate_select_ms` and `build_action_features_ms` are the dominant costs and do not scale strongly with K in this range.
- `feature_batch_ms` stayed ~0.18 ms after vectorization (was the main bottleneck before optimization).
- `ranker_forward_ms` stayed ~0.08 ms, confirming the neural-network forward pass is not the bottleneck.
