# SA-HMARL v1.3 Post-Decision Ranker Upgrade — Implementation Report

This report documents the Stage 0–5 implementation for moving the R-side ranker from a pre-action state-action scorer to an explicit post-decision / afterstate opportunity-cost ranker while keeping the frozen PPO-C policy and the fixed-C conditional R-side boundary intact.

## 1. Stage 0 — Diagnostic Audit

### Deliverables
- `sa_hmarl/sa_hmarl/evaluation/v13_pds_stage0_diagnostic.py`
- `sa_hmarl/experiments/v13_pds_stage0_diagnostic.json`
- `sa_hmarl/experiments/v13_pds_stage0_diagnostic.md`

### What it audits
1. **Common-random-number fairness** of counterfactual rollouts.
2. **Afterstate variance** across successful R candidates under fixed `(s, a_C)`.
3. **Horizon comparison** (H=5, 12, 20) on the same states and candidates.
4. **Mutually-exclusive failure-cause counters** (optical, overload, other).
5. **Candidate recall** vs an all-legal H=5 oracle.

### Key findings from the first run
- Server post-load is identical across successful R candidates under fixed `a_C`; afterstate differences are purely optical (fragmentation, occupied slot hops, path conflict).
- Return ranges are nonzero at all horizons, but the top-1/top-2 gap is tiny at H=5.
- H=5 and H=12/H=20 labels are correlated (Spearman ≈ 0.51 for H=12) but not identical, so horizon matters.
- The current training candidate pool (`ppo_r_topk_only`, top-30) recalls the all-legal H=5 oracle top-1 only 66.7% of the time (top-5 73.3%), confirming candidate-set expansion is a bottleneck.
- In the moderate-load states sampled, future failures were not observed; overload dynamics require high-load snapshots.

## 2. Stage 1 — Post-Decision Feature Schema (`poststate_v1`)

### Deliverables
- `sa_hmarl/sa_hmarl/evaluation/r_poststate_features.py` (existing module now wired into the training/inference pipeline)
- Updates to `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_dataset.py`
- Updates to `sa_hmarl/sa_hmarl/agents/r_ranker_policy.py`
- Updates to `sa_hmarl/sa_hmarl/evaluation/r_ranker_features.py`

### What changed
- Added `--feature_schema {v1, poststate_v1}` to the dataset generator.
- `generate_dataset` sets `args._feature_names` to the 41-d `POSTSTATE_V1_FEATURE_NAMES` when `poststate_v1` is selected.
- `_build_candidate_feature(...)` dispatches between `_r_feature_vector` (25-d) and `build_poststate_v1_feature` (41-d).
- `_concat_groups` now receives the correct `feature_dim` so empty splits do not assume the old 25-d shape.
- The online `CounterfactualRRankerPolicy` dispatches to `build_poststate_v1_feature_batch` when `feature_names == POSTSTATE_V1_FEATURE_NAMES`.
- `r_ranker_features.build_r_ranker_feature_batch` also dispatches to the post-state batch builder for schema-robustness.

### Afterstate features added
- `path_lfb_after`, `path_free_ratio_after`
- `global_lfb_after`, `global_lfb_ratio_after`
- `frag_after`, `delta_frag`
- `free_block_count_after`, `free_block_count_delta`
- `min_edge_lfb_margin_after`, `min_edge_free_ratio_after`, `bottleneck_edge_util_after`
- `occupied_slot_hops`
- `path_conflict_after`
- `server_util_context`, `server_margin_context`, `holding_time_norm`

These are computed analytically without mutating the environment, so training and online inference see exactly the same feature distribution.

## 3. Stage 2 — Label Refactor with Exclusive Causes and Longer Horizon

### Deliverables
- Updated `_rollout_future` and `_compute_return` in `generate_r_counterfactual_ranking_dataset.py`

### What changed
- `_rollout_future` now returns mutually-exclusive cause counts:
  - `no_suitable_block` (optical)
  - `server_overload`
  - `allocation_failed`
  - `other` = blocked − optical − overload − allocation_failed
- Future delay and FS sums are discounted by `gamma**offset` when `gamma < 1`.
- `_compute_return` supports `--return_mode {v1, exclusive_cause}`.
  - `v1`: legacy formula (double-counts NSB/overload).
  - `exclusive_cause`: partitions the penalty into optical/overload/other with configurable per-cause weights.
- New arguments:
  - `--return_future_optical_coef`
  - `--return_future_overload_coef`
  - `--return_future_other_coef`
- The `viability_phi_coef` post-decision probe is skipped when its coefficient is zero, removing a per-candidate `build_agent_c_observation` call and speeding up dataset generation.

## 4. Stage 3 — Explicit Post-Decision Candidate Mode (`poststate_anchor48`)

### Deliverables
- `poststate_anchor48` candidate selection in dataset generator and online policy wrapper.

### What changed
- Added `_select_candidate_actions_poststate_anchor48` to the dataset generator.
- Added `CounterfactualRRankerPolicy._build_poststate_anchor48` for online inference.
- The 48-slot candidate pool mixes:
  1. PPO-R top-K proposals.
  2. KSP-FF anchors (plain + highest modulation).
  3. Pre-action resource anchors (shortest path, lowest required FS).
  4. Explicit afterstate anchors:
     - min post-allocation fragmentation
     - max bottleneck LFB margin
     - min occupied slot hops
     - min bottleneck edge utilization
     - max path LFB
  5. Highest-logit legal fillers.
- To keep candidate construction fast, afterstate features are computed only on a reduced pool (PPO top-K + structural anchors); the afterstate anchors are selected from that pool.
- `candidate_mode` choices updated to include `poststate_anchor48` in the dataset generator and the policy wrapper.

## 5. Stage 4 — Training / Evaluation Harness

### Deliverables
- `sa_hmarl/sa_hmarl/evaluation/eval_v13_fair_comparison.py`
- Existing `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py` (verified compatible)

### What changed
- New closed-loop evaluation script compares arbitrary ranker checkpoints (loaded via `CounterfactualRRankerPolicy`) against PPO-R, KSP-FF plain, and KSP-FF highest on a user-specified seed set.
- Ranker specs use `name=path[:candidate_mode]` syntax, allowing the same checkpoint to be evaluated under different candidate modes (e.g., original `ppo_r_topk_only` vs new `poststate_anchor48`).
- Reports per-seed and aggregate blocking, overload, NSB, delay, FS, decision latency, and PPO-agreement.
- The training script reads `feature_names` and `feature_dim` from dataset metadata, so it trains the new 41-d poststate_v1 ranker without modification.

## 6. Evaluator Bug Fix

The first version of `eval_v13_fair_comparison.py` had two protocol mistakes:

1. **Warmup was ignored**: the script generated only `requests_per_episode` requests and counted every request, even though it reported a `warmup_requests` parameter.
2. **C-side path support was wrong**: the script created the environment with `k=50`, so the frozen PPO-C policy saw 50 candidate paths at inference time even though it was trained with K=5.

Both errors inflated blocking and changed the result scale. The evaluator was fixed to:
- generate `warmup_requests + requests_per_episode` total requests and exclude the warmup window from metrics;
- expose `--k_paths_c` (default 5) and `--k_paths_r` (default 50) and set `env.k`, `env.path_sort_strategy`, and `env.block_sort_strategy` separately before building the C and R observations.

The corrected script is `sa_hmarl/sa_hmarl/evaluation/eval_v13_fair_comparison.py`.

## 7. Stage 4 Results — Fair Closed-Loop Comparison (fixed protocol)

### Deliverables
- `sa_hmarl/experiments/v13_fair_comparison_fixed_5k.json`
- `sa_hmarl/experiments/v13_fair_comparison_fixed_5k.md`
- Updated `sa_hmarl/experiments/ABLATION_RESULTS.md`
- Updated `sa_hmarl/experiments/PAPER_INTERPRETATION.md`

### Experimental protocol
- Network: COST239 (`xlron_cost239_ptrnet_real`), 320 slots, 4 servers.
- C/R checkpoints frozen: `agent_c_cost239_r_feasibility_safe_last.pt`, `agent_r_mixed.pt`.
- Seeds: `3030, 4040, 5050, 6060, 7070`.
- 1000 warm-up requests, 5000 evaluation requests per seed.
- C-side path support K=5; R-side path support K=50; `path_sort=hops`, `block_sort=start_asc`.
- **Both rankers evaluated with the same candidate mode `ppo_r_topk_only`** to isolate the pure feature effect.

### Results

| Mode | Blocking % | Decision ms | Notes |
|---|---:|---:|---|
| ranker_v13 | 6.01% ± 0.75% | ~14 ms | 25-d pre-action ranker |
| ppo_r | 6.85% ± 1.72% | ~10 ms | Frozen PPO-R baseline |
| ksp_ff_highest | 7.03% ± 0.86% | ~10 ms | KSP-FF highest-modulation first |
| ksp_ff_plain | 7.88% ± 1.35% | ~10 ms | KSP-FF lowest-modulation first |
| ranker_poststate | 7.42% ± 0.63% | ~43 ms | 41-d poststate_v1 ranker |

Per-seed details are in `v13_fair_comparison_fixed_5k.json`.

### Interpretation
- The fixed protocol restores the historical ~6–8% blocking scale (v1.3 baseline ≈ 6.01%, close to the historical 6.29%).
- `ranker_v13` remains the best method, beating PPO-R and KSP-FF.
- `ranker_poststate` is worse than `ranker_v13` on **every seed** and ~3× slower, even though both use the same candidate pool.
- The failure mix is still dominated by `server_overload`; NSB is small but nonzero on some seeds under the longer run.
- Because the only difference between `ranker_v13` and `ranker_poststate` is the feature vector, this is a clean negative result for the current H=5, 240-group poststate feature upgrade.

## 8. Known Limitations & Next Steps

- **Training signal is weak**: test Spearman ≈ 0.03, top-1 accuracy ≈ 10%. The H=5 label does not strongly distinguish candidates; longer horizons (H=12/20) showed higher overload variance in Stage 0 and may be needed.
- **Dataset size**: only 240 train groups. The 41-d poststate feature space may require a larger corpus.
- **Label mode**: the current checkpoint was trained with `return_mode=v1`, not the mutually-exclusive `exclusive_cause` labels that Stage 2 implemented. Re-training with `exclusive_cause` and a non-zero overload weight is the next label experiment.
- **Speed**: with `ppo_r_topk_only`, poststate feature computation still costs ~43 ms/decision vs ~14 ms for v1.3. With `poststate_anchor48` it is substantially slower; that mode should be restricted to offline dataset generation unless afterstate computation is vectorized.
- **C-side bottleneck**: the most promising remaining direction is still to let the ranker’s signal feed back into C-side/server selection, or to train C with overload-aware reward, rather than expecting an R-side reranker to compensate for a fixed C decision.
