# SA-HMARL v1.3 Implementation Audit

**Protocol:** PPO-R Proposal-Supported RMSA Domain-Constrained Common-Future Counterfactual Reranking  
**Topology:** COST239, 320 slots, 4 servers  
**Hyper-parameters:** K_C=5, K_path=50, K_prop=30, H=5, γ=1.0  
**Path sort:** hops; **Block sort:** start_asc  
**Date:** 2026-07-13

## Files

| File | Role | Status |
|---|---|---|
| `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py` | Dataset generator | Implemented, smoke-tested |
| `sa_hmarl/sa_hmarl/evaluation/merge_r_counterfactual_ranking_shards.py` | Shard → train/val/test merger | Implemented |
| `sa_hmarl/sa_hmarl/evaluation/v13_kpath50_hops_counterfactual_smoke.py` | Nine-invariant smoke test | PASS |
| `sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_cost239_kpath50_hops_fair.py` | Fair closed-loop evaluator | Implemented, quick-run validated |
| `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py` | Ranker training backend | Reused, awaiting pilot data |

## Protocol enforcement

1. **Strict K switching.** The generator sets `env.k = K_C` before every `build_agent_c_observation`, and `env.k = K_path` before every `build_agent_r_observation` and `env.step`. The evaluator uses the same discipline.
2. **PPO-R Top-K_prop candidates only.** Candidates are obtained by a stable argsort over PPO-R policy logits on the legal mask. No KSP anchors, random fillers, or diversity actions are added.
3. **Common-future trace.** For every candidate branch the generator deep-copies the pre-R-decision environment snapshot and rolls out the *same* future request sequence (H=5). A per-request trace hash is recorded and validated to be identical across candidates.
4. **25-d v1 features.** The feature vector is `_r_feature_vector(..., feature_names=FEATURE_NAMES)` from `generate_r_post_decision_dataset`, i.e. the 25-d pre-decision schema.
5. **Legacy v1.3 label.** `-3·B_t - 4·ΣB - 3·ΣNSB - 0.03·D - 0.05·FS`. No path penalty, FS penalty, or fragmentation proxy.
6. **No old checkpoint reuse.** The old `r_counterfactual_ranking_postdec_topk_k50_s100` checkpoint is only used as a comparison baseline in the fair evaluator.

## Smoke-test invariants (all PASS)

| # | Invariant | Result |
|---|---|---|
| 1 | Metadata K_C/K_path/K_prop/H correct | PASS |
| 2 | Feature dimension == 25 | PASS |
| 3 | At least one non-empty group | PASS |
| 4 | Mask length == K_prop (candidate-budget padding) | PASS |
| 5 | Per-group candidates ∈ [1, K_prop] | PASS |
| 6 | Action IDs inside the K_path flat action space | PASS |
| 7 | Path diversity observed (path_idx ≥ 5, action_id ≥ 200) | PASS |
| 8 | Provenance strictly PPO-R Top-K (bit 1) | PASS |
| 9 | Finite features/returns | PASS |
| 10 | Trace-hash present and consistent | PASS |
| 11 | Unique group IDs | PASS |

## Known limitations / bounded claims

- The generator uses `copy.deepcopy(env)` per candidate branch. This is correct but slow (~1.3 s/request on the test CPU). Full-scale generation is expected to run overnight.
- The R mask length is dynamic: `num_paths × num_mods × max_blocks`, where `num_paths ≤ K_path`. The smoke test was updated to check `mask_len == K_prop` (listwise group padding) rather than the full flat space.
- The fair evaluator restricts every ranker (old and new) to the same PPO-R Top-30 candidate set. This bounds paper claims to: *inside PPO-R proposal support, common-future counterfactual supervision improves ranking*.

## Pilot dataset

- Generated 4 shards × 1000 requests each (2 train / 1 val / 1 test, 250 warmup per shard).
- Merged to `train.npz`/`val.npz`/`test.npz`.
- Dataset diagnostics: see `DATASET_REPORT.md`.

## Pilot training

| Run | Hyperparameters | Val top-1 | Test top-1 |
|---|---|---:|---:|
| Default | listwise KL + SmoothL1 MSE + pairwise + hard-negative | 9.79% | 1.66% |
| Simpler | `--lambda_pair 0.0 --lambda_hard 0.0 --reg_weight 1.0` | 8.39% | 5.15% |

Both values are low, confirming the pilot dataset is too small to learn a robust ranker.

## Fair closed-loop comparison (pilot)

Two seeds × 2500 requests (500 warmup, 2000 evaluated):

| Mode | Blocking % | Overload % | NSB % | PPO agree | Top-1 in cand |
|---|---:|---:|---:|---:|---:|
| ppo_r_top1 | 8.50% ± 2.80% | 8.20% | 0.30% | 100.00% | 0.00% |
| ksp_ff_plain | 10.33% ± 2.17% | 9.57% | 0.75% | 29.73% | 0.00% |
| ksp_ff_highest | 6.65% ± 1.25% | 6.62% | 0.03% | 7.15% | 0.00% |
| ranker_old_v13 | 6.68% ± 0.87% | 6.58% | 0.10% | 10.20% | 93.32% |
| ranker_new_v13 | 8.92% ± 3.82% | 8.62% | 0.30% | 14.00% | 91.07% |

The new v1.3 ranker is competitive on seed 4001 (5.10% vs PPO-R 5.70%) but unstable on seed 4002 (12.75%). The old ranker remains the strongest under the strict pool.

## Completed / next steps

- ✅ Generator, merger, smoke test, fair evaluator all functional.
- ✅ Pilot dataset, merge, training, and fair comparison complete.
- ⏳ Full-scale dataset generation (e.g. 16 train / 4 val / 4 test shards × 5000 requests) to stabilise the new ranker.
- ⏳ Finalise `FINAL_METHOD_AND_RESULT.md` after full-scale run.
