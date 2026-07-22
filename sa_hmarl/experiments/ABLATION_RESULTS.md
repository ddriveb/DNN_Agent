# Ablation Results: v1.3 vs poststate_v1 / poststate_anchor48

## Datasets

| Dataset | Feature schema | Candidate mode | Horizon | Gamma | Overload coef | Max cand | Groups (train/val/test) |
|---|---|---|---|---|---:|---:|---|
| `r_counterfactual_ranking_postdec_topk_k50_s100` (baseline) | v1 (25-d) | `ppo_r_topk_only` | H=5 | 1.0 | 0.0 | 30 | pre-trained |
| `r_poststate_v1_h5_anchor48_k50` | poststate_v1 (41-d) | `poststate_anchor48` | H=5 | 0.99 | 4.0 | 24 | 240 / 80 / 80 |

## Training metrics

| Model | Val top-1 | Val Spearman | Test top-1 | Test Spearman | PPO regret | Selection metric |
|---|---:|---:|---:|---:|---:|---|
| v1.3 baseline | — | — | — | — | — | top1 |
| poststate_v1 H5 anchor48 | 22.50% | 0.184 | 10.00% | 0.029 | 0.0336 | top1 |

Training was run for up to 150 epochs with early stopping at epoch 21. The low test top-1 (10%) and near-zero test Spearman indicate that 240 training groups is insufficient for the 41-d poststate_v1 model to generalize reliably.

## Closed-loop fair comparison (fixed evaluator protocol)

The first version of the evaluator ignored `warmup_requests` and used `k=50` for the C-side observation. After fixing both bugs (C-K=5, R-K=50, warmup excluded), the historical ~6–8% scale is restored. The corrected aggregate numbers are from `v13_fair_comparison_fixed_5k.json` (5 seeds, 5000 evaluated requests/seed, 1000 warmup, same `ppo_r_topk_only` candidate mode for both rankers):

| Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---:|---:|---:|---:|
| ranker_v13 | 6.01% ± 0.75% | 5.98% | 0.02% | 14.31 |
| ppo_r | 6.85% ± 1.72% | 6.68% | 0.17% | 9.86 |
| ksp_ff_highest | 7.03% ± 0.86% | 7.03% | 0.00% | 9.97 |
| ksp_ff_plain | 7.88% ± 1.35% | 7.82% | 0.06% | 9.77 |
| ranker_poststate | 7.42% ± 0.63% | 7.39% | 0.04% | 42.84 |

Per-seed:

| Seed | ranker_v13 | ranker_poststate | Δ (pp) |
|---|---:|---:|---:|
| 3030 | 6.48% | 7.26% | +0.78 |
| 4040 | 4.80% | 7.44% | +2.64 |
| 5050 | 5.46% | 6.52% | +1.06 |
| 6060 | 6.60% | 7.42% | +0.82 |
| 7070 | 6.70% | 8.48% | +1.78 |

## Takeaways

1. **The earlier ~21% numbers were an evaluator artifact**, not a real change in the ranker. With the fixed protocol, v1.3 returns to ~6%, matching historical results.
2. **The poststate_v1 feature upgrade is still negative.** Even with the same candidate pool (`ppo_r_topk_only`), the 41-d poststate ranker is worse than the 25-d v1.3 ranker on every seed.
3. **The failure mode is still overload-dominated**, with a small NSB component on seed 6060. The R-ranker cannot fix C-side/server-selection-driven overload.
4. **Latency remains a concern.** Poststate feature computation costs ~3× more than v1.3 even with the fast candidate mode.

## Why the poststate checkpoint may still fail

- **No training signal:** test Spearman ≈ 0.03, top-1 ≈ 10% on 240 train groups.
- **Label does not penalize overload:** the checkpoint was trained with `return_mode=v1` and H=5; Stage 0 showed future overload variance only appears at longer horizons or higher load snapshots.
- **Feature/candidate entanglement removed:** the corrected comparison evaluates both checkpoints with `ppo_r_topk_only`, so the remaining gap is purely the 41-d feature vector and the model trained on it.

## Recommended next ablations

1. **Re-train with `exclusive_cause` labels** and a non-zero overload coefficient, keeping H=5 and `ppo_r_topk_only`. This tests whether the label, not the features, is the bottleneck.
2. **If (1) still fails, try a larger dataset** (≥1000 groups) before concluding that explicit afterstate features do not help.
3. **Only then extend horizon to H=12/20** with util-biased sampling, and only if the new labels show candidate-dependent future overload variance.
