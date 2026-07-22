# SA-HMARL v1.3 Pilot Training Report

**Dataset:** `sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5`  
**Splits:** train=1,285 groups, val=429 groups, test=602 groups  
**Feature dim:** 25, **Candidate budget:** K_prop=30  
**Model:** MLP [128, 64]

## Pilot runs

Two training runs were performed to probe the effect of the auxiliary losses.

### Run 1 — default loss

| Hyperparameter | Value |
|---|---|
| Optimizer | Adam, lr=3e-4, weight_decay=1e-5 |
| Epochs | 80 |
| Batch size | 64 |
| `tau_label` / `tau_model` | 0.5 / 1.0 |
| `reg_weight` | 0.1 |
| `lambda_pair` / `pair_margin` | 0.5 / 0.1 |
| `lambda_hard` / `alpha_hard` | 1.0 / 2.0 |
| Patience | 15 epochs |

| Metric | Value |
|---|---:|
| Best val top-1 | 9.79% |
| Test top-1 | 1.66% |
| Test top-3 | 7.81% |
| Test tie-aware top-1 | 36.71% |
| Test NDCG@3 | 0.613 |
| Test Spearman mean | 0.056 |
| Test model regret | 0.0529 |
| Test PPO regret | 0.0547 |
| Test PPO agreement | 1.83% |
| Test KL | 0.0140 |

**Checkpoint:** `sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_pilot/ranking_model.pt`

### Run 2 — simpler loss

Same as Run 1 except:

| Hyperparameter | Value |
|---|---|
| `reg_weight` | 1.0 |
| `lambda_pair` | 0.0 |
| `lambda_hard` | 0.0 |

| Metric | Value |
|---|---:|
| Best val top-1 | 8.39% |
| Test top-1 | 5.15% |
| Test top-3 | 14.12% |
| Test tie-aware top-1 | 35.55% |
| Test NDCG@3 | 0.586 |
| Test Spearman mean | -0.037 |
| Test model regret | 0.0421 |
| Test PPO regret | 0.0547 |
| Test PPO agreement | 7.31% |
| Test KL | 0.0137 |

**Checkpoint:** `sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_pilot_v2/ranking_model.pt`

## Observations

- Both runs achieve very low absolute top-1 accuracy (<10%), consistent with the small pilot dataset size.
- Removing the pairwise and hard-negative terms and increasing the regression weight (Run 2) improves test top-1 from 1.66% to 5.15% and lowers model regret from 0.053 to 0.042. This suggests the auxiliary ranking losses are hurting generalisation on limited data.
- Test tie-aware top-1 is ≈36% because many candidates have nearly identical returns; exact top-1 matching is a strict metric.
- NDCG@3 ≈0.6 indicates the model is learning some useful ranking signal.

## Closed-loop fair comparison

When Run 2 checkpoint is used inside the strict PPO-R Top-30 pool (see `FAIR_COMPARISON_FULL.md`):

- Seed 4001: 5.10% blocking vs PPO-R Top-1 5.70% (improvement).
- Seed 4002: 12.75% blocking vs PPO-R Top-1 11.30% (regression).
- Average: 8.92% ± 3.82% vs 8.50% ± 2.80%.

The ranker is competitive but unstable across seeds, again indicating insufficient training data.

## Recommendations

1. **Scale the dataset.** The primary blocker is data volume, not architecture.
2. **Try a larger model** (e.g. DeepSet / Set Transformer) once more data is available.
3. **Tune temperature and loss weights** on the full dataset; the pairwise/hard-negative losses may become helpful with more examples.
4. **Consider data augmentation** by generating multiple common-future traces per request (different RNG seeds) to multiply effective training groups.
