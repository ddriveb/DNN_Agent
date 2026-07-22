# SA-HMARL v1.3 Medium Dataset Training Report (Corrected)

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- Model: MLP [128,64]
- Loss: listwise KL + SmoothL1 MSE (reg_weight=1.0, lambda_pair=0, lambda_hard=0)
- Seeds: [42, 123, 456]
- Checkpoint selection: **lowest validation model regret** (selection_metric=regret)

## Per-seed results

| Seed | Val selection score | Test model regret | Test PPO regret | Abs. improvement | Aggregate reduction | Model better rate | Oracle tie hit |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 42 | -0.0465 | 0.0682 | 0.1050 | 0.0368 | 35.04% | 26.34% | 46.22% |
| 123 | -0.0432 | 0.0696 | 0.1050 | 0.0354 | 33.71% | 27.35% | 47.76% |
| 456 | -0.0450 | 0.0791 | 0.1050 | 0.0259 | 24.65% | 23.50% | 45.02% |

## Selected checkpoint

- Seed: 42
- Path: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt`
- Selection was based on validation regret only; test metrics are reported for diagnostics.