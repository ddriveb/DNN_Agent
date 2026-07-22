# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/checkpoints/full_state_stratified_depth/seed_42/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 2.30% |
| Best `regret` selection score | -0.1602 |
| Test top-1 | 0.86% |
| Test top-3 | 6.21% |
| Test score-tie fractional top-1 | 50.00% |
| Test oracle tie hit rate | 50.17% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.682 |
| Test Spearman mean | -0.060 |
| Test Kendall tau-b | -0.048 |
| Test pairwise accuracy | 0.454 |
| Test model regret | 0.0306 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0013 |
| Test aggregate regret reduction | 4.10% |
| Test conditional mean relative reduction | -0.0256 |
| Test model better / equal / worse rate | 16.21% / 66.03% / 17.76% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0166 / 0.0000 / 0.0194 |
| Test PPO agreement | 0.52% |
| Test score std | 0.1371 |
| Test KL | 0.0017 |