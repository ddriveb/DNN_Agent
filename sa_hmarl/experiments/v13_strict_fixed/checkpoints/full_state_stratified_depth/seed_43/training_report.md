# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/checkpoints/full_state_stratified_depth/seed_43/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 6.71% |
| Best `regret` selection score | -0.1704 |
| Test top-1 | 25.00% |
| Test top-3 | 35.86% |
| Test score-tie fractional top-1 | 50.00% |
| Test oracle tie hit rate | 50.17% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.685 |
| Test Spearman mean | -0.081 |
| Test Kendall tau-b | -0.063 |
| Test pairwise accuracy | 0.400 |
| Test model regret | 0.0306 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0014 |
| Test aggregate regret reduction | 4.30% |
| Test conditional mean relative reduction | -0.0431 |
| Test model better / equal / worse rate | 13.62% / 71.55% / 14.83% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0100 / 0.0000 / 0.0103 |
| Test PPO agreement | 27.07% |
| Test score std | 0.1904 |
| Test KL | 0.0017 |