# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 3.53% |
| Best `regret` selection score | -0.1518 |
| Test top-1 | 9.83% |
| Test top-3 | 13.79% |
| Test score-tie fractional top-1 | 49.83% |
| Test oracle tie hit rate | 49.83% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.678 |
| Test Spearman mean | -0.076 |
| Test Kendall tau-b | -0.061 |
| Test pairwise accuracy | 0.446 |
| Test model regret | 0.0312 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0008 |
| Test aggregate regret reduction | 2.37% |
| Test conditional mean relative reduction | -0.0627 |
| Test model better / equal / worse rate | 16.90% / 65.86% / 17.24% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0193 / 0.0000 / 0.0200 |
| Test PPO agreement | 10.34% |
| Test score std | 0.9241 |
| Test KL | 0.0033 |