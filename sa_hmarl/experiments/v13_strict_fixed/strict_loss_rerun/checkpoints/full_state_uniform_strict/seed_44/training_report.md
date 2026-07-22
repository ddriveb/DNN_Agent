# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_44/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 2.47% |
| Best `regret` selection score | -0.1749 |
| Test top-1 | 3.10% |
| Test top-3 | 12.07% |
| Test score-tie fractional top-1 | 51.03% |
| Test oracle tie hit rate | 51.03% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.718 |
| Test Spearman mean | 0.058 |
| Test Kendall tau-b | 0.047 |
| Test pairwise accuracy | 0.583 |
| Test model regret | 0.0284 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0035 |
| Test aggregate regret reduction | 11.03% |
| Test conditional mean relative reduction | 0.0568 |
| Test model better / equal / worse rate | 18.97% / 66.72% / 14.31% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0094 / 0.0000 / 0.0271 |
| Test PPO agreement | 5.00% |
| Test score std | 0.3602 |
| Test KL | 0.0021 |