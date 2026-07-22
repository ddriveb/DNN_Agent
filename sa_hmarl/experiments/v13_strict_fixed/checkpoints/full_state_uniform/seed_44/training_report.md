# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/checkpoints/full_state_uniform/seed_44/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 4.24% |
| Best `regret` selection score | -0.1650 |
| Test top-1 | 7.93% |
| Test top-3 | 12.07% |
| Test score-tie fractional top-1 | 50.34% |
| Test oracle tie hit rate | 50.52% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.692 |
| Test Spearman mean | -0.080 |
| Test Kendall tau-b | -0.063 |
| Test pairwise accuracy | 0.417 |
| Test model regret | 0.0294 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0025 |
| Test aggregate regret reduction | 7.93% |
| Test conditional mean relative reduction | -0.0230 |
| Test model better / equal / worse rate | 17.07% / 67.76% / 15.17% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0100 / 0.0000 / 0.0201 |
| Test PPO agreement | 7.24% |
| Test score std | 0.3377 |
| Test KL | 0.0024 |