# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/checkpoints/full_state_uniform/seed_43/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 3.36% |
| Best `regret` selection score | -0.1600 |
| Test top-1 | 18.62% |
| Test top-3 | 22.93% |
| Test score-tie fractional top-1 | 50.52% |
| Test oracle tie hit rate | 50.69% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.695 |
| Test Spearman mean | -0.071 |
| Test Kendall tau-b | -0.057 |
| Test pairwise accuracy | 0.428 |
| Test model regret | 0.0293 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0026 |
| Test aggregate regret reduction | 8.22% |
| Test conditional mean relative reduction | -0.0535 |
| Test model better / equal / worse rate | 17.24% / 66.90% / 15.86% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0124 / 0.0000 / 0.0245 |
| Test PPO agreement | 17.93% |
| Test score std | 0.4018 |
| Test KL | 0.0020 |