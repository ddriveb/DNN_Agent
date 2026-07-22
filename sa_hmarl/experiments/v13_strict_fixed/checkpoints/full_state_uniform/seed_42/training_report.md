# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/checkpoints/full_state_uniform/seed_42/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 2.30% |
| Best `regret` selection score | -0.1610 |
| Test top-1 | 1.03% |
| Test top-3 | 6.21% |
| Test score-tie fractional top-1 | 50.00% |
| Test oracle tie hit rate | 50.17% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.685 |
| Test Spearman mean | -0.065 |
| Test Kendall tau-b | -0.051 |
| Test pairwise accuracy | 0.443 |
| Test model regret | 0.0309 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0011 |
| Test aggregate regret reduction | 3.41% |
| Test conditional mean relative reduction | -0.0401 |
| Test model better / equal / worse rate | 16.72% / 65.17% / 18.10% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0193 / 0.0000 / 0.0194 |
| Test PPO agreement | 0.69% |
| Test score std | 0.1416 |
| Test KL | 0.0018 |