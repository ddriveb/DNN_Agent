# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/checkpoints/full_state_stratified_depth/seed_44/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 5.30% |
| Best `regret` selection score | -0.1650 |
| Test top-1 | 7.41% |
| Test top-3 | 11.03% |
| Test score-tie fractional top-1 | 50.69% |
| Test oracle tie hit rate | 50.86% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.689 |
| Test Spearman mean | -0.067 |
| Test Kendall tau-b | -0.053 |
| Test pairwise accuracy | 0.444 |
| Test model regret | 0.0281 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0038 |
| Test aggregate regret reduction | 11.97% |
| Test conditional mean relative reduction | 0.0485 |
| Test model better / equal / worse rate | 18.79% / 65.00% / 16.21% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0100 / 0.0000 / 0.0277 |
| Test PPO agreement | 6.21% |
| Test score std | 0.2930 |
| Test KL | 0.0018 |