# Counterfactual R-Side Ranking Training

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_456/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 6.17% |
| Best `regret` selection score | -0.0450 |
| Test top-1 | 5.78% |
| Test top-3 | 13.05% |
| Test score-tie fractional top-1 | 43.38% |
| Test oracle tie hit rate | 45.02% |
| Test PPO oracle tie hit rate | 43.28% |
| Test NDCG@3 | 0.617 |
| Test Spearman mean | -0.006 |
| Test Kendall tau-b | -0.005 |
| Test pairwise accuracy | 0.510 |
| Test model regret | 0.0791 |
| Test PPO regret | 0.1050 |
| Test absolute regret improvement | 0.0259 |
| Test aggregate regret reduction | 24.65% |
| Test conditional mean relative reduction | -2.0384 |
| Test model better / equal / worse rate | 23.50% / 57.10% / 19.40% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0173 / 0.0000 / 0.0238 |
| Test PPO agreement | 8.23% |
| Test score std | 0.5609 |
| Test KL | 0.0297 |