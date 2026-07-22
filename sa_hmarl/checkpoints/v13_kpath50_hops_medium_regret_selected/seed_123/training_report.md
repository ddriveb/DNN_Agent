# Counterfactual R-Side Ranking Training

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_123/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 3.60% |
| Best `regret` selection score | -0.0432 |
| Test top-1 | 4.57% |
| Test top-3 | 10.78% |
| Test score-tie fractional top-1 | 46.00% |
| Test oracle tie hit rate | 47.76% |
| Test PPO oracle tie hit rate | 43.28% |
| Test NDCG@3 | 0.638 |
| Test Spearman mean | 0.030 |
| Test Kendall tau-b | 0.024 |
| Test pairwise accuracy | 0.515 |
| Test model regret | 0.0696 |
| Test PPO regret | 0.1050 |
| Test absolute regret improvement | 0.0354 |
| Test aggregate regret reduction | 33.71% |
| Test conditional mean relative reduction | -0.5246 |
| Test model better / equal / worse rate | 27.35% / 53.44% / 19.21% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0154 / 0.0000 / 0.0320 |
| Test PPO agreement | 4.67% |
| Test score std | 0.6336 |
| Test KL | 0.0307 |