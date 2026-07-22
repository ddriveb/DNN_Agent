# Counterfactual R-Side Ranking Training

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 5.49% |
| Best `regret` selection score | -0.0465 |
| Test top-1 | 4.19% |
| Test top-3 | 11.84% |
| Test score-tie fractional top-1 | 44.85% |
| Test oracle tie hit rate | 46.22% |
| Test PPO oracle tie hit rate | 43.28% |
| Test NDCG@3 | 0.627 |
| Test Spearman mean | -0.003 |
| Test Kendall tau-b | -0.003 |
| Test pairwise accuracy | 0.526 |
| Test model regret | 0.0682 |
| Test PPO regret | 0.1050 |
| Test absolute regret improvement | 0.0368 |
| Test aggregate regret reduction | 35.04% |
| Test conditional mean relative reduction | -0.7040 |
| Test model better / equal / worse rate | 26.34% / 53.59% / 20.08% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0173 / 0.0000 / 0.0285 |
| Test PPO agreement | 4.33% |
| Test score std | 0.5981 |
| Test KL | 0.0302 |