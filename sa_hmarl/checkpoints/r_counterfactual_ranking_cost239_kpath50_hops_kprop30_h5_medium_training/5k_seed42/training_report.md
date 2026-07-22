# Counterfactual R-Side Ranking Training

- Dataset: `/tmp/medium_training_9pjead_4/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_5k`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/5k_seed42/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 7.71% |
| Best `top1` selection score | 0.0771 |
| Test top-1 | 6.55% |
| Test top-3 | 15.41% |
| Test tie-aware top-1 | 45.38% |
| Test NDCG@3 | 0.628 |
| Test Spearman mean | 0.012 |
| Test Kendall tau-b | 0.009 |
| Test pairwise accuracy | 0.532 |
| Test relative regret reduction | -0.0983 |
| Test model regret | 0.0913 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 8.18% |
| Test score std | 0.6471 |
| Test KL | 0.0315 |