# Counterfactual R-Side Ranking Training

- Dataset: `/tmp/medium_training_9pjead_4/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_5k`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/5k_seed43/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 7.54% |
| Best `top1` selection score | 0.0754 |
| Test top-1 | 5.92% |
| Test top-3 | 15.31% |
| Test tie-aware top-1 | 45.06% |
| Test NDCG@3 | 0.625 |
| Test Spearman mean | 0.002 |
| Test Kendall tau-b | 0.001 |
| Test pairwise accuracy | 0.498 |
| Test relative regret reduction | -0.1265 |
| Test model regret | 0.0933 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 8.23% |
| Test score std | 0.7016 |
| Test KL | 0.0317 |