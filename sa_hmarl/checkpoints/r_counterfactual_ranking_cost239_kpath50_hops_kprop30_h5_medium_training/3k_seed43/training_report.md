# Counterfactual R-Side Ranking Training

- Dataset: `/tmp/medium_training_9pjead_4/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_3k`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/3k_seed43/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 6.80% |
| Best `top1` selection score | 0.0680 |
| Test top-1 | 5.97% |
| Test top-3 | 15.31% |
| Test tie-aware top-1 | 44.82% |
| Test NDCG@3 | 0.619 |
| Test Spearman mean | -0.002 |
| Test Kendall tau-b | -0.001 |
| Test pairwise accuracy | 0.508 |
| Test relative regret reduction | -0.3197 |
| Test model regret | 0.0721 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 8.14% |
| Test score std | 0.5533 |
| Test KL | 0.0308 |