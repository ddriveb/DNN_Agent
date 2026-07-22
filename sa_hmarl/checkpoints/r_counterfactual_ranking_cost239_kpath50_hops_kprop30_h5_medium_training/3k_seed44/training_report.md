# Counterfactual R-Side Ranking Training

- Dataset: `/tmp/medium_training_9pjead_4/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_3k`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/3k_seed44/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 6.74% |
| Best `top1` selection score | 0.0674 |
| Test top-1 | 6.50% |
| Test top-3 | 15.70% |
| Test tie-aware top-1 | 45.38% |
| Test NDCG@3 | 0.624 |
| Test Spearman mean | 0.002 |
| Test Kendall tau-b | 0.001 |
| Test pairwise accuracy | 0.521 |
| Test relative regret reduction | -0.1893 |
| Test model regret | 0.0658 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 7.22% |
| Test score std | 0.6171 |
| Test KL | 0.0311 |