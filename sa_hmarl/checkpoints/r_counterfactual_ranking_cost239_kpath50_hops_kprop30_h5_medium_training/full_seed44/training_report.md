# Counterfactual R-Side Ranking Training

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/full_seed44/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 6.91% |
| Best `top1` selection score | 0.0691 |
| Test top-1 | 3.90% |
| Test top-3 | 10.40% |
| Test tie-aware top-1 | 44.92% |
| Test NDCG@3 | 0.633 |
| Test Spearman mean | 0.022 |
| Test Kendall tau-b | 0.017 |
| Test pairwise accuracy | 0.532 |
| Test relative regret reduction | -1.3242 |
| Test model regret | 0.0817 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 4.67% |
| Test score std | 0.7316 |
| Test KL | 0.0322 |