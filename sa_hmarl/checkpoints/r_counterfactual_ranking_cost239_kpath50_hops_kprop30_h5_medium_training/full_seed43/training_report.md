# Counterfactual R-Side Ranking Training

- Dataset: `/mnt/d/project/DNN_Agent/sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/full_seed43/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 7.83% |
| Best `top1` selection score | 0.0783 |
| Test top-1 | 6.55% |
| Test top-3 | 15.84% |
| Test tie-aware top-1 | 45.35% |
| Test NDCG@3 | 0.638 |
| Test Spearman mean | 0.008 |
| Test Kendall tau-b | 0.006 |
| Test pairwise accuracy | 0.518 |
| Test relative regret reduction | -1.1403 |
| Test model regret | 0.0868 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 8.23% |
| Test score std | 0.5850 |
| Test KL | 0.0311 |