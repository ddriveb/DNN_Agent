# Counterfactual R-Side Ranking Training

- Dataset: `/tmp/medium_training_9pjead_4/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_1k3`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/1k3_seed42/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 4.23% |
| Best `top1` selection score | 0.0423 |
| Test top-1 | 5.44% |
| Test top-3 | 13.77% |
| Test tie-aware top-1 | 46.65% |
| Test NDCG@3 | 0.641 |
| Test Spearman mean | 0.026 |
| Test Kendall tau-b | 0.021 |
| Test pairwise accuracy | 0.507 |
| Test relative regret reduction | -0.6903 |
| Test model regret | 0.0678 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 5.34% |
| Test score std | 0.1133 |
| Test KL | 0.0296 |