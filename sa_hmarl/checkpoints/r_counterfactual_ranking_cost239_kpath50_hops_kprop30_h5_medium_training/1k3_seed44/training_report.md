# Counterfactual R-Side Ranking Training

- Dataset: `/tmp/medium_training_9pjead_4/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_1k3`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `/mnt/d/project/DNN_Agent/sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium_training/1k3_seed44/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `top1`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 4.17% |
| Best `top1` selection score | 0.0417 |
| Test top-1 | 2.94% |
| Test top-3 | 8.38% |
| Test tie-aware top-1 | 46.94% |
| Test NDCG@3 | 0.643 |
| Test Spearman mean | 0.066 |
| Test Kendall tau-b | 0.053 |
| Test pairwise accuracy | 0.543 |
| Test relative regret reduction | -0.6517 |
| Test model regret | 0.0727 |
| Test PPO regret | 0.1050 |
| Test PPO agreement | 3.23% |
| Test score std | 0.1002 |
| Test KL | 0.0304 |