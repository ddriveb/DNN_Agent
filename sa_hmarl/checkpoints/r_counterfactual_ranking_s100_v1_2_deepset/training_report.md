# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_mixed_low`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, path_penalty=0.05, fs_penalty=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_deepset/ranking_model.pt`

- Model type: `deepset`

- Selection metric: `kl`

| Metric | Value |
|---|---:|
| Best val top-1 | 29.38% |
| Best selection score | -0.0075 |
| Test top-1 | 26.34% |
| Test top-3 | 69.39% |
| Test Spearman mean | 0.792 |
| Test model regret | 0.0478 |
| Test PPO regret | 0.1354 |
| Test PPO agreement | 18.05% |
| Test score std | 0.4558 |
| Test KL | 0.0297 |