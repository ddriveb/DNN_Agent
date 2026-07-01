# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_v1_2_fs`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, path_penalty=0.0, fs_penalty=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_fs/ranking_model.pt`

| Metric | Value |
|---|---:|
| Best val top-1 | 45.16% |
| Test top-1 | 58.04% |
| Test top-3 | 85.31% |
| Test Spearman mean | 0.517 |
| Test model regret | 0.0062 |
| Test PPO regret | 0.2217 |
| Test PPO agreement | 27.97% |
| Test score std | 0.5946 |
| Test KL | 0.0347 |