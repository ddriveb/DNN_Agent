# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_v1_2_mixed_low`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, path_penalty=0.05, fs_penalty=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`

| Metric | Value |
|---|---:|
| Best val top-1 | 72.81% |
| Test top-1 | 76.22% |
| Test top-3 | 97.20% |
| Test Spearman mean | 0.655 |
| Test model regret | 0.0536 |
| Test PPO regret | 0.2334 |
| Test PPO agreement | 34.97% |
| Test score std | 0.6489 |
| Test KL | 0.0343 |