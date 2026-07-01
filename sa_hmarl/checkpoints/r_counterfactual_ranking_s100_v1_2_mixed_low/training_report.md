# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_mixed_low`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, path_penalty=0.05, fs_penalty=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_mixed_low/ranking_model.pt`

| Metric | Value |
|---|---:|
| Best val top-1 | 41.37% |
| Test top-1 | 45.86% |
| Test top-3 | 80.08% |
| Test Spearman mean | 0.786 |
| Test model regret | 0.0220 |
| Test PPO regret | 0.1354 |
| Test PPO agreement | 17.78% |
| Test score std | 0.5079 |
| Test KL | 0.0292 |