# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_v1_2_mixed_mid`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, path_penalty=0.1, fs_penalty=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_mid/ranking_model.pt`

| Metric | Value |
|---|---:|
| Best val top-1 | 77.88% |
| Test top-1 | 75.52% |
| Test top-3 | 96.50% |
| Test Spearman mean | 0.841 |
| Test model regret | 0.0538 |
| Test PPO regret | 0.2521 |
| Test PPO agreement | 32.87% |
| Test score std | 0.6391 |
| Test KL | 0.0349 |