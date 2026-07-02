# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_dagger1_legalctx48_h5`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_dagger1_mlp_legalctx48_h5/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `kl`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 11.79% |
| Best `kl` selection score | -0.0056 |
| Test top-1 | 19.92% |
| Test top-3 | 46.70% |
| Test Spearman mean | 0.004 |
| Test model regret | 0.0354 |
| Test PPO regret | 0.0309 |
| Test PPO agreement | 21.57% |
| Test score std | 0.0709 |
| Test KL | 0.0127 |