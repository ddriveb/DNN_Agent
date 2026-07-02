# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_legalctx48_h10_full`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 10
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_mlp_legalctx48_h10_full/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `kl`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 10.95% |
| Best `kl` selection score | -0.0165 |
| Test top-1 | 17.38% |
| Test top-3 | 41.98% |
| Test Spearman mean | 0.028 |
| Test model regret | 0.1266 |
| Test PPO regret | 0.2360 |
| Test PPO agreement | 17.91% |
| Test score std | 0.4635 |
| Test KL | 0.0633 |