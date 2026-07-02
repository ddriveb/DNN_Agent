# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_structured_legalctx48_mid`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, path_penalty=0.05, fs_penalty=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_settransformer_structured_mid/ranking_model.pt`

- Model type: `set_transformer`

- Selection metric: `kl`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 4.89% |
| Best `kl` selection score | -0.0183 |
| Test top-1 | 7.41% |
| Test top-3 | 60.19% |
| Test Spearman mean | 0.735 |
| Test model regret | 0.0971 |
| Test PPO regret | 0.1814 |
| Test PPO agreement | 29.63% |
| Test score std | 0.0962 |
| Test KL | 0.0420 |