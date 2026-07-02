# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/r_counterfactual_ranking_s100_v1_2_legalctx48`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, path_penalty=0.05, fs_penalty=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_settransformer_legalctx48/ranking_model.pt`

- Model type: `set_transformer`

- Selection metric: `kl`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 16.49% |
| Best `kl` selection score | -0.0074 |
| Test top-1 | 24.47% |
| Test top-3 | 61.10% |
| Test Spearman mean | 0.786 |
| Test model regret | 0.0484 |
| Test PPO regret | 0.1360 |
| Test PPO agreement | 16.98% |
| Test score std | 0.6029 |
| Test KL | 0.0267 |