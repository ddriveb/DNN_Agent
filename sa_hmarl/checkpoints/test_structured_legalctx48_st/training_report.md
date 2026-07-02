# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/datasets/test_structured_legalctx48`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/checkpoints/test_structured_legalctx48_st/ranking_model.pt`

- Model type: `set_transformer`

- Selection metric: `kl`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 5.00% |
| Best `kl` selection score | -0.0056 |
| Test top-1 | 30.00% |
| Test top-3 | 100.00% |
| Test Spearman mean | 0.000 |
| Test model regret | 0.0000 |
| Test PPO regret | 0.0000 |
| Test PPO agreement | 30.00% |
| Test score std | 0.0537 |
| Test KL | 0.0001 |