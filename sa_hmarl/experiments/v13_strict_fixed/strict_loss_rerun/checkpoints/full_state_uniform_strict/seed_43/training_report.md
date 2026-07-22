# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_43/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 5.12% |
| Best `regret` selection score | -0.1526 |
| Test top-1 | 19.14% |
| Test top-3 | 25.69% |
| Test score-tie fractional top-1 | 51.38% |
| Test oracle tie hit rate | 51.55% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.717 |
| Test Spearman mean | -0.005 |
| Test Kendall tau-b | -0.005 |
| Test pairwise accuracy | 0.511 |
| Test model regret | 0.0280 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0039 |
| Test aggregate regret reduction | 12.26% |
| Test conditional mean relative reduction | 0.0584 |
| Test model better / equal / worse rate | 19.31% / 65.52% / 15.17% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0100 / 0.0000 / 0.0346 |
| Test PPO agreement | 17.76% |
| Test score std | 0.2493 |
| Test KL | 0.0019 |