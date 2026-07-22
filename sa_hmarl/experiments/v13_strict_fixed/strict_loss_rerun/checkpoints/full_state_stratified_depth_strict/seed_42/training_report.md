# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_stratified_depth_strict/seed_42/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 3.18% |
| Best `regret` selection score | -0.1522 |
| Test top-1 | 12.59% |
| Test top-3 | 18.62% |
| Test score-tie fractional top-1 | 50.34% |
| Test oracle tie hit rate | 50.34% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.680 |
| Test Spearman mean | -0.076 |
| Test Kendall tau-b | -0.061 |
| Test pairwise accuracy | 0.438 |
| Test model regret | 0.0304 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0015 |
| Test aggregate regret reduction | 4.76% |
| Test conditional mean relative reduction | -0.0992 |
| Test model better / equal / worse rate | 20.17% / 62.41% / 17.41% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0201 / 0.0000 / 0.0256 |
| Test PPO agreement | 11.21% |
| Test score std | 0.4465 |
| Test KL | 0.0020 |