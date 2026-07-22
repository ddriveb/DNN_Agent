# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_stratified_depth_strict/seed_43/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 5.48% |
| Best `regret` selection score | -0.1386 |
| Test top-1 | 9.48% |
| Test top-3 | 22.76% |
| Test score-tie fractional top-1 | 51.90% |
| Test oracle tie hit rate | 52.24% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.716 |
| Test Spearman mean | -0.015 |
| Test Kendall tau-b | -0.013 |
| Test pairwise accuracy | 0.499 |
| Test model regret | 0.0284 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0035 |
| Test aggregate regret reduction | 11.08% |
| Test conditional mean relative reduction | 0.0842 |
| Test model better / equal / worse rate | 20.00% / 64.83% / 15.17% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0100 / 0.0000 / 0.0346 |
| Test PPO agreement | 8.45% |
| Test score std | 0.1885 |
| Test KL | 0.0017 |