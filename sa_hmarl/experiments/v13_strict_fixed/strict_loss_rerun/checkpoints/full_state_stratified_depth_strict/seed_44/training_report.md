# Counterfactual R-Side Ranking Training

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- Return coefs: current_block=3.0, future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05
- Horizon: 5
- Checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_stratified_depth_strict/seed_44/ranking_model.pt`

- Model type: `mlp`

- Selection metric: `regret`

| Metric | Value |
|---|---:|
| Best checkpoint val top-1 | 2.12% |
| Best `regret` selection score | -0.1606 |
| Test top-1 | 8.28% |
| Test top-3 | 16.21% |
| Test score-tie fractional top-1 | 51.21% |
| Test oracle tie hit rate | 51.21% |
| Test PPO oracle tie hit rate | 52.07% |
| Test NDCG@3 | 0.714 |
| Test Spearman mean | 0.053 |
| Test Kendall tau-b | 0.042 |
| Test pairwise accuracy | 0.579 |
| Test model regret | 0.0283 |
| Test PPO regret | 0.0319 |
| Test absolute regret improvement | 0.0037 |
| Test aggregate regret reduction | 11.49% |
| Test conditional mean relative reduction | 0.0923 |
| Test model better / equal / worse rate | 20.52% / 63.79% / 15.69% |
| Test median regret delta | 0.0000 |
| Test regret delta P10/P50/P90 | -0.0090 / 0.0000 / 0.0250 |
| Test PPO agreement | 10.00% |
| Test score std | 0.3844 |
| Test KL | 0.0021 |