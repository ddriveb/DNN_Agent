# Full-State Pilot Dataset Report (COST239)

- Dataset: `sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239`
- K_C=5, K_path=50, K_prop=30, H=5
- group_filter=all
- candidate_mode=ppo_r_topk_only

## Split statistics

| Split | Groups | Avg cand | E=0 | E=1 | E=1 rate | Return min | Return max | PPO regret mean | PPO headroom | Future block diff rate |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 593 | 30.00 | 59 | 534 | 90.05% | -20.0000 | -0.0000 | 0.0337 | 0.4746 | 0.84% |
| val | 580 | 25.24 | 50 | 530 | 91.38% | -20.0000 | -0.0000 | 0.2719 | 8.1958 | 7.24% |
| test | 580 | 24.62 | 120 | 460 | 79.31% | -20.0000 | -0.0000 | 0.0319 | 0.3204 | 0.86% |

## Train depth-stratum learnability

| Stratum | Groups | Return mean | Return std | Nonzero range rate | Label variance proxy |
|---|---|---:|---:|---:|---:|
| 0 | 59 | -0.9762 | 2.9407 | 76.27% | 0.0010 |
| 1 | 23 | -1.9736 | 4.6065 | 91.30% | 0.0011 |
| 2 | 406 | -1.5422 | 3.9587 | 91.13% | 0.0038 |
| 3 | 105 | -0.7651 | 2.4534 | 93.33% | 0.0085 |