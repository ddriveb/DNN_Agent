# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `v1`
- Horizon H: 5
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs - 0.1*norm(path_km) - 0.05*norm(required_fs)`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 1433 | 0 (0.0%) | 8.14 | 7 | 15 | 81.23% | 42.43% | 0.0070 | 0.0107 |
| val | 217 | 0 (0.0%) | 8.79 | 8 | 15 | 82.95% | 49.77% | 0.0175 | 0.0185 |
| test | 143 | 0 (0.0%) | 6.97 | 4 | 15 | 69.23% | 46.85% | 0.0098 | 0.0180 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 2.080 | 10.77 | 0.518 | 0.405 | 0.360 | 0.010 |
| val | 2.003 | 9.62 | 0.504 | 0.475 | 0.334 | 0.008 |
| test | 2.020 | 7.41 | 0.402 | 0.348 | 0.340 | 0.003 |

**Verdict: STOP_AND_AUDIT_DATASET**