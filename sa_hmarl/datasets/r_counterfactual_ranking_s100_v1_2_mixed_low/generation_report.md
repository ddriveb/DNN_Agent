# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `v1`
- Horizon H: 5
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs - 0.05*norm(path_km) - 0.05*norm(required_fs)`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2307 | 0 (0.0%) | 12.07 | 15 | 20 | 83.18% | 29.56% | 0.0019 | 0.0027 |
| val | 776 | 0 (0.0%) | 14.95 | 16 | 21 | 91.75% | 31.06% | 0.0010 | 0.0021 |
| test | 748 | 0 (0.0%) | 9.91 | 6 | 19 | 77.27% | 31.15% | 0.0048 | 0.0072 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 1.951 | 42.88 | 0.635 | 0.369 | 0.317 | 0.064 |
| val | 1.914 | 45.98 | 0.676 | 0.406 | 0.305 | 0.074 |
| test | 1.966 | 32.67 | 0.514 | 0.300 | 0.322 | 0.068 |

**Verdict: STOP_AND_AUDIT_DATASET**