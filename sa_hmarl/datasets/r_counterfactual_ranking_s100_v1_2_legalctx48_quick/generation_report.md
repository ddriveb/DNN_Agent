# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `larger_legal_context`
- Horizon H: 5
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs - 0.05*norm(path_km) - 0.05*norm(required_fs)`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 240 | 0 (0.0%) | 8.38 | 4 | 16 | 78.33% | 10.83% | 0.0000 | 0.0000 |
| val | 80 | 0 (0.0%) | 9.66 | 12 | 15 | 81.25% | 7.50% | 0.0000 | 0.0000 |
| test | 80 | 0 (0.0%) | 4.88 | 4 | 4 | 70.00% | 15.00% | 0.0000 | 0.0000 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 2.170 | 69.77 | 0.897 | 0.316 | 0.390 | 0.013 |
| val | 2.092 | 80.40 | 0.950 | 0.385 | 0.364 | 0.002 |
| test | 2.338 | 49.38 | 0.841 | 0.090 | 0.446 | 0.021 |

### Phi-after viability profile

| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |
|---|---:|---:|---:|---:|
| train | 1.1165 | 0.2673 | 94.68% | -0.180 |
| val | 1.1076 | 0.2451 | 95.60% | 0.158 |
| test | 0.9585 | 0.3948 | 85.90% | -0.422 |

**Verdict: STOP_AND_AUDIT_DATASET**