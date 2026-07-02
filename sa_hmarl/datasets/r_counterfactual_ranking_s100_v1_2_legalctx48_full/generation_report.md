# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `legalctx48`
- Horizon H: 5
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2307 | 0 (0.0%) | 17.20 | 15 | 44 | 58.60% | 55.87% | 0.0019 | 0.0027 |
| val | 776 | 0 (0.0%) | 23.40 | 17 | 48 | 75.77% | 42.78% | 0.0010 | 0.0021 |
| test | 748 | 0 (0.0%) | 13.70 | 6 | 33 | 42.38% | 67.38% | 0.0048 | 0.0074 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 2.091 | 35.83 | 0.600 | 0.380 | 0.364 | 0.095 |
| val | 2.114 | 36.50 | 0.619 | 0.403 | 0.371 | 0.114 |
| test | 2.040 | 26.85 | 0.488 | 0.340 | 0.347 | 0.096 |

### Phi-after viability profile

| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |
|---|---:|---:|---:|---:|
| train | 1.1018 | 0.2580 | 95.60% | 0.004 |
| val | 1.1579 | 0.2038 | 97.30% | 0.138 |
| test | 1.0660 | 0.2944 | 93.79% | -0.017 |

**Verdict: STOP_AND_AUDIT_DATASET**