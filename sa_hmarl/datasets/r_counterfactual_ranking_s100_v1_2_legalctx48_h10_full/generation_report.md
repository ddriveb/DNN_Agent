# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `legalctx48`
- Horizon H: 10
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2307 | 0 (0.0%) | 17.20 | 15 | 44 | 63.94% | 49.41% | 0.0022 | 0.0093 |
| val | 776 | 0 (0.0%) | 23.40 | 17 | 48 | 81.44% | 35.18% | 0.0013 | 0.0049 |
| test | 748 | 0 (0.0%) | 13.70 | 6 | 33 | 47.86% | 61.36% | 0.0056 | 0.0234 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 2.091 | 35.83 | 0.600 | 0.380 | 0.364 | 0.095 |
| val | 2.114 | 36.50 | 0.619 | 0.403 | 0.371 | 0.114 |
| test | 2.040 | 26.85 | 0.488 | 0.340 | 0.347 | 0.096 |

### Phi-after viability profile

| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |
|---|---:|---:|---:|---:|
| train | 1.1018 | 0.2580 | 95.60% | 0.019 |
| val | 1.1579 | 0.2038 | 97.30% | 0.118 |
| test | 1.0660 | 0.2944 | 93.79% | 0.027 |

**Verdict: STOP_AND_AUDIT_DATASET**