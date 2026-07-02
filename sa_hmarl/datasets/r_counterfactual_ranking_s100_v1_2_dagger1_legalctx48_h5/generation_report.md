# Counterfactual R-Side Ranking Dataset Generation

- Candidate mode: `legalctx48`
- Horizon H: 5
- Trajectory policy: `ranker`
- Trajectory ranker checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_s100_v1_2_mixed_low/ranking_model.pt`
- Trajectory ranker epsilon: 0.0
- Return: `-3.0*cur_blocked - 4.0*future_blocked - 3.0*future_nsb - 0.03*delay_mean - 0.05*avg_fs`

| Split | Groups | Skipped | Avg cand | P50 | P90 | Nonzero range | PPO top-1 | Oracle headroom (pp) | Future blocked var |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 2371 | 0 (0.0%) | 17.92 | 15 | 44 | 61.03% | 58.20% | 0.0012 | 0.0017 |
| val | 797 | 0 (0.0%) | 23.34 | 17 | 48 | 79.55% | 41.78% | 0.0005 | 0.0020 |
| test | 788 | 0 (0.0%) | 15.36 | 8 | 43 | 50.76% | 67.51% | 0.0008 | 0.0044 |

### Candidate feature profile

| Split | Avg SE | Avg block_size | Avg block_waste | Avg path_norm | Avg mod_norm | Avg block_norm |
|---|---:|---:|---:|---:|---:|---:|
| train | 2.083 | 34.55 | 0.550 | 0.337 | 0.361 | 0.134 |
| val | 2.093 | 35.91 | 0.571 | 0.357 | 0.364 | 0.151 |
| test | 2.148 | 23.49 | 0.400 | 0.285 | 0.383 | 0.148 |

### Phi-after viability profile

| Split | Phi mean | Phi std | Nonzero rate | Spearman w/ return |
|---|---:|---:|---:|---:|
| train | 1.0942 | 0.2148 | 97.26% | 0.009 |
| val | 1.1107 | 0.2000 | 97.83% | -0.001 |
| test | 1.1010 | 0.2241 | 96.84% | 0.196 |

**Verdict: STOP_AND_AUDIT_DATASET**