# C-Side Interaction: Strict v1.3 vs KSP-FF K=50 hops

Δ_PPO-C = blocking(Strict, PPO-C) - blocking(KSP-FF, PPO-C)
Δ_DF_C  = blocking(Strict, DF_C) - blocking(KSP-FF, DF_C)
Interaction = Δ_DF_C - Δ_PPO-C

| Quantity | Mean | Median | Std | Bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Δ_PPO-C | -2.4392pp | -2.0667pp | 0.9219pp | [-2.8433%, -2.0592%] |
| Δ_DF_C  | -0.0342pp | 0.0000pp | 0.1528pp | [-0.1025%, 0.0000%] |
| Interaction | 2.4050pp | 2.0667pp | 0.8968pp | [2.0392%, 2.8008%] |

Wilcoxon p for interaction: 0.000088

**Interpretation:** Strict v1.3's advantage over KSP-FF is smaller under DF_C than under PPO-C (interaction significant).