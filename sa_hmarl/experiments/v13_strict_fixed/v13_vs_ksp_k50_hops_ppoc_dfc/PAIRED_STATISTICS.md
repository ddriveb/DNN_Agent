# Paired Statistics

All comparisons are paired by traffic seed. n = number of seeds.

## Primary comparisons (Holm-corrected)

| Comparison | Mean Δ | Median Δ | 95% CI | Wilcoxon p | Holm-adj p | Strict better / total | Relative reduction |
|---|---:|---:|---:|---:|---:|---:|---:|
| PPO-C Strict vs KSP-FF | -2.44pp | -2.07pp | [-2.8433%, -2.0592%] | 0.000088 | 0.000177 | 20/20 | -26.42% |
| DF_C Strict vs KSP-FF | -0.03pp | 0.00pp | [-0.1025%, 0.0000%] | 0.317311 | 0.000177 | 1/20 | -0.55% |

## Secondary comparisons

| Comparison | Mean Δ | 95% CI | Wilcoxon p | Sign-flip p | Better / total |
|---|---:|---:|---:|---:|---:|
| PPO-C Strict vs PPO-R | 0.11pp | [-0.1400%, 0.3425%] | 0.262597 | 0.406955 | 7/20 |
| DF_C Strict vs PPO-R | 0.00pp | [0.0000%, 0.0000%] | nan | 1.000000 | 0/20 |
| PPO-C KSP-FF vs PPO-R | 2.55pp | [2.0658%, 3.0725%] | 0.000002 | 0.000002 | 0/20 |
| DF_C KSP-FF vs PPO-R | 0.03pp | [0.0000%, 0.1025%] | 0.317311 | 1.000000 | 0/20 |