# DF_C Comparison: Strict v1.3 vs KSP-FF K=50 hops

Seeds: 20
Requests per seed: 120000 total evaluated

## Summary statistics

| Method | Mean blocking | Std | Median | P25 | P75 | Overload | NSB | Avg delay ms | Avg FS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DF_C + Strict v1.3 | 6.1642% | 1.4036% | 5.9583% | 5.1542% | 6.7583% | 6.1633% | 0.0008% | 9.53 | 3.16 |
| DF_C + KSP-FF K=50 hops | 6.1983% | 1.3610% | 5.9583% | 5.1542% | 6.7583% | 6.1983% | 0.0000% | 8.01 | 2.22 |
| DF_C + PPO-R Top-1 | 6.1642% | 1.4036% | 5.9583% | 5.1542% | 6.7583% | 6.1633% | 0.0008% | 8.64 | 3.10 |
| DF_C + Legacy E1-trained gated baseline | 6.1642% | 1.4036% | 5.9583% | 5.1542% | 6.7583% | 6.1633% | 0.0008% | 9.40 | 2.98 |

## Strict v1.3 vs KSP-FF K=50 hops

Mean difference: -0.03pp
Median difference: 0.00pp
Bootstrap 95% CI: [-0.1025%, 0.0000%]
Wilcoxon p: 0.317311
Sign-flip permutation p: 1.000000
Strict better seeds: 1/20
Relative reduction: -0.55%

## Strict v1.3 vs PPO-R Top-1

Mean difference: 0.00pp
Bootstrap 95% CI: [0.0000%, 0.0000%]
Wilcoxon p: nan
Sign-flip permutation p: 1.000000
Strict better seeds: 0/20

## KSP-FF K=50 hops vs PPO-R Top-1

Mean difference: 0.03pp
Bootstrap 95% CI: [0.0000%, 0.1025%]
Wilcoxon p: 0.317311
Sign-flip permutation p: 1.000000