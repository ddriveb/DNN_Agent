# PPO_C Comparison: Strict v1.3 vs KSP-FF K=50 hops

Seeds: 20
Requests per seed: 120000 total evaluated

## Summary statistics

| Method | Mean blocking | Std | Median | P25 | P75 | Overload | NSB | Avg delay ms | Avg FS |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO-C + Strict v1.3 | 6.7933% | 1.4090% | 6.3167% | 5.7708% | 7.6417% | 6.7925% | 0.0008% | 9.59 | 3.14 |
| PPO-C + KSP-FF K=50 hops | 9.2325% | 1.5645% | 9.0250% | 7.8583% | 10.6250% | 9.2308% | 0.0008% | 8.69 | 2.31 |
| PPO-C + PPO-R Top-1 | 6.6850% | 1.4338% | 6.3500% | 5.5583% | 7.6958% | 6.6850% | 0.0000% | 8.64 | 3.19 |
| PPO-C + Legacy E1-trained gated baseline | 6.7917% | 1.3293% | 6.4583% | 5.8000% | 7.4375% | 6.7908% | 0.0008% | 9.71 | 3.06 |

## Strict v1.3 vs KSP-FF K=50 hops

Mean difference: -2.44pp
Median difference: -2.07pp
Bootstrap 95% CI: [-2.8433%, -2.0592%]
Wilcoxon p: 0.000088
Sign-flip permutation p: 0.000002
Strict better seeds: 20/20
Relative reduction: -26.42%

## Strict v1.3 vs PPO-R Top-1

Mean difference: 0.11pp
Bootstrap 95% CI: [-0.1400%, 0.3425%]
Wilcoxon p: 0.262597
Sign-flip permutation p: 0.406955
Strict better seeds: 7/20

## KSP-FF K=50 hops vs PPO-R Top-1

Mean difference: 2.55pp
Bootstrap 95% CI: [2.0658%, 3.0725%]
Wilcoxon p: 0.000002
Sign-flip permutation p: 0.000002