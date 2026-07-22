# Strict v1.3 Exact Latency Optimization

## Protocol

- fixed-C / all-OD; no PPO-C or DF_C
- PPO-R legal Top-30; K=50 hops; 25-d frozen Ranker
- CPU, serial latency measurement
- Diagnostic KSP-FF/FF-KSP coverage calls excluded from production selector timing

## Equivalence

| Topology | States | Passed | Top-K mismatch | Action mismatch | Feature max error | Score max error |
|---|---:|---:|---:|---:|---:|---:|
| xlron_nsfnet_deeprmsa | 20 | True | 0 | 0 | 0.000e+00 | 0.000e+00 |

## Latency

| Topology | Method | Selector mean ms | Selector p95 ms | E2E mean ms | E2E p95 ms |
|---|---|---:|---:|---:|---:|
| xlron_nsfnet_deeprmsa | strict_reference | 8.692 | 11.293 | 14.654 | 18.401 |
| xlron_nsfnet_deeprmsa | strict_optimized | 4.063 | 3.495 | 9.903 | 11.477 |
| xlron_nsfnet_deeprmsa | ksp_ff | 0.014 | 0.021 | 5.713 | 9.040 |
| xlron_nsfnet_deeprmsa | ff_ksp | 0.022 | 0.033 | 5.126 | 6.879 |

## Speedup

- `xlron_nsfnet_deeprmsa`: selector 2.14x; end-to-end 1.48x.
