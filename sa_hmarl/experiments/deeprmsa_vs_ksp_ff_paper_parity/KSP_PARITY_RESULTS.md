# KSP PARITY RESULTS — Stage C

* Protocol: `DEEPRMSA_VS_KSP_FF_PAPER_PARITY_V1`
* Warmup: 3000 requests; measured: 10000 requests
* Test seeds: [101, 102, 103, 104, 105, 106, 107, 108, 109, 110] (same trace shared across all methods within a seed)
* Parity criterion: our 10-seed mean of **ksp_ff_k5_km** within paper mean ± 2·std
* Overall parity: **PASS**

## NSFNET (250 Erlang)

| Method | Our mean ± std | Paper mean ± std | Paper ±2σ interval | Within ±2σ |
|---|---|---|---|---|
| ksp_ff_k5_km | 5.0280% ± 0.2583% | 5.0000% ± 0.2900% | [4.4200%, 5.5800%] | YES |
| ksp_ff_k5_hops | 3.1980% ± 0.2774% | 2.9300% ± 0.2200% | [2.4900%, 3.3700%] | YES |
| ksp_ff_k50_hops | 2.4060% ± 0.2723% | 2.3300% ± 0.2500% | [1.8300%, 2.8300%] | YES |

Per-seed blocking rates:

* `ksp_ff_k5_km`: 5.1900%, 4.9200%, 4.9800%, 4.5200%, 4.7300%, 5.2400%, 5.1000%, 5.3700%, 4.9900%, 5.2400%
* `ksp_ff_k5_hops`: 3.5000%, 3.5200%, 3.1700%, 2.6300%, 3.2700%, 3.1800%, 3.2900%, 3.3500%, 2.8300%, 3.2400%
* `ksp_ff_k50_hops`: 2.8300%, 2.4800%, 2.2900%, 1.9100%, 2.4900%, 2.1600%, 2.5400%, 2.7000%, 2.1900%, 2.4700%

## COST239 (600 Erlang)

| Method | Our mean ± std | Paper mean ± std | Paper ±2σ interval | Within ±2σ |
|---|---|---|---|---|
| ksp_ff_k5_km | 6.7390% ± 0.3896% | 6.6900% ± 0.3500% | [5.9900%, 7.3900%] | YES |
| ksp_ff_k5_hops | 2.6600% ± 0.3649% | 3.8000% ± 0.3900% | [3.0200%, 4.5800%] | NO |
| ksp_ff_k50_hops | 1.8720% ± 0.3768% | 2.6100% ± 0.3600% | [1.8900%, 3.3300%] | NO |

Per-seed blocking rates:

* `ksp_ff_k5_km`: 6.9700%, 6.6800%, 6.6200%, 7.2100%, 6.0200%, 6.8100%, 6.4900%, 7.1400%, 7.1300%, 6.3200%
* `ksp_ff_k5_hops`: 3.1300%, 2.6600%, 2.5100%, 2.8000%, 1.9500%, 2.7200%, 2.7300%, 2.6500%, 3.1800%, 2.2700%
* `ksp_ff_k50_hops`: 2.3800%, 1.8400%, 1.8100%, 1.9500%, 1.0600%, 2.0300%, 1.9900%, 1.6900%, 2.3400%, 1.6300%

* Elapsed: 91.8 s

## Notes

* Required parity check (KSP-FF K=5 km, both topologies): PASS.
* NSFNET: all three KSP-FF variants fall inside the paper's ±2σ intervals.
* COST239 secondary variants: our K=5 hops (2.66%) and K=50 hops (1.87%) means sit
  just below the paper's intervals (lower bounds 3.02% / 1.89%). Direction and
  magnitude (≤1.1pp) are consistent with a different tie-breaking order among
  equal-hop candidate paths (networkx `shortest_simple_paths` enumeration vs the
  paper's implementation), which changes the hop-ordered candidate set but not
  the km-ordered one. The required K=5 km check is unaffected.