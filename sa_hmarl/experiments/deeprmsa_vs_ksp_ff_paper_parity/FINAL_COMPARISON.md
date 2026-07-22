# FINAL COMPARISON — DeepRMSA vs KSP-FF (paper-standard pure RMSA)

* Protocol: `DEEPRMSA_VS_KSP_FF_PAPER_PARITY_V1` (see PROTOCOL_LOCK.md)
* Warmup 3000, measured 10000 requests; 10 shared test seeds [101, 102, 103, 104, 105, 106, 107, 108, 109, 110]
* KSP-FF std = across 10 test seeds. DeepRMSA local: mean over 3 training seeds; std shown as (training-seed std / pooled test-seed std).

## NSFNET — 250 Erlang, 100 slots

| Method | Ours | Paper | Within paper ±2σ |
|---|---|---|---|
| KSP-FF K=5 km | 5.0280% ± 0.2583% | 5.0000% ± 0.2900% | YES |
| KSP-FF K=5 hops | 3.1980% ± 0.2774% | 2.9300% ± 0.2200% | YES |
| KSP-FF K=50 hops | 2.4060% ± 0.2723% | 2.3300% ± 0.2500% | YES |
| DeepRMSA (local, K=5) | 7.2093% (±1.8614% train / ±1.5910% test) | 4.0000% (published) | — |

Per-training-seed DeepRMSA means: seed 42: 6.3890%, seed 43: 9.3400%, seed 44: 5.8990%

## COST239 — 600 Erlang, 100 slots

| Method | Ours | Paper | Within paper ±2σ |
|---|---|---|---|
| KSP-FF K=5 km | 6.7390% ± 0.3896% | 6.6900% ± 0.3500% | YES |
| KSP-FF K=5 hops | 2.6600% ± 0.3649% | 3.8000% ± 0.3900% | NO |
| KSP-FF K=50 hops | 1.8720% ± 0.3768% | 2.6100% ± 0.3600% | NO |
| DeepRMSA (local, K=5) | 9.5650% (±2.2588% train / ±1.9469% test) | 5.7500% (published) | — |

Per-training-seed DeepRMSA means: seed 42: 11.9520%, seed 43: 9.2820%, seed 44: 7.4610%

## Appendix (excluded from the main comparison)

* Legacy masked DeepRMSA-style K=3/M=1 on custom SNAP24: blocking 25.7067% (different topology, masked action space, K=3/M=1 — not comparable; kept for provenance only).
