# Implementation Audit: Phase-R0 Expanded-Path Rescue

## Rescue action selector

`expanded_ksp_ff_rescue_action` is defined in `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py`. It scans expanded candidate paths in hops/km/lexicographic node-tuple order, selects the feasible modulation with the smallest required FS (highest spectral efficiency on ties, then smallest mod index), and selects the lowest start-slot feasible block.

## Rescue trigger

Rescue is only triggered when the K=50 R-mask is empty. If K=100, K=200, and K=500 all fail, the request is rejected with reason `r_no_valid_action_after_k500_rescue`.

## Environment K restoration

During rescue, `env.k` is temporarily set to the expansion K, the rescue action is executed via `env.step`, and `env.k` is restored to 50 immediately after the step.
