# Status: SUPERSEDED / MECHANICAL_INVALID

This v1 trajectory policy-effect diagnosis is **superseded** by the corrected v2 deliverable:

`sa_hmarl/experiments/v13_trajectory_policy_effect_diagnosis_v2/`

## Reason for invalidation

The v1 continuation function used an off-by-one horizon definition: after applying the first fork action it processed `future_requests[:H]` future requests. Consequently, `H=1` simulated **one** future request instead of **zero**, violating the mechanical invariant that both forks must report 0 blocks and ΔB_1 = 0 when both first actions succeed.

Observed failure: 7 out of 301 × 5 H=1 trace records had non-zero ΔB (mean ΔB ≈ -0.001).

## Correction in v2

* Horizon H now means: 1 first action + `max(0, H-1)` future requests.
* Hard assertions enforce `blocks_strict[1] == 0`, `blocks_ksp[1] == 0`, and therefore `ΔB_1 == 0`.
* Snapshot-level cluster inference (bootstrap CI, sign-flip permutation test, Holm correction, TOST equivalence) is now reported.

Do not cite the v1 report for any quantitative claim.
