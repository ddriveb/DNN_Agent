# H=1 Invariant Failure Root Cause

**v1 status:** `MECHANICAL_INVALID / SUPERSEDED_PENDING_H1_FIX`
**v2 status:** `CORRECTED`

## Root cause
The v1 continuation function used an off-by-one horizon definition: after applying the first fork action it looped over `future_requests[:H]`. Consequently, `H=1` simulated **one** future request instead of **zero**.

## Observed failure
* Eligible snapshots: 301
* Future traces per snapshot: 5
* H=1 non-zero ΔB records: 7
* Example: snapshot 3033, trace 0 → ΔB_1 = 1; The first future request (id 3034) blocked in the Strict fork but not in the KSP fork.

## Fix applied in v2
* Code change: `Process max(0, H-1) future requests after the first action.`
* Added hard assertions in `_evaluate_snapshots`:
  - `blocks_strict[1] == 0`
  - `blocks_ksp[1] == 0`
  - `delta_b[1] == 0 (implied)`
* Verification deliverable: `H1_ALL_ZERO_PROOF`
