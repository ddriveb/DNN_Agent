# Next Step Decision: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / fixed-OD)

## Result Summary

- Strict v1.3 blocking: 5.6133%
- KSP-FF K=50 hops blocking: 5.5933%
- Difference: 0.02 pp
- strict_win: 25
- ksp_win: 31
- KSP action in Strict Top-30 rate: 14.6533%

## Decision

**Case 3: Blocking rates are close.**

The fixed-OD setting may not create enough spectral pressure, or the chosen OD pair is not sensitive to the differences between the two policies.

Recommended follow-ups:
- Try a different traffic matrix or OD distribution with higher spectral contention.
- Increase load (smaller arrival_interval or longer holding times).
- Verify that num_slots=320 is not overly generous for this OD pair.

## Fixed-C Verification

- no PPO-C loaded: True
- no PPO-C action selected: True
- traffic_matrix: uniform_all_od
- fixed_split_id: 0
- server_node_ids: [0, 1, 2, 3]
