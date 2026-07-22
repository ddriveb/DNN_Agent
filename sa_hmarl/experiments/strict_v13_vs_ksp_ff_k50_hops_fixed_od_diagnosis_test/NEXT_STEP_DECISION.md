# Next Step Decision: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / fixed-OD)

## Result Summary

- Strict v1.3 blocking: 42.9000%
- KSP-FF K=50 hops blocking: 42.9000%
- Difference: 0.00 pp
- strict_win: 0
- ksp_win: 0
- KSP action in Strict Top-30 rate: 23.5500%

## Decision

**Case 3: Blocking rates are close.**

The fixed-OD setting may not create enough spectral pressure, or the chosen OD pair is not sensitive to the differences between the two policies.

Recommended follow-ups:
- Try a different fixed_src_node / fixed_server_id pair with higher spectral contention.
- Increase load (smaller arrival_interval or longer holding times).
- Verify that num_slots=320 is not overly generous for this OD pair.

## Fixed-C Verification

- no PPO-C loaded: True
- no PPO-C action selected: True
- fixed_src_node: 1
- fixed_split_id: 0
- fixed_server_id: 0
- fixed_dst_node: 0
