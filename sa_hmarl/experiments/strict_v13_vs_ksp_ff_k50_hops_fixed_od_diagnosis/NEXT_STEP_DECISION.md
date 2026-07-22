# Next Step Decision: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / fixed-OD)

## Result Summary

- Strict v1.3 blocking: 43.9067%
- KSP-FF K=50 hops blocking: 43.9067%
- Difference: 0.00 pp
- strict_win: 0
- ksp_win: 0
- KSP action in Strict Top-30 rate: 20.7142%
- Seeds: 20/20 identical blocking

## Decision

**Case 3: Blocking rates are identical.**

Under fixed-C / fixed-OD, Strict v1.3 and KSP-FF K=50 hops produce exactly the same blocking rate across all 20 seeds. The difference is 0.00 pp with strict_win=0 and ksp_win=0.

### Key observation from request-level trace

- All 10,739 requests where the two methods selected **different R actions** resulted in **both_success**.
- All blocked requests (both_block) failed with reason `r_no_valid_action`, i.e. the R-side legal mask was empty.
- No request was blocked because the chosen action was infeasible or suboptimal; blocking occurs only when no legal (path, modulation, block) tuple exists.

### Interpretation

The identical blocking rate is **not** because the two policies make the same choices. They do not:

- Strict v1.3 prefers longer paths (avg path idx 11.77, avg hops 3.23, avg path km 1756.51).
- KSP-FF K=50 hops prefers the shortest path (avg path idx 0.00, avg hops 1.00, avg path km 1310.00).
- Strict v1.3 also selects higher-waste blocks (avg block waste 0.9820 vs 0.2227).

The identical blocking rate is because, under fixed-OD, the bottleneck is **spectrum exhaustion on the fixed OD pair**, not action quality. When spectrum is available, both policies find a feasible action; when spectrum is exhausted, neither policy can find a legal action. The R-side decision has no degree of freedom to avoid blocking in this regime.

## Recommended follow-ups

1. **OD sensitivity check**: The current OD pair (fixed_src_node=1 -> fixed_dst_node=0) may not be sensitive to R-side policy differences. Try a longer or more congested OD pair where path choice has a larger impact on spectrum lifetime.

2. **Load regime check**: arrival_interval=0.5 with holding 20-30 produces 43.9% blocking driven purely by r_no_valid_action. Consider a lower blocking regime (e.g. 5-15%) where action quality, not just spectrum availability, can influence outcomes.

3. **Action-quality isolation**: To test whether Strict v1.3's longer-path preference hurts or helps, evaluate a mixed-OD or PPO-C setting where R actions influence future C-side decisions and server load. The fixed-OD setting removes that coupling.

4. **Candidate-pool audit**: KSP-FF K=50 hops action is inside Strict v1.3's PPO-R Top-30 candidate pool in only 20.71% of requests. If future work shows KSP-FF outperforming Strict v1.3 under a different setting, this low coverage is the first place to investigate.

## Fixed-C Verification

- no PPO-C loaded: True
- no PPO-C action selected: True
- fixed_src_node: 1
- fixed_split_id: 0
- fixed_server_id: 0
- fixed_dst_node: 0
