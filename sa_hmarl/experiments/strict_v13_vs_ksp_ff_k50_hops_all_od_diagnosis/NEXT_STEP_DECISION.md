# Next Step Decision: Strict v1.3 vs KSP-FF K=50 hops (fixed-C / all-OD)

## Result Summary

- Strict v1.3 blocking: 5.6342%
- KSP-FF K=50 hops blocking: 5.6342%
- Difference: 0.00 pp
- strict_win: 0
- ksp_win: 0
- KSP action in Strict Top-30 rate: 32.7117%
- Seeds: 20/20 identical blocking

## Deep-dive: Why is blocking identical despite different action preferences?

### 1. Action preferences differ substantially

| Metric | Strict v1.3 | KSP-FF K=50 hops |
|---|---|---|
| Avg path idx | 4.38 | 0.00 |
| Avg hops | 1.95 | 1.45 |
| Avg path km | 976.81 | 831.55 |
| Avg block start | 63.56 | 8.17 |
| Avg block waste | 0.9666 | 0.4755 |
| Avg required FS | 2.08 | 2.06 |
| Avg decision ms | 13.47 | 3.14 |

Strict v1.3 consistently chooses longer paths (higher path_idx, more hops, longer km) and higher-start, higher-waste blocks. KSP-FF K=50 hops always chooses the first hops-ordered path and the lowest-start First-Fit block.

### 2. Yet outcomes are identical

- All 18,586 requests with **different R actions** resulted in **both_success**.
- All blocked requests (both_block) failed with reason `r_no_valid_action` (empty R mask).
- No request was blocked because the chosen action was infeasible or suboptimal.

### 3. Interpretation

Under fixed-C / all-OD with arrival_interval=0.3, the bottleneck is **spectrum availability**, not **action quality**:

- When spectrum is available, both policies find a feasible action. Strict v1.3 may pick a longer path or a higher-start block, but that action still succeeds.
- When spectrum is exhausted, neither policy can find a legal (path, modulation, block) tuple. The R mask is empty, so the request blocks regardless of which policy is in control.

In this regime, the R-side decision has **no degree of freedom to avoid blocking**. The identical blocking rate is a property of the environment, not of the policies.

### 4. Candidate-pool coverage

KSP-FF K=50 hops action is inside Strict v1.3's PPO-R Top-30 candidate pool in **32.71%** of evaluated requests. This is higher than in the fixed-OD setting (20.71%) because all-OD traffic produces more diverse path/mod combinations, giving PPO-R more candidates to rank.

## Decision

**Case 3: Blocking rates are identical.**

The fixed-C / all-OD setting with arrival_interval=0.3 produces ~5.6% blocking driven purely by `r_no_valid_action`. The two policies cannot be distinguished by blocking rate in this regime.

## Recommended follow-ups

1. **Regime shift**: To make action quality matter, move to a regime where blocking is caused by suboptimal actions, not empty masks. This requires either:
   - Lower load (blocking < 1%) where almost all actions succeed, and differences appear in delay/FS efficiency.
   - A topology with tighter spectrum (fewer slots) or longer paths where path choice has a larger impact on feasibility.

2. **Coupled C+R setting**: Re-enable PPO-C. In the coupled setting, R actions change server load and future C-side observations, so Strict v1.3's trajectory-level advantage can emerge. The fixed-C setting removes that coupling.

3. **Delay/FS efficiency analysis**: Even with identical blocking, Strict v1.3 uses more FS (2.08 vs 2.06) and longer paths (976.81 km vs 831.55 km). Analyze whether this translates into better future spectrum availability in a coupled setting.

4. **Candidate-pool audit**: 32.71% coverage means that in 67.29% of requests, the KSP-FF action is not in Strict v1.3's Top-30. If future work shows KSP-FF outperforming Strict v1.3, this coverage gap is the first place to investigate.

## Fixed-C Verification

- no PPO-C loaded: True
- no PPO-C action selected: True
- traffic_matrix: uniform_all_od
- fixed_split_id: 0
- server_node_ids: [0, 1, 2, 3]
