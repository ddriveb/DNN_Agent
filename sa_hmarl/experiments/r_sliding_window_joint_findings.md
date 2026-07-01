# Sliding-Window Joint RMSA Oracle Findings

## Question

Can we beat the per-request greedy ceiling by buffering a small request window and jointly choosing R actions?

This tests a different mechanism from v1.2:

- v1.2: one request arrives, rank legal R actions, execute one action.
- Sliding-window joint: buffer `N` requests, enumerate Top-K R choices in a small tree, execute the best sequence.

No model is trained.  This is an oracle/diagnostic mechanism probe.

## Important Protocol Note

This diagnostic buffers requests until the window is full.  The queueing time is added to reported delay, but this first probe does **not** reject a request again if the queueing time makes it exceed its deadline.

Therefore, it is an optimistic blocking probe and a realistic delay-cost probe.

## Scenario

- Topology: `snap24_gnutella_reach`
- Slots: 80
- Workload: `default3`
- Arrival interval: `0.15s`
- Holding time: `4-10s`
- Request size: `5-40MB`
- R-ranker: v1.2 mixed-low
- DeepRMSA: S80 checkpoint

## Results

### Window `N=2`, Top-K `K=2`

Seeds `3030,4040,5050`, 5 episodes per seed.

| Method | Blocking | NSB | Overload | Delay mean/P95 | Wait mean/P95 | Decision mean/P95 | Branches mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1.2 rank-only | 7.42% | 6.25% | 1.17% | 7.994/16.450 | 0.000/0.000 | 14.3/23.8 | 1.0/1.0 |
| window joint N2K2 | 7.33% | 6.08% | 1.25% | 85.224/163.545 | 75.000/150.000 | 29.4/42.8 | 3.7/4.0 |
| DeepRMSA | 7.75% | 6.67% | 1.08% | 9.906/18.621 | 0.000/0.000 | 11.0/14.7 | 1.0/1.0 |

### Window `N=3`, Top-K `K=2`

Seeds `3030,4040,5050`, 3 episodes per seed.

| Method | Blocking | NSB | Overload | Delay mean/P95 | Wait mean/P95 | Decision mean/P95 | Branches mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1.2 rank-only | 6.94% | 5.69% | 1.25% | 7.664/15.307 | 0.000/0.000 | 14.5/23.2 | 1.0/1.0 |
| window joint N3K2 | 7.22% | 5.97% | 1.25% | 160.864/311.895 | 148.125/300.000 | 42.4/62.1 | 7.1/8.0 |
| DeepRMSA | 7.22% | 6.11% | 1.11% | 9.500/17.742 | 0.000/0.000 | 11.2/15.0 | 1.0/1.0 |

## Interpretation

1. Window joint optimization does not produce a meaningful blocking gain.
   - `N=2,K=2` improves blocking by only `0.09 pp`.
   - `N=3,K=2` is worse than v1.2.

2. The delay cost is very large.
   - `N=2` adds about `75 ms` mean waiting and `150 ms` P95 waiting.
   - `N=3` adds about `148 ms` mean waiting and `300 ms` P95 waiting.

3. The tree search itself is computationally manageable for small windows, but the queueing delay dominates the tradeoff.

4. This result aligns with prior diagnostics:
   - v1.2 already saturates single-request R ranking.
   - Short online rollout did not change v1.2's choices.
   - C/R no-valid states and resource-envelope limits dominate the remaining blocking.

## Verdict

**STOP_SLIDING_WINDOW_JOINT_FOR_CURRENT_WORKLOAD.**

Sliding-window joint RMSA is an interesting mechanism, but under the current Poisson-like request stream and S80 workload, the small blocking benefit does not justify the waiting delay.

## Recommendation

Do not use fixed-size request-window batching as the next main method.

If revisiting this mechanism, use an event-triggered version instead:

- only open a window when the current request is at high blocking risk;
- cap queueing by deadline slack;
- allow immediate service when the v1.2 action is clearly safe;
- compare against deadline-aware waiting, because both are queueing mechanisms.

For the current paper direction, v1.2 remains the main R-side method. Larger gains likely require defragmentation or admission/waiting mechanisms, not joint batching.
