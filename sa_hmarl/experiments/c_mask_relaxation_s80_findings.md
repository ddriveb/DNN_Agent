# C-Mask Relaxation Findings on S80

## Goal

Test whether relaxing Agent-C's server-utilization mask can reduce blocking in the S80 medium-low-blocking scenario.

The relaxation keeps the important physical guard:

- `feasible_count > 0` is still required.

Only the server utilization cutoff is relaxed:

- strict: `util_threshold = 0.95`
- relaxed: `util_threshold in {1.00, 1.05, 1.10}`

This avoids admitting candidates with no downstream R action.

## Setup

- Topology: `snap24_gnutella_reach`
- Slots: 80
- Workload: `default3`, arrival interval `0.15s`, holding time `4-10s`, size `5-40MB`
- Seeds: `3030,4040,5050`
- Episodes: `5` per seed
- Requests: `80` per episode
- R methods: v1.2 counterfactual ranker and DeepRMSA-S80

## Results

| Method | Blocking | Raw empty | Added candidates | Selected relaxed-only | NSB | Overload | Delay mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| v1.2 strict | 7.42% | 7.42% | 0.00% | 0.00% | 6.25% | 1.17% | 7.994/16.450 |
| v1.2 relax-policy u=1.00 | 7.42% | 7.42% | 7.50% | 2.00% | 6.33% | 1.08% | 8.145/16.963 |
| v1.2 relax-policy u=1.05 | 7.42% | 7.42% | 7.50% | 2.00% | 6.33% | 1.08% | 8.145/16.963 |
| v1.2 relax-policy u=1.10 | 7.42% | 7.42% | 7.50% | 2.00% | 6.33% | 1.08% | 8.145/16.963 |
| v1.2 relax-maxR u=1.00 | 8.83% | 8.83% | 16.50% | 7.08% | 8.33% | 0.50% | 10.720/21.426 |
| DeepRMSA strict | 7.75% | 7.75% | 0.00% | 0.00% | 6.67% | 1.08% | 9.906/18.621 |
| DeepRMSA relax-policy u=1.00 | 7.92% | 7.92% | 8.00% | 2.92% | 6.83% | 1.08% | 10.095/19.441 |
| DeepRMSA relax-maxR u=1.00 | 12.50% | 12.50% | 13.67% | 6.00% | 12.08% | 0.42% | 14.230/24.841 |

The `1.05` and `1.10` results are identical to `1.00`, which indicates that useful relaxed candidates are already captured by moving from `0.95` to `1.00`; further relaxation does not add effective choices in this workload.

## Interpretation

1. Relaxing the utilization threshold does create additional C candidates in about `7-8%` of requests.

2. PPO-C selects relaxed-only candidates rarely (`2-3%`), and this does not reduce blocking.

3. The max-R oracle-style C choice is worse. This is important: simply choosing the split/server with more immediate R actions is not a safe proxy for lower future blocking.

4. Relaxation reduces overload slightly in the max-R setting, but it increases no-suitable-block and delay enough to make total blocking worse.

## Verdict

**FAIL.**

C-mask relaxation is not a good next mechanism in this scenario. The strict C-mask is not leaving a meaningful set of useful candidates unused. Relaxed candidates exist, but selecting them changes the trajectory in a harmful direction.

## Recommendation

Do not continue threshold-only C-mask relaxation.

If we revisit C-side mechanisms, they should not be simple mask relaxation. They need one of:

- a trained C objective that predicts long-horizon success under the selected R policy;
- deadline-aware waiting/queueing to handle temporary infeasibility;
- active defragmentation or admission control for states where the raw C-mask is genuinely empty.
