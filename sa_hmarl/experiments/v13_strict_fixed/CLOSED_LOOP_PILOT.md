# Closed-Loop Pilot (COST239)

Fair closed-loop comparison on shared traffic seeds (4001–4005). Each method sees the exact same request sequence per seed. Evaluated requests = 2000 per seed after 500 warmup.

## Overall blocking rate

| Gate | Mode | Blocking % | Overload % | NSB % | Delay ms | FS | PPO agree |
|---|---|---:|---:|---:|---:|---:|---:|
| all | ppo_r_top1 | 5.63% ± 2.94% | 5.51% | 0.12% | 11.18 | 3.45 | 100.00% |
| all | ksp_ff_plain | 9.57% ± 1.55% | 9.27% | 0.30% | 10.50 | 3.17 | 26.77% |
| all | ksp_ff_highest | 6.44% ± 1.62% | 6.43% | 0.01% | 10.20 | 2.36 | 9.53% |
| all | ranker_old_v13 | 4.84% ± 1.40% | 4.78% | 0.06% | 13.05 | 3.17 | 7.26% |
| all | ranker_full_v13_pilot | 5.40% ± 3.29% | 5.16% | 0.24% | 14.72 | 3.42 | 9.36% |
| deep_path_only | ppo_r_top1 | 5.63% ± 2.94% | 5.51% | 0.12% | 11.18 | 3.45 | 100.00% |
| deep_path_only | ksp_ff_plain | 9.57% ± 1.55% | 9.27% | 0.30% | 10.50 | 3.17 | 26.77% |
| deep_path_only | ksp_ff_highest | 6.44% ± 1.62% | 6.43% | 0.01% | 10.20 | 2.36 | 9.53% |
| deep_path_only | ranker_old_v13 | 4.44% ± 1.32% | 4.36% | 0.08% | 13.04 | 3.33 | 30.83% |
| deep_path_only | ranker_full_v13_pilot | 5.82% ± 3.70% | 5.69% | 0.13% | 14.50 | 3.55 | 34.04% |

## Key comparisons

| Comparison | Absolute Δ blocking | Relative Δ |
|---|---|---:|
| Current v1.3 all vs PPO-R | -0.7900% | -14.03% |
| Gated v1.3 vs PPO-R | -1.1900% | -21.14% |
| Gated v1.3 vs Current v1.3 all | -0.4000% | -8.26% |
| Full v1.3 pilot all vs PPO-R | -0.2300% | -4.09% |
| Full v1.3 pilot all vs Current v1.3 all | 0.5600% | 11.57% |

## E-gate subgroup notes

The online subgroup counters record states where at least one PPO-R candidate exists. In this pilot, all such states resulted in immediate admission; blocking only occurs when no legal candidate is available. Therefore E=0/E=1 blocking rates within the ranker-invoked subgroups are 0%, while aggregate blocking is driven by no-candidate fall-through. This means the subgroup blocking-rate metric is not discriminative here; the gate effect is visible instead in overall blocking and PPO-agreement.

Observations:
- Current v1.3 all-gate improves over PPO-R (4.84% vs 5.63%).
- Gated Current v1.3 further improves over all-gate (4.44% vs 4.84%),   suggesting that bypassing the E=1-only ranker on E=0 states avoids OOD harm.
- Full v1.3 pilot all-gate (5.40%) does not beat Current v1.3 all-gate (4.84%) on this small pilot.
- Full v1.3 pilot has higher variance across seeds than Current v1.3.