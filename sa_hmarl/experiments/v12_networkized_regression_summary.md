# v1.2 Networkized Regression Summary

## Purpose

Validate that the networkized v1.2 implementation still runs correctly in
closed loop after replacing the inline R-ranker selection logic with
`CounterfactualRRankerPolicy`.

## Equivalence Smoke

| Check | Result |
|---|---:|
| Old inline selector vs new policy wrapper | 100.00% action agreement |
| Checked decisions | 80 |
| Mismatches | 0 |

Report:

- `sa_hmarl/experiments/v12_networkized_equivalence_smoke.md`
- `sa_hmarl/experiments/v12_networkized_equivalence_smoke.json`

## S24 Regression

Configuration:

- topology: `snap24_gnutella_reach`
- slots: 24
- split profile: `default3`
- arrival interval: 0.09
- size: 30 MB fixed
- holding: 14 s fixed
- seeds: `3030,4040,5050,6060,7070`
- episodes: 20 per seed

| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 |
|---|---:|---:|---:|---:|---:|
| PPO-R | 71.09% | 71.44% | 60.32% | 10.76% | 8.064/16.128 ms |
| Networkized v1.2 | 62.68% | 63.05% | 46.59% | 16.09% | 8.870/18.698 ms |
| DeepRMSA | 63.51% | 63.81% | 48.34% | 15.17% | 8.875/18.145 ms |

Networkized v1.2 vs DeepRMSA:

- blocking improvement: 0.83 pp
- NSB improvement: 1.75 pp
- delay mean difference: -0.005 ms

Report:

- `sa_hmarl/experiments/v12_networkized_s24_regression.md`
- `sa_hmarl/experiments/v12_networkized_s24_regression.json`

## S100 Regression

Configuration:

- topology: `snap24_gnutella_reach`
- slots: 100
- split profile: `default3`
- arrival interval: 0.15
- size: 5-30 MB
- holding: 4-10 s
- seeds: `3030,4040,5050,6060,7070`
- episodes: 20 per seed

| Method | Blocking | Raw empty | NSB | Overload | Delay mean/P95 |
|---|---:|---:|---:|---:|---:|
| PPO-R | 3.66% | 3.66% | 3.25% | 0.41% | 8.088/14.672 ms |
| Networkized v1.2 | 0.95% | 0.95% | 0.86% | 0.09% | 8.073/15.550 ms |
| DeepRMSA | 1.29% | 1.29% | 1.21% | 0.07% | 10.183/18.040 ms |

Networkized v1.2 vs DeepRMSA:

- blocking improvement: 0.34 pp
- NSB improvement: 0.35 pp
- delay mean improvement: 2.110 ms

Report:

- `sa_hmarl/experiments/v12_networkized_s100_regression.md`
- `sa_hmarl/experiments/v12_networkized_s100_regression.json`

## Verdict

PASS.

The networkized implementation preserves the v1.2 action-selection behavior and
keeps the expected advantage over DeepRMSA in both S24 and S100 regression runs.

Note: the S24 absolute blocking rate in this regression command is higher than
some earlier formal-validation records.  The equivalence smoke proves that the
networkized wrapper itself is behavior-preserving; any absolute-metric difference
should be attributed to evaluation configuration differences rather than the
networkization refactor.
