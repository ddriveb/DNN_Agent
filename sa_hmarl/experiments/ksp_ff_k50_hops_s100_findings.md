# KSP-FF K50 Hops S100 Findings

This experiment adds a tuned KSP-FF baseline inspired by the pure optical RMSA
setting: `K=50`, paths ordered by hop count, and first-fit spectrum allocation.
The goal is to test whether this stronger heuristic remains a competitive
R-side backend in the DNN offloading-driven compute-optical setting.

## Implementation

- Path generation now supports `sort_by="km"` and `sort_by="hops"`:
  `sa_hmarl/sa_hmarl/network/ksp.py`
- `SMDPEnv`, `make_env`, and C/R observation builders propagate
  `path_sort_strategy`.
- Main S100 evaluation adds the R backend `ksp_ff_k50_hops`.
- PPO-C is kept fixed on the main experiment setting (`K=5`, `km` order).
- Only the R-side observation/execution path set is widened to `K=50`,
  `hops` order for `ksp_ff_k50_hops`.
- The strict tuned baseline uses path-order scanning, highest feasible
  modulation (minimum required FS), and lowest-start-slot First-Fit.

Important correction: the first diagnostic used the historical framework
`ksp_ff_action`, which simply returned the first legal flat action.  In this
project that meant `mixed` block ordering and low-efficiency modulation could
be selected first.  That was not a strict Doherty-style tuned KSP-FF baseline.
The strict rerun below fixes this.

## S100 Result

Output files:

- JSON: `sa_hmarl/experiments/main_s100_ksp_ff_k50_hops_strict.json`
- Markdown: `sa_hmarl/experiments/main_s100_ksp_ff_k50_hops_strict.md`

Configuration:

- Topology: `snap24_gnutella_reach`
- Slots: `100`
- Seeds: `3030,4040,5050,6060,7070`
- Episodes per seed: `20`
- Requests per episode: `80`
- Base PPO-C path setting: `K=5`, `km` order
- Tuned heuristic R setting: `K=50`, `hops` order

| Method | Blocking | NSB | Overload | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|
| PPO-C + final v1.2 | 0.92% | 0.84% | 0.09% | 8.377/16.304 ms | 12.253/21.055 ms |
| PPO-C + historical KSP-FF K=5/km | 7.38% | 5.94% | 1.44% | 8.201/16.280 ms | 8.684/12.104 ms |
| PPO-C + strict KSP-FF K=50/hops | 1.05% | 0.94% | 0.11% | 7.919/15.229 ms | 10.056/14.604 ms |
| PPO-C + DeepRMSA | 1.29% | 1.21% | 0.07% | 10.183/18.040 ms | 9.145/12.607 ms |

## Relative Improvement

Against `KSP-FF K=50/hops`, final v1.2 reduces blocking by:

```text
(1.05% - 0.92%) / 1.05% = 11.90%
```

Against DeepRMSA, final v1.2 reduces blocking by:

```text
(1.29% - 0.92%) / 1.29% = 28.16%
```

## Interpretation

The strict `K=50/hops` KSP-FF baseline is much stronger than the historical
framework KSP-FF baseline (`1.05%` vs `7.38%`) and also stronger than DeepRMSA
under the fixed PPO-C setup (`1.05%` vs `1.29%`).  This is the fairer
heuristic ceiling to report.

final v1.2 still beats this tuned heuristic, but the margin is now modest:
`0.92%` vs `1.05%`, or about `11.90%` relative blocking reduction.  This is a
stronger and more credible story than comparing only against the weak
historical KSP-FF implementation.

This supports the paper/PPT narrative:

- The tuned pure-optical heuristic remains highly competitive when implemented
  carefully.
- final v1.2 still outperforms it under the same PPO-C traffic-shaping policy.
- The gain over the strongest heuristic should be reported as a careful,
  incremental improvement, not an order-of-magnitude gap.
- The large gap remains valid only against the historical weak KSP-FF baseline.
