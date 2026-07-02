# XLRON Topology R-Backend Comparison

Fixed setting:

- C policy: `ppo_c`
- Slots: `100`
- Split profile: `default3`
- Seeds: `3030,4040,5050,6060,7070`
- Episodes per seed: `20`
- Requests per episode: `80`
- v1.2 base path setting: `k_paths=5`, `path_sort_strategy=km`
- KSP-FF tuned baseline: `K=50`, `path_sort_strategy=hops`

DeepRMSA note:

- DeepRMSA was not run on these four XLRON topologies because the available
  DeepRMSA checkpoints are topology-specific and the loader rejects node-count
  mismatches (`DeepRMSA checkpoint topology mismatch`).
- A fair DeepRMSA comparison requires retraining one checkpoint per topology.

## Results

| Topology | R backend | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | Decision mean/P95 |
|---|---|---:|---:|---:|---:|---:|---:|
| `xlron_cost239_ptrnet_real` | v1.2 | **0.20%** | 0.20% | 0.00% | 0.20% | 11.060/20.027 ms | 10.890/16.479 ms |
| `xlron_cost239_ptrnet_real` | KSP-FF K50 hops | 3.95% | 0.00% | 0.00% | 0.00% | 10.294/19.427 ms | 17.818/24.355 ms |
| `xlron_german17` | v1.2 | **0.97%** | 0.97% | 0.27% | 0.70% | 13.317/23.363 ms | 14.597/22.793 ms |
| `xlron_german17` | KSP-FF K50 hops | 2.96% | 1.31% | 2.19% | 0.59% | 13.014/22.640 ms | 24.902/32.221 ms |
| `xlron_nsfnet_deeprmsa` | v1.2 | **3.09%** | 3.15% | 0.76% | 2.18% | 19.526/33.812 ms | 7.748/10.932 ms |
| `xlron_nsfnet_deeprmsa` | KSP-FF K50 hops | 9.55% | 2.24% | 2.25% | 1.40% | 18.555/32.276 ms | 20.250/26.999 ms |
| `xlron_jpn48` | v1.2 | **6.19%** | 6.19% | 4.59% | 1.05% | 16.066/28.934 ms | 28.237/43.844 ms |
| `xlron_jpn48` | KSP-FF K50 hops | 9.70% | 4.26% | 4.40% | 0.99% | 15.142/27.527 ms | 76.292/128.185 ms |

## Blocking Gap

| Topology | KSP-FF K50 - v1.2 | Relative reduction by v1.2 |
|---|---:|---:|
| `xlron_cost239_ptrnet_real` | +3.75 pp | 94.9% |
| `xlron_german17` | +1.99 pp | 67.2% |
| `xlron_nsfnet_deeprmsa` | +6.46 pp | 67.6% |
| `xlron_jpn48` | +3.51 pp | 36.2% |

## Takeaway

Across the four imported XLRON topologies, v1.2 consistently beats the tuned
KSP-FF K=50 hops baseline under the same PPO-C traffic-shaping policy.  The
absolute blocking rate is topology-dependent: COST239 is easy, German17 is
close to the previous S100 range, while NSFNET and JPN48 are substantially
harder under the same request/load configuration.
