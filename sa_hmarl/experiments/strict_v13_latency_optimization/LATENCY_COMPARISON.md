# Strict v1.3 Latency Comparison

All latency runs used CPU and serial method execution. Diagnostic-only heuristic coverage calls were excluded from the production selector timing. Values are means over three repeats of 1,000 evaluated requests after 500 warmup requests.

| Topology | Reference selector | Optimized selector | Selector speedup | Reference E2E | Optimized E2E | E2E reduction | KSP-FF E2E | FF-KSP E2E |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| NSFNET | 10.277 ms | 3.362 ms | 3.06x | 15.044 ms | 8.119 ms | 46.0% | 4.111 ms | 4.085 ms |
| USNET | 17.934 ms | 3.752 ms | 4.78x | 25.670 ms | 11.454 ms | 55.4% | 6.424 ms | 6.699 ms |
| JPN48 | 21.818 ms | 4.380 ms | 4.98x | 34.203 ms | 17.081 ms | 50.1% | 10.659 ms | 10.420 ms |

## Optimized Non-empty Decision Components

| Topology | PPO feature build | Ranker feature build | Ranker forward |
|---|---:|---:|---:|
| NSFNET | 1.512 ms | 0.322 ms | 0.084 ms |
| USNET | 1.410 ms | 0.522 ms | 0.096 ms |
| JPN48 | 1.854 ms | 0.677 ms | 0.114 ms |

The Ranker MLP forward is not the bottleneck. Remaining cost is dominated by PPO-R action-feature construction, PPO proposal work, and request observation construction.
