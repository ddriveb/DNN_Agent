# Optimized Profile

The optimized selector reuses the PPO-R action-feature matrix and constructs
the 30 Ranker rows as one batch. No model, feature, normalization, candidate,
or action semantics changed.

| Topology | Optimized selector | Selector speedup | Optimized end-to-end | E2E reduction |
|---|---:|---:|---:|---:|
| NSFNET | 3.362 ms | 3.06x | 8.119 ms | 46.0% |
| USNET | 3.752 ms | 4.78x | 11.454 ms | 55.4% |
| JPN48 | 4.380 ms | 4.98x | 17.081 ms | 50.1% |

The Ranker forward itself takes only 0.084-0.114 ms on non-empty decisions.
The next exact optimization target is PPO-R/observation feature construction,
not Ranker compression.
