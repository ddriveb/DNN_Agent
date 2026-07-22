# Snapshot-Cluster Statistics

*Per-snapshot mean ΔB across future traces; inference treats snapshots as independent clusters.*

| H | N | mean ΔB | std | harmful | beneficial | 95% CI | perm p (raw) | perm p (Holm) | TOST ±0.05 |
|---|:-:|--------:|----:|--------:|-----------:|--------|-------------:|--------------:|:----------:|
|   1 | 301 |  +0.000 | 0.000 |   0.00% |      0.00% | [+0.000, +0.000] |       1.0000 |        1.0000 | yes        |
|   5 | 301 |  +0.000 | 0.067 |   3.32% |      3.32% | [-0.008, +0.007] |       1.0000 |        1.0000 | yes        |
|  20 | 301 |  +0.041 | 0.150 |  20.60% |      6.64% | [+0.023, +0.058] |       0.0000 |        0.0002 | no         |
|  50 | 301 |  +0.020 | 0.210 |  21.26% |     16.61% | [-0.003, +0.044] |       0.1073 |        0.4292 | yes        |
| 100 | 301 |  +0.017 | 0.333 |  28.90% |     25.25% | [-0.021, +0.054] |       0.4069 |        1.0000 | yes        |

Methods:
* **Cluster bootstrap CI**: 20,000 resamples of snapshots with replacement.
* **Sign-flip permutation test**: 20,000 random sign flips of per-snapshot means; tests H0: mean ΔB = 0.
* **Holm correction**: applied across the five horizon tests.
* **TOST equivalence**: two one-sided t-tests for equivalence region ±0.05 blocks per snapshot.
