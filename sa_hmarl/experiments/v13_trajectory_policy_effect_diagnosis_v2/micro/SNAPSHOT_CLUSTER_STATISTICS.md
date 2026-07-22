# Snapshot-Cluster Statistics

*Per-snapshot mean ΔB across future traces; inference treats snapshots as independent clusters.*

| H | N | mean ΔB | std | harmful | beneficial | 95% CI | perm p (raw) | perm p (Holm) | TOST ±0.05 |
|---|:-:|--------:|----:|--------:|-----------:|--------|-------------:|--------------:|:----------:|
|   1 |  10 |  +0.000 | 0.000 |   0.00% |      0.00% | [+0.000, +0.000] |       1.0000 |        1.0000 | no         |
|   5 |  10 |  +0.000 | 0.000 |   0.00% |      0.00% | [+0.000, +0.000] |       1.0000 |        1.0000 | no         |
|  20 |  10 |  +0.050 | 0.158 |  10.00% |      0.00% | [+0.000, +0.150] |       1.0000 |        1.0000 | no         |
|  50 |  10 |  +0.000 | 0.236 |  10.00% |     10.00% | [-0.150, +0.150] |       1.0000 |        1.0000 | no         |
| 100 |  10 |  +0.000 | 0.577 |  40.00% |     20.00% | [-0.350, +0.300] |       1.0000 |        1.0000 | no         |

Methods:
* **Cluster bootstrap CI**: 20,000 resamples of snapshots with replacement.
* **Sign-flip permutation test**: 20,000 random sign flips of per-snapshot means; tests H0: mean ΔB = 0.
* **Holm correction**: applied across the five horizon tests.
* **TOST equivalence**: two one-sided t-tests for equivalence region ±0.05 blocks per snapshot.
