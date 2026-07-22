# Topology Comparison Summary: Strict v1.3 vs Heuristics (fixed-C / all-OD)

| Topology | Nodes | Links | Strict blocking | Main heuristic | Main blocking | Aux heuristic | Aux blocking | Strict-main pp | Strict-aux pp |
|---|---:|---:|---:|---|---|---|---:|---:|---:|
| xlron_nsfnet_deeprmsa | 14 | 22 | 5.3733% | KSP-FF K=50 hops | 5.3600% | FF-KSP K=50 hops | 5.3667% | +0.01 | +0.01 |
| xlron_usnet_gcnrmsa | 24 | 43 | 5.4400% | FF-KSP K=50 hops | 5.4400% | KSP-FF K=50 hops | 5.4400% | +0.00 | +0.00 |
| xlron_jpn48 | 48 | 82 | 5.6067% | FF-KSP K=50 hops | 5.5933% | KSP-FF K=50 hops | 5.5933% | +0.01 | +0.01 |

## Interpretation

Negative `Strict-main pp` / `Strict-aux pp` means Strict v1.3 has lower blocking. Positive means the heuristic has lower blocking. The topology with the largest negative gap is where Strict v1.3's cross-topology frozen transfer provides the strongest spectral-trajectory advantage.
