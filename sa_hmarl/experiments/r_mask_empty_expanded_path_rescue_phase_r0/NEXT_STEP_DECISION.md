# Next Step Decision: Phase-R0 Expanded-Path Rescue Ceiling

## Aggregate results

| Topology | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| xlron_cost239_ptrnet_real | 5.5933% | 5.5933% | 5.5933% | 5.5933% |
| xlron_nsfnet_deeprmsa | 5.3733% | 5.3600% | 5.3733% | 5.3600% |
| xlron_usnet_gcnrmsa | 5.4400% | 5.4400% | 5.4400% | 5.4400% |
| xlron_jpn48 | 5.6067% | 5.5933% | 5.6067% | 5.5933% |

## Interpretation

> **Decision: STOP Phase-R0 and do not expand static path pool.**

The evidence shows that K=500 rescue availability is **exactly 0** across all four topologies. The binding constraint is not that the 50 shortest paths are missing a feasible route; the bottleneck is spectrum/contention or reach/deadline on the available simple paths. Expanding to K=1000 or beyond would not address this and is not recommended without new evidence.

Recommended next step (if continued): investigate **one-active-connection defragmentation / reconfiguration rescue** or **load-regime / spectrum-efficiency** improvements, not larger static KSP pools.
