# Smoke Test Results: Phase-R0 Expanded-Path Rescue Ceiling

| Topology | Seed | Evaluated | K50-empty events | Strict rescue success | Preferred rescue success |
|---|---|---:|---:|---:|---:|
| xlron_cost239_ptrnet_real | 5001 | 200 | 22 | 0 | 0 |
| xlron_nsfnet_deeprmsa | 6101 | 200 | 18 | 0 | 0 |
| xlron_usnet_gcnrmsa | 7101 | 200 | 12 | 0 | 0 |
| xlron_jpn48 | 8101 | 200 | 36 | 0 | 0 |

## Per-method blocking rates

### xlron_cost239_ptrnet_real
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.5000% | 200 | 189 | 11 | 0 | 0 |
| KSP-FF K=50 hops | 5.5000% | 200 | 189 | 11 | 0 | 0 |
| Strict v1.3 + K500 rescue | 5.5000% | 200 | 189 | 11 | 11 | 0 |
| KSP-FF K=50 hops + K500 rescue | 5.5000% | 200 | 189 | 11 | 11 | 0 |

### xlron_nsfnet_deeprmsa
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 4.5000% | 200 | 191 | 9 | 0 | 0 |
| KSP-FF K=50 hops | 4.5000% | 200 | 191 | 9 | 0 | 0 |
| Strict v1.3 + K500 rescue | 4.5000% | 200 | 191 | 9 | 9 | 0 |
| KSP-FF K=50 hops + K500 rescue | 4.5000% | 200 | 191 | 9 | 9 | 0 |

### xlron_usnet_gcnrmsa
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 3.0000% | 200 | 194 | 6 | 0 | 0 |
| FF-KSP K=50 hops | 3.0000% | 200 | 194 | 6 | 0 | 0 |
| Strict v1.3 + K500 rescue | 3.0000% | 200 | 194 | 6 | 6 | 0 |
| FF-KSP K=50 hops + K500 rescue | 3.0000% | 200 | 194 | 6 | 6 | 0 |

### xlron_jpn48
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 9.0000% | 200 | 182 | 18 | 0 | 0 |
| FF-KSP K=50 hops | 9.0000% | 200 | 182 | 18 | 0 | 0 |
| Strict v1.3 + K500 rescue | 9.0000% | 200 | 182 | 18 | 18 | 0 |
| FF-KSP K=50 hops + K500 rescue | 9.0000% | 200 | 182 | 18 | 18 | 0 |
