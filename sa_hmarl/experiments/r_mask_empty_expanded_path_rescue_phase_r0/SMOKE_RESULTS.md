# Smoke Test Results: Phase-R0 Expanded-Path Rescue Ceiling

| Topology | Seed | Evaluated | K50-empty events | Strict rescue success | Preferred rescue success |
|---|---|---:|---:|---:|---:|
| xlron_cost239_ptrnet_real | 5001 | 2000 | 228 | 0 | 0 |
| xlron_nsfnet_deeprmsa | 6101 | 2000 | 245 | 0 | 0 |
| xlron_usnet_gcnrmsa | 6101 | 2000 | 258 | 0 | 0 |
| xlron_jpn48 | 6101 | 2000 | 236 | 0 | 0 |

## Per-method blocking rates

### xlron_cost239_ptrnet_real
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.7000% | 2000 | 1886 | 114 | 0 | 0 |
| KSP-FF K=50 hops | 5.7000% | 2000 | 1886 | 114 | 0 | 0 |
| Strict v1.3 + K500 rescue | 5.7000% | 2000 | 1886 | 114 | 114 | 0 |
| KSP-FF K=50 hops + K500 rescue | 5.7000% | 2000 | 1886 | 114 | 114 | 0 |

### xlron_nsfnet_deeprmsa
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 6.1000% | 2000 | 1878 | 122 | 0 | 0 |
| KSP-FF K=50 hops | 6.1500% | 2000 | 1877 | 123 | 0 | 0 |
| Strict v1.3 + K500 rescue | 6.1000% | 2000 | 1878 | 122 | 122 | 0 |
| KSP-FF K=50 hops + K500 rescue | 6.1500% | 2000 | 1877 | 123 | 123 | 0 |

### xlron_usnet_gcnrmsa
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 6.4500% | 2000 | 1871 | 129 | 0 | 0 |
| FF-KSP K=50 hops | 6.4500% | 2000 | 1871 | 129 | 0 | 0 |
| Strict v1.3 + K500 rescue | 6.4500% | 2000 | 1871 | 129 | 129 | 0 |
| FF-KSP K=50 hops + K500 rescue | 6.4500% | 2000 | 1871 | 129 | 129 | 0 |

### xlron_jpn48
| Method | Blocking | Total | Admitted | Blocked | Rescue attempted | Rescue success |
|---|---|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.9000% | 2000 | 1882 | 118 | 0 | 0 |
| FF-KSP K=50 hops | 5.9000% | 2000 | 1882 | 118 | 0 | 0 |
| Strict v1.3 + K500 rescue | 5.9000% | 2000 | 1882 | 118 | 118 | 0 |
| FF-KSP K=50 hops + K500 rescue | 5.9000% | 2000 | 1882 | 118 | 118 | 0 |
