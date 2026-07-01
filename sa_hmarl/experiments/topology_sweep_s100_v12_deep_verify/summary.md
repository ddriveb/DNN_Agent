# S100 V1.2 vs DeepRMSA Topology Verification

5 seeds x 10 episodes x 80 requests, 100 slots, default3 workload.

| Topology | v1.2 Blocking | DeepRMSA Blocking | Gap (Deep-v1.2) | v1.2 Raw | Deep Raw | v1.2 Overload | Deep Overload | v1.2 Delay | Deep Delay |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| metro24 | 0.00% | 0.00% | +0.00 pp | 0.00% | 0.00% | 0.00% | 0.00% | 7.933 | 8.368 |
| metro24_bottleneck | 0.30% | 0.33% | +0.03 pp | 0.30% | 0.33% | 0.30% | 0.33% | 11.046 | 11.208 |
| metro24_c_sensitive | 0.00% | 0.00% | +0.00 pp | 0.00% | 0.00% | 0.00% | 0.00% | 9.602 | 10.071 |
| snap24_gnutella_reach | 0.80% | 1.07% | +0.27 pp | 0.80% | 1.07% | 0.10% | 0.10% | 8.346 | 9.982 |