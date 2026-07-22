# Experiment Matrix

| Method | K_path | |M| | max_blocks | Total Actions | Policy Head | Backbone | Training |
|---|---|---|---:|---:|---|---|---|
| Strict v1.3 | 50 | 4 | 10 | 2000 | N/A (PPO-R + ranker) | PPO-R + ranker MLP | Frozen checkpoints |
| KSP-FF K=50 hops | 50 | 4 | 10 | 2000 | N/A (rule-based) | N/A | N/A |
| Topology-matched adapted DeepRMSA K=50 hops | 50 | 4 | 10 | 2000 | 2000 | 5-layer 128-unit ELU MLP | A2C from scratch |

## Per-Seed Blocking Rates

| seed | Strict v1.3 | KSP-FF | DeepRMSA |
|---|---:|---:|---:|
| 5001 | 5.3833% | 5.3833% | 11.8667% |
| 5002 | 5.6667% | 5.6667% | 12.0500% |
| 5003 | 5.2667% | 5.2667% | 12.3333% |
| 5004 | 5.8333% | 5.8333% | 12.1333% |
| 5005 | 5.8167% | 5.8167% | 12.0333% |
| **overall** | **5.5933%** | **5.5933%** | **12.0833%** |
