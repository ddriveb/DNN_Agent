# DeepRMSA Baseline Comparison

This report compares the newly added PyTorch DeepRMSA-style A2C/A3C baseline with existing SA-HMARL baselines on `metro24_c_sensitive`.

## Setup

All commands were run from:

```bash
cd /mnt/d/project/DNN_Agent
export PYTHONPATH=/mnt/d/project/DNN_Agent/sa_hmarl
```

Traffic and topology setting:

```text
topology              = metro24_c_sensitive
num_slots             = 24
num_servers           = 4
requests_per_episode  = 60
arrival_interval      = 0.25
holding_time          = [4, 10]
deadline_ms           = [30, 100]
size_mb               = [5, 30]
edge_compute_cost     = [0.5, 15]
slot_bw_hz            = 1.25e9
guard_band_fs         = 1
```

DeepRMSA was trained as a topology-specific checkpoint because its state contains source/destination node one-hot vectors.

```bash
/mnt/d/project/DNN_Agent/.venv/bin/python -u -m sa_hmarl.training.train_deep_rmsa \
  --topology metro24_c_sensitive \
  --num_slots 24 \
  --num_servers 4 \
  --episodes 1000 \
  --requests_per_episode 60 \
  --arrival_interval 0.25 \
  --holding_min 4 \
  --holding_max 10 \
  --deadline_min 30 \
  --deadline_max 100 \
  --size_min_mb 5 \
  --size_max_mb 30 \
  --edge_cost_min 0.5 \
  --edge_cost_max 15 \
  --slot_bw_hz 1.25e9 \
  --guard_band_fs 1 \
  --mixed_splits \
  --lr 1e-4 \
  --checkpoint_name deep_rmsa_c_sensitive_mixed.pt
```

Checkpoint:

```text
sa_hmarl/checkpoints/deep_rmsa_c_sensitive_mixed.pt
```

## System-Level Unified Comparison

Evaluation scale: `5 seeds x 20 episodes x 60 requests = 6000 requests / method`.

Command:

```bash
/mnt/d/project/DNN_Agent/.venv/bin/python -u -m sa_hmarl.evaluation.eval_unified \
  --topologies metro24_c_sensitive \
  --seeds 42,123,456,789,2024 \
  --episodes 20 \
  --requests_per_episode 60 \
  --arrival_interval 0.25 \
  --holding_min 4 \
  --holding_max 10 \
  --deadline_min 30 \
  --deadline_max 100 \
  --size_min_mb 5 \
  --size_max_mb 30 \
  --edge_cost_min 0.5 \
  --edge_cost_max 15 \
  --slot_bw_hz 1.25e9 \
  --guard_band_fs 1 \
  --num_slots 24 \
  --num_servers 4 \
  --stage2_c_checkpoint sa_hmarl/checkpoints/agent_c_frozen_r_c_sensitive_best.pt \
  --stage2_r_checkpoint sa_hmarl/checkpoints/joint_mappo_v2_cshape_metro24_fs002_r_best.pt \
  --stage3_v3_c_checkpoint sa_hmarl/checkpoints/stage3_v3_c_sensitive_c_best.pt \
  --stage3_v3_r_checkpoint sa_hmarl/checkpoints/stage3_v3_c_sensitive_r_best.pt \
  --ppo_r_checkpoint sa_hmarl/checkpoints/joint_mappo_v2_cshape_metro24_fs002_r_best.pt \
  --deep_rmsa_checkpoint sa_hmarl/checkpoints/deep_rmsa_c_sensitive_mixed.pt \
  --output_json sa_hmarl/experiments/metro24_c_sensitive_deeprmsa_eval.json
```

### Main Results

| Method | Blocking | Success | AvgReward | Delay(ms) | AvgFS | Waste | Path(km) |
|---|---:|---:|---:|---:|---:|---:|---:|
| Compute-Greedy + DeepRMSA-A2C | 0.004 | 0.996 | +0.507 | 6.6 | 2.05 | 0.511 | 210.8 |
| Stage2 Frozen-R+PPO-C | 0.008 | 0.992 | +0.513 | 7.2 | 2.10 | 0.487 | 202.4 |
| Stage3 v3 MAPPO KL=0.05 | 0.011 | 0.989 | +0.474 | 7.6 | 2.07 | 0.505 | 203.8 |
| Stage3 v4 MAPPO KL=0.01 | 0.014 | 0.986 | +0.502 | 7.4 | 2.08 | 0.479 | 196.7 |
| Compute-Greedy + PPO-R | 0.023 | 0.977 | +0.498 | 7.1 | 2.14 | 0.485 | 225.8 |
| Separate C+R (DQN) | 0.027 | 0.973 | +0.434 | 6.5 | 2.28 | 0.554 | 159.1 |
| Compute-Greedy + KSP-BF | 0.092 | 0.908 | +0.454 | 6.5 | 2.97 | 0.406 | 194.1 |

### Key Distributions

| Method | Main modulation behavior | Main server behavior |
|---|---|---|
| Stage2 Frozen-R+PPO-C | 16QAM 47.0%, 8QAM 41.5%, QPSK 11.3% | s0 38.8%, s3 33.6%, s1 14.4%, s2 13.3% |
| Compute-Greedy + PPO-R | 8QAM 52.6%, 16QAM 33.4%, QPSK 13.4% | s0 50.1%, s3 43.2%, s1 6.6%, s2 0.1% |
| Compute-Greedy + DeepRMSA-A2C | 16QAM 99.9% | s0 51.2%, s3 45.2%, s1 2.6%, s2 0.9% |
| Compute-Greedy + KSP-BF | BPSK 76.9%, QPSK 16.2% | s0 51.0%, s3 37.7%, s1 8.7%, s2 2.6% |

Failure reasons were almost entirely `no_suitable_block`; server overload was negligible in this scenario.

## Agent-R-Only Comparison

This isolates the lower-layer RMSA behavior under a fixed split/server evaluation protocol.

```bash
/mnt/d/project/DNN_Agent/.venv/bin/python -m sa_hmarl.evaluation.eval_agent_r \
  --episodes 20 \
  --requests_per_episode 60 \
  --arrival_interval 0.25 \
  --holding_min 4 \
  --holding_max 10 \
  --deadline_min 30 \
  --deadline_max 100 \
  --size_min_mb 5 \
  --size_max_mb 30 \
  --edge_cost_min 0.5 \
  --edge_cost_max 15 \
  --slot_bw_hz 1.25e9 \
  --guard_band_fs 1 \
  --topology metro24_c_sensitive \
  --num_slots 24 \
  --num_servers 4 \
  --checkpoint sa_hmarl/checkpoints/separate_agent_r.pt \
  --deep_rmsa_checkpoint sa_hmarl/checkpoints/deep_rmsa_c_sensitive_mixed.pt
```

| Method | Blocking | Success | AvgReward | Delay(ms) | Waste | AvgFS | Path(km) |
|---|---:|---:|---:|---:|---:|---:|---:|
| DeepRMSA-A3C | 0.606 | 0.394 | -0.382 | 13.47 | 0.581 | 2.07 | 172.26 |
| Agent-R-DQN | 0.612 | 0.388 | -0.429 | 13.79 | 0.680 | 2.17 | 182.07 |
| KSP-FF | 0.619 | 0.381 | -0.401 | 13.37 | 0.550 | 2.39 | 174.19 |
| KSP-BF | 0.621 | 0.379 | -0.396 | 13.40 | 0.515 | 2.39 | 174.88 |

The R-only setting is intentionally harsh because the fixed server/split setup creates heavy server overload. DeepRMSA is slightly better than Agent-R-DQN and KSP methods, but the gap is much smaller than in the full system-level comparison.

## Interpretation

DeepRMSA is a strong additional RMSA baseline, but it should be described carefully:

- It is a PyTorch DeepRMSA-style single-worker A2C implementation, not the original asynchronous multi-worker TensorFlow A3C.
- Its action space is smaller than Agent-R PPO/DQN: it chooses path/block, while modulation is selected by a highest-SE feasible rule.
- On `metro24_c_sensitive`, links are short enough that high-order modulation is usually feasible. This makes DeepRMSA's highest-SE rule very strong, leading to 99.9% 16QAM usage and the lowest blocking in this specific system-level comparison.
- Stage2 Frozen-R+PPO-C remains the best learned C policy by average reward and demonstrates better server diversification than Compute-Greedy variants.

For the paper, this baseline is valuable because it shows that SA-HMARL is being compared against a strong RMSA-oriented RL method, not only against KSP heuristics.
