# K=5 System Comparison — 7-Method Benchmark

**Topology:** snap24_gnutella_reach  **Slots:** 20  **Servers:** 4  **k_paths:** 5  **MaxBlocks:** 10  **BlockSort:** mixed

**Agent-C:** `sa_hmarl/checkpoints/agent_c_r_feasibility_s42_s20_r80_best.pt`  
**Agent-R:** `sa_hmarl/checkpoints/agent_r_mixed.pt`  

**Seeds:** 42,123,456,789,101112  **Eps/seed:** 20  **Requests/ep:** 80

**Total time:** 605s (10.1 min)

## Results

| Method | Blocking | Reward | Delay(ms) | AvgFS | Waste | PathKm | noC% | avgRacts | NSB% | SvrOv% |
|--------|----------|--------|-----------|-------|-------|--------|------|----------|------|--------|
| Agent-C + PPO-R | 0.4614±0.2824 | -0.138 | 8.3±0.8 | 3.09 | 0.333 | 658.1 | 52.0% | 4 | 100.0% | 0.0% |
| WO-C + PPO-R | 0.4807±0.2757 | -0.149 | 8.4±1.2 | 3.22 | 0.321 | 688.5 | 53.9% | 3 | 100.0% | 0.0% |
| DF-C + PPO-R | 0.4770±0.2677 | -0.158 | 7.1±1.4 | 3.14 | 0.330 | 546.4 | 54.5% | 3 | 99.5% | 0.5% |
| RF-C + PPO-R | 0.4829±0.2741 | -0.158 | 8.1±1.0 | 3.23 | 0.333 | 673.4 | 54.3% | 3 | 100.0% | 0.0% |
| IWD-C + PPO-R | 0.4351±0.2787 | -0.128 | 8.6±2.3 | 2.75 | 0.349 | 562.3 | 49.9% | 3 | 92.7% | 7.3% |
| Greedy-C + PPO-R | 0.4795±0.2682 | -0.163 | 6.9±1.0 | 3.16 | 0.344 | 561.5 | 54.4% | 3 | 100.0% | 0.0% |
| Agent-C + KSP-BF | 0.5008±0.2767 | -0.182 | 8.3±0.8 | 3.54 | 0.307 | 679.9 | 55.4% | 3 | 100.0% | 0.0% |

### KSP-BF Stats

- Calls: 8000
- Success: 3566 (44.6%)
- Total BF time: 46.1s
- Δblocking (Agent-C → KSP-BF): -0.0394

## Failure Reasons

- **Agent-C + PPO-R**: no_suitable_block=3690 (100.0%), server_overload=1 (0.0%)
- **WO-C + PPO-R**: no_suitable_block=3846 (100.0%)
- **DF-C + PPO-R**: no_suitable_block=3798 (99.5%), server_overload=18 (0.5%)
- **RF-C + PPO-R**: no_suitable_block=3863 (100.0%)
- **IWD-C + PPO-R**: no_suitable_block=3226 (92.7%), server_overload=255 (7.3%)
- **Greedy-C + PPO-R**: no_suitable_block=3836 (100.0%)
- **Agent-C + KSP-BF**: no_suitable_block=4005 (100.0%), server_overload=1 (0.0%)

## Split Distribution

- **Agent-C + PPO-R**: split0=56.5%, split4=21.9%, split3=7.6%, split2=7.1%, split1=7.0%
- **WO-C + PPO-R**: split0=54.6%, split4=37.9%, split3=4.1%, split2=1.8%, split1=1.6%
- **DF-C + PPO-R**: split0=54.9%, split4=39.2%, split3=4.0%, split2=1.2%, split1=0.7%
- **RF-C + PPO-R**: split0=54.8%, split4=38.2%, split3=4.0%, split2=1.7%, split1=1.3%
- **IWD-C + PPO-R**: split0=56.5%, split4=13.2%, split3=12.7%, split2=9.8%, split1=7.7%
- **Greedy-C + PPO-R**: split0=54.5%, split4=40.6%, split3=3.6%, split2=0.9%, split1=0.4%
- **Agent-C + KSP-BF**: split0=58.3%, split4=23.3%, split3=6.8%, split2=5.9%, split1=5.7%

## Server Distribution

- **Agent-C + PPO-R**: s0=69.8%, s3=23.5%, s1=3.9%, s2=2.9%
- **WO-C + PPO-R**: s0=63.4%, s3=15.6%, s1=11.3%, s2=9.7%
- **DF-C + PPO-R**: s0=79.5%, s3=16.6%, s1=2.1%, s2=1.8%
- **RF-C + PPO-R**: s0=66.5%, s3=16.1%, s1=10.0%, s2=7.4%
- **IWD-C + PPO-R**: s0=75.2%, s3=17.3%, s1=4.2%, s2=3.2%
- **Greedy-C + PPO-R**: s0=74.3%, s3=21.8%, s1=3.1%, s2=0.9%
- **Agent-C + KSP-BF**: s0=73.0%, s3=20.5%, s1=3.7%, s2=2.9%

## Modulation Distribution

- **Agent-C + PPO-R**: BPSK=73.3%, QPSK=13.9%, 8QAM=6.9%, 16QAM=6.0%
- **WO-C + PPO-R**: BPSK=49.9%, QPSK=24.6%, 8QAM=14.2%, 16QAM=11.4%
- **DF-C + PPO-R**: BPSK=51.5%, QPSK=21.7%, 8QAM=15.8%, 16QAM=11.0%
- **RF-C + PPO-R**: BPSK=51.2%, QPSK=23.5%, 8QAM=13.6%, 16QAM=11.7%
- **IWD-C + PPO-R**: BPSK=58.6%, QPSK=22.6%, 8QAM=12.1%, 16QAM=6.7%
- **Greedy-C + PPO-R**: BPSK=49.6%, QPSK=20.0%, 8QAM=15.6%, 16QAM=14.8%
- **Agent-C + KSP-BF**: BPSK=84.9%, QPSK=11.9%, 8QAM=2.9%, 16QAM=0.4%

## Runtime

| Method | Time (s) |
|---|---|
| Agent-C + PPO-R | 0.0 |
| WO-C + PPO-R | 0.0 |
| DF-C + PPO-R | 0.0 |
| RF-C + PPO-R | 0.0 |
| IWD-C + PPO-R | 0.0 |
| Greedy-C + PPO-R | 0.0 |
| Agent-C + KSP-BF | 46.1 |
