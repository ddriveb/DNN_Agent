# K=5 System Comparison — 7-Method Benchmark

**Topology:** snap24_gnutella_reach  **Slots:** 20  **Servers:** 4  **k_paths:** 5  **MaxBlocks:** 10  **BlockSort:** mixed

**Agent-C:** `sa_hmarl/checkpoints/agent_c_typed_mean_field_s123_s20_r80_best.pt`  
**Agent-R:** `sa_hmarl/checkpoints/agent_r_mixed.pt`  

**Seeds:** 42,123,456,789,101112  **Eps/seed:** 20  **Requests/ep:** 80

**Total time:** 592s (9.9 min)

## Results

| Method | Blocking | Reward | Delay(ms) | AvgFS | Waste | PathKm | noC% | avgRacts | NSB% | SvrOv% |
|--------|----------|--------|-----------|-------|-------|--------|------|----------|------|--------|
| Agent-C + PPO-R | 0.4352±0.2650 | -0.101 | 8.6±1.3 | 2.71 | 0.335 | 648.0 | 50.6% | 2 | 99.7% | 0.3% |
| WO-C + PPO-R | 0.4807±0.2757 | -0.149 | 8.4±1.2 | 3.22 | 0.321 | 688.5 | 53.9% | 3 | 100.0% | 0.0% |
| DF-C + PPO-R | 0.4770±0.2677 | -0.158 | 7.1±1.4 | 3.14 | 0.330 | 546.4 | 54.5% | 3 | 99.5% | 0.5% |
| RF-C + PPO-R | 0.4829±0.2741 | -0.158 | 8.1±1.0 | 3.23 | 0.333 | 673.4 | 54.3% | 3 | 100.0% | 0.0% |
| IWD-C + PPO-R | 0.4351±0.2787 | -0.128 | 8.6±2.3 | 2.75 | 0.349 | 562.3 | 49.9% | 3 | 92.7% | 7.3% |
| Greedy-C + PPO-R | 0.4795±0.2682 | -0.163 | 6.9±1.0 | 3.16 | 0.344 | 561.5 | 54.4% | 3 | 100.0% | 0.0% |
| Agent-C + KSP-BF | 0.4477±0.2625 | -0.111 | 8.5±1.3 | 2.91 | 0.324 | 650.1 | 51.9% | 2 | 99.8% | 0.2% |

### KSP-BF Stats

- Calls: 8000
- Success: 3851 (48.1%)
- Total BF time: 36.8s
- Δblocking (Agent-C → KSP-BF): -0.0125

## Failure Reasons

- **Agent-C + PPO-R**: no_suitable_block=3470 (99.7%), server_overload=12 (0.3%)
- **WO-C + PPO-R**: no_suitable_block=3846 (100.0%)
- **DF-C + PPO-R**: no_suitable_block=3798 (99.5%), server_overload=18 (0.5%)
- **RF-C + PPO-R**: no_suitable_block=3863 (100.0%)
- **IWD-C + PPO-R**: no_suitable_block=3226 (92.7%), server_overload=255 (7.3%)
- **Greedy-C + PPO-R**: no_suitable_block=3836 (100.0%)
- **Agent-C + KSP-BF**: no_suitable_block=3576 (99.8%), server_overload=6 (0.2%)

## Split Distribution

- **Agent-C + PPO-R**: split0=55.8%, split4=28.4%, split3=6.2%, split2=4.9%, split1=4.6%
- **WO-C + PPO-R**: split0=54.6%, split4=37.9%, split3=4.1%, split2=1.8%, split1=1.6%
- **DF-C + PPO-R**: split0=54.9%, split4=39.2%, split3=4.0%, split2=1.2%, split1=0.7%
- **RF-C + PPO-R**: split0=54.8%, split4=38.2%, split3=4.0%, split2=1.7%, split1=1.3%
- **IWD-C + PPO-R**: split0=56.5%, split4=13.2%, split3=12.7%, split2=9.8%, split1=7.7%
- **Greedy-C + PPO-R**: split0=54.5%, split4=40.6%, split3=3.6%, split2=0.9%, split1=0.4%
- **Agent-C + KSP-BF**: split0=57.0%, split4=28.4%, split3=5.8%, split1=4.4%, split2=4.4%

## Server Distribution

- **Agent-C + PPO-R**: s0=67.6%, s3=12.4%, s1=12.1%, s2=7.9%
- **WO-C + PPO-R**: s0=63.4%, s3=15.6%, s1=11.3%, s2=9.7%
- **DF-C + PPO-R**: s0=79.5%, s3=16.6%, s1=2.1%, s2=1.8%
- **RF-C + PPO-R**: s0=66.5%, s3=16.1%, s1=10.0%, s2=7.4%
- **IWD-C + PPO-R**: s0=75.2%, s3=17.3%, s1=4.2%, s2=3.2%
- **Greedy-C + PPO-R**: s0=74.3%, s3=21.8%, s1=3.1%, s2=0.9%
- **Agent-C + KSP-BF**: s0=67.4%, s3=12.7%, s1=11.7%, s2=8.2%

## Modulation Distribution

- **Agent-C + PPO-R**: BPSK=43.2%, QPSK=23.3%, 8QAM=18.2%, 16QAM=15.3%
- **WO-C + PPO-R**: BPSK=49.9%, QPSK=24.6%, 8QAM=14.2%, 16QAM=11.4%
- **DF-C + PPO-R**: BPSK=51.5%, QPSK=21.7%, 8QAM=15.8%, 16QAM=11.0%
- **RF-C + PPO-R**: BPSK=51.2%, QPSK=23.5%, 8QAM=13.6%, 16QAM=11.7%
- **IWD-C + PPO-R**: BPSK=58.6%, QPSK=22.6%, 8QAM=12.1%, 16QAM=6.7%
- **Greedy-C + PPO-R**: BPSK=49.6%, QPSK=20.0%, 8QAM=15.6%, 16QAM=14.8%
- **Agent-C + KSP-BF**: BPSK=62.9%, QPSK=21.1%, 8QAM=14.2%, 16QAM=1.8%

## Runtime

| Method | Time (s) |
|---|---|
| Agent-C + PPO-R | 0.0 |
| WO-C + PPO-R | 0.0 |
| DF-C + PPO-R | 0.0 |
| RF-C + PPO-R | 0.0 |
| IWD-C + PPO-R | 0.0 |
| Greedy-C + PPO-R | 0.0 |
| Agent-C + KSP-BF | 36.8 |
