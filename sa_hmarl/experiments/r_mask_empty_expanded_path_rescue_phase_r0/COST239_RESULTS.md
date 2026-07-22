# Results: xlron_cost239_ptrnet_real

## Protocol Lock

- Topology: `xlron_cost239_ptrnet_real`
- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)
- Traffic matrix: uniform all-OD
- fixed_split_id: 0
- server_node_ids: [0, 1, 2, 3]
- R-side K_path: 50
- R-side path_sort_strategy: `hops`
- R-side block_sort_strategy: `start_asc`
- max_blocks: 10
- Preferred heuristic: `KSP-FF K=50 hops`, rescue enabled at K=100/200/500
- Seeds: None
- Warmup: 500, Evaluated: 6000

## Main Result

| Method | Blocking | R-no-valid | Rescue attempted | Rescue success | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.5933% | 5.5933% | 0 | 0 | 977.09 | 27.07 |
| KSP-FF K=50 hops | 5.5933% | 5.5933% | 0 | 0 | 831.02 | 13.16 |
| Strict v1.3 + K500 rescue | 5.5933% | 0.0000% | 1678 | 0 | 977.09 | 28.39 |
| KSP-FF K=50 hops + K500 rescue | 5.5933% | 0.0000% | 1678 | 0 | 831.02 | 13.04 |

## Per-seed summaries

| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| 5001 | 5.3833% | 5.3833% | 5.3833% | 5.3833% |
| 5002 | 5.6667% | 5.6667% | 5.6667% | 5.6667% |
| 5003 | 5.2667% | 5.2667% | 5.2667% | 5.2667% |
| 5004 | 5.8333% | 5.8333% | 5.8333% | 5.8333% |
| 5005 | 5.8167% | 5.8167% | 5.8167% | 5.8167% |
