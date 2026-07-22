# Results: xlron_nsfnet_deeprmsa

## Protocol Lock

- Topology: `xlron_nsfnet_deeprmsa`
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
| Strict v1.3 frozen cross-topology transfer | 5.3733% | 5.3733% | 0 | 0 | 2554.00 | 16.65 |
| KSP-FF K=50 hops | 5.3600% | 5.3600% | 0 | 0 | 2324.19 | 6.91 |
| Strict v1.3 + K500 rescue | 5.3733% | 0.0000% | 1612 | 0 | 2554.00 | 16.05 |
| KSP-FF K=50 hops + K500 rescue | 5.3600% | 0.0000% | 1608 | 0 | 2324.19 | 6.91 |

## Per-seed summaries

| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| 6101 | 5.8667% | 5.8667% | 5.8667% | 5.8667% |
| 6102 | 5.2667% | 5.2167% | 5.2667% | 5.2167% |
| 6103 | 4.9833% | 4.9833% | 4.9833% | 4.9833% |
| 6104 | 5.2333% | 5.2500% | 5.2333% | 5.2500% |
| 6105 | 5.5167% | 5.4833% | 5.5167% | 5.4833% |
