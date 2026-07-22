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
- Seeds: 5001
- Warmup: 50, Evaluated: 50

## Main Result

| Method | Blocking | R-no-valid | Rescue attempted | Rescue success | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 4.0000% | 4.0000% | 0 | 0 | 2734.38 | 11.52 |
| KSP-FF K=50 hops | 4.0000% | 4.0000% | 0 | 0 | 2262.50 | 4.49 |
| Strict v1.3 + K500 rescue | 4.0000% | 0.0000% | 2 | 0 | 2734.38 | 11.38 |
| KSP-FF K=50 hops + K500 rescue | 4.0000% | 0.0000% | 2 | 0 | 2262.50 | 4.63 |

## Per-seed summaries

| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| 5001 | 4.0000% | 4.0000% | 4.0000% | 4.0000% |
