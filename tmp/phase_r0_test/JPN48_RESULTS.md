# Results: xlron_jpn48

## Protocol Lock

- Topology: `xlron_jpn48`
- C-side: FIXED (no PPO-C loaded, no PPO-C action selected)
- Traffic matrix: uniform all-OD
- fixed_split_id: 0
- server_node_ids: [0, 1, 2, 3]
- R-side K_path: 50
- R-side path_sort_strategy: `hops`
- R-side block_sort_strategy: `start_asc`
- max_blocks: 10
- Preferred heuristic: `FF-KSP K=50 hops`, rescue enabled at K=100/200/500
- Seeds: 5001
- Warmup: 50, Evaluated: 50

## Main Result

| Method | Blocking | R-no-valid | Rescue attempted | Rescue success | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 2.0000% | 2.0000% | 0 | 0 | 1433.24 | 29.76 |
| FF-KSP K=50 hops | 2.0000% | 2.0000% | 0 | 0 | 1211.22 | 13.34 |
| Strict v1.3 + K500 rescue | 2.0000% | 0.0000% | 1 | 0 | 1433.24 | 27.47 |
| FF-KSP K=50 hops + K500 rescue | 2.0000% | 0.0000% | 1 | 0 | 1211.22 | 13.19 |

## Per-seed summaries

| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| 5001 | 2.0000% | 2.0000% | 2.0000% | 2.0000% |
