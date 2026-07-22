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
- Seeds: None
- Warmup: 500, Evaluated: 6000

## Main Result

| Method | Blocking | R-no-valid | Rescue attempted | Rescue success | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.6067% | 5.6067% | 0 | 0 | 1270.49 | 43.34 |
| FF-KSP K=50 hops | 5.5933% | 5.5933% | 0 | 0 | 1175.85 | 24.05 |
| Strict v1.3 + K500 rescue | 5.6067% | 0.0000% | 1682 | 0 | 1270.49 | 38.33 |
| FF-KSP K=50 hops + K500 rescue | 5.5933% | 0.0000% | 1678 | 0 | 1175.85 | 19.98 |

## Per-seed summaries

| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| 6101 | 5.7500% | 5.7333% | 5.7500% | 5.7333% |
| 6102 | 5.1500% | 5.1500% | 5.1500% | 5.1500% |
| 6103 | 5.7667% | 5.7500% | 5.7667% | 5.7500% |
| 6104 | 5.3500% | 5.3333% | 5.3500% | 5.3333% |
| 6105 | 6.0167% | 6.0000% | 6.0167% | 6.0000% |
