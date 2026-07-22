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
- Seeds: 5001
- Warmup: 50, Evaluated: 50

## Main Result

| Method | Blocking | R-no-valid | Rescue attempted | Rescue success | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 0.0000% | 0.0000% | 0 | 0 | 1020.68 | 13.67 |
| KSP-FF K=50 hops | 0.0000% | 0.0000% | 0 | 0 | 791.68 | 3.46 |
| Strict v1.3 + K500 rescue | 0.0000% | 0.0000% | 0 | 0 | 1020.68 | 13.90 |
| KSP-FF K=50 hops + K500 rescue | 0.0000% | 0.0000% | 0 | 0 | 791.68 | 3.26 |

## Per-seed summaries

| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| 5001 | 0.0000% | 0.0000% | 0.0000% | 0.0000% |
