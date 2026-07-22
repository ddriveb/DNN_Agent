# Results: xlron_usnet_gcnrmsa

## Protocol Lock

- Topology: `xlron_usnet_gcnrmsa`
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
| Strict v1.3 frozen cross-topology transfer | 5.4400% | 5.4400% | 0 | 0 | 1400.85 | 35.20 |
| FF-KSP K=50 hops | 5.4400% | 5.4400% | 0 | 0 | 1265.59 | 18.71 |
| Strict v1.3 + K500 rescue | 5.4400% | 0.0000% | 1632 | 0 | 1400.85 | 35.68 |
| FF-KSP K=50 hops + K500 rescue | 5.4400% | 0.0000% | 1632 | 0 | 1265.59 | 18.85 |

## Per-seed summaries

| Seed | Strict blocking | Preferred blocking | Strict rescue blocking | Preferred rescue blocking |
|---|---:|---:|---:|---:|
| 6101 | 5.9833% | 5.9833% | 5.9833% | 5.9833% |
| 6102 | 5.8167% | 5.8167% | 5.8167% | 5.8167% |
| 6103 | 5.2167% | 5.2167% | 5.2167% | 5.2167% |
| 6104 | 5.1167% | 5.1167% | 5.1167% | 5.1167% |
| 6105 | 5.0667% | 5.0667% | 5.0667% | 5.0667% |
