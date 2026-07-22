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
- Main heuristic: `FF-KSP K=50 hops`
- Auxiliary heuristic: `KSP-FF K=50 hops`
- Seeds: 6101,6102,6103,6104,6105
- Warmup: 500, Evaluated: 6000

## Main Result

| Method | Blocking | Overload | NSB | R-no-valid | Deadline | Avg delay ms | Avg FS | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.4433% | 0.0000% | 0.0000% | 5.4433% | 0.0000% | 16.31 | 2.41 | 1386.04 | 22.65 |
| FF-KSP K=50 hops | 5.4400% | 0.0000% | 0.0000% | 5.4400% | 0.0000% | 15.11 | 2.17 | 1265.59 | 6.86 |
| KSP-FF K=50 hops | 5.4400% | 0.0000% | 0.0000% | 5.4400% | 0.0000% | 13.44 | 2.10 | 1060.65 | 6.80 |

**Blocking differences**: Strict v1.3 - FF-KSP K=50 hops = **0.00 pp**, Strict v1.3 - KSP-FF K=50 hops = **0.00 pp**

## Per-seed summaries

| Seed | Strict blocking | Main blocking | Aux blocking | Strict-main pp | Strict-aux pp |
|---|---:|---:|---:|---:|---:|
| 6101 | 6.0000% | 5.9833% | 5.9833% | +0.02 | +0.02 |
| 6102 | 5.8167% | 5.8167% | 5.8167% | +0.00 | +0.00 |
| 6103 | 5.2167% | 5.2167% | 5.2167% | +0.00 | +0.00 |
| 6104 | 5.1167% | 5.1167% | 5.1167% | +0.00 | +0.00 |
| 6105 | 5.0667% | 5.0667% | 5.0667% | +0.00 | +0.00 |
