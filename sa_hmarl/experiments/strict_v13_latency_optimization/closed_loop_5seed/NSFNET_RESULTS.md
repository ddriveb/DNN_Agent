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
- Main heuristic: `KSP-FF K=50 hops`
- Auxiliary heuristic: `FF-KSP K=50 hops`
- Seeds: 6101,6102,6103,6104,6105
- Warmup: 500, Evaluated: 6000

## Main Result

| Method | Blocking | Overload | NSB | R-no-valid | Deadline | Avg delay ms | Avg FS | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.3733% | 0.0000% | 0.0000% | 5.3733% | 0.0000% | 20.44 | 2.73 | 2554.00 | 34.13 |
| KSP-FF K=50 hops | 5.3600% | 0.0000% | 0.0000% | 5.3600% | 0.0000% | 18.64 | 2.68 | 2324.19 | 8.58 |
| FF-KSP K=50 hops | 5.3667% | 0.0000% | 0.0000% | 5.3667% | 0.0000% | 19.37 | 2.72 | 2439.41 | 6.85 |

**Blocking differences**: Strict v1.3 - KSP-FF K=50 hops = **0.01 pp**, Strict v1.3 - FF-KSP K=50 hops = **0.01 pp**

## Per-seed summaries

| Seed | Strict blocking | Main blocking | Aux blocking | Strict-main pp | Strict-aux pp |
|---|---:|---:|---:|---:|---:|
| 6101 | 5.8667% | 5.8667% | 5.8667% | +0.00 | +0.00 |
| 6102 | 5.2667% | 5.2167% | 5.2500% | +0.05 | +0.02 |
| 6103 | 4.9833% | 4.9833% | 4.9833% | +0.00 | +0.00 |
| 6104 | 5.2333% | 5.2500% | 5.2500% | -0.02 | -0.02 |
| 6105 | 5.5167% | 5.4833% | 5.4833% | +0.03 | +0.03 |
