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
- Main heuristic: `FF-KSP K=50 hops`
- Auxiliary heuristic: `KSP-FF K=50 hops`
- Seeds: 6101,6102,6103,6104,6105
- Warmup: 500, Evaluated: 6000

## Main Result

| Method | Blocking | Overload | NSB | R-no-valid | Deadline | Avg delay ms | Avg FS | Avg path km | Avg decision ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Strict v1.3 frozen cross-topology transfer | 5.6133% | 0.0000% | 0.0000% | 5.6133% | 0.0000% | 19.19 | 2.69 | 1278.14 | 28.34 |
| FF-KSP K=50 hops | 5.5933% | 0.0000% | 0.0000% | 5.5933% | 0.0000% | 17.77 | 2.32 | 1175.85 | 9.23 |
| KSP-FF K=50 hops | 5.5933% | 0.0000% | 0.0000% | 5.5933% | 0.0000% | 16.32 | 2.26 | 1074.91 | 8.43 |

**Blocking differences**: Strict v1.3 - FF-KSP K=50 hops = **0.02 pp**, Strict v1.3 - KSP-FF K=50 hops = **0.02 pp**

## Per-seed summaries

| Seed | Strict blocking | Main blocking | Aux blocking | Strict-main pp | Strict-aux pp |
|---|---:|---:|---:|---:|---:|
| 6101 | 5.7500% | 5.7333% | 5.7333% | +0.02 | +0.02 |
| 6102 | 5.1333% | 5.1500% | 5.1500% | -0.02 | -0.02 |
| 6103 | 5.7833% | 5.7500% | 5.7500% | +0.03 | +0.03 |
| 6104 | 5.3667% | 5.3333% | 5.3333% | +0.03 | +0.03 |
| 6105 | 6.0333% | 6.0000% | 6.0000% | +0.03 | +0.03 |
