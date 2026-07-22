# JPN48 df_c Heuristic C + R Backend Comparison

- Topology: xlron_jpn48
- Seeds: 3030,4040,5050,6060,7070
- Methods: df_c+ksp_ff_k50_hops,df_c+v12_k50_hops

## Aggregate Results

| Method | Blocking % | Overload % | NSB % | Delay ms |
|---|---:|---:|---:|---:|
| df_c+ksp_ff_k50_hops | 11.28 | 10.80 | 0.15 | 16.35 |
| df_c+v12_k50_hops | 10.85 | 10.10 | 0.22 | 17.88 |

## Per-Seed Blocking

| Seed | df + KSP-FF | df + v1.3 |
|---:|---:|---:|
| 3030 | 14.01 | 13.71 |
| 4040 | 9.68 | 9.15 |
| 5050 | 12.03 | 11.68 |
| 6060 | 10.49 | 9.90 |
| 7070 | 10.21 | 9.79 |
