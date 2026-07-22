# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_jpn48`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.15`
- `holding_min`: `4.0`
- `holding_max`: `10.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 5.04% | 6.56% | 1.94% | 2.81% | 0.27% | 0.01% | 15.695/28.622 ms | 24.134/38.008 ms |
| ppo_c+v12 | ppo_c | v12 | 6.19% | 6.19% | 4.59% | 1.05% | 0.51% | 0.04% | 16.066/28.934 ms | 26.799/42.532 ms |
| ppo_c+ppo_r_proposer_postdec | ppo_c | ppo_r_proposer_postdec | 4.07% | 4.16% | 3.01% | 0.72% | 0.30% | 0.04% | 16.567/29.702 ms | 34.843/49.568 ms |
| ppo_c+mixed32_postdec | ppo_c | mixed32_postdec | 3.75% | 4.25% | 1.97% | 1.26% | 0.46% | 0.05% | 16.314/29.667 ms | 32.098/45.737 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | -1.15 pp | -2.65 pp | +1.76 pp | -0.370 ms |
| ppo_c+ppo_r_proposer_postdec | -2.11 pp | -1.57 pp | -0.33 pp | +0.502 ms |
| ppo_c+mixed32_postdec | -2.44 pp | -2.61 pp | +0.21 pp | +0.249 ms |