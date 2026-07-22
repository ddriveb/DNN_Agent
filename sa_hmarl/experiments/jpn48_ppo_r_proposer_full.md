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
| ppo_c+v12 | ppo_c | v12 | 6.19% | 6.19% | 4.59% | 1.05% | 0.51% | 0.04% | 16.066/28.934 ms | 25.511/39.682 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 5.04% | 6.56% | 1.94% | 2.81% | 0.27% | 0.01% | 15.695/28.622 ms | 24.145/37.897 ms |
| ppo_c+ppo_r | ppo_c | ppo_r | 12.34% | 12.34% | 11.65% | 0.26% | 0.41% | 0.01% | 15.550/28.133 ms | 22.072/36.506 ms |
| ppo_c+ppo_r_k50_hops | ppo_c | ppo_r_k50_hops | 7.59% | 9.28% | 4.31% | 2.90% | 0.35% | 0.03% | 16.973/29.590 ms | 25.809/39.749 ms |
| ppo_c+ppo_r_proposer_ksp_ff | ppo_c | ppo_r_proposer_ksp_ff | 7.25% | 9.38% | 4.04% | 2.80% | 0.36% | 0.05% | 16.290/29.075 ms | 25.875/40.303 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | -1.15 pp | -2.65 pp | +1.76 pp | -0.370 ms |
| ppo_c+ppo_r | +6.15 pp | +7.06 pp | -0.79 pp | -0.516 ms |
| ppo_c+ppo_r_k50_hops | +1.40 pp | -0.27 pp | +1.85 pp | +0.907 ms |
| ppo_c+ppo_r_proposer_ksp_ff | +1.06 pp | -0.55 pp | +1.75 pp | +0.225 ms |