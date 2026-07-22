# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_german17`
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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 1.09% | 1.62% | 0.07% | 1.01% | 0.00% | 0.00% | 13.157/23.037 ms | 11.549/15.020 ms |
| ppo_c+v12 | ppo_c | v12 | 0.97% | 0.97% | 0.27% | 0.70% | 0.00% | 0.00% | 13.317/23.363 ms | 13.170/20.621 ms |
| ppo_c+ppo_r_proposer_postdec | ppo_c | ppo_r_proposer_postdec | 0.47% | 0.53% | 0.00% | 0.46% | 0.01% | 0.00% | 13.796/24.054 ms | 20.795/28.062 ms |
| ppo_c+mixed32_postdec | ppo_c | mixed32_postdec | 0.54% | 0.59% | 0.01% | 0.50% | 0.03% | 0.00% | 13.555/23.762 ms | 18.629/25.056 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | +0.11 pp | -0.20 pp | +0.31 pp | -0.160 ms |
| ppo_c+ppo_r_proposer_postdec | -0.50 pp | -0.27 pp | -0.24 pp | +0.480 ms |
| ppo_c+mixed32_postdec | -0.44 pp | -0.26 pp | -0.20 pp | +0.238 ms |