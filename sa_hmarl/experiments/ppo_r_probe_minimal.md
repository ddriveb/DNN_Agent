# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `snap24_gnutella_reach`
- `num_slots`: `100`
- `split_profile`: `default3`
- `arrival_interval`: `0.15`
- `holding_min`: `4.0`
- `holding_max`: `10.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040`
- `episodes`: `5`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 0.38% | 0.38% | 0.38% | 0.00% | 0.00% | 0.00% | 8.581/16.483 ms | 12.315/20.300 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 0.38% | 0.38% | 0.38% | 0.00% | 0.00% | 0.00% | 7.928/14.882 ms | 10.440/14.238 ms |
| ppo_c+ppo_r | ppo_c | ppo_r | 3.38% | 3.38% | 3.12% | 0.25% | 0.00% | 0.00% | 8.357/15.405 ms | 9.266/12.293 ms |
| ppo_c+ppo_r_k50_hops | ppo_c | ppo_r_k50_hops | 3.25% | 3.25% | 3.00% | 0.25% | 0.00% | 0.00% | 9.183/16.034 ms | 11.846/17.144 ms |
| ppo_c+ppo_r_proposer_ksp_ff | ppo_c | ppo_r_proposer_ksp_ff | 0.50% | 0.50% | 0.50% | 0.00% | 0.00% | 0.00% | 8.226/14.869 ms | 12.009/17.137 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | +0.00 pp | +0.00 pp | +0.00 pp | -0.653 ms |
| ppo_c+ppo_r | +3.00 pp | +2.75 pp | +0.25 pp | -0.225 ms |
| ppo_c+ppo_r_k50_hops | +2.88 pp | +2.63 pp | +0.25 pp | +0.602 ms |
| ppo_c+ppo_r_proposer_ksp_ff | +0.13 pp | +0.13 pp | +0.00 pp | -0.355 ms |