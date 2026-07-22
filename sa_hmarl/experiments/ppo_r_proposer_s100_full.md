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
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `20`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12 | ppo_c | v12 | 0.92% | 0.92% | 0.84% | 0.09% | 0.00% | 0.00% | 8.377/16.304 ms | 17.037/29.160 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 1.05% | 1.05% | 0.94% | 0.11% | 0.00% | 0.00% | 7.919/15.229 ms | 13.268/18.637 ms |
| ppo_c+ppo_r | ppo_c | ppo_r | 3.66% | 3.66% | 3.25% | 0.41% | 0.00% | 0.00% | 8.088/14.672 ms | 9.233/12.556 ms |
| ppo_c+ppo_r_k50_hops | ppo_c | ppo_r_k50_hops | 3.78% | 3.78% | 3.36% | 0.41% | 0.00% | 0.00% | 8.761/16.367 ms | 11.306/16.387 ms |
| ppo_c+ppo_r_proposer_ksp_ff | ppo_c | ppo_r_proposer_ksp_ff | 1.38% | 1.38% | 1.29% | 0.09% | 0.00% | 0.00% | 8.102/15.301 ms | 10.994/16.005 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | +0.12 pp | +0.10 pp | +0.02 pp | -0.457 ms |
| ppo_c+ppo_r | +2.74 pp | +2.41 pp | +0.33 pp | -0.289 ms |
| ppo_c+ppo_r_k50_hops | +2.85 pp | +2.53 pp | +0.33 pp | +0.384 ms |
| ppo_c+ppo_r_proposer_ksp_ff | +0.45 pp | +0.45 pp | -0.00 pp | -0.275 ms |