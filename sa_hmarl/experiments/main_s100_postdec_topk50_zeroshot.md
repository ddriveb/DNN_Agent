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
| ppo_c+v12 | ppo_c | v12 | 0.92% | 0.92% | 0.84% | 0.09% | 0.00% | 0.00% | 8.377/16.304 ms | 11.832/20.231 ms |
| ppo_c+ppo_r_proposer_ksp_ff | ppo_c | ppo_r_proposer_ksp_ff | 1.38% | 1.38% | 1.29% | 0.09% | 0.00% | 0.00% | 8.102/15.301 ms | 10.970/15.907 ms |
| ppo_c+ppo_r_proposer_postdec | ppo_c | ppo_r_proposer_postdec | 1.01% | 1.01% | 0.88% | 0.14% | 0.00% | 0.00% | 8.553/16.606 ms | 17.006/27.133 ms |

## Delta vs PPO-C + v1.2

| Method | Δ Blocking | Δ NSB | Δ Overload | Δ Delay mean |
|---|---:|---:|---:|---:|
| ppo_c+ppo_r_proposer_ksp_ff | +0.45 pp | +0.45 pp | -0.00 pp | -0.275 ms |
| ppo_c+ppo_r_proposer_postdec | +0.09 pp | +0.04 pp | +0.05 pp | +0.176 ms |