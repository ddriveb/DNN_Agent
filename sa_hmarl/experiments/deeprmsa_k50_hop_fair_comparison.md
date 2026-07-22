# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.0625`
- `holding_min`: `20.0`
- `holding_max`: `30.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `1`
- `requests_per_episode`: `10000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 9.30% | 12.13% | 0.00% | 9.29% | 0.00% | 0.00% | 10.339/18.213 ms | 10.702/15.374 ms |
| ppo_c+deep_rmsa_style_k50_hops | ppo_c | deep_rmsa_style_k50_hops | 9.62% | 12.52% | 0.00% | 9.62% | 0.00% | 0.00% | 10.519/19.161 ms | 9.144/13.272 ms |
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 9.93% | 12.97% | 0.00% | 9.93% | 0.00% | 0.00% | 9.086/17.171 ms | 9.074/13.292 ms |
| ppo_c+ksp_ff | ppo_c | ksp_ff | 7.46% | 9.69% | 0.00% | 7.46% | 0.00% | 0.00% | 8.989/16.470 ms | 6.280/8.910 ms |

## Reproducibility Metadata

- Git commit: `6f6337f44c036518886ce8e1146b07b6b6f6d1af`
- Git dirty: `True`
- Python: `3.14.4`
- Torch: `2.11.0+cu130`
- Numpy: `2.4.4`
- Evaluator SHA-256: `df3b496e4ed0393dad002c70ddecb89ddf7f10295756c47560963f2c1b1b3de0`
- Agent-C checkpoint SHA-256: `a5c9eb39bc33cf3b77a06d80e9c45315880dcb963ad9c3c95663eadaded9f2c8`
- Agent-R checkpoint SHA-256: `e32e908196ac0c39898aab6e98516842c53f817150eeaba97e4ac172a61f8b94`
- Ranking checkpoint SHA-256: `9e9f06c766bdd8ea6c37400a5fed6cc3ed3c7593804183a58be13a65dfaaeb21`