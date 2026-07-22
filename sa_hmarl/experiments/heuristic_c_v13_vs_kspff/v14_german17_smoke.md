# Main S100 System-Level Comparison

This is the recommended main comparison table for paper/PPT use.

## Configuration

- `topology`: `xlron_german17`
- `num_slots`: `320`
- `split_profile`: `default3`
- `arrival_interval`: `0.0625`
- `holding_min`: `20.0`
- `holding_max`: `30.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`
- `seeds`: `3030`
- `episodes`: `1`
- `requests_per_episode`: `1000`
- `k_paths`: `5`
- `path_sort_strategy`: `hops`
- `ksp_ff_k50_hops_k_paths`: `50`

## Results

| Method | C policy | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 9.12% | 11.38% | 0.00% | 9.12% | 0.00% | 0.00% | 9.780/18.418 ms | 9.790/14.410 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 11.25% | 13.88% | 0.00% | 11.25% | 0.00% | 0.00% | 10.890/19.436 ms | 12.723/18.879 ms |
| fr_count_c+ksp_ff_k50_hops | fr_count_c | ksp_ff_k50_hops | 69.75% | 75.50% | 0.00% | 69.75% | 0.00% | 0.00% | 18.801/29.592 ms | 6.950/11.865 ms |
| fr_count_c+v12_k50_hops | fr_count_c | v12_k50_hops | 70.88% | 77.75% | 0.00% | 70.88% | 0.00% | 0.00% | 20.187/29.866 ms | 8.419/14.590 ms |
| fr_quality_c+ksp_ff_k50_hops | fr_quality_c | ksp_ff_k50_hops | 46.38% | 51.62% | 0.00% | 46.38% | 0.00% | 0.00% | 13.962/24.464 ms | 8.658/15.245 ms |
| fr_quality_c+v12_k50_hops | fr_quality_c | v12_k50_hops | 49.00% | 56.75% | 0.00% | 49.00% | 0.00% | 0.00% | 15.619/28.395 ms | 10.494/18.149 ms |

## Reproducibility Metadata

- Git commit: `6f6337f44c036518886ce8e1146b07b6b6f6d1af`
- Git dirty: `True`
- Python: `3.14.4`
- Torch: `2.11.0+cu130`
- Numpy: `2.4.4`
- Evaluator SHA-256: `784f9e4237eaf2a42b68ad4ffef473c362f5be74b2b21378255d90770b387eb3`
- Agent-C checkpoint SHA-256: `a5c9eb39bc33cf3b77a06d80e9c45315880dcb963ad9c3c95663eadaded9f2c8`
- Agent-R checkpoint SHA-256: `e32e908196ac0c39898aab6e98516842c53f817150eeaba97e4ac172a61f8b94`
- Ranking checkpoint SHA-256: `9e9f06c766bdd8ea6c37400a5fed6cc3ed3c7593804183a58be13a65dfaaeb21`