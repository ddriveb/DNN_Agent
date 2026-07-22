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
| ppo_c+ksp_ff_k50_hops | ppo_c | ksp_ff_k50_hops | 10.37% | 13.00% | 0.00% | 10.37% | 0.00% | 0.00% | 8.800/16.777 ms | 8.287/12.699 ms |
| ppo_c+v12_k50_hops | ppo_c | v12_k50_hops | 7.68% | 9.71% | 0.00% | 7.68% | 0.00% | 0.00% | 9.539/17.268 ms | 12.372/18.550 ms |
| fr_count_c+ksp_ff_k50_hops | fr_count_c | ksp_ff_k50_hops | 70.17% | 77.99% | 0.00% | 70.17% | 0.00% | 0.00% | 17.372/26.922 ms | 5.592/9.486 ms |
| fr_count_c+v12_k50_hops | fr_count_c | v12_k50_hops | 69.69% | 77.57% | 0.00% | 69.68% | 0.00% | 0.00% | 18.861/28.711 ms | 7.485/13.576 ms |
| fr_quality_c+ksp_ff_k50_hops | fr_quality_c | ksp_ff_k50_hops | 56.59% | 64.48% | 0.00% | 56.58% | 0.00% | 0.00% | 14.093/25.375 ms | 6.561/11.257 ms |
| fr_quality_c+v12_k50_hops | fr_quality_c | v12_k50_hops | 53.03% | 60.68% | 0.00% | 53.02% | 0.00% | 0.00% | 14.810/26.585 ms | 9.140/15.916 ms |

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