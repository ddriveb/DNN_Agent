# v1.2 K=50 Candidate Coverage Probe (COST239 S100)

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `seeds`: `3030,4040`
- `episodes`: `3`
- `requests_per_episode`: `80`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `block_sort_strategy`: `mixed`
- `arrival_interval`: `0.15`
- `holding_min`: `4.0`
- `holding_max`: `10.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`

## Blocking results

| Probe | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 (k=5 km mixed, all_legal) | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 9.336/16.545 ms | 9.121/13.597 ms |
| v1.2 k=50 hops all_legal | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 9.582/17.505 ms | 17.462/40.397 ms |
| v1.2 k=50 hops legalctx48 + ensure KSP | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 9.582/17.505 ms | 17.585/41.371 ms |
| KSP-FF K=50 hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 9.261/18.276 ms | 8.957/12.791 ms |

## Delta vs old v1.2

| Probe | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) |
|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | +0.00 | +0.00 | +0.00 |
| v1.2 k=50 hops legalctx48 + ensure KSP | +0.00 | +0.00 | +0.00 |
| KSP-FF K=50 hops | +0.00 | +0.00 | +0.00 |

## Selected action feature summary

| Probe | Feature | Mean | Median | Std |
|---|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | path_length_km | 579.9417 | 550.0000 | 554.4186 |
| v1.2 k=50 hops all_legal | hop_count | 1.2125 | 1.0000 | 1.1764 |
| v1.2 k=50 hops all_legal | spectral_efficiency | 2.6958 | 3.0000 | 1.1899 |
| v1.2 k=50 hops all_legal | required_fs | 2.1083 | 2.0000 | 0.3662 |
| v1.2 k=50 hops all_legal | block_size | 69.3896 | 89.0000 | 37.9951 |
| v1.2 k=50 hops all_legal | block_waste | 0.8718 | 0.9767 | 0.2287 |
| v1.2 k=50 hops legalctx48 + ensure KSP | path_length_km | 579.9417 | 550.0000 | 554.4186 |
| v1.2 k=50 hops legalctx48 + ensure KSP | hop_count | 1.2125 | 1.0000 | 1.1764 |
| v1.2 k=50 hops legalctx48 + ensure KSP | spectral_efficiency | 2.6958 | 3.0000 | 1.1899 |
| v1.2 k=50 hops legalctx48 + ensure KSP | required_fs | 2.1083 | 2.0000 | 0.3662 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_size | 69.3896 | 89.0000 | 37.9951 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_waste | 0.8718 | 0.9767 | 0.2287 |
| KSP-FF K=50 hops | path_length_km | 461.3958 | 390.0000 | 409.2619 |
| KSP-FF K=50 hops | hop_count | 0.9313 | 1.0000 | 0.8046 |
| KSP-FF K=50 hops | spectral_efficiency | 1.9250 | 2.0000 | 0.8258 |
| KSP-FF K=50 hops | required_fs | 2.1021 | 2.0000 | 0.3594 |
| KSP-FF K=50 hops | block_size | 71.9562 | 92.0000 | 37.9466 |
| KSP-FF K=50 hops | block_waste | 0.8000 | 0.9777 | 0.3595 |
| v1.2 (k=5 km mixed, all_legal) | path_length_km | 540.2083 | 550.0000 | 498.3577 |
| v1.2 (k=5 km mixed, all_legal) | hop_count | 1.1854 | 1.0000 | 1.1497 |
| v1.2 (k=5 km mixed, all_legal) | spectral_efficiency | 2.7083 | 3.0000 | 1.2103 |
| v1.2 (k=5 km mixed, all_legal) | required_fs | 2.1167 | 2.0000 | 0.4069 |
| v1.2 (k=5 km mixed, all_legal) | block_size | 64.6708 | 89.0000 | 42.1433 |
| v1.2 (k=5 km mixed, all_legal) | block_waste | 0.7740 | 0.9766 | 0.3417 |