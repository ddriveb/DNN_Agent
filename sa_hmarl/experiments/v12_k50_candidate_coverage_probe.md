# v1.2 K=50 Candidate Coverage Probe (COST239 S100)

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `10`
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
| v1.2 (k=5 km mixed, all_legal) | 0.12% | 0.12% | 0.00% | 0.12% | 0.00% | 0.00% | 10.907/19.855 ms | 9.614/14.963 ms |
| v1.2 k=50 hops all_legal | 0.20% | 0.20% | 0.00% | 0.20% | 0.00% | 0.00% | 11.142/20.571 ms | 21.829/55.964 ms |
| v1.2 k=50 hops legalctx48 + ensure KSP | 0.20% | 0.20% | 0.00% | 0.20% | 0.00% | 0.00% | 11.142/20.571 ms | 21.925/55.768 ms |
| KSP-FF K=50 hops | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 0.00% | 10.464/19.466 ms | 9.736/13.395 ms |

## Delta vs old v1.2

| Probe | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) |
|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | +0.07 | +0.00 | +0.07 |
| v1.2 k=50 hops legalctx48 + ensure KSP | +0.07 | +0.00 | +0.07 |
| KSP-FF K=50 hops | -0.12 | +0.00 | -0.12 |

## Selected action feature summary

| Probe | Feature | Mean | Median | Std |
|---|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | path_length_km | 727.4675 | 684.0000 | 582.1851 |
| v1.2 k=50 hops all_legal | hop_count | 1.6670 | 2.0000 | 1.2995 |
| v1.2 k=50 hops all_legal | spectral_efficiency | 2.4405 | 2.0000 | 1.1358 |
| v1.2 k=50 hops all_legal | required_fs | 2.1480 | 2.0000 | 0.4556 |
| v1.2 k=50 hops all_legal | block_size | 59.9943 | 73.0000 | 38.4414 |
| v1.2 k=50 hops all_legal | block_waste | 0.8677 | 0.9697 | 0.2124 |
| v1.2 k=50 hops legalctx48 + ensure KSP | path_length_km | 727.4675 | 684.0000 | 582.1851 |
| v1.2 k=50 hops legalctx48 + ensure KSP | hop_count | 1.6670 | 2.0000 | 1.2995 |
| v1.2 k=50 hops legalctx48 + ensure KSP | spectral_efficiency | 2.4405 | 2.0000 | 1.1358 |
| v1.2 k=50 hops legalctx48 + ensure KSP | required_fs | 2.1480 | 2.0000 | 0.4556 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_size | 59.9943 | 73.0000 | 38.4414 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_waste | 0.8677 | 0.9697 | 0.2124 |
| KSP-FF K=50 hops | path_length_km | 679.5355 | 660.0000 | 583.7059 |
| KSP-FF K=50 hops | hop_count | 1.2190 | 1.0000 | 0.8928 |
| KSP-FF K=50 hops | spectral_efficiency | 1.8787 | 2.0000 | 0.8140 |
| KSP-FF K=50 hops | required_fs | 2.2218 | 2.0000 | 0.6458 |
| KSP-FF K=50 hops | block_size | 68.4167 | 86.0000 | 37.5150 |
| KSP-FF K=50 hops | block_waste | 0.7909 | 0.9753 | 0.3609 |
| v1.2 (k=5 km mixed, all_legal) | path_length_km | 685.9220 | 684.0000 | 516.4538 |
| v1.2 (k=5 km mixed, all_legal) | hop_count | 1.6262 | 2.0000 | 1.2428 |
| v1.2 (k=5 km mixed, all_legal) | spectral_efficiency | 2.4808 | 2.0000 | 1.1587 |
| v1.2 (k=5 km mixed, all_legal) | required_fs | 2.1643 | 2.0000 | 0.4825 |
| v1.2 (k=5 km mixed, all_legal) | block_size | 56.1430 | 73.0000 | 41.7190 |
| v1.2 (k=5 km mixed, all_legal) | block_waste | 0.7591 | 0.9683 | 0.3448 |