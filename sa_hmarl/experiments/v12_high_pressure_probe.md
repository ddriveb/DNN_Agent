# v1.2 K=50 Candidate Coverage Probe (COST239 S100)

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `seeds`: `3030,4040,5050,6060,7070`
- `episodes`: `10`
- `requests_per_episode`: `90`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `block_sort_strategy`: `mixed`
- `arrival_interval`: `0.09`
- `holding_min`: `4.0`
- `holding_max`: `13.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`

## Blocking results

| Probe | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 (k=5 km mixed, all_legal) | 12.51% | 13.09% | 0.00% | 12.51% | 0.00% | 0.00% | 12.320/22.262 ms | 8.423/12.921 ms |
| v1.2 k=50 hops all_legal | 12.91% | 13.33% | 0.00% | 12.91% | 0.00% | 0.00% | 12.818/22.713 ms | 17.459/29.908 ms |
| v1.2 k=50 hops legalctx48 + ensure KSP | 12.91% | 13.33% | 0.00% | 12.91% | 0.00% | 0.00% | 12.818/22.713 ms | 17.840/30.268 ms |
| KSP-FF K=50 hops | 12.53% | 13.18% | 0.00% | 12.53% | 0.00% | 0.00% | 11.385/21.352 ms | 9.815/13.660 ms |

## Delta vs old v1.2

| Probe | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) |
|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | +0.40 | +0.00 | +0.40 |
| v1.2 k=50 hops legalctx48 + ensure KSP | +0.40 | +0.00 | +0.40 |
| KSP-FF K=50 hops | +0.02 | +0.00 | +0.02 |

## Selected action feature summary

| Probe | Feature | Mean | Median | Std |
|---|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | path_length_km | 956.6378 | 1034.0000 | 538.7837 |
| v1.2 k=50 hops all_legal | hop_count | 1.9636 | 2.0000 | 1.0938 |
| v1.2 k=50 hops all_legal | spectral_efficiency | 2.0147 | 2.0000 | 0.9754 |
| v1.2 k=50 hops all_legal | required_fs | 1.9589 | 2.0000 | 1.0086 |
| v1.2 k=50 hops all_legal | block_size | 44.9480 | 41.0000 | 37.9687 |
| v1.2 k=50 hops all_legal | block_waste | 0.8741 | 0.9649 | 0.1970 |
| v1.2 k=50 hops legalctx48 + ensure KSP | path_length_km | 956.6378 | 1034.0000 | 538.7837 |
| v1.2 k=50 hops legalctx48 + ensure KSP | hop_count | 1.9636 | 2.0000 | 1.0938 |
| v1.2 k=50 hops legalctx48 + ensure KSP | spectral_efficiency | 2.0147 | 2.0000 | 0.9754 |
| v1.2 k=50 hops legalctx48 + ensure KSP | required_fs | 1.9589 | 2.0000 | 1.0086 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_size | 44.9480 | 41.0000 | 37.9687 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_waste | 0.8741 | 0.9649 | 0.1970 |
| KSP-FF K=50 hops | path_length_km | 797.3338 | 740.0000 | 514.2851 |
| KSP-FF K=50 hops | hop_count | 1.4446 | 2.0000 | 0.7667 |
| KSP-FF K=50 hops | spectral_efficiency | 1.8910 | 2.0000 | 0.7578 |
| KSP-FF K=50 hops | required_fs | 2.2660 | 2.0000 | 0.7003 |
| KSP-FF K=50 hops | block_size | 70.2035 | 78.0000 | 29.0751 |
| KSP-FF K=50 hops | block_waste | 0.8852 | 0.9718 | 0.2604 |
| v1.2 (k=5 km mixed, all_legal) | path_length_km | 887.5493 | 987.0000 | 473.5769 |
| v1.2 (k=5 km mixed, all_legal) | hop_count | 1.9973 | 2.0000 | 1.0889 |
| v1.2 (k=5 km mixed, all_legal) | spectral_efficiency | 2.0371 | 2.0000 | 1.0030 |
| v1.2 (k=5 km mixed, all_legal) | required_fs | 1.9764 | 2.0000 | 1.0013 |
| v1.2 (k=5 km mixed, all_legal) | block_size | 45.1167 | 42.0000 | 38.1209 |
| v1.2 (k=5 km mixed, all_legal) | block_waste | 0.8451 | 0.9655 | 0.2596 |