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
- `arrival_interval`: `0.1`
- `holding_min`: `4.0`
- `holding_max`: `12.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`

## Blocking results

| Probe | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 (k=5 km mixed, all_legal) | 0.42% | 0.42% | 0.00% | 0.42% | 0.00% | 0.00% | 10.198/18.476 ms | 9.218/13.804 ms |
| v1.2 k=50 hops all_legal | 0.62% | 0.62% | 0.00% | 0.62% | 0.00% | 0.00% | 10.510/19.463 ms | 17.390/30.936 ms |
| v1.2 k=50 hops legalctx48 + ensure KSP | 0.62% | 0.62% | 0.00% | 0.62% | 0.00% | 0.00% | 10.510/19.463 ms | 18.065/32.256 ms |
| KSP-FF K=50 hops | 3.12% | 3.12% | 0.00% | 3.12% | 0.00% | 0.00% | 9.446/16.966 ms | 10.943/15.511 ms |

## Delta vs old v1.2

| Probe | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) |
|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | +0.21 | +0.00 | +0.21 |
| v1.2 k=50 hops legalctx48 + ensure KSP | +0.21 | +0.00 | +0.21 |
| KSP-FF K=50 hops | +2.71 | +0.00 | +2.71 |

## Selected action feature summary

| Probe | Feature | Mean | Median | Std |
|---|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | path_length_km | 669.3250 | 660.0000 | 546.1812 |
| v1.2 k=50 hops all_legal | hop_count | 1.4292 | 1.0000 | 1.2038 |
| v1.2 k=50 hops all_legal | spectral_efficiency | 2.4854 | 2.0000 | 1.1291 |
| v1.2 k=50 hops all_legal | required_fs | 2.1646 | 2.0000 | 0.4958 |
| v1.2 k=50 hops all_legal | block_size | 65.7479 | 80.0000 | 36.2722 |
| v1.2 k=50 hops all_legal | block_waste | 0.8956 | 0.9740 | 0.1832 |
| v1.2 k=50 hops legalctx48 + ensure KSP | path_length_km | 669.3250 | 660.0000 | 546.1812 |
| v1.2 k=50 hops legalctx48 + ensure KSP | hop_count | 1.4292 | 1.0000 | 1.2038 |
| v1.2 k=50 hops legalctx48 + ensure KSP | spectral_efficiency | 2.4854 | 2.0000 | 1.1291 |
| v1.2 k=50 hops legalctx48 + ensure KSP | required_fs | 2.1646 | 2.0000 | 0.4958 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_size | 65.7479 | 80.0000 | 36.2722 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_waste | 0.8956 | 0.9740 | 0.1832 |
| KSP-FF K=50 hops | path_length_km | 540.5591 | 660.0000 | 412.9741 |
| KSP-FF K=50 hops | hop_count | 0.9914 | 1.0000 | 0.7302 |
| KSP-FF K=50 hops | spectral_efficiency | 1.9419 | 2.0000 | 0.7944 |
| KSP-FF K=50 hops | required_fs | 2.1828 | 2.0000 | 0.6037 |
| KSP-FF K=50 hops | block_size | 77.7656 | 87.0000 | 28.0901 |
| KSP-FF K=50 hops | block_waste | 0.9031 | 0.9762 | 0.2395 |
| v1.2 (k=5 km mixed, all_legal) | path_length_km | 635.9750 | 660.0000 | 489.7242 |
| v1.2 (k=5 km mixed, all_legal) | hop_count | 1.3646 | 1.0000 | 1.1210 |
| v1.2 (k=5 km mixed, all_legal) | spectral_efficiency | 2.5229 | 2.0000 | 1.1490 |
| v1.2 (k=5 km mixed, all_legal) | required_fs | 2.1833 | 2.0000 | 0.5662 |
| v1.2 (k=5 km mixed, all_legal) | block_size | 65.1479 | 79.0000 | 37.0304 |
| v1.2 (k=5 km mixed, all_legal) | block_waste | 0.8533 | 0.9737 | 0.2653 |