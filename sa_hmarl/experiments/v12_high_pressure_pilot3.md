# v1.2 K=50 Candidate Coverage Probe (COST239 S100)

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `seeds`: `3030,4040`
- `episodes`: `3`
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
| v1.2 (k=5 km mixed, all_legal) | 15.37% | 16.30% | 0.00% | 15.37% | 0.00% | 0.00% | 12.024/22.423 ms | 9.684/15.039 ms |
| v1.2 k=50 hops all_legal | 13.33% | 13.89% | 0.00% | 13.33% | 0.00% | 0.00% | 11.978/22.053 ms | 19.566/35.679 ms |
| v1.2 k=50 hops legalctx48 + ensure KSP | 13.33% | 13.89% | 0.00% | 13.33% | 0.00% | 0.00% | 11.978/22.053 ms | 19.633/35.444 ms |
| KSP-FF K=50 hops | 17.04% | 17.78% | 0.00% | 17.04% | 0.00% | 0.00% | 11.070/21.134 ms | 10.375/15.224 ms |

## Delta vs old v1.2

| Probe | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) |
|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | -2.04 | +0.00 | -2.04 |
| v1.2 k=50 hops legalctx48 + ensure KSP | -2.04 | +0.00 | -2.04 |
| KSP-FF K=50 hops | +1.67 | +0.00 | +1.67 |

## Selected action feature summary

| Probe | Feature | Mean | Median | Std |
|---|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | path_length_km | 833.1815 | 854.0000 | 499.9782 |
| v1.2 k=50 hops all_legal | hop_count | 1.6963 | 2.0000 | 1.0330 |
| v1.2 k=50 hops all_legal | spectral_efficiency | 2.1185 | 2.0000 | 1.0474 |
| v1.2 k=50 hops all_legal | required_fs | 1.9611 | 2.0000 | 1.0094 |
| v1.2 k=50 hops all_legal | block_size | 47.7315 | 47.5000 | 38.9603 |
| v1.2 k=50 hops all_legal | block_waste | 0.8880 | 0.9682 | 0.1836 |
| v1.2 k=50 hops legalctx48 + ensure KSP | path_length_km | 833.1815 | 854.0000 | 499.9782 |
| v1.2 k=50 hops legalctx48 + ensure KSP | hop_count | 1.6963 | 2.0000 | 1.0330 |
| v1.2 k=50 hops legalctx48 + ensure KSP | spectral_efficiency | 2.1185 | 2.0000 | 1.0474 |
| v1.2 k=50 hops legalctx48 + ensure KSP | required_fs | 1.9611 | 2.0000 | 1.0094 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_size | 47.7315 | 47.5000 | 38.9603 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_waste | 0.8880 | 0.9682 | 0.1836 |
| KSP-FF K=50 hops | path_length_km | 724.4866 | 690.0000 | 530.6435 |
| KSP-FF K=50 hops | hop_count | 1.2946 | 1.0000 | 0.7779 |
| KSP-FF K=50 hops | spectral_efficiency | 1.8527 | 2.0000 | 0.7710 |
| KSP-FF K=50 hops | required_fs | 2.3192 | 2.0000 | 0.9441 |
| KSP-FF K=50 hops | block_size | 73.1027 | 82.0000 | 28.4762 |
| KSP-FF K=50 hops | block_waste | 0.8962 | 0.9737 | 0.2427 |
| v1.2 (k=5 km mixed, all_legal) | path_length_km | 816.1444 | 900.0000 | 466.8506 |
| v1.2 (k=5 km mixed, all_legal) | hop_count | 1.8093 | 2.0000 | 1.0675 |
| v1.2 (k=5 km mixed, all_legal) | spectral_efficiency | 2.0481 | 2.0000 | 1.0600 |
| v1.2 (k=5 km mixed, all_legal) | required_fs | 1.9389 | 2.0000 | 1.1197 |
| v1.2 (k=5 km mixed, all_legal) | block_size | 47.4889 | 48.0000 | 39.0138 |
| v1.2 (k=5 km mixed, all_legal) | block_waste | 0.8747 | 0.9698 | 0.2312 |