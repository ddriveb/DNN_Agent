# v1.2 K=50 Candidate Coverage Probe (COST239 S100)

## Configuration

- `topology`: `xlron_cost239_ptrnet_real`
- `num_slots`: `100`
- `seeds`: `3030,4040`
- `episodes`: `3`
- `requests_per_episode`: `100`
- `k_paths`: `5`
- `path_sort_strategy`: `km`
- `block_sort_strategy`: `mixed`
- `arrival_interval`: `0.08`
- `holding_min`: `4.0`
- `holding_max`: `15.0`
- `size_min_mb`: `5.0`
- `size_max_mb`: `30.0`

## Blocking results

| Probe | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| v1.2 (k=5 km mixed, all_legal) | 22.33% | 23.50% | 0.00% | 22.33% | 0.00% | 0.00% | 13.179/24.377 ms | 9.181/14.727 ms |
| v1.2 k=50 hops all_legal | 18.33% | 19.17% | 0.00% | 18.33% | 0.00% | 0.00% | 13.416/23.270 ms | 19.522/35.448 ms |
| v1.2 k=50 hops legalctx48 + ensure KSP | 18.33% | 19.17% | 0.00% | 18.33% | 0.00% | 0.00% | 13.416/23.270 ms | 19.700/33.803 ms |
| KSP-FF K=50 hops | 20.00% | 20.67% | 0.00% | 20.00% | 0.00% | 0.00% | 12.024/21.591 ms | 10.335/15.442 ms |

## Delta vs old v1.2

| Probe | Δ Blocking (pp) | Δ NSB (pp) | Δ Overload (pp) |
|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | -4.00 | +0.00 | -4.00 |
| v1.2 k=50 hops legalctx48 + ensure KSP | -4.00 | +0.00 | -4.00 |
| KSP-FF K=50 hops | -2.33 | +0.00 | -2.33 |

## Selected action feature summary

| Probe | Feature | Mean | Median | Std |
|---|---|---:|---:|---:|
| v1.2 k=50 hops all_legal | path_length_km | 974.2767 | 1050.0000 | 541.5307 |
| v1.2 k=50 hops all_legal | hop_count | 1.9633 | 2.0000 | 1.0887 |
| v1.2 k=50 hops all_legal | spectral_efficiency | 1.8467 | 2.0000 | 0.8980 |
| v1.2 k=50 hops all_legal | required_fs | 1.8650 | 2.0000 | 1.0631 |
| v1.2 k=50 hops all_legal | block_size | 42.7550 | 39.0000 | 36.8923 |
| v1.2 k=50 hops all_legal | block_waste | 0.8993 | 0.9661 | 0.1618 |
| v1.2 k=50 hops legalctx48 + ensure KSP | path_length_km | 974.2767 | 1050.0000 | 541.5307 |
| v1.2 k=50 hops legalctx48 + ensure KSP | hop_count | 1.9633 | 2.0000 | 1.0887 |
| v1.2 k=50 hops legalctx48 + ensure KSP | spectral_efficiency | 1.8467 | 2.0000 | 0.8980 |
| v1.2 k=50 hops legalctx48 + ensure KSP | required_fs | 1.8650 | 2.0000 | 1.0631 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_size | 42.7550 | 39.0000 | 36.8923 |
| v1.2 k=50 hops legalctx48 + ensure KSP | block_waste | 0.8993 | 0.9661 | 0.1618 |
| KSP-FF K=50 hops | path_length_km | 826.3042 | 740.0000 | 453.8208 |
| KSP-FF K=50 hops | hop_count | 1.5271 | 2.0000 | 0.7687 |
| KSP-FF K=50 hops | spectral_efficiency | 1.8854 | 2.0000 | 0.6811 |
| KSP-FF K=50 hops | required_fs | 2.2708 | 2.0000 | 0.7916 |
| KSP-FF K=50 hops | block_size | 68.6479 | 76.5000 | 29.9339 |
| KSP-FF K=50 hops | block_waste | 0.8736 | 0.9712 | 0.2779 |
| v1.2 (k=5 km mixed, all_legal) | path_length_km | 913.6300 | 1050.0000 | 484.8420 |
| v1.2 (k=5 km mixed, all_legal) | hop_count | 2.0817 | 2.0000 | 1.1853 |
| v1.2 (k=5 km mixed, all_legal) | spectral_efficiency | 1.8417 | 2.0000 | 0.9451 |
| v1.2 (k=5 km mixed, all_legal) | required_fs | 1.7683 | 2.0000 | 1.0930 |
| v1.2 (k=5 km mixed, all_legal) | block_size | 40.5217 | 33.0000 | 37.5350 |
| v1.2 (k=5 km mixed, all_legal) | block_waste | 0.8814 | 0.9688 | 0.2260 |