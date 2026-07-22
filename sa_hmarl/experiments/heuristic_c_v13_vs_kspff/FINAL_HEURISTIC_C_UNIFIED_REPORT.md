# Unified Heuristic-C Main Table Report

**Date:** 2026-07-05

This report presents a single, protocol-unified comparison of the adapted heuristic C-side baselines (`df_c` and `rf_c`) against the KSP-FF K50 hops and v1.3 hybrid R-side backends across four imported topologies. All entries in the main table were produced under the exact same protocol, so they can be placed directly in a paper table without risk of mixing different evaluation configurations.

## 1. df_c Audit Summary

Existing `df_c` full results were audited against the unified protocol. None of the four topologies satisfied the v1.3 configuration requirement because the previous runs did not explicitly set `ranker_candidate_mode=legalctx48`, `ranker_max_candidates=48`, or `ranker_ensure_ksp=true`. Although the checkpoint defaults may have produced similar behavior in some cases, for a paper-ready main table we require all runs to use the same explicit configuration. Therefore all four `df_c` topologies were re-run under the unified protocol. The old non-compliant files (`xlron_*_df_full.json`) are retained for reference but are not used in the main table.

| Topology | Existing file | Compliant | Action | Reason |
|---|---:|---|---|---|
| COST239 | xlron_cost239_ptrnet_real_df_full.json | No | re-run | ranker_candidate_mode=None, ranker_max_candidates=None, ranker_ensure_ksp=False |
| German17 | xlron_german17_df_full.json | No | re-run | ranker_candidate_mode=None, ranker_max_candidates=None, ranker_ensure_ksp=False |
| NSFNET | xlron_nsfnet_deeprmsa_df_full.json | No | re-run | ranker_candidate_mode=None, ranker_max_candidates=None, ranker_ensure_ksp=False |
| JPN48 | xlron_jpn48_df_full.json | No | re-run | ranker_candidate_mode=None, ranker_max_candidates=None, ranker_ensure_ksp=False |

## 2. Unified Protocol

| Parameter | Value |
|---|---|
| `block_sort_strategy` | `start_asc` |
| `path_sort_strategy` | `hops` |
| `ksp_ff_k50_hops_k_paths` | 50 |
| `requests_per_episode` | 10000 |
| `warmup_requests` | 2000 |
| `seeds` | 3030, 4040, 5050, 6060, 7070 |
| `episodes` | 1 |
| `poisson_arrivals` | true |
| `exponential_holding` | true |
| `num_slots` | 320 |
| `num_servers` | 4 |
| v1.3 `ranker_candidate_mode` | `legalctx48` |
| v1.3 `ranker_max_candidates` | 48 |
| v1.3 `ranker_ensure_ksp` | true |

The R-side column labeled `v12_k50_hops` in the script interface is the **unified v1.3 hybrid configuration** above. We keep the script method name for backward compatibility but note explicitly in this report that the backend is v1.3 hybrid.

## 3. Main Table: df_c and rf_c under Unified Protocol

### 3.1 df_c

| Topology | df_c + KSP-FF | df_c + v1.3 | Δ (pp) | p-value | overload | NSB |
|---|---:|---:|---:|---:|---:|---:|
| COST239 | 14.84% | 14.84% | +0.00 pp | NA | 14.84% | 0.00% |
| German17 | 12.69% | 12.80% | +0.11 pp | 0.3483 | 12.67% | 0.02% |
| NSFNET | 20.33% | 20.26% | -0.07 pp | 0.8628 | 20.10% | 0.07% |
| JPN48 | 11.28% | 11.06% | -0.22 pp | 0.2015 | 10.80% | 0.15% |

### 3.2 rf_c

| Topology | rf_c + KSP-FF | rf_c + v1.3 | Δ (pp) | p-value | overload | NSB |
|---|---:|---:|---:|---:|---:|---:|
| COST239 | 15.17% | 15.17% | +0.00 pp | NA | 15.17% | 0.00% |
| German17 | 13.32% | 12.78% | -0.54 pp | 0.3116 | 13.31% | 0.01% |
| NSFNET | 20.53% | 21.10% | +0.56 pp | 0.3125 | 20.25% | 0.10% |
| JPN48 | 12.84% | 11.67% | -1.17 pp | 0.0225 | 12.40% | 0.18% |

### 3.3 Cross-baseline comparison (v1.3 column)

| Topology | PPO-C + v1.3 | df_c + v1.3 | rf_c + v1.3 |
|---|---:|---:|---:|
| COST239 | 7.68% | 14.84% | 15.17% |
| German17 | 7.63% | 12.80% | 12.78% |
| NSFNET | 14.92% | 20.26% | 21.10% |
| JPN48 | 8.03% | 11.06% | 11.67% |

## 4. Fixed-Split Supporting Ablation (Qualitative)

The previously generated `df_fixed0_c` and `rf_fixed0_c` results are retained as strong supporting evidence. They robustly show that removing split adaptivity collapses performance to 68–75% blocking and eliminates observable R-side gain (v1.3 vs KSP-FF delta within ±0.36 pp, all p > 0.37). Because these ablations were produced under the same evaluation script but with a pinned split, they are used **qualitatively** rather than as row-by-row directly comparable entries in the main heuristic-C table.

| Topology | df_fixed0 + v1.3 | rf_fixed0 + v1.3 |
|---|---:|---:|
| COST239 | 73.97% | 74.43% |
| German17 | 72.58% | 72.30% |
| NSFNET | 72.80% | 72.68% |
| JPN48 | 68.53% | 68.42% |

## 5. Interpretation

1. **PPO-C remains the strongest C-side policy.** Learned PPO-C blocking is substantially lower than both `df_c` and `rf_c` on every topology, confirming that the learned split/server allocation is a key component of the overall system.

2. **Heuristic C attenuates v1.3 gains and makes them topology-dependent.** Under unified `df_c` and `rf_c`, the v1.3 hybrid advantage over KSP-FF is smaller and less systematic than under PPO-C. This reinforces that v1.3's benefit is conditional on the C-side action distribution.

3. **Residual blocking is dominated by `server_overload`.** In every main-table entry the primary failure mode is server overload; `no_suitable_block` remains small. The bottleneck is C-side server/split allocation, not R-side candidate set size.

4. **Fixed-split ablation confirms split adaptivity is essential.** Pinning the split to `split0` raises blocking to 68–75% and removes the v1.3 advantage entirely. This provides a clean lower-bound argument: without the ability to choose splits, even an optimal R-side ranker cannot compensate.

## 6. Paper-Ready Summary Draft

> Table X compares the adapted heuristic C-side baselines `df_c` and `rf_c` against the
> KSP-FF K50 hops and v1.3 hybrid R-side backends under a single unified protocol across
> COST239, German17, NSFNET and JPN48. The v1.3 column (`v12_k50_hops` in the script label)
> uses the same configuration everywhere: `ranker_candidate_mode=legalctx48`,
> `ranker_max_candidates=48`, `ranker_ensure_ksp=true`. Learned PPO-C remains the strongest
> C-side policy, while heuristic C yields higher blocking and a weaker, topology-dependent
> v1.3 advantage. A fixed-split / no-partition-style ablation (`df_fixed0_c`, `rf_fixed0_c`)
> raises blocking to 68–75% and eliminates observable R-side gain, confirming that adaptive
> split selection is a prerequisite for the v1.3 ranker to matter.

## 7. Suggested Division of Labour in the Paper

- **Main heuristic-C table:** include only the unified `df_c` and `rf_c` rows from this report (Section 3). They share one protocol and one v1.3 config and can be directly compared across topologies.

- **Comparison to learned PPO-C:** use the PPO-C + v1.3 column (Section 3.3) to show that learned C-side allocation sets a much lower blocking floor.

- **Supporting ablation:** place the fixed-split results (Section 4) in a separate paragraph or small table, labelled as a no-partition-style lower-bound ablation. Use them to argue that split adaptivity is essential and that R-side gains vanish without it.

## 8. Commands Used

### df_c unified re-runs
```bash
cd /mnt/d/project/DNN_Agent
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_unified_topology.sh xlron_cost239_ptrnet_real 0.0625 2.2 xlron_cost239_ptrnet_real
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_unified_topology.sh xlron_german17 0.07142857142857142 3.0 xlron_german17
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_unified_topology.sh xlron_nsfnet_deeprmsa 0.07692307692307693 4.0 xlron_nsfnet_deeprmsa
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_df_unified_topology.sh xlron_jpn48 0.1 5.2 xlron_jpn48
```

### rf_c unified runs
```bash
cd /mnt/d/project/DNN_Agent
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_unified_topology.sh xlron_cost239_ptrnet_real 0.0625 2.2 xlron_cost239_ptrnet_real
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_unified_topology.sh xlron_german17 0.07142857142857142 3.0 xlron_german17
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_unified_topology.sh xlron_nsfnet_deeprmsa 0.07692307692307693 4.0 xlron_nsfnet_deeprmsa
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_rf_unified_topology.sh xlron_jpn48 0.1 5.2 xlron_jpn48
```

## 9. Status Checklist

- [x] Audit existing `df_c` results against unified protocol
- [x] Re-run all `df_c` topologies under unified protocol
- [x] Run all `rf_c` topologies under unified protocol
- [x] Generate unified main-table report
- [x] Retain fixed-split ablation as qualitative supporting evidence
