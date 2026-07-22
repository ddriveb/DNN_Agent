# v1.3 K=50 Final Fair Comparison — COST239 Old Protocol

## Protocol

- topology: `xlron_cost239_ptrnet_real`
- num_slots: 320
- num_servers: 4
- edge_cost_min: 0.1
- edge_cost_max: 2.2
- requests_per_episode: 10000
- warmup_requests: 2000
- seeds: 3030,4040,5050,6060,7070
- path_sort_strategy: hops
- block_sort_strategy: start_asc
- C checkpoint: `agent_c_cost239_r_feasibility_safe_last.pt`
- v1.3 ranking checkpoint: `r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt`

## Method Definitions

| Method | K | Selector / Ranker |
|---|---|---|
| `ppo_c+v13_k50_hops` | 50 | v1.3 post-decision ranker (postdec_topk_k50_s100) |
| `ppo_c+ksp_ff_plain_k50_hops` | 50 | plain KSP-FF |
| `ppo_c+ksp_ff_highest_mod_k50_hops` | 50 | highest-modulation KSP-FF |
| `ppo_c+deep_rmsa_style_k50_hops` | 50 | hand-crafted DeepRMSA-style proxy |
| `ppo_c+v13_k5_hops` | 5 | v1.3 post-decision ranker |
| `ppo_c+ksp_ff_plain_k5_hops` | 5 | plain KSP-FF |

## Table 1: Primary Blocking & Delay Metrics

| Method | K | Selector / Ranker | Blocking mean ± std | Overload | NSB | raw_empty | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ppo_c+v13_k50_hops | 50 | v1.3 post-decision ranker (postdec_topk_k50_s100) | 6.29% ± 1.08% | 6.29% | 0.00% | 8.34% | 9.118/16.899 | 12.483/18.191 |
| ppo_c+ksp_ff_plain_k50_hops | 50 | plain KSP-FF | 7.92% ± 0.57% | 7.92% | 0.00% | 10.34% | 8.964/16.562 | 8.534/12.773 |
| ppo_c+ksp_ff_highest_mod_k50_hops | 50 | highest-modulation KSP-FF | 9.93% ± 1.42% | 9.93% | 0.00% | 12.97% | 9.086/17.171 | 7.312/11.530 |
| ppo_c+deep_rmsa_style_k50_hops | 50 | hand-crafted DeepRMSA-style proxy | 9.62% ± 1.34% | 9.62% | 0.00% | 12.52% | 10.519/19.161 | 7.581/11.966 |
| ppo_c+v13_k5_hops | 5 | v1.3 post-decision ranker | 7.38% ± 1.10% | 7.38% | 0.00% | 9.70% | 9.246/16.638 | 6.034/8.947 |
| ppo_c+ksp_ff_plain_k5_hops | 5 | plain KSP-FF | 7.46% ± 0.94% | 7.46% | 0.00% | 9.69% | 8.989/16.470 | 4.944/7.685 |

## Table 2: Selected R Action Characteristics

| Method | K | path length mean | hop count mean | required FS mean | selected mod distribution | block start mean | block size mean |
|---|---:|---:|---:|---:|---|---:|---:|
| ppo_c+v13_k50_hops | 50 | 768.06 | 1.45 | 2.88 | 16QAM=24.6%, 8QAM=3.8%, BPSK=41.2%, QPSK=30.4% | 85.37 | 98.39 |
| ppo_c+ksp_ff_plain_k50_hops | 50 | 735.66 | 1.32 | 3.49 | BPSK=100.0%, QPSK=0.0% | 44.66 | 91.55 |
| ppo_c+ksp_ff_highest_mod_k50_hops | 50 | 731.79 | 1.33 | 2.33 | 16QAM=3.3%, 8QAM=17.0%, BPSK=25.6%, QPSK=54.1% | 30.45 | 94.90 |
| ppo_c+deep_rmsa_style_k50_hops | 50 | 925.99 | 1.80 | 2.61 | 16QAM=28.9%, 8QAM=18.9%, BPSK=3.9%, QPSK=48.3% | 53.97 | 73.06 |
| ppo_c+v13_k5_hops | 5 | 758.97 | 1.51 | 2.88 | 16QAM=22.8%, 8QAM=2.6%, BPSK=36.3%, QPSK=38.3% | 58.25 | 83.26 |
| ppo_c+ksp_ff_plain_k5_hops | 5 | 743.79 | 1.33 | 3.50 | BPSK=100.0%, QPSK=0.0% | 45.59 | 92.42 |

## Table 3: Split & Server Selection Distributions

| Method | K | split0 | split1 | split2 | server0 | server1 | server2 | server3 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| ppo_c+v13_k50_hops | 50 | 8.34% | 0.48% | 91.18% | 27.98% | 24.41% | 23.35% | 24.27% |
| ppo_c+ksp_ff_plain_k50_hops | 50 | 10.48% | 1.04% | 88.48% | 28.50% | 23.71% | 23.76% | 24.03% |
| ppo_c+ksp_ff_highest_mod_k50_hops | 50 | 13.33% | 1.23% | 85.45% | 30.02% | 22.96% | 23.34% | 23.69% |
| ppo_c+deep_rmsa_style_k50_hops | 50 | 12.80% | 1.57% | 85.63% | 29.59% | 23.28% | 23.50% | 23.63% |
| ppo_c+v13_k5_hops | 5 | 9.87% | 1.04% | 89.10% | 28.11% | 23.70% | 24.22% | 23.97% |
| ppo_c+ksp_ff_plain_k5_hops | 5 | 9.81% | 1.04% | 89.14% | 28.26% | 23.79% | 23.80% | 24.15% |

## Table 4: Pairwise Fairness Comparison

| Pair | Same K? | Same C? | Main fairness? | Δ Blocking (pp) | Interpretation |
|---|---|---|---:|---:|---|
| v13_k50_hops vs ksp_ff_plain_k50_hops | true | yes (all ppo_c) | main | -1.63 | ranker vs plain FF |
| v13_k50_hops vs ksp_ff_highest_mod_k50_hops | true | yes (all ppo_c) | main | -3.64 | ranker vs optical heuristic |
| v13_k50_hops vs deep_rmsa_style_k50_hops | true | yes (all ppo_c) | auxiliary | -3.33 | ranker vs DeepRMSA-style proxy |
| v13_k50_hops vs v13_k5_hops | false | yes (all ppo_c) | diagnostic | -1.10 | candidate horizon sensitivity |
| v13_k50_hops vs ksp_ff_plain_k5_hops | false | yes (all ppo_c) | boundary only | -1.17 | not primary fairness claim |

## Per-Seed Blocking Rates

| Method | 3030 | 4040 | 5050 | 6060 | 7070 |
|---|---:|---:|---:|---:|---:|
| ppo_c+v13_k50_hops | 6.09% | 6.55% | 4.80% | 8.11% | 5.88% |
| ppo_c+ksp_ff_plain_k50_hops | 7.49% | 8.28% | 7.07% | 8.66% | 8.09% |
| ppo_c+ksp_ff_highest_mod_k50_hops | 8.50% | 11.11% | 8.64% | 12.09% | 9.31% |
| ppo_c+deep_rmsa_style_k50_hops | 10.09% | 10.21% | 7.15% | 11.15% | 9.49% |
| ppo_c+v13_k5_hops | 7.19% | 7.89% | 6.04% | 9.21% | 6.58% |
| ppo_c+ksp_ff_plain_k5_hops | 7.49% | 8.28% | 6.11% | 8.66% | 6.75% |

## Conclusions

1. **v1.3 K50 is the strongest method under matched K50 path support.**
   - vs plain FF K50: -1.63 pp (lower is better)
   - vs highest-mod FF K50: -3.64 pp
   - vs DeepRMSA-style K50 proxy: -3.33 pp

2. **All failures remain `server_overload`**, confirming COST239 is a compute-overload-dominated regime.

3. **v1.3 K50 even beats v1.3 K5 and plain FF K5**, so the improvement is not merely from changing the candidate horizon.

4. **The primary fair claim is v1.3 K50 vs KSP-FF K50 variants** (same K, same C, same path support).
   K5 results are diagnostic boundary cases only.

5. `deep_rmsa_style_k50_hops` is a hand-crafted proxy, not the neural DeepRMSA checkpoint.
   Any comparison against it should be labeled accordingly.

## What Can Go Into the Paper

- **Main text**: Table 1 and Table 4 rows 1–3 (v1.3 K50 vs KSP-FF K50 variants).
- **Appendix / diagnostic**: Table 4 rows 4–5 (K5 boundary) and Table 2/3 detailed action characteristics.
- **Avoid**: presenting v1.3 K5 as the primary baseline; the fairness claim is K50-matched.

## Paper-Ready English Paragraph

We evaluate the proposed v1.3 post-decision ranker under matched K=50 hop-ordered path support. To avoid conflating path-horizon effects with selector effects, KSP-FF is separated into plain first-fit and highest-modulation first-fit variants. The primary comparison is therefore v1.3 K50 against KSP-FF variants using the same K50 path support and the same PPO-C policy. K5 results are reported only as diagnostic boundary cases, because they change the R-side candidate support and are not the primary fairness claim.

## Reproducibility Metadata

- Git commit: `6f6337f44c036518886ce8e1146b07b6b6f6d1af`
- Git dirty: `True`
- Python: `3.14.4`
- Torch: `2.11.0+cu130`
- Numpy: `2.4.4`