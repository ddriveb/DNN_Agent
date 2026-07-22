# df_c Existing Results Audit Against Unified Protocol

## Unified Protocol

| Parameter | Value |
|---|---|
| block_sort_strategy | start_asc |
| path_sort_strategy | hops |
| ksp_ff_k50_hops_k_paths | 50 |
| requests_per_episode | 10000 |
| warmup_requests | 2000 |
| seeds | 3030,4040,5050,6060,7070 |
| episodes | 1 |
| poisson_arrivals | True |
| exponential_holding | True |
| num_slots | 320 |
| num_servers | 4 |
| split_profile | default3 |
| num_splits | 3 |
| v1.3 ranker_candidate_mode | legalctx48 |
| v1.3 ranker_max_candidates | 48 |
| v1.3 ranker_ensure_ksp | true |

| Topology | arrival_interval | edge_cost_max |
|---|---|---|
| xlron_cost239_ptrnet_real | 0.0625 | 2.2 |
| xlron_german17 | 0.07142857142857142 | 3.0 |
| xlron_nsfnet_deeprmsa | 0.07692307692307693 | 4.0 |
| xlron_jpn48 | 0.1 | 5.2 |

## Audit Results

| Topology | Compliant | Mismatches | Present Methods | Action |
|---|---|---|---|---|
| xlron_cost239_ptrnet_real | False | ranker_candidate_mode: expected 'legalctx48', got None; ranker_max_candidates: expected 48, got None; ranker_ensure_ksp: expected True, got False | df_c+ksp_ff_k50_hops, df_c+v12_k50_hops | re-run |
| xlron_german17 | False | ranker_candidate_mode: expected 'legalctx48', got None; ranker_max_candidates: expected 48, got None; ranker_ensure_ksp: expected True, got False | df_c+ksp_ff_k50_hops, df_c+v12_k50_hops | re-run |
| xlron_nsfnet_deeprmsa | False | ranker_candidate_mode: expected 'legalctx48', got None; ranker_max_candidates: expected 48, got None; ranker_ensure_ksp: expected True, got False | df_c+ksp_ff_k50_hops, df_c+v12_k50_hops | re-run |
| xlron_jpn48 | False | ranker_candidate_mode: expected 'legalctx48', got None; ranker_max_candidates: expected 48, got None; ranker_ensure_ksp: expected True, got False | df_c+ksp_ff_k50_hops, df_c+v12_k50_hops | re-run |
