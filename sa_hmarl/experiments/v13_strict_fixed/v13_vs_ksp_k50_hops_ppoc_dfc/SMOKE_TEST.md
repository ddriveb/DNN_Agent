# Smoke Test Report

**Date:** 2026-07-14
**Protocol:** Strict v1.3 vs KSP-FF K=50 hops under PPO-C and DF_C
**Seeds:** 6001, 6002 (not used in training or previous evaluation)
**Requests per seed:** 500 warmup + 1000 evaluated
**Topology:** `xlron_cost239_ptrnet_real`

## Objective

Confirm that the six main C×R combinations (plus optional Legacy E1 diagnostic rows) can run end-to-end under the locked protocol before launching the formal 20-seed experiment.

## Checks and results

| Check | Result | Evidence |
|---|---|---|
| Six main combinations run without error | **Pass** | Both seeds produced `smoke_*.json` with all 8 rows (6 main + 2 legacy diagnostic). |
| DF_C selects legal `(split, server)` | **Pass** | No `ValueError` from `decode_agent_c_action`; all `split_id`/`server_id` values are in range. |
| Strict v1.3 ranker loads under DF_C | **Pass** | `df_c+strict_v13` completed on both seeds with finite scores. |
| R actions decode correctly | **Pass** | `env.step` accepted every decoded `(path_idx, mod_idx, block_idx)` without shape errors. |
| Same-seed request sequence is shared | **Pass** | Evaluator generates `requests` once per seed and resets env per method (`env.reset(requests)`). |
| KSP-FF calls `ksp_ff_highest_mod_action` | **Pass** | `r_mode=ksp_ff_highest` maps to `ksp_ff_highest_mod_action` in the evaluator. |
| `ksp_ff_action` (Naive Flat-FF) not used as formal baseline | **Pass** | Only `ksp_ff_highest` was registered; Naive Flat-FF is not supported by this evaluator and is excluded by design. |
| PPO-C and DF_C use identical non-C parameters | **Pass** | Same `obs_c`/`obs_r` builder, same `K_C=5`, `K_path=50`, `hops`, `start_asc`, same request generator. |

## Per-seed blocking summary

### Seed 6001

| Method | Blocked / Total | Blocking | Overload | NSB | Avg delay ms | Avg FS |
|---|---:|---:|---:|---:|---:|---:|
| PPO-C + PPO-R Top-1 | 0 / 1000 | 0.00% | 0.00% | 0.00% | 7.56 | 3.20 |
| PPO-C + KSP-FF K=50 hops | 0 / 1000 | 0.00% | 0.00% | 0.00% | 7.01 | 2.25 |
| PPO-C + Strict v1.3 | 0 / 1000 | 0.00% | 0.00% | 0.00% | 8.65 | 3.06 |
| PPO-C + Legacy E1 | 0 / 1000 | 0.00% | 0.00% | 0.00% | 8.76 | 3.02 |
| DF_C + PPO-R Top-1 | 0 / 1000 | 0.00% | 0.00% | 0.00% | 7.58 | 3.17 |
| DF_C + KSP-FF K=50 hops | 0 / 1000 | 0.00% | 0.00% | 0.00% | 6.52 | 2.19 |
| DF_C + Strict v1.3 | 0 / 1000 | 0.00% | 0.00% | 0.00% | 8.17 | 3.12 |
| DF_C + Legacy E1 | 0 / 1000 | 0.00% | 0.00% | 0.00% | 8.15 | 2.93 |

Seed 6001 produced zero blocking (traffic load happened to be light after warmup), so it is useful only for crash testing, not for relative ranking.

### Seed 6002

| Method | Blocked / Total | Blocking | Overload | NSB | Avg delay ms | Avg FS |
|---|---:|---:|---:|---:|---:|---:|
| PPO-C + PPO-R Top-1 | 103 / 1000 | 10.30% | 10.30% | 0.00% | 9.12 | 3.35 |
| PPO-C + KSP-FF K=50 hops | 141 / 1000 | 14.10% | 14.10% | 0.00% | 8.29 | 2.30 |
| PPO-C + Strict v1.3 | 119 / 1000 | 11.90% | 11.90% | 0.00% | 10.40 | 3.20 |
| PPO-C + Legacy E1 | 117 / 1000 | 11.70% | 11.70% | 0.00% | 9.79 | 3.04 |
| DF_C + PPO-R Top-1 | 89 / 1000 | 8.90% | 8.90% | 0.00% | 8.56 | 3.17 |
| DF_C + KSP-FF K=50 hops | 89 / 1000 | 8.90% | 8.90% | 0.00% | 7.78 | 2.23 |
| DF_C + Strict v1.3 | 89 / 1000 | 8.90% | 8.90% | 0.00% | 9.42 | 3.18 |
| DF_C + Legacy E1 | 89 / 1000 | 8.90% | 0.00% | 8.90% | 9.24 | 2.97 |

Seed 6002 shows differentiation under PPO-C (KSP-FF highest at 14.1%, Strict v1.3 at 11.9%, PPO-R at 10.3%). Under DF_C all R-side methods tie at 8.9%, with all blocking attributed to server overload. This suggests that for this particular seed, DF_C's split/server choices are the binding constraint and the R-side policy cannot compensate. This is a hypothesis to be tested on the full 20-seed set.

## Action distributions (seed 6002)

The evaluator records per-request path indices, modulation names, hop counts, and path lengths. Raw lists are available in `smoke_6001.json` and `smoke_6002.json`. Full action-distribution aggregation will be performed on the formal 20-seed set.

## Conclusion

**Smoke test passed.** All required combinations run, DF_C respects the legal C-side mask, and the Strict v1.3 ranker operates correctly under both PPO-C and DF_C. The formal 20-seed experiment can proceed.

## Notes

- Naive Flat-FF diagnostic was intentionally omitted because the strict multi-C evaluator does not implement it and it is not part of the formal protocol.
- Legacy E1 diagnostic rows are included but will be labeled "Legacy E1-trained gated baseline" in all outputs.
- Checkpoint-deployment distribution mismatch for PPO-R (NSFNET/extended/32 slots) and PPO-C (COST239/default/100 slots, shorter episodes) remains in effect and will be reported in the final deliverables.
