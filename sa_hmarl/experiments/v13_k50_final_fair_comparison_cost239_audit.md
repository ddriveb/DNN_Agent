# v1.3 K=50 Final Fair Comparison — Code & Fairness Audit

## 1. Code Confirmation

### New r_mode added

- `v13_k50_hops`: v1.3 post-decision ranker with K=50, `path_sort_strategy=hops`, `block_sort_strategy=start_asc`.
- `v13_k5_hops`: v1.3 post-decision ranker with K=5 (diagnostic boundary).

### Files modified

- `sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py`
  - `_r_backend_env_config`: added `v13_k50_hops` to K50 group, `v13_k5_hops` to K5 group.
  - `_select_r_action_idx`: both modes route to `rank_policy.select_action(...)`.
  - `_run_episode`: profile collection includes `v13_k50_hops`.
  - `evaluate`: aggregate `mod_dist` and `server_dist` across seeds.
- `sa_hmarl/sa_hmarl/evaluation/eval_main_s100_system_comparison.py`: same r_mode mappings.
- `sa_hmarl/sa_hmarl/evaluation/eval_c_demand_potential_closed_loop.py`: extended `PerMethodMetrics` and `_aggregate_metrics` with hop count, block start/size, mod/server distributions.
- `sa_hmarl/sa_hmarl/evaluation/eval_c_post_decision_closed_loop.py`: `_record_outcome` now records the extra fields.

### Name mapping: code name → paper name

| Code r_mode | Paper description |
|---|---|
| `v13_k50_hops` | v1.3 post-decision ranker, K=50 hops |
| `v13_k5_hops` | v1.3 post-decision ranker, K=5 hops (diagnostic) |
| `ksp_ff_plain_k50_hops` | Plain First-Fit KSP-FF, K=50 |
| `ksp_ff_plain_k5_hops` | Plain First-Fit KSP-FF, K=5 |
| `ksp_ff_highest_mod_k50_hops` | Highest-modulation First-Fit KSP-FF, K=50 |
| `deep_rmsa_style_k50_hops` | Hand-crafted DeepRMSA-style proxy, K=50 |

> Note: `v12_k50_hops` is preserved for backward compatibility; the paper uses the explicit `v13_k50_hops` label.

### Checkpoint used for v1.3

- Ranking checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt`
- Candidate mode (from checkpoint): `checkpoint default (ppo_r_topk_only)`
- Max candidates (from checkpoint): `checkpoint default (30)`
- Ensure KSP action: `False`

## 2. Fairness Checklist

| Check | Status | Evidence |
|---|---|---|
| All methods use same PPO-C policy | ✅ | `--agent_c_checkpoint agent_c_cost239_r_feasibility_safe_last.pt` for all |
| All methods use same traffic seeds | ✅ | seeds=3030,4040,5050,6060,7070 for all |
| C-side observation is built before R-side K override | ✅ | `eval_long_horizon_system_comparison.py` sets `env.k=args.k_paths` before `build_agent_c_observation`, then overrides after C selection |
| `v13_k50_hops` vs `v13_k5_hops` differ only in R-side K | ✅ | Both route to same ranker; `_r_backend_env_config` returns K=50 vs K=5 |
| `ksp_ff_plain_k50_hops` vs `ksp_ff_plain_k5_hops` differ only in K | ✅ | Same `ksp_ff_action` selector |
| `ksp_ff_highest_mod_k50_hops` vs `ksp_ff_highest_mod_k5_hops` differ only in K | ✅ | Same `ksp_ff_highest_mod_action` selector |
| `deep_rmsa_style_k50_hops` is labeled as proxy, not neural DeepRMSA | ✅ | Method table and report text explicitly state this |

## 3. Results Summary

| Method | K | Selector / Ranker | Blocking mean ± std | Overload | NSB | raw_empty | Delay mean/P95 | Decision mean/P95 |
|---|---|---|---:|---:|---:|---:|---:|---:|
| ppo_c+v13_k50_hops | 50 | v1.3 post-decision ranker (postdec_topk_k50_s100) | 6.29% ± 1.08% | 6.29% | 0.00% | 8.34% | 9.118/16.899 | 12.483/18.191 |
| ppo_c+ksp_ff_plain_k50_hops | 50 | plain KSP-FF | 7.92% ± 0.57% | 7.92% | 0.00% | 10.34% | 8.964/16.562 | 8.534/12.773 |
| ppo_c+ksp_ff_highest_mod_k50_hops | 50 | highest-modulation KSP-FF | 9.93% ± 1.42% | 9.93% | 0.00% | 12.97% | 9.086/17.171 | 7.312/11.530 |
| ppo_c+deep_rmsa_style_k50_hops | 50 | hand-crafted DeepRMSA-style proxy | 9.62% ± 1.34% | 9.62% | 0.00% | 12.52% | 10.519/19.161 | 7.581/11.966 |
| ppo_c+v13_k5_hops | 5 | v1.3 post-decision ranker | 7.38% ± 1.10% | 7.38% | 0.00% | 9.70% | 9.246/16.638 | 6.034/8.947 |
| ppo_c+ksp_ff_plain_k5_hops | 5 | plain KSP-FF | 7.46% ± 0.94% | 7.46% | 0.00% | 9.69% | 8.989/16.470 | 4.944/7.685 |

### Pairwise Fairness Comparison

| Pair | Same K? | Same C? | Main fairness? | Δ Blocking (pp) | Interpretation |
|---|---|---|---:|---:|---|
| v13_k50_hops vs ksp_ff_plain_k50_hops | true | yes (all ppo_c) | main | -1.63 | ranker vs plain FF |
| v13_k50_hops vs ksp_ff_highest_mod_k50_hops | true | yes (all ppo_c) | main | -3.64 | ranker vs optical heuristic |
| v13_k50_hops vs deep_rmsa_style_k50_hops | true | yes (all ppo_c) | auxiliary | -3.33 | ranker vs DeepRMSA-style proxy |
| v13_k50_hops vs v13_k5_hops | false | yes (all ppo_c) | diagnostic | -1.10 | candidate horizon sensitivity |
| v13_k50_hops vs ksp_ff_plain_k5_hops | false | yes (all ppo_c) | boundary only | -1.17 | not primary fairness claim |

## 4. Interpretation Rules Applied

1. `v13 K50` is better than both plain and highest-mod KSP-FF K50 → v1.3 outperforms traditional local heuristics under matched path support.
2. `v13 K50` is better than DeepRMSA-style K50 proxy → ranker is at least not weaker than the hand-crafted proxy (not a neural DeepRMSA checkpoint comparison).
3. `v13 K5` is slightly worse than `v13 K50` → ranker checkpoint/candidate distribution is tuned for K50; K5 is a diagnostic horizon-sensitivity case.
4. All blocking is `server_overload` → COST239 is compute-overload dominated; R-side gains are bounded by C/server bottleneck.
5. Primary paper claim is `v13 K50` vs KSP-FF K50 variants, not vs K5 baselines.

## 5. What to Cite in the Paper

- Main table: v13 K50 vs plain/highest-mod KSP-FF K50 (Table 1 in main report).
- Auxiliary: v13 K50 vs DeepRMSA-style proxy, with explicit proxy disclaimer.
- Appendix: v13 K50 vs v13 K5 / plain FF K5 horizon sensitivity.

## 6. English Paragraph for the Paper

We evaluate the proposed v1.3 post-decision ranker under matched K=50 hop-ordered path support. To avoid conflating path-horizon effects with selector effects, KSP-FF is separated into plain first-fit and highest-modulation first-fit variants. The primary comparison is therefore v1.3 K50 against KSP-FF variants using the same K50 path support and the same PPO-C policy. K5 results are reported only as diagnostic boundary cases, because they change the R-side candidate support and are not the primary fairness claim.