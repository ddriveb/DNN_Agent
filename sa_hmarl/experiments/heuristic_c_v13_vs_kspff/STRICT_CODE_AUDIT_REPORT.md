# Strict Code Audit: SA-HMARL Experimental Correctness, Fairness, and Baseline Consistency

**Date:** 2026-07-05  
**Auditor:** Kimi Code CLI  
**Scope:** C-side heuristic baselines (`df_c`, `rf_c`, `df_fixed0_c`, `rf_fixed0_c`), Yin-style evaluator, PPO-C fixed v1.3 vs KSP-FF, R-side ranker implementation, C-R interface, and report/code consistency.

---

## 1. Executive Summary

This audit reviewed the SA-HMARL evaluation pipeline that produced the heuristic-C and PPO-C long-horizon comparisons. **No critical code-correctness bug was found** that invalidates the main conclusions. However, several **High** and **Medium** severity issues affect reproducibility, report clarity, and the risk of mis-citation:

- The unified main table (`FINAL_HEURISTIC_C_UNIFIED_REPORT.md`) is internally consistent and was produced under the declared v1.3 protocol.
- An older report (`FINAL_HEURISTIC_C_REPORT.md`) with different `df_c` v1.3 numbers is still present and could be cited by mistake.
- The training-script defaults that produced the stable PPO-C checkpoint still do **not** match the recommended mixed-stress validation regime documented in the synthesis report.
- Fixed-split ablation scripts rely on implicit defaults rather than explicit unified-protocol flags, creating a reproducibility hazard.
- The comparison of PPO-C fixed + v1.3 vs PPO-C fixed + KSP-FF is fair; the comparison of adapted `df_c`/`rf_c` to PPO-C is fair within the SA-HMARL action-space design but must not be sold as a reproduction of prior "server-first" DF/RF baselines.

**Overall verdict:** the reported unified main table is scientifically defensible, provided the taxonomy and caveats below are respected in the paper.

---

## 2. Audit Scope and Methodology

Files reviewed:

- `sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py`
- `sa_hmarl/sa_hmarl/evaluation/eval_yin_style_offloading_sweep.py`
- `sa_hmarl/sa_hmarl/agents/r_agent.py`
- `sa_hmarl/sa_hmarl/agents/r_frag_selector.py`
- `sa_hmarl/sa_hmarl/agents/r_ranker_policy.py`
- `sa_hmarl/sa_hmarl/training/utils.py`
- `sa_hmarl/sa_hmarl/training/train_agent_c_with_frozen_r.py`
- `sa_hmarl/sa_hmarl/evaluation/sweep_r_action_space.py`
- `sa_hmarl/sa_hmarl/evaluation/eval_c_closed_loop.py`
- `sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py`
- `sa_hmarl/sa_hmarl/evaluation/eval_c_demand_potential_closed_loop.py`
- `sa_hmarl/sa_hmarl/evaluation/eval_c_post_decision_closed_loop.py`
- `sa_hmarl/sa_hmarl/env/c_action_risk.py`
- `sa_hmarl/tests/test_offloading_baselines.py`
- `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/FINAL_HEURISTIC_C_UNIFIED_REPORT.md`
- `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/FINAL_HEURISTIC_C_REPORT.md`
- `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/FIXED_SPLIT_DF_RF_REPORT.md`
- `sa_hmarl/experiments/cside_step3_v13_system_synthesis.md`
- Run scripts: `run_df_unified_topology.sh`, `run_rf_unified_topology.sh`, `run_df_fixed0_topology.sh`, `run_rf_fixed0_topology.sh`

Methodology:

1. Read source code for each baseline/evaluator path.
2. Verified JSON outputs against report tables for `df_c`, `rf_c`, `df_fixed0_c`, and `rf_fixed0_c` on all four topologies.
3. Confirmed unit tests for fixed-split baselines pass.
4. Cross-checked command-line flags in run scripts against saved `config` blocks.
5. Compared synthesis-report recommendations to current training-script defaults.

---

## 3. Findings by Severity

### 3.1 High

#### H1. Superseded report still contains different numbers

- **Location:** `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/FINAL_HEURISTIC_C_REPORT.md`
- **Issue:** This report (2026-07-04) presents `df_c + v1.3` blocking of 10.85 % on JPN48 and 12.96 % on German17. The newer `FINAL_HEURISTIC_C_UNIFIED_REPORT.md` (2026-07-05) updates these to 11.06 % and 12.80 % because the earlier run did not explicitly set `ranker_candidate_mode=legalctx48`, `ranker_max_candidates=48`, and `ranker_ensure_ksp=true`. The older report is not marked as superseded.
- **Risk:** A reader or future author may cite the outdated table.
- **Mitigation:** Add a prominent supersession notice at the top of `FINAL_HEURISTIC_C_REPORT.md` pointing to the unified report, or move it to an `archive/` directory.

#### H2. Training-script defaults contradict the synthesis recommendation

- **Location:** `sa_hmarl/sa_hmarl/training/train_agent_c_with_frozen_r.py`, lines 1388–1393
- **Issue:** The synthesis report (`cside_step3_v13_system_synthesis.md`, section 8.1) explicitly recommends changing defaults to `--checkpoint_metric max_blocking` and `--validation_stress_levels standard,medium,heavy`. The current defaults remain `--checkpoint_metric mean_blocking` and `--validation_stress_levels ""`.
- **Risk:** Future C-side training runs will not reproduce the stable `r_feasibility_safe` checkpoint that underlies the long-horizon benchmark, unless the user manually passes the recommended flags.
- **Mitigation:** Update the argparse defaults. The synthesis report notes the change is backward-compatible because an empty `--validation_stress_levels` still falls back to single-stress behavior when explicitly supplied.

### 3.2 Medium

#### M1. Fixed-split run scripts rely on implicit defaults

- **Location:** `run_df_fixed0_topology.sh`, `run_rf_fixed0_topology.sh`
- **Issue:** These scripts do not pass `--block_sort_strategy start_asc`, `--path_sort_strategy hops`, or `--ksp_ff_k50_hops_k_paths 50`. They rely on the script defaults (`path_sort_strategy=hops`, `ksp_ff_k50_hops_k_paths=50`, but `block_sort_strategy=mixed`). The actual R observation still uses `start_asc` because `_r_backend_env_config` overrides it for `ksp_ff_k50_hops`/`v12_k50_hops`, but the saved JSON config records `block_sort_strategy="mixed"`, and the C-side observation is built with `mixed`.
- **Risk:** (a) Reproducibility hazard if defaults change. (b) Config file is misleading. (c) Although the fixed-split heuristics do not use block-sort-dependent features, any future reuse of these scripts for mask-sensitive baselines could silently deviate from the unified protocol.
- **Mitigation:** Add the explicit unified-protocol flags to both fixed0 scripts.

#### M2. Default `agent_c_checkpoint` is topology-mismatched

- **Location:** `eval_long_horizon_system_comparison.py`, line 768
- **Issue:** The default checkpoint is `agent_c_delayaware_v2_snap24_best.pt`. For imported-topology experiments the actual runs override this, but for heuristic-C modes the checkpoint is simply unused. If a user runs `ppo_c` on an imported topology without overriding, they will silently evaluate a snap24-trained policy on the wrong graph.
- **Risk:** Misleading config output and silent misuse.
- **Mitigation:** Add a runtime warning when `ppo_c` is used with the default checkpoint, or make the default `None` and require an explicit checkpoint for `ppo_c`.

#### M3. C-side observation uses k=5 while R backend may use k=50

- **Location:** `eval_long_horizon_system_comparison.py`, `_run_episode`
- **Issue:** The C observation (and therefore `agent_c_mask`) is built with `args.k_paths=5` for all C-side policies, including PPO-C. The R backend is then widened to k=50 for `ksp_ff_k50_hops`/`v12_k50_hops`. This is consistent across all C modes, but it means the C-side policy has a narrower view of R feasibility than the R backend it deploys.
- **Risk:** Could be perceived as handicapping C-side policies. In practice it is the deployment mode used during training and is applied equally to PPO-C and heuristics.
- **Mitigation:** Disclose this design choice explicitly in the paper methods section.

#### M4. v1.3 ranker candidate construction depends on PPO-R logits

- **Location:** `r_ranker_policy.py`, `_build_core_candidates`
- **Issue:** `legalctx48` candidates are seeded by PPO-R top-k logits and filled with highest-logit legal actions. If PPO-R has path/mod biases, v1.3 inherits them. The KSP-ensured fallback mitigates this but does not eliminate the dependence.
- **Risk:** v1.3's performance is not purely a learned planner-distillation signal; it is also a reranking of PPO-R's preferences.
- **Mitigation:** Acknowledge in the paper that v1.3 is a PPO-R-guided ranker, not an independent oracle.

#### M5. Version labels are inconsistent

- **Location:** `r_ranker_policy.py` module docstring; method name `v12_k50_hops` in `eval_long_horizon_system_comparison.py`
- **Issue:** The ranker file still says "final v1.2 R-ranker" although the implementation now supports v1.3 features (`legalctx48`, `ensure_ksp`). The R backend method is still called `v12_k50_hops` for backward compatibility.
- **Risk:** Reader confusion about which version is being evaluated.
- **Mitigation:** Update the docstring and add a code comment explaining the `v12` alias.

### 3.3 Low

#### L1. "Global best heuristic" claim is under-validated

- **Location:** `FINAL_HEURISTIC_C_REPORT.md`, section 2
- **Issue:** `df_c` was selected as the global heuristic based on performance on `snap24_gnutella_reach` and a COST239 smoke. German17, JPN48, and NSFNET were not used for heuristic selection.
- **Risk:** A per-topology oracle might choose differently.
- **Mitigation:** The unified report addresses this by presenting both `df_c` and `rf_c`; the limitation should be restated in the paper.

#### L2. Yin-style evaluator is not comparable to main table

- **Location:** `eval_yin_style_offloading_sweep.py`
- **Issue:** Yin-style DF/RF uses server-first + best-split logic, ignores `agent_c_mask`, and fixes the R backend to shortest-path/highest-mod/first-fit. Main-table DF/RF uses joint (split, server) selection, respects masks, and uses ksp_ff_k50_hops/v1.3 backends.
- **Risk:** The shared name "DF/RF" invites cross-evaluation comparison.
- **Mitigation:** Maintain the clear taxonomy used in `FIXED_SPLIT_DF_RF_REPORT.md` section 1; do not place Yin-style and main-table numbers in the same table.

#### L3. WO fallback to RF is surprising

- **Location:** `offloading_baselines.py`, `select_offloading_action`
- **Issue:** If `server_selected_count` is not provided to `select_wo`, it silently falls back to `select_rf`. This is documented but easy to miss.
- **Risk:** A caller expecting WO behavior may get RF behavior.
- **Mitigation:** No action required for current experiments; consider raising an error instead of falling back.

#### L4. v1.3 is guaranteed to contain the KSP-FF action

- **Location:** `r_ranker_policy.py`, `_ensure_ksp_action`
- **Issue:** Because `ensure_ksp=true` prepends the KSP-FF action, v1.3 can always select at least the KSP-FF solution. This guarantees v1.3 is never structurally disadvantaged relative to KSP-FF.
- **Risk:** Could be framed as an unfair advantage for v1.3.
- **Mitigation:** This is a deliberate fairness mechanism; state it clearly in the paper so reviewers understand why candidate-set restriction does not handicap v1.3.

---

## 4. Fairness Audit Verdict

| Comparison | Verdict | Rationale |
|---|---|---|
| PPO-C fixed + v1.3 vs PPO-C fixed + KSP-FF | **FAIR** | Same checkpoint, same request traces, same seeds, same R observation config (`k=50`, `hops`, `start_asc`). v1.3 includes the KSP-FF action in its candidate set. |
| df_c/rf_c vs PPO-C | **FAIR within SA-HMARL** | Same R backends, same protocol, same mask semantics. `df_c`/`rf_c` are adapted joint-action heuristics; they are not claimed to reproduce prior server-first literature. |
| df_fixed0_c/rf_fixed0_c vs df_c/rf_c | **FAIR ablation** | Identical code path; only split is pinned to 0. No fallback masks behavior is unit-tested. |
| Yin-style DF/RF vs main-table DF/RF | **NOT COMPARABLE** | Different R backend, different mask semantics, different evaluator. |
| v1.3 vs KSP-FF candidate space | **FAIR by construction** | KSP-FF searches all legal actions greedily; v1.3 scores a curated subset that always contains the KSP-FF action. |

**No evidence of hidden bias that systematically favors v1.3 over KSP-FF was found.** The observed topology-dependent gains/neutrality are consistent with the code behavior: v1.3 helps most when the C-side policy keeps servers out of deep saturation and the topology offers path diversity.

---

## 5. Baseline Taxonomy

| Baseline | Split choice | Server choice | Respects `agent_c_mask` | R backend | Native? / Notes |
|---|---|---|---|---|---|
| **PPO-C `r_feasibility_safe`** | learned | learned | yes | any | Learned SA-HMARL upper-bound reference. |
| **Adapted `df_c`** | distance-first over legal `(split, server)` pairs | same as split choice | yes | KSP-FF / v1.3 | SA-HMARL-adapted heuristic; not a reproduction of prior server-first DF. |
| **Adapted `rf_c`** | resource-first over legal `(split, server)` pairs | same as split choice | yes | KSP-FF / v1.3 | SA-HMARL-adapted heuristic; not a reproduction of prior server-first RF. |
| **Fixed-split `df_fixed0_c`** | pinned to `split0` | distance-first within split0 | yes, no fallback | KSP-FF / v1.3 | New no-partition-style lower-bound ablation. |
| **Fixed-split `rf_fixed0_c`** | pinned to `split0` | resource-first within split0 | yes, no fallback | KSP-FF / v1.3 | New no-partition-style lower-bound ablation. |
| **Yin-style DF/RF** | best split for chosen server | server-first (distance/resource) | **no** | SP-HM-FF only | Reference evaluator; not comparable to main table. |

**Key invariant:** all main-table baselines respect `agent_c_mask` and use the same R backend configuration. Yin-style is a separate experimental regime.

---

## 6. Final Recommendations

1. **Treat `FINAL_HEURISTIC_C_UNIFIED_REPORT.md` as the sole authoritative heuristic-C main table.** Mark `FINAL_HEURISTIC_C_REPORT.md` as superseded.
2. **Update `train_agent_c_with_frozen_r.py` defaults** to `--checkpoint_metric max_blocking` and `--validation_stress_levels standard,medium,heavy` so future training reproduces the stable C-side checkpoint.
3. **Make fixed-split run scripts explicit** by adding `--block_sort_strategy start_asc --path_sort_strategy hops --ksp_ff_k50_hops_k_paths 50`.
4. **Add a guard or warning** in `eval_long_horizon_system_comparison.py` when `ppo_c` is run with the default snap24 checkpoint.
5. **Disclose in the paper methods** that C-side observations use `k=5` while the KSP-FF/v1.3 R backend uses `k=50`.
6. **Clarify terminology:** use "adapted DF/RF" or "SA-HMARL DF/RF" for main-table heuristics, and reserve "Yin-style" for the separate evaluator.
7. **Update `r_ranker_policy.py` docstring** from "v1.2" to "v1.3" and add a comment explaining the `v12` alias.
8. **Do not increase the R candidate set beyond 48** without first demonstrating that `no_suitable_block` (not `server_overload`) is the bottleneck.

---

## 7. Action Items Completed During Audit

- Verified all five `test_offloading_baselines.py` unit tests pass.
- Verified unified `df_c`/`rf_c` JSON outputs match `FINAL_HEURISTIC_C_UNIFIED_REPORT.md` tables.
- Verified fixed-split JSON outputs match `FIXED_SPLIT_DF_RF_REPORT.md` tables and show 100 % `split0` share.
- Confirmed `ranker_candidate_mode=legalctx48`, `ranker_max_candidates=48`, and `ranker_ensure_ksp=true` are recorded in the unified config.
- Updated `run_df_fixed0_topology.sh` and `run_rf_fixed0_topology.sh` to explicitly pass `--block_sort_strategy start_asc --path_sort_strategy hops --ksp_ff_k50_hops_k_paths 50`.
- Updated `train_agent_c_with_frozen_r.py` defaults to `--checkpoint_metric max_blocking` and `--validation_stress_levels standard,medium,heavy`.
- Added a runtime warning in `eval_long_horizon_system_comparison.py` when `ppo_c` is run with the placeholder default checkpoint.
- Updated `r_ranker_policy.py` module docstring from v1.2 to v1.3.

## 8. Action Items Pending

- [ ] Mark `FINAL_HEURISTIC_C_REPORT.md` as superseded (retain file, add header notice).
- [ ] Update any paper-facing text that refers to "v1.2" ranker to "v1.3".
