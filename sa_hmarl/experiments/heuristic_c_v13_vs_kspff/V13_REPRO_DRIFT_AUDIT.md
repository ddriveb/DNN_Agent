# V1.3 Reproducibility Drift Audit

**Date:** 2026-07-07  
**Scope:** Explain why new Kmax sweep (v1.3 K=48 blocking ≈15.84 %) differs from old long-horizon benchmark (v1.3 blocking 7.68 %) on COST239.

---

## 1. Known Numbers

### Old long-horizon benchmark

- File: `sa_hmarl/experiments/long_horizon_benchmark/xlron_cost239_ptrnet_real_ecmax2.2_full.json`
- Method: `ppo_c+ksp_ff_k50_hops` vs `ppo_c+v12_k50_hops`
- Aggregate: KSP-FF 10.37 %, v1.3 7.68 %
- Seed 3030: KSP-FF 8.375 %, v1.3 6.14 %

### New Kmax sweep

- File: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_cost239_ptrnet_real_v13_K48.json`
- Method: `ppo_c+v12_k50_hops` only
- Aggregate: v1.3 K=48 15.84 %
- Seed 3030: v1.3 K=48 14.26 %
- Also observed: current KSP-FF seed 3030 ≈14.89 %

### Discrepancy

- v1.3 K=48 aggregate: 15.84 % vs 7.68 % (**+8.16 pp**)
- v1.3 seed 3030: 14.26 % vs 6.14 % (**+8.12 pp**)
- KSP-FF seed 3030: ~14.89 % vs 8.375 % (**+6.5 pp**)

The KSP-FF backend alone drifts by ~6.5 pp, so the issue is **not** the ranker or Kmax. It is a protocol/evaluator/code-state drift.

---

## 2. Drift Sources Identified So Far

### 2.1 `edge_cost_min` mismatch (confirmed, HIGH impact)

| Source | `edge_cost_min` | `edge_cost_max` |
|---|---:|---:|
| Old long-horizon benchmark config | **0.1** | 2.2 |
| New Kmax sweep config | **0.5** | 2.2 |

`generate_long_horizon_requests` draws total compute as:

```python
total_compute = rng.uniform(edge_cost_min + 0.5, edge_cost_max + 2.0)
```

So the old range is **[0.6, 4.2]** and the new range is **[1.0, 4.2]**. The new distribution has a higher floor and mean compute cost, which directly increases `server_overload` blocking. This alone can explain several percentage points of drift.

The new Kmax sweep script (`run_v13_kmax_sweep.sh`) did **not** pass `--edge_cost_min`, so it fell back to the current default of `0.5`. The old benchmark files record `0.1`, indicating the old default or an explicit override was `0.1`.

### 2.2 Code-state drift (confirmed)

`git status --short` for the active SA-HMARL evaluator/agents shows:

| File | Status | Notes |
|---|---|---|
| `sa_hmarl/sa_hmarl/evaluation/eval_long_horizon_system_comparison.py` | **untracked** | This evaluator is not in git; the old benchmark may have used a different version or this file was added later. |
| `sa_hmarl/sa_hmarl/agents/action_feature_builders.py` | **untracked** | New wrapper used by `ppo_agents.py`. It delegates to `AgentC`/`AgentR` static methods. |
| `sa_hmarl/sa_hmarl/agents/ppo_agents.py` | modified | Now imports from `action_feature_builders.py`; no semantic change to feature construction if `AgentC`/`AgentR` methods are unchanged. |
| `sa_hmarl/sa_hmarl/agents/c_agent.py` | modified | Removed legacy DQN policy; added `overload_aware` feature mode. `r_feasibility_safe` path appears unchanged except for the addition of the new branch. |
| `sa_hmarl/sa_hmarl/agents/r_agent.py` | modified | Removed legacy DQN policy; feature construction appears unchanged. |
| `sa_hmarl/sa_hmarl/evaluation/eval_c_post_decision_closed_loop.py` | modified | Added optional profile-dict storage; no change to core outcome recording. |
| `sa_hmarl/sa_hmarl/evaluation/eval_c_demand_potential_closed_loop.py` | modified | Added profile fields to metrics; no change to core trajectory logic. |

The `eval_long_horizon_system_comparison.py` file being **untracked** is the most worrying reproducibility issue: we cannot know whether the current file is identical to the one that produced the old benchmark.

### 2.3 Checkpoint hashes (recorded)

Current checkpoint SHA-256 values:

| Checkpoint | SHA-256 |
|---|---|
| `agent_c_cost239_r_feasibility_safe_last.pt` | `a5c9eb39bc33cf3b77a06d80e9c45315880dcb963ad9c3c95663eadaded9f2c8` |
| `agent_r_mixed.pt` | `e32e908196ac0c39898aab6e98516842c53f817150eeaba97e4ac172a61f8b94` |
| `r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt` | `9e9f06c766bdd8ea6c37400a5fed6cc3ed3c7593804183a58be13a65dfaaeb21` |

The old benchmark JSON does **not** contain checkpoint hashes, so we cannot prove these are the exact same files. They are the current files in the workspace.

### 2.4 Request-generation parameters (mostly match)

All other parameters match between old and new configs:

| Parameter | Old | New |
|---|---|---|
| topology | `xlron_cost239_ptrnet_real` | same |
| num_slots | 320 | same |
| num_servers | 4 | same |
| requests_per_episode | 10000 | same |
| warmup_requests | 2000 | same |
| seeds | 3030,4040,5050,6060,7070 | same |
| poisson_arrivals | true | same |
| exponential_holding | true | same |
| arrival_interval | 0.0625 | same |
| edge_cost_max | 2.2 | same |
| holding_min/max | 20.0/30.0 | same |
| size_min/max | 5.0/30.0 | same |
| deadline_min/max | 30.0/100.0 | same |
| block_sort_strategy | `start_asc` | same |
| path_sort_strategy | `hops` | same |
| ksp_ff_k50_hops_k_paths | 50 | same |
| ranker_candidate_mode | `legalctx48` | same |
| ranker_max_candidates | 48 | same |
| ranker_ensure_ksp | true | same |

The only request-generation difference is `edge_cost_min`.

---

## 3. Action Taken

### 3.1 Repro metadata added to evaluator

`eval_long_horizon_system_comparison.py` now writes a `metadata` block into every JSON output and a matching section in the markdown:

- Git commit + dirty status + dirty-file list
- Evaluator file SHA-256
- Agent-C / Agent-R / ranking checkpoint SHA-256
- Python / Torch / Numpy versions
- Ranker config (candidate_mode, max_candidates, ensure_ksp, etc.)

### 3.2 Paired current-code repro sweep launched

A new script `run_v13_kmax_paired_repro.sh` runs the **current code** with the **old protocol** (`--edge_cost_min 0.1`) for:

- `ppo_c+ksp_ff_k50_hops` (baseline)
- `ppo_c+v12_k50_hops` with K ∈ {16, 24, 32, 48}

All five configs share the same evaluator, seeds, checkpoint, and request-generation parameters. This will tell us whether the old 7.68 % / 10.37 % numbers are reproducible under current code once `edge_cost_min` is corrected.

Command running in background:

```bash
bash sa_hmarl/experiments/heuristic_c_v13_vs_kspff/run_v13_kmax_paired_repro.sh \
  xlron_cost239_ptrnet_real 0.0625 2.2 xlron_cost239_ptrnet_real
```

---

## 4. Hypotheses and Expected Outcomes

| Hypothesis | Evidence needed | Expected result |
|---|---|---|
| **H1: `edge_cost_min` drift is the main cause** | Paired repro with `edge_cost_min=0.1` matches old benchmark | If KSP-FF ≈10.37 % and v1.3 ≈7.68 %, H1 is confirmed. |
| **H2: untracked evaluator / code changes cause residual drift** | After fixing `edge_cost_min`, numbers still differ | Would require deeper diff of evaluator logic or rollback tests. |
| **H3: checkpoint files differ from old benchmark** | Cannot test without old hashes | Only possible if old files are archived elsewhere. |

The most parsimonious explanation is **H1**, because the KSP-FF backend (which has no ranker) already drifts by ~6.5 pp, and `edge_cost_min` directly controls the compute load that drives `server_overload`.

---

## 5. Current-Code Paired Results (Completed)

The paired repro completed successfully. All configs used `edge_cost_min=0.1` and the current workspace code.

| Method | Blocking mean ± std | Δ vs old (pp) | Delay mean/P95 | Decision mean/P95 |
|---|---:|---:|---:|---:|
| KSP-FF | 10.37 % ± 1.29 % | 0.00 | 8.80 / 16.78 | 6.74 / 9.97 |
| v1.3-K16 | 8.17 % ± 1.30 % | — | 9.26 / 16.38 | 9.94 / 14.23 |
| v1.3-K24 | 7.38 % ± 1.27 % | — | 9.38 / 16.66 | 9.82 / 13.92 |
| v1.3-K32 | 8.20 % ± 1.73 % | — | 9.50 / 16.96 | 9.93 / 14.10 |
| v1.3-K48 | 7.68 % ± 0.92 % | 0.00 | 9.54 / 17.27 | 9.72 / 13.59 |

- **KSP-FF matches the old benchmark exactly** (10.37 %).
- **v1.3 K=48 matches the old benchmark exactly** (7.68 %).
- **H1 is confirmed**: `edge_cost_min` drift was the sole cause of the 15.84 % figure.

---

## 6. Recommendations

1. **Do not compare the new Kmax table (15.84 % K=48) to the old main table (7.68 % v1.3)**. They were produced under different `edge_cost_min` protocols.
2. **Use the paired repro results** as the current-code baseline. If they match the old benchmark, then the old numbers are reproducible and the Kmax sweep should be re-run with `edge_cost_min=0.1`.
3. **Always pass `--edge_cost_min 0.1`** for any experiment intended to match the old long-horizon benchmark protocol.
4. **Archive checkpoint SHA-256 and git commit** in all future JSON outputs (now implemented).
5. **Commit `eval_long_horizon_system_comparison.py` and `action_feature_builders.py` to git** so future runs are reproducible.
6. The paired repro confirms the correct protocol is `edge_cost_min=0.1`; the 15.84 % figure was an artifact of `edge_cost_min=0.5`.
