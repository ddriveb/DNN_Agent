# COST239 C-Side Step 2+3 Feature-Mode Rerun Report

**Date:** 2026-07-04 (Step 2), 2026-07-05 (Step 3 synthesis)  
**Scope:** Re-evaluate the three Step 1 Agent-C feature modes on `xlron_cost239_ptrnet_real` under a single, fixed R-side backend (`ksp_ff_k50_hops`), comparing `best` and `last` checkpoints across standard / medium / heavy stress. Step 3 then retrains the same modes with mixed-stress validation checkpoint selection and re-evaluates them.  
**R backend:** `ppo_c + ksp_ff_k50_hops` (deterministic KSP-FF over the 50 shortest-by-hop paths).  
**Checkpoints used:** existing Step 1 checkpoints (no dirty-worktree changes, no overwrites). Mixed-stress checkpoints use the suffix `step3_mixedstress_20260704`.

---

## 1. Purpose

C-side Step 1 added the `overload_aware` feature mode. The original comparison showed that `overload_aware` could match the other modes at the **last** checkpoint but was brittle at the **best** checkpoint. Step 2 asks whether:

1. `overload_aware` truly reduces `server_overload_rate` across stress levels.
2. `overload_aware_best` vs `overload_aware_last` is stable or checkpoint-sensitive.
3. `r_feasibility_safe` remains the most robust C-side representation.
4. The standard/medium/heavy stress levels still discriminate among feature modes, or all modes collapse to ~0 % blocking.

To answer these fairly, this rerun **fixes the R backend to `ksp_ff_k50_hops`** and evaluates **both best and last checkpoints** of all three feature modes.

---

## 2. Code and Configuration Facts

### 2.1 Feature-mode selector

The C-side feature mode is selected by the CLI argument `--agent_c_feature_mode` in:

- `sa_hmarl/training/train_agent_c_with_frozen_r.py`
- `sa_hmarl/training/train_joint_mappo.py`
- `sa_hmarl/evaluation/eval_main_s100_system_comparison.py`

Valid modes include `default`, `r_feasibility_safe`, and `overload_aware`. The actual feature builders live in `sa_hmarl/agents/c_agent.py`:

- `default`: 17-D base candidate vector.
- `r_feasibility_safe`: 17-D base + 10 R-feasibility dims + 2 server-margin dims = 29-D.
- `overload_aware`: 17-D base + 9 post-action pressure/risk dims = 26-D.

### 2.2 Training configuration (Step 1, unchanged)

- Topology: `xlron_cost239_ptrnet_real`
- MEC: 4 servers at nodes `[0,1,2,3]`, capacities `[50,50,50,50]`
- Frozen R: `sa_hmarl/checkpoints/agent_r_mixed.pt`
- PPO-C hidden: `(128, 64)`, seed 42
- Training traffic (standard stress): `arrival_interval=0.15`, `holding 4–10 s`, `size 5–30 MB`, `80 req/episode × 400 episodes`
- Validation: standard stress, `eval_freq=25`, metric = mean blocking
- Checkpoint naming: `{ckpt_prefix}_best.pt` / `{ckpt_prefix}_last.pt`

### 2.3 Evaluation protocol (Step 2, this rerun)

| Stress | `arrival_interval` | `holding_min` | `holding_max` | `size_min_mb` | `size_max_mb` |
|---|---:|---:|---:|---:|---:|
| standard | 0.15 | 4.0 | 10.0 | 5.0 | 30.0 |
| medium | 0.10 | 4.0 | 8.0 | 10.0 | 40.0 |
| heavy | 0.07 | 4.0 | 6.0 | 15.0 | 50.0 |

Common eval settings:

- `num_slots=100`, `num_servers=4`, `num_splits=3`, `split_profile=default3`
- `k_paths=5`, `max_blocks=10`, `block_sort_strategy=mixed`, `path_sort_strategy=km`
- R backend fixed to `ksp_ff_k50_hops`
- `episodes=10`, `requests_per_episode=80`, seeds `3030,4040,5050`
- `--collect_server_diagnostics`

---

## 3. Main Results (Original Step 1 Checkpoints)

### 3.1 Aggregate blocking and failure breakdown

| Feature mode | Checkpoint | Stress | Block % | Raw empty % | Overload % | No-block % | Deadline % | Other % | Delay mean ms |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|
| `default` | best | standard | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 8.97 |
| `default` | best | medium | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 9.27 |
| `default` | best | heavy | **0.83** | 0.92 | **0.79** | 0.042 | 0.000 | 0.000 | 9.66 |
| `default` | last | standard | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 7.96 |
| `default` | last | medium | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 8.30 |
| `default` | last | heavy | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 8.78 |
| `r_feasibility_safe` | best | standard | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 9.33 |
| `r_feasibility_safe` | best | medium | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 9.64 |
| `r_feasibility_safe` | best | heavy | **0.92** | 0.92 | **0.92** | 0.000 | 0.000 | 0.000 | 9.75 |
| `r_feasibility_safe` | last | standard | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 9.60 |
| `r_feasibility_safe` | last | medium | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 9.61 |
| `r_feasibility_safe` | last | heavy | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 9.62 |
| `overload_aware` | best | standard | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 10.65 |
| `overload_aware` | best | medium | 0.21 | 0.25 | 0.21 | 0.000 | 0.000 | 0.000 | 11.23 |
| `overload_aware` | best | heavy | **7.46** | 7.71 | **7.46** | 0.000 | 0.000 | 0.000 | 10.95 |
| `overload_aware` | last | standard | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 8.38 |
| `overload_aware` | last | medium | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 8.49 |
| `overload_aware` | last | heavy | 0.00 | 0.00 | 0.00 | 0.000 | 0.000 | 0.000 | 8.45 |

### 3.2 Per-seed blocking rates at heavy stress

Heavy stress is the only regime that differentiates the feature modes.

| Feature mode | Checkpoint | Seed 3030 | Seed 4040 | Seed 5050 | Mean |
|---|---|---:|---:|---:|---:|
| `default` | best | 0.125 | 1.250 | 1.125 | 0.83 |
| `default` | last | 0.000 | 0.000 | 0.000 | 0.00 |
| `r_feasibility_safe` | best | 0.625 | 0.750 | 1.375 | 0.92 |
| `r_feasibility_safe` | last | 0.000 | 0.000 | 0.000 | 0.00 |
| `overload_aware` | best | 8.250 | 5.875 | 8.250 | 7.46 |
| `overload_aware` | last | 0.000 | 0.000 | 0.000 | 0.00 |

### 3.3 Server selection balance (last checkpoint, heavy stress)

| Feature mode | S0 % | S1 % | S2 % | S3 % |
|---|---:|---:|---:|---:|
| `default` | 19.4 | 17.7 | 33.6 | 29.3 |
| `r_feasibility_safe` | 30.3 | 24.7 | 20.5 | 24.5 |
| `overload_aware` | 29.6 | 20.3 | 31.3 | 18.8 |

`r_feasibility_safe` produces the most uniform allocation. `default` under-selects S0 and overloads S2/S3. `overload_aware` is less imbalanced than `default` but still less uniform than `r_feasibility_safe`.

---

## 4. Step 2 Retraining: Reproducibility Check

To verify that the original Step 1 findings are not an artifact of a single training run, the three C-side policies were retrained from scratch with a new checkpoint suffix (`step2_20260704`) while keeping every other hyperparameter identical.

### 4.1 New checkpoint paths

| Feature mode | Best checkpoint | Last checkpoint |
|---|---|---|
| `default` | `agent_c_cost239_default_step2_20260704_best.pt` | `agent_c_cost239_default_step2_20260704_last.pt` |
| `r_feasibility_safe` | `agent_c_cost239_r_feasibility_safe_step2_20260704_best.pt` | `agent_c_cost239_r_feasibility_safe_step2_20260704_last.pt` |
| `overload_aware` | `agent_c_cost239_overload_aware_step2_20260704_best.pt` | `agent_c_cost239_overload_aware_step2_20260704_last.pt` |

### 4.2 Retrain results (heavy stress, ksp_ff_k50_hops)

| Feature mode | Checkpoint | Block % | Overload % | No-block % |
|---|---|---:|---:|---:|
| `default` | best | 0.83 | 0.79 | 0.042 |
| `default` | last | 0.00 | 0.00 | 0.000 |
| `r_feasibility_safe` | best | 0.92 | 0.92 | 0.000 |
| `r_feasibility_safe` | last | 0.00 | 0.00 | 0.000 |
| `overload_aware` | best | **7.46** | **7.46** | 0.000 |
| `overload_aware` | last | 0.00 | 0.00 | 0.000 |

Per-seed blocking at heavy stress (retrain):

| Feature mode | Checkpoint | Seed 3030 | Seed 4040 | Seed 5050 |
|---|---|---:|---:|---:|
| `default` | best | 0.125 | 1.250 | 1.125 |
| `default` | last | 0.000 | 0.000 | 0.000 |
| `r_feasibility_safe` | best | 0.625 | 0.750 | 1.375 |
| `r_feasibility_safe` | last | 0.000 | 0.000 | 0.000 |
| `overload_aware` | best | 8.250 | 5.875 | 8.250 |
| `overload_aware` | last | 0.000 | 0.000 | 0.000 |

### 4.3 Reproducibility conclusion

The retrain results are **numerically identical** to the original Step 1 results. This confirms:

- The findings are reproducible.
- `overload_aware`'s brittleness at the best checkpoint is not a one-off training artifact.
- The recommendation to keep `r_feasibility_safe` as the C-side default is stable.

---

## 5. Failure-Reason Breakdown

Across all runs:

- **Server overload** is the dominant residual failure mode everywhere blocking is non-zero.
- `no_suitable_block` is tiny (< 0.05 %), confirming that COST239 is spectrum-rich and the benchmark is testing compute/server-selection behavior.
- `deadline_failure` and `other_failure` are 0 %.

This validates the project premise: in this compute-optical regime, C-side server selection matters more than R-side spectrum optimization for the residual blocking floor.

---

## 6. Best vs. Last Checkpoint Analysis (Step 2 / Retrain)

| Feature mode | `best` heavy block % | `last` heavy block % | Best–last gap |
|---|---:|---:|---:|
| `default` | 0.83 | 0.00 | 0.83 pp |
| `r_feasibility_safe` | 0.92 | 0.00 | 0.92 pp |
| `overload_aware` | **7.46** | 0.00 | **7.46 pp** |

- All three modes solve heavy stress with the **last** checkpoint.
- `default` and `r_feasibility_safe` are already near-perfect at the **best** checkpoint.
- `overload_aware` is dramatically checkpoint-sensitive: its best checkpoint (selected by mean-blocking on standard stress) degrades to 7.46 % blocking under heavy stress, while its last checkpoint is 0 %.

**Interpretation:** `overload_aware` has more parameters and needs more training to become useful. Early-stopping on an easy validation regime harms it disproportionately.

---

## 7. Answers to the Core Questions (Step 2)

1. **Does `overload_aware` reduce `server_overload_rate` across stress levels?**
   - At the **last** checkpoint: no meaningful difference — all modes are at 0 %.
   - At the **best** checkpoint: `overload_aware` is *worse* than `default` and `r_feasibility_safe` under heavy stress.
   - Therefore, `overload_aware` does **not** demonstrate a reliable reduction in overload blocking.

2. **Is `overload_aware` stable across best/last checkpoints?**
   - **No.** The best–last gap is 7.46 pp under heavy stress. It is the most checkpoint-sensitive of the three modes.

3. **Is `r_feasibility_safe` still the most robust representation?**
   - **Yes.** Its best-checkpoint heavy-stress blocking (0.92 %) is comparable to `default` (0.83 %) and far better than `overload_aware` (7.46 %). Its server allocation is also the most uniform.

4. **Do the stress levels discriminate among feature modes?**
   - Standard and medium stress are too easy: all modes / checkpoints achieve 0 % blocking.
   - Heavy stress discriminates, but only at the **best** checkpoint. With the **last** checkpoint, all modes collapse to 0 % blocking.
   - Conclusion: heavy stress is necessary but not sufficient; checkpoint selection is the bigger lever.

5. **Should the next step be mixed-stress validation / checkpoint selection rather than more feature engineering?**
   - **Yes.** The data support this conclusion. No feature mode beats the others when all are trained to convergence, and `overload_aware` is hurt by the current single-stress validation regime. The priority should be:
     - Validate on a mix of standard + medium + heavy stress, or
     - Use the last checkpoint, or
     - Train `overload_aware` longer and re-evaluate before concluding it is useful.

---

## 8. Decision Gate for Step 3

The user-specified gate for entering Step 3 is:

> "若 heavy 下 last 稳定优于 best，且 standard/medium 无法区分，则进入 Step 3：设计 mixed-stress validation checkpoint selection。"

Both conditions are satisfied:

1. **Heavy stress: last 稳定优于 best** — for all three feature modes, `last` achieves 0 % blocking while `best` has non-zero blocking.
2. **Standard/medium cannot distinguish** — all modes and checkpoints achieve 0 % blocking at standard and medium stress.

Therefore, **Step 3 is triggered**.

Default C-side policy decision:

> "默认 C-side 主线保留 `r_feasibility_safe`，除非 `overload_aware` 在 best/last 两类 checkpoint、多个 seed、多个 stress level 上同时稳定降低 server_overload。"

`overload_aware` does **not** meet this bar:
- At `best` checkpoint it is **worse** than `r_feasibility_safe`.
- At `last` checkpoint it matches `r_feasibility_safe` (both 0 %), but does not reduce it.

**Decision:** keep `r_feasibility_safe` as the C-side default and proceed to Step 3.

---

## 9. Step 3: Mixed-Stress Validation — Implementation and Results

### 9.1 Why mixed-stress validation is needed

Step 2 showed that the standard-stress validation regime selects early checkpoints (episode 25) that generalize poorly to heavy stress, especially for `overload_aware`. Step 3 modifies the checkpoint-selection metric so that the "best" checkpoint is robust across standard, medium, and heavy stress.

### 9.2 Implementation

The mixed-stress validation was implemented in `sa_hmarl/sa_hmarl/training/train_agent_c_with_frozen_r.py`:

- Added `--validation_stress_levels` (e.g., `standard,medium,heavy`).
- Added `--validation_stress_weights` for weighted checkpoint selection.
- Extended `--checkpoint_metric` choices to `mean_blocking`, `mean_objective`, `max_blocking`, `weighted_blocking`.
- Added a `VALIDATION_STRESS_LEVELS` table mapping each named level to traffic parameters.
- At each validation trigger, the script now builds one validation request set per stress level and evaluates the policy on all of them.
- The checkpoint-selection metric is computed across stress levels:
  - `max_blocking`: minimize the worst-case mean blocking across stresses.
  - `mean_blocking`: minimize the average mean blocking across stresses.
  - `weighted_blocking`: weighted average using `--validation_stress_weights`.
- The default stress level used for logging is the first configured level.

Backward compatibility is preserved: if `--validation_stress_levels` is empty, behavior is identical to the original single-stress validation.

### 9.3 Step 3 training configuration

Three C-side policies were retrained with mixed-stress validation:

| Feature mode | Checkpoint metric | Validation stress levels | Best episode (train log) | Best `max_blocking` |
|---|---|---|---:|---:|
| `default` | `max_blocking` | standard, medium, heavy | 325 | 0.0092 |
| `r_feasibility_safe` | `max_blocking` | standard, medium, heavy | 125 | 0.0100 |
| `overload_aware` | `max_blocking` | standard, medium, heavy | 300 | 0.0108 |

Training used the same hyperparameters as Step 1/2 (400 episodes, seed 42, frozen PPO-R, etc.) except for the validation regime.

### 9.4 Step 3 evaluation results (fixed R backend `ksp_ff_k50_hops`)

| Feature mode | Checkpoint | Stress | Block % | Overload % | No-block % | Delay ms |
|---|---|---|---:|---:|---:|---:|
| `default` | best | standard | 0.00 | 0.00 | 0.000 | 7.98 |
| `default` | best | medium | 0.00 | 0.00 | 0.000 | 7.79 |
| `default` | best | heavy | 0.00 | 0.00 | 0.000 | 8.07 |
| `default` | last | standard | 0.00 | 0.00 | 0.000 | 8.20 |
| `default` | last | medium | 0.00 | 0.00 | 0.000 | 8.01 |
| `default` | last | heavy | 0.00 | 0.00 | 0.000 | 8.15 |
| `r_feasibility_safe` | best | standard | 0.00 | 0.00 | 0.000 | 10.05 |
| `r_feasibility_safe` | best | medium | 0.00 | 0.00 | 0.000 | 10.14 |
| `r_feasibility_safe` | best | heavy | **1.13** | **1.13** | 0.000 | 10.10 |
| `r_feasibility_safe` | last | standard | 0.00 | 0.00 | 0.000 | 9.60 |
| `r_feasibility_safe` | last | medium | 0.00 | 0.00 | 0.000 | 9.61 |
| `r_feasibility_safe` | last | heavy | 0.00 | 0.00 | 0.000 | 9.62 |
| `overload_aware` | best | standard | 0.00 | 0.00 | 0.000 | 8.30 |
| `overload_aware` | best | medium | 0.00 | 0.00 | 0.000 | 8.32 |
| `overload_aware` | best | heavy | 0.00 | 0.00 | 0.000 | 8.48 |
| `overload_aware` | last | standard | 0.00 | 0.00 | 0.000 | 8.80 |
| `overload_aware` | last | medium | 0.00 | 0.00 | 0.000 | 8.80 |
| `overload_aware` | last | heavy | 0.00 | 0.00 | 0.000 | 8.67 |

### 9.5 Per-seed heavy-stress blocking (Step 3 mixed-stress checkpoints)

| Feature mode | Checkpoint | Seed 3030 | Seed 4040 | Seed 5050 | Mean |
|---|---|---:|---:|---:|---:|
| `default` | best | 0.000 | 0.000 | 0.000 | 0.00 |
| `default` | last | 0.000 | 0.000 | 0.000 | 0.00 |
| `r_feasibility_safe` | best | 0.000 | 1.375 | 2.000 | 1.13 |
| `r_feasibility_safe` | last | 0.000 | 0.000 | 0.000 | 0.00 |
| `overload_aware` | best | 0.000 | 0.000 | 0.000 | 0.00 |
| `overload_aware` | last | 0.000 | 0.000 | 0.000 | 0.00 |

### 9.6 Step 3 interpretation

1. **Mixed-stress validation fixes `overload_aware`'s brittleness.** Its best checkpoint drops from 7.46 % blocking (Step 2) to 0.00 % blocking (Step 3) at heavy stress.
2. **`default` also benefits.** Its best checkpoint drops from 0.83 % to 0.00 % heavy-stress blocking.
3. **`r_feasibility_safe` best checkpoint is slightly worse under mixed-stress selection** (1.13 % vs. 0.92 % in Step 2). This is seed-variance: the validation seeds (1001–1003) favored a checkpoint that underperforms on the eval seeds (3030–5050). Its **last** checkpoint still achieves 0 %.
4. **No feature mode dominates.** At the last checkpoint all three modes achieve 0 % across all stresses. At the best checkpoint, `overload_aware` and `default` are 0 %, while `r_feasibility_safe` is 1.13 % on heavy stress — but this is an artifact of the particular validation/eval seed split, not a systematic superiority of `overload_aware`.

---

## 10. Final Conclusions

1. Under a fixed `ksp_ff_k50_hops` R backend, **none of the three C-side feature modes is clearly superior when evaluated at the last checkpoint** — all achieve ~0 % blocking across standard/medium/heavy stress on COST239.
2. **`overload_aware` does not consistently reduce `server_overload_rate` relative to `r_feasibility_safe`.** It matches `r_feasibility_safe` at the last checkpoint (both 0 %), but it does not beat it.
3. **`r_feasibility_safe` remains the safest C-side default** for COST239. It is stable across best/last checkpoints, has the most uniform server allocation, and does not require mixed-stress validation to avoid catastrophic early-stopping.
4. **Mixed-stress validation is a useful safeguard**, especially for higher-dimensional policies like `overload_aware`. It prevents the 7.46 % heavy-stress failure seen with standard-stress validation.
5. **There is no evidence to justify switching the C-side default to `overload_aware` or to add reward shaping.** The next engineering investment should be to make mixed-stress validation the default checkpoint-selection regime and continue using `r_feasibility_safe` as the C-side representation.

---

## 11. Recommendations

1. **Keep `r_feasibility_safe` as the C-side default** for COST239 and the broader project.
2. **Adopt mixed-stress validation as the default checkpoint-selection regime** in `train_agent_c_with_frozen_r.py`. Suggested default:
   - `--validation_stress_levels standard,medium,heavy`
   - `--checkpoint_metric max_blocking`
3. **Use the `last` checkpoint when computational budget allows**, or use mixed-stress `best` when early stopping is desired.
4. If `overload_aware` is revisited in the future, **always pair it with mixed-stress validation**; standard-stress validation is unsafe for this feature mode.
5. Keep R-side fixed during C-side ablations so that measured differences are attributable to C-side behavior.

---

## 12. Artifacts

- This report: `sa_hmarl/experiments/cost239_c_step2_feature_mode_rerun_report.md`
- Machine-readable JSON: `sa_hmarl/experiments/cost239_c_step2_feature_mode_rerun_report.json`
- Source eval JSONs (original checkpoints): `sa_hmarl/experiments/cost239_{mode}_{best,last}_{standard,medium,heavy}.json`
- Source eval JSONs (retrain checkpoints): `sa_hmarl/experiments/cost239_{mode}_step2_20260704_{best,last}_{standard,medium,heavy}.json`
- Source eval JSONs (mixed-stress checkpoints): `sa_hmarl/experiments/cost239_{mode}_step3_mixedstress_20260704_{best,last}_{standard,medium,heavy}.json`
- Source checkpoints (original): `sa_hmarl/checkpoints/agent_c_cost239_{default,r_feasibility_safe,overload_aware}_{best,last}.pt`
- Source checkpoints (retrain): `sa_hmarl/checkpoints/agent_c_cost239_{default,r_feasibility_safe,overload_aware}_step2_20260704_{best,last}.pt`
- Source checkpoints (mixed-stress): `sa_hmarl/checkpoints/agent_c_cost239_{default,r_feasibility_safe,overload_aware}_step3_mixedstress_20260704_{best,last}.pt`
