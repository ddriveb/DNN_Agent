# SA-HMARL v1.3 Metric Correction

> **Status:** Corrected metric is now implemented, unit-tested, and used for all v1.3 gates.  
> **Date:** 2026-07-08  
> **Supersedes:** Any prior report that uses `relative_regret_reduction` as the primary offline gate.

---

## 1. What was wrong

The original offline gate used a metric called `relative_regret_reduction`:

```
relative_regret_reduction = mean over groups where PPO regret > 0 of
                            (ppo_regret - model_regret) / ppo_regret
```

This metric has two problems:

1. **Conditioning on PPO regret > 0** throws away groups where PPO is already optimal, which exaggerates variance and can reverse sign when the model is better on easy groups.
2. **Ratio inside the mean** is unbounded and unstable. When `ppo_regret` is tiny (e.g. `1e-4`) and `model_regret` is slightly larger (e.g. `0.15`), the ratio can become **−1500**, even though the absolute harm is small. Averaging a handful of such outliers can dominate the entire metric and make a genuinely improving model look catastrophic.

### Example from the old full-medium run

| Seed | Model regret | PPO regret | Old conditional ratio | New aggregate reduction |
|---:|---:|---:|---:|---:|
| 42 | 0.0682 | 0.1050 | **−1.154 ± 0.134** | **35.04%** |

The old metric classified the trained ranker as *much worse* than PPO; the corrected metric shows it is *substantially better*.

---

## 2. Corrected metric

All gates now use the stable aggregate regret metrics:

```
absolute_regret_improvement = mean_ppo_regret - mean_model_regret
aggregate_regret_reduction  = absolute_regret_improvement / max(mean_ppo_regret, 1e-12)
```

Properties:

* Bounded: if model regret is non-negative, `aggregate_regret_reduction` is in `[−∞, 1]`, but in practice it stays well behaved because the denominator is the **mean** PPO regret, not a per-group tiny value.
* No conditioning: uses **all** groups.
* Monotonic in model quality for a fixed PPO baseline.
* Directly interpretable: 35% aggregate reduction means the ranker reduces average per-group regret by 35% relative to PPO-R.

### Additional diagnostics kept

The training/evaluation code also reports:

* `model_better_than_ppo_rate` / `model_worse_than_ppo_rate` / `model_equal_to_ppo_rate`
* Regret-delta percentiles (p10 / p50 / p90)
* PPO-regret buckets (low/medium/high regret groups)
* `oracle_tie_hit_rate` and `ppo_oracle_tie_hit_rate`
* The old conditional metric is preserved under the name `conditional_mean_relative_reduction_on_ppo_error_groups` for backward compatibility, but it is **not used for gating**.

---

## 3. Code changes

* `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py`
  * Replaced the conditional ratio with `_compute_group_metrics` returning `absolute_regret_improvement`, `aggregate_regret_reduction`, rescue/harm/tie rates, percentiles, buckets, and tie-hit rates.
  * `_selection_score(metric_name="regret")` now selects the checkpoint with the **lowest validation model regret** (highest `−model_regret`).
* `sa_hmarl/tests/test_regret_metrics.py`
  * 4 unit tests covering positive reduction when model is better, stability with tiny PPO regret, rate summation, and rescue-rate correctness.

---

## 4. Unit-test result

```bash
PYTHONPATH=sa_hmarl python3 -m pytest sa_hmarl/tests/test_regret_metrics.py -v
```

```
sa_hmarl/tests/test_regret_metrics.py::test_aggregate_regret_reduction_positive_when_model_better PASSED
sa_hmarl/tests/test_regret_metrics.py::test_aggregate_regret_reduction_does_not_explode_with_small_ppo_regret PASSED
sa_hmarl/tests/test_regret_metrics.py::test_better_equal_worse_rates_sum_to_one PASSED
sa_hmarl/tests/test_regret_metrics.py::test_model_better_rate_reflects_actual_rescues PASSED

4 passed in ~18 s
```

---

## 5. Impact on decisions

| Decision item | Before correction (old metric) | After correction |
|---|---|---|
| Full-medium offline gate | FAIL (`relative_regret_reduction = −1.15`) | **PASS** (`aggregate_regret_reduction = 35.04%`) |
| Checkpoint selection | Selected by Top-1 accuracy | Selected by **lowest validation model regret** |
| Closed-loop launch | Blocked | **Launched** (5 seeds × 6000 requests) |
| v1.35 launch | Not allowed | **Pilot allowed**, full launch still gated |

---

## 6. Recommendations for future reporting

1. Always report `aggregate_regret_reduction` and `absolute_regret_improvement` together.
2. Report rescue/harm/tie rates and regret-delta percentiles to diagnose *where* the improvement comes from.
3. Do not use `conditional_mean_relative_reduction_on_ppo_error_groups` as a primary gate; keep it only as a diagnostic.
4. When sample size is small (e.g. 5 seeds), pair aggregate metrics with bootstrap CIs and permutation/Wilcoxon tests before making launch decisions.
