# v1.3 Spectrum-Viability Label Report

> Date: 2026-06-28  
> Scenario: `snap24_gnutella_reach`, 24 slots, `default3`, AI=0.09, size_max=30 MB, holding_max=14 s  
> Eval: 5 seeds × 20 episodes × 80 requests (formal)

## What was changed

The v1.2 counterfactual R-ranker label is:

```
G_v12 = -3*cur_blocked - 4*future_blocked - 3*future_nsb - 0.03*delay - 0.05*avg_fs
        - 0.05*norm(path_km) - 0.05*norm(required_fs)
```

v1.3 adds a post-decision spectrum-viability potential:

```
Phi_after = alpha * log(1 + K_C_valid_after) / log(1 + num_c_actions)
          + beta  * log(1 + K_R_total_after) / log(1 + num_c_actions * max_r_per_c)

G_v13 = G_v12 + eta_phi * Phi_after
```

with `alpha=1.0`, `beta=0.3`, computed from the next-request Agent-C observation
(`agent_c_mask` and `feasible_counts`) on the post-decision branch env.

No new inference component was added; the deployed ranker is still a single
forward pass `argmax score`.

## Results

### Screening (3 seeds × 5 episodes)

| eta_phi | blocking (v1.3) |
|---:|---:|
| 0.1 | 56.58% |
| 0.3 | 56.58% |
| 0.5 | 56.58% |
| 1.0 | 56.50% |

All four coefficients are essentially tied in screening.

### Formal evaluation (5 seeds × 20 episodes)

| Method | Blocking | Raw empty | NSB | Overload | Delay mean | Delay p95 | Avg FS | Avg path km |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| PPO-R | 60.17% | 60.31% | 53.52% | 6.65% | 8.067 | 15.257 | 2.424 | 554.9 |
| KSP-BF | 61.79% | 61.94% | 56.25% | 5.54% | 7.806 | 14.204 | 2.746 | 543.7 |
| **v1.2 ranker** | **53.92%** | **54.26%** | **45.87%** | 8.05% | 8.360 | 16.781 | 2.041 | 565.3 |
| DeepRMSA | 54.94% | 55.24% | 47.49% | 7.45% | 8.405 | 16.235 | 2.062 | 595.7 |
| v1.3 eta=1.0 | 54.19% | 54.39% | 47.47% | **6.71%** | **8.278** | 16.125 | 2.051 | 567.8 |
| v1.3 eta=0.1 | 54.30% | 54.50% | 47.58% | **6.73%** | **8.306** | 16.272 | 2.052 | 567.2 |

### Delta vs v1.2

| eta_phi | Δ blocking | Δ raw empty | Δ NSB | Δ overload | Δ delay |
|---:|---:|---:|---:|---:|---:|
| 1.0 | +0.26 pp | +0.13 pp | +1.60 pp | -1.34 pp | -0.082 ms |
| 0.1 | +0.37 pp | +0.24 pp | +1.71 pp | -1.32 pp | -0.053 ms |

### Delta vs DeepRMSA

| eta_phi | Δ blocking vs DeepRMSA |
|---:|---:|
| 1.0 | -0.75 pp |
| 0.1 | -0.64 pp |

v1.2 still has the largest gap over DeepRMSA (-1.01 pp).

## Verdict

**FAIL** for all eta_phi values.

- v1.3 does **not** reduce blocking vs v1.2; it makes it slightly worse.
- The small gains are in overload and delay, not in the primary metric.
- v1.3 also shrinks the margin over DeepRMSA from -1.01 pp to about -0.7 pp.

## Diagnostic analysis

### 1. Does Phi_after distinguish candidates?

Yes. Across the training split:

- Phi_after mean = **1.006**, std = **0.351**
- Non-zero range rate = **91.8%**

So Phi_after has plenty of variance within candidate groups.

### 2. Is Phi_after aligned with the true H-step return?

**No.** The Spearman correlation between Phi_after and the v1.2 return on the
training split is **-0.061**. In other words, candidates that look like they
leave more feasible options for the next request are, if anything, slightly
associated with *worse* H-step returns.

This is the root cause of the failure: the viability bonus is not a reliable
proxy for long-term blocking reduction.

### 3. Is eta_phi too large?

Even `eta_phi=0.1` (which adds only ~0.1 to the return on average) degrades
blocking by +0.37 pp. So the problem is not just scale; the signal itself is
misaligned.

### 4. Does v1.2 already internalize downstream viability?

Yes, indirectly. The H-step rollout in v1.2 already penalizes future blocking,
future NSB, path length, and FS consumption. The learned ranker therefore
already accounts for how an R action affects future feasibility. Adding a
single-step feasibility-count proxy is redundant and, because it is only weakly
(and negatively) correlated with the true return, it perturbs the ranking in
the wrong direction.

## Answers to the requested questions

1. **Is v1.3 better than v1.2?**  
   No. v1.3 blocking is higher by +0.26 pp (eta=1.0) and +0.37 pp (eta=0.1).

2. **Does any improvement come from fewer raw_empty/NSB?**  
   No. raw_empty and NSB both increase slightly with v1.3. Overload and delay
   improve marginally, but those were not the target.

3. **Is it more effective than Lyapunov dynamic rerank?**  
   No. Lyapunov at least did not materially hurt v1.2 (best +0.06 pp). v1.3
   actively hurts v1.2.

4. **Should v1.3 replace v1.2 as the final method?**  
   No. Keep v1.2 as the final method.

## Recommendation

- **Stop the v1.3 viability-label line.**
- The post-decision feasibility-count proxy is not a useful addition to the
  v1.2 H-step return.
- Continue with v1.2 (`r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`)
  as the final ranker.
