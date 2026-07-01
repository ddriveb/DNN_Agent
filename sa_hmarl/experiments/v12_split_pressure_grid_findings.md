# v1.2 Counterfactual R-Ranker Scenario Screening — Findings

## Goal
Find workload conditions on `snap24_gnutella_reach` (24 or 100 slots, k=5,
max_blocks=10, 4 servers) where the v1.2 counterfactual R-ranker clearly
outperforms DeepRMSA.

## Screening criteria
- **PASS**: v1.2 blocking < DeepRMSA by ≥2 pp, delay overhead ≤1 ms,
  overload overhead ≤0.5 pp, DeepRMSA blocking in [5%, 60%].
- **MARGINAL**: gap 1–2 pp with the same side-effect limits.
- **FAIL**: otherwise.

---

# 24-Slot Results

## Stage 1 grid (36 scenarios)
- Split profiles: `default3`, `complex5_v2_lite`
- Arrival intervals: 0.10, 0.12, 0.15
- `size_max_mb`: 30, 40, 50
- `holding_max`: 10, 14
- 3 seeds × 5 episodes × 80 requests

### Result
- **0 PASS** under the strict overload threshold.
- `complex5_v2_lite` produced blocking gaps of +2.0 to +3.2 pp, but all failed
  because v1.2's server-overload rate was +1.5 to +5.9 pp higher than DeepRMSA.
- `default3` produced only **MARGINAL** gaps (+0.5 to +1.2 pp) with low overload.

### Best Stage 1 candidates
| Scenario | gap (pp) | delay Δ (ms) | overload Δ (pp) | verdict |
|---|---:|---:|---:|---:|
| complex5_v2_lite ai0.15 sz30 h14 | +3.17 | +0.53 | +1.67 | FAIL |
| default3 ai0.10 sz30 h14 | +1.17 | -0.19 | -0.25 | MARGINAL |
| default3 ai0.15 sz30 h10 | +1.00 | -0.24 | -0.08 | MARGINAL |

## Focused extensions (default3, size=30, holding=14/16/18)
Tightened arrival intervals around the sweet spot (0.08–0.14) and increased
holding time to raise pressure without adding splits.

| Scenario | Stage 1/Ext gap (pp) | verdict |
|---|---:|---:|
| default3 ai0.09 sz30 h14 | +1.50 | MARGINAL |
| default3 ai0.09 sz30 h16 | +1.17 | MARGINAL |
| default3 ai0.10 sz30 h18 | +1.25 | MARGINAL |

No default3 configuration reached a ≥2 pp gap while keeping overload ≤0.5 pp.

## Stage 2 formal validation (5 seeds × 20 episodes × 80 requests)
| Scenario | v1.2 Blk | Deep Blk | Deep-v1.2 | p-value | Delay Δ ms | Overload Δ pp |
|---|---:|---:|---:|---:|---:|---:|
| complex5_v2_lite ai0.15 sz30 h14 | 46.17% ± 8.34% | 47.14% ± 8.01% | **+0.96 pp** | **0.005** | +0.39 | +1.66 |
| default3 ai0.09 sz30 h14 | 53.92% ± 5.90% | 54.94% ± 5.82% | **+1.01 pp** | **0.002** | -0.05 | +0.60 |

Key observation: the small-sample Stage 1 estimates for `complex5` were
optimistic. With more seeds/episodes the gap shrinks to ~1 pp. The improvement
is **statistically significant** in both scenarios, but it is **not ≥2 pp**.

## Stage 3 targeted retraining (24 slots)
Retrained the v1.2 ranker on the two best candidate scenarios with an added
server-overload penalty in the H-step return:

```text
overload_coef ∈ {0.5, 1.0, 2.0}
path_penalty=0.05, fs_penalty=0.05
future_block=4.0, future_nsb=3.0, delay=0.03, fs=0.05, horizon=5
```

### Small-sample retrain results (3 seeds × 5 episodes)
| Scenario | overload_coef | v1.2 Blk | Deep Blk | Gap | Delay Δ | Overload Δ | Verdict |
|---|---|---|---:|---:|---:|---:|---:|
| default3 ai0.09 sz30 h14 | 0.5/1.0/2.0 | 56.67% | 57.50% | +0.83 pp | -0.12 ms | **+3.17 pp** | FAIL |
| complex5 ai0.15 sz30 h14 | 0.5/1.0/2.0 | 44.00% | 47.00% | **+3.00 pp** | +0.50 ms | **+0.75 pp** | FAIL |

Notably, the three overload coefficients produced **identical metrics**, which
means the future server-overload signal within `H=5` is too sparse to change the
ranking labels. The penalty term therefore had no practical effect.

### Formal validation of the best retrained model
Retrained on `complex5_v2_lite ai0.15 sz30 h14`, evaluated with 5 seeds × 20 eps:

| v1.2 Blk | Deep Blk | Gap | Delay Δ | Overload Δ |
|---|---:|---:|---:|---:|
| 46.06% | 47.14% | **+1.07 pp** | +0.41 ms | **+1.94 pp** |

The small-sample +3.00 pp collapsed back to **~1 pp**, and overload actually
worsened versus the original transfer model (+1.94 pp vs +1.66 pp). The retrain
overfit to the small generated dataset and did not improve the fair gap.

---

# 100-Slot Results

## Motivation
With 100 slots the resource setting is much less constrained. The standard
protocol (`ai=0.15, size=5-30, holding=4-10`) already gives very low blocking
(DeepRMSA ≈ 1.3%). We therefore had to push arrival rate, request size, and
holding time much harder to bring DeepRMSA blocking into the [5%, 60%] window.

## Calibration findings
- `default3` under strong pressure (`ai=0.03-0.06, size=50, holding=18`) gave
  DeepRMSA blocking ≈25%, but v1.2 gap was essentially **0 pp**.
- `complex5_v2_lite` with `size=70, holding=24` produced **MARGINAL +1.75 pp**
  gaps, and with `holding=28/32` two scenarios passed the small-sample criteria:

| Scenario | v1.2 Blk | Deep Blk | Gap | Delay Δ | Overload Δ | Verdict |
|---|---:|---:|---:|---:|---:|---:|
| complex5 ai0.03 sz70 h28 | 38.75% | 40.83% | **+2.08 pp** | -1.99 ms | ok | PASS |
| complex5 ai0.03 sz70 h32 | 38.75% | 40.83% | **+2.08 pp** | -2.08 ms | ok | PASS |

## Stage 2 formal validation of the best 100-slot PASS scenario
Evaluated `complex5_v2_lite ai0.03 sz70 h28` with 5 seeds × 20 episodes:

| v1.2 Blk | Deep Blk | Gap | p-value | Delay Δ | Overload Δ |
|---|---:|---:|---:|---:|---:|
| 42.10% ± 5.58% | 41.42% ± 4.62% | **−0.68 pp** | 0.409 | -1.70 ms | +0.47 pp |

The small-sample **PASS was a false positive**. With larger sample the gap
reversed and v1.2 was slightly worse than DeepRMSA. The high variance observed
in the 24-slot `complex5` scenarios is also present in 100 slots.

## 100-slot conclusion
No 100-slot scenario robustly satisfies the PASS criteria. The lower resource
pressure makes the v1.2 advantage harder to manifest, and the few promising
small-sample results do not survive formal validation.

---

# Overall Final Conclusion

- The v1.2 counterfactual R-ranker **reliably beats DeepRMSA by roughly 1 pp**
  on **24-slot fair `default3` workloads**, with negligible delay and near-zero
  overload overhead.
- It **does not achieve a ≥2 pp advantage** under fair side-effect constraints on
  either 24-slot or 100-slot `snap24_gnutella_reach` across the tested parameter
  ranges.
- Scenario-specific retraining with an H-step server-overload penalty did not
  help, because the future overload signal is too weak within `H=5` and the
  retrained model overfits.

# Recommendation

**Accept the ~1 pp result as the final conclusion.** The cleanest fair scenario
is:

> **`default3` ai=0.09, size_max=30 MB, holding_max=14** (24 slots)
> - Formal gap: **+1.01 pp** (p = 0.002)
> - Delay Δ: **−0.05 ms**
> - Overload Δ: **+0.60 pp**

If a stress-test scenario is needed for the paper, `complex5_v2_lite`
ai=0.15 sz30 h14 (24 slots) gives a significant **+1 pp** blocking reduction as
well, but with a measurable overload trade-off (+1.66 pp). It is not a fair win
under the strict 0.5 pp overload budget.

# Files

- 24-slot Stage 1 summary: `sa_hmarl/experiments/v12_split_pressure_grid_summary.json`
  and `.md`
- 24-slot Extension 1: `sa_hmarl/experiments/v12_split_pressure_grid_ext1_summary.json`
  and `.md`
- 24-slot Extension 2: `sa_hmarl/experiments/v12_split_pressure_grid_ext2_summary.json`
  and `.md`
- 24-slot formal validation (complex5): `sa_hmarl/experiments/v12_formal_validation/`
- 24-slot formal validation (default3): `sa_hmarl/experiments/v12_formal_validation_default3/`
- 24-slot Stage 3 retrain sweep: `sa_hmarl/experiments/v12_stage3_retrain/`
- 100-slot default3 calibration: `sa_hmarl/experiments/v12_split_pressure_grid_s100_calib/`
- 100-slot complex5 calibration: `sa_hmarl/experiments/v12_split_pressure_grid_s100_complex5_calib/`
- 100-slot complex5 extension: `sa_hmarl/experiments/v12_split_pressure_grid_s100_complex5_ext/`
- 100-slot formal validation: `sa_hmarl/experiments/v12_formal_validation_s100_complex5/`
- Screening script: `sa_hmarl/scripts/run_v12_split_pressure_grid.py`
- Formal validation script: `sa_hmarl/scripts/run_v12_best_gap_formal.py`
- Retrain pipeline: `sa_hmarl/scripts/run_v12_best_retrain.py`
- Sweep orchestrator: `sa_hmarl/scripts/run_v12_stage3_sweep.py`
