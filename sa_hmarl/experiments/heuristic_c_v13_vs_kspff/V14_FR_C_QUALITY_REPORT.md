# V1.4 Feasible-Region C-Side Quality Selector Report

**Date:** 2026-07-07  
**Scope:** Evaluate `fr_count_c` (ablation) and `fr_quality_c` (v1.4 proposed C-side) paired with v1.3 R-side ranker, under the old long-horizon protocol.

---

## 1. Implementation Summary

Two new C-side selectors were added to `sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py` and wired into `eval_long_horizon_system_comparison.py`:

- **`fr_count_c`**: Maximizes the normalized count of legal R actions for each legal C action.
- **`fr_quality_c`**: Quality-aware feasible-region selector. For each legal `(split, server)` C action it builds the R observation, samples up to 64 diverse legal R actions (KSP-FF, per-path first-fit, per-path largest-block, high-SE anchors), computes a lightweight optical-quality proxy `q_fast`, and scores the C action as

  \[
  \Psi = \lambda_N \widetilde{N}_R + \lambda_Q \, \text{TopAvg}_M(q_{fast}) - \lambda_U u_v^{post}
  \]

  Default weights: \(\lambda_N=0.45, \lambda_Q=0.40, \lambda_U=0.15, M=8\).

- **`fr_quality_strong_c`**: Overload-strong variant with \(\lambda_N=0.40, \lambda_Q=0.35, \lambda_U=0.25\).

All selectors respect `agent_c_mask`. No R-side ranker is invoked during C-side selection, and no checkpoint was changed.

A small bug was fixed in `_select_c_action`: heuristic C-mode names previously had the first `_c` occurrence removed (`fr_count_c` → `frount`), which broke the new modes. The evaluator now strips the `_c` suffix correctly.

---

## 2. COST239 Full Results (Old Protocol)

| C-side | R-side | Blocking mean ± std | Raw empty | Server overload | Delay mean / P95 (ms) | Decision mean / P95 (ms) |
|---|---|---:|---:|---:|---:|---:|
| PPO-C | KSP-FF K50 | **10.37 % ± 1.29 %** | 12.99 % | 10.37 % | 8.80 / 16.78 | 8.29 / 12.70 |
| PPO-C | v1.3 | **7.68 % ± 0.92 %** | 9.71 % | 7.68 % | 9.54 / 17.27 | 12.37 / 18.55 |
| FR-Count | KSP-FF K50 | 70.17 % ± 0.83 % | 77.99 % | 70.17 % | 17.37 / 26.92 | 5.59 / 9.49 |
| FR-Count | v1.3 | 69.69 % ± 1.09 % | 77.57 % | 69.68 % | 18.86 / 28.71 | 7.49 / 13.58 |
| FR-Quality | KSP-FF K50 | 56.59 % ± 0.89 % | 64.48 % | 56.58 % | 14.09 / 25.38 | 6.56 / 11.26 |
| FR-Quality | v1.3 | 53.03 % ± 2.26 % | 60.68 % | 53.02 % | 14.81 / 26.59 | 9.14 / 15.92 |

Protocol: `edge_cost_min=0.1`, `edge_cost_max=2.2`, seeds `3030,4040,5050,6060,7070`, 10 000 requests / seed, 2 000 warmup.

### Key observations

- **FR-Count is dominated by server overload.** Maximizing R-side legal-action volume ignores the compute cost placed on the edge server, so it consistently overloads servers.
- **FR-Quality substantially improves over FR-Count** (−14 to −17 pp blocking), confirming that the `u_v^{post}` term and `q_fast` optical proxy are working in the right direction.
- **FR-Quality is still far from PPO-C.** The gap is ~45 pp, so it cannot replace PPO-C as the main C-side policy.

---

## 3. v1.4 Delta vs Baselines

| Method | Δ blocking vs same-C KSP-FF | Δ blocking vs PPO-C + v1.3 | Comment |
|---|---:|---:|:---|
| FR-Count + v1.3 | −0.48 pp | +62.01 pp | Count-only is catastrophic; overload dominates. |
| FR-Quality + KSP-FF | — | +48.91 pp | Quality term reduces but does not fix overload. |
| FR-Quality + v1.3 | −3.56 pp | +45.35 pp | R-side ranker still helps, but C-side overload swamps gains. |

Conclusion: the bottleneck is **C-side server-overload control**, not R-side optical selection.

---

## 4. Cross-Topology Smoke (Confirmation)

A single-seed smoke was run on German17 to verify the COST239 pattern.

| Topology | PPO-C + KSP-FF | PPO-C + v1.3 | FR-Count + v1.3 | FR-Quality + v1.3 | Best interpretable C? |
|---|---:|---:|---:|---:|:---|
| COST239 (full) | 10.37 % | **7.68 %** | 69.69 % | 53.03 % | None — PPO-C wins. |
| German17 (smoke) | 9.13 % | 11.25 % | 70.88 % | 49.00 % | None — PPO-C wins. |

German17 smoke confirms the same qualitative ranking: FR-Quality < PPO-C, and FR-Count is the worst.

Full cross-topology sweeps on JPN48 / NSFNET were **deferred**: COST239 full and German17 smoke already establish that v1.4 is an ablation, not a main-method replacement, so the additional compute would not change the conclusion.

---

## 5. Overload-Strong Variant (Smoke)

| Variant | KSP-FF | v1.3 |
|---|---:|---:|
| FR-Quality (default, smoke) | 51.75 % | 53.12 % |
| FR-Quality-Strong (λ_U=0.25, smoke) | 51.25 % | 51.75 % |

The overload-strong variant gives only a marginal improvement (~1 pp) on a single seed. It does not close the gap to PPO-C.

---

## 6. Decision and Recommendations

Per the pre-defined decision criteria, this is **Scenario 2**:

> `FR-Quality + v1.3` is明显弱于 `PPO-C + v1.3`， but stronger than `FR-Count` / fixed C.

**Decision:**
- **Do not promote v1.4 as the main method.** Keep the paper's main experimental result as **PPO-C + v1.3**.
- **Present FR-Quality as an interpretable C-side ablation** that shows:
  1. Maximizing R-side feasible-region volume alone is insufficient (FR-Count).
  2. Adding a lightweight optical-quality proxy and overload-awareness helps, but a learned PPO-C policy is still required for low blocking.
- **Theoretical framing remains valid:** the fixed-C conditional post-decision regret decomposition is the R-side theoretical core; C-side is a separate feasibility-preserving/overload-aware selector.

**Future work (if desired):**
- Investigate why FR-Quality still overloads servers despite the `u_v^{post}` term. Possible directions:
  - Add a stronger hinge penalty or dynamic λ_U that scales with system load.
  - Include edge-compute-cost-aware normalization or a server capacity headroom term.
  - Use the fixed-C oracle (`oracle_r_query`) as an upper-bound reference to quantify how much C-side selection alone can improve.
- Run full German17 / JPN48 / NSFNET only if a future FR-Quality variant gets within ~5 pp of PPO-C on COST239.

---

## 7. Output Files

| File | Description |
|---|---|
| `v14_cost239_smoke.json/.md` | COST239 smoke with all four v1.4 methods |
| `v14_cost239_full.json/.md` | COST239 full (6 methods, 5 seeds) |
| `v14_cost239_strong_smoke.json/.md` | Overload-strong variant smoke |
| `v14_german17_smoke.json/.md` | German17 confirmation smoke |
| `V14_FR_C_QUALITY_REPORT.md` | This report |

All COST239/German17 experiments used `edge_cost_min=0.1` (verified in each JSON config). Old `edge_cost_min=0.5` files were not overwritten.
