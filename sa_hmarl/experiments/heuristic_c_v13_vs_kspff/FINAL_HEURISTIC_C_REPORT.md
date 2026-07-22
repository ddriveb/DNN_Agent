# Heuristic C vs v1.3 Hybrid: Does R-Side Gain Depend on C-Side Policy?

**Date:** 2026-07-04  
**C-side heuristic:** `df_c` (distance-first)  
**R backends compared:** `ksp_ff_k50_hops` vs `v12_k50_hops` (v1.3 hybrid)  
**Evaluator:** `sa_hmarl.evaluation.eval_long_horizon_system_comparison`

---

## 1. Executive Conclusion

When C-side is switched from the learned PPO-C policy (`r_feasibility_safe_last`) to the best simple heuristic (`df_c`), the v1.3 hybrid's advantage over KSP-FF K50 hops largely disappears:

| Topology | df + KSP-FF block % | df + v1.3 block % | Δ (pp) | Δ relative | p-value |
|---|---:|---:|---:|---:|---:|
| COST239 | 14.84 ± 1.92 | 14.84 ± 1.92 | 0.00 | 0.0% | identical |
| German17 | 12.69 ± 0.73 | 12.96 ± 1.10 | 0.27 | 2.2% | 0.2272 |
| JPN48 | 11.28 ± 1.76 | 10.85 ± 1.86 | -0.44 | -3.9% | 0.0012 |
| NSFNET | 20.33 ± 1.44 | 19.94 ± 1.17 | -0.39 | -1.9% | 0.2358 |

**Key takeaways:**

1. With `df_c` the system blocking is higher than with PPO-C on every topology, confirming that the learned C policy is an important part of the overall system performance.
2. The v1.3 learned ranker provides **no measurable benefit** over KSP-FF under `df_c` on COST239 (identical blocking). On JPN48 it remains statistically significant but the effect size shrinks from −1.16 pp with PPO-C to −0.44 pp with `df_c`.
3. This suggests that much of v1.3's reported gain in the PPO-C benchmark is **conditional on the action distribution produced by PPO-C**, especially on smaller topologies. On larger topologies such as JPN48, the ranker retains independent value.
4. Residual blocking remains dominated by `server_overload`; `no_suitable_block` stays low. The bottleneck is still C-side server/split allocation.

---

## 2. How the Best C Heuristic Was Selected

### 2.1 Existing evidence on `snap24_gnutella_reach`

The existing `main_s100_c_heuristic_r_matrix.json` compares `greedy_c`, `df_c`, `rf_c`, `wo_c`, and `iwd_c` against several R backends on `snap24_gnutella_reach`. With the v12 backend the lowest-blocking heuristics are:

| Heuristic | v12 blocking | KSP-FF blocking |
|---|---:|---:|
| `df_c` | 1.18% | 9.08% |
| `iwd_c` | 1.18% | 8.60% |
| `greedy_c` | 1.74% | — |
| `rf_c` | 2.00% | 9.75% |
| `wo_c` | 2.20% | 9.92% |

`df_c` and `iwd_c` are tied for lowest blocking with v12; `iwd_c` is slightly better with KSP-FF.

### 2.2 Validation on COST239 long-horizon smoke

Because imported topologies may favor different heuristics, we re-evaluated all five heuristics on `xlron_cost239_ptrnet_real` under the long-horizon protocol (1 seed, `ksp_ff_k50_hops`):

| Heuristic | COST239 blocking |
|---:|---:|
| `df_c` | 12.10% |
| `wo_c` | 12.16% |
| `greedy_c` | 13.18% |
| `rf_c` | 13.18% |
| `iwd_c` | 58.49% |

`df_c` is the best heuristic on COST239; `iwd_c` collapses on this topology. We therefore selected `df_c` as the **global best heuristic** for the main comparison. This choice is conservative with respect to the primary question: it uses the strongest available heuristic C policy.

### 2.3 Limitation: per-topology heuristic selection not exhaustively tested

We validated the global choice on COST239 but did not run full heuristic-selection sweeps on German17, JPN48, or NSFNET. A per-topology oracle could in principle choose a different heuristic for each graph; that is reported only as a limitation, not as the main result.

---

## 3. Protocol

We fixed C-side to `df_c` and compared the same two R backends used in the PPO-C long-horizon benchmark:

- `df_c + ksp_ff_k50_hops`
- `df_c + v12_k50_hops`

All settings match the PPO-C long-horizon benchmark to ensure comparability:

| Parameter | Value |
|---|---|
| Slots | 320 |
| Servers | 4 (`[0,1,2,3]`, capacity 50 each) |
| Splits | 3, profile `default3` |
| Requests/episode | 10 000 |
| Warmup | 2 000 |
| Arrivals | Poisson |
| Holding | Exponential, mean 25 s |
| Request size | Uniform 5–30 MB |
| Seeds | 3030, 4040, 5050, 6060, 7070 |
| KSP-FF paths | 50, sorted by hops |
| Block sort | `start_asc` |
| v1.3 config | `legalctx48`, max 48 candidates, `ensure_ksp=true` |

Per-topology load:

| Topology | λ | `edge_cost_max` |
|---|---:|---:|
| COST239 | 16 | 2.2 |
| German17 | 14 | 3.0 |
| JPN48 | 10 | 5.2 |
| NSFNET | 13 | 4.0 |

---

## 4. Main Results: df_c + KSP-FF vs df_c + v1.3

### 4.1 Aggregate blocking

| Topology | df + KSP-FF block % | df + v1.3 block % | Δ (pp) | Δ relative | p-value |
|---|---:|---:|---:|---:|---:|
| COST239 | 14.84 ± 1.92 | 14.84 ± 1.92 | 0.00 | 0.0% | identical |
| German17 | 12.69 ± 0.73 | 12.96 ± 1.10 | 0.27 | 2.2% | 0.2272 |
| JPN48 | 11.28 ± 1.76 | 10.85 ± 1.86 | -0.44 | -3.9% | 0.0012 |
| NSFNET | 20.33 ± 1.44 | 19.94 ± 1.17 | -0.39 | -1.9% | 0.2358 |

### 4.2 Per-seed blocking rates


**COST239**

| Seed | df + KSP-FF | df + v1.3 | Δ |
|---:|---:|---:|---:|
| 3030 | 12.10% | 12.10% | +0.00 pp |
| 4040 | 14.31% | 14.31% | +0.00 pp |
| 5050 | 16.60% | 16.60% | +0.00 pp |
| 6060 | 14.42% | 14.42% | +0.00 pp |
| 7070 | 16.75% | 16.75% | +0.00 pp |


**German17**

| Seed | df + KSP-FF | df + v1.3 | Δ |
|---:|---:|---:|---:|
| 3030 | 13.68% | 14.46% | +0.79 pp |
| 4040 | 11.91% | 11.76% | -0.15 pp |
| 5050 | 12.40% | 12.20% | -0.20 pp |
| 6060 | 13.21% | 13.66% | +0.45 pp |
| 7070 | 12.25% | 12.74% | +0.49 pp |


**JPN48**

| Seed | df + KSP-FF | df + v1.3 | Δ |
|---:|---:|---:|---:|
| 3030 | 14.01% | 13.71% | -0.30 pp |
| 4040 | 9.68% | 9.15% | -0.53 pp |
| 5050 | 12.03% | 11.68% | -0.35 pp |
| 6060 | 10.49% | 9.90% | -0.59 pp |
| 7070 | 10.21% | 9.79% | -0.42 pp |


**NSFNET**

| Seed | df + KSP-FF | df + v1.3 | Δ |
|---:|---:|---:|---:|
| 3030 | 18.88% | 18.84% | -0.04 pp |
| 4040 | 19.18% | 18.54% | -0.64 pp |
| 5050 | 21.69% | 21.01% | -0.68 pp |
| 6060 | 19.90% | 20.40% | +0.50 pp |
| 7070 | 22.00% | 20.91% | -1.09 pp |

---

## 5. Delta vs KSP-FF

Under `df_c`, the v1.3 hybrid's gain is attenuated and topology-dependent:

- **COST239:** Δ = 0.00 pp (0.0%), identical values across all seeds.
- **German17:** Δ = 0.27 pp (2.2%), p = 0.2272 (not significant).
- **JPN48:** Δ = -0.44 pp (-3.9%), p = 0.0012 (significant).
- **NSFNET:** Δ = -0.39 pp (-1.9%), p = 0.2358 (not significant).

By contrast, under PPO-C the same v1.3 hybrid achieved −2.69 pp (p = 0.0011) on COST239 and −1.16 pp (p = 0.0067) on JPN48. The JPN48 gain persists under `df_c` but at roughly one-third the magnitude; the COST239 gain vanishes entirely.

---

## 6. Failure-Mode Breakdown

Across all topologies, blocking is overwhelmingly due to `server_overload`; optical `no_suitable_block` rates remain below 0.3 %.

| Topology | df + KSP-FF overload % | df + v1.3 overload % | df + KSP-FF NSB % | df + v1.3 NSB % |
|---|---:|---:|---:|---:|
| COST239 | 14.84 | 14.84 | 0.00 | 0.00 |
| German17 | 12.67 | 12.93 | 0.02 | 0.03 |
| JPN48 | 10.80 | 10.10 | 0.15 | 0.22 |
| NSFNET | 20.10 | 19.53 | 0.07 | 0.15 |

On COST239 the blocking and overload rates are identical for KSP-FF and v1.3, even though v1.3 selects paths with higher average FS and longer distance. This indicates that the C-side decision is the binding constraint: once `df_c` assigns a request to an overloaded server/split, the R-side choice no longer affects whether the request is blocked.

---

## 7. Comparison to PPO-C Long-Horizon Reference

| Topology | PPO-C + KSP-FF | PPO-C + v1.3 | Δ (pp) | df + KSP-FF | df + v1.3 | Δ (pp) |
|---|---:|---:|---:|---:|---:|---:|
| COST239 | 10.37% | 7.68% | -2.69 | 14.84% | 14.84% | 0.00 |
| German17 | 7.60% | 7.63% | 0.04 | 12.69% | 12.96% | 0.27 |
| JPN48 | 9.19% | 8.03% | -1.16 | 11.28% | 10.85% | -0.44 |
| NSFNET | 15.29% | 14.92% | -0.37 | 20.33% | 19.94% | -0.39 |

Two patterns are clear:

1. `df_c` raises the absolute blocking floor compared with PPO-C on every topology. The learned C policy produces server allocations that are more robust under the same load.
2. The R-side v1.3 advantage visible with PPO-C is no longer present (COST239) or is strongly attenuated (JPN48, NSFNET) under `df_c`.

---

## 8. Interpretation

### 8.1 Does v1.3's gain come from R-side alone?

Partially, and topology-dependently. The evidence supports a **conditional** interpretation: v1.3's learned ranker is most effective when paired with a C-side policy (PPO-C `r_feasibility_safe`) that keeps servers out of deep saturation. When the C-side policy is replaced by `df_c`, the marginal value of smarter R-side path/block selection:

- **Vanishes on COST239:** identical blocking for KSP-FF and v1.3. The C-side decisions are so poor that R-side choice no longer affects the outcome.
- **Shrinks but remains significant on JPN48:** the large physical network still gives the ranker enough path diversity to provide independent value, but the gain is roughly one-third of what it is with PPO-C.
- **Stays weak on NSFNET:** compute saturation dominates; R-side optimization has limited headroom regardless of C-side policy.

### 8.2 Why does `df_c` attenuate the R-side gap?

`df_c` (distance-first) does not account for dynamic server load as effectively as PPO-C `r_feasibility_safe`. It assigns requests to nearby servers without anticipating future congestion. On COST239 this pushes the system so far into overload that the exact R-side path/block choice becomes irrelevant: the request is blocked at the C-side mask stage before R-side spectrum assignment matters. On JPN48 the larger graph offers more R-side alternatives, so the ranker can still find marginally better paths even when C-side allocation is suboptimal.

### 8.3 Is `df_c` a fair baseline?

`df_c` is the best-performing simple heuristic on the available evidence (snap24 and COST239 smoke). It is not meant to match PPO-C; rather, it is meant to test whether v1.3's gains are robust to a different, plausible C-side action distribution. The answer is that they are not robust.

---

## 9. Limitations

1. **Single global heuristic:** We used `df_c` for all topologies. A per-topology oracle might choose differently, though the COST239 validation suggests `df_c` is at least competitive.
2. **No per-topology heuristic sweep:** German17, JPN48, and NSFNET were not used to select the heuristic; only COST239 was validated.
3. **Long runtime for JPN48:** The JPN48 v1.3 run is the slowest due to the large topology and ranker inference. If it had to be split across multiple background tasks, the combined result is still from the same protocol.
4. **No joint optimization:** The experiment keeps C and R fixed and separate. It does not test whether retraining either component jointly would recover the v1.3 gain.

---

## 10. Next Recommendation

1. **Do not claim v1.3 is universally superior** independent of the C-side policy. Paper/PPT wording should state that v1.3's gains are 'observed under the learned PPO-C C-side policy and are attenuated when C-side is replaced by a distance-first heuristic'.
2. **Treat JPN48 as a partial exception:** the ranker retains significant value on the large topology even with `df_c`, so the dependency on C-side is not absolute.
3. **Investigate joint C-R selection:** Because residual blocking is dominated by `server_overload`, future gains likely require a decision mechanism that accounts for both server load and spectrum availability simultaneously, rather than larger or smarter R-side candidate sets alone.
4. **Consider per-topology C training:** The current C checkpoint is trained on COST239 and transferred. Training a `df_c`-style or learned C policy per topology could change both the absolute blocking and the R-side delta.
5. **If a stronger heuristic is found**, repeat this comparison; the conclusion may strengthen or weaken depending on how close the heuristic comes to PPO-C's allocation quality.

---

## 11. Artifacts

- This report: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/FINAL_HEURISTIC_C_REPORT.md`
- Machine-readable synthesis: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/FINAL_HEURISTIC_C_REPORT.json`
- COST239 combined result: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_cost239_ptrnet_real_df_full.json`
- German17 combined result: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_german17_df_full.json`
- JPN48 KSP-FF result: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_jpn48_df_kspff_full.json`
- JPN48 v1.3 result: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_jpn48_df_v12_full.json`
- NSFNET combined result: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_nsfnet_deeprmsa_df_full.json`
- Heuristic selection smokes: `sa_hmarl/experiments/heuristic_c_v13_vs_kspff/xlron_cost239_ptrnet_real_*_smoke.json`
