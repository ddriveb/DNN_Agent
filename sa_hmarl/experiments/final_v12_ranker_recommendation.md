# Final Recommendation: v1.2 Static Counterfactual Ranker

> Date: 2026-06-28
> Project: SA-HMARL counterfactual R-side ranking for DNN offloading in optical-MEC networks

## Decision

Accept the **v1.2 static counterfactual ranker** as the final method. No further retraining or inference-time reranking is warranted.

## Why

- It beats DeepRMSA in the hard 24-slot scenario by **+1.01 pp** (p = 0.002).
- It beats DeepRMSA in the easy 100-slot scenario by **−0.36 pp** (p = 0.007).
- Transfer v1.2 is as good as S100-specific retraining, so no extra training is needed.
- Lyapunov rerank, scenario-specific retraining, and the v1.3 spectrum-viability label do not add practical value.

## Primary result

| Method | Blocking | Config |
|---|---|---|
| v1.2 ranker | **53.92 ± 5.90%** | `snap24_gnutella_reach`, 24 slots, `default3`, AI=0.09, size_max=30, holding_max=14 |
| DeepRMSA | 54.94 ± 5.82% | same |
| Gap | **+1.01 pp** (p = 0.002) | |

- Checkpoint: `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`
- Side effects vs DeepRMSA: delay −0.05 ms, overload +0.60 pp.

## Supplementary low-blocking result

| Method | Blocking | Config |
|---|---|---|
| v1.2 transfer | **0.925 ± 0.487%** | `snap24_gnutella_reach`, 100 slots, `default3`, k=5, max_blocks=10 |
| v1.2 S100 retrained | 0.950 ± 0.505% | same |
| DeepRMSA S100 | 1.288 ± 0.555% | same |
| Gap vs DeepRMSA | **−0.36 pp** (p = 0.007) | |

## What was ruled out

1. **Scenario-specific retraining (Stage 3)** — FAIL. Future server-overload penalties in the return produced identical metrics because H=5 rollouts have too few overload events to change ranking labels.
2. **Lyapunov dynamic rerank** — FAIL. Best formal blocking improvement only +0.06 pp; <2% of actions changed; `H_spec` saturated at the clip limit.
3. **v1.3 spectrum-viability label** — FAIL. Adding a post-decision `Phi_after` bonus based on next-request feasible counts degraded blocking by +0.26–0.37 pp; the bonus is weakly anti-correlated with the true H-step return (Spearman ≈ -0.06).
4. **Aggressive 24-slot / 100-slot scenario screening** — FAIL to find a second fair PASS beyond `default3`, but the primary PASS is robust.

## Paper plan

- Use the 24-slot `default3` result as the main experimental claim.
- Use the 100-slot standard result as a supplementary low-blocking validation.
- Mention Lyapunov only as a theoretical interpretation, not as a proposed method.
