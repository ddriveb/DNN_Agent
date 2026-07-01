# SA-HMARL Experiment Summary

> Last updated: 2026-06-28

## Final method

**v1.2 static counterfactual R-ranker** (`r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt`)

## Main results

| Scenario | Method | Blocking | p vs DeepRMSA | Notes |
|---|---|---|---|---|
| Hard 24-slot (`default3`, AI=0.09, size=30, hold=14) | v1.2 | **53.92 ± 5.90%** | 0.002 | +1.01 pp vs DeepRMSA |
| Hard 24-slot | DeepRMSA | 54.94 ± 5.82% | — | — |
| Easy 100-slot (`default3`) | v1.2 transfer | **0.925 ± 0.487%** | 0.007 | −0.36 pp vs DeepRMSA |
| Easy 100-slot | DeepRMSA S100 | 1.288 ± 0.555% | — | — |

## Negative results

- **Stage 3 scenario-specific retraining**: no gain; H=5 rollouts lack overload events.
- **Lyapunov inference-time rerank**: ≤0.06 pp improvement; not practically useful.
- **v1.3 spectrum-viability label**: degrades v1.2 by +0.26–0.37 pp; `Phi_after` is weakly anti-correlated with the true H-step return.

## Recommendation

Use 24-slot `default3` as the paper's primary result and 100-slot as supplementary. Do not include Lyapunov as a method.
