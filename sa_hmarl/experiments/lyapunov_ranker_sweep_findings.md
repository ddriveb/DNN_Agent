# Lyapunov Drift-Plus-Penalty Inference-Time Reranking — Findings

## What was tested
A lightweight inference-time dynamic rerank that maintains a scalar risk queue
`H_spec(t)` and subtracts `λ_spec * H_spec(t) * D_spec(a)` from the v1.2 ranker
score.  No model retraining.  Two scenarios:

- **A (clean)**：`default3`, `ai=0.09`, `size_max=30`, `holding_max=14`
- **B (stress)**：`complex5_v2_lite`, `ai=0.15`, `size_max=30`, `holding_max=14`

Screening grid:
- `λ_spec ∈ {0.02, 0.05, 0.10, 0.20}`
- `ε_spec ∈ {0.02, 0.05, 0.10}`
- `damage_mode ∈ {fs_path, fs_only, path_only}`
- 3 seeds × 5 episodes × 80 requests

Top 3 screening configs were formally validated with 5 seeds × 20 episodes.

## Results

### Screening summary
| Scenario | best Δ blocking vs v1.2 | Δ overload | Δ delay | changed% | verdict |
|---|---:|---:|---:|---:|---:|
| A default3 | 0.00 pp | 0.00 pp | ~0 ms | <0.3% | FAIL |
| B complex5 | −0.08 pp (slightly worse) | up to −0.25 pp | up to −0.12 ms | 1.3–1.6% | FAIL |

No configuration met the PASS threshold (blocking reduction ≥0.5 pp with
overload/delay within budget).

### Formal validation (top 3 from screening)
| Scenario | λ_spec | ε_spec | damage_mode | Δ blocking | Δ overload | Δ delay | changed% | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| A default3 | 0.02 | 0.02 | path_only | +0.06 pp | −0.12 pp | −0.01 ms | 0.5% | FAIL |
| A default3 | 0.02 | 0.02 | fs_path   | +0.04 pp | −0.10 pp | −0.01 ms | 0.5% | FAIL |
| A default3 | 0.02 | 0.02 | fs_only   | 0.00 pp  |  0.00 pp |  0.00 ms | 0.0% | FAIL |

The formal validation confirms the screening: the dynamic rerank gives at most
~0.06 pp blocking reduction, far below the 0.5 pp PASS bar.

## Why it did not help

1. **Queue saturation**. In every run `p95_H_spec` hit the clip value of 20.0.
   The binary update mode adds a pressure of 1.0 whenever the raw C mask is
   empty, while `ε_spec` was at most 0.10.  The queue therefore grows to the
   clip almost immediately and stays there, turning the dynamic penalty into an
   effectively constant static penalty.

2. **Tiny action-change rate**. Even when `H_spec` is at the clip, the penalty
   term changes the ranker’s selected action in fewer than 2% of decisions.
   The v1.2 ranker scores are already well-separated, and the added penalty is
   too small (mean adjustment <0.25) to flip the argmax for most candidates.

3. **Damage proxy correlates with what v1.2 already does**. The v1.2 ranker was
   explicitly trained with path/FS penalties (`path_penalty=0.05`,
   `fs_penalty=0.05`).  Adding a similar inference-time penalty on top of an
   already resource-aware model provides little additional signal.

## Answers to the required questions

1. **Is the dynamic Lyapunov queue better than fixed v1.2?**  
   No. At best it gives ~0.06 pp blocking reduction on the clean scenario and
   slightly hurts blocking on the stress scenario.

2. **Does it lower raw_empty/NSB or just trade overload/delay?**  
   It slightly lowers overload and delay in some configurations, but blocking is
   essentially unchanged.  The effect is not a meaningful trade-off; it is
   within noise.

3. **Does H_spec participate in decisions?**  
   Mechanically yes (`changed_action_rate_vs_v12` is non-zero), but practically
   no: <2% of decisions change, and the queue is saturated at the clip, so the
   dynamic information is lost.

4. **Should the dynamic queue feature be baked into training?**  
   Not on the basis of these results.  The current penalty formulation does not
   add value beyond what v1.2 already learned.  A different formulation
   (e.g. non-saturating queue, normalized scores, or C-side server-pressure
   reranking) might be worth testing, but the R-side spec queue is not
   promising.

5. **What is the conclusion?**  
   Keep v1.2 as a static, resource-aware ranker and treat the Lyapunov
   interpretation as a theoretical explanation rather than an engineering
   module.  The static penalty already encodes the relevant preferences.

## Files
- Screening summary: `sa_hmarl/experiments/lyapunov_ranker_sweep_screening.json`
  and `.md`
- Formal summary: `sa_hmarl/experiments/lyapunov_ranker_sweep_formal.json`
  and `.md`
- Per-run outputs: `sa_hmarl/experiments/lyapunov_ranker_sweep/`
- Implementation:
  `sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_closed_loop.py`
- Sweep script: `sa_hmarl/scripts/run_lyapunov_ranker_sweep.py`
