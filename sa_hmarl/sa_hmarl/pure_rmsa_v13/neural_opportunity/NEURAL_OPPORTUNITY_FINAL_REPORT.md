# Neural Opportunity v1 Final Report

## Executive decision

Two decisions are required because the experiment had two distinct targets.

1. **GO: replace the Opportunity-only analytic deployment variant.** On the
   frozen 10-seed confirmatory set, Neural Opportunity differs from its exact
   Opportunity-only Teacher by only +0.023/+0.002/+0.041 percentage points on
   NSFNET/USNET/JPN48, with every confidence interval crossing zero. It is
   2.7-3.2x faster than that Teacher.
2. **HOLD: do not claim a lossless replacement for Full Direct.** It trails
   Full Direct by +0.210/+0.009/+0.201 pp. USNET is indistinguishable, but the
   NSFNET and JPN48 gaps are real enough that Full Direct remains the primary
   performance algorithm.

The user's accepted deployment tradeoff was Opportunity-only, not Full Direct.
Under that scope, the central hypothesis is confirmed: training can move the
expensive opportunity-price calculation offline, leaving deterministic
feasibility plus one small neural forward online.

## Online algorithm

For each request:

1. Scan the locked K=50 hops path pool with bit-parallel feasibility.
2. Keep the first three feasible paths and all legal start slots.
3. Build one 110-dimensional row per path:
   - ten request/path scalars;
   - a 50-slot end-to-end common-free map;
   - a 50-slot path-conditioned conflict-pressure map.
4. Run one shared `110 -> 16 -> 50` ReLU MLP over at most three rows.
5. Mask illegal starts and minimize `path_rank + neural_price`.

There is no online Shadow pricer, compressed feasible-window sketch, exact
Opportunity pricer, rollout, GNN, fallback, or C kernel.

## Conflict-pressure mechanism

The first cheap model used only global mean occupancy. It was too lossy because
it could not identify where routes competing with the candidate were under
pressure. The final representation precomputes a static route co-occurrence
kernel:

`C = sum_q (1 / hops(q)) * incidence(q) outer incidence(q)`

For candidate path `p`, its normalized conflict weights are the sum of rows of
`C` corresponding to arcs in `p`. Online, one matrix product between these
weights and the current arc-slot bitmap yields a 50-slot pressure profile. This
is a compressed, topology-aware approximation of the Teacher's probe logic,
not action-label classification.

## Training

- Exact Teacher: locked `OpportunityOnlyDirectSelector`.
- Dense label: exact weighted opportunity contribution for every legal start,
  not only the selected action.
- Data per topology: about 27.8k training states from seeds 60001-60003 and
  about 9.2k validation states from seed 60101.
- Loss: Huber price regression + 0.5 listwise KL + 0.5 pairwise margin.
- Per-topology P95 target normalization is folded into the final layer before
  NumPy deployment.
- Models were frozen before pilot and confirmatory seeds were read.
- DAgger was not run: the frozen policy already reproduced the Teacher in
  closed loop, so additional imitation would add cost without a causal target.

## Representation ablation

| Topology | Global-mean Top-1 | Conflict Top-1 | Global regret | Conflict regret |
|---|---:|---:|---:|---:|
| NSFNET | 42.5% | 47.9% | 0.120 | 0.094 |
| USNET | 38.0% | 43.5% | 0.182 | 0.134 |
| JPN48 | 30.1% | 32.4% | 0.218 | 0.180 |

Top-1 recall remains modest because many exact starts are near ties. Closed-loop
blocking, rather than exact argmin identity, is therefore the primary test.

## Frozen 10-seed confirmation

Seeds 23001-23010 were used only after the model was frozen. Current KSP runs
were required to reproduce the archived KSP blocked counts exactly before the
locked Full Direct and Opportunity-only fact table was joined.

| Topology | KSP-FF K=50 | Full Direct | Opp Teacher | Neural | Neural vs KSP | Neural vs Full | Neural vs Opp |
|---|---:|---:|---:|---:|---:|---:|---:|
| NSFNET | 7.739% | 6.117% | 6.304% | 6.327% | -1.412 pp | +0.210 pp | +0.023 pp |
| USNET | 8.599% | 7.698% | 7.705% | 7.707% | -0.892 pp | +0.009 pp | +0.002 pp |
| JPN48 | 8.216% | 6.950% | 7.110% | 7.151% | -1.065 pp | +0.201 pp | +0.041 pp |

Neural vs KSP is 10/0/0 on all topologies and all 95% CIs are below zero.
Neural vs Opportunity-only is 5/0/5, 4/0/6, and 5/0/5; every CI crosses zero.
Full Direct KSP-gain retention is 87.1%, 99.0%, and 84.1% respectively.

## Fair Python latency

| Topology | KSP-FF | Full Direct | Opp Teacher | Neural | Neural/Opp | Neural/KSP |
|---|---:|---:|---:|---:|---:|---:|
| NSFNET | 0.0162 ms | 0.3580 ms | 0.1397 ms | 0.0448 ms | 0.32x | 2.77x |
| USNET | 0.0195 ms | 0.3484 ms | 0.1220 ms | 0.0446 ms | 0.37x | 2.28x |
| JPN48 | 0.0250 ms | 0.5138 ms | 0.1542 ms | 0.0476 ms | 0.31x | 1.91x |

All measurements are single worker, pure Python/NumPy, native kernel unset.
Initialization is excluded identically. The final method is within roughly 2x
of KSP and decisively in the same sub-0.05-ms order of magnitude.

## Interpretation

This is not simple behavioral distillation. The network does not predict the
Teacher's action ID. It learns a dense resource price map, while legal action
construction, path prior, masking, and argmin remain analytic. The
route-co-occurrence kernel explicitly preserves the Teacher's opportunity-cost
structure in compressed form.

The result also explains why the old NPK underperformed: its additive per-link
price field lost path-level conflict structure and its GNN/input construction
cost more online. The new method uses a single path-conditioned conflict
profile and a 2.6k-parameter tiny MLP, so the useful structure is kept
without carrying the exact probe machinery into deployment.

## Artifacts

- Protocol: `PHASE_N1_PROTOCOL_LOCK.md`
- Latency gate: `artifacts/latency_ceiling_conflict_v1/LATENCY_CEILING.json`
- Teacher data: `artifacts/datasets_conflict_v1/`
- Frozen models: `artifacts/training_conflict_v1/`
- Five-seed pilot: `artifacts/frozen_pilot_5seed/FROZEN_PILOT_RESULTS.json`
- Ten-seed confirmation: `artifacts/confirmatory_10seed/CONFIRMATORY_RESULTS.json`

## Next decision

Promote Neural Opportunity as the **fast deployment variant** and keep Full
Direct as the **best-blocking reference**. Do not spend another round on DAgger
or a larger neural model under this protocol. The remaining 0.20 pp on
NSFNET/JPN48 is the confirmed price of deleting Shadow plus amortizing the
Opportunity computation, not an unmeasured training failure.
