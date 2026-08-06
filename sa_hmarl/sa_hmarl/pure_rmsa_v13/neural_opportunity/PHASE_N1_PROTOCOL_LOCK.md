# Neural Opportunity v1 Protocol Lock

## Objective

Approximate the locked Opportunity-only Direct score offline, then deploy only
deterministic feasibility, cheap candidate features, and one shared neural
forward pass online. This is an amortized pricing experiment, not policy
distillation and not an RL policy.

## Locked online algorithm

- KSP pool: K=50, hops then km then path tuple.
- Candidate paths: first three feasible paths in the locked Direct order.
- Candidate starts: every legal contiguous start on those paths.
- Score: `path_rank + neural_opportunity(state, action)`.
- Tie break: score, path rank, start slot through stable candidate order.
- No Shadow pricer, compressed sketch, exact Opportunity pricer, rollout,
  neural fallback, or native C kernel online.
- One batched MLP call over at most three path rows. Each row contains the
  50-slot common-free map, 50-slot path-conditioned conflict-pressure profile,
  and ten
  static/request scalars. The shared MLP is `110 -> 16 -> 50` with ReLU and
  predicts every start price at once. The per-path occupancy-count profile was
  removed before Teacher labels were read because it duplicated common-free
  structure and made JPN48 latency topology-dependent. The initial global-mean
  profile then failed offline fidelity on all three topologies. It was replaced
  by a static route-co-occurrence conflict kernel before any DAgger or pilot:
  the online profile is one matrix product between the candidate's precomputed
  conflict weights and the current arc bitmap. Hidden
  width 16 was locked before Teacher labels were read, after width 24 missed
  the pre-training latency gate by 0.0007-0.0016 ms on two topologies.
- Deployment is pure Python/NumPy. Training uses PyTorch.

## Teacher and losses

The offline teacher is the locked `OpportunityOnlyDirectSelector` with the
topology-specific budget and beta from Phase A. It supplies the exact weighted
opportunity contribution for every legal candidate.

Training loss is locked before data inspection:

`L = Huber(price) + 0.5 * listwise_KL + 0.5 * pairwise_margin`.

The price target is `beta * normalized_opportunity`; path rank remains an
analytic term and is never learned.

## Seed isolation

- Train: 60001-60003.
- Validation: 60101.
- Smoke: 60201.
- Closed-loop pilot: 60301-60305.
- DAgger collection: 60401-60402, at most two rounds.
- Confirmatory: 23001-23010 only after the model is frozen. These seeds are
  reused solely to pair against the locked Phase-A Full Direct/Opportunity-only
  fact table; they remain forbidden for tuning.

No seed may move between splits. Formal seeds 23001-23010 and all prior
diagnosis/ablation seeds are forbidden for tuning.

## Gates

1. Latency ceiling before training: complete selector mean <=0.060 ms/request
   under single-worker pure-Python timing.
2. Offline: Top-1 recall >=80% and normalized regret <=0.10.
3. Pilot: student no more than +0.20 pp worse than Opportunity-only and retains
   at least 85% of Opportunity-only gain over paired KSP-FF K=50.
4. At most two DAgger rounds. Failure after round two is a STOP, not a request
   for a larger model.

After the untouched five-seed frozen pilot, no DAgger was run because the
student already retained at least 89% of Full Direct's KSP gain. The
confirmatory gate is: Neural beats paired KSP with CI below zero on all three
topologies, retains at least 85% of Full Direct's gain, is non-inferior to Full
Direct within +0.20 pp, and has mean selector latency <=0.060 ms.

Full Direct remains the performance reference. Opportunity-only's confirmed
0.16-0.19 pp loss on NSFNET/JPN48 is accepted as the deployment tradeoff and
must not be hidden by the neural comparison.
