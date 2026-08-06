# N1-FUSE0: fused feature-construction + first layer

Minimal, clean latency optimization for the deployed Neural Opportunity
selector.  No weight changes, no candidate/feature definition changes, no
cross-request dynamic caching.

## Mechanism

The N1 first layer is `h = ReLU(x_static W_s + x_free W_f + x_conflict W_c + b1)`
with `x` a materialized 110-dim per-candidate array.  FUSE0 never builds
that array:

- path-static features (rank, hops, km, centrality sum/max, overlap) are
  combined with their `W_s` rows **once at init** into `_static_matrix`
  (static route knowledge, not a dynamic cache);
- conflict weights per path are pre-stacked at init into `_cw_matrix`;
- per request, only the dynamic blocks are computed
  (`dyn | common_free | conflict-profile`) and multiplied by the fused
  `W1` block rows in one matmul;
- per-candidate Python loops are replaced by uint64 shift/popcount
  vectorization.

Summation order differs from the materialized matmul (float32
reassociation), so equivalence is established empirically: three
topologies × three fresh seeds × 10,000 decisions, **0 action
mismatches**, and identical per-request blocking.

## Contents

- `selector.py` — `FusedNeuralOpportunitySelector` (subclass of the N1
  selector; reuses its scan/sync; overrides only `select`).
- `run_fuse0_bench.py` — parity driver + fair latency benchmark
  (same traces/warmup, single worker, pure NumPy, includes the cached
  bit-parallel early-exit KSP-FF K=50 reference).
- `tests/` — fused-vs-original price equality on live states, small-scale
  action parity, static-table immutability.
- `artifacts/FUSE0_RESULTS.json` — full parity + latency record.

## Result summary

See `artifacts/FUSE0_RESULTS.json` and the report in the final summary.
Parity: 0/90,000 mismatches per topology pair across three topologies
(NSFNET/USNET/JPN48 × seeds 61201-61203 × 10,000 post-warmup decisions).
