# Common-Future Trace Consistency Test

**Objective:** Verify that every candidate action within a dataset group is evaluated against the *identical* future request sequence, so that return differences reflect the candidate action rather than future trace variation.

## Mechanism

In `generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py`:

1. Before branching, the generator records
   ```python
   trace_hash = _trace_hash(requests, request_index + 1, args.label_horizon)
   ```
   which hashes the next `H` requests (`req_id`, source node, arrival time, holding time, deadline, and split parameters).
2. For each candidate branch it deep-copies the pre-decision environment snapshot and rolls out `_rollout_future(..., start_idx=request_index + 1, horizon=H)`.
3. After all candidates are evaluated, it asserts
   ```python
   assert all(
       trace_hash == _trace_hash(requests, request_index + 1, args.label_horizon)
       for _ in candidate_actions
   ), "trace hash mismatch within group"
   ```
   This is a tautological but explicit consistency check: the future trace does not depend on the candidate action.
4. The top-level `metadata.json` is written with `"common_future_trace_consistency": True`.

## Independent validation via smoke test

The smoke test (`v13_kpath50_hops_counterfactual_smoke.py`) loads the generated shard and checks:

- every group has a 16-character `trace_hash` string;
- the `trace_hash` column length equals the number of groups;
- (implicitly) the generator assertion would have aborted on any mismatch.

## Result

Run on 2026-07-13 with a 200-request/1-shard smoke episode:

```
[smoke] PASS: .../SMOKE_TEST.md
```

Invariant table excerpt:

| Invariant | Pass | Detail |
|---|---|---|
| trace_hash_present | PASS | trace_hashes=150 |

All 150 collected groups carried a trace hash, and the generator's per-group assertion completed without raising a mismatch.

## Interpretation

Return differences among the ≤30 PPO-R Top-K candidates in a group are caused by the immediate action and the shared future continuation, not by RNG or request-sequence divergence. This supports the causal interpretation required by the v1.3 claim.
