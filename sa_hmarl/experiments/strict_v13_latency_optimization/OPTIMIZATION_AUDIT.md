# Optimization Audit

## Removed work

1. PPO-R action features were previously built once for Top-30 proposal and a second time for Ranker base features. The optimized selector reuses the first matrix.
2. The legacy Ranker feature loop recomputed request-level field/context values for every candidate. The optimized path computes them once and vectorizes the 30 candidate rows.
3. KSP-FF/FF-KSP candidate-coverage checks are controlled by an explicit flag and are excluded from production latency.

## Preserved semantics

- Stable legal Top-30 ordering is unchanged.
- Raw and normalized 25-d features are unchanged.
- Frozen PPO-R and Ranker checkpoint hashes are unchanged.
- Ranker scores and selected actions are unchanged.
- Non-default feature schemas retain the legacy per-candidate fallback.

## Verification

- 1,500 same-state checks: zero Top-30, feature, score, or action mismatch.
- 90,000 five-seed closed-loop requests: zero action, outcome, or blocked-identity mismatch.
