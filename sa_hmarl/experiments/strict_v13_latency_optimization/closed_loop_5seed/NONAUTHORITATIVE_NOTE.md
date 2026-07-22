# Non-authoritative Historical Comparison

This directory is an optimized Strict v1.3 versus heuristic rerun. It is not
the exact reference-versus-optimized regression result.

The old multi-topology pilot has a different evaluator code hash and its exact
untracked source snapshot cannot be reconstructed. Although checkpoint,
topology, request configuration, and request hashes agree, its warmup spectrum
trajectory is not byte-for-byte identical to the current implementation.

Use `../closed_loop_regression/CLOSED_LOOP_REGRESSION.json` for the authoritative
exactness result. That paired run compares the current reference and optimized
selectors from identical initial conditions and reports zero action, outcome,
and blocked-request identity mismatches across 90,000 evaluated requests.
