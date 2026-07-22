# Horizon Semantics Audit

A horizon H is the total number of decision points evaluated in a fork.
* Composition: H = 1 (the first counterfactual action) + (H - 1) future requests continued under Formal KSP-FF.
* H=1 meaning: H=1 evaluates only the first action; zero future requests are processed.
* H=1 invariant: If both first actions are successful, both forks must report 0 blocks and ΔB_1 must be 0.
* v1 bug: v1 used `future_requests[:H]` instead of `future_requests[:H-1]`, violating the H=1 invariant.
* v2 correction: `_continue_with_ksp` now iterates over `future_requests[:max(0, horizon - 1)]`.
* Horizons evaluated: [1, 5, 20, 50, 100]
