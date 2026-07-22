# Mask Removal Audit

This audit verifies that the source-semantic port does **not** use a legal-action mask.

## Requirements

1. Action space cardinality = K * M = 50 * 1 = 50.
2. Policy outputs are always selectable.
3. Selecting an invalid path/block produces a blocked request and reward -1.
4. No re-sampling, fallback, or masking is applied.

## Evidence

* `DeepRMSASourceSemanticAgent.select_action` computes softmax over all 50 outputs
  and never calls `_build_action_mask`.
* The stored action mask is `np.ones(self.n_actions, dtype=bool)` so the A2C optimizer
  does not mask any logits.
* `store_transition` is called even when `_decode_to_sahmarl` returns `None`, storing
  reward -1 for that policy output.

## Unit test

`sa_hmarl/tests/test_deeprmsa_optical_k50.py::test_source_semantic_agent_does_not_mask_unavailable_action`
fills the network and confirms the agent still selects a policy output and records a blocking transition.
