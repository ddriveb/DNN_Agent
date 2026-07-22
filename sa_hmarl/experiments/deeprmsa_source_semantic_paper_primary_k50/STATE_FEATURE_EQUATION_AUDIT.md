# State Feature Equation Audit

The source-semantic state vector uses the upstream DeepRMSA `model1` equations.

## Header

* Source one-hot: vector of length NODE_NUM with `src_node` set to 1.0.
* Destination one-hot: vector of length NODE_NUM with `dst_node` set to 1.0.

## Per-path block (repeated K=50 times)

For each candidate path the port first picks the highest-spectral-efficiency feasible
modulation (matching upstream reach thresholds).

| Feature | Equation | Upstream line | Port line |
|---------|----------|---------------|-----------|
| required_fs_norm | `(req_fs - 5.5) / 3.5` | DeepRMSA_Agent.py:395 | deep_rmsa_source_semantic_agent.py:73 |
| block_start_norm | `2 * (start - 0.5 * num_slots) / num_slots` | DeepRMSA_Agent.py:402 | deep_rmsa_source_semantic_agent.py:77 |
| block_size_norm | `(size - 8.0) / 8.0` | DeepRMSA_Agent.py:403 | deep_rmsa_source_semantic_agent.py:78 |
| total_avail_norm | `2 * (sum(block_sizes) - 0.5 * num_slots) / num_slots` | DeepRMSA_Agent.py:407 | deep_rmsa_source_semantic_agent.py:82 |
| mean_size_norm | `(mean(block_sizes) - 4.0) / 4.0` | DeepRMSA_Agent.py:408 | deep_rmsa_source_semantic_agent.py:83 |

## Padding

If `num_paths < k_path`, remaining per-path blocks are filled with `-1.0`.
If a path has no feasible modulation or no sufficiently large contiguous block, the whole
per-path segment is filled with `-1.0` (upstream DeepRMSA_Agent.py:380,392).
