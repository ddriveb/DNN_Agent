# Invalid Results Notice — Strict v1.3 Multi-Topology C-side Verified (v1)

**Status:** INVALID / SUPERSEDED

The results stored under this directory were produced with a buggy evaluator
(`sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_verified.py`,
pre-fix revision) and must **not** be used for paper tables, v1.35 decisions,
or any conclusion about Strict v1.3 performance.

## Root cause

The evaluator's `_run_episode()` loop iterated over the externally generated
`requests` list but only advanced the environment's `event_queue` when
`env.step()` was called.  When the C-side or R-side had no legal action, the
loop executed `continue` **without** consuming the current request.  Therefore,
starting from the first no-valid-action occurrence, every subsequent iteration
used `req_{t+1}` to build the observation/action while `env.step()` still
processed `req_t`.  The request stream, selected path, destination node,
modulation, and action became permanently desynchronized.

The same bug affected the warmup phase, so even episodes whose formal
statistics reported `c_no_valid_action=0` could already be desynchronized.

## Direct evidence in the old results

COST239 seed 3030 examples:

- `ppo_c + ksp_ff_highest` reported **515** `modulation_reach` failures.
- `ppo_c + strict_v13` reported **193** `invalid_path` failures.
- `ppo_c + strict_v13` reported **1170** `modulation_reach` failures.

The formal KSP-FF K=50 hops selector (`ksp_ff_highest_mod_action`) chooses
only from the legal mask and only reachable modulations.  A correctly
synchronized evaluation cannot produce hundreds of `invalid_path` or
`modulation_reach` failures for this baseline.  These counts are a direct
consequence of request/action misalignment.

## Consequences

- The previous 120-cell full results and the previous `Category E`
  classification are invalid.
- They are superseded by the v2 re-run under
  `sa_hmarl/experiments/v13_strict_multitopology_cside_verified_v2/`.
- This directory and its contents are preserved only for auditability.

## Remediation

A fixed evaluator was implemented with:

1. `SMDPEnv.reject_next_request(expected_req_id, reason)` to safely consume
   requests in hard-blocking branches.
2. Per-loop queue-head assertions (`req_id`, `arrival_time`) and per-loop
   queue-length conservation checks.
3. Mandatory R-mask check before calling PPO-R / KSP-FF / Strict ranker.
4. Hard validation of selected R actions against the mask, required FS,
   modulation reach, and block size.
5. Fixed E=0/E=1 definition from PPO-R Top-30 legal candidates (not
   `split_id`).
6. Consistent `delta = baseline - strict` statistical direction and
   independent failure-decomposition columns.

All results used for v1.35 decisions must come from the v2 directory.
