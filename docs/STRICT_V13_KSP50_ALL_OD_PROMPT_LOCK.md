# Strict v1.3 vs KSP-FF K=50 Hops Prompt Lock

This document is the mandatory protocol lock for writing any prompt related to
Strict v1.3, KSP-FF K=50 hops, fixed-C, and all-OD diagnosis in this project.

Before writing or executing any new prompt for this line of experiments, read
this document first and explicitly preserve these definitions.

## 1. Current Target

The current diagnosis target is:

```text
fixed-C + all-OD dynamic traffic distribution
Strict v1.3 vs KSP-FF K=50 hops
```

The goal is to explain the action-level and trajectory-level difference between
Strict v1.3 and KSP-FF K=50 hops under all-OD traffic when C-side decisions are
deterministic and PPO-C is not used.

## 2. Fixed-C Does Not Mean Fixed-OD

Fixed-C means:

- Do not call PPO-C.
- Do not load a PPO-C checkpoint.
- Do not use PPO-C to select split/server.
- Do not call `_load_ppo_c`.
- Do not call `_select_c_action_from_obs`.
- Use a deterministic C-side mapping.

Fixed-C does not mean fixing a single source-destination pair.

The experiment must use all-OD traffic:

- `src_node` varies according to the project/default traffic distribution.
- `dst_node` varies according to the all-OD traffic distribution.
- Do not fix a single `src_node`.
- Do not fix a single `dst_node`.
- Do not fix a single `server_id` if doing so fixes `dst_node`.

If the environment defines:

```text
dst_node = env.mec.servers[server_id].node_id
```

then `server_id` cannot be a global constant in all-OD experiments.

Use:

```text
split_id = fixed_split_id
server_id = deterministic_server_for(sampled_dst_node)
```

If each node has one server, `server_id` is uniquely determined by `dst_node`.
If a destination node has multiple servers, use a fixed deterministic rule, such
as the smallest server id on that node.

Every report must verify:

- no PPO-C loaded
- no PPO-C action selected
- all-OD enabled
- fixed split id
- deterministic server mapping rule
- number of unique source nodes
- number of unique destination nodes
- number of unique OD pairs

If the unique OD pair count is close to 1, the run is fixed-OD and invalid for
the current all-OD target.

## 3. Strict v1.3 Definition

Strict v1.3 has exactly one meaning:

```text
PPO-R proposal-supported, full-state, common-future counterfactual RMSA reranker.
```

It is not a new PPO strategy.
It is not direct search over all legal RMSA actions.
It is not a mixed candidate-pool method.

Strict v1.3 online execution:

1. Build R observation from deterministic fixed-C mapping.
2. Frozen PPO-R scores legal RMSA actions.
3. Candidate set is only PPO-R legal Top-30.
4. Existing 25-dimensional candidate features are built.
5. The ranker scores each candidate once.
6. Execute the highest-scoring ranker candidate.

Strict v1.3 constraints:

- Candidate pool must be PPO-R legal Top-30 only.
- No KSP anchor.
- No heuristic filler.
- No diversity/random candidate.
- No all-legal candidate pool.
- No E1-only gate.
- `ranker_gate=all`.
- E=0 and E=1 both call the ranker.
- Online inference does not run H=5 rollout.
- If there are no PPO-R candidates, fall back according to the existing evaluated
  code path and record the fallback.

Use the existing 25-dimensional feature definition. Do not silently redefine a
new feature set and still call it Strict v1.3.

Legacy E1-only checkpoints must be named:

```text
Legacy E1-trained gated baseline
```

They are not current Strict v1.3.

## 4. KSP-FF K=50 Hops Definition

KSP-FF K=50 hops has exactly one formal implementation:

```text
ksp_ff_highest_mod_action
```

It means distance-adaptive KSP-FF:

- `K_path=50`
- `path_sort_strategy="hops"`
- tie-break same-hop paths by km
- `block_sort_strategy="start_asc"`
- `max_blocks=10`
- scan paths in hops order
- choose the highest feasible modulation on each path
- choose the First-Fit spectrum block

Never use the following as the formal KSP-FF K=50 hops baseline:

- `ksp_ff_action`
- naive flat First-Fit
- BPSK-first baseline
- K=5
- 5km
- default KSP
- legacy KSP-FF

Every prompt must spell out "KSP-FF K=50 hops" and must explicitly say that the
formal implementation is `ksp_ff_highest_mod_action`.

## 5. Paired Comparison Requirements

Strict v1.3 and KSP-FF K=50 hops must be compared with paired conditions:

- same seed
- same request trace
- same all-OD traffic matrix
- same arrival sequence
- same holding-time sequence
- same demand sequence
- same deterministic C mapping
- independent environment copies starting from the same initial state

Recommended default: 20 seeds for diagnosis. Larger seed counts can be used for
final reporting.

## 6. Required Per-Request Diagnostics

Record at least:

- seed
- request id
- src node
- dst node
- split id
- server id
- method
- success / blocked
- block reason
- flattened R action
- path index
- modulation index
- block index
- path hops
- path length km
- required FS
- block start
- block size
- block waste
- free ratio before action
- largest free block before action
- fragmentation before action
- `phi_spec` before action
- legal R action count
- whether KSP action is in Strict v1.3 PPO-R Top-30
- rank of KSP action if contained
- ranker-selected candidate rank in PPO-R Top-30
- pair outcome type:
  - both success
  - both block
  - strict win
  - ksp win

## 7. Required Outputs

For all-OD fixed-C diagnosis, write outputs under a clearly named directory such
as:

```text
sa_hmarl/experiments/strict_v13_vs_ksp_ff_k50_hops_all_od_fixed_c_diagnosis/
```

Required files:

- `DIAGNOSIS.json`
- `DIAGNOSIS.md`
- `NEXT_STEP_DECISION.md`
- `per_seed_summary.csv`
- `paired_trace_sample.jsonl.gz`

## 8. Required Analysis Questions

`DIAGNOSIS.md` must answer:

1. Was the run truly all-OD?
2. Was PPO-C completely absent?
3. What deterministic C mapping was used?
4. Which method has lower blocking: Strict v1.3 or KSP-FF K=50 hops?
5. What is the blocking-rate gap in percentage points?
6. What are the paired per-seed blocking rates?
7. How many strict-win and ksp-win requests occurred?
8. Which block reasons explain the gap?
9. Does Strict v1.3 choose longer paths?
10. Does Strict v1.3 choose different modulation levels?
11. Does Strict v1.3 trade current block waste for lower fragmentation?
12. Does KSP-FF K=50 hops suffer from start-ascending First-Fit congestion?
13. Is the KSP action often absent from PPO-R Top-30?
14. Is the failure mainly proposer coverage, ranker ordering, label mismatch, or
    an action-insensitive bottleneck?

`NEXT_STEP_DECISION.md` must choose the next action:

- If Strict v1.3 is better, analyze path/mod/block contribution to spectrum
  state management.
- If KSP-FF K=50 hops is better, analyze PPO-R Top-30 coverage, ranker regret,
  and same-state counterfactual forks.
- If the gap is near zero, run load sweep or topology sweep.
- If PPO-C was called or OD collapsed to one pair, mark the result invalid and
  rerun.

## 9. Parallel Execution Requirement

When running a batch of independent experiments, use parallel execution to
increase CPU utilization.

Recommended approach:

- Split seeds into independent jobs.
- Run multiple seed jobs concurrently, bounded by available CPU cores and memory.
- Avoid oversubscribing so heavily that each process becomes I/O-bound or causes
  memory pressure.
- Use a stable output directory per seed or per worker.
- Merge outputs only after all workers finish.
- Make the merge step deterministic and idempotent.
- Preserve paired comparison within each seed. Do not split Strict v1.3 and
  KSP-FF K=50 hops into incompatible traces.

Suggested wording for prompts:

```text
Please run independent seeds in parallel where safe, with a worker count matched
to available CPU cores and memory. Keep paired Strict v1.3 and KSP-FF K=50 hops
comparisons within the same seed/trace. Write per-seed outputs separately and
merge them deterministically after all workers complete.
```

## 10. First Line For Future Prompts

Every future prompt for this experiment family should start with:

```text
Before writing or executing this experiment, read
docs/STRICT_V13_KSP50_ALL_OD_PROMPT_LOCK.md and obey it exactly.
```
