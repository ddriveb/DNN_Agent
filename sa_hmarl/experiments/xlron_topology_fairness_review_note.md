# XLRON Topology Experiment Fairness Review Note

This note documents the experimental setup, key implementation logic, and
fairness caveats for the XLRON topology R-backend comparison.

The purpose is to let an external reviewer judge whether the comparison between
`v1.2` and `KSP-FF K=50 hops` is fair, and what is still missing before calling
it a complete three-way comparison with DeepRMSA.

## 1. Experiment Goal

We evaluate how the R-side backend affects blocking rate under the same fixed
C-side policy.

Compared R backends:

| Backend | Description |
|---|---|
| `v1.2` | Planner-distilled counterfactual R-ranker. |
| `KSP-FF K=50 hops` | Tuned heuristic baseline: 50 candidate paths, path order by hop count, first-fit spectrum assignment. |
| `DeepRMSA` | Not run in this batch because available checkpoints are topology-specific and incompatible with the four newly imported XLRON topologies. |

## 2. Fixed Experiment Configuration

The following settings are held constant across topologies and R backends unless
explicitly stated otherwise.

| Item | Value |
|---|---|
| C policy | `ppo_c` |
| C checkpoint | `sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt` |
| v1.2 ranker checkpoint | `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt` |
| Number of slots | `100` |
| Number of MEC servers | `4` |
| Split profile | `default3` |
| Arrival interval | `0.15` |
| Holding time | `4.0` to `10.0` |
| Request size | `5.0` to `30.0` MB |
| Deadline | `30.0` to `100.0` ms |
| Edge compute cost | `0.5` to `15.0` |
| Seeds | `3030,4040,5050,6060,7070` |
| Episodes per seed | `20` |
| Requests per episode | `80` |
| Requests per method/topology | `8000` |

Topologies:

| Topology name |
|---|
| `xlron_cost239_ptrnet_real` |
| `xlron_german17` |
| `xlron_nsfnet_deeprmsa` |
| `xlron_jpn48` |

## 3. Backend-Specific Settings

### v1.2

For `v1.2`, the main path configuration is:

| Item | Value |
|---|---|
| `k_paths` | `5` |
| Path order | `km` |
| Block sort strategy | `mixed` |

Online decision procedure:

1. PPO-C selects `(split_id, server_id)`.
2. The environment builds Agent-R observation for that C decision.
3. v1.2 enumerates legal R candidates or the checkpoint-specific candidate subset.
4. v1.2 builds candidate features.
5. Features are normalized by checkpoint statistics.
6. The ranker scores candidates.
7. The highest-score candidate is selected.

Implementation excerpt:

```python
def score_legal_actions(...):
    r_features, r_mask = agent_r.build_action_features(obs_r)
    legal = np.flatnonzero(np.asarray(r_mask, dtype=bool)).astype(int).tolist()
    if not legal:
        return [], np.empty((0,), dtype=np.float32)
    if len(legal) == 1:
        return legal, np.asarray([0.0], dtype=np.float32)

    candidate_actions = self._select_online_candidates(
        obs_r, r_features, legal, env.max_blocks, agent_r
    )

    features = build_r_ranker_feature_batch(
        env, req, obs_c, obs_r, r_features, candidate_actions,
        split_id, server_id, feature_names=self.feature_names,
    )
    normalized = (features - self.feature_mean) / self.feature_std
    with torch.no_grad():
        scores = self.model(
            torch.as_tensor(normalized, dtype=torch.float32, device=self.device)
        ).detach().cpu().numpy()
    return candidate_actions, np.asarray(scores, dtype=np.float32).reshape(-1)
```

Action selection:

```python
legal, scores = self.score_legal_actions(...)
if not legal:
    return self.empty_action
if len(legal) == 1:
    return int(legal[0])
return int(legal[int(np.argmax(scores))])
```

### KSP-FF K=50 hops

For the tuned heuristic baseline, the R-side observation/execution path set is
expanded only for the R backend:

| Item | Value |
|---|---|
| `k_paths` | `50` |
| Path order | `hops` |
| Block sort strategy | `start_asc` |

Implementation excerpt for backend-specific path configuration:

```python
def _r_backend_env_config(r_mode, args):
    if r_mode in ("ksp_ff_k50_hops", "v12_k50_hops"):
        return args.ksp_ff_k50_hops_k_paths, "hops", "start_asc"
    return args.k_paths, args.path_sort_strategy, args.block_sort_strategy
```

In each request, PPO-C is selected first under the main experiment path
configuration.  Then the R-side backend may override the R observation path set:

```python
env.k, env.path_sort_strategy, env.block_sort_strategy = _r_backend_env_config(
    r_mode, args
)
obs_r = build_agent_r_observation(env, req, split_id, server_id)
```

The tuned KSP-FF heuristic scans paths in the provided order, chooses the
feasible modulation requiring the fewest frequency slots on that path, then
selects the lowest-start valid block.

Implementation excerpt:

```python
def ksp_ff_highest_mod_action(obs):
    mask = np.asarray(obs["agent_r_mask"], dtype=bool)
    valid = np.flatnonzero(mask)
    if len(valid) == 0:
        return None

    num_paths = len(obs["candidate_paths"])
    num_mods = len(obs["mod_names"])
    num_blocks = len(mask) // (num_paths * num_mods)

    for path_idx in range(num_paths):
        best_mod = None
        best_req_fs = float("inf")
        for mod_idx in range(num_mods):
            req_fs = obs["required_fs_per_path_mod"][path_idx][mod_idx]
            if req_fs is None or req_fs <= 0:
                continue
            start = path_idx * num_mods * num_blocks + mod_idx * num_blocks
            end = start + num_blocks
            if not mask[start:end].any():
                continue
            if float(req_fs) < best_req_fs:
                best_req_fs = float(req_fs)
                best_mod = mod_idx

        if best_mod is None:
            continue

        start = path_idx * num_mods * num_blocks + best_mod * num_blocks
        blocks = obs["candidate_blocks_per_path_mod"][path_idx][best_mod]
        valid_blocks = []
        for block_idx in range(num_blocks):
            action_idx = start + block_idx
            if not mask[action_idx]:
                continue
            block_start = blocks[block_idx][0] if block_idx < len(blocks) else block_idx
            valid_blocks.append((block_start, block_idx, action_idx))
        if valid_blocks:
            valid_blocks.sort(key=lambda item: (item[0], item[1]))
            return int(valid_blocks[0][2])

    return None
```

### DeepRMSA

DeepRMSA was requested but not included in the result table because the currently
available checkpoints are topology-specific:

| Available DeepRMSA checkpoint family |
|---|
| `deep_rmsa_snap24_*` |
| `deep_rmsa_c_sensitive_mixed.pt` |

When attempting to run DeepRMSA on the imported XLRON topologies, the loader
raised:

```text
ValueError: DeepRMSA checkpoint topology mismatch
```

The relevant check is:

```python
ckpt_nodes = ckpt.get("num_nodes")
if ckpt_nodes is not None and ckpt_nodes != env.net.NUM_NODES:
    raise ValueError("DeepRMSA checkpoint topology mismatch")
```

Therefore, a fair DeepRMSA comparison requires retraining DeepRMSA separately on
each imported topology.

## 4. Topology Registration

The four XLRON topologies are registered as normal topology names in the project:

```python
TOPOLOGY_REGISTRY = {
    ...
    "xlron_nsfnet_deeprmsa": XLRON_NSFNET_DEEPRMSA_EDGES,
    "xlron_cost239_ptrnet_real": XLRON_COST239_PTRNET_REAL_EDGES,
    "xlron_german17": XLRON_GERMAN17_EDGES,
    "xlron_jpn48": XLRON_JPN48_EDGES,
}
```

The imported topology edge lists were relabeled to 0-based contiguous node IDs
to match the request generator and existing environment assumptions.

## 5. Commands Used

Template:

```bash
PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology TOPOLOGY_NAME \
  --methods ppo_c+v12,ppo_c+ksp_ff_k50_hops \
  --episodes 20 \
  --requests_per_episode 80 \
  --seeds 3030,4040,5050,6060,7070 \
  --output_json sa_hmarl/experiments/TOPOLOGY_NAME_s100_r_compare.json \
  --output_md sa_hmarl/experiments/TOPOLOGY_NAME_s100_r_compare.md
```

Concrete output files:

| Topology | Output markdown |
|---|---|
| `xlron_cost239_ptrnet_real` | `sa_hmarl/experiments/xlron_cost239_s100_r_compare.md` |
| `xlron_german17` | `sa_hmarl/experiments/xlron_german17_s100_r_compare.md` |
| `xlron_nsfnet_deeprmsa` | `sa_hmarl/experiments/xlron_nsfnet_deeprmsa_s100_r_compare.md` |
| `xlron_jpn48` | `sa_hmarl/experiments/xlron_jpn48_s100_r_compare.md` |

## 6. Results

Metric definitions:

| Metric | Meaning |
|---|---|
| `Blocking` | All failed requests, i.e. environment returns `success=False`. |
| `Raw empty` | Diagnostic: the raw Agent-C mask had no valid C action before the final step.  This is not a mutually exclusive failure category and should not be added to `NSB/Overload/Deadline/Other`. |
| `NSB` | `no_suitable_block`: after the C decision and R action, no continuous spectrum block can satisfy the required FS on the selected path/modulation.  This is our project shorthand for spectrum block shortage. |
| `Overload` | Compute-side server failure.  The saved report counted `server_overload`; the evaluation code has now been updated so future reports also count `server_saturated`. |
| `Deadline` | `deadline_infeasible`: total estimated delay violates the request deadline. |
| `Other` | Blocked requests not categorized as `NSB`, `Overload`, or `Deadline` in the compact saved aggregate.  This can include invalid action/path/modulation, modulation reach failure, `fs_too_large`, `allocation_failed`, older uncounted `server_saturated`, or other environment reasons. |

| Topology | R backend | Blocking | Raw empty | NSB | Overload | Deadline | Other | Delay mean/P95 | Decision mean/P95 |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `xlron_cost239_ptrnet_real` | v1.2 | **0.20%** | 0.20% | 0.00% | 0.20% | 0.00% | 0.00% | 11.060/20.027 ms | 10.890/16.479 ms |
| `xlron_cost239_ptrnet_real` | KSP-FF K50 hops | 3.95% | 0.00% | 0.00% | 0.00% | 0.00% | 3.95% | 10.294/19.427 ms | 17.818/24.355 ms |
| `xlron_german17` | v1.2 | **0.97%** | 0.97% | 0.27% | 0.70% | 0.00% | 0.00% | 13.317/23.363 ms | 14.597/22.793 ms |
| `xlron_german17` | KSP-FF K50 hops | 2.96% | 1.31% | 2.19% | 0.59% | 0.00% | 0.19% | 13.014/22.640 ms | 24.902/32.221 ms |
| `xlron_nsfnet_deeprmsa` | v1.2 | **3.09%** | 3.15% | 0.76% | 2.18% | 0.14% | 0.01% | 19.526/33.812 ms | 7.748/10.932 ms |
| `xlron_nsfnet_deeprmsa` | KSP-FF K50 hops | 9.55% | 2.24% | 2.25% | 1.40% | 0.06% | 5.84% | 18.555/32.276 ms | 20.250/26.999 ms |
| `xlron_jpn48` | v1.2 | **6.19%** | 6.19% | 4.59% | 1.05% | 0.51% | 0.04% | 16.066/28.934 ms | 28.237/43.844 ms |
| `xlron_jpn48` | KSP-FF K50 hops | 9.70% | 4.26% | 4.40% | 0.99% | 0.26% | 4.05% | 15.142/27.527 ms | 76.292/128.185 ms |

Important interpretation note:

```text
Raw empty is a diagnostic, not part of the mutually exclusive reason
decomposition.  The approximate decomposition of Blocking is:

Blocking ~= NSB + Overload + Deadline + Other
```

The COST239 KSP-FF row is the clearest example: its `3.95%` blocking is not
explained by `NSB`, `Overload`, or `Deadline` in the compact saved aggregate,
so those blocked requests are shown as `Other`.  A future rerun with the updated
evaluation code will also store raw `reason_counts`, allowing this `Other`
bucket to be split exactly.

## 7. Fairness Interpretation

This comparison is fair for the following claim:

```text
With the same fixed PPO-C policy and the same traffic/load configuration,
the transferred v1.2 R-ranker achieves lower blocking than the tuned KSP-FF
K=50 hops heuristic on the four imported XLRON topologies.
```

This comparison is not sufficient for the following stronger claims:

```text
v1.2 is universally better than DeepRMSA on these topologies.
```

DeepRMSA was not included because topology-specific checkpoints are missing.

```text
The reported results are the best possible C-R system on each topology.
```

PPO-C was fixed and not retrained per topology.

```text
KSP-FF was disadvantaged by a smaller path set.
```

KSP-FF was given a larger R action-space breadth (`K=50`) and hop-ordered paths.
The v1.2 run used the normal `k_paths=5`, `km`-ordered configuration.

## 8. Main Caveats

1. The C-side policy was trained for the original project setting and transferred
   to the XLRON topologies without retraining.
2. The v1.2 ranker was also transferred, not retrained per topology.
3. KSP-FF uses no learning and is evaluated directly on each topology.
4. KSP-FF was given the stronger `K=50 hops` configuration.
5. DeepRMSA needs per-topology retraining before a complete three-way comparison
   can be claimed.
