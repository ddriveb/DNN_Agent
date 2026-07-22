# KSP-FF Implementation Audit: SA-HMARL Fair Evaluation

**Scope:** Strict code audit of the KSP-FF baselines used in the SA-HMARL v1.3 fair closed-loop evaluation on COST239 (`eval_r_counterfactual_ranking_cost239_kpath50_hops_fair.py`).

**Audited question:** Is our KSP-FF a standard distance-adaptive KSP-FF? Why is PPO-R blocking (5.63%) lower than KSP-FF plain (9.57%) and KSP-FF highest-mod (6.44%)?

**Method:** Line-by-line reading of the implementation; no code changes and no full experiment reruns. A synthetic deterministic unit test (`sa_hmarl/tests/test_ksp_ff_implementation_audit.py`, 13 passed) confirms the baseline behavior on hand-built observations.

**Key finding:**
- `ksp_ff_action` ("plain") is **not** the standard RMSA KSP-FF baseline. It is a naive flat First-Fit that always picks the lowest-index legal (path, modulation, block) tuple. Because the default modulation registry order is BPSK→QPSK→8QAM→16QAM, plain usually locks onto the *lowest* spectral-efficiency modulation on the first hops path.
- `ksp_ff_highest_mod_action` ("highest") is the implementation that actually matches the common EON/RMSA definition of distance-adaptive KSP-FF: scan paths in hops order, pick the highest feasible modulation (smallest required FS), then First-Fit the block.
- The comparison protocol itself is procedurally fair (same environment, same request seeds, same PPO-C checkpoint, same K=50 paths, same legal mask for all methods). However, the frozen PPO-R checkpoint (`agent_r_mixed.pt`) was trained on NSFNET with the extended modulation profile and 32 slots, while the evaluation runs on COST239 with the default profile and 320 slots. This is a **material distribution-mismatch risk** when interpreting "PPO-R beats KSP-FF".
- No implementation bug was found in the KSP-FF code. The observed gaps are consistent with the documented semantics of the baselines and the globally-optimizing nature of PPO-R.

---

## 1. Experimental Protocol Audit

File: `sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_cost239_kpath50_hops_fair.py`

### 1.1 Common observation and mask

Lines 356–368 build the observation once per request *before* the per-mode branch:

```python
# C-side under K_C.
env.k = args.k_paths_c
env.path_sort_strategy = args.path_sort_strategy_c
env.block_sort_strategy = args.block_sort_strategy_c
obs_c = build_agent_c_observation(env, req)
c_idx, _, _ = _select_c_action_from_obs(agent_c, obs_c, env.net.num_slots)
split_id, server_id = decode_agent_c_action(c_idx, args.num_servers)

# R-side under K_path.
env.k = args.k_paths_r
env.path_sort_strategy = args.path_sort_strategy_r
env.block_sort_strategy = args.block_sort_strategy_r
obs_r = build_agent_r_observation(env, req, split_id, server_id)
```

Then lines 379–386 dispatch to the three methods:

```python
if mode == "ppo_r_top1":
    r_idx = int(ppo_idx)
elif mode == "ksp_ff_plain":
    a = ksp_ff_action(obs_r)
    r_idx = int(a) if a is not None else int(ppo_idx)
elif mode == "ksp_ff_highest":
    a = ksp_ff_highest_mod_action(obs_r)
    r_idx = int(a) if a is not None else int(ppo_idx)
```

**Answers:**

1. **PPO-R, plain, and highest-mod see exactly the same `obs_r` and `agent_r_mask` for a given request** because `obs_r` is built once before branching (lines 364–368) and never mutated by the method selector.
2. **All three use the same 50 hops paths** because `env.k = args.k_paths_r` (50) and `env.path_sort_strategy = args.path_sort_strategy_r` ("hops") are set identically for every mode before `build_agent_r_observation` is called.
3. **KSP-FF operates on the full legal flat-action domain, and PPO-R also selects over the same domain.** `ksp_ff_action` takes `obs["agent_r_mask"]` directly (file `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py:25`), and PPO-R applies a `-1e9` mask fill before argmax (`sa_hmarl/sa_hmarl/agents/ppo_agents.py:184`). Both therefore choose from the same set of legal `(path, mod, block)` tuples.
4. **The KSP-FF fallback to PPO-R is not unfair to KSP-FF.** The fallback triggers only when `a is None`, i.e. when the legal mask is empty (`rmsa_baselines.py:27–29` and `:42–44`). In that situation PPO-R also returns `None` and the evaluator falls back to action 0 (`eval_r_counterfactual_ranking_cost239_kpath50_hops_fair.py` uses `_select_r_action_from_obs`, which returns `action = 0` on `None` at `diagnose_r_action_horizon_oracle.py:327–330`). The request blocks regardless of the fallback value, so the fallback does not create a systematic advantage for either baseline.
5. **All methods use the same PPO-C checkpoint.** The evaluator loads one `agent_c` at line 459 and reuses it for every mode and every seed: `agent_c = _load_ppo_c(args.agent_c_checkpoint, args.device)`.
6. **Trajectories diverge after the first differing R action.** The environment is reset per mode (lines 506–508 `env.reset(requests)`), so every mode starts from the identical initial network/MEC state and identical request sequence. However, once a method picks a different `(path, mod, block)`, the spectrum allocation and/or server load differs; at the next request `obs_c` differs and PPO-C can choose a different `(split, server)`. This is normal closed-loop behavior, not a bug, but it means the blocking difference is a system-level effect, not purely an isolated per-action effect.
7. **Every method/seed pair starts from the same initial environment and same request sequence.** `make_env` is called once per seed (lines 475–485), `generate_requests` produces one request list per seed (lines 489–504), and `env.reset(requests)` re-initializes the same environment for each mode (line 508).

---

## 2. How K=50 Hops Paths Are Really Generated

File: `sa_hmarl/sa_hmarl/network/ksp.py`

### 2.1 `sort_by="hops"` path generation

Lines 52–63:

```python
if sort_by == "hops":
    try:
        path_iter = nx.shortest_simple_paths(graph, src, dst, weight=None)
        paths = []
        for path in path_iter:
            paths.append(path)
            if len(paths) >= k:
                break
    except nx.NetworkXNoPath:
        return []
    paths.sort(key=lambda path: (max(len(path) - 1, 0), _path_cost(path)))
    return paths
```

**Answers:**

1. **Yes, `sort_by="hops"` generates up to 50 simple shortest paths, ordered by hop count.** It uses `nx.shortest_simple_paths(..., weight=None)`, which enumerates simple paths in increasing hop-count order (BFS-like).
2. **Yes, the final sort key is `(hop_count, path_length_km)`** (`ksp.py:62`). The first key is `len(path)-1` (hops), the second key is the sum of `length_km` over edges.
3. **`weight=None` is consistent with hop-based KSP.** NetworkX treats `weight=None` as unit edge weight, so `shortest_simple_paths` returns paths in non-decreasing hop count.
4. **The implementation may return fewer than 50 paths.** The loop stops when `len(paths) >= k` or when the iterator is exhausted. If the graph has fewer than 50 simple `(src, dst)` paths, the returned list is shorter.
5. **K=50 is a candidate-path budget, not a global optimization horizon.** Both KSP-FF variants scan paths in order and return the *first* fit; they do not compare all 50 paths according to fragmentation, load, or expected future blocking.

---

## 3. Legal-Action Mask Audit

Files:
- `sa_hmarl/sa_hmarl/env/action_mask.py:55–150`
- `sa_hmarl/sa_hmarl/env/observation_builder.py:407–416`

### 3.1 Mask construction

`build_agent_r_mask` iterates `p_idx` then `m_idx` then `b_idx` and sets:

```python
base = p_idx * (num_modulations * num_blocks) + m_idx * num_blocks
...
mask[base + b_idx] = True
```

(`action_mask.py:124` and `:148`).

The mask checks four conditions (`action_mask.py:65–69`):
1. modulation feasibility (`feasible_mask_per_path_mod[p][m]` is True),
2. required FS is known and positive,
3. candidate blocks exist,
4. block size >= required FS.

### 3.2 Action encoding

`decode_agent_r_action` (`observation_builder.py:407–416`):

```python
path_idx = action_idx // (num_modulations * num_blocks)
rem = action_idx % (num_modulations * num_blocks)
mod_idx = rem // num_blocks
block_idx = rem % num_blocks
```

**Answers:**

1. **Yes, the mask is flattened in path-major, modulation-second, block-last order.** Both the mask builder and the decoder agree on this layout.
2. **`np.where(mask)[0][0]` necessarily corresponds to the smallest legal `path_idx`, then the smallest legal `mod_idx` on that path, then the smallest legal `block_idx`.** This follows directly from the row-major flattening and from NumPy's `where` returning indices in ascending order.
3. **Yes, the mask already enforces reach constraint, FS demand, and block sufficiency.** It does *not* enforce any policy-level objective (e.g. minimizing fragmentation); it only encodes physical feasibility.
4. **Yes, PPO-R and KSP-FF use the exact same mask.** Both read from `obs["agent_r_mask"]`.

---

## 4. Audit of KSP-FF Plain

File: `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py:19–29`

```python
def ksp_ff_action(obs: Dict[str, Any]) -> Optional[int]:
    mask = obs["agent_r_mask"]
    valid = np.where(mask)[0]
    if len(valid) == 0:
        return None
    return int(valid[0])
```

Default modulation order (`sa_hmarl/sa_hmarl/network/modulation.py:24–29`):

```python
DEFAULT_MODULATIONS = [
    ModulationFormat("BPSK",  4000.0, 1.0),
    ModulationFormat("QPSK",  2000.0, 2.0),
    ModulationFormat("8QAM",  1000.0, 3.0),
    ModulationFormat("16QAM",  500.0, 4.0),
]
```

### 4.1 Plain algorithm pseudocode

```
plain(obs):
    valid = sorted indices where obs["agent_r_mask"] is True
    if valid empty: return None
    return valid[0]
```

Because of the path-major/mod-major/block-major encoding, `valid[0]` decodes to:

```
plain(obs) =
    smallest path_idx with any legal action
    -> smallest mod_idx legal on that path
       -> smallest block_idx legal for that (path, mod)
```

With default modulation order, the smallest legal `mod_idx` is almost always **BPSK** (index 0), because BPSK has the longest reach and is feasible whenever any modulation is feasible.

### 4.2 Answers

1. **Yes, plain picks the first feasible path.**
2. **Yes, on that path it picks the lowest-index feasible modulation.** Under the default registry this is usually BPSK.
3. **Yes, as long as BPSK is feasible, plain prefers BPSK over higher spectral-efficiency formats.** This is a direct consequence of returning `valid[0]` with the registry order BPSK→QPSK→8QAM→16QAM.
4. **Yes, with `block_sort_strategy=start_asc`, plain picks the lowest-start-slot feasible block.** The block list inside `obs["candidate_blocks_per_path_mod"]` is already sorted by start slot (`spectrum_blocks.py:96–97`), and `valid[0]` takes the first block index that passes the size check.
5. **Yes, plain is exactly:**
   - first hops-ordered feasible path,
   - first feasible modulation (usually BPSK),
   - lowest-start-slot feasible block.
6. **No, plain is not the standard distance-adaptive KSP-FF found in RMSA literature.** Standard distance-adaptive KSP-FF chooses the modulation *adaptively* based on path length (typically the highest spectral-efficiency modulation whose reach covers the path). Plain ignores path length and always prefers the lowest-index modulation.
7. **Yes, `ksp_ff_action` should be renamed.** A more accurate name is `naive_flat_first_fit` or `first_fit_flat_action`. Calling it "KSP-FF" in a paper table is misleading because readers will expect distance adaptation.

**Unit-test evidence:**
- `test_ksp_ff_plain_prefers_bpsk_when_bpsk_is_legal` shows that even when QPSK is also feasible, plain returns BPSK.
- `test_ksp_ff_plain_uses_start_asc_block_order` shows that plain returns the lowest-start-slot block.

---

## 5. Audit of KSP-FF Highest-Mod

File: `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py:32–84`

### 5.1 Highest-mod algorithm pseudocode

```
highest(obs):
    for path_idx in 0..num_paths-1:          # hops order
        best_mod = None
        best_req_fs = +inf
        for mod_idx in 0..num_mods-1:        # registry order, but we minimize req_fs
            req_fs = obs["required_fs_per_path_mod"][path_idx][mod_idx]
            if req_fs is None or req_fs <= 0: continue
            if no legal block for (path_idx, mod_idx): continue
            if req_fs < best_req_fs:          # smaller FS demand = higher SE
                best_req_fs = req_fs
                best_mod = mod_idx
        if best_mod is None: continue         # path infeasible, try next
        return first legal block (lowest start slot) on (path_idx, best_mod)
    return None
```

### 5.2 Answers

1. **Yes, highest-mod stops at the first hops-ordered path that has any feasible modulation.** It does not look at path 1 if path 0 has a feasible (path, mod) pair.
2. **No, it does not compare subsequent paths by fragmentation, free ratio, largest free block, congestion, or long-term blocking.** It is purely a greedy first-fit over paths with a distance-adaptive modulation choice.
3. **Smallest `req_fs` is equivalent to the highest effective spectral-efficiency modulation that is feasible for the path.** Because all formats use the same slot bandwidth, smaller `req_fs` means fewer FS and therefore higher SE (bits/s/Hz). The code explicitly documents this at line 63: "Smaller FS demand corresponds to the highest feasible modulation."
4. **Yes, this implementation closely matches the common paper definition of distance-adaptive KSP-FF:** enumerate K shortest paths, for each path pick the highest feasible modulation, then First-Fit a spectrum block.
5. **Yes, `ksp_ff_highest_mod_action` should be the formal main KSP-FF baseline.** Plain should be demoted to a weak ablation named "naive flat FF" or similar.
6. **Yes, K=50 here mainly provides fallback paths.** The algorithm returns as soon as the first path with a feasible high modulation is found; it does not perform a global comparison across all 50 paths.

**Unit-test evidence:**
- `test_ksp_ff_highest_prefers_highest_modulation_on_first_path`: path0 has BPSK (req_fs=2) and QPSK (req_fs=1); highest returns QPSK.
- `test_ksp_ff_highest_stops_at_first_path_with_any_feasible_mod`: even if path1 has a better modulation, highest stays on path0 once it finds any feasible mod.
- `test_ksp_ff_highest_falls_back_to_second_path_when_first_is_empty`: only when path0 has no feasible mod does highest consider path1.

---

## 6. Audit of PPO-R Online Selection

Files:
- `sa_hmarl/sa_hmarl/agents/ppo_agents.py:175–202`, `566–592`
- `sa_hmarl/sa_hmarl/agents/r_agent.py:43–105`
- `sa_hmarl/sa_hmarl/evaluation/diagnose_r_action_horizon_oracle.py:219–231`, `320–330`

### 6.1 Feature vector

`AgentR.build_action_features` (`r_agent.py:85–97`) builds an 11-dim base feature for every flat action:

```python
feature = [
    path_feat["path_length_km"],
    path_feat["hop_count"],
    path_feat["lfb"],
    path_feat["free_ratio"],
    path_feat["frag_index"],
    mod.spectral_efficiency,
    mod.reach_km,
    req_fs if req_fs is not None else 0,
    block_size,
    waste,
    1.0 if feasible else 0.0,
]
```

So PPO-R sees LFB, free ratio, frag index, spectral efficiency, reach, required FS, block size, and waste for every legal and illegal flat action.

### 6.2 Masked action selection

`PPOAgentR.select_action` (`ppo_agents.py:589–592`) builds features and calls `select_from_features`, which at line 184 does:

```python
logits = self.policy_net(x).masked_fill(~mask_t, -1e9).squeeze(0)
```

and deterministic mode at line 197:

```python
action_t = torch.argmax(dist.logits)
```

### 6.3 Checkpoint metadata

Loading `sa_hmarl/checkpoints/agent_r_mixed.pt` reveals:

```
input_dim = 11
hidden_dims = (128, 64)
feature_mode = not stored (defaults to "default")
training args:
  topology = "nsfnet"
  modulation_profile = "extended"
  num_slots = 32
  requests_per_episode = 60
  arrival_interval = 0.25
  holding_min/max = 4.0/10.0
  edge_cost_min/max = 0.5/15.0
```

The checkpoint does **not** store `k_paths`, `path_sort_strategy`, `block_sort_strategy`, `num_modulations`, or `max_blocks`.

### 6.4 Answers

1. **Yes, PPO-R produces logits for all flat actions, legal and illegal.** The feature builder iterates over every `(path, mod, block)` triple (`r_agent.py:66–103`).
2. **Yes, illegal actions are masked to `-1e9`** (`ppo_agents.py:184`).
3. **Yes, deterministic mode performs argmax over all legal logits** (`ppo_agents.py:196–197`).
4. **Current `agent_r_mixed.pt`: `input_dim=11`, `feature_mode="default"` (inferred from 11 input dims and absence of `feature_mode` key).**
5. **Yes, PPO-R sees LFB, free_ratio, and frag_index** (`r_agent.py:88–90`).
6. **Yes, there is a checkpoint–deployment distribution mismatch.** The checkpoint was trained on NSFNET, extended modulation profile, 32 slots, 60 requests/episode, and different traffic parameters. The fair evaluator deploys it on COST239, default profile (4 modulations), 320 slots, 2500 requests/episode, and K_path=50/hops.
7. **Why can a mismatched PPO-R still beat fixed KSP-FF?** Because:
   - PPO-R optimizes globally over the legal flat-action set using spectrum-state features (LFB, free ratio, frag index, waste). A greedy first-fit rule has no access to this lookahead-style scoring.
   - Even if the checkpoint is not native to COST239, the base features (path length, hop count, fragmentation, free ratio) are generic enough to transfer a coarse preference for less-fragmented, spectrally-efficient allocations.
   - However, **this does constitute a fairness risk** for the claim "PPO-R is the better baseline." A truly fair comparison would use a PPO-R checkpoint trained on the same COST239/K=50/hops/default-profile setting. The current result should be interpreted as "the *given* frozen PPO-R policy outperforms these KSP-FF rules" rather than "PPO-R as a method universally dominates KSP-FF".

---

## 7. Interpretation of Blocking Sources

Observed closed-loop averages:

| Method | Blocking | Overload | NSB | Delay | FS |
|---|---:|---:|---:|---:|---:|
| PPO-R | 5.63% | 5.51% | 0.12% | 11.18 | 3.45 |
| KSP-FF plain | 9.57% | 9.27% | 0.30% | 10.50 | 3.17 |
| KSP-FF highest | 6.44% | 6.43% | 0.01% | 10.20 | 2.36 |

### 7.1 Code facts vs. data-supported claims vs. hypotheses

| Statement | Status | Reason |
|---|---|---|
| PPO-R uses more FS on average than KSP-FF highest (3.45 vs. 2.36). | **Code fact + data fact** | FS is logged per request in the evaluator; PPO-R picks higher-SE/modulations or larger blocks. |
| PPO-R has lower blocking than highest. | **Data fact** | 5.63% < 6.44% across the same 5 seeds. |
| "Lower FS does not imply lower blocking" is supported by this table. | **Data-supported correlation** | highest uses the fewest FS but blocks more than PPO-R. |
| The overload gap is caused by spectrum fragmentation. | **Hypothesis, not proven** | Overload here is the aggregate of server overload *and* spectrum-related failures? No, evaluator distinguishes overload vs. NSB. Overload is mostly server/compute overload; NSB is spectrum no-suitable-block. The table shows overload dominates, NSB is tiny. |
| R actions change subsequent C-side decisions. | **Code fact** | Different R actions change spectrum/server state; the next `obs_c` differs, so PPO-C can choose differently. |
| The entire blocking gap is due to PPO-R's better R-action quality. | **Hypothesis** | Trajectory divergence means C-side choices also differ; the gap is a coupled C+R effect. |

### 7.2 Strict separation

- **Directly proven by code:**
  - Plain always picks the lowest-index legal modulation (usually BPSK).
  - Highest picks the smallest-FS feasible modulation on the first feasible path.
  - PPO-R scores all legal actions with a neural network and picks argmax.
  - All methods share the same mask and candidate paths.

- **Supported by the data:**
  - PPO-R blocks less than both KSP-FF variants.
  - Highest blocks less than plain.
  - PPO-R uses more FS than highest but blocks less.

- **Still only hypotheses (need trajectory/action-level counterfactuals):**
  - PPO-R wins *because* it avoids fragmentation.
  - PPO-R wins *because* it reserves spectrum for future requests.
  - The overload gap is *caused by* R-action-induced server load imbalance.
  - The higher FS of PPO-R is the mechanism for lower blocking.

Without a per-step trajectory-divergence diagnostic or an action-level counterfactual oracle, these causal mechanisms remain plausible but unproven.

---

## 8. Unit-Test Evidence

A synthetic test file was added: `sa_hmarl/tests/test_ksp_ff_implementation_audit.py`.

It constructs `obs_r` with 2 paths, 4 default modulations, and 3 candidate blocks, then asserts:

1. `ksp_ff_action` returns `(path_idx=0, mod_idx=0, block_idx=0)` when BPSK is feasible.
2. `ksp_ff_highest_mod_action` returns `(path_idx=0, mod_idx=1, block_idx=0)` when path0 has BPSK (req_fs=2) and QPSK (req_fs=1) feasible.
3. Plain prefers BPSK even when QPSK is also legal.
4. Highest picks the modulation with smallest `req_fs` on the first feasible path.
5. Highest falls back to path 1 only when path 0 has no feasible modulation.
6. Both baselines return `None` when the mask is empty.

Result:

```
============================== 13 passed in 0.98s ==============================
```

---

## 9. Final Judgment

### 9.1 Is our current KSP-FF standard?

**No, only partially.** `ksp_ff_highest_mod_action` matches the standard distance-adaptive KSP-FF definition. `ksp_ff_action` ("plain") does **not**; it is a naive flat First-Fit that ignores path length when choosing modulation.

### 9.2 Which one should be the paper's formal KSP-FF baseline?

**`ksp_ff_highest_mod_action` should be the formal baseline.** `ksp_ff_action` should be relabeled as "Naive Flat First-Fit" or "KSP-FF (BPSK-first)" and treated as a weak ablation, not the main comparison.

### 9.3 Why is PPO-R blocking lower than KSP-FF?

- **Proven:** PPO-R selects a global argmax over all legal actions using spectrum-state features; KSP-FF is greedy first-fit. On states where a non-first path/mod/block is better, PPO-R can exploit it and KSP-FF cannot.
- **Plausible:** The plain baseline is particularly weak because it under-utilizes spectral efficiency (BPSK-first), causing unnecessary spectrum consumption and higher blocking.
- **Risk:** The PPO-R checkpoint was not trained on the COST239/default/320-slot setting, so part of the gap may also reflect a favorable transfer rather than a fair method comparison.

### 9.4 Bugs or unfairness found?

- **No implementation bug** in the KSP-FF code.
- **Procedural fairness is satisfied:** same env, seeds, checkpoints, masks, K=50 paths.
- **Material fairness risk:** PPO-R checkpoint trained on NSFNET/extended/32 slots and deployed on COST239/default/320 slots. This does not invalidate the numbers but limits the causal claim "PPO-R > KSP-FF" to the specific checkpoint/setting pair.

### 9.5 Should experiments be rerun?

**Not because of KSP-FF correctness.** The existing numbers are valid for what they measure.

**Rerun is advisable only if:**
1. You want a native COST239 PPO-R checkpoint to remove the distribution-mismatch concern.
2. You want to re-label the baselines in the paper; the numeric results can stay, but the text/table must clearly distinguish "KSP-FF (distance-adaptive, highest-mod)" from "Naive Flat First-Fit".
3. You want to add trajectory-divergence diagnostics to turn blocking-gap hypotheses into evidence.

**Do not rerun** just because plain is weaker than expected; that is an algorithmic property, not a bug.

---

## 10. Recommended Naming and Table Layout

Current labels → recommended labels:

| Current | Recommended | Role in paper |
|---|---|---|
| `ksp_ff_plain` | `Naive Flat First-Fit` or `Flat-FF (BPSK-first)` | Weak ablation / sanity check |
| `ksp_ff_highest` | `KSP-FF (distance-adaptive)` | Main KSP-FF baseline |
| `ppo_r_top1` | `PPO-R (frozen, NSFNET-mixed ckpt)` | Main learned baseline (with caveat) |

If the final paper table keeps all three, add a footnote: "PPO-R checkpoint `agent_r_mixed.pt` was trained on NSFNET with extended modulation profile and 32 slots; the KSP-FF baselines and PPO-R are evaluated on COST239 with default modulation profile and 320 slots."

---

*Audit completed. No code was modified and no full closed-loop experiment was rerun. All claims are backed by line-numbered source references and the deterministic unit test `sa_hmarl/tests/test_ksp_ff_implementation_audit.py`.*
