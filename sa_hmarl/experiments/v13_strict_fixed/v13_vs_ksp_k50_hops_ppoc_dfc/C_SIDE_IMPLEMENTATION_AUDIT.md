# C-Side Implementation Audit

**Scope:** Identify and verify the canonical implementations of PPO-C and DF_C used in the SA-HMARL closed-loop evaluation, confirm they operate on the same action space and observation, and rule out ambiguous or alternative DF_C candidates.

**Conclusion:** A unique DF_C implementation was identified: `select_df` in `sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py:157–196`. It respects the Agent-C legal mask, uses the same `(split_id, server_id)` action space as PPO-C, and is deterministic. No blocker.

---

## 1. PPO-C

### 1.1 Class and checkpoint

- **Class:** `PPOAgentC` in `sa_hmarl/sa_hmarl/agents/ppo_agents.py`.
- **Checkpoint:** `sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt`.
- **SHA256:** `a5c9eb39bc33cf3b77a06d80e9c45315880dcb963ad9c3c95663eadaded9f2c8`.
- **Feature mode:** `r_feasibility_safe` (recorded in checkpoint args).
- **Input dim:** 29.
- **Hidden dims:** `(128, 64)`.
- **Activation:** `tanh`.

### 1.2 Selection function

PPO-C action selection in closed-loop evaluation is performed by `_select_c_action_from_obs` in `sa_hmarl/sa_hmarl/evaluation/diagnose_r_action_horizon_oracle.py:296–317`:

```python
def _select_c_action_from_obs(
    agent_c: PPOAgentC,
    obs_c: Dict[str, Any],
    num_slots: int,
) -> Tuple[int, np.ndarray, np.ndarray]:
    features, raw_mask = agent_c.build_action_features(obs_c)
    risk_mask = raw_mask.copy()
    ckpt_args = getattr(agent_c, "checkpoint_args", {}) or {}
    if ckpt_args:
        risk_kwargs = risk_kwargs_from_args(ckpt_args)
        min_valid = int(risk_kwargs.pop("min_valid_after_mask", 1))
        risk_mask = apply_agent_c_risk_mask(
            obs_c, risk_mask,
            num_slots_total=num_slots,
            min_valid_after_mask=min_valid,
            **risk_kwargs,
        )
    action, _, _ = agent_c.select_from_features(features, risk_mask, deterministic=True)
    if action is None:
        action = 0
    return int(action), raw_mask, risk_mask
```

Key properties:

- Builds features for every `(split_id, server_id)` candidate.
- Applies a risk mask derived from checkpoint args on top of the raw legal mask.
- Selects deterministically via argmax over the risk-masked logits.

### 1.3 Action space

Agent-C action encoding is split-major (`sa_hmarl/sa_hmarl/env/observation_builder.py:397–404`):

```python
split_id = action_idx // num_servers
server_id = action_idx % num_servers
```

The number of splits is 3 and the number of servers is 4, so the action space size is 12.

### 1.4 Training/deployment distribution note

Checkpoint args show that PPO-C was trained on:

- Topology: `xlron_cost239_ptrnet_real`
- Modulation profile: `default`
- `num_slots`: 100
- `requests_per_episode`: 80
- `holding_min/max`: 4.0 / 10.0
- `arrival_interval`: 0.15

This experiment deploys it on:

- `num_slots`: 320
- `requests_per_episode`: 6000
- `holding_min/max`: 20.0 / 30.0
- `arrival_interval`: 0.0625

Topology and modulation profile match, but traffic scale and slot count differ. This is a deployment-scale mismatch, not a topology mismatch. The checkpoint is used as-is because the task requires evaluating the existing PPO-C policy.

---

## 2. DF_C

### 2.1 Candidate implementations found

A repository search for `df_c` / `DF_C` / `select_df` returned two main functions:

1. `select_df(env, obs_c, mask)` in `sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py:157–196`.
2. `_select_df(env, obs_c, num_servers)` in `sa_hmarl/sa_hmarl/evaluation/eval_yin_style_offloading_sweep.py:87–93`.

### 2.2 Why `select_df` is the canonical DF_C

The multi-C-side strict v1.3 evaluator (`sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_fair.py`) explicitly maps `df_c` to `df` and calls `select_offloading_action("df", ...)`:

```python
baseline_map = {"df_c": "df", "rf_c": "rf", "wo_c": "wo", "greedy_c": "greedy", "iwd_c": "iwd"}
baseline_name = baseline_map.get(c_mode, c_mode)
if baseline_name in ("df", "rf", "wo", "greedy", "iwd"):
    mask = np.asarray(obs_c["agent_c_mask"], dtype=bool)
    action_idx = select_offloading_action(baseline_name, env, req, obs_c, mask)
```

`select_offloading_action` then dispatches to `select_df` (`offloading_baselines.py:1113–1114`). This is the implementation used by the existing heuristic-C comparison reports (`experiments/heuristic_c_v13_vs_kspff/FINAL_HEURISTIC_C_UNIFIED_REPORT.md` and `STRICT_CODE_AUDIT_REPORT.md`), which explicitly describe DF_C as "distance-first over legal `(split, server)` pairs."

The Yin-style `_select_df` (`eval_yin_style_offloading_sweep.py:87–93`) does **not** respect the Agent-C mask: it first picks the nearest server and then picks the best split for that server, even if the resulting `(split, server)` is illegal. That implementation is a literature-style baseline, not the SA-HMARL main-experiment DF_C. The strict evaluator does not import from `eval_yin_style_offloading_sweep.py`.

### 2.3 Canonical DF_C algorithm

Function: `sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py:157–196`

```python
def select_df(
    env,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
) -> Optional[int]:
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]
    src_node = obs_c["request_features"]["src_node"]

    distances = []
    for server_id in range(num_servers):
        server_node = env.mec.servers[server_id].node_id
        distances.append(_estimate_distance_km(env, src_node, server_node))

    best = int(valid[0])
    best_dist = distances[_decode_c(best, num_servers)[1]]
    best_delay = features[best]["edge_compute_ms"]
    if best_delay == float("inf"):
        best_delay = 1e9

    for idx in valid[1:]:
        idx = int(idx)
        _, server_id = _decode_c(idx, num_servers)
        dist = distances[server_id]
        delay = features[idx]["edge_compute_ms"]
        if delay == float("inf"):
            delay = 1e9

        if dist < best_dist or (dist == best_dist and delay < best_delay):
            best = idx
            best_dist = dist
            best_delay = delay

    return best
```

Distance is computed by `_estimate_distance_km` (`offloading_baselines.py:32–43`) using `get_k_shortest_paths(..., k=1, weight="length_km")`.

### 2.4 DF_C decision rule summary

1. Enumerate all legal `(split_id, server_id)` actions from `agent_c_mask`.
2. Compute shortest-path-in-km distance from request source to each server.
3. Select the legal action whose server has the smallest distance.
4. Tie-break by smaller `edge_compute_ms` (estimated edge compute delay for that split/server).
5. Return the flat action index; if no legal action exists, return `None` (the evaluator then falls back to action 0).

### 2.5 Same action space as PPO-C

Yes. Both PPO-C and DF_C select from the same flat `(split_id, server_id)` space of size `num_splits * num_servers = 12`. DF_C further restricts itself to the legal subset encoded by `agent_c_mask`, which is the same raw mask that PPO-C uses before risk masking.

### 2.6 Same C-side observation

Yes. The strict evaluator builds `obs_c` once per request with `build_agent_c_observation(env, req)` (`eval_strict_v13_multitopology_cside_fair.py:372`) and passes the same `obs_c` to either `_select_c_action_from_obs` (PPO-C) or `select_offloading_action` (DF_C).

### 2.7 Information used by DF_C

- `obs_c["agent_c_mask"]` for legality.
- `obs_c["request_features"]["src_node"]` for distance computation.
- `env.mec.servers[server_id].node_id` for server locations.
- `obs_c["candidate_features"][idx]["edge_compute_ms"]` for tie-breaking.

DF_C does **not** use server utilization or queue delay (those are used by RF_C / WO_C). It does not use feasible R action counts directly, although the legality mask already encodes whether at least one R action exists for each `(split, server)`.

### 2.8 Determinism

DF_C is fully deterministic: given `env`, `obs_c`, and `mask`, `select_df` contains no random sampling.

### 2.9 Not a replacement by another heuristic

The strict evaluator maps `df_c` only to `select_df`. It does not map it to `greedy_c`, `rf_c`, `wo_c`, or `iwd_c`. Those are separate `c_mode` strings handled by the same dispatcher but are not DF_C.

---

## 3. Ambiguity Resolution

| Candidate | File | Mask-aware? | Canonical? | Reason |
|---|---|---|---|---|
| `select_df` | `offloading_baselines.py:157–196` | Yes | **Yes** | Used by `eval_strict_v13_multitopology_cside_fair.py` and the existing heuristic-C reports. |
| `_select_df` | `eval_yin_style_offloading_sweep.py:87–93` | No | No | Server-first Yin-style baseline; not imported by the main strict evaluator. |

No blocker. DF_C is uniquely identified.

---

## 4. Summary Table

| Property | PPO-C | DF_C |
|---|---|---|
| Class / function | `PPOAgentC` | `select_df` |
| File | `sa_hmarl/sa_hmarl/agents/ppo_agents.py` | `sa_hmarl/sa_hmarl/evaluation/offloading_baselines.py` |
| Selection function | `_select_c_action_from_obs` | `select_df` |
| Line(s) | `diagnose_r_action_horizon_oracle.py:296–317` | `offloading_baselines.py:157–196` |
| Action space | `(split_id, server_id)`, 12 actions | `(split_id, server_id)`, 12 actions |
| Uses legal mask | Yes (plus risk mask) | Yes (`agent_c_mask`) |
| Uses K_C=5 paths | Yes (via `build_agent_c_observation`) | Yes (via `build_agent_c_observation`) |
| Uses server utilization | Yes (in features) | No |
| Uses queue delay | Yes (in features) | No |
| Uses feasible R counts | Yes (in features) | Indirectly via mask |
| Deterministic | Yes | Yes |
| Randomness | None | None |

---

*Audit complete. DF_C is unambiguously `select_df` in `offloading_baselines.py`. The experiment can proceed under both PPO-C and DF_C.*
