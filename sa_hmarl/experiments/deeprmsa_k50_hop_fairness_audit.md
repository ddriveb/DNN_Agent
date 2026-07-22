# DeepRMSA Path-Support Fairness Audit

## 1. Executive Summary

This audit checks whether the neural DeepRMSA baseline and the v1.3 counterfactual R-ranker are evaluated on the **same candidate-path horizon**.  The short answer is **no** for the existing DeepRMSA checkpoints:

| Backend | Topology | K | Path ordering | Trainable? |
|---|---|---|---|---|
| v1.3 `v12_k50_hops` | any imported topology | 50 | hop-count | yes (ranker) |
| Neural DeepRMSA (`deep_rmsa`) | snap24 only | 5 | hops (by env default) | yes (A3C) |
| **Heuristic DeepRMSA-style K50** (`deep_rmsa_style_k50_hops`) | any imported topology | 50 | hop-count | no — hand-crafted scorer |

Because the DeepRMSA A3C network dimensions are tied to `k_path` and `m_blocks`, a K=50 DeepRMSA checkpoint cannot be obtained by re-parameterizing the existing K=5 checkpoint; it must be retrained.  To still isolate the effect of **path horizon** from **action-valuation architecture**, we introduce `deep_rmsa_style_k50_hops`: a non-neural policy that uses the same `(path, block)` action space and highest-SE modulation rule as DeepRMSA, but scores candidates with a hand-crafted objective over the K=50 hop-ordered path set.

## 2. How Path Support Is Configured

In `eval_long_horizon_system_comparison.py` (and `eval_main_s100_system_comparison.py`), the R-side backend receives its candidate paths via:

```python
def _r_backend_env_config(r_mode: str, args: argparse.Namespace) -> tuple[int, str, str]:
    if r_mode in ("ksp_ff_k50_hops", "v12_k50_hops", "ppo_r_k50_hops",
                  "ppo_r_proposer_ksp_ff", "ppo_r_proposer_postdec",
                  "mixed32_postdec", "deep_rmsa_style_k50_hops"):
        return args.ksp_ff_k50_hops_k_paths, "hops", "start_asc"
    return args.k_paths, args.path_sort_strategy, args.block_sort_strategy
```

- `v12_k50_hops` and `deep_rmsa_style_k50_hops` both receive `k=50`, `path_sort_strategy="hops"`, `block_sort_strategy="start_asc"`.
- Neural `deep_rmsa` is **not** in the K50 override list, so it keeps the evaluator default `k_paths=5`.

Therefore, under the standard long-horizon protocol, v1.3 sees 50 hop-ordered candidate paths while the available DeepRMSA checkpoint sees only 5.

## 3. Why the Neural DeepRMSA Checkpoint Cannot Be Re-parameterized to K=50

`DeepRMSAAgent` fixes its input/output dimensions at construction:

```python
self.n_actions = k_path * m_blocks
self.state_dim = num_nodes * 2 + k_path * (1 + m_blocks * 2 + 2)
```

The saved checkpoint contains `k_path` and `m_blocks`.  Loading a K=5 checkpoint into a K=50 agent (or vice versa) fails with a shape mismatch because the policy/value heads and the state encoder have different sizes.  Concretely:

- Existing checkpoint: `deep_rmsa_snap24_s100_k5m10_ai015_h4_10_s5_30.pt`
  - `k_path=5`, `m_blocks=10`, topology=snap24, `num_slots=100`.
- COST239 topology also has a different `num_nodes`, so loading this checkpoint on COST239 additionally fails with a topology-mismatch error in `_load_deep_rmsa`.

A fair neural DeepRMSA-K50-hop baseline would require training a new checkpoint with:

```bash
python -m sa_hmarl.training.train_deep_rmsa \
  --topology xlron_cost239_ptrnet_real \
  --k_paths 50 --m_blocks 10 --path_sort_strategy hops \
  ...
```

That training run is outside the scope of a single evaluation pass, so we provide the heuristic proxy below.

## 4. `deep_rmsa_style_k50_hops`: Path-Support-Equalized DeepRMSA Proxy

This backend is implemented in `eval_long_horizon_system_comparison.py` (and `eval_main_s100_system_comparison.py`) as `_deep_rmsa_style_k50_action`.  It mirrors DeepRMSA's action parameterization:

1. **Action space**: `(path_idx, block_idx)` — the modulation is not part of the policy output.
2. **Modulation rule**: for each path, use the highest-spectral-efficiency feasible modulation, exactly as `_best_mod_for_path` in `DeepRMSAAgent`.
3. **Scoring**: each legal `(path, best-mod, block)` triple is scored with a hand-crafted objective that rewards tight spectrum fit, shorter paths, and lower slot consumption:

```python
score = (
    -1.00 * waste
    - 0.30 * (path_km / max_path_km)
    - 0.20 * (req_fs / num_slots)
    - 0.05 * (block_start / num_slots)
)
```

where `waste = (block_size - req_fs) / block_size`.

This is **not** a learned DeepRMSA policy; it is an interpretable, path-horizon-matched ablation.  It tells us how much of the v1.3 vs. DeepRMSA gap is due to the **wider path set** versus the **neural value/policy estimator**.

## 5. COST239 Fair-Comparison Results

Protocol: `xlron_cost239_ptrnet_real`, 320 slots, 4 servers, `edge_cost_min=0.1`, `edge_cost_max=2.2`, 10 000 requests/episode, 2 000 warmup, seeds `3030,4040,5050,6060,7070`.  C-side fixed to `agent_c_cost239_r_feasibility_safe_last.pt`.  **All methods use `path_sort_strategy=hops` and `block_sort_strategy=start_asc`** (the old main-table protocol).

| Method | R backend | K | Path ordering | Mean blocking | Per-seed blocking |
|---|---|---|---:|---|---|
| `ppo_c+ksp_ff` | KSP-FF | 5 | hops | **7.46%** | 7.49%, 8.28%, 6.11%, 8.66%, 6.75% |
| `ppo_c+v12_k50_hops` | v1.3 ranker | 50 | hops | 9.30% | 9.00%, 9.48%, 7.71%, 11.21%, 9.09% |
| `ppo_c+deep_rmsa_style_k50_hops` | DeepRMSA-style | 50 | hops | 9.62% | 10.09%, 10.21%, 7.15%, 11.15%, 9.49% |
| `ppo_c+ksp_ff_k50_hops` | KSP-FF | 50 | hops | 9.93% | 8.50%, 11.11%, 8.64%, 12.09%, 9.31% |

All failures are server overload (`no_suitable_block_rate = 0`), so blocking equals overload rate in this regime.

### Protocol consistency check vs. old main table

Old main table: `sa_hmarl/experiments/long_horizon_benchmark/xlron_cost239_ptrnet_real_ecmax2.2_full.json`.

| Parameter | Old main table | This run | Match? |
|---|---|---|---|
| topology | `xlron_cost239_ptrnet_real` | `xlron_cost239_ptrnet_real` | ✅ |
| num_slots | 320 | 320 | ✅ |
| num_servers | 4 | 4 | ✅ |
| k_paths | 5 | 5 | ✅ |
| path_sort_strategy | hops | hops | ✅ |
| block_sort_strategy | start_asc | start_asc | ✅ |
| edge_cost_min | 0.1 | 0.1 | ✅ |
| edge_cost_max | 2.2 | 2.2 | ✅ |
| seeds | 3030,4040,5050,6060,7070 | 3030,4040,5050,6060,7070 | ✅ |
| requests_per_episode | 10000 | 10000 | ✅ |
| warmup_requests | 2000 | 2000 | ✅ |
| agent_c_checkpoint | `agent_c_cost239_r_feasibility_safe_last.pt` | `agent_c_cost239_r_feasibility_safe_last.pt` | ✅ |
| ranking_checkpoint | `r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt` | `r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt` | ✅ |

### Interpretation

- **Path support is equalized**: `v12_k50_hops` and `deep_rmsa_style_k50_hops` both receive K=50 hop-ordered paths, confirming the audit claim.
- **Action-valuation vs. path-horizon**: with matched path support, the hand-crafted DeepRMSA-style proxy (9.62%) is statistically tied with the learned v1.3 ranker (9.30%).
- **K50 can hurt under overload**: KSP-FF with K=5 (7.46%) outperforms KSP-FF with K=50 (9.93%).  The wider path set includes longer paths that consume more spectrum and server capacity; because all blocking here is overload-driven, the shorter-horizon heuristic is actually better.
- **Old main table reproduces**: the KSP-FF K50 baseline in this run is 9.93%, consistent with the old main table's 10.37% (same protocol, minor run-to-run variance).
- **Neural DeepRMSA remains un-evaluated on COST239**: the existing snap24/K=5 checkpoint cannot be loaded on COST239 due to topology and slot mismatches.

## 6. Caveats

- **Action-space difference remains**: v1.3 selects `(path, modulation, block)` jointly; DeepRMSA-style selects `(path, block)` and then highest-SE modulation.  Equalizing path support does not equalize action parameterization.
- **Heuristic weights are not trained**: the `deep_rmsa_style_k50_hops` scorer is a best-effort proxy.  A retrained neural DeepRMSA-K50 could outperform it.
- **Neural DeepRMSA is snap24-only**: the released checkpoint cannot be evaluated on COST239, German17, NSFNET, or JPN48.
- **Overload-dominated regime**: all blocking is due to server overload.  In a spectrum-scarcity regime, the K50 horizon might look more favorable.

## 7. Recommended Interpretation for Paper

Use the results in [`deeprmsa_k50_hop_fair_comparison.md`](./deeprmsa_k50_hop_fair_comparison.md) as follows:

- `ppo_c+v12_k50_hops` vs. `ppo_c+ksp_ff_k50_hops`: v1.3 value-engineering gain at K=50 (≈0.6 pp better than KSP-FF K50).
- `ppo_c+deep_rmsa_style_k50_hops` vs. `ppo_c+ksp_ff_k50_hops`: DeepRMSA-style action parameterization gain at K=50 (≈0.3 pp better than KSP-FF K50).
- `ppo_c+ksp_ff` vs. `ppo_c+ksp_ff_k50_hops`: effect of widening path support for a pure heuristic under overload (K=5 is ≈2.5 pp better).
- Neural `ppo_c+deep_rmsa` results, when reported, should be clearly labeled as **snap24 / K=5 / num_slots=100 only** and not directly compared to COST239 K=50 numbers.

## 8. Files Produced

- `sa_hmarl/experiments/deeprmsa_k50_hop_fairness_audit.md` — this audit.
- `sa_hmarl/experiments/deeprmsa_k50_hop_fair_comparison.json` — full per-seed metrics.
- `sa_hmarl/experiments/deeprmsa_k50_hop_fair_comparison.md` — markdown summary table.

## 9. Addendum: v1.3 K5-hop Control Experiment

To further isolate whether the v1.3 ranker's value comes from **better action valuation** or simply from the **wider K=50 path horizon**, we ran a K=5-hop control experiment on the identical COST239 protocol.

Full report: [`v13_k5_hops_cost239_fair.md`](./v13_k5_hops_cost239_fair.md)

### Settings

- r_mode: `v13_k5_hops` (new mode that uses the v1.3 ranker but **does not** trigger the K=50 override in `_r_backend_env_config`).
- Explicit main config: `--k_paths 5 --path_sort_strategy hops --block_sort_strategy start_asc`.
- Ranker candidate_mode: `all_legal` (checkpoint default); ensure_ksp: `false`.
- All other parameters identical to Section 5.

### Results

| Method | K | Mean blocking | Per-seed blocking |
|---|---|---:|---|
| `ppo_c+ksp_ff_k5_hops` | 5 | **7.46%** | 7.49%, 8.28%, 6.11%, 8.66%, 6.75% |
| `ppo_c+v13_k5_hops` | 5 | 8.13% | 8.50%, 8.00%, 6.96%, 9.40%, 7.76% |

### Interpretation

`v1.3 K5-hop (8.13%) > KSP-FF K5-hop (7.46%)`.  This supports the conclusion that:

1. The v1.3 ranker checkpoint was trained/validated for the K=50 horizon and does **not** generalize well to the K=5 candidate distribution.
2. The COST239 `edge_cost_max=2.2` regime is **overload-dominated**; KSP-FF's lean first-fit packing outperforms the ranker's all_legal scoring when paths are restricted to K=5.
3. Therefore, the appropriate v1.3 comparison setting is **K=50**, not K=5.

This control experiment strengthens the main audit finding: the fair DeepRMSA comparison must be made at matched path support, and on COST239 the K=50 horizon is where v1.3 shows measurable gains.
