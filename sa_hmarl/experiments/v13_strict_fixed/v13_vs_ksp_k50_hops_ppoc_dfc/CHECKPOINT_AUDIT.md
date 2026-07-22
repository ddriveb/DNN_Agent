# Checkpoint Audit

This audit records the canonical checkpoints used in the pre-registered experiment and verifies that the Strict v1.3 checkpoint matches the required loss-matched protocol.

---

## 1. Strict v1.3 Ranker

| Property | Value |
|---|---|
| Path | `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt` |
| SHA256 | `ae2f17b3644e17e19b3967918d654177c7d84666bdb7469e40c1b8c13fa2e2de` |
| Training seed | 42 |
| Selection | Best validation regret (checkpoint chosen by `selection_metric=regret`, not by test or closed-loop results) |
| Model type | `mlp` |
| input_dim | 25 |
| hidden_dims | `[128, 64]` |
| dropout | 0.0 |
| feature_names | 25 names, matching `FEATURE_NAMES` in `generate_r_post_decision_dataset.py` |
| candidate_mode | `ppo_r_topk_only` |
| ppo_top_k | 30 |
| max_candidates | 30 |
| reg_weight | 1.0 |
| lambda_pair | 0.0 |
| lambda_hard | 0.0 |

Feature names (in order):

```json
[
  "path_length_km", "hop_count", "lfb", "free_ratio", "frag_index",
  "spectral_efficiency", "reach_km", "required_fs", "block_size", "block_waste",
  "path_mod_feasible", "split_norm", "server_norm", "deadline_norm", "holding_norm",
  "intermediate_size_norm", "server_utilization", "selected_valid_r_ratio",
  "k_c_valid_ratio", "k_r_total_ratio", "phi_spec_norm", "raw_r_valid_ratio",
  "path_idx_norm", "mod_idx_norm", "block_idx_norm"
]
```

Protocol compliance:

- Loss-matched strict protocol (`reg_weight=1.0`, `lambda_pair=0.0`, `lambda_hard=0.0`): yes.
- Candidate source: PPO-R Top-30 legal actions only (`ppo_r_topk_only`, `ppo_top_k=30`): yes.
- Full-state / `group_filter=all`: yes (trained on the merged full-state pilot dataset).
- Sampling mode: `uniform`.
- No test-set checkpoint selection: yes (selected by validation regret).

This is the same checkpoint used in the previous Strict v1.3 closed-loop report (`FINAL_STRICT_DECISION.md`). It was chosen by the orchestrator at `run_v13_strict_fixed_pilot.py:555`.

---

## 2. PPO-C

| Property | Value |
|---|---|
| Path | `sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt` |
| SHA256 | `a5c9eb39bc33cf3b77a06d80e9c45315880dcb963ad9c3c95663eadaded9f2c8` |
| Class | `PPOAgentC` |
| input_dim | 29 |
| hidden_dims | `(128, 64)` |
| feature_mode | `r_feasibility_safe` |
| Activation | `tanh` |

Training environment (from checkpoint args):

- Topology: `xlron_cost239_ptrnet_real`
- Modulation profile: `default`
- `num_slots`: 100
- `requests_per_episode`: 80
- `holding_min/max`: 4.0 / 10.0
- `arrival_interval`: 0.15
- `k_paths`: 5

Deployment environment in this experiment:

- Topology: `xlron_cost239_ptrnet_real`
- Modulation profile: `default`
- `num_slots`: 320
- `requests_per_episode`: 6000
- `holding_min/max`: 20.0 / 30.0
- `arrival_interval`: 0.0625
- `k_paths`: 5

Note: topology and modulation profile match, but traffic scale and slot count differ.

---

## 3. PPO-R

| Property | Value |
|---|---|
| Path | `sa_hmarl/checkpoints/agent_r_mixed.pt` |
| SHA256 | `e32e908196ac0c39898aab6e98516842c53f817150eeaba97e4ac172a61f8b94` |
| Class | `PPOAgentR` |
| input_dim | 11 |
| hidden_dims | `(128, 64)` |
| feature_mode | default (inferred from 11-dim input; no `feature_mode` key stored) |

Training environment (from checkpoint args):

- Topology: `nsfnet`
- Modulation profile: `extended`
- `num_slots`: 32
- `requests_per_episode`: 60
- `holding_min/max`: 4.0 / 10.0
- `arrival_interval`: 0.25

Deployment environment in this experiment:

- Topology: `xlron_cost239_ptrnet_real`
- Modulation profile: `default`
- `num_slots`: 320
- `requests_per_episode`: 6000
- `holding_min/max`: 20.0 / 30.0
- `arrival_interval`: 0.0625

Note: This is a substantial checkpoint–deployment distribution mismatch (topology, modulation profile, slot count, traffic parameters). The checkpoint is used as-is because the task evaluates the existing frozen PPO-R policy. The mismatch must be reported in any conclusion.

---

## 4. Legacy E1-Trained Gated Baseline (optional diagnostic)

The legacy ranker checkpoint is:

- Path: `sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt`
- SHA256: `f62852c8ee2f1bb0ce59c3a66dfd6ffbd9183ebdfdac8e45c7b4241bb28543fb`
- Usage: diagnostic only; must be labeled "Legacy E1-trained gated baseline" in any table.
- It was trained on E1-only (`deep_path_only`) data and reused only as a baseline reference.
- Model type: `mlp`, input_dim=25, hidden_dims=[128,64], candidate_mode=`ppo_r_topk_only`, ppo_top_k=30.

This checkpoint will be loaded only for the optional diagnostic rows `ppo_c+legacy_e1_gated` and `df_c+legacy_e1_gated`.

---

## 5. Verification Commands

SHA256 sums were computed with:

```bash
sha256sum \
  sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt \
  sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt \
  sa_hmarl/checkpoints/agent_r_mixed.pt
```

Checkpoint metadata was inspected with:

```python
import torch
ckpt = torch.load(path, map_location='cpu', weights_only=False)
print(ckpt.keys())
for k in ['input_dim', 'hidden_dims', 'feature_names', 'model_type', 'dropout',
          'selection_metric', 'ppo_top_k', 'candidate_mode', 'max_candidates']:
    print(k, ckpt.get(k))
```

---

*All checkpoints are frozen for this experiment. No checkpoint will be retrained or replaced.*
