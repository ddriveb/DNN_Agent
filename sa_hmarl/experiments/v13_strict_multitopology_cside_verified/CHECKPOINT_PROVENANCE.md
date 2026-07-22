# Checkpoint Provenance

| Name | Path | SHA-256 | Deployment label |
|---|---|---|---|
| agent_c_cost239_native | sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt | `a5c9eb39bc33cf3b...` | native for xlron_cost239_ptrnet_real PPO-C |
| agent_c_delayaware_transfer | sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt | `707b504e9b3dda71...` | transfer/zero-shot PPO-C for German17/NSFNET/JPN48 |
| agent_r_mixed | sa_hmarl/checkpoints/agent_r_mixed.pt | `e32e908196ac0c39...` | mixed/multi-topology PPO-R (deployed transfer; verify training args) |
| strict_v13_seed42 | sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt | `ae2f17b3644e17e1...` | Strict v1.3 ranker trained on COST239; source-matched for COST239, zero-shot transfer to others |
| strict_v13_seed43 | sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_43/ranking_model.pt | `a4737b092bbe9c12...` | Strict v1.3 ranker trained on COST239; source-matched for COST239, zero-shot transfer to others |
| strict_v13_seed44 | sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_44/ranking_model.pt | `390ec5840a3bef2c...` | Strict v1.3 ranker trained on COST239; source-matched for COST239, zero-shot transfer to others |
| legacy_old_v13 | sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt | `f62852c8ee2f1bb0...` | legacy/old v1.3 diagnostic reference |

## Notes
- COST239 PPO-C is source-matched (native).
- German17/NSFNET/JPN48 PPO-C uses the delay-aware v2 checkpoint in zero-shot/transfer mode.
- Strict v1.3 ranker is trained on COST239; COST239 is source-matched, other topologies are zero-shot.
- PPO-R (agent_r_mixed.pt) is a mixed/multi-topology checkpoint; deployment topology/protocol may differ.