# Checkpoint Transfer Audit

Strict v1.3 is evaluated as a frozen cross-topology transfer: the same PPO-R and Ranker checkpoints are loaded for every topology without any retraining or topology-specific adaptation.

| Checkpoint | Path | SHA-256 |
|---|---|---|
| PPO-R | `sa_hmarl/checkpoints/agent_r_mixed.pt` | e32e908196ac0c39898aab6e98516842c53f817150eeaba97e4ac172a61f8b94 |
| Ranker | `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt` | ae2f17b3644e17e19b3967918d654177c7d84666bdb7469e40c1b8c13fa2e2de |

## Verification

- All model parameters are set to `requires_grad=False` after loading.
- No optimizer state is loaded.
- No checkpoint file is written.
- Topology-specific node/slot mismatch is not checked because the ranker input dim is fixed at 25 and the PPO-R action space is determined by the environment's K_path, num_mods, and max_blocks, which are held constant across topologies.
