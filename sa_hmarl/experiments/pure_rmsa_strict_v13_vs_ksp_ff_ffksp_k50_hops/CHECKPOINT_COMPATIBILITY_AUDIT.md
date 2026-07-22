# Checkpoint Compatibility Audit

Status: **PASS_WITH_DISTRIBUTION_SHIFT**

- PPO-R: `sa_hmarl/checkpoints/agent_r_mixed.pt`
- SHA256: `e32e908196ac0c39898aab6e98516842c53f817150eeaba97e4ac172a61f8b94`
- Ranker: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt`
- SHA256: `ae2f17b3644e17e19b3967918d654177c7d84666bdb7469e40c1b8c13fa2e2de`
- Ranker input: 25 dimensions in stored feature-name order.
- Action space: 50 paths x 4 modulations x 10 blocks = 2000.
- C-only fields are neutralized to checkpoint means; this is frozen cross-domain transfer.

## Disclosed Shifts

- checkpoint training used SA-HMARL C/MEC context
- checkpoint evaluation used 320 slots; paper protocol uses 100 slots
- paper reach table is 10000/2500/1250/625 km
- C-only ranker fields are neutralized to checkpoint means
