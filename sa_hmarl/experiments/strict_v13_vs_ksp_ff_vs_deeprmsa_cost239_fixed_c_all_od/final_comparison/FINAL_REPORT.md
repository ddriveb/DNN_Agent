# FINAL REPORT: COST239 Fixed-C / All-OD Pure RMSA Three-Method Fair Comparison

## 1. Executive Summary

This 5-seed pilot compares three RMSA methods under fixed-C / all-OD on `xlron_cost239_ptrnet_real` with 320 slots.

| Method | Overall Blocking Rate | Accepted | R-no-valid | Avg Hops | Avg FS | Avg Block Waste |
|---|---:|---:|---:|---:|---:|---:|
| Strict v1.3 | 5.5933% | 28322 | 1678 | 1.95 | 2.08 | 0.9667 |
| KSP-FF K=50 hops | 5.5933% | 28322 | 1678 | 1.45 | 2.05 | 0.4785 |
| Topology-matched adapted DeepRMSA K=50 hops | 12.0833% | 26375 | 3625 | 3.68 | 2.48 | 0.3720 |

## 2. Pairwise Comparisons

### Strict v1.3 vs KSP-FF K=50 hops
- Mean blocking difference (Strict - KSP): 0.00 pp
- 95% bootstrap CI: [0.00, 0.00] pp
- Strict wins: 0, KSP wins: 0, both success: 28322, both block: 1678
- Action divergence rate: 93.2167%

### Strict v1.3 vs Topology-matched adapted DeepRMSA K=50 hops
- Mean blocking difference (Strict - DeepRMSA): -6.49 pp
- 95% bootstrap CI: [-6.79, -6.28] pp
- Strict wins: 2803, DeepRMSA wins: 856, both success: 25519, both block: 822
- Action divergence rate: 97.1567%

### KSP-FF K=50 hops vs Topology-matched adapted DeepRMSA K=50 hops
- Mean blocking difference (KSP - DeepRMSA): -6.49 pp
- 95% bootstrap CI: [-6.79, -6.28] pp
- KSP wins: 2803, DeepRMSA wins: 856, both success: 25519, both block: 822
- Action divergence rate: 97.2600%

## 3. KSP-FF Parity Verification

- Source-of-truth overall blocking rate (20 seeds): 5.6342%
- Pilot KSP-FF blocking rate (5 seeds): 5.5933%
- The pilot reproduces the source-of-truth rate within sampling noise.

## 4. Implementation Notes for Adapted DeepRMSA

- Preserves DeepRMSA's 5-layer 128-unit ELU MLP backbone and A2C episode-level updates.
- Output head size is 2000 (50 paths x 4 modulations x 10 blocks).
- State encoding uses src/dst one-hot + per-path DeepRMSA-style features, adapted for K=50 and B=10.
- Action selection is masked over the same physical R mask used by PPO-R and KSP-FF.
- Training from scratch on the current COST239 fixed-C / all-OD environment; no external checkpoint loaded.

## 5. Resource & Timing

- Adapted DeepRMSA trainable parameters: 690,513
- Training was stopped early at epoch ~74 after validation blocking showed no improvement beyond epoch 1.
- Best checkpoint: `deep_rmsa_adapted/seed_42/best.pt` saved at epoch 1.
- Best validation blocking rate: 12.067% (validation seeds 6001, 6002, 6003).
- DeepRMSA 5-seed evaluation blocking rate: 12.0833%.
- DeepRMSA evaluation time: 122.4s.

## 6. Recommendation

Under COST239 fixed-C / all-OD pure RMSA with 320 slots, the pilot shows:

- **Strict v1.3** and **KSP-FF K=50 hops** are statistically tied at ~5.59% blocking.
- **Topology-matched adapted DeepRMSA K=50 hops** is substantially worse at **12.08% blocking**.

The 6.49 pp gap between DeepRMSA and the other two methods is large and consistent across all 5 evaluation seeds. Scaling to 20 seeds would tighten confidence intervals, but the pilot already strongly suggests DeepRMSA is not competitive in this adapted form. Before any further comparison, DeepRMSA's state encoding / network / training hyperparameters would need substantial redesign for the 2000-action K=50 / B=10 space.
