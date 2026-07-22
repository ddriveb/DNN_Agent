# Protocol Audit: COST239 fixed-C / all-OD Three-Method RMSA Comparison

## Scenario

This experiment compares three RMSA methods under fixed-C / all-OD pure RMSA:

1. Strict v1.3
2. KSP-FF K=50 hops
3. Topology-matched adapted DeepRMSA K=50 hops

## Protocol Lock Statement

> This is fixed-C / all-OD pure RMSA. PPO-C and DF_C are not invoked.
> `split_id` is fixed to 0 and `server_id` is deterministically mapped from `dst_node`.

## Configuration

| Parameter | Value |
|---|---|
| Topology | `xlron_cost239_ptrnet_real` |
| Slots | 320 |
| Servers | 4 |
| Server nodes | [0, 1, 2, 3] |
| K_path (R) | 50 |
| Path sort | hops |
| Block sort | start_asc |
| Max blocks | 10 |
| Modulations | 4 |
| Unified action space | 2000 |
| Arrival interval | 0.3 |
| Holding time | [20.0, 30.0] |
| Warmup | 500 |
| Evaluated requests | 6000 |
| Poisson arrivals | False |
| Exponential holding | False |
| Traffic matrix | uniform_all_od |

## Verification Hashes

- Topology config SHA256: `73c17618bf93e5aabdee916f8e3fc1837f94765f8bbdd61824dfb823e78d656e`
- Action space SHA256: `6c785f837afdc68e98224a653e6ccf89d078a1aebca3f2bc9237bde26df11eb3`
- Traffic config SHA256: `3527f83928d22a8151757c72fea09911a2b5e24ea8e393fa2c7d64ddc5cab2bc`

## Methods

### Strict v1.3

- PPO-R checkpoint: `sa_hmarl/checkpoints/agent_r_mixed.pt`
- Ranker checkpoint: `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt`
- Candidate pool: ppo_r_legal_topk_only
- Top-K: 30
- Feature dim: 25

### KSP-FF K=50 hops

- Function: `ksp_ff_highest_mod_action`
- K_path: 50
- Path sort: hops
- Block sort: start_asc

### Topology-matched adapted DeepRMSA K=50 hops

- Base architecture: DeepRMSA 5-layer 128-unit ELU MLP A2C
- Output dim: 2000
- Train from scratch: True
- Action mask: same physical R mask as PPO-R

## Evaluation Seeds

[5001, 5002, 5003, 5004, 5005]

## Provenance

This configuration is copied from the previous fixed-C / all-OD diagnosis
that reported 5.6342% blocking for both Strict v1.3 and KSP-FF K=50 hops.
Any deviation from these parameters must be reported as protocol drift.

## Parity Verification

**Canonical runner**: `sa_hmarl/sa_hmarl/evaluation/ksp_ff_k50_hops_only_parity.py`

**Buggy runner (fixed)**: `sa_hmarl/sa_hmarl/evaluation/ksp_ff_k50_parity_check.py` had a warmup
queue misalignment bug: it skipped the first 500 warmup requests with `continue`,
without calling `env.step()` or `env.reject_next_request()`. Since `env.reset(requests)`
had already enqueued all requests, the inner queue popped `req=0` while the outer
loop was at `req=500`, causing a permanent 500-request offset and a spurious
1.0267% blocking rate. The bug was fixed by executing the environment operation
during warmup and only gating metric recording.

**Parity result (5 seeds)**:

| Seed | KSP-FF K=50 hops blocking |
|---|---:|
| 5001 | 5.3833% |
| 5002 | 5.6667% |
| 5003 | 5.2667% |
| 5004 | 5.8333% |
| 5005 | 5.8167% |
| **Mean** | **5.5933%** |

The 5-seed mean 5.5933% reproduces the source-of-truth 5-seed subset from the
original 20-seed diagnosis (also 5.5933%). The 20-seed overall mean 5.6342%
remains the broader reference.

**Parity status**: PASSED. Proceeding to DeepRMSA training and three-method evaluation.