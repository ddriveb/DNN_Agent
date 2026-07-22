# COST239 C-Side Feature Mode Comparison

Formal training and validation of three Agent-C feature modes on `xlron_cost239_ptrnet_real`:
- `default` (17-D)
- `r_feasibility_safe` (29-D: base 17 + 10 R-feasibility + 2 server-margin)
- `overload_aware` (26-D: base 17 + 9 post-action pressure/risk features)

## Training Configuration

- Topology: `xlron_cost239_ptrnet_real`
- MEC: 4 servers, `server_nodes=[0,1,2,3]`, `capacities=[50,50,50,50]`
- Training traffic (standard stress):
  - `arrival_interval=0.15`
  - `holding_min=4.0`, `holding_max=10.0`
  - `size_min_mb=5.0`, `size_max_mb=30.0`
  - `requests_per_episode=80`, `episodes=400`
- Agent: frozen PPO-R (`agent_r_mixed.pt`)
- C policy: PPOAgentC, hidden `(128, 64)`, seed 42
- Validation: same standard stress, `eval_freq=25`, `validation_episodes=5`

## Key Finding: Best vs. Last Checkpoint

The training script selects the "best" checkpoint by validation mean-blocking on the **standard** stress. Because standard stress is already near-saturated for a well-trained policy, the best checkpoint was consistently found at episode 25 (0% validation blocking) for all three modes.

However, the **last checkpoint (episode 399) generalized dramatically better to heavier stress**, especially for `overload_aware`:

| C mode | Checkpoint | Heavy stress blocking (v12_k50_hops) | Heavy stress blocking (ksp_ff_k50_hops) |
|---|---|---:|---:|
| default | best (ep 25) | 1.13% | 0.83% |
| default | last (ep 399) | 0.00% | 0.00% |
| r_feasibility_safe | best (ep 25) | 0.00% | 0.92% |
| r_feasibility_safe | last (ep 399) | 0.00% | 0.00% |
| overload_aware | best (ep 25) | 6.92% | 7.46% |
| overload_aware | last (ep 399) | 0.00% | 0.00% |

**Interpretation:** early-stopping on a too-easy validation regime is dangerous for the higher-dimensional `overload_aware` policy. With enough training (last checkpoint), all three feature modes solve the COST239 overload problem.

## Full Last-Checkpoint Results

R backend: both `v12_k50_hops` (v1.3 hybrid, ensure_ksp) and `ksp_ff_k50_hops`.

| Stress | C mode | R backend | Blocking | Raw empty | NSB | Overload | Delay mean/P95 | S0% | S1% | S2% | S3% |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| standard | default | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.029/15.321 | 14.3% | 19.4% | 37.3% | 29.0% |
| standard | default | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 7.957/13.304 | 13.7% | 16.7% | 41.3% | 28.3% |
| standard | r_feasibility_safe | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.921/16.182 | 27.4% | 20.7% | 20.9% | 31.0% |
| standard | r_feasibility_safe | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.602/16.260 | 32.1% | 22.0% | 22.4% | 23.5% |
| standard | overload_aware | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.127/16.147 | 27.4% | 25.9% | 27.7% | 19.0% |
| standard | overload_aware | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 8.378/16.111 | 25.0% | 28.8% | 27.9% | 18.3% |
| medium | default | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.159/15.559 | 19.2% | 16.3% | 34.0% | 30.4% |
| medium | default | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 8.304/13.794 | 16.8% | 17.2% | 36.4% | 29.7% |
| medium | r_feasibility_safe | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 10.190/16.304 | 28.8% | 21.0% | 20.8% | 29.5% |
| medium | r_feasibility_safe | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.606/16.033 | 29.9% | 23.7% | 19.8% | 26.6% |
| medium | overload_aware | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.212/15.997 | 27.5% | 22.8% | 26.1% | 23.6% |
| medium | overload_aware | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 8.494/15.269 | 26.8% | 26.8% | 30.8% | 15.5% |
| heavy | default | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.639/16.566 | 21.2% | 18.2% | 31.5% | 29.1% |
| heavy | default | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 8.778/15.355 | 19.4% | 17.7% | 33.6% | 29.3% |
| heavy | r_feasibility_safe | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.890/15.749 | 27.4% | 23.0% | 21.8% | 27.8% |
| heavy | r_feasibility_safe | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.624/15.461 | 30.3% | 24.7% | 20.5% | 24.5% |
| heavy | overload_aware | v12_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 9.378/16.149 | 29.4% | 21.5% | 24.3% | 24.9% |
| heavy | overload_aware | ksp_ff_k50_hops | 0.00% | 0.00% | 0.00% | 0.00% | 8.447/14.447 | 29.6% | 20.3% | 31.3% | 18.8% |

### Averaged over R backends

| Stress | C mode | Avg Blocking | Avg Overload | Avg Delay | S0% | S1% | S2% | S3% |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| standard | default | 0.00% | 0.00% | 8.493 | 14.0% | 18.0% | 39.3% | 28.6% |
| standard | r_feasibility_safe | 0.00% | 0.00% | 9.762 | 29.8% | 21.4% | 21.6% | 27.3% |
| standard | overload_aware | 0.00% | 0.00% | 8.752 | 26.2% | 27.4% | 27.8% | 18.7% |
| medium | default | 0.00% | 0.00% | 8.732 | 18.0% | 16.7% | 35.2% | 30.1% |
| medium | r_feasibility_safe | 0.00% | 0.00% | 9.898 | 29.3% | 22.4% | 20.3% | 28.0% |
| medium | overload_aware | 0.00% | 0.00% | 8.853 | 27.1% | 24.8% | 28.5% | 19.6% |
| heavy | default | 0.00% | 0.00% | 9.209 | 20.3% | 17.9% | 32.6% | 29.2% |
| heavy | r_feasibility_safe | 0.00% | 0.00% | 9.757 | 28.9% | 23.8% | 21.2% | 26.1% |
| heavy | overload_aware | 0.00% | 0.00% | 8.912 | 29.5% | 20.9% | 27.8% | 21.8% |

## Server Allocation Balance Analysis

All three modes eliminate server-overload blocking with the last checkpoint, but they choose different server-balance strategies:

- **`default`** consistently under-selects Server 0 (the historically overloaded server) and overloads Servers 2 and 3. This is a learned avoidance strategy, not an explicit balance objective.
- **`r_feasibility_safe`** produces the most uniform server distribution across all stress levels (e.g., heavy: S0=28.9%, S1=23.8%, S2=21.2%, S3=26.1%).
- **`overload_aware`** is also fairly balanced, with slightly higher S0/S2 selection than `r_feasibility_safe` (e.g., heavy: S0=29.5%, S2=27.8%).

## Conclusions

1. **The COST239 server-overload problem is solvable with all three feature modes**, provided training is allowed to run long enough (last checkpoint) rather than early-stopping on a too-easy validation set.
2. **`overload_aware` does not beat `r_feasibility_safe`**. Both improve balance over `default`, but `r_feasibility_safe` gives the most uniform server allocation and is more robust to early stopping.
3. **Early-stopping on standard-stress validation is dangerous for `overload_aware`**: the best checkpoint (ep 25) yields 6.9–7.5% blocking under heavy stress, while the last checkpoint yields 0%. This suggests the extra 9 overload dimensions need more training to become useful, or the validation metric must include heavier stress.
4. **There is no clear justification to proceed to reward shaping solely based on `overload_aware`**, because it does not outperform the simpler `r_feasibility_safe` representation.

## Recommendations

1. **Adopt `r_feasibility_safe` as the C-side representation for COST239** unless further feature engineering can demonstrate a consistent advantage.
2. **Change checkpoint selection** for C-side training: either
   - use the last checkpoint, or
   - validate on a mix of standard + medium/heavy stress so the best checkpoint is stress-robust.
3. If reward shaping is still desired, **shape the reward to penalize server imbalance explicitly** (e.g., variance of projected utilization) rather than relying only on feature engineering. This could be combined with any of the three feature modes.
4. Consider training `overload_aware` longer (e.g., 800–1000 episodes) or adding feature normalization/input scaling, since its high-dimensional state may need more optimization budget.

## Artifacts

- Checkpoints: `sa_hmarl/checkpoints/agent_c_cost239_{default,r_feasibility_safe,overload_aware}_{best,last}.pt`
- Eval reports: `sa_hmarl/experiments/cost239_{mode}_{best,last}_{standard,medium,heavy}.{md,json}`
- Training logs: `sa_hmarl/experiments/train_cost239_{mode}.log`
