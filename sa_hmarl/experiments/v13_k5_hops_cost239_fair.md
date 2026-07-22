# v1.3 K5-hop COST239 Fair Control Experiment

Controlled comparison of v1.3 ranker vs. KSP-FF on the **same** K=5 hop-ordered path support.

## Protocol

All parameters identical to the COST239 old main table and the corrected KSP-FF K5-hop run:

| Parameter | Value |
|---|---|
| topology | `xlron_cost239_ptrnet_real` |
| num_slots | 320 |
| num_servers | 4 |
| k_paths | 5 |
| path_sort_strategy | `hops` |
| block_sort_strategy | `start_asc` |
| edge_cost_min | 0.1 |
| edge_cost_max | 2.2 |
| seeds | 3030, 4040, 5050, 6060, 7070 |
| requests_per_episode | 10000 |
| warmup_requests | 2000 |
| agent_c_checkpoint | `sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt` |
| ranking_checkpoint | `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt` |

## v1.3 Ranker Candidate Settings

| Setting | Value | Note |
|---|---|---|
| r_mode | `v13_k5_hops` | Uses `_r_backend_env_config` default (`args.k_paths=5`, `args.path_sort_strategy=hops`, `args.block_sort_strategy=start_asc`), **no K=50 override**. |
| candidate_mode | `all_legal` | Checkpoint stores `candidate_mode=null`; policy default is `all_legal`. No `--ranker_candidate_mode` override was passed. |
| ensure_ksp | `false` | No `--ranker_ensure_ksp` flag was passed. |
| max_candidates | 48 | Policy default. |

## Results

| Method | K | path_order | block_order | candidate_mode | ensure_ksp | blocking mean ± std | per-seed blocking | delay mean / P95 (ms) | decision mean / P95 (ms) | raw_empty | NSB | overload |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `ppo_c+v13_k5_hops` | 5 | hops | start_asc | all_legal | false | 8.13% ± 0.81% | 8.50%, 8.00%, 6.96%, 9.40%, 7.76% | 9.863 / 17.257 | 7.169 / 10.122 | 10.66% | 0.00% | 8.13% |
| `ppo_c+ksp_ff_k5_hops` | 5 | hops | start_asc | — | — | 7.46% ± 0.94% | 7.49%, 8.28%, 6.11%, 8.66%, 6.75% | 8.989 / 16.470 | 6.564 / 9.307 | 9.69% | 0.00% | 7.46% |

All failures are server overload (`NSB = 0`).

## Comparison to K=50 Horizon

From the parallel K=50 experiment (`deeprmsa_k50_hop_fair_comparison.json`, same protocol):

| Method | K | Mean blocking |
|---|---|---:|
| `ppo_c+ksp_ff` | 5 | **7.46%** |
| `ppo_c+v13_k5_hops` | 5 | 8.13% |
| `ppo_c+v12_k50_hops` | 50 | 9.30% |
| `ppo_c+ksp_ff_k50_hops` | 50 | 9.93% |

## Judgment

**v1.3 K5-hop (8.13%) > KSP-FF K5-hop (7.46%)**.

This falls into the third case:

> *If v1.3 K5-hop > KSP-FF K5-hop, the ranker checkpoint and K5-hop candidate distribution are mismatched, or the scenario is primarily overload-driven and not a good v1.3 victory scenario.*

Both effects likely contribute:

1. **Overload-dominated regime**: all blocking is server overload. KSP-FF's first-fit packing keeps spectrum/server capacity lean, while the ranker's all_legal scoring may pick actions that increase resource consumption.
2. **Candidate distribution mismatch**: the v1.3 ranker checkpoint was trained/validated with K=50 (`v12_k50_hops`) and may not generalize to the much smaller K=5 candidate set.

Consequently, the COST239 `edge_cost_max=2.2` / K=5 scenario is **not** a strong demonstration of v1.3's advantage; the K=50 horizon is the appropriate setting for that comparison.

## Files

- `sa_hmarl/experiments/v13_k5_hops_cost239_fair.json` — full per-seed metrics.
- `sa_hmarl/experiments/v13_k5_hops_cost239_fair.md` — this report.
