# PPO-C + v1.2 vs. KSP-FF K5-hop on COST239

Direct comparison of the planner-distilled v1.2 ranker and KSP-FF on the **same K=5 hop-ordered path support**.

## Protocol

| Parameter | Value |
|---|---|
| topology | `xlron_cost239_ptrnet_real` |
| num_slots | 320 |
| num_servers | 4 |
| k_paths | 5 |
| path_sort_strategy | `hops` |
| block_sort_strategy | `start_asc` |
| edge_cost_min / max | 0.1 / 2.2 |
| seeds | 3030, 4040, 5050, 6060, 7070 |
| requests_per_episode | 10000 |
| warmup_requests | 2000 |
| agent_c_checkpoint | `sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt` |
| ranking_checkpoint | `sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt` |

## v1.2 Ranker Candidate Settings

| Setting | Value | Note |
|---|---|---|
| r_mode | `v12` | Uses main config (`k_paths=5`, `path_sort_strategy=hops`, `block_sort_strategy=start_asc`), no K=50 override. |
| candidate_mode | `all_legal` | Default; v1.2 original behavior scores every legal candidate. |
| ensure_ksp | `false` | Default; no KSP action injection. |
| max_candidates | 48 | Policy default (irrelevant in `all_legal` mode). |

## Results

| Method | K | path_order | block_order | candidate_mode | ensure_ksp | blocking mean ± std | per-seed blocking | delay mean / P95 (ms) | decision mean / P95 (ms) | raw_empty | NSB | overload |
|---|---|---|---|---|---:|---|---|---:|---:|---:|---:|---:|
| `ppo_c+v12` | 5 | hops | start_asc | all_legal | false | 8.13% ± 0.81% | 8.50%, 8.00%, 6.96%, 9.40%, 7.76% | 9.863 / 17.257 | 7.255 / 10.268 | 10.66% | 0.00% | 8.13% |
| `ppo_c+ksp_ff` | 5 | hops | start_asc | — | — | **7.46% ± 0.94%** | 7.49%, 8.28%, 6.11%, 8.66%, 6.75% | 8.989 / 16.470 | 6.493 / 9.233 | 9.69% | 0.00% | 7.46% |

All failures are server overload (`NSB = 0`).

## Conclusion

**PPO-C + v1.2 K5-hop (8.13%) is worse than PPO-C + KSP-FF K5-hop (7.46%)** by **0.67 pp** on average.

This means:
- On COST239 under this protocol, the v1.2 ranker does **not** beat a simple KSP-FF first-fit heuristic when both are restricted to K=5 hop-ordered paths.
- The v1.2 checkpoint was trained for the wider candidate horizon (K=50 / `legalctx48` in the old main table); its all_legal scoring does not add value in the K=5 overload-dominated regime.
- This reinforces that the appropriate v1.2/v1.3 comparison setting is **K=50** (or at least the horizon it was trained on), not K=5.

## Files

- `sa_hmarl/experiments/v12_k5_hops_cost239_fair.json` — full per-seed metrics.
- `sa_hmarl/experiments/v12_k5_hops_cost239_fair.md` — this report.
