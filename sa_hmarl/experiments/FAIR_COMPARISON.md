# Fair Comparison: v1.3 Ranker vs poststate_v1 Upgrade

Protocol: COST239 (`xlron_cost239_ptrnet_real`), 320 slots, 4 servers, K=50 shortest paths sorted by hops, `start_asc` block ordering, 5000 requests/seed after 1000-request warmup, seeds `3030,4040,5050,6060,7070`.

Checkpoints:
- `ranker_v13`: `sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt` (25-d features, `ppo_r_topk_only`, H=5).
- `ranker_poststate`: `sa_hmarl/checkpoints/r_poststate_v1_h5_anchor48_k50/ranking_model.pt` (41-d features, `poststate_anchor48`, H=5). For evaluation speed, the poststate checkpoint was also run with `ppo_r_topk_only`; on every measured seed the selected action was identical to the full `poststate_anchor48` run, so the per-seed numbers below are taken from the faster run.

## Aggregate results

| Mode | Blocking % | Overload % | NSB % | Avg delay ms | Avg FS | Decision ms | PPO agree |
|---|---:|---:|---:|---:|---:|---:|---:|
| ppo_r | 21.29% ± 9.09% | 21.28% | 0.00% | 11.49 | 3.29 | 27.20 | 100.00% |
| ranker_v13 | 21.69% ± 8.22% | 21.68% | 0.01% | 11.75 | 2.97 | 32.71 | 26.52% |
| ranker_poststate | 22.07% ± 6.29% | 22.06% | 0.01% | 12.20 | 3.29 | 64.71 | 41.34% |
| ksp_ff_plain | 24.36% ± 5.38% | 24.34% | 0.01% | 10.62 | 3.24 | 24.18 | 43.17% |
| ksp_ff_highest | 26.76% ± 8.67% | 26.76% | 0.00% | 10.73 | 2.28 | 24.02 | 32.21% |

## Per-seed details

| Seed | Mode | Blocking % | Overload % | NSB % | Decision ms |
|---|---|---|---:|---:|---:|
| 3030 | ppo_r | 13.78% | 13.78% | 0.00% | 25.73 |
| 3030 | ranker_v13 | 13.00% | 13.00% | 0.00% | 34.18 |
| 3030 | ranker_poststate | 16.62% | 16.62% | 0.00% | 68.16 |
| 3030 | ksp_ff_plain | 20.42% | 20.42% | 0.00% | 22.23 |
| 3030 | ksp_ff_highest | 20.30% | 20.30% | 0.00% | 22.17 |
| 4040 | ppo_r | 25.96% | 25.96% | 0.00% | 35.48 |
| 4040 | ranker_v13 | 27.92% | 27.92% | 0.00% | 41.79 |
| 4040 | ranker_poststate | 24.50% | 24.48% | 0.02% | 79.46 |
| 4040 | ksp_ff_plain | 35.00% | 34.96% | 0.04% | 27.02 |
| 4040 | ksp_ff_highest | 27.80% | 27.80% | 0.00% | 28.30 |
| 5050 | ppo_r | 17.94% | 17.94% | 0.00% | 23.88 |
| 5050 | ranker_v13 | 16.48% | 16.48% | 0.00% | 28.10 |
| 5050 | ranker_poststate | 19.06% | 19.06% | 0.00% | 55.61 |
| 5050 | ksp_ff_plain | 22.34% | 22.34% | 0.00% | 22.25 |
| 5050 | ksp_ff_highest | 22.04% | 22.04% | 0.00% | 22.43 |
| 6060 | ppo_r | 36.72% | 36.70% | 0.02% | 27.15 |
| 6060 | ranker_v13 | 34.66% | 34.60% | 0.04% | 32.51 |
| 6060 | ranker_poststate | 33.30% | 33.26% | 0.04% | 62.81 |
| 6060 | ksp_ff_plain | 21.32% | 21.30% | 0.02% | 27.66 |
| 6060 | ksp_ff_highest | 43.22% | 43.22% | 0.00% | 25.05 |
| 7070 | ppo_r | 12.04% | 12.04% | 0.00% | 23.77 |
| 7070 | ranker_v13 | 16.38% | 16.38% | 0.00% | 26.95 |
| 7070 | ranker_poststate | 16.86% | 16.86% | 0.00% | 57.50 |
| 7070 | ksp_ff_plain | 22.70% | 22.70% | 0.00% | 21.73 |
| 7070 | ksp_ff_highest | 20.46% | 20.46% | 0.00% | 22.15 |

## Discussion

- **PPO-R remains the strongest single policy on average** under this protocol. The distilled rankers do not reliably beat it; they win on individual seeds but lose on others.
- **v1.3 ranker (`ranker_v13`)** is close to PPO-R on average (within 0.4 pp) and is a valid low-regret amortized policy. Its PPO agreement is low (~26%), confirming it actively reranks rather than copying PPO-R.
- **poststate_v1 ranker** is slightly worse on average than both PPO-R and `ranker_v13` in this run. Two likely causes: (1) the training dataset is small (240 train groups), so the 41-d model underfits; (2) the H=5 label still does not capture the overload dynamics that dominate blocking.
- **All failures are `server_overload`** (NSB ≈ 0), which is consistent with the Stage 0 observation that the dominant failure mode under fixed `a_C` is compute overload, not optical blocking. Since overload is largely determined by the C-side split/server choice, there is limited room for an R-side ranker to improve it.
- **KSP-FF baselines** are generally worse than PPO-R, with `ksp_ff_highest` showing high variance (43.22% on seed 6060). This confirms that a learned policy is preferable to simple KSP heuristics.
- **Latency**: `ranker_poststate` costs ~65 ms/decision vs ~33 ms for `ranker_v13` and ~27 ms for PPO-R, because poststate feature computation evaluates each candidate's optical afterstate. The anchor48 candidate set did not change the selected action compared with top-K in these seeds, so the extra latency did not translate into better decisions here.

## Caveats

- The evaluation uses 5000 requests/seed rather than the historical 10000 to fit the background-time budget. Variance across seeds is therefore higher.
- The poststate ranker was trained on only 240 groups. A larger training set and longer horizon (H=12/20) may change the comparison.
- Because blocking is overwhelmingly `server_overload`, R-side reranking has a low ceiling unless the label explicitly weights overload differences across candidates and the horizon is long enough for those differences to appear.
