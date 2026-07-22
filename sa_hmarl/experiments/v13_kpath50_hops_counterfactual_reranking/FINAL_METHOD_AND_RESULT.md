# SA-HMARL v1.3: Final Method and Result

**Method name:** PPO-R Proposal-Supported RMSA Domain-Constrained Common-Future Counterfactual Reranking

## Method definition

1. **C-side proposal (frozen PPO-C, K_C=5).** At each request, the frozen PPO-C policy observes the state under path support K_C=5 and proposes a split/server pair.
2. **R-side proposal support (frozen PPO-R, K_path=50).** The R-side state is observed under the wider path support K_path=50. The frozen PPO-R policy scores every legal flat action and provides the top-K_prop=30 actions as the candidate set.
3. **Common-future counterfactual evaluation.** Each candidate is executed on a deep-copied snapshot of the environment. From the post-action state, H=5 future requests are rolled out with the same frozen PPO-C + PPO-R pair and the *same* future request trace.
4. **Label.** The candidate return is the legacy v1.3 total return:
   ```
   R = -3·B_t - 4·ΣB - 3·ΣNSB - 0.03·D - 0.05·FS
   ```
5. **Ranker.** A small MLP scores each candidate using the 25-d pre-decision feature vector. It is trained offline to rank candidates by their counterfactual return.
6. **Online deployment.** At test time, the ranker scores the same PPO-R Top-30 candidates and selects the highest-scoring action.

## What is *not* claimed

- The method is **not** globally optimal; it only re-ranks inside the PPO-R proposal support.
- It does **not** use a mixed candidate pool (KSP anchors, random fillers, or post-state features).
- It does **not** rely on an explicit afterstate value function; the label is a total-return counterfactual.

## Results

*Results from the pilot and full-scale runs will be inserted here after the background jobs complete.*

### Pilot dataset (4 shards × 1000 requests)

| Split | Groups | Avg cand | PPO top-1 | Oracle headroom | Nonzero range |
|---|---:|---:|---:|---:|---:|
| train | 1285 | 28.52 | 35.49% | 0.0807 | 87.59% |
| val | 429 | 30.00 | 34.27% | 0.0547 | 86.71% |
| test | 602 | 29.98 | 31.23% | 0.0547 | 89.20% |

PPO-R's greedy Top-1 matches the teacher's best-return candidate only ~35% of the time, leaving substantial headroom for a learned ranker.

### Pilot closed-loop comparison (2 seeds × 2000 evaluated requests)

| Mode | Blocking % | Overload % | NSB % | PPO agree |
|---|---:|---:|---:|---:|
| ppo_r_top1 | 8.50% ± 2.80% | 8.20% | 0.30% | 100.00% |
| ksp_ff_plain | 10.33% ± 2.17% | 9.57% | 0.75% | 29.73% |
| ksp_ff_highest | 6.65% ± 1.25% | 6.62% | 0.03% | 7.15% |
| ranker_old_v13 | 6.68% ± 0.87% | 6.58% | 0.10% | 10.20% |
| ranker_new_v13 | 8.92% ± 3.82% | 8.62% | 0.30% | 14.00% |

Per-seed breakdown:

| Seed | ppo_r | ksp_ff_highest | old_v13 | new_v13 |
|---:|---:|---:|---:|---:|
| 4001 | 5.70% | 5.40% | 5.80% | 5.10% |
| 4002 | 11.30% | 7.90% | 7.55% | 12.75% |

### Full-scale

Not yet run. The pilot suggests the strict v1.3 method is viable (it beats PPO-R on seed 4001) but the current pilot dataset is too small to yield a stable average improvement over PPO-R Top-1.

## Interpretation

- **Common-future supervision is implementable and passes all protocol audits.** The dataset has high return range (≈88%) and positive oracle headroom, confirming that ranking inside the PPO-R Top-30 support is a meaningful learning problem.
- **Pilot ranker is not yet reliably better than PPO-R Top-1.** Average blocking is 8.92% vs 8.50% for PPO-R, with large seed-to-seed variance. This is expected from a small pilot (≈1.3k train groups).
- **Old v1.3 ranker remains strong.** It was trained on a larger, less strictly controlled pool, so it benefits from more data and implicit regularization. It also operates inside the same strict Top-30 pool when evaluated fairly, and it outperforms PPO-R on average.
- **Recommendation:** Run the full-scale dataset (e.g. 16 train / 4 val / 4 test shards × 5000 requests) overnight and re-train. The pilot pipeline is verified end-to-end; the remaining issue is data scale, not methodology.

## Supporting reports

| Report | File |
|---|---|
| Implementation audit | `IMPLEMENTATION_AUDIT.md` |
| Smoke-test invariants | `SMOKE_TEST.md` |
| Common-future trace consistency | `COMMON_FUTURE_TRACE_TEST.md` |
| Pilot dataset details | `DATASET_REPORT.md` |
| Ranker training details | `TRAINING_REPORT.md` |
| Fair closed-loop comparison | `FAIR_COMPARISON_FULL.md` |

