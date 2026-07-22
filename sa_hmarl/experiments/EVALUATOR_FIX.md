# Evaluator Fix Note — `eval_v13_fair_comparison.py`

## Bugs found

The first version of `sa_hmarl/sa_hmarl/evaluation/eval_v13_fair_comparison.py` had two protocol errors:

1. **Warmup was not implemented.**
   - The script declared `--warmup_requests` and reported it in the Markdown.
   - However, it generated only `args.requests_per_episode` requests and counted every one of them.
   - This made the reported "1000 warmup + 5000 eval" protocol false.

2. **C-side path support was wrong.**
   - The environment was created with `k=args.k_paths` (default 50).
   - The frozen PPO-C checkpoint was trained with K=5 candidate paths, so giving it 50 paths at inference time changed its action distribution and inflated blocking.
   - The historical fair-comparison protocol therefore uses C-K=5 and only widens the R-side path support to K=50.

## Fixes applied

- The script now generates `warmup_requests + requests_per_episode` total requests.
- Metrics are collected only after the warmup window.
- Added `--k_paths_c` (default 5) and `--k_paths_r` (default 50) plus matching sort-strategy flags.
- Before `build_agent_c_observation`, the environment is set to the C-side configuration.
- Before `build_agent_r_observation`, the environment is set to the R-side configuration.
- The Markdown header now reports total/evaluated/warmup counts and both C/R K values.

## Impact

| Issue | Before fix | After fix |
|---|---|---|
| PPO-R baseline (seed 3030) | 13.73% blocking | 5.58% (10k eval) / 7.62% (5k eval) |
| v1.3 ranker aggregate | 21.69% blocking | 6.01% blocking |
| KSP-FF plain aggregate | 24.36% blocking | 7.88% blocking |

The corrected numbers restore the historical ~6–8% COST239 blocking scale and match the previously reported v1.3 result (~6.29%).

## Files

- Fixed evaluator: `sa_hmarl/sa_hmarl/evaluation/eval_v13_fair_comparison.py`
- Corrected comparison report: `sa_hmarl/experiments/v13_fair_comparison_fixed_5k.md`
- Corrected comparison data: `sa_hmarl/experiments/v13_fair_comparison_fixed_5k.json`
