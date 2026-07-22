# SA-HMARL v1.3 Counterfactual Ranking Dataset Report

- Dataset dir: ``
- Topology: `xlron_cost239_ptrnet_real`
- K_C=5, K_path=50, K_prop=30, H=5
- Candidate mode: `ppo_r_topk_only`
- Path sort (C/R): `hops` / `hops`
- Block sort (C/R): `start_asc` / `start_asc`
- Gamma: 1.0

## Label coefficients

```json
{
  "current_block": 3.0,
  "future_block": 4.0,
  "future_nsb": 3.0,
  "delay": 0.03,
  "fs": 0.05,
  "future_server_overload": 0.0,
  "future_optical": 0.0,
  "future_overload": 0.0,
  "future_other": 0.0,
  "path_penalty": 0.0,
  "fs_penalty": 0.0,
  "return_mode": "legacy_total_return"
}
```

## Diagnostics

| Split | Groups | Avg cand | P50 cand | PPO top-1 | Oracle headroom | Nonzero range | Return mean | Return std |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| train | 7303 | 29.01 | 30 | 37.37% | 0.0471 | 85.35% | -1.6781 | 3.5232 |
| val | 1750 | 29.98 | 30 | 36.74% | 0.0624 | 84.00% | -1.4912 | 3.3315 |
| test | 2079 | 28.90 | 30 | 40.69% | 0.1049 | 83.15% | -2.6595 | 4.4472 |

## Label distribution (train)

```json
{
  "n": 211835,
  "mean": -1.6781061887741089,
  "std": 3.5231878757476807,
  "min": -20.0,
  "max": -0.0,
  "p5": -8.690373420715332,
  "p25": -0.600240170955658,
  "p50": -0.5182443261146545,
  "p75": -0.44228604435920715,
  "p95": -0.3113773763179779
}
```

## Feature statistics (train valid candidates)

- feature_dim=25
- mean[0:5]=[1515.6993408203125, 2.9925696849823, 32.1075553894043, 0.258091002702713, 0.6716752052307129]
- std[0:5]=[582.5891723632812, 1.0331953763961792, 41.996238708496094, 0.19735829532146454, 0.20761097967624664]