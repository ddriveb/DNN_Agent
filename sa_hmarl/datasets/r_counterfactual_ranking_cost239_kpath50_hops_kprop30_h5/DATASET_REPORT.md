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
| train | 1285 | 28.52 | 30 | 35.49% | 0.0807 | 87.59% | -2.1969 | 4.1557 |
| val | 429 | 30.00 | 30 | 34.27% | 0.0547 | 86.71% | -1.0438 | 2.6864 |
| test | 602 | 29.98 | 30 | 31.23% | 0.0547 | 89.20% | -1.1200 | 2.4466 |

## Label distribution (train)

```json
{
  "n": 36642,
  "mean": -2.196859836578369,
  "std": 4.155702114105225,
  "min": -20.0,
  "max": -0.0,
  "p5": -12.623092651367188,
  "p25": -0.6356097459793091,
  "p50": -0.5311126708984375,
  "p75": -0.4667377769947052,
  "p95": -0.3762584924697876
}
```

## Feature statistics (train valid candidates)

- feature_dim=25
- mean[0:5]=[1620.4447021484375, 3.0373342037200928, 21.75105094909668, 0.2160637080669403, 0.6999605298042297]
- std[0:5]=[655.2786254882812, 1.0623914003372192, 27.030231475830078, 0.18145740032196045, 0.17941993474960327]