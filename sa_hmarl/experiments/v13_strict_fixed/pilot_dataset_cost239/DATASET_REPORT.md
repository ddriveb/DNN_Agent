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
| train | 593 | 30.00 | 30 | 34.40% | 0.0337 | 90.05% | -1.3650 | 3.6869 |
| val | 580 | 25.24 | 30 | 30.52% | 0.2719 | 92.58% | -1.6130 | 3.5335 |
| test | 580 | 24.62 | 30 | 52.07% | 0.0319 | 65.34% | -0.7115 | 2.1987 |

## Label distribution (train)

```json
{
  "n": 17790,
  "mean": -1.3650364875793457,
  "std": 3.686924457550049,
  "min": -20.0,
  "max": -0.0,
  "p5": -12.415472030639648,
  "p25": -0.5152604579925537,
  "p50": -0.44554388523101807,
  "p75": -0.36718717217445374,
  "p95": -0.2585863471031189
}
```

## Feature statistics (train valid candidates)

- feature_dim=25
- mean[0:5]=[1248.1943359375, 2.7780776023864746, 42.84941101074219, 0.3333793580532074, 0.6306625008583069]
- std[0:5]=[512.01953125, 0.9524929523468018, 41.229888916015625, 0.19838538765907288, 0.21649514138698578]

## E-gate and depth stratum (v1.3 fix)

- group_filter=all
- total E=0 groups: 229
- total E=1 groups: 1524
- total groups: 1753
- depth stratum counts (total): {'0': 229, '1': 42, '2': 731, '3': 751}

| Split | E=0 | E=1 | E=1 rate | Stratum 0 | Stratum 1 | Stratum 2 | Stratum 3 |
|---|---|---:|---:|---:|---:|---:|---:|
| train | 59 | 534 | 90.05% | 59 | 23 | 406 | 105 |
| val | 50 | 530 | 91.38% | 50 | 10 | 81 | 439 |
| test | 120 | 460 | 79.31% | 120 | 9 | 244 | 207 |