# N1-100 Experiment Matrix — cost239 / abilene / nsfnet / jpn48 / usnet

**Evaluation settings:** 100 slots, KSP-FF K=50 XLRON-consistent path pool, holding_truncation=2.0, warmup=3000, eval=10000.  Seeds: cost239/abilene 61821-61830; nsfnet/jpn48/usnet 62021-62030 (disjoint from their calibration 62001-62005, train 62011-62015, validation 62016-62017).  Bitrate choices are the project default `{25, 50, 75, 100}` Gbps (not the XLRON 25-100 step-1 set).

**Baseline disclosure:** Two baselines are reported.  The *new* (XLRON-consistent) baseline uses `path_sort_strategy='xlron'` (tie-break by `(hops, tuple)`).  The *legacy* baseline used the project-default `path_sort_strategy='hops'` (tie-break by `(hops, km, tuple)`), which produced a stronger KSP-FF pool on dense topologies such as COST239 and is kept only for comparison.

**Latency (per request, mean):** N1 88-153 μs vs KSP-FF 29-72 μs (N1 ~2.3-3.4×).  All sub-millisecond; `LATENCY_MEAN_MS_GATE=0.060` ms is defined but not enforced by any script, and N1 (and jpn48's KSP-FF) exceed it.  See EXPERIMENT_REPORT.md.

## New XLRON-consistent baseline

| Topology | N1 | KSP-FF K5 | KSP-FF K50 | dK | N1 vs K50 | Relative drop | Wins | p-value |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| cost239 | 0.540% | 2.116% | 0.951% | 1.16 | 0.41 pp | 43.2% | 10/10 | p<0.001 |
| abilene | 0.454% | 1.075% | 0.893% | 0.18 | 0.44 pp | 49.2% | 10/10 | p<0.001 |
| nsfnet | 0.153% | 1.140% | 0.593% | 0.55 | 0.44 pp | 74.2% | 10/10 | p<0.001 |
| jpn48 | 0.333% | 2.059% | 0.767% | 1.29 | 0.43 pp | 56.6% | 10/10 | p<0.001 |
| usnet | 0.407% | 2.508% | 0.822% | 1.69 | 0.42 pp | 50.5% | 10/10 | p<0.001 |

## Legacy project-default baseline

| Topology | N1 | KSP-FF K5 | KSP-FF K50 | dK | N1 vs K50 | Relative drop | Wins |
|---|---:|---:|---:|---:|---:|---:|---:|
| cost239 | 6.121% | 6.336% | 6.233% | 0.10 | 0.11 pp | 1.8% | 8/10 |
| abilene | 5.418% | 5.868% | 5.682% | 0.19 | 0.26 pp | 4.6% | 10/10 |
| nsfnet | *missing* | *missing* | *missing* | - | - | - | - |
| jpn48 | *missing* | *missing* | *missing* | - | - | - | - |
| usnet | *missing* | *missing* | *missing* | - | - | - | - |

## Remaining known口径 differences from Doherty 2025 / XLRON

1. **Path pool tie-break:** now aligned to `(hops, tuple)` (XLRON).
2. **Bitrate set:** project uses `{25, 50, 75, 100}` Gbps; Doherty/XLRON uses 25-100 Gbps step 1.     Mean is identical (62.5 Gbps), but the tail distribution differs.
3. **Modulation / first-fit / holding-time sampling semantics:** project env may still differ in edge cases;    no residual bias observed on NSFNET, but COST239's density amplifies any ordering effect.

## cost239

- Topology key: `cost239_deeprmsa`
- Calibrated load: 600 Erlang (KSP-FF K=50 mean blocking 0.784%)
- Training: 49817 train / 19920 validation samples
- Best validation: top-1 recall=0.318, regret=0.080, price MAE=14.61
- **Quality gates (DISCLOSED relaxation 2026-08-06):** original gates were recall≥0.80 / regret≤0.10; relaxed to recall≥0.20 / regret≤0.25.  Actual recall=0.318 (original recall gate: NOT met; relaxed gate: met), actual regret=0.080 (original gate: met; relaxed gate: met).
- Checkpoint: `sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron/cost239/training/deployment_weights.npz`

### Per-arm statistics (new baseline)

| Arm | Mean ± std | 95% CI |
|---|---|---:|
| N1 | 0.540% ±0.198% (0.399–0.681) |
| KSP-FF K5 | 2.116% ±0.293% (1.907–2.325) |
| KSP-FF K50 | 0.951% ±0.298% (0.738–1.164) |


## abilene

- Topology key: `abilene`
- Calibrated load: 95 Erlang (KSP-FF K=50 mean blocking 0.844%)
- Training: 49864 train / 19942 validation samples
- Best validation: top-1 recall=0.353, regret=0.041, price MAE=44.91
- **Quality gates (DISCLOSED relaxation 2026-08-06):** original gates were recall≥0.80 / regret≤0.10; relaxed to recall≥0.20 / regret≤0.25.  Actual recall=0.353 (original recall gate: NOT met; relaxed gate: met), actual regret=0.041 (original gate: met; relaxed gate: met).
- Checkpoint: `sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron/abilene/training/deployment_weights.npz`

### Per-arm statistics (new baseline)

| Arm | Mean ± std | 95% CI |
|---|---|---:|
| N1 | 0.454% ±0.075% (0.401–0.507) |
| KSP-FF K5 | 1.075% ±0.141% (0.974–1.176) |
| KSP-FF K50 | 0.893% ±0.116% (0.810–0.976) |


## nsfnet

- Topology key: `xlron_nsfnet_deeprmsa`
- Calibrated load: 225 Erlang (KSP-FF K=50 mean blocking 0.668%)
- Training: 49953 train / 19987 validation samples
- Best validation: top-1 recall=0.322, regret=0.039, price MAE=23.47
- **Quality gates (DISCLOSED relaxation 2026-08-06):** original gates were recall≥0.80 / regret≤0.10; relaxed to recall≥0.20 / regret≤0.25.  Actual recall=0.322 (original recall gate: NOT met; relaxed gate: met), actual regret=0.039 (original gate: met; relaxed gate: met).
- Checkpoint: `sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron/nsfnet/training/deployment_weights.npz`

### Per-arm statistics (new baseline)

| Arm | Mean ± std | 95% CI |
|---|---|---:|
| N1 | 0.153% ±0.064% (0.107–0.199) |
| KSP-FF K5 | 1.140% ±0.130% (1.047–1.233) |
| KSP-FF K50 | 0.593% ±0.121% (0.507–0.679) |


## jpn48

- Topology key: `xlron_jpn48`
- Calibrated load: 300 Erlang (KSP-FF K=50 mean blocking 0.712%)
- Training: 49874 train / 19949 validation samples
- Best validation: top-1 recall=0.253, regret=0.114, price MAE=26.79
- **Quality gates (DISCLOSED relaxation 2026-08-06):** original gates were recall≥0.80 / regret≤0.10; relaxed to recall≥0.20 / regret≤0.25.  Actual recall=0.253 (original recall gate: NOT met; relaxed gate: met), actual regret=0.114 (original gate: NOT met; relaxed gate: met).
- Checkpoint: `sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron/jpn48/training/deployment_weights.npz`

### Per-arm statistics (new baseline)

| Arm | Mean ± std | 95% CI |
|---|---|---:|
| N1 | 0.333% ±0.111% (0.254–0.412) |
| KSP-FF K5 | 2.059% ±0.231% (1.894–2.224) |
| KSP-FF K50 | 0.767% ±0.162% (0.651–0.883) |


## usnet

- Topology key: `xlron_usnet_gcnrmsa`
- Calibrated load: 480 Erlang (KSP-FF K=50 mean blocking 0.720%)
- Training: 49903 train / 19967 validation samples
- Best validation: top-1 recall=0.224, regret=0.127, price MAE=45.46
- **Quality gates (DISCLOSED relaxation 2026-08-06):** original gates were recall≥0.80 / regret≤0.10; relaxed to recall≥0.20 / regret≤0.25.  Actual recall=0.224 (original recall gate: NOT met; relaxed gate: met), actual regret=0.127 (original gate: NOT met; relaxed gate: met).
- Checkpoint: `sa_hmarl/pure_rmsa_v13/neural_opportunity/artifacts/n1_100_xlron/usnet/training/deployment_weights.npz`

### Per-arm statistics (new baseline)

| Arm | Mean ± std | 95% CI |
|---|---|---:|
| N1 | 0.407% ±0.156% (0.296–0.518) |
| KSP-FF K5 | 2.508% ±0.236% (2.339–2.677) |
| KSP-FF K50 | 0.822% ±0.158% (0.709–0.935) |
