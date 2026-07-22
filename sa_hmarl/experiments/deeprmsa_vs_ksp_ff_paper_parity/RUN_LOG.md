# RUN LOG — DeepRMSA vs KSP-FF paper parity pipeline

All stages executed 2026-07-17 on this machine (12 CPU workers).

| Stage | Command | Outcome | Elapsed |
|---|---|---|---|
| A unit tests | `python run_unit_tests.py` | PASS (10/10) | 1.8 s |
| B smoke | `python run_smoke.py` | OK | 6.3 s |
| C KSP parity | `python run_ksp_parity.py` | parity PASS | 91.8 s |
| D train nsfnet s42 | `python train_deeprmsa_paper.py --topology nsfnet --seed 42` | best val 5.9222% | 1309 s |
| D train nsfnet s43 | `python train_deeprmsa_paper.py --topology nsfnet --seed 43` | best val 8.6889% | 1305 s |
| D train nsfnet s44 | `python train_deeprmsa_paper.py --topology nsfnet --seed 44` | best val 5.5222% | 1296 s |
| D train cost239 s42 | `python train_deeprmsa_paper.py --topology cost239 --seed 42` | best val 10.9556% | 1303 s |
| D train cost239 s43 | `python train_deeprmsa_paper.py --topology cost239 --seed 43` | best val 8.4667% | 1295 s |
| D train cost239 s44 | `python train_deeprmsa_paper.py --topology cost239 --seed 44` | best val 6.7889% | 1302 s |
| D eval | `python eval_deeprmsa_paper.py` | done | 156.9 s |

## Notable events

* Stage C first run used `xlron_cost239_ptrnet_real` for COST239 and FAILED parity
  (our 3.18% vs paper 6.69±0.35%). Root cause: wrong COST239 variant. Switched to
  upstream/XLRON `cost239_deeprmsa` (2× distances) and reran — see PREFLIGHT_AUDIT.md
  addendum. NSFNET was unaffected and matched on the first run.
* Stage D training required a measured hyperparameter investigation (all deviations
  disclosed in PREFLIGHT_AUDIT.md D6/D7): lr=1e-5 stalls at the SP-FF collapse at this
  compute budget; lr=1e-4 + grad-clip 40 diverged one seed; advantage normalization
  prevents collapse escape; final config lr=1e-4 + grad-clip 5 + raw advantages let all
  seeds converge (one NSFNET seed via a collapse→diverge→recover trajectory).
* Final COST239/NSFNET parity status is recorded in KSP_PARITY_RESULTS.md.
