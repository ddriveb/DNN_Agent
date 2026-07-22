# Protocol Lock: PAPER_PRIMARY_K50

This experiment fixes the comparison protocol described in the task specification.

## Formal KSP-FF
* `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py::ksp_ff_highest_mod_action`
* K=50, paths sorted by hops, highest feasible modulation, First-Fit block selection.

## Strict v1.3
* Candidate pool: PPO-R legal-action Top-30
* Feature dim: 25
* Hidden dims: [128, 64]

## Locked environment configuration
```json
{
  "topology": "xlron_cost239_ptrnet_real",
  "num_slots": 100,
  "k_paths": 50,
  "max_blocks": 10,
  "path_sort_strategy": "hops",
  "block_sort_strategy": "start_asc",
  "modulation_profile": "default",
  "slot_bw_hz": 12500000000.0,
  "guard_band_fs": 1,
  "mean_holding_time": 10.0,
  "bitrate_min_gbps": 25,
  "bitrate_max_gbps": 100
}
```

* Arrival interval: 0.025
* Warm-up requests: 3000
* Evaluated requests: 10000
* Test seeds: [3030, 4040, 5050, 6060, 7070, 8080, 9090, 1010, 2020, 3031, 4041, 5051]
* DeepRMSA training seeds: [42, 123, 456]
