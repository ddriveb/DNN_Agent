# UNIT TEST REPORT — Stage A

* Protocol: `DEEPRMSA_VS_KSP_FF_PAPER_PARITY_V1`
* Date: 2026-07-17
* Overall: **PASS** (10/10 tests)
* Elapsed: 1.8 s

Oracle: `deep_rmsa_parity_adapter.py` (behavioral extraction of upstream
DeepRMSA, audited in v135_r_only_standard_and_blocking_diagnosis).

| Test | Result | Key details |
|---|---|---|
| `fs_boundaries` | PASS | [{"bw": 25, "len": 624.0, "expected_mod": "16QAM", "got_mod": "16QAM", "fs": 2, "oracle_fs": 2, "pass": true}, {"bw": 25, "len": 625.0, "expected_mod": "16QAM", "got_mod": "16QAM", "fs": 2, "oracle_fs": 2, "pass": true}, {"bw": 25, "len": 626.0, "expected_mod": "8QAM", "got_mod": "8QAM", "fs": 2, "oracle_fs": 2, "pass": true}, {"bw": 100, "len": 624.0, "expected_mod": "16QAM", "got_mod": "16QAM", … |
| `direction_independence` | PASS | {"nsfnet_directed_arcs": 44, "cost239_directed_arcs": 52, "forward_occupied": true, "reverse_untouched": true, "reverse_allocation_ok": true, "conflict_rejected": true, "adjacent_ok": true} |
| `continuity_contiguity` | PASS | {"path": [0, 1, 3], "blocks_after_alloc": [[0, 40], [46, 54]], "first_fit_41": 46, "eligible_41": [[46, 54]]} |
| `first_fit` | PASS | {"static_checks": [true, true, true, true, true, true], "oracle_fuzz": "200/200"} |
| `ksp_ff_oracle_fuzz` | PASS | {"total": 113, "agree": 113, "admitted": 61, "mismatches": []} |
| `release_fifo` | PASS | {"release_order": [10, 20, 0], "tie_fifo_ok": true, "a_held_until_5": true, "released_before_arrival": true} |
| `holding_truncation` | PASS | {"n": 40000, "min": 6.819525916679872e-06, "max": 19.99937888591711, "mean": 6.850864393823785, "theoretical_mean": 6.869647145006687} |
| `uniform_all_od` | PASS | {"src_neq_dst": true, "pairs_seen": 182, "pairs_total": 182, "chi2": 183.76606666666666, "dof": 181, "p_approx": 0.44220493924877363, "bitrate_range": [25, 100], "mean_inter_arrival": 0.04006430861259163} |
| `ksp_path_order` | PASS | {"xlron_nsfnet_deeprmsa": {"od_pairs_with_different_ordering": 178, "min_paths_found_k50": 50}, "cost239_deeprmsa": {"od_pairs_with_different_ordering": 108, "min_paths_found_k50": 50}} |
| `no_excluded_components` | PASS | {"offenders": []} |
