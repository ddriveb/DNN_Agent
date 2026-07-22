# SA-HMARL v1.3 Counterfactual Dataset Smoke Test

**Result:** FAIL
**Elapsed:** 14.7s

## Summary

```json
{}
```

## Invariants

| Invariant | Pass | Detail |
|---|---|---|

## Failures

- generator crashed

## Generator stdout (last 80 lines)

```text
```

## Generator stderr (last 40 lines)

```text
Traceback (most recent call last):
  File "<frozen runpy>", line 198, in _run_module_as_main
  File "<frozen runpy>", line 88, in _run_code
  File "/mnt/d/project/DNN_Agent/sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py", line 880, in <module>
    main()
    ~~~~^^
  File "/mnt/d/project/DNN_Agent/sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py", line 766, in main
    "agent_c": _sha256_file(args.agent_c_checkpoint),
               ~~~~~~~~~~~~^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/mnt/d/project/DNN_Agent/sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py", line 82, in _sha256_file
    with open(path, "rb") as f:
         ~~~~^^^^^^^^^^^^
FileNotFoundError: [Errno 2] No such file or directory: 'sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt'
```