# SA-HMARL v1.3 Counterfactual Dataset Smoke Test

**Result:** PASS
**Elapsed:** 262.4s

## Summary

```json
{
  "groups": 150,
  "nonempty_groups": 150,
  "max_action_id": 1481,
  "max_path_idx": 37,
  "cand_counts": {
    "min": 30,
    "max": 30,
    "mean": 30.0
  }
}
```

## Invariants

| Invariant | Pass | Detail |
|---|---|---|
| metadata_K_constants | PASS | K_C=5 K_path=50 K_prop=30 H=5 |
| feature_dim | PASS | feature_dim=25 names=25 |
| nonempty_groups | PASS | groups=150 nonempty=150 |
| mask_length | PASS | mask_len=30 |
| candidate_budget | PASS | min=30 max=30 |
| action_id_range | PASS | min=0 max=1481 |
| path_diversity | PASS | max_path_idx=37 max_action_id=1481 |
| provenance_ppo_r_topk | PASS | unique=[1] |
| finite_features | PASS | non-finite feature values detected |
| finite_returns | PASS | non-finite return values detected |
| trace_hash_present | PASS | trace_hashes=150 |
| unique_group_ids | PASS | unique=150 groups=150 |

## Generator stdout (last 80 lines)

```text
[launcher] total=1 done=0 todo=1
[launcher] Using max_workers=1
[launcher] Finished: ok=1 skipped=0 error=0 elapsed=4.14min
[dataset] Wrote top-level metadata to /tmp/v13_kpath50_smoke_hsr2m5xe/metadata.json
[dataset] Finished: ok=1 error=0 shards=1
```

## Generator stderr (last 40 lines)

```text
```