# SA-HMARL v1.3 Label Component Regression Audit

**Result:** PASS
**Generator elapsed:** 1196.7s

## Compared shards

| Shard | Common groups | Action IDs | Features | Returns | Reconstruct | Failures |
|---|---:|---|---|---|---|---|
| train_s1001_e0 | 581 | PASS | PASS | PASS | PASS | 0 |
| val_s2001_e0 | 429 | PASS | PASS | PASS | PASS | 0 |
| test_s3001_e0 | 602 | PASS | PASS | PASS | PASS | 0 |

## Protocol invariants

| Check | Result | Detail |
|---|---|---|
| common_groups | PASS | common=581 |
| path_indices_identical | PASS |  |
| action_id_range | PASS | max_action_id=1921 |
| path_diversity | PASS | max_path_idx=48 |
| candidate_budget | PASS | max_candidates=30 |
| trace_hash_present | PASS | n=581 |
| trace_hash_consistency | PASS | groups_with_hash=581 |
| common_groups | PASS | common=429 |
| path_indices_identical | PASS |  |
| action_id_range | PASS | max_action_id=1803 |
| path_diversity | PASS | max_path_idx=45 |
| candidate_budget | PASS | max_candidates=30 |
| trace_hash_present | PASS | n=429 |
| trace_hash_consistency | PASS | groups_with_hash=429 |
| common_groups | PASS | common=602 |
| path_indices_identical | PASS |  |
| action_id_range | PASS | max_action_id=1683 |
| path_diversity | PASS | max_path_idx=42 |
| candidate_budget | PASS | max_candidates=30 |
| trace_hash_present | PASS | n=602 |
| trace_hash_consistency | PASS | groups_with_hash=602 |

## Smoke test

Return code: 0
