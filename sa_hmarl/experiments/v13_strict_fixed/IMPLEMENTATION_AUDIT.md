# SA-HMARL Strict v1.3 Implementation Audit

## Scope
This audit documents the code changes made to fix the train-deployment distribution shift in the strict v1.3 counterfactual ranking pipeline.

## Files changed

### 1. `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py`
- **Removed** the implicit hard filter `any(path_idx >= 5) and any(action_id >= 200)`.
- **Added** `_keep_group(max_path_idx, group_filter)` helper.
- **Added** `--group_filter {all,deep_path_only}` CLI argument (default `all`).
- Each saved group now records:
  - `deep_path_gate` (`0/1`, i.e. E=0/E=1)
  - `max_path_idx`
  - `depth_stratum` (`0/1/2/3`)
- `_pad_groups` now emits `deep_path_gate` and `depth_stratum` arrays.
- The old invariant is replaced by an assertion `len(candidate_actions) <= max_candidates`.
- Top-level `metadata.json` records `group_filter`, `groups_e0`, `groups_e1`, `groups_total`, and `depth_stratum_counts`.

### 2. `sa_hmarl/sa_hmarl/evaluation/merge_r_counterfactual_ranking_shards.py`
- `_split_diagnostics` computes E=0/E=1 group counts, E=1 rate, and per-stratum counts.
- `DATASET_REPORT.md` includes an E-gate / depth-stratum table.
- Int arrays (`deep_path_gate`, `depth_stratum`) are preserved automatically by the object-array filter.

### 3. `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py`
- **Added** `--sampling_mode {uniform,balanced,stratified_depth}` (default `uniform`).
- `uniform` is the unweighted full-state natural distribution and the main v1.3-fix version.
- `stratified_depth` balances the four depth strata during training.
- Legacy `--balanced_sampler` flag maps to `sampling_mode=balanced` for backward compatibility.
- Added `--stratified_depth_contrast` to optionally upweight high-contrast groups within each stratum.

### 4. `sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_cost239_kpath50_hops_fair.py`
- **Added** `--ranker_gate {all,deep_path_only}` CLI argument (default `all`).
- `all`: ranker is invoked on every non-empty candidate state.
- `deep_path_only`: E=0 states bypass the ranker and use PPO-R top-1 directly.
- Added `SubgroupCounters` dataclass tracking per E-gate:
  - request count and state share
  - blocking rate (total, overload, NSB, other)
  - delay / FS averages
  - PPO-R / Ranker action agreement
  - Ranker change count
  - improvement / deterioration rates (via counterfactual PPO-R top-1 branch)
- `MethodResult.to_dict` includes `e0` and `e1` subgroups.
- Summary JSON/Markdown includes E-gate subgroup tables.

### 5. `sa_hmarl/tests/test_v13_full_state_group_filter.py`
- Tests `_depth_stratum` boundaries.
- Tests `_action_to_path_idx` with the standard action encoding.
- Tests `_keep_group` for `all` and `deep_path_only`.
- Tests `_e_gate_from_candidates`.
- Tests that legacy NPZ files without `deep_path_gate` still load.

## Protocol invariants preserved
- `K_C = 5`, `K_path = 50`, `path_sort_strategy = hops`, `block_sort_strategy = start_asc`
- `K_prop = min(30, |A_legal|)` enforced by `_ppo_r_topk_actions_with_scores` and clipping.
- `H = 5` future transitions, `gamma = 1.0`
- `candidate_mode = ppo_r_topk_only`
- 25-d pre-decision state-action features (`FEATURE_NAMES`)
- Frozen PPO-C / PPO-R continuation policies
- No KSP-FF anchors, fillers, diversity quotas, or hybrid candidates added.

## Backward compatibility
- Existing old-v1.3 datasets and checkpoints are **not overwritten**.
- `group_filter=deep_path_only` reproduces the legacy E=1-only distribution.
- Online evaluator defaults to `ranker_gate=all`; old scripts that do not pass the flag still work.
- Training script defaults to `sampling_mode=uniform`; old `--balanced_sampler` still works.
