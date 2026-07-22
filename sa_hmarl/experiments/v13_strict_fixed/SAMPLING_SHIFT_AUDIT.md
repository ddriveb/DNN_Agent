# Sampling Shift Audit

{
  "old_dataset": "sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_medium/metadata.json",
  "old_groups_total": "unknown",
  "old_groups_e0": "unknown",
  "old_groups_e1": "unknown",
  "old_group_filter": "implicit deep_path_only (path_idx>=5)",
  "new_dataset": "sa_hmarl/experiments/v13_strict_fixed/pilot_dataset_cost239",
  "new_groups_total": 1753,
  "new_groups_e0": 229,
  "new_groups_e1": 1524,
  "new_group_filter": "all",
  "shift_conclusion": "The old v1.3 dataset conditioned on E=1; the new full-state dataset includes E=0 states, removing the train-deployment distribution shift."
}