# RL Cleanup Audit

This audit does not delete files. The retained mainline is frozen Strict v1.3:
PPO-R legal Top-30 plus the 25-D common-future counterfactual Ranker.

## Must keep for Strict v1.3

- `sa_hmarl/sa_hmarl/agents/ppo_agents.py`
- `sa_hmarl/sa_hmarl/agents/r_agent.py`
- `sa_hmarl/sa_hmarl/agents/c_agent.py`
- `sa_hmarl/sa_hmarl/agents/counterfactual_r_ranker.py`
- `sa_hmarl/sa_hmarl/evaluation/strict_v13_online.py`
- `sa_hmarl/sa_hmarl/evaluation/generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5.py`
- `sa_hmarl/sa_hmarl/evaluation/eval_r_counterfactual_ranking_cost239_kpath50_hops_fair.py`
- `sa_hmarl/sa_hmarl/evaluation/generate_multitopology_strict_v13_vs_heuristics.py`
- `sa_hmarl/sa_hmarl/training/train_r_counterfactual_ranking.py`
- shared environment, observation, action-mask, topology, KSP and RMSA baseline code
- `sa_hmarl/checkpoints/agent_r_mixed.pt`
- the canonical PPO-C checkpoint used by the retained evaluation protocol
- `sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt`
- canonical v1.3 datasets, protocol locks and final reports

## Direct cleanup candidates

### DeepRMSA family

- agents: `deep_rmsa_agent.py`, `deep_rmsa_adapted_k50_agent.py`,
  `deep_rmsa_source_semantic_agent.py`
- training/evaluation scripts whose names contain `deep_rmsa` or `deeprmsa`
- DeepRMSA checkpoints in `sa_hmarl/checkpoints/`
- experiments: `deeprmsa_*`, `strict_v13_vs_ksp_ff_vs_deeprmsa_*`,
  `topology_sweep_s100_v12_deep_verify`
- DeepRMSA-specific tests after shared baseline tests are separated

### v1.35 family

- evaluation scripts named `v135_*`, `analyze_v135_*`, `slice_v135_*`
- experiment directories named `v135_*`
- dataset directories named `v135_*`

### v2.0 family

- evaluation scripts named `*v20*`
- experiment directories named `v20_*` or `v20e_*`
- sliding-window, multiscale-credit and utility-gate generated datasets

### TD/post-decision value family

- `agents/post_decision_value.py`
- `agents/post_decision_finalizer_policy.py`
- `training/train_r_td_post_decision_value.py`
- `evaluation/collect_r_td_post_decision_replay.py`
- `evaluation/diagnose_r_td_post_decision.py`
- `evaluation/eval_r_td_post_decision_closed_loop.py`
- other R-postdecision checkpoints/datasets after confirming they are not the
  canonical v1.3 25-D Ranker data

### Joint and alternative-policy RL

- `training/train_joint_mappo.py`
- `training/train_joint_mappo_dual_critic.py`
- `evaluation/eval_joint_mappo.py`
- fragment-aware PPO, future-aware selector and iterative-refinement studies
- experiment directories `r_iterative_refinement_*`

## Archive before deleting

### Pure-RMSA v1.3-R

This is topology-matched native retraining, not frozen Strict v1.3. It includes:

- `training/train_pure_rmsa_native_proposer.py`
- `training/train_pure_rmsa_native_ranker.py`
- `evaluation/diagnose_pure_rmsa_native_common_future.py`
- `evaluation/eval_pure_rmsa_native_ranker.py`
- `evaluation/run_pure_rmsa_native_common_future_dataset.py`
- experiments `pure_rmsa_native_retrain_r1` and `pure_rmsa_native_retrain_r2`

Its proposer produced the verified 1.70% pure-RMSA result, so retain a compressed
results/checkpoint archive if that comparison may appear in a paper or rebuttal.

### PPO-C variants

Do not delete all Agent-C code or checkpoints. Strict v1.3 native SA-HMARL
evaluation uses PPO-C. Keep the canonical checkpoint named in the protocol;
archive or remove only obsolete default/overload/smoke/step variants after a
checkpoint provenance table is produced.

## Approximate removable result/data size

- DeepRMSA experiment families: about 39 MB, excluding root checkpoints.
- Pure-RMSA native R1/R2: about 44 MB.
- v1.35 experiment plus listed datasets: at least 30 MB.
- v2.0 experiment families: about 37 MB.
- Iterative refinement: about 1 MB.

The first cleanup pass can therefore recover roughly 150 MB before counting
all logs, duplicated checkpoints and temporary output.

## Safety rule

Before deletion, create an explicit path manifest and verify that no retained
Strict v1.3 module imports any candidate file. Existing dirty-worktree changes
must not be reverted or folded into cleanup.
