# IMMUTABLE PROTOCOL AND NAMING LOCK

This file must be placed at the beginning of any subsequent prompt or hand-off to another AI. It must not be redefined, paraphrased, or ignored.

## KSP-FF K=50 hops

The terms "KSP-FF", "KSP-FF K=50", and "KSP-FF K=50 hops" uniquely refer to:

- Function: `sa_hmarl.baselines.rmsa_baselines.ksp_ff_highest_mod_action`
- File: `sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py`, lines 32–84

Fixed protocol:

- K_path = 50
- path_sort_strategy = hops
- Tie-break on equal hop count: path length (km)
- Per path: select highest feasible modulation (smallest required FS)
- block_sort_strategy = start_asc
- Spectrum assignment = First-Fit
- Stop at the first path that has a legal RMSA action

`ksp_ff_action` (`sa_hmarl/sa_hmarl/baselines/rmsa_baselines.py:19–29`) is **not** KSP-FF. It must be called:

- Naive Flat-FF

Under no circumstances may Naive Flat-FF results be used to claim superiority over KSP-FF.

## Strict v1.3

The term "v1.3" in this project uniquely refers to the current loss-matched configuration:

- SA-HMARL Strict v1.3 full-state / all-gate

Fixed protocol:

- K_C = 5
- K_path = 50
- path_sort_strategy_c = hops
- path_sort_strategy_r = hops
- block_sort_strategy = start_asc
- K_prop = 30
- candidate_mode = ppo_r_topk_only
- Candidate set is taken only from the frozen PPO-R Top-30 legal actions
- No KSP anchor / filler / diversity candidates added
- H = 5
- gamma = 1.0
- common-future counterfactual rollout
- group_filter = all
- sampling_mode = uniform
- ranker_gate = all
- MLP hidden_dims = [128, 64]
- reg_weight = 1.0
- lambda_pair = 0.0
- lambda_hard = 0.0
- Checkpoint selected by validation regret
- It is forbidden to select a checkpoint based on test or closed-loop results

The old E1-only checkpoint must be called:

- Legacy E1-trained gated baseline

It must not be abbreviated as the current v1.3.
