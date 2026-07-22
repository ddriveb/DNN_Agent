#!/usr/bin/env bash
# P0 smoke: COST239 high-pressure counterfactual ranking dataset
# Compare baseline continuation / overload return vs stronger PPO-R teacher vs ranker continuation.
set -euo pipefail
cd /mnt/d/project/DNN_Agent

run_smoke() {
  PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.generate_r_counterfactual_ranking_dataset "$@"
}

COMMON=(
  --topology xlron_cost239_ptrnet_real
  --num_slots 100
  --num_servers 4
  --k_paths 5
  --path_sort_strategy km
  --block_sort_strategy mixed
  --max_blocks 10
  --splits train
  --train_episodes 2
  --requests_per_episode 60
  --horizon 1
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_delayaware_v2_snap24_best.pt
  --arrival_interval 0.15
  --holding_min 4.0
  --holding_max 10.0
  --size_min_mb 5.0
  --size_max_mb 30.0
)

echo "=== P0 smoke A: baseline (ppo_r future, no overload penalty) ==="
run_smoke "${COMMON[@]}" \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --future_rollout_policy ppo_r \
  --return_future_server_overload_coef 0.0 \
  --output_dir sa_hmarl/experiments/p0_cost239_smoke/a_baseline

echo "=== P0 smoke B: strong PPO-R teacher + overload penalty ==="
run_smoke "${COMMON[@]}" \
  --agent_r_checkpoint sa_hmarl/checkpoints/ppo_r_deeprmsa_bc_snap24_best.pt \
  --future_rollout_policy ppo_r \
  --return_future_server_overload_coef 8.0 \
  --output_dir sa_hmarl/experiments/p0_cost239_smoke/b_strong_ppo_r_overload8

echo "=== P0 smoke C: ranker future (cost239 retrain v2) + overload penalty ==="
run_smoke "${COMMON[@]}" \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --trajectory_policy ppo_r \
  --future_rollout_policy ranker \
  --future_ranker_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_xlron_cost239_v1_2_retrain_v2/ranking_model.pt \
  --return_future_server_overload_coef 8.0 \
  --output_dir sa_hmarl/experiments/p0_cost239_smoke/c_ranker_future_overload8

echo "=== P0 smoke complete ==="
