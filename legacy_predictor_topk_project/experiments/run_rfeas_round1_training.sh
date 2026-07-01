#!/bin/bash
# Round 1 formal training: default vs r_feasibility, seeds 42/123/456
# Run up to 4 jobs concurrently.
set -euo pipefail

cd /mnt/d/project/DNN_Agent
export PYTHONPATH=sa_hmarl

COMMON="--topologies snap24_gnutella_reach \
  --num_slots 20 --num_servers 4 --max_blocks 10 --block_sort_strategy mixed \
  --num_splits 3 --split_profile default3 \
  --k_paths 5 --frozen_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt --frozen_r_type ppo \
  --episodes 300 --requests_per_episode 80 --device cpu \
  --c_valid_action_pressure_coef 0.2 --c_spectrum_block_penalty 0.3"

run_train() {
  local mode=$1
  local seed=$2
  local prefix="agent_c_${mode}_s${seed}_s20_r80"
  echo "[$(date)] Starting $mode seed=$seed -> $prefix"
  .venv/bin/python -u -m sa_hmarl.training.train_agent_c_with_frozen_r \
    $COMMON \
    --agent_c_feature_mode "$mode" \
    --seed "$seed" \
    --ckpt_prefix "$prefix" \
    --eval_freq 50 --eval_episodes 5 \
    > "experiments/${prefix}_train.log" 2>&1
  echo "[$(date)] Finished $mode seed=$seed"
}

export -f run_train
export PYTHONPATH COMMON

MAX_JOBS=4
JOBS=0
for seed in 42 123 456; do
  for mode in default r_feasibility; do
    run_train "$mode" "$seed" &
    JOBS=$((JOBS + 1))
    if [ "$JOBS" -ge "$MAX_JOBS" ]; then
      wait -n 2>/dev/null || wait
      JOBS=$((JOBS - 1))
    fi
  done
done
wait

echo "[$(date)] Round 1 training complete"
