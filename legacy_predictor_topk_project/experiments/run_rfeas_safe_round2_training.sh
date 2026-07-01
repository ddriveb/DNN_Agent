#!/bin/bash
# Round 2: r_feasibility_safe, seeds 42/123/456
# Run up to 3 jobs concurrently.
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
  local seed=$1
  local prefix="agent_c_r_feasibility_safe_s${seed}_s20_r80"
  echo "[$(date)] Starting r_feasibility_safe seed=$seed -> $prefix"
  .venv/bin/python -u -m sa_hmarl.training.train_agent_c_with_frozen_r \
    $COMMON \
    --agent_c_feature_mode r_feasibility_safe \
    --seed "$seed" \
    --ckpt_prefix "$prefix" \
    --eval_freq 50 --eval_episodes 5 \
    > "experiments/${prefix}_train.log" 2>&1
  echo "[$(date)] Finished r_feasibility_safe seed=$seed"
}

export -f run_train
export PYTHONPATH COMMON

MAX_JOBS=3
JOBS=0
for seed in 42 123 456; do
  run_train "$seed" &
  JOBS=$((JOBS + 1))
  if [ "$JOBS" -ge "$MAX_JOBS" ]; then
    wait -n 2>/dev/null || wait
    JOBS=$((JOBS - 1))
  fi
done
wait

echo "[$(date)] Round 2 training complete"
