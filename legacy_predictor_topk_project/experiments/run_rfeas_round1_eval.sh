#!/bin/bash
# Round 1 formal evaluation: default vs r_feasibility on 5 eval seeds.
set -euo pipefail

cd /mnt/d/project/DNN_Agent
export PYTHONPATH=sa_hmarl

COMMON="--topology snap24_gnutella_reach \
  --num_slots 20 --num_servers 4 --max_blocks 10 --block_sort_strategy mixed \
  --num_splits 3 --split_profile default3 \
  --k_paths 5 \
  --seeds 42,123,456,789,101112 \
  --episodes 20 --requests_per_episode 80"

run_eval() {
  local mode=$1
  local seed=$2
  local prefix="agent_c_${mode}_s${seed}_s20_r80"
  local ckpt="sa_hmarl/checkpoints/${prefix}_best.pt"
  if [ ! -f "$ckpt" ]; then
    echo "Missing checkpoint: $ckpt"
    return 1
  fi
  echo "[$(date)] Evaluating $mode seed=$seed"
  .venv/bin/python -m sa_hmarl.evaluation.k5_system_comparison \
    $COMMON \
    --agent_c_checkpoint "$ckpt" \
    --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
    --out_json "sa_hmarl/experiments/k5_${mode}_s${seed}_round1.json" \
    --out_md "sa_hmarl/experiments/k5_${mode}_s${seed}_round1.md"
}

for seed in 42 123 456; do
  for mode in default r_feasibility; do
    run_eval "$mode" "$seed"
  done
done

echo "[$(date)] Round 1 evaluation complete"
