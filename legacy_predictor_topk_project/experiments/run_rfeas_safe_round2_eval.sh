#!/bin/bash
# Round 2 evaluation: r_feasibility_safe on 5 eval seeds.
set -euo pipefail

cd /mnt/d/project/DNN_Agent
export PYTHONPATH=sa_hmarl

COMMON="--topology snap24_gnutella_reach \
  --num_slots 20 --num_servers 4 --max_blocks 10 --block_sort_strategy mixed \
  --num_splits 3 --split_profile default3 \
  --k_paths 5 \
  --seeds 42,123,456,789,101112 \
  --episodes 20 --requests_per_episode 80"

for seed in 42 123 456; do
  prefix="agent_c_r_feasibility_safe_s${seed}_s20_r80"
  ckpt="sa_hmarl/checkpoints/${prefix}_best.pt"
  if [ ! -f "$ckpt" ]; then
    echo "Missing checkpoint: $ckpt"
    exit 1
  fi
  echo "[$(date)] Evaluating r_feasibility_safe seed=$seed"
  .venv/bin/python -m sa_hmarl.evaluation.k5_system_comparison \
    $COMMON \
    --agent_c_checkpoint "$ckpt" \
    --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
    --out_json "sa_hmarl/experiments/k5_r_feasibility_safe_s${seed}_round2.json" \
    --out_md "sa_hmarl/experiments/k5_r_feasibility_safe_s${seed}_round2.md"
done

echo "[$(date)] Round 2 evaluation complete"
