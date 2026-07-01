#!/bin/bash
# Compare default, typed_mean_field, candidate_mean_field, candidate_mean_field_count_only
# on seeds 42 and 123.
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=sa_hmarl
PYTHON=".venv/bin/python"
TRAINS="sa_hmarl.training.train_agent_c_with_frozen_r"

MODES=(
  "default"
  "typed_mean_field"
  "candidate_mean_field"
  "candidate_mean_field_count_only"
)
SEEDS=(42 123)

for seed in "${SEEDS[@]}"; do
  for mode in "${MODES[@]}"; do
    prefix="agent_c_${mode}_s${seed}_s20_r80"
    echo "=== Training seed=$seed mode=$mode prefix=$prefix ==="
    $PYTHON -m "$TRAINS" \
      --topologies snap24_gnutella_reach \
      --num_slots 20 --num_servers 4 --max_blocks 10 --block_sort_strategy mixed \
      --k_paths 5 --agent_c_feature_mode "$mode" \
      --frozen_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
      --episodes 300 --requests_per_episode 80 --arrival_interval 0.2 \
      --seed "$seed" --eval_freq 50 --eval_episodes 5 \
      --ckpt_prefix "$prefix"
  done
done

echo "=== Sweep complete ==="
