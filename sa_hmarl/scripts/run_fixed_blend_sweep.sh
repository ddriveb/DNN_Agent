#!/bin/bash
# Fixed blend alpha sweep for fixed_blend_typed_mean_field Agent-C.
# Usage: bash sa_hmarl/scripts/run_fixed_blend_sweep.sh
set -euo pipefail

cd "$(dirname "$0")/../.."

export PYTHONPATH=sa_hmarl
PYTHON=".venv/bin/python"
TRAINS="sa_hmarl.training.train_agent_c_with_frozen_r"

ALPHAS=(0.00 0.25 0.50 0.75 1.00)
SEEDS=(42 123)

for seed in "${SEEDS[@]}"; do
  for alpha in "${ALPHAS[@]}"; do
    alpha_str=$(printf "a%.2f" "$alpha" | tr '.' '_')
    prefix="agent_c_fixed_blend_${alpha_str}_s${seed}_s20_r80"
    echo "=== Training seed=$seed alpha=$alpha prefix=$prefix ==="
    $PYTHON -m "$TRAINS" \
      --topologies snap24_gnutella_reach \
      --num_slots 20 --num_servers 4 --max_blocks 10 --block_sort_strategy mixed \
      --k_paths 5 --agent_c_feature_mode fixed_blend_typed_mean_field \
      --fixed_blend_alpha "$alpha" \
      --frozen_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
      --episodes 300 --requests_per_episode 80 --arrival_interval 0.2 \
      --seed "$seed" --eval_freq 50 --eval_episodes 5 \
      --ckpt_prefix "$prefix"
  done
done

echo "=== Sweep complete ==="
