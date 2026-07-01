#!/bin/bash
# Auto-pipeline: replay buffer -> offline DQN -> evaluation
set -e

cd "$(dirname "$0")"
source ../../.venv/bin/activate

echo "========================================"
echo "PIPELINE: Replay -> Offline DQN -> Eval"
echo "========================================"

# Step 1: Collect replay buffer
echo "[1/3] Collecting replay buffer (30K)..."
python collect_replay.py > results/collect_replay_v2.log 2>&1
echo "[1/3] Done."

# Step 2: Train offline DQN
echo "[2/3] Training offline DQN..."
python train_offline_dqn.py > results/train_offline_dqn.log 2>&1
echo "[2/3] Done."

# Step 3: Evaluate
echo "[3/3] Evaluating offline DQN..."
python eval_dqn.py > results/eval_offline_dqn.log 2>&1
echo "[3/3] Done."

echo ""
echo "========================================"
echo "PIPELINE COMPLETE"
echo "========================================"
echo "Results: results/eval_offline_dqn.log"
echo "Checkpoint: checkpoints/offline_dqn.pt"
