#!/usr/bin/env bash
# Full v1.3 counterfactual ranking pipeline (overnight scale).
# Generates a larger dataset, trains the ranker, and runs the fair comparison.
set -euo pipefail

cd /mnt/d/project/DNN_Agent
export PYTHONPATH=sa_hmarl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATASET="sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_full"
CKPT="sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_full"
OLD_CKPT="sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt"
EXP="sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking"

echo "[full] Step 1/4: Generate dataset"
.venv/bin/python3 -m sa_hmarl.evaluation.generate_r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5 \
    --output_dir "${DATASET}" \
    --n_train_shards 16 \
    --n_val_shards 4 \
    --n_test_shards 4 \
    --requests_per_episode 5000 \
    --warmup 1500 \
    --max_workers 5 \
    --seeds 1001,2001,3001 \
    --device cpu

echo "[full] Step 2/4: Merge shards"
.venv/bin/python3 -m sa_hmarl.evaluation.merge_r_counterfactual_ranking_shards \
    --dataset_dir "${DATASET}"

echo "[full] Step 3/4: Train ranker"
.venv/bin/python3 -m sa_hmarl.training.train_r_counterfactual_ranking \
    --dataset_dir "${DATASET}" \
    --output_dir "${CKPT}" \
    --epochs 100 \
    --batch_size 64 \
    --eval_batch_size 256 \
    --log_every 5 \
    --device cpu

echo "[full] Step 4/4: Fair comparison"
.venv/bin/python3 -m sa_hmarl.evaluation.eval_r_counterfactual_ranking_cost239_kpath50_hops_fair \
    --seeds 4001,4002,4003 \
    --requests_per_episode 5000 \
    --warmup_requests 1500 \
    --ranker_specs "old_v13=${OLD_CKPT}" "new_v13=${CKPT}/ranking_model.pt" \
    --modes ppo_r_top1,ksp_ff_plain,ksp_ff_highest \
    --device cpu \
    --output_dir "${EXP}" \
    --output_json "fair_comparison_full.json" \
    --output_md "FAIR_COMPARISON_FULL.md"

echo "[full] Done. Final report: ${EXP}/FAIR_COMPARISON_FULL.md"
