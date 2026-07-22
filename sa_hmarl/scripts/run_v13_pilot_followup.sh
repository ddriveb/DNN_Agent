#!/usr/bin/env bash
# Follow-up pipeline for the v1.3 pilot dataset.
# Waits until the pilot generator has written all 4 ok shards, then merges,
# trains a ranker, and runs the fair comparison against PPO-R, KSP-FF and the
# old v1.3 ranker.
set -euo pipefail

cd /mnt/d/project/DNN_Agent
export PYTHONPATH=sa_hmarl
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

DATASET="sa_hmarl/datasets/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5"
CKPT="sa_hmarl/checkpoints/r_counterfactual_ranking_cost239_kpath50_hops_kprop30_h5_pilot"
OLD_CKPT="sa_hmarl/checkpoints/r_counterfactual_ranking_postdec_topk_k50_s100/ranking_model.pt"
EXP="sa_hmarl/experiments/v13_kpath50_hops_counterfactual_reranking"

echo "[followup] Waiting for pilot dataset to finish: ${DATASET}"
while true; do
    n_ok=$(python3 - <<PY
import json, sys, pathlib
n = 0
for p in pathlib.Path('${DATASET}/shards').glob('*.metadata.json'):
    try:
        if json.loads(p.read_text()).get('status') == 'ok':
            n += 1
    except Exception:
        pass
sys.stdout.write(str(n))
PY
)
    if [[ "$n_ok" -ge 4 ]]; then
        echo "[followup] Pilot dataset ready with ${n_ok} ok shards."
        break
    fi
    sleep 30
done

echo "[followup] Merging shards..."
.venv/bin/python3 -m sa_hmarl.evaluation.merge_r_counterfactual_ranking_shards \
    --dataset_dir "${DATASET}"

echo "[followup] Training ranker on pilot data..."
.venv/bin/python3 -m sa_hmarl.training.train_r_counterfactual_ranking \
    --dataset_dir "${DATASET}" \
    --output_dir "${CKPT}" \
    --epochs 80 \
    --batch_size 64 \
    --eval_batch_size 256 \
    --log_every 5 \
    --device cpu

echo "[followup] Running fair comparison..."
.venv/bin/python3 -m sa_hmarl.evaluation.eval_r_counterfactual_ranking_cost239_kpath50_hops_fair \
    --seeds 4001,4002 \
    --requests_per_episode 2000 \
    --warmup_requests 500 \
    --ranker_specs "old_v13=${OLD_CKPT}" "new_v13=${CKPT}/ranking_model.pt" \
    --modes ppo_r_top1,ksp_ff_plain,ksp_ff_highest \
    --device cpu \
    --output_dir "${EXP}" \
    --output_json "fair_comparison_full.json" \
    --output_md "FAIR_COMPARISON_FULL.md"

echo "[followup] Done. Report: ${EXP}/FAIR_COMPARISON_FULL.md"
