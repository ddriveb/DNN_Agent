#!/usr/bin/env bash
# Part A: JPN48 online candidate-set / latency sweep for v1.3 selection.
set -euo pipefail
cd /mnt/d/project/DNN_Agent

OUTDIR="sa_hmarl/experiments/jpn48_candidate_latency_sweep"
mkdir -p "$OUTDIR"

run_eval() {
  PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison "$@"
}

COMMON=(
  --topology xlron_jpn48
  --num_slots 100
  --num_servers 4
  --split_profile default3
  --arrival_interval 0.15
  --holding_min 4.0
  --holding_max 10.0
  --size_min_mb 5.0
  --size_max_mb 30.0
  --deadline_min 30.0
  --deadline_max 100.0
  --edge_cost_min 0.5
  --edge_cost_max 15.0
  --seeds 3030,4040,5050,6060,7070
  --episodes 20
  --requests_per_episode 80
  --max_blocks 10
)

run_cfg() {
  local label=$1
  shift
  echo "=== $label ==="
  run_eval "${COMMON[@]}" "$@" \
    --output_json "$OUTDIR/${label}.json" \
    --output_md "$OUTDIR/${label}.md"
}

run_cfg baseline_v12 \
  --methods ppo_c+v12 \
  --k_paths 5 \
  --path_sort_strategy km \
  --block_sort_strategy mixed \
  --ranker_candidate_mode all_legal

run_cfg v12_k20_hops \
  --methods ppo_c+v12_k50_hops \
  --k_paths 5 \
  --path_sort_strategy km \
  --block_sort_strategy mixed \
  --ksp_ff_k50_hops_k_paths 20 \
  --ranker_candidate_mode all_legal

run_cfg v12_k30_hops \
  --methods ppo_c+v12_k50_hops \
  --k_paths 5 \
  --path_sort_strategy km \
  --block_sort_strategy mixed \
  --ksp_ff_k50_hops_k_paths 30 \
  --ranker_candidate_mode all_legal

run_cfg v12_k50_hops_all_legal \
  --methods ppo_c+v12_k50_hops \
  --k_paths 5 \
  --path_sort_strategy km \
  --block_sort_strategy mixed \
  --ksp_ff_k50_hops_k_paths 50 \
  --ranker_candidate_mode all_legal

run_cfg v12_k50_hops_legalctx48 \
  --methods ppo_c+v12_k50_hops \
  --k_paths 5 \
  --path_sort_strategy km \
  --block_sort_strategy mixed \
  --ksp_ff_k50_hops_k_paths 50 \
  --ranker_candidate_mode legalctx48 \
  --ranker_max_candidates 48 \
  --ranker_ppo_top_k 8 \
  --ranker_num_random_candidates 5 \
  --ranker_min_candidates 15

run_cfg v12_k50_hops_legalctx48_ensure_ksp \
  --methods ppo_c+v12_k50_hops \
  --k_paths 5 \
  --path_sort_strategy km \
  --block_sort_strategy mixed \
  --ksp_ff_k50_hops_k_paths 50 \
  --ranker_candidate_mode legalctx48 \
  --ranker_max_candidates 48 \
  --ranker_ppo_top_k 8 \
  --ranker_num_random_candidates 5 \
  --ranker_min_candidates 15 \
  --ranker_ensure_ksp

echo "=== JPN48 sweep complete ==="
