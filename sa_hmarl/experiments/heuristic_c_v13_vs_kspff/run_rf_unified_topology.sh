#!/usr/bin/env bash
# Run rf_c unified main-table comparison for one imported topology.
# Usage: bash run_rf_unified_topology.sh <topology> <arrival_interval> <edge_cost_max> <output_basename>
set -euo pipefail

cd /mnt/d/project/DNN_Agent

TOPOLOGY="${1}"
ARRIVAL_INTERVAL="${2}"
EDGE_COST_MAX="${3}"
OUT_BASE="${4}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_long_horizon_system_comparison \
  --topology "${TOPOLOGY}" \
  --num_slots 320 --num_servers 4 \
  --methods "rf_c+ksp_ff_k50_hops,rf_c+v12_k50_hops" \
  --seeds 3030,4040,5050,6060,7070 \
  --episodes 1 --requests_per_episode 10000 --warmup_requests 2000 \
  --poisson_arrivals --exponential_holding \
  --arrival_interval "${ARRIVAL_INTERVAL}" --edge_cost_max "${EDGE_COST_MAX}" \
  --block_sort_strategy start_asc --path_sort_strategy hops \
  --ksp_ff_k50_hops_k_paths 50 \
  --ranker_candidate_mode legalctx48 --ranker_max_candidates 48 --ranker_ensure_ksp \
  --output_json "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_rf_unified_full.json" \
  --output_md "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_rf_unified_full.md"
