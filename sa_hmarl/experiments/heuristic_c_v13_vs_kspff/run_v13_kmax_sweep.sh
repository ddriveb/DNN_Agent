#!/usr/bin/env bash
# Sweep ranker_max_candidates for v1.3 as a new variant/ablation.
# This does NOT replace the main v1.3 (K=48)口径; it produces a blocking-latency trade-off table.
#
# Usage:
#   bash run_v13_kmax_sweep.sh <topology> <arrival_interval> <edge_cost_max> <output_basename>
set -euo pipefail

cd /mnt/d/project/DNN_Agent

TOPOLOGY="${1}"
ARRIVAL_INTERVAL="${2}"
EDGE_COST_MAX="${3}"
OUT_BASE="${4}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

# Fixed v1.3 backbone; only K varies.
COMMON_ARGS=(
  --topology "${TOPOLOGY}"
  --num_slots 320 --num_servers 4
  --seeds 3030,4040,5050,6060,7070
  --episodes 1 --requests_per_episode 10000 --warmup_requests 2000
  --poisson_arrivals --exponential_holding
  --arrival_interval "${ARRIVAL_INTERVAL}" --edge_cost_max "${EDGE_COST_MAX}"
  --block_sort_strategy start_asc --path_sort_strategy hops
  --ksp_ff_k50_hops_k_paths 50
  --ranker_candidate_mode legalctx48
  --ranker_ensure_ksp
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt
)

for K in 16 24 32 48; do
  echo "=== Running v1.3 K=${K} ==="
  PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_long_horizon_system_comparison \
    "${COMMON_ARGS[@]}" \
    --ranker_max_candidates "${K}" \
    --methods "ppo_c+v12_k50_hops" \
    --output_json "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_v13_K${K}.json" \
    --output_md "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_v13_K${K}.md"
done

echo "=== Sweep complete. Summaries: ==="
for K in 16 24 32 48; do
  PYTHONPATH=sa_hmarl .venv/bin/python -c "
	import json
	p = 'sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_v13_K${K}.json'
	with open(p) as f:
	    d = json.load(f)
	agg = d['methods']['ppo_c+v12_k50_hops']['aggregate']
	print(f'K=${K}: block={agg[\"blocking_rate\"]:.4f} delay_mean={agg[\"mean_delay_ms\"]:.2f} decision_mean={agg[\"mean_decision_time_ms\"]:.2f} decision_p95={agg[\"p95_decision_time_ms\"]:.2f}')
	"
done
