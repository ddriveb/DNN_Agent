#!/usr/bin/env bash
# Quick Kmax sweep for validation/table generation (short episodes).
set -euo pipefail

cd /mnt/d/project/DNN_Agent

TOPOLOGY="${1:-xlron_cost239_ptrnet_real}"
ARRIVAL_INTERVAL="${2:-0.0625}"
EDGE_COST_MAX="${3:-2.2}"
OUT_BASE="${4:-xlron_cost239_ptrnet_real}"
SEEDS="${5:-3030,4040}"
REQS="${6:-1000}"
WARMUP="${7:-200}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

COMMON_ARGS=(
  --topology "${TOPOLOGY}"
  --num_slots 320 --num_servers 4
  --seeds "${SEEDS}"
  --episodes 1 --requests_per_episode "${REQS}" --warmup_requests "${WARMUP}"
  --poisson_arrivals --exponential_holding
  --arrival_interval "${ARRIVAL_INTERVAL}" --edge_cost_max "${EDGE_COST_MAX}"
  --block_sort_strategy start_asc --path_sort_strategy hops
  --ksp_ff_k50_hops_k_paths 50
  --ranker_candidate_mode legalctx48
  --ranker_ensure_ksp
  --ranker_enable_profile
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt
)

for K in 16 24 32 48; do
  echo "=== Quick sweep K=${K} ==="
  PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_long_horizon_system_comparison \
    "${COMMON_ARGS[@]}" \
    --ranker_max_candidates "${K}" \
    --methods "ppo_c+v12_k50_hops" \
    --output_json "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_v13_K${K}_quick.json" \
    --output_md "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_v13_K${K}_quick.md"
done

echo ""
echo "=== Quick Kmax sweep summary ==="
echo "| K | Blocking | Delay mean | Delay P95 | Decision mean | Decision P95 |"
echo "|---:|---:|---:|---:|---:|---:"
for K in 16 24 32 48; do
  PYTHONPATH=sa_hmarl .venv/bin/python -c "
import json
p = 'sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_v13_K${K}_quick.json'
with open(p) as f:
    d = json.load(f)
agg = d['methods']['ppo_c+v12_k50_hops']['aggregate']
print(f'| ${K} | {agg[\"blocking_rate\"]:.4f} | {agg[\"mean_delay_ms\"]:.2f} | {agg[\"p95_delay_ms\"]:.2f} | {agg[\"mean_decision_time_ms\"]:.2f} | {agg[\"p95_decision_time_ms\"]:.2f} |')
"
done
