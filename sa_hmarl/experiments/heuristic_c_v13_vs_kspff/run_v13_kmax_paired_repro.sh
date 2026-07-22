#!/usr/bin/env bash
# Paired Kmax sweep under the OLD long-horizon benchmark protocol (edge_cost_min=0.1).
# This establishes the current-code reproducible baseline and does NOT overwrite
# the original long-horizon benchmark files.
#
# Usage:
#   bash run_v13_kmax_paired_repro.sh <topology> <arrival_interval> <edge_cost_max> <output_basename>
set -euo pipefail

cd /mnt/d/project/DNN_Agent

TOPOLOGY="${1}"
ARRIVAL_INTERVAL="${2}"
EDGE_COST_MAX="${3}"
OUT_BASE="${4}"

export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export OPENBLAS_NUM_THREADS=2

COMMON_ARGS=(
  --topology "${TOPOLOGY}"
  --num_slots 320 --num_servers 4
  --seeds 3030,4040,5050,6060,7070
  --episodes 1 --requests_per_episode 10000 --warmup_requests 2000
  --poisson_arrivals --exponential_holding
  --arrival_interval "${ARRIVAL_INTERVAL}"
  --edge_cost_min 0.1
  --edge_cost_max "${EDGE_COST_MAX}"
  --block_sort_strategy start_asc --path_sort_strategy hops
  --ksp_ff_k50_hops_k_paths 50
  --ranker_candidate_mode legalctx48
  --ranker_ensure_ksp
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt
)

# Baseline KSP-FF (same C policy, same R backend as old benchmark)
echo "=== Paired baseline: ppo_c+ksp_ff_k50_hops ==="
PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_long_horizon_system_comparison \
  "${COMMON_ARGS[@]}" \
  --methods "ppo_c+ksp_ff_k50_hops" \
  --output_json "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_repro_kspff.json" \
  --output_md "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_repro_kspff.md"

# v1.3 Kmax sweep
for K in 16 24 32 48; do
  echo "=== Paired v1.3 K=${K} ==="
  PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_long_horizon_system_comparison \
    "${COMMON_ARGS[@]}" \
    --ranker_max_candidates "${K}" \
    --methods "ppo_c+v12_k50_hops" \
    --output_json "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_repro_v13_K${K}.json" \
    --output_md "sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}_repro_v13_K${K}.md"
done

echo ""
echo "=== Paired current-code repro summary (edge_cost_min=0.1) ==="
echo "| Method | Blocking mean | Blocking std | Delay mean | Delay P95 | Decision mean | Decision P95 |"
echo "|---|---:|---:|---:|---:|---:|---:"
PYTHONPATH=sa_hmarl .venv/bin/python -c "
import json, numpy as np
base='sa_hmarl/experiments/heuristic_c_v13_vs_kspff/${OUT_BASE}'
files=[(base+'_repro_kspff.json','KSP-FF')]
for K in [16,24,32,48]:
    files.append((base+'_repro_v13_K'+str(K)+'.json','v1.3-K'+str(K)))
for p,name in files:
    with open(p) as f:
        d=json.load(f)
    spec=list(d['methods'].values())[0]
    agg=spec['aggregate']
    blocks=[row['blocking_rate'] for row in spec['per_seed'].values()]
    print(f\"| {name} | {agg['blocking_rate']:.4f} | {np.std(blocks):.4f} | {agg['mean_delay_ms']:.2f} | {agg['p95_delay_ms']:.2f} | {agg['mean_decision_time_ms']:.2f} | {agg['p95_decision_time_ms']:.2f} |\")
"
