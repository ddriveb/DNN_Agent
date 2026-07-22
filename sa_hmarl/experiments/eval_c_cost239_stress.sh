#!/usr/bin/env bash
# Evaluate three Agent-C feature modes on COST239 under standard/medium/heavy stress.
# Usage: bash eval_c_cost239_stress.sh
set -euo pipefail
cd /mnt/d/project/DNN_Agent

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_default_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.15 --holding_min 4.0 --holding_max 10.0 \
  --size_min_mb 5.0 --size_max_mb 30.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_default_standard.json \
  --output_md sa_hmarl/experiments/cost239_default_standard.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.15 --holding_min 4.0 --holding_max 10.0 \
  --size_min_mb 5.0 --size_max_mb 30.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_rfeas_standard.json \
  --output_md sa_hmarl/experiments/cost239_rfeas_standard.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_overload_aware_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.15 --holding_min 4.0 --holding_max 10.0 \
  --size_min_mb 5.0 --size_max_mb 30.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_overload_standard.json \
  --output_md sa_hmarl/experiments/cost239_overload_standard.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_default_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.10 --holding_min 4.0 --holding_max 8.0 \
  --size_min_mb 10.0 --size_max_mb 40.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_default_medium.json \
  --output_md sa_hmarl/experiments/cost239_default_medium.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.10 --holding_min 4.0 --holding_max 8.0 \
  --size_min_mb 10.0 --size_max_mb 40.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_rfeas_medium.json \
  --output_md sa_hmarl/experiments/cost239_rfeas_medium.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_overload_aware_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.10 --holding_min 4.0 --holding_max 8.0 \
  --size_min_mb 10.0 --size_max_mb 40.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_overload_medium.json \
  --output_md sa_hmarl/experiments/cost239_overload_medium.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_default_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.07 --holding_min 4.0 --holding_max 6.0 \
  --size_min_mb 15.0 --size_max_mb 50.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_default_heavy.json \
  --output_md sa_hmarl/experiments/cost239_default_heavy.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.07 --holding_min 4.0 --holding_max 6.0 \
  --size_min_mb 15.0 --size_max_mb 50.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_rfeas_heavy.json \
  --output_md sa_hmarl/experiments/cost239_rfeas_heavy.md

PYTHONPATH=sa_hmarl .venv/bin/python -m sa_hmarl.evaluation.eval_main_s100_system_comparison \
  --topology xlron_cost239_ptrnet_real \
  --num_slots 100 --num_servers 4 --num_splits 3 --split_profile default3 \
  --k_paths 5 --max_blocks 10 --block_sort_strategy mixed --path_sort_strategy km \
  --agent_c_checkpoint sa_hmarl/checkpoints/agent_c_cost239_overload_aware_best.pt \
  --agent_r_checkpoint sa_hmarl/checkpoints/agent_r_mixed.pt \
  --ranking_checkpoint sa_hmarl/checkpoints/r_counterfactual_ranking_v1_2_mixed_low/ranking_model.pt \
  --ranker_ensure_ksp \
  --methods "ppo_c+v12_k50_hops,ppo_c+ksp_ff_k50_hops" \
  --seeds "3030,4040,5050" --episodes 10 --requests_per_episode 80 \
  --arrival_interval 0.07 --holding_min 4.0 --holding_max 6.0 \
  --size_min_mb 15.0 --size_max_mb 50.0 \
  --collect_server_diagnostics \
  --output_json sa_hmarl/experiments/cost239_overload_heavy.json \
  --output_md sa_hmarl/experiments/cost239_overload_heavy.md

echo "All COST239 C-mode evaluations complete."
