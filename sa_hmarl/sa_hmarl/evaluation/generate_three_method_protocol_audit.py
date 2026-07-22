#!/usr/bin/env python3
"""Generate PROTOCOL_AUDIT.md and EXPERIMENT_MANIFEST.json for the COST239
fixed-C / all-OD three-method RMSA comparison.
"""
import hashlib
import json
from pathlib import Path

ROOT = Path("/mnt/d/project/DNN_Agent")
OUT = ROOT / "sa_hmarl/experiments/strict_v13_vs_ksp_ff_vs_deeprmsa_cost239_fixed_c_all_od/protocol"
OUT.mkdir(parents=True, exist_ok=True)

PROTOCOL = {
    "experiment_id": "strict_v13_vs_ksp_ff_vs_deeprmsa_cost239_fixed_c_all_od",
    "scenario": "fixed-C / all-OD pure RMSA",
    "topology": "xlron_cost239_ptrnet_real",
    "num_slots": 320,
    "num_servers": 4,
    "server_node_ids": [0, 1, 2, 3],
    "fixed_split_id": 0,
    "k_paths_r": 50,
    "path_sort_strategy_r": "hops",
    "block_sort_strategy_r": "start_asc",
    "max_blocks": 10,
    "modulation_profile": "default",
    "n_modulations": 4,
    "unified_action_space_size": 50 * 4 * 10,
    "action_encoding": "a = p * (|M| * B) + m * B + b",
    "arrival_interval": 0.3,
    "holding_min": 20.0,
    "holding_max": 30.0,
    "deadline_min": 30.0,
    "deadline_max": 100.0,
    "size_min_mb": 5.0,
    "size_max_mb": 30.0,
    "edge_cost_min": 0.1,
    "edge_cost_max": 2.2,
    "split_profile": "default3",
    "num_splits": 3,
    "warmup_requests": 500,
    "requests_per_episode": 6000,
    "poisson_arrivals": False,
    "exponential_holding": False,
    "traffic_matrix": "uniform_all_od",
    "evaluation_seeds": [5001, 5002, 5003, 5004, 5005],
    "no_ppo_c_loaded": True,
    "no_ppo_c_action_selected": True,
    "no_df_c": True,
    "methods": {
        "strict_v13": {
            "name": "Strict v1.3",
            "ppo_r_checkpoint": "sa_hmarl/checkpoints/agent_r_mixed.pt",
            "ranker_checkpoint": "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt",
            "candidate_pool": "ppo_r_legal_topk_only",
            "topk": 30,
            "feature_dim": 25,
        },
        "ksp_ff_k50_hops": {
            "name": "KSP-FF K=50 hops",
            "function": "ksp_ff_highest_mod_action",
            "k_path": 50,
            "path_sort_strategy": "hops",
            "block_sort_strategy": "start_asc",
            "max_blocks": 10,
        },
        "deep_rmsa_adapted_k50": {
            "name": "Topology-matched adapted DeepRMSA K=50 hops",
            "base_architecture": "DeepRMSA 5-layer 128-unit ELU MLP A2C",
            "output_dim": 2000,
            "train_from_scratch": True,
            "no_pretrained_checkpoint": True,
            "action_mask": "same physical R mask as PPO-R",
        },
    },
}

def _sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

# Hashes of key configurations
topology_config_text = json.dumps({
    "topology": PROTOCOL["topology"],
    "num_slots": PROTOCOL["num_slots"],
    "k_paths_r": PROTOCOL["k_paths_r"],
    "path_sort_strategy_r": PROTOCOL["path_sort_strategy_r"],
    "block_sort_strategy_r": PROTOCOL["block_sort_strategy_r"],
    "max_blocks": PROTOCOL["max_blocks"],
    "modulation_profile": PROTOCOL["modulation_profile"],
}, sort_keys=True)

action_space_text = json.dumps({
    "k_path": 50,
    "n_modulations": 4,
    "max_blocks": 10,
    "encoding": PROTOCOL["action_encoding"],
}, sort_keys=True)

traffic_text = json.dumps({
    "traffic_matrix": PROTOCOL["traffic_matrix"],
    "arrival_interval": PROTOCOL["arrival_interval"],
    "holding_min": PROTOCOL["holding_min"],
    "holding_max": PROTOCOL["holding_max"],
    "deadline_min": PROTOCOL["deadline_min"],
    "deadline_max": PROTOCOL["deadline_max"],
    "size_min_mb": PROTOCOL["size_min_mb"],
    "size_max_mb": PROTOCOL["size_max_mb"],
    "edge_cost_min": PROTOCOL["edge_cost_min"],
    "edge_cost_max": PROTOCOL["edge_cost_max"],
    "split_profile": PROTOCOL["split_profile"],
    "num_splits": PROTOCOL["num_splits"],
    "warmup_requests": PROTOCOL["warmup_requests"],
    "requests_per_episode": PROTOCOL["requests_per_episode"],
    "poisson_arrivals": PROTOCOL["poisson_arrivals"],
    "exponential_holding": PROTOCOL["exponential_holding"],
}, sort_keys=True)

PROTOCOL["hashes"] = {
    "topology_config_sha256": _sha256_text(topology_config_text),
    "action_space_sha256": _sha256_text(action_space_text),
    "traffic_config_sha256": _sha256_text(traffic_text),
}

# Write manifest
manifest_path = OUT / "EXPERIMENT_MANIFEST.json"
with open(manifest_path, "w", encoding="utf-8") as f:
    json.dump(PROTOCOL, f, indent=2)

# Write audit markdown
audit_lines = [
    "# Protocol Audit: COST239 fixed-C / all-OD Three-Method RMSA Comparison",
    "",
    "## Scenario",
    "",
    "This experiment compares three RMSA methods under fixed-C / all-OD pure RMSA:",
    "",
    "1. Strict v1.3",
    "2. KSP-FF K=50 hops",
    "3. Topology-matched adapted DeepRMSA K=50 hops",
    "",
    "## Protocol Lock Statement",
    "",
    "> This is fixed-C / all-OD pure RMSA. PPO-C and DF_C are not invoked.",
    "> `split_id` is fixed to 0 and `server_id` is deterministically mapped from `dst_node`.",
    "",
    "## Configuration",
    "",
    "| Parameter | Value |",
    "|---|---|",
    f"| Topology | `{PROTOCOL['topology']}` |",
    f"| Slots | {PROTOCOL['num_slots']} |",
    f"| Servers | {PROTOCOL['num_servers']} |",
    f"| Server nodes | {PROTOCOL['server_node_ids']} |",
    f"| K_path (R) | {PROTOCOL['k_paths_r']} |",
    f"| Path sort | {PROTOCOL['path_sort_strategy_r']} |",
    f"| Block sort | {PROTOCOL['block_sort_strategy_r']} |",
    f"| Max blocks | {PROTOCOL['max_blocks']} |",
    f"| Modulations | {PROTOCOL['n_modulations']} |",
    f"| Unified action space | {PROTOCOL['unified_action_space_size']} |",
    f"| Arrival interval | {PROTOCOL['arrival_interval']} |",
    f"| Holding time | [{PROTOCOL['holding_min']}, {PROTOCOL['holding_max']}] |",
    f"| Warmup | {PROTOCOL['warmup_requests']} |",
    f"| Evaluated requests | {PROTOCOL['requests_per_episode']} |",
    f"| Poisson arrivals | {PROTOCOL['poisson_arrivals']} |",
    f"| Exponential holding | {PROTOCOL['exponential_holding']} |",
    f"| Traffic matrix | {PROTOCOL['traffic_matrix']} |",
    "",
    "## Verification Hashes",
    "",
    f"- Topology config SHA256: `{PROTOCOL['hashes']['topology_config_sha256']}`",
    f"- Action space SHA256: `{PROTOCOL['hashes']['action_space_sha256']}`",
    f"- Traffic config SHA256: `{PROTOCOL['hashes']['traffic_config_sha256']}`",
    "",
    "## Methods",
    "",
    "### Strict v1.3",
    "",
    f"- PPO-R checkpoint: `{PROTOCOL['methods']['strict_v13']['ppo_r_checkpoint']}`",
    f"- Ranker checkpoint: `{PROTOCOL['methods']['strict_v13']['ranker_checkpoint']}`",
    f"- Candidate pool: {PROTOCOL['methods']['strict_v13']['candidate_pool']}",
    f"- Top-K: {PROTOCOL['methods']['strict_v13']['topk']}",
    f"- Feature dim: {PROTOCOL['methods']['strict_v13']['feature_dim']}",
    "",
    "### KSP-FF K=50 hops",
    "",
    f"- Function: `{PROTOCOL['methods']['ksp_ff_k50_hops']['function']}`",
    f"- K_path: {PROTOCOL['methods']['ksp_ff_k50_hops']['k_path']}",
    f"- Path sort: {PROTOCOL['methods']['ksp_ff_k50_hops']['path_sort_strategy']}",
    f"- Block sort: {PROTOCOL['methods']['ksp_ff_k50_hops']['block_sort_strategy']}",
    "",
    "### Topology-matched adapted DeepRMSA K=50 hops",
    "",
    f"- Base architecture: {PROTOCOL['methods']['deep_rmsa_adapted_k50']['base_architecture']}",
    f"- Output dim: {PROTOCOL['methods']['deep_rmsa_adapted_k50']['output_dim']}",
    f"- Train from scratch: {PROTOCOL['methods']['deep_rmsa_adapted_k50']['train_from_scratch']}",
    f"- Action mask: {PROTOCOL['methods']['deep_rmsa_adapted_k50']['action_mask']}",
    "",
    "## Evaluation Seeds",
    "",
    f"{PROTOCOL['evaluation_seeds']}",
    "",
    "## Provenance",
    "",
    "This configuration is copied from the previous fixed-C / all-OD diagnosis",
    "that reported 5.6342% blocking for both Strict v1.3 and KSP-FF K=50 hops.",
    "Any deviation from these parameters must be reported as protocol drift.",
]

audit_path = OUT / "PROTOCOL_AUDIT.md"
with open(audit_path, "w", encoding="utf-8") as f:
    f.write("\n".join(audit_lines))

print(f"Wrote {manifest_path}")
print(f"Wrote {audit_path}")
