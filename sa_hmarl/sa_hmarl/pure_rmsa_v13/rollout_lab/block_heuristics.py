"""Path-ordered spectrum heuristics for mechanism-controlled comparisons."""
from __future__ import annotations

from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, PDSRMSAEnv, RMSAAction
from sa_hmarl.pds_rmsa.protocol import (
    Request,
    highest_modulation_for_distance,
    required_fs,
)


def ksp_best_fit_action(
    env: PDSRMSAEnv, request: Request
) -> RMSAAction | BlockedAction:
    """KSP-BF: first feasible path, then the tightest sufficient free run."""
    paths = env._cached_k_paths(request.src_node, request.dst_node, env.k_paths)
    reachable = 0
    for path_rank, path in enumerate(paths):
        modulation = highest_modulation_for_distance(
            env.topology.path_length_km(path)
        )
        if modulation is None:
            continue
        reachable += 1
        modulation_name, spectral_efficiency = modulation
        required = required_fs(request.bitrate_gbps, spectral_efficiency)
        available = env.spectrum.get_available_slots(path)
        blocks = env._free_blocks(available, required)
        if not blocks:
            continue
        start_slot, _run_size = min(blocks, key=lambda block: (block[1], block[0]))
        return RMSAAction(
            action_id=(path_rank, modulation_name, start_slot, required),
            path_rank=path_rank,
            path=path,
            modulation=modulation_name,
            start_slot=start_slot,
            required_fs=required,
        )

    reason = (
        "no_candidate_path"
        if not paths
        else "no_reach_feasible_path_mod"
        if reachable == 0
        else "insufficient_spectrum"
    )
    return BlockedAction(reason=reason)


def global_best_fit_action(
    env: PDSRMSAEnv, request: Request
) -> RMSAAction | BlockedAction:
    """Choose the lowest-waste FF block across all K=50 candidate paths."""
    paths = env._cached_k_paths(request.src_node, request.dst_node, env.k_paths)
    candidates: list[tuple[tuple[float, int, int, int], RMSAAction]] = []
    reachable = 0
    for path_rank, path in enumerate(paths):
        modulation = highest_modulation_for_distance(
            env.topology.path_length_km(path)
        )
        if modulation is None:
            continue
        reachable += 1
        modulation_name, spectral_efficiency = modulation
        required = required_fs(request.bitrate_gbps, spectral_efficiency)
        available = env.spectrum.get_available_slots(path)
        for start_slot, run_size in env._free_blocks(available, required):
            waste = (run_size - required) / run_size
            action = RMSAAction(
                action_id=(path_rank, modulation_name, start_slot, required),
                path_rank=path_rank,
                path=path,
                modulation=modulation_name,
                start_slot=start_slot,
                required_fs=required,
            )
            candidates.append(
                ((float(waste), run_size - required, path_rank, start_slot), action)
            )
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]

    reason = (
        "no_candidate_path"
        if not paths
        else "no_reach_feasible_path_mod"
        if reachable == 0
        else "insufficient_spectrum"
    )
    return BlockedAction(reason=reason)


def path_penalized_best_fit_action(
    env: PDSRMSAEnv, request: Request, path_penalty: float
) -> RMSAAction | BlockedAction:
    """Trade block waste against deviating from the hops-ordered path rank."""
    if path_penalty < 0:
        raise ValueError("path_penalty must be non-negative")
    paths = env._cached_k_paths(request.src_node, request.dst_node, env.k_paths)
    candidates: list[tuple[tuple[float, float, int, int], RMSAAction]] = []
    reachable = 0
    for path_rank, path in enumerate(paths):
        modulation = highest_modulation_for_distance(
            env.topology.path_length_km(path)
        )
        if modulation is None:
            continue
        reachable += 1
        modulation_name, spectral_efficiency = modulation
        required = required_fs(request.bitrate_gbps, spectral_efficiency)
        available = env.spectrum.get_available_slots(path)
        for start_slot, run_size in env._free_blocks(available, required):
            waste = (run_size - required) / run_size
            score = waste + path_penalty * path_rank
            action = RMSAAction(
                action_id=(path_rank, modulation_name, start_slot, required),
                path_rank=path_rank,
                path=path,
                modulation=modulation_name,
                start_slot=start_slot,
                required_fs=required,
            )
            candidates.append(
                ((float(score), float(waste), path_rank, start_slot), action)
            )
    if candidates:
        return min(candidates, key=lambda item: item[0])[1]
    reason = (
        "no_candidate_path"
        if not paths
        else "no_reach_feasible_path_mod"
        if reachable == 0
        else "insufficient_spectrum"
    )
    return BlockedAction(reason=reason)
