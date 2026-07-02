"""Shared utilities for Agent-R training and evaluation."""
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.env.event_env import SMDPEnv


def make_env(topology: str = "net1", num_slots: int = 32,
             num_servers: int = 2, seed: int = 42,
             slot_bw_hz: float = 1.25e9, guard_band_fs: int = 1,
             capacities: List[float] = None,
             server_nodes: List[int] = None,
             modulation_profile: str = "default",
             max_blocks: int = 5,
             block_sort_strategy: str = "size_desc",
             path_sort_strategy: str = "km",
             k: int = 3):
    """Create a standard environment.

    Args:
        topology: Network topology name.
        num_slots: Number of spectrum slots.
        num_servers: Number of MEC servers.
        seed: Random seed.
        slot_bw_hz: Slot bandwidth in Hz.
        guard_band_fs: Guard band in FS.
        capacities: Per-server compute capacity in GFLOPS. Defaults to [50.0]*num_servers.
        server_nodes: Optional explicit server node IDs.
        modulation_profile: "default" (4 formats) or "extended" (7 formats).
        max_blocks: Number of candidate spectrum blocks exposed per path/mod.
        block_sort_strategy: Candidate block ranking strategy.  Use "mixed"
            to combine size_desc, waste_asc, start_asc, and center_asc.
        path_sort_strategy: Candidate path output ordering, either "km" or
            "hops".
        k: Number of shortest paths to offer (R action space breadth).
    """
    net = OpticalNetwork(topology, num_slots=num_slots)
    # Topology-specific defaults can create deliberate compute/spectrum
    # asymmetry while still allowing callers to override with server_nodes.
    if server_nodes is None and topology == "metro24_c_sensitive":
        server_nodes = [0, 8, 13, 20][:num_servers]
    if server_nodes is None and topology == "snap24_gnutella_reach":
        server_nodes = [6, 1, 7, 12][:num_servers]
    if server_nodes is None:
        # Default server nodes: first num_servers nodes of the topology.
        server_nodes = list(range(min(num_servers, net.NUM_NODES)))

    if capacities is None:
        if topology == "metro24_c_sensitive":
            capacities = [90.0, 55.0, 32.0, 85.0][:num_servers]
        elif topology == "snap24_gnutella_reach":
            capacities = [90.0, 55.0, 32.0, 85.0][:num_servers]
        else:
            capacities = [50.0] * num_servers
    mec = MECCluster(
        num_nodes=net.NUM_NODES,
        num_servers=num_servers,
        seed=seed,
        server_nodes=server_nodes,
        capacities=capacities,
    )
    mod_reg = ModulationRegistry.from_profile(modulation_profile)
    fs_calc = FSDemandCalculator(slot_bw_hz=slot_bw_hz, guard_band_fs=guard_band_fs)
    env = SMDPEnv(
        net,
        mec,
        mod_reg,
        fs_calc,
        k=k,
        max_blocks=max_blocks,
        block_sort_strategy=block_sort_strategy,
        path_sort_strategy=path_sort_strategy,
    )
    return env


# Split profile definitions.
# Each profile defines (size_multiplier_range, edge_ratio_range) per split.
# As split_id increases: data size increases, edge_ratio decreases.
SPLIT_PROFILES = {
    "default3": {
        "size_multipliers": [(0.2, 0.5), (0.4, 0.8), (0.7, 1.0)],
        "edge_ratio_ranges": [(0.7, 0.95), (0.4, 0.6), (0.1, 0.3)],
    },
    "complex5": {
        "size_multipliers": [
            (0.25, 0.25),   # split0: smallest data, highest edge compute
            (0.45, 0.45),   # split1
            (0.70, 0.70),   # split2: balanced
            (1.00, 1.00),   # split3
            (1.35, 1.35),   # split4: largest data, lowest edge compute
        ],
        "edge_ratio_ranges": [
            (0.75, 0.95),   # split0: high edge_ratio → high edge_compute_cost
            (0.55, 0.75),   # split1
            (0.40, 0.60),   # split2
            (0.25, 0.45),   # split3
            (0.05, 0.25),   # split4: low edge_ratio → low edge_compute_cost
        ],
    },
    "complex5_v2": {
        "size_multipliers": [
            (0.35, 0.35),   # split0: small data, very high edge compute → server overload risk
            (0.55, 0.55),   # split1
            (0.85, 0.85),   # split2: balanced trade-off
            (1.25, 1.25),   # split3
            (2.00, 2.00),   # split4: huge data, moderate edge compute → spectrum pressure risk
        ],
        "edge_ratio_ranges": [
            (0.85, 1.00),   # split0: very high edge_ratio → very high edge_compute_cost
            (0.70, 0.85),   # split1
            (0.50, 0.65),   # split2
            (0.30, 0.45),   # split3
            (0.15, 0.30),   # split4: moderate edge_ratio → non-trivial edge_compute_cost
        ],
    },
    "complex5_v2_lite": {
        "size_multipliers": [
            (0.35, 0.35),   # split0: small data, high edge compute → server compute pressure
            (0.55, 0.55),   # split1
            (0.85, 0.85),   # split2: balanced trade-off
            (1.20, 1.20),   # split3
            (1.60, 1.60),   # split4: larger data, lower edge compute → spectrum pressure
        ],
        "edge_ratio_ranges": [
            (0.80, 0.95),   # split0: high edge_ratio → high edge_compute_cost
            (0.65, 0.80),   # split1
            (0.45, 0.65),   # split2
            (0.30, 0.45),   # split3
            (0.18, 0.35),   # split4: lower edge_ratio → lower edge_compute_cost
        ],
    },
}


def generate_requests(env: SMDPEnv,
                      rng: np.random.RandomState,
                      src_node: int,
                      num_requests: int = 30,
                      arrival_interval: float = 0.25,
                      holding_min: float = 4.0,
                      holding_max: float = 10.0,
                      deadline_min: float = 40.0,
                      deadline_max: float = 120.0,
                      size_min_mb: float = 1.0,
                      size_max_mb: float = 8.0,
                      edge_cost_min: float = 0.5,
                      edge_cost_max: float = 15.0,
                      num_splits: int = 3,
                      split_profile: str = "default3",
                      traffic_mode: str = "iid",
                      regime_stay_prob: float = 0.9) -> List[DNNRequest]:
    """Generate a sequence of random requests from the same source node.

    Supports configurable split profiles:
        default3: 3 splits (backward compatible)
        complex5: 5 splits with expanded data/compute trade-off space
        complex5_v2: 5 splits with stronger trade-offs and trap options

    ``markov_regime`` preserves the configured long-run demand ranges while
    adding persistent light/medium/heavy request regimes. The default ``iid``
    mode preserves historical behavior and random-number consumption.
    """
    profile = SPLIT_PROFILES.get(split_profile, SPLIT_PROFILES["default3"])
    size_mults = profile["size_multipliers"]
    edge_ranges = profile["edge_ratio_ranges"]

    # Validate num_splits against profile
    if num_splits != len(size_mults):
        raise ValueError(
            f"Profile '{split_profile}' expects {len(size_mults)} splits, "
            f"but num_splits={num_splits}."
        )

    if traffic_mode not in ("iid", "markov_regime"):
        raise ValueError(f"Unknown traffic_mode: {traffic_mode}")
    if not 0.0 <= regime_stay_prob <= 1.0:
        raise ValueError("regime_stay_prob must be in [0, 1]")

    def regime_interval(lo: float, hi: float, regime_id: int) -> Tuple[float, float]:
        width = (hi - lo) / 3.0
        return lo + regime_id * width, lo + (regime_id + 1) * width

    requests = []
    regime = None
    for i in range(num_requests):
        if traffic_mode == "markov_regime":
            if regime is None:
                regime = int(rng.randint(0, 3))
            elif rng.rand() > regime_stay_prob:
                alternative = int(rng.randint(0, 2))
                regime = alternative if alternative < regime else alternative + 1
            size_lo, size_hi = regime_interval(size_min_mb, size_max_mb, regime)
            compute_lo, compute_hi = regime_interval(
                edge_cost_min + 0.5, edge_cost_max + 2.0, regime
            )
            deadline_lo, deadline_hi = regime_interval(
                deadline_min, deadline_max, 2 - regime
            )
            base_size = rng.uniform(size_lo, size_hi)
            total_compute = rng.uniform(compute_lo, compute_hi)
        else:
            # Preserve historical IID random-number consumption exactly.
            base_size = rng.uniform(size_min_mb, size_max_mb)
            total_compute = rng.uniform(edge_cost_min + 0.5, edge_cost_max + 2.0)

        splits = []
        for split_id in range(num_splits):
            lo_size, hi_size = size_mults[split_id]
            lo_edge, hi_edge = edge_ranges[split_id]
            size_mb = base_size * rng.uniform(lo_size, hi_size)
            edge_ratio = rng.uniform(lo_edge, hi_edge)

            edge_cost = total_compute * edge_ratio
            local_cost = total_compute * (1.0 - edge_ratio)
            splits.append(SplitProfile(
                split_id=split_id,
                intermediate_size_mb=size_mb,
                local_compute_cost=local_cost,
                edge_compute_cost=edge_cost,
            ))

        req = DNNRequest(
            req_id=i,
            src_node=src_node,
            arrival_time=i * arrival_interval,
            holding_time=rng.uniform(holding_min, holding_max),
            deadline_ms=rng.uniform(
                deadline_lo if traffic_mode == "markov_regime" else deadline_min,
                deadline_hi if traffic_mode == "markov_regime" else deadline_max,
            ),
            splits=splits,
        )
        requests.append(req)
    return requests


def compute_reward(info: Dict, waste_coef: float = 0.8,
                   mod_name: str = None, pm_bpsk_penalty: float = 0.0) -> float:
    """Reward shaping for Agent-R.

    Success: +1 minus block-waste penalty.
    Fail:   -1.

    Args:
        info: Environment step info dict.
        waste_coef: Penalty coefficient for spectrum block waste.
        mod_name: Selected modulation name (e.g. "PM-BPSK", "QPSK").
        pm_bpsk_penalty: If >0 and mod_name == "PM-BPSK", subtract this
            penalty from the success reward to discourage over-reliance
            on the lowest-SE modulation.
    """
    if info.get("success", False):
        waste = info.get("block_waste", 0.0)
        reward = 1.0 - waste_coef * waste
        if mod_name is not None and mod_name == "PM-BPSK" and pm_bpsk_penalty > 0:
            reward -= pm_bpsk_penalty
        return reward
    return -1.0


def compute_agent_c_reward(info: Dict,
                           deadline_ms: float = 100.0,
                           waste_coef: float = 0.8,
                           server_utilization_after: float = 0.0) -> float:
    """Reward shaping for Agent-C (hierarchical upper agent).

    Penalizes delay, waste, and server overload more granularly than Agent-R.
    """
    if info.get("success", False):
        delay_ms = info.get("delay_ms", 0.0)
        waste = info.get("block_waste", 0.0)
        delay_penalty = 0.3 * (delay_ms / max(deadline_ms, 1.0))
        util_penalty = 0.2 * server_utilization_after
        return 1.0 - delay_penalty - waste_coef * waste - util_penalty

    reason = info.get("reason", "unknown")
    if reason == "server_overload":
        return -1.2
    if reason in ("no_suitable_block", "fs_too_large"):
        return -1.0
    return -1.0


def compute_agent_c_reward_delay_aware(
    info: Dict,
    deadline_ms: float = 100.0,
    num_slots_total: int = 24,
    success_reward: float = 1.0,
    block_penalty: float = 2.0,
    delay_coef: float = 0.5,
    deadline_penalty: float = 1.0,
    server_overload_penalty: float = 1.0,
    no_block_penalty: float = 1.2,
    fs_penalty_coef: float = 0.0,
    server_util_coef: float = 0.0,
    server_utilization_after: float = 0.0,
    spectrum_fail_extra: float = 0.0,
    server_fail_extra: float = 0.0,
) -> float:
    """Delay-aware reward shaping for Agent-C.

    Objectives (in order of priority):
        1. Minimize blocking rate
        2. Minimize end-to-end inference delay
        3. Keep server utilization balanced
        4. Encourage spectrum efficiency

    AvgFS, waste, and modulation are diagnostic only; fs_penalty_coef
    defaults to 0.0 so they do not influence the training objective.
    """
    delay_ms = info.get("delay_ms", 0.0)
    norm_delay = delay_ms / max(deadline_ms, 1.0) if deadline_ms > 0 else delay_ms / 100.0

    if info.get("success", False):
        reward = success_reward - delay_coef * norm_delay
        if server_util_coef > 0:
            reward -= server_util_coef * server_utilization_after
        if fs_penalty_coef > 0:
            num_fs = info.get("num_slots", 0)
            norm_fs = num_fs / max(num_slots_total, 1)
            reward -= fs_penalty_coef * norm_fs
        if delay_ms > deadline_ms:
            reward -= deadline_penalty
        return reward

    # Failure case
    reason = info.get("reason", "unknown")
    reward = -block_penalty - delay_coef * norm_delay

    if reason in ("server_overload", "server_saturated"):
        reward -= server_overload_penalty
        if server_fail_extra > 0:
            reward -= server_fail_extra
    elif reason == "no_suitable_block":
        reward -= no_block_penalty
        if spectrum_fail_extra > 0:
            reward -= spectrum_fail_extra
    elif reason == "deadline_infeasible":
        reward -= deadline_penalty

    # Extra deadline-violation check (covers both explicit reason and delay>deadline)
    if delay_ms > deadline_ms:
        reward -= deadline_penalty

    if fs_penalty_coef > 0:
        num_fs = info.get("num_slots", 0)
        norm_fs = num_fs / max(num_slots_total, 1)
        reward -= fs_penalty_coef * norm_fs

    return reward


def compute_spectrum_pressure(
    obs_c: Dict[str, Any],
    action_idx: int,
    num_servers: int,
    source: str = "auto",
    num_slots_total: int = 24,
    pressure_clip_min: float = 0.0,
    pressure_clip_max: float = 1.0,
) -> float:
    """Compute spectrum pressure for the selected (split, server) action.

    Args:
        obs_c: Agent-C observation dict.
        action_idx: Selected flat action index.
        num_servers: Number of servers.
        source: "auto", "feasible_count", "pressure", or "required_over_lfb".
        num_slots_total: Total number of spectrum slots.
        pressure_clip_min: Minimum clipped pressure value.
        pressure_clip_max: Maximum clipped pressure value.

    Returns:
        Clipped spectrum pressure in [pressure_clip_min, pressure_clip_max].
    """
    if action_idx is None or action_idx < 0:
        return pressure_clip_min

    candidate_features = obs_c.get("candidate_features", [])
    if action_idx >= len(candidate_features):
        return pressure_clip_min

    feat = candidate_features[action_idx]
    spectrum_summary = feat.get("spectrum_summary", [])
    if isinstance(spectrum_summary, np.ndarray):
        spectrum_summary = spectrum_summary.tolist()

    # spectrum_summary vector: [lfb_max, lfb_mean, lfb_min, lfb_p25,
    #                           frag_mean, frag_std, free_mean, num_feas_paths,
    #                           path_diversity, min_delay_est]
    free_mean = float(spectrum_summary[6]) if len(spectrum_summary) > 6 else 1.0
    lfb_max = float(spectrum_summary[0]) if len(spectrum_summary) > 0 else float(num_slots_total)
    frag_mean = float(spectrum_summary[4]) if len(spectrum_summary) > 4 else 0.0

    feasible_count = feat.get("feasible_count", 0)
    best_fs = feat.get("best_fs_estimate")
    if best_fs is None:
        best_fs = num_slots_total

    pressure = None

    if source == "pressure":
        # Use 1 - free_mean as congestion proxy
        pressure = 1.0 - free_mean
    elif source == "feasible_count":
        # Normalize feasible_count by approximate max (k=3 paths * 4 mods * 5 blocks = 60)
        max_feas = 60.0
        pressure = 1.0 - min(float(feasible_count) / max_feas, 1.0)
    elif source == "required_over_lfb":
        # Required FS relative to largest free block
        pressure = float(best_fs) / max(lfb_max, 1.0)
    else:  # auto
        # Prefer pressure (free_mean) if available; fallback to feasible_count
        if free_mean >= 0.0 and free_mean <= 1.0:
            pressure = 1.0 - free_mean
        elif feasible_count is not None:
            max_feas = 60.0
            pressure = 1.0 - min(float(feasible_count) / max_feas, 1.0)
        else:
            pressure = float(best_fs) / max(lfb_max, 1.0)

    # Use frag_mean as secondary signal if free_mean doesn't vary
    if pressure is None or pressure < 1e-6:
        pressure = frag_mean

    pressure = float(np.clip(pressure, pressure_clip_min, pressure_clip_max))
    return pressure


def compute_agent_c_reward_pressure_aware(
    info: Dict,
    obs_c: Dict[str, Any],
    action_idx: int,
    num_servers: int,
    deadline_ms: float = 100.0,
    num_slots_total: int = 24,
    success_reward: float = 1.0,
    base_block_penalty: float = 2.0,
    base_delay_coef: float = 0.5,
    deadline_penalty: float = 1.0,
    server_overload_penalty: float = 1.0,
    base_no_block_penalty: float = 1.2,
    pressure_penalty_coef: float = 0.5,
    pressure_source: str = "auto",
    pressure_clip_min: float = 0.0,
    pressure_clip_max: float = 1.0,
) -> Tuple[float, float]:
    """Pressure-aware reward shaping for Agent-C.

    Key idea: penalize the agent more when it selects a high-pressure candidate
    and the environment is already congested. This encourages the agent to
    reserve low-pressure candidates for critical requests.

    The pressure penalty scales both the base reward and failure penalties:
      - success_reward is reduced by pressure_penalty_coef * pressure
      - block penalties are increased by (1 + pressure)

    Returns:
        (reward, pressure)
    """
    pressure = compute_spectrum_pressure(
        obs_c, action_idx, num_servers,
        source=pressure_source,
        num_slots_total=num_slots_total,
        pressure_clip_min=pressure_clip_min,
        pressure_clip_max=pressure_clip_max,
    )

    delay_ms = info.get("delay_ms", 0.0)
    norm_delay = delay_ms / max(deadline_ms, 1.0) if deadline_ms > 0 else delay_ms / 100.0

    # Pressure-adjusted coefficients
    adjusted_success_reward = success_reward - pressure_penalty_coef * pressure
    adjusted_block_penalty = base_block_penalty * (1.0 + 0.5 * pressure)
    adjusted_delay_coef = base_delay_coef * (1.0 + 0.3 * pressure)
    adjusted_no_block_penalty = base_no_block_penalty * (1.0 + 0.5 * pressure)

    if info.get("success", False):
        reward = adjusted_success_reward - adjusted_delay_coef * norm_delay
        if delay_ms > deadline_ms:
            reward -= deadline_penalty
        return reward, pressure

    # Failure case
    reason = info.get("reason", "unknown")
    reward = -adjusted_block_penalty - adjusted_delay_coef * norm_delay

    if reason in ("server_overload", "server_saturated"):
        reward -= server_overload_penalty * (1.0 + pressure)
    elif reason == "no_suitable_block":
        reward -= adjusted_no_block_penalty
    elif reason == "deadline_infeasible":
        reward -= deadline_penalty

    if delay_ms > deadline_ms:
        reward -= deadline_penalty

    return reward, pressure


def compute_agent_c_reward_state_aware(
    info: Dict,
    obs_c: Dict[str, Any],
    action_idx: int,
    num_servers: int,
    deadline_ms: float = 100.0,
    num_slots_total: int = 24,
    success_reward: float = 1.0,
    base_block_penalty: float = 2.0,
    base_delay_coef: float = 0.8,
    deadline_penalty: float = 1.0,
    server_overload_penalty: float = 1.0,
    base_no_block_penalty: float = 1.2,
    pressure_source: str = "auto",
    pressure_clip_min: float = 0.0,
    pressure_clip_max: float = 1.0,
) -> Tuple[float, float, float, float, float]:
    """State-aware blocking-delay reward shaping for Agent-C.

    Dynamically balances blocking and delay based on current spectrum pressure:
        - Low pressure  → emphasize delay reduction (allow split2)
        - High pressure → emphasize blocking reduction (prefer split0)

    Returns:
        (reward, pressure, delay_weight, block_weight, no_block_weight)
    """
    pressure = compute_spectrum_pressure(
        obs_c, action_idx, num_servers,
        source=pressure_source,
        num_slots_total=num_slots_total,
        pressure_clip_min=pressure_clip_min,
        pressure_clip_max=pressure_clip_max,
    )

    # Dynamic weights
    delay_weight = base_delay_coef * (1.0 - pressure)
    block_weight = base_block_penalty * (1.0 + pressure)
    no_block_weight = base_no_block_penalty * (1.0 + pressure)

    delay_ms = info.get("delay_ms", 0.0)
    norm_delay = delay_ms / max(deadline_ms, 1.0) if deadline_ms > 0 else delay_ms / 100.0

    if info.get("success", False):
        reward = success_reward - delay_weight * norm_delay
        if delay_ms > deadline_ms:
            reward -= deadline_penalty
        return reward, pressure, delay_weight, block_weight, no_block_weight

    # Failure case
    reason = info.get("reason", "unknown")
    reward = -block_weight - delay_weight * norm_delay

    if reason == "server_overload":
        reward -= server_overload_penalty
    elif reason == "no_suitable_block":
        reward -= no_block_weight
    elif reason == "deadline_infeasible":
        reward -= deadline_penalty

    if delay_ms > deadline_ms:
        reward -= deadline_penalty

    return reward, pressure, delay_weight, block_weight, no_block_weight
