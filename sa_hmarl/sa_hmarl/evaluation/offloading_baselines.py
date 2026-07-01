"""Offloading baselines for Agent-C (split/server selection).

Baselines:
  - greedy:   minimum edge_compute_ms
  - wo:       distributed offloading (load balancing + utilization)
  - df:       distance first
  - rf:       resource first (lightest server)
  - iwd:      lightweight online Intelligent Water Droplet
  - oracle_r_query: query frozen R for each valid candidate, pick best delay among successful
  - oracle_pressure: select candidate with lowest spectrum+server pressure score

All baselines respect agent_c_mask.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import decode_agent_c_action, build_agent_r_observation
from sa_hmarl.network.ksp import get_k_shortest_paths


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_valid_actions(mask: np.ndarray) -> np.ndarray:
    return np.where(mask)[0]


def _estimate_distance_km(env, src_node: int, server_node: int) -> float:
    """Shortest path distance in km from src to server node."""
    try:
        paths = get_k_shortest_paths(env.net.G, src_node, server_node, k=1, weight="length_km")
        if paths:
            return env.net.path_distance_km(paths[0])
    except Exception:
        pass
    # Fallback: direct edge or large value
    if env.net.G.has_edge(src_node, server_node):
        return env.net.G.edges[(min(src_node, server_node), max(src_node, server_node))]["length_km"]
    return float("inf")


def _decode_c(action_idx: int, num_servers: int) -> Tuple[int, int]:
    return decode_agent_c_action(action_idx, num_servers)


def _finite_or_large(value: Any, large: float = 1e9) -> float:
    """Convert missing/invalid numeric costs to a large finite penalty."""
    if value is None:
        return large
    try:
        val = float(value)
    except (TypeError, ValueError):
        return large
    if np.isnan(val) or np.isinf(val):
        return large
    return val


# ---------------------------------------------------------------------------
# Greedy
# ---------------------------------------------------------------------------

def select_greedy(obs_c: Dict[str, Any], mask: np.ndarray) -> Optional[int]:
    """Select action with minimum edge_compute_ms among valid actions."""
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None
    features = obs_c["candidate_features"]
    best = int(valid[0])
    best_val = features[best]["edge_compute_ms"]
    if best_val == float("inf"):
        best_val = 1e9
    for idx in valid[1:]:
        val = features[int(idx)]["edge_compute_ms"]
        if val == float("inf"):
            val = 1e9
        if val < best_val:
            best = int(idx)
            best_val = val
    return best


# ---------------------------------------------------------------------------
# WO: Distributed Offloading
# ---------------------------------------------------------------------------

def select_wo(
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    server_selected_count: np.ndarray,
    alpha: float = 1.0,
    beta: float = 0.5,
    gamma: float = 0.2,
) -> Optional[int]:
    """Distributed offloading: balance across servers using selection history + load.

    Args:
        server_selected_count: array of shape (num_servers,) tracking selections.
    """
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]

    max_count = max(server_selected_count.max(), 1)
    max_util = max(max(obs_c["server_utilizations"]), 1e-6)

    best = int(valid[0])
    best_score = float("inf")

    for idx in valid:
        idx = int(idx)
        split_id, server_id = _decode_c(idx, num_servers)
        feat = features[idx]
        edge_ms = feat["edge_compute_ms"]
        if edge_ms == float("inf"):
            edge_ms = 1e9

        norm_count = server_selected_count[server_id] / max_count
        norm_util = obs_c["server_utilizations"][server_id] / max_util
        norm_delay = edge_ms / 100.0  # rough normalization

        score = alpha * norm_count + beta * norm_util + gamma * norm_delay
        if score < best_score:
            best_score = score
            best = idx

    return best


# ---------------------------------------------------------------------------
# DF: Distance First
# ---------------------------------------------------------------------------

def select_df(
    env,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
) -> Optional[int]:
    """Distance first: select nearest server, tie-break by compute delay."""
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]
    src_node = obs_c["request_features"]["src_node"]

    # Precompute distances
    distances = []
    for server_id in range(num_servers):
        server_node = env.mec.servers[server_id].node_id
        distances.append(_estimate_distance_km(env, src_node, server_node))

    best = int(valid[0])
    best_dist = distances[_decode_c(best, num_servers)[1]]
    best_delay = features[best]["edge_compute_ms"]
    if best_delay == float("inf"):
        best_delay = 1e9

    for idx in valid[1:]:
        idx = int(idx)
        _, server_id = _decode_c(idx, num_servers)
        dist = distances[server_id]
        delay = features[idx]["edge_compute_ms"]
        if delay == float("inf"):
            delay = 1e9

        if dist < best_dist or (dist == best_dist and delay < best_delay):
            best = idx
            best_dist = dist
            best_delay = delay

    return best


# ---------------------------------------------------------------------------
# RF: Resource First
# ---------------------------------------------------------------------------

def select_rf(
    obs_c: Dict[str, Any],
    mask: np.ndarray,
) -> Optional[int]:
    """Resource first: select lightest server, tie-break by compute delay."""
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]
    utils = obs_c["server_utilizations"]

    best = int(valid[0])
    best_util = utils[_decode_c(best, num_servers)[1]]
    best_delay = features[best]["edge_compute_ms"]
    if best_delay == float("inf"):
        best_delay = 1e9

    for idx in valid[1:]:
        idx = int(idx)
        _, server_id = _decode_c(idx, num_servers)
        util = utils[server_id]
        delay = features[idx]["edge_compute_ms"]
        if delay == float("inf"):
            delay = 1e9

        if util < best_util or (util == best_util and delay < best_delay):
            best = idx
            best_util = util
            best_delay = delay

    return best


# ---------------------------------------------------------------------------
# IWD: Lightweight online Intelligent Water Droplet
# ---------------------------------------------------------------------------

def select_iwd(
    env,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
    n_droplets: int = 20,
    n_iters: int = 10,
    rho: float = 0.1,
    epsilon: float = 1e-6,
    w_delay: float = 0.30,
    w_load: float = 0.25,
    w_dist: float = 0.20,
    w_fs: float = 0.25,
) -> Optional[int]:
    """Lightweight online IWD for split/server selection.

    Each valid action is a candidate solution. IWD droplets probabilistically
    explore actions, updating soil values based on composite cost.
    """
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    valid = np.array(valid, dtype=int)
    n_actions = len(valid)

    # Small action space: direct cost minimization fallback
    if n_actions <= 3:
        costs = _compute_iwd_costs(env, obs_c, valid, w_delay, w_load, w_dist, w_fs)
        return int(valid[np.argmin(costs)])

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]
    src_node = obs_c["request_features"]["src_node"]

    # Precompute distances
    distances = []
    for server_id in range(num_servers):
        server_node = env.mec.servers[server_id].node_id
        distances.append(_estimate_distance_km(env, src_node, server_node))

    # Precompute costs for all valid actions
    costs = np.zeros(n_actions)
    max_delay = 1e-6
    max_util = 1e-6
    max_dist = 1e-6
    max_fs = 1e-6

    for i, idx in enumerate(valid):
        idx = int(idx)
        split_id, server_id = _decode_c(idx, num_servers)
        feat = features[idx]
        delay = feat["edge_compute_ms"]
        if delay == float("inf"):
            delay = 1e9
        util = obs_c["server_utilizations"][server_id]
        dist = distances[server_id]
        fs_est = _finite_or_large(feat["best_fs_estimate"])

        costs[i] = 0.0
        max_delay = max(max_delay, delay)
        max_util = max(max_util, util)
        max_dist = max(max_dist, dist)
        max_fs = max(max_fs, fs_est)

    # Normalize and compute composite cost
    for i, idx in enumerate(valid):
        idx = int(idx)
        split_id, server_id = _decode_c(idx, num_servers)
        feat = features[idx]
        delay = feat["edge_compute_ms"]
        if delay == float("inf"):
            delay = 1e9
        util = obs_c["server_utilizations"][server_id]
        dist = distances[server_id]
        fs_est = _finite_or_large(feat["best_fs_estimate"])

        c_delay = delay / max_delay
        c_load = util / max_util
        c_dist = dist / max_dist if max_dist > 0 else 0.0
        c_fs = fs_est / max_fs if max_fs > 0 else 0.0
        costs[i] = w_delay * c_delay + w_load * c_load + w_dist * c_dist + w_fs * c_fs

    # Guard against NaN
    if np.any(np.isnan(costs)):
        costs = np.nan_to_num(costs, nan=1e9, posinf=1e9, neginf=1e9)

    # Initialize soil
    soil = np.ones(n_actions, dtype=float)
    visit_counts = np.zeros(n_actions, dtype=int)

    if rng is None:
        rng = np.random.RandomState(42)

    for _ in range(n_iters):
        for _ in range(n_droplets):
            # Selection probability: inverse of soil + cost
            denom = epsilon + soil + costs
            denom = np.maximum(denom, 1e-12)
            inv = 1.0 / denom
            s = inv.sum()
            if s < 1e-12:
                probs = np.ones(n_actions) / n_actions
            else:
                probs = inv / s
            choice = rng.choice(n_actions, p=probs)
            visit_counts[choice] += 1
            # Update soil
            soil[choice] = (1 - rho) * soil[choice] + rho * costs[choice]

    # Select most visited action
    return int(valid[np.argmax(visit_counts)])


def _compute_iwd_costs(env, obs_c, valid_actions, w_delay, w_load, w_dist, w_fs):
    """Compute normalized composite costs for valid actions."""
    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]
    src_node = obs_c["request_features"]["src_node"]
    distances = []
    for server_id in range(num_servers):
        server_node = env.mec.servers[server_id].node_id
        distances.append(_estimate_distance_km(env, src_node, server_node))

    costs = []
    max_delay = 1e-6
    max_util = 1e-6
    max_dist = 1e-6
    max_fs = 1e-6

    for idx in valid_actions:
        idx = int(idx)
        _, server_id = _decode_c(idx, num_servers)
        feat = features[idx]
        delay = feat["edge_compute_ms"]
        if delay == float("inf"):
            delay = 1e9
        max_delay = max(max_delay, delay)
        max_util = max(max_util, obs_c["server_utilizations"][server_id])
        max_dist = max(max_dist, distances[server_id])
        max_fs = max(max_fs, _finite_or_large(feat["best_fs_estimate"]))

    for idx in valid_actions:
        idx = int(idx)
        _, server_id = _decode_c(idx, num_servers)
        feat = features[idx]
        delay = feat["edge_compute_ms"]
        if delay == float("inf"):
            delay = 1e9
        c_delay = delay / max_delay
        c_load = obs_c["server_utilizations"][server_id] / max_util
        c_dist = distances[server_id] / max_dist if max_dist > 0 else 0.0
        c_fs = _finite_or_large(feat["best_fs_estimate"]) / max_fs if max_fs > 0 else 0.0
        costs.append(w_delay * c_delay + w_load * c_load + w_dist * c_dist + w_fs * c_fs)

    return np.array(costs)


# ---------------------------------------------------------------------------
# Oracle: Query frozen R for each valid candidate
# ---------------------------------------------------------------------------

def select_oracle_r_query(
    env,
    request,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    agent_r,
) -> Optional[int]:
    """Oracle-C: query frozen Agent-R for every valid (split, server) candidate.

    Returns the candidate with minimum expected delay among those where
    Agent-R can find a valid (path, mod, block).  This measures the
    theoretical upper bound of C-side performance with perfect foresight
    of R's capabilities.
    """
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]

    best_action = None
    best_delay = float("inf")

    for idx in valid:
        idx = int(idx)
        split_id, server_id = _decode_c(idx, num_servers)

        # Build R observation for this candidate
        try:
            obs_r = build_agent_r_observation(env, request, split_id, server_id)
        except Exception:
            continue

        # Query R (deterministic)
        if agent_r is None:
            continue
        try:
            action_r = agent_r.select_action(obs_r, deterministic=True)
        except Exception:
            continue

        if action_r is None:
            continue

        # Check if R's choice is valid according to its mask
        r_mask = obs_r["agent_r_mask"]
        if action_r < 0 or action_r >= len(r_mask) or not r_mask[action_r]:
            continue

        # Candidate is viable: R can serve it
        feat = features[idx]
        edge_ms = feat["edge_compute_ms"]
        if edge_ms == float("inf"):
            edge_ms = 1e9
        local_ms = feat["local_compute_ms"]
        if local_ms == float("inf"):
            local_ms = 1e9
        est_delay = edge_ms + local_ms

        if est_delay < best_delay:
            best_delay = est_delay
            best_action = idx

    # Fallback: if no candidate is viable for R, use greedy
    if best_action is None:
        return select_greedy(obs_c, mask)

    return best_action


# ---------------------------------------------------------------------------
# Oracle: Pressure-based selection (no R query)
# ---------------------------------------------------------------------------

def select_oracle_pressure(
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    w_spectrum: float = 0.35,
    w_server: float = 0.25,
    w_frag: float = 0.15,
    w_delay: float = 0.25,
    pressure_clip: float = 10.0,
    **kwargs,
) -> Optional[int]:
    """Oracle-C pressure: select candidate with lowest composite pressure score.

    Pressure components:
      - spectrum: best_fs / lfb_max  (tightness of spectrum fit)
      - server:   server_utilization  (compute load)
      - frag:     frag_mean           (fragmentation penalty)
      - delay:    normalized edge_compute_ms

    Lower score = less pressure = better candidate.
    """
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]

    best_action = None
    best_score = float("inf")

    # Normalize delay
    max_delay = 1e-6
    for idx in valid:
        idx = int(idx)
        edge_ms = features[idx]["edge_compute_ms"]
        if edge_ms != float("inf"):
            max_delay = max(max_delay, edge_ms)

    for idx in valid:
        idx = int(idx)
        feat = features[idx]
        split_id, server_id = _decode_c(idx, num_servers)

        spec = feat.get("spectrum_summary", [])
        if isinstance(spec, np.ndarray):
            spec = spec.tolist()

        lfb_max = float(spec[0]) if len(spec) > 0 else 24.0
        frag_mean = float(spec[4]) if len(spec) > 4 else 0.0
        best_fs = feat.get("best_fs_estimate")
        if best_fs is None:
            best_fs = 24.0

        server_util = obs_c["server_utilizations"][server_id]
        edge_ms = feat["edge_compute_ms"]
        if edge_ms == float("inf"):
            edge_ms = 1e9

        spectrum_pressure = min(float(best_fs) / max(lfb_max, 1.0), 1.0)
        server_pressure = min(server_util, 1.0)
        frag_pressure = min(frag_mean, 1.0)
        delay_pressure = min(edge_ms / max_delay, 1.0) if max_delay > 0 else 0.0

        score = (
            w_spectrum * spectrum_pressure
            + w_server * server_pressure
            + w_frag * frag_pressure
            + w_delay * delay_pressure
        )
        score = min(score, pressure_clip)

        if score < best_score:
            best_score = score
            best_action = idx

    return best_action


# ---------------------------------------------------------------------------
# Unified interface
# ---------------------------------------------------------------------------

def select_offloading_action(
    method_name: str,
    env,
    request,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    rng: Optional[np.random.RandomState] = None,
    server_selected_count: Optional[np.ndarray] = None,
    **kwargs,
) -> Optional[int]:
    """Unified interface for all offloading baselines.

    Args:
        method_name: "ppo" | "greedy" | "wo" | "df" | "rf" | "iwd"
        env: SMDPEnv instance.
        request: DNNRequest.
        obs_c: Agent-C observation dict.
        mask: agent_c_mask bool array.
        rng: RandomState for stochastic baselines (IWD).
        server_selected_count: array tracking per-server selections (WO only).

    Returns:
        Flat action index or None if no valid action.
    """
    method = method_name.lower()
    if method == "ppo":
        # Handled outside (needs agent model)
        raise ValueError("select_offloading_action does not handle 'ppo'; call agent.select_action directly")
    elif method == "greedy":
        return select_greedy(obs_c, mask)
    elif method == "wo":
        if server_selected_count is None:
            # Fallback to RF if no history
            return select_rf(obs_c, mask)
        return select_wo(obs_c, mask, server_selected_count, **kwargs)
    elif method == "df":
        return select_df(env, obs_c, mask)
    elif method == "rf":
        return select_rf(obs_c, mask)
    elif method == "iwd":
        return select_iwd(env, obs_c, mask, rng=rng, **kwargs)
    elif method == "oracle_r_query":
        agent_r = kwargs.get("agent_r")
        if agent_r is None:
            raise ValueError("oracle_r_query requires agent_r in kwargs")
        return select_oracle_r_query(env, request, obs_c, mask, agent_r)
    elif method == "oracle_pressure":
        return select_oracle_pressure(obs_c, mask, **kwargs)
    else:
        raise ValueError(f"Unknown offloading method: {method_name}")
