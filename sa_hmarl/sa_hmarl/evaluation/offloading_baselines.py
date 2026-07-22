"""Offloading baselines for Agent-C (split/server selection).

Baselines:
  - greedy:   minimum edge_compute_ms
  - wo:       distributed offloading (load balancing + utilization)
  - df:       distance first
  - rf:       resource first (lightest server)
  - df_fixed0: distance first, constrained to split 0 (fixed-split / no-partition-style ablation)
  - rf_fixed0: resource first, constrained to split 0 (fixed-split / no-partition-style ablation)
  - iwd:      lightweight online Intelligent Water Droplet
  - oracle_r_query: query frozen R for each valid candidate, pick best delay among successful
  - oracle_pressure: select candidate with lowest spectrum+server pressure score

All baselines respect agent_c_mask.
"""
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from sa_hmarl.env.observation_builder import decode_agent_c_action, decode_agent_r_action, build_agent_r_observation
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


def _valid_actions_for_split(
    mask: np.ndarray, split_id: int, num_servers: int
) -> np.ndarray:
    """Return valid flat action indices belonging to a single split.

    Action encoding is split-major: idx = split_id * num_servers + server_id.
    This helper never falls back to other splits; if the requested split has
    no valid (split, server) pair, an empty array is returned.
    """
    start = split_id * num_servers
    end = start + num_servers
    split_indices = np.arange(start, end, dtype=int)
    split_mask = mask[start:end]
    return split_indices[split_mask]


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
# Fixed-split DF/RF (no-partition-style ablation)
# ---------------------------------------------------------------------------

def select_df_fixed0(
    env,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
) -> Optional[int]:
    """Distance-first server selection forced to split 0.

    This is a fixed-split / no-partition-style ablation: the split is not
    learned or heuristically chosen, it is pinned to split0.  Server selection
    follows the same distance-first rule as select_df, but only split0 actions
    that are legal under agent_c_mask are considered.  No fallback to other
    splits is performed.
    """
    num_servers = len(obs_c["server_utilizations"])
    valid = _valid_actions_for_split(mask, split_id=0, num_servers=num_servers)
    if len(valid) == 0:
        return None

    features = obs_c["candidate_features"]
    src_node = obs_c["request_features"]["src_node"]

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


def select_rf_fixed0(
    obs_c: Dict[str, Any],
    mask: np.ndarray,
) -> Optional[int]:
    """Resource-first server selection forced to split 0.

    Fixed-split / no-partition-style ablation.  Only split0 actions that are
    legal under agent_c_mask are considered; no fallback to other splits.
    """
    num_servers = len(obs_c["server_utilizations"])
    valid = _valid_actions_for_split(mask, split_id=0, num_servers=num_servers)
    if len(valid) == 0:
        return None

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
# Feasible-Region C selectors (v1.4)
# ---------------------------------------------------------------------------

def select_fr_count(
    env,
    request,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    n_max: Optional[int] = None,
) -> Optional[int]:
    """Feasible-region count selector: maximize number of legal R actions.

    This is an ablation for v1.4. It ignores optical quality and server
    overload, selecting the (split, server) pair that gives the largest
    feasible R-side action set.
    """
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    counts: List[int] = []

    for idx in valid:
        idx = int(idx)
        split_id, server_id = _decode_c(idx, num_servers)
        try:
            obs_r = build_agent_r_observation(env, request, split_id, server_id)
            n_valid = int(np.sum(np.asarray(obs_r["agent_r_mask"], dtype=bool)))
        except Exception:
            n_valid = 0
        counts.append(n_valid)

    counts_arr = np.asarray(counts, dtype=float)
    if n_max is None:
        n_max = int(max(counts_arr.max(), 1.0))
    else:
        n_max = max(int(n_max), 1)

    scores = np.log1p(counts_arr) / np.log1p(n_max)
    best_local = int(np.argmax(scores))
    return int(valid[best_local])


def _sample_r_actions_for_quality(
    obs_r: Dict[str, Any],
    sample_k: int,
) -> np.ndarray:
    """Return a diverse subset of legal R actions for q_fast evaluation.

    Always includes KSP-FF, per-path first-fit and largest-block anchors, and
    high-spectral-efficiency candidates. Falls back to all legal actions if the
    legal set is smaller than ``sample_k``.
    """
    mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    legal = np.flatnonzero(mask)
    n_legal = len(legal)
    if n_legal == 0:
        return np.array([], dtype=int)
    if n_legal <= sample_k:
        return legal

    num_paths = len(obs_r["candidate_paths"])
    num_mods = len(obs_r["mod_names"])
    num_blocks = len(mask) // (num_paths * num_mods)

    selected: set = set()

    # KSP-FF anchor
    try:
        from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action
        kspff = ksp_ff_action(obs_r)
        if kspff is not None and mask[kspff]:
            selected.add(int(kspff))
    except Exception:
        pass

    # Per-path first valid block and largest block
    for path_idx in range(num_paths):
        first_action: Optional[int] = None
        largest_action: Optional[int] = None
        largest_size = -1
        for mod_idx in range(num_mods):
            start = path_idx * num_mods * num_blocks + mod_idx * num_blocks
            blocks_matrix = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
            for block_idx in range(num_blocks):
                action_idx = start + block_idx
                if not mask[action_idx]:
                    continue
                if block_idx >= len(blocks_matrix):
                    continue
                block_size = int(blocks_matrix[block_idx][1])
                if first_action is None:
                    first_action = action_idx
                if block_size > largest_size:
                    largest_size = block_size
                    largest_action = action_idx
        if first_action is not None:
            selected.add(int(first_action))
        if largest_action is not None:
            selected.add(int(largest_action))

    # Fill with high spectral-efficiency / large-block candidates.
    # For a fixed request, required FS is inversely proportional to SE.
    scored: List[Tuple[float, int, int]] = []
    for action_idx in legal:
        path_idx, mod_idx, block_idx = decode_agent_r_action(action_idx, num_mods, num_blocks)
        req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
        if req_fs is None or req_fs <= 0:
            continue
        blocks_matrix = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        if block_idx >= len(blocks_matrix):
            continue
        block_size = int(blocks_matrix[block_idx][1])
        se_proxy = 1.0 / float(req_fs)
        scored.append((-se_proxy, -block_size, int(action_idx)))

    scored.sort()
    for _, _, action_idx in scored:
        selected.add(action_idx)
        if len(selected) >= sample_k:
            break

    return np.array(sorted(selected), dtype=int)


def _compute_q_fast_for_actions(
    request,
    obs_r: Dict[str, Any],
    c_feat: Dict[str, Any],
    action_indices: np.ndarray,
    w_slack: float,
    w_lfb: float,
    w_eff: float,
    w_reqfs: float,
    w_waste: float,
) -> np.ndarray:
    """Compute q_fast for a fixed set of flat R actions."""
    if len(action_indices) == 0:
        return np.array([], dtype=float)

    num_paths = len(obs_r["candidate_paths"])
    num_mods = len(obs_r["mod_names"])
    mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
    num_blocks = len(mask) // (num_paths * num_mods)

    edge_compute_ms = float(c_feat.get("edge_compute_ms", 0.0))
    local_compute_ms = float(c_feat.get("local_compute_ms", 0.0))
    if edge_compute_ms == float("inf"):
        edge_compute_ms = 1e9
    if local_compute_ms == float("inf"):
        local_compute_ms = 1e9

    slack_vals: List[float] = []
    lfb_vals: List[float] = []
    eff_vals: List[float] = []
    reqfs_vals: List[float] = []
    waste_vals: List[float] = []

    for action_idx in action_indices:
        path_idx, mod_idx, block_idx = decode_agent_r_action(int(action_idx), num_mods, num_blocks)
        path_features = obs_r["path_features"][path_idx]
        path_km = float(path_features.get("path_length_km", 0.0))
        hops = int(path_features.get("hop_count", 0))

        # Transmission delay: prop + proc + setup (matches _path_transmission_delay_ms)
        from sa_hmarl.env.fs_demand import (
            DEFAULT_PROP_SPEED_KM_S,
            DEFAULT_PROC_PER_HOP_S,
            DEFAULT_SETUP_TIME_S,
        )
        trans_ms = (
            (path_km / DEFAULT_PROP_SPEED_KM_S) * 1000.0
            + hops * DEFAULT_PROC_PER_HOP_S * 1000.0
            + DEFAULT_SETUP_TIME_S * 1000.0
        )
        slack_ms = float(request.deadline_ms) - (trans_ms + edge_compute_ms + local_compute_ms)
        slack_vals.append(slack_ms)

        req_fs = obs_r["required_fs_per_path_mod"][path_idx][mod_idx]
        req_fs_val = float(req_fs) if req_fs is not None and req_fs > 0 else 0.0
        reqfs_vals.append(req_fs_val)

        blocks_matrix = obs_r["candidate_blocks_per_path_mod"][path_idx][mod_idx]
        if block_idx < len(blocks_matrix):
            block_size = int(blocks_matrix[block_idx][1])
        else:
            block_size = req_fs_val
        lfb_vals.append(float(block_size))

        waste = float(block_size - req_fs_val)
        waste_vals.append(max(waste, 0.0))

        # Spectral-efficiency proxy: for fixed data rate, SE ~ 1/required_fs.
        eff_vals.append(1.0 / max(req_fs_val, 1.0))

    def _normalize_higher_is_better(vals: List[float]) -> np.ndarray:
        arr = np.asarray(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (arr - mn) / (mx - mn)
        return np.full_like(arr, 0.5)

    def _normalize_lower_is_better(vals: List[float]) -> np.ndarray:
        arr = np.asarray(vals, dtype=float)
        mn, mx = arr.min(), arr.max()
        if mx > mn:
            return (mx - arr) / (mx - mn)
        return np.full_like(arr, 0.5)

    norm_slack = _normalize_higher_is_better(slack_vals)
    norm_lfb = _normalize_higher_is_better(lfb_vals)
    norm_eff = _normalize_higher_is_better(eff_vals)
    norm_reqfs = _normalize_lower_is_better(reqfs_vals)
    norm_waste = _normalize_lower_is_better(waste_vals)

    q = (
        w_slack * norm_slack
        + w_lfb * norm_lfb
        + w_eff * norm_eff
        - w_reqfs * norm_reqfs
        - w_waste * norm_waste
    )
    return q


def select_fr_quality(
    env,
    request,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    lambda_N: float = 0.45,
    lambda_Q: float = 0.40,
    lambda_U: float = 0.15,
    top_m: int = 8,
    sample_k: int = 64,
    w_slack: float = 0.25,
    w_lfb: float = 0.25,
    w_eff: float = 0.20,
    w_reqfs: float = 0.15,
    w_waste: float = 0.15,
    use_overload_hinge: bool = False,
    tau_u: float = 0.8,
    lambda_O: float = 0.20,
) -> Optional[int]:
    """Quality-aware feasible-region C selector (v1.4 main C-side rule).

    Selects the legal (split, server) pair that maximizes a composite score:

        Psi = lambda_N * tilde{N}_R
            + lambda_Q * TopAvg_M(q_fast)
            - lambda_U * u_v_post
            - lambda_O * max(0, u_v_post - tau_u)   [optional hinge]

    where q_fast is a lightweight optical-quality proxy (no ranker, no rollout).
    """
    return _select_fr_quality_impl(
        env,
        request,
        obs_c,
        mask,
        lambda_N=lambda_N,
        lambda_Q=lambda_Q,
        lambda_U=lambda_U,
        top_m=top_m,
        sample_k=sample_k,
        w_slack=w_slack,
        w_lfb=w_lfb,
        w_eff=w_eff,
        w_reqfs=w_reqfs,
        w_waste=w_waste,
        use_overload_hinge=use_overload_hinge,
        tau_u=tau_u,
        lambda_O=lambda_O,
    )


def select_fr_quality_strong(
    env,
    request,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
) -> Optional[int]:
    """Overload-strong variant of fr_quality (lambda_U=0.25)."""
    return _select_fr_quality_impl(
        env,
        request,
        obs_c,
        mask,
        lambda_N=0.40,
        lambda_Q=0.35,
        lambda_U=0.25,
        top_m=8,
        sample_k=64,
        w_slack=0.25,
        w_lfb=0.25,
        w_eff=0.20,
        w_reqfs=0.15,
        w_waste=0.15,
        use_overload_hinge=False,
        tau_u=0.8,
        lambda_O=0.20,
    )


def _select_fr_quality_impl(
    env,
    request,
    obs_c: Dict[str, Any],
    mask: np.ndarray,
    lambda_N: float = 0.45,
    lambda_Q: float = 0.40,
    lambda_U: float = 0.15,
    top_m: int = 8,
    sample_k: int = 64,
    w_slack: float = 0.25,
    w_lfb: float = 0.25,
    w_eff: float = 0.20,
    w_reqfs: float = 0.15,
    w_waste: float = 0.15,
    use_overload_hinge: bool = False,
    tau_u: float = 0.8,
    lambda_O: float = 0.20,
) -> Optional[int]:
    """Shared implementation for fr_quality variants.

    Selects the legal (split, server) pair that maximizes a composite score:

        Psi = lambda_N * tilde{N}_R
            + lambda_Q * TopAvg_M(q_fast)
            - lambda_U * u_v_post
            - lambda_O * max(0, u_v_post - tau_u)   [optional hinge]

    where q_fast is a lightweight optical-quality proxy (no ranker, no rollout).
    """
    valid = _get_valid_actions(mask)
    if len(valid) == 0:
        return None

    num_servers = len(obs_c["server_utilizations"])
    features = obs_c["candidate_features"]

    count_scores: List[float] = []
    quality_scores: List[float] = []
    util_scores: List[float] = []
    hinge_scores: List[float] = []

    for idx in valid:
        idx = int(idx)
        split_id, server_id = _decode_c(idx, num_servers)
        feat = features[idx]

        # Post-admission server utilization
        server = env.mec.servers[server_id]
        edge_cost = float(feat.get("edge_compute_cost", 0.0))
        capacity = float(server.compute_capacity)
        u_post = float(server.utilization)
        if capacity > 0:
            u_post += edge_cost / capacity
        util_scores.append(u_post)
        hinge_scores.append(max(0.0, u_post - tau_u) if use_overload_hinge else 0.0)

        try:
            obs_r = build_agent_r_observation(env, request, split_id, server_id)
        except Exception:
            count_scores.append(0.0)
            quality_scores.append(float("-inf"))
            continue

        r_mask = np.asarray(obs_r["agent_r_mask"], dtype=bool)
        n_valid = int(r_mask.sum())
        count_scores.append(float(n_valid))

        if n_valid == 0:
            quality_scores.append(float("-inf"))
            continue

        sample_actions = _sample_r_actions_for_quality(obs_r, sample_k)
        q_vals = _compute_q_fast_for_actions(
            request, obs_r, feat, sample_actions,
            w_slack, w_lfb, w_eff, w_reqfs, w_waste,
        )

        top_m_actual = min(top_m, len(q_vals))
        if top_m_actual > 0:
            top_avg = float(np.mean(np.sort(q_vals)[-top_m_actual:]))
        else:
            top_avg = 0.0
        quality_scores.append(top_avg)

    counts_arr = np.asarray(count_scores, dtype=float)
    n_max = max(int(counts_arr.max()), 1)
    norm_counts = np.log1p(counts_arr) / np.log1p(n_max)

    q_arr = np.asarray(quality_scores, dtype=float)
    q_valid_mask = q_arr > -1e18  # distinguish -inf sentinel
    norm_q = np.zeros_like(q_arr)
    if q_valid_mask.any():
        q_min = q_arr[q_valid_mask].min()
        q_max = q_arr[q_valid_mask].max()
        if q_max > q_min:
            norm_q[q_valid_mask] = (q_arr[q_valid_mask] - q_min) / (q_max - q_min)
        else:
            norm_q[q_valid_mask] = 0.5

    util_arr = np.asarray(util_scores, dtype=float)
    norm_u = np.clip(util_arr, 0.0, 1.0)

    hinge_arr = np.asarray(hinge_scores, dtype=float)
    norm_hinge = np.clip(hinge_arr, 0.0, 1.0)

    total_scores = lambda_N * norm_counts + lambda_Q * norm_q - lambda_U * norm_u
    if use_overload_hinge:
        total_scores -= lambda_O * norm_hinge

    # If no candidate has any legal R action, fall back to count-util only.
    if not q_valid_mask.any():
        total_scores = norm_counts - lambda_U * norm_u
        if use_overload_hinge:
            total_scores -= lambda_O * norm_hinge

    best_local = int(np.argmax(total_scores))
    return int(valid[best_local])


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
        method_name: "ppo" | "greedy" | "wo" | "df" | "rf" | "df_fixed0" | "rf_fixed0" | "iwd"
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
    elif method == "df_fixed0":
        return select_df_fixed0(env, obs_c, mask)
    elif method == "rf_fixed0":
        return select_rf_fixed0(obs_c, mask)
    elif method == "iwd":
        return select_iwd(env, obs_c, mask, rng=rng, **kwargs)
    elif method == "oracle_r_query":
        agent_r = kwargs.get("agent_r")
        if agent_r is None:
            raise ValueError("oracle_r_query requires agent_r in kwargs")
        return select_oracle_r_query(env, request, obs_c, mask, agent_r)
    elif method == "oracle_pressure":
        return select_oracle_pressure(obs_c, mask, **kwargs)
    elif method == "fr_count":
        return select_fr_count(env, request, obs_c, mask, **kwargs)
    elif method == "fr_quality":
        return select_fr_quality(env, request, obs_c, mask, **kwargs)
    elif method == "fr_quality_strong":
        return select_fr_quality_strong(env, request, obs_c, mask)
    else:
        raise ValueError(f"Unknown offloading method: {method_name}")
