"""PreD-DQN input feature construction."""
from __future__ import annotations

from typing import Dict, List, Union

import numpy as np

from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, RMSAAction
from sa_hmarl.pds_rmsa.env.reservations import PostDecisionState, TrueState
from sa_hmarl.pds_rmsa.env.topology import Topology, load_topology
from sa_hmarl.pds_rmsa.features.afterstate import (
    build_afterstate_features,
    build_afterstate_features_batch,
)
from sa_hmarl.pds_rmsa.protocol import (
    BITRATE_CHOICES_GBPS,
    MEAN_HOLDING_TIME,
    MODULATION_TABLE,
    Request,
)


_MODULATION_ORDER = [name for name, _reach, _se in MODULATION_TABLE]
_BITRATE_ORDER = list(BITRATE_CHOICES_GBPS)

_topology_cache: Dict[str, Topology] = {}


def _get_topology(topology_name: str) -> Topology:
    if topology_name not in _topology_cache:
        _topology_cache[topology_name] = load_topology(topology_name)
    return _topology_cache[topology_name]


def _one_hot(index: int, dim: int) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    if 0 <= index < dim:
        vec[index] = 1.0
    return vec


def _build_action_context(
    request: Request,
    action: Union[RMSAAction, BlockedAction],
    feature_config: dict,
) -> np.ndarray:
    """Build the PreD request/action context vector.

    Dimensions: src one-hot(11) + dst one-hot(11) + bitrate one-hot(4) +
    holding norm(1) + path_rank one-hot(5) + modulation one-hot(4) +
    start_slot norm(1) + required_fs norm(1) + path_hops norm(1) + path_km norm(1) = 40.
    """
    topology_name = feature_config.get("topology_name")
    topology = _get_topology(topology_name) if topology_name else None

    num_nodes = int(feature_config.get("num_nodes", 11))
    num_slots = int(feature_config.get("num_slots", 50))
    k_paths = int(feature_config.get("k_paths", 5))
    max_hops = float(feature_config.get("max_hops", 10))
    max_path_km = float(feature_config.get("max_path_km", 10000.0))
    max_required_fs = float(feature_config.get("max_required_fs", 10.0))
    mean_holding = float(feature_config.get("mean_holding_time", MEAN_HOLDING_TIME))

    src = _one_hot(request.src_node, num_nodes)
    dst = _one_hot(request.dst_node, num_nodes)
    bitrate = _one_hot(_BITRATE_ORDER.index(request.bitrate_gbps), len(_BITRATE_ORDER))
    holding = np.array([request.holding_time / mean_holding], dtype=np.float32)

    if isinstance(action, RMSAAction):
        path_rank = _one_hot(action.path_rank, k_paths)
        modulation = _one_hot(_MODULATION_ORDER.index(action.modulation), len(_MODULATION_ORDER))
        start_slot = np.array([action.start_slot / max(1, num_slots - 1)], dtype=np.float32)
        required_fs = np.array([action.required_fs / max_required_fs], dtype=np.float32)
        path_hops = np.array([(len(action.path) - 1) / max_hops], dtype=np.float32)
        path_km = (
            np.array([topology.path_length_km(action.path) / max_path_km], dtype=np.float32)
            if topology is not None
            else np.zeros(1, dtype=np.float32)
        )
    else:
        path_rank = np.zeros(k_paths, dtype=np.float32)
        modulation = np.zeros(len(_MODULATION_ORDER), dtype=np.float32)
        start_slot = np.zeros(1, dtype=np.float32)
        required_fs = np.zeros(1, dtype=np.float32)
        path_hops = np.zeros(1, dtype=np.float32)
        path_km = np.zeros(1, dtype=np.float32)

    return np.concatenate(
        [src, dst, bitrate, holding, path_rank, modulation, start_slot, required_fs, path_hops, path_km]
    )


def build_pred_features(
    pre_state: Union[PostDecisionState, TrueState],
    request: Request,
    action: Union[RMSAAction, BlockedAction],
    feature_config: dict,
) -> np.ndarray:
    """Build PreD-DQN input features: resource features + request/action context.

    The resource features are the same as the afterstate builder applied to the
    pre-state.  The PDS-style approach uses the same resource features with a
    zeroed context vector.
    """
    probe_suite = feature_config.get("probe_suite")
    if probe_suite is None:
        raise ValueError("feature_config must contain 'probe_suite'")

    afterstate = build_afterstate_features(pre_state, probe_suite, feature_config)
    context = _build_action_context(request, action, feature_config)
    return np.concatenate([afterstate, context]).astype(np.float32)


def build_pds_features(
    post_state: Union[PostDecisionState, TrueState],
    feature_config: dict,
) -> np.ndarray:
    """Build PDS-MLP input features: resource features + zero context vector.

    The context coordinates are all zeros, matching the PreD input dimension so
    that the PDS and PreD networks share the same architecture and parameter
    count.
    """
    probe_suite = feature_config.get("probe_suite")
    if probe_suite is None:
        raise ValueError("feature_config must contain 'probe_suite'")

    context_dim = int(feature_config.get("context_dim", 40))
    afterstate = build_afterstate_features(post_state, probe_suite, feature_config)
    zeros = np.zeros(context_dim, dtype=np.float32)
    return np.concatenate([afterstate, zeros]).astype(np.float32)


def build_pred_features_for_actions(
    pre_state: Union[PostDecisionState, TrueState],
    request,
    actions,
    feature_config: dict,
) -> np.ndarray:
    """Efficiently build PreD features for many actions sharing the same pre-state.

    The expensive afterstate/resource features are computed once and shared across
    all actions; only the small action-context vectors are built per action.
    """
    probe_suite = feature_config.get("probe_suite")
    if probe_suite is None:
        raise ValueError("feature_config must contain 'probe_suite'")

    resource = build_afterstate_features(pre_state, probe_suite, feature_config)
    features = []
    for action in actions:
        context = _build_action_context(request, action, feature_config)
        features.append(np.concatenate([resource, context]))
    return np.array(features, dtype=np.float32)


def build_pred_features_batch(
    pre_states: List[Union[PostDecisionState, TrueState]],
    requests: List[Request],
    actions: List[Union[RMSAAction, BlockedAction]],
    feature_config: dict,
) -> np.ndarray:
    """Build one PreD feature per state/request/action without repeated probes."""
    if not (len(pre_states) == len(requests) == len(actions)):
        raise ValueError("pre_states, requests, and actions must have equal length")
    if not pre_states:
        return np.zeros((0, 0), dtype=np.float32)
    probe_suite = feature_config.get("probe_suite")
    if probe_suite is None:
        raise ValueError("feature_config must contain 'probe_suite'")
    resources = build_afterstate_features_batch(
        pre_states, probe_suite, feature_config
    )
    contexts = np.stack([
        _build_action_context(request, action, feature_config)
        for request, action in zip(requests, actions)
    ]).astype(np.float32)
    return np.concatenate([resources, contexts], axis=1).astype(np.float32)


def build_pred_features_grouped(
    pre_states: List[Union[PostDecisionState, TrueState]],
    requests: List[Request],
    action_groups: List[List[Union[RMSAAction, BlockedAction]]],
    feature_config: dict,
) -> tuple[np.ndarray, List[int]]:
    """Flatten ragged candidate groups while computing each resource state once."""
    if not (len(pre_states) == len(requests) == len(action_groups)):
        raise ValueError("pre_states, requests, and action_groups must have equal length")
    counts = [len(actions) for actions in action_groups]
    total = sum(counts)
    if total == 0:
        return np.zeros((0, 0), dtype=np.float32), counts

    probe_suite = feature_config.get("probe_suite")
    if probe_suite is None:
        raise ValueError("feature_config must contain 'probe_suite'")
    resources = build_afterstate_features_batch(
        pre_states, probe_suite, feature_config
    )
    repeated_resources = np.repeat(resources, counts, axis=0)
    contexts = np.stack([
        _build_action_context(request, action, feature_config)
        for request, actions in zip(requests, action_groups)
        for action in actions
    ]).astype(np.float32)
    return np.concatenate(
        [repeated_resources, contexts], axis=1
    ).astype(np.float32), counts


def build_pds_features_batch(
    post_states: List[Union[PostDecisionState, TrueState]],
    feature_config: dict,
) -> np.ndarray:
    """Batch PDS feature construction: resource features + zero context vectors.

    This is much faster than looping over ``build_pds_features`` when many
    post-decision states must be evaluated at once (e.g. for PDS target candidates).
    """
    probe_suite = feature_config.get("probe_suite")
    if probe_suite is None:
        raise ValueError("feature_config must contain 'probe_suite'")

    context_dim = int(feature_config.get("context_dim", 40))
    afterstates = build_afterstate_features_batch(post_states, probe_suite, feature_config)
    zeros = np.zeros((len(post_states), context_dim), dtype=np.float32)
    return np.concatenate([afterstates, zeros], axis=1).astype(np.float32)


def context_dim(feature_config: dict) -> int:
    """Return the context dimensionality (default 40 for COST239)."""
    return int(feature_config.get("context_dim", 40))
