"""Unit and integration tests for expanded-path K=500 rescue.

These tests cover the deterministic rescue action selector
(`expanded_ksp_ff_rescue_action`) and the rescue wrapper logic in the Phase-R0
evaluation script.
"""
import sys
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pytest

from sa_hmarl.baselines.rmsa_baselines import expanded_ksp_ff_rescue_action
from sa_hmarl.env.observation_builder import decode_agent_r_action
from sa_hmarl.evaluation import generate_r_mask_empty_rescue_phase_r0 as rescue
from sa_hmarl.network.modulation import ModulationRegistry


# ---------------------------------------------------------------------------
# Synthetic obs builder
# ---------------------------------------------------------------------------
def _make_obs(
    num_paths: int,
    num_mods: int,
    num_blocks: int,
    feasible: list,  # list of (path_idx, mod_idx, block_idx, start_slot, size)
    mod_names: list = None,
) -> dict:
    """Build a synthetic observation for `expanded_ksp_ff_rescue_action`."""
    if mod_names is None:
        mod_names = ["BPSK", "QPSK", "8QAM", "16QAM"][:num_mods]
    mask = np.zeros(num_paths * num_mods * num_blocks, dtype=bool)
    required_fs = [[None] * num_mods for _ in range(num_paths)]
    blocks = [[[] for _ in range(num_mods)] for _ in range(num_paths)]
    for path_idx, mod_idx, block_idx, start_slot, size in feasible:
        action_idx = path_idx * num_mods * num_blocks + mod_idx * num_blocks + block_idx
        mask[action_idx] = True
        if required_fs[path_idx][mod_idx] is None:
            # Use the first feasible block size as the required FS for this mod.
            required_fs[path_idx][mod_idx] = size
        blocks[path_idx][mod_idx].append((start_slot, size))
    return {
        "candidate_paths": [list(range(i, i + 3)) for i in range(num_paths)],
        "mod_names": mod_names,
        "required_fs_per_path_mod": required_fs,
        "candidate_blocks_per_path_mod": blocks,
        "agent_r_mask": mask,
    }


# ---------------------------------------------------------------------------
# Unit tests for expanded_ksp_ff_rescue_action
# ---------------------------------------------------------------------------

def test_k50_non_empty_returns_none():
    """A feasible action in the K=50 pool should not trigger rescue."""
    obs = _make_obs(num_paths=60, num_mods=2, num_blocks=2, feasible=[(0, 0, 0, 10, 4)])
    assert expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59) is None


def test_path_50_feasible_returns_path_50():
    """If paths 0-49 are infeasible and path 50 is feasible, rescue picks path 50."""
    obs = _make_obs(num_paths=60, num_mods=2, num_blocks=2, feasible=[(50, 0, 0, 10, 4)])
    action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59)
    assert action == 50 * 2 * 2 + 0 * 2 + 0


def test_path_101_feasible_returns_path_101():
    """Rescue can find a feasible action deep in the expanded path list."""
    obs = _make_obs(num_paths=102, num_mods=2, num_blocks=2, feasible=[(101, 0, 0, 10, 4)])
    action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=101)
    assert action == 101 * 2 * 2 + 0 * 2 + 0


def test_picks_lowest_ranked_path():
    """Multiple feasible expanded paths: the lowest-ranked path wins."""
    obs = _make_obs(
        num_paths=120, num_mods=2, num_blocks=2,
        feasible=[(55, 0, 0, 10, 4), (52, 0, 0, 10, 4)],
    )
    action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=119)
    expected = 52 * 2 * 2 + 0 * 2 + 0
    assert action == expected


def test_picks_smallest_required_fs():
    """On the same path, the modulation requiring the fewest FS is chosen."""
    # mod0 (BPSK) needs 4 FS, mod1 (QPSK) needs 2 FS.
    obs = _make_obs(
        num_paths=60, num_mods=2, num_blocks=2,
        feasible=[(50, 0, 0, 10, 4), (50, 1, 0, 10, 2)],
    )
    action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59)
    expected = 50 * 2 * 2 + 1 * 2 + 0
    assert action == expected


def test_tie_break_higher_spectral_efficiency():
    """If two modulations require the same FS, the higher-SE one wins."""
    # BPSK (mod0) and QPSK (mod1) both need 3 FS; QPSK has higher SE.
    obs = _make_obs(
        num_paths=60, num_mods=2, num_blocks=2,
        feasible=[(50, 0, 0, 10, 3), (50, 1, 0, 10, 3)],
    )
    action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59)
    expected = 50 * 2 * 2 + 1 * 2 + 0
    assert action == expected


def test_picks_lowest_start_slot():
    """For the same path/mod, the lowest start-slot block is chosen."""
    obs = _make_obs(
        num_paths=60, num_mods=2, num_blocks=2,
        feasible=[(50, 0, 0, 5, 4), (50, 0, 1, 2, 4)],
    )
    action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59)
    expected = 50 * 2 * 2 + 0 * 2 + 1
    assert action == expected


def test_k500_fully_infeasible_returns_none():
    """When no expanded path is feasible, rescue returns None."""
    obs = _make_obs(num_paths=500, num_mods=2, num_blocks=2, feasible=[])
    assert expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=499) is None


def test_min_path_idx_enforced():
    """A feasible path below min_path_idx is ignored."""
    obs = _make_obs(num_paths=60, num_mods=2, num_blocks=2, feasible=[(49, 0, 0, 10, 4)])
    assert expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59) is None


def test_obs_not_mutated():
    """The rescue selector must not mutate the input observation."""
    import copy
    obs = _make_obs(
        num_paths=60, num_mods=2, num_blocks=2,
        feasible=[(50, 0, 0, 10, 4)],
    )
    obs_before = copy.deepcopy(obs)
    expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59)
    assert np.array_equal(obs_before["agent_r_mask"], obs["agent_r_mask"])
    assert obs_before["required_fs_per_path_mod"] == obs["required_fs_per_path_mod"]
    assert obs_before["candidate_blocks_per_path_mod"] == obs["candidate_blocks_per_path_mod"]


def test_returned_action_satisfies_mask():
    """The returned action index must be legal in the rescue mask."""
    obs = _make_obs(
        num_paths=60, num_mods=2, num_blocks=2,
        feasible=[(55, 1, 0, 7, 2)],
    )
    action = expanded_ksp_ff_rescue_action(obs, min_path_idx=50, max_path_idx=59)
    assert action is not None
    assert obs["agent_r_mask"][action]
    path_idx, mod_idx, block_idx = decode_agent_r_action(action, 2, 2)
    assert path_idx >= 50


# ---------------------------------------------------------------------------
# Integration test: run a short fixed-C episode and check rescue invariants
# ---------------------------------------------------------------------------
@dataclass
class _TestArgs:
    """Minimal args namespace for the integration test."""
    num_slots: int = 320
    num_servers: int = 4
    k_paths_r: int = 50
    path_sort_strategy_r: str = "hops"
    block_sort_strategy_r: str = "start_asc"
    modulation_profile: str = "default"
    max_blocks: int = 10
    split_profile: str = "default3"
    num_splits: int = 3
    arrival_interval: float = 0.3
    holding_min: float = 20.0
    holding_max: float = 30.0
    deadline_min: float = 30.0
    deadline_max: float = 100.0
    size_min_mb: float = 5.0
    size_max_mb: float = 30.0
    edge_cost_min: float = 0.1
    edge_cost_max: float = 2.2
    requests_per_episode: int = 500
    warmup_requests: int = 500
    poisson_arrivals: bool = False
    exponential_holding: bool = False
    fixed_split_id: int = 0
    device: str = "cpu"


def test_rescue_episode_invariants():
    """Run a short fixed-C episode and verify rescue invariants."""
    args = _TestArgs()
    topology = "xlron_cost239_ptrnet_real"
    seed = 5001
    total_requests = args.warmup_requests + args.requests_per_episode

    env_proto = rescue._base._build_env(args, topology, seed)
    server_node_ids = [int(s.node_id) for s in env_proto.mec.servers]
    rng = np.random.RandomState(seed)
    requests = rescue._base._generate_all_od_requests(
        num_nodes=env_proto.net.NUM_NODES,
        server_node_ids=server_node_ids,
        rng=rng,
        num_requests=total_requests,
        arrival_interval=args.arrival_interval,
        holding_min=args.holding_min,
        holding_max=args.holding_max,
        deadline_min=args.deadline_min,
        deadline_max=args.deadline_max,
        size_min_mb=args.size_min_mb,
        size_max_mb=args.size_max_mb,
        edge_cost_min=args.edge_cost_min,
        edge_cost_max=args.edge_cost_max,
        num_splits=args.num_splits,
        split_profile=args.split_profile,
        poisson_arrivals=args.poisson_arrivals,
        exponential_holding=args.exponential_holding,
    )

    # Baseline preferred heuristic (no rescue) for the same request trace.
    env_baseline = rescue._base._build_env(args, topology, seed)
    env_baseline.reset(requests)
    _, records_baseline, _ = rescue._run_episode(
        env_baseline, requests, None, None, "preferred_heuristic", args, seed, topology
    )

    # Rescue variant of the same preferred heuristic.
    env_rescue = rescue._base._build_env(args, topology, seed)
    env_rescue.reset(requests)
    _, records_rescue, _ = rescue._run_episode(
        env_rescue, requests, None, None, "preferred_heuristic_k500_rescue", args, seed, topology
    )

    # Invariants after the episode.
    assert env_rescue.k == 50
    assert env_rescue.path_sort_strategy == "hops"
    assert env_rescue.block_sort_strategy == "start_asc"

    baseline_k50_empty = 0
    for rec_b, rec_r in zip(records_baseline, records_rescue):
        if rec_r.warmup:
            continue
        if rec_r.rescue_attempted:
            # Rescue is only attempted when the baseline K=50 mask was empty.
            assert rec_b.failure_reason == "r_no_valid_action", (
                f"step {rec_r.step_idx}: rescue attempted but baseline reason is {rec_b.failure_reason}"
            )
            if rec_r.rescue_success:
                assert rec_r.rescue_path_idx is not None and rec_r.rescue_path_idx >= 50
            else:
                assert rec_r.failure_reason == "r_no_valid_action_after_k500_rescue"
        if not rec_b.success and rec_b.failure_reason == "r_no_valid_action":
            baseline_k50_empty += 1

    # Rescue successes cannot exceed K50-empty baseline events.
    rescue_success = sum(
        1 for r in records_rescue if not r.warmup and r.rescue_success
    )
    assert rescue_success <= baseline_k50_empty

    # Sanity: there should be some evaluated requests.
    assert len(records_rescue) == args.requests_per_episode
    print(f"test_rescue_episode_invariants: K50-empty={baseline_k50_empty}, rescue_success={rescue_success}")
