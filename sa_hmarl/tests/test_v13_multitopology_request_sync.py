"""Regression tests for Strict v1.3 request-sync and action-conservation fixes."""
from __future__ import annotations

import json
import tempfile
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from sa_hmarl.env.event_env import SMDPEnv
from sa_hmarl.env.request import DNNRequest, SplitProfile
from sa_hmarl.evaluation import eval_strict_v13_multitopology_cside_verified as ev
from sa_hmarl.evaluation import run_strict_v13_multitopology_cside_verified as runner
from sa_hmarl.evaluation import analyze_strict_v13_multitopology_cside_verified as ana
from sa_hmarl.mec.cluster import MECCluster
from sa_hmarl.network.modulation import ModulationRegistry
from sa_hmarl.network.optical_network import OpticalNetwork
from sa_hmarl.env.fs_demand import FSDemandCalculator


# ---------------------------------------------------------------------------
# E-stratum / depth-stratum definitions
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("max_path_idx, expected_e, expected_depth", [
    (4, "E=0", "D=0"),
    (5, "E=1", "D=1"),
    (9, "E=1", "D=1"),
    (10, "E=1", "D=2"),
    (20, "E=1", "D=3"),
])
def test_e_and_depth_stratum_definitions(max_path_idx, expected_e, expected_depth):
    has_candidates = True
    assert ev._e_label(max_path_idx, has_candidates) == expected_e
    assert ev._depth_stratum(max_path_idx) == int(expected_depth.split("=")[1])


def test_e_label_no_candidates():
    assert ev._e_label(-1, False) == "E=NA/no_candidate"


def test_no_e2_label_allowed():
    labels = {ev._e_label(i, True) for i in range(51)}
    assert "E=2" not in labels
    assert labels <= {"E=0", "E=1"}


# ---------------------------------------------------------------------------
# event_env.reject_next_request semantics
# ---------------------------------------------------------------------------

def _make_env_and_requests(n: int = 3):
    net = OpticalNetwork("net1", num_slots=32)
    mec = MECCluster(num_nodes=net.NUM_NODES, num_servers=1, seed=42, server_nodes=[0])
    mod_reg = ModulationRegistry()
    fs_calc = FSDemandCalculator()
    env = SMDPEnv(net, mec, mod_reg, fs_calc, k=3)
    requests = [
        DNNRequest(
            req_id=i,
            src_node=1,
            arrival_time=float(i),
            holding_time=2.0,
            deadline_ms=100.0,
            splits=[SplitProfile(0, 1.0, 0.5, 1.0)],
        )
        for i in range(n)
    ]
    env.reset(requests)
    return env, requests


def test_reject_next_request_consumes_expected_request():
    env, requests = _make_env_and_requests(3)
    assert len(env.event_queue) == 3
    req = requests[0]
    _, _, _, info = env.reject_next_request(req.req_id, "c_no_valid_action")
    assert info["success"] is False
    assert info["reason"] == "c_no_valid_action"
    assert info["req_id"] == req.req_id
    assert len(env.event_queue) == 2
    assert env.event_queue[0][2].req_id == requests[1].req_id
    assert env.stats["total_requests"] == 1
    assert env.stats["blocked"] == 1
    assert env.stats["accepted"] == 0


def test_reject_next_request_enforces_req_id_match():
    env, requests = _make_env_and_requests(3)
    with pytest.raises(RuntimeError, match="request id mismatch"):
        env.reject_next_request(999, "c_no_valid_action")


def test_reject_next_request_enforces_arrival_time_match():
    env, requests = _make_env_and_requests(3)
    # Corrupt the request object locally without changing the queued one
    bad_req = requests[0]
    bad_req.arrival_time = 99.0
    with pytest.raises(RuntimeError, match="arrival_time mismatch"):
        env.reject_next_request(bad_req.req_id, "c_no_valid_action")


def test_reject_next_request_empty_queue_raises():
    env, _ = _make_env_and_requests(0)
    with pytest.raises(RuntimeError, match="empty event queue"):
        env.reject_next_request(0, "c_no_valid_action")


def test_normal_env_step_info_req_id_matches_loop():
    env, requests = _make_env_and_requests(3)
    for req in requests:
        _, _, _, info = env.step((0, 0), (0, 0, 0))
        assert info["req_id"] == req.req_id
    assert len(env.event_queue) == 0


# ---------------------------------------------------------------------------
# C / R no-valid-action branches must consume exactly one request
# ---------------------------------------------------------------------------

def test_c_no_valid_consumes_request_and_does_not_call_r():
    env, requests = _make_env_and_requests(3)
    req = requests[0]
    queue_len_before = len(env.event_queue)
    with patch.object(env, "step") as mock_step:
        _, _, _, info = env.reject_next_request(req.req_id, "c_no_valid_action")
    mock_step.assert_not_called()
    assert len(env.event_queue) == queue_len_before - 1
    assert info["req_id"] == req.req_id


def test_r_no_valid_consumes_request_without_action_zero():
    env, requests = _make_env_and_requests(3)
    req = requests[0]
    queue_len_before = len(env.event_queue)
    with patch.object(env, "step") as mock_step:
        _, _, _, info = env.reject_next_request(req.req_id, "r_no_valid_action")
    mock_step.assert_not_called()
    assert len(env.event_queue) == queue_len_before - 1
    assert info["req_id"] == req.req_id


# ---------------------------------------------------------------------------
# Warmup + evaluated request sync
# ---------------------------------------------------------------------------

def test_warmup_rejects_do_not_desync_evaluated_requests():
    env, requests = _make_env_and_requests(5)
    # Simulate two warmup rejects followed by normal steps for evaluated requests
    for req in requests[:2]:
        env.reject_next_request(req.req_id, "c_no_valid_action")
    for req in requests[2:]:
        assert env.event_queue[0][2].req_id == req.req_id
        _, _, _, info = env.step((0, 0), (0, 0, 0))
        assert info["req_id"] == req.req_id
    assert len(env.event_queue) == 0


# ---------------------------------------------------------------------------
# Action-observation consistency validation
# ---------------------------------------------------------------------------

def _make_valid_obs():
    mod_reg = ModulationRegistry.from_profile("default")
    num_paths = 3
    num_mods = mod_reg.num_formats
    num_blocks = 2
    mask = np.zeros(num_paths * num_mods * num_blocks, dtype=bool)
    # Make action 1 legal: path=0, mod=0, block=1
    mask[1] = True
    return {
        "agent_r_mask": mask,
        "candidate_paths": [[0, 1], [0, 1, 2], [0, 1, 2, 3]],
        "mod_names": list(mod_reg.names),
        "path_features": [
            {"path_length_km": 50.0, "hop_count": 1.0},
            {"path_length_km": 100.0, "hop_count": 2.0},
            {"path_length_km": 200.0, "hop_count": 3.0},
        ],
        "required_fs_per_path_mod": [[2] * num_mods for _ in range(num_paths)],
        "candidate_blocks_per_path_mod": [
            [[(0, 1), (5, 2)] for _ in range(num_mods)] for _ in range(num_paths)
        ],
    }


def test_validate_selected_r_action_accepts_legal_action():
    obs = _make_valid_obs()
    mod_reg = ModulationRegistry.from_profile("default")
    r_action = (0, 0, 1)
    ev._validate_selected_r_action(obs, 1, r_action, mod_reg)


def test_validate_selected_r_action_rejects_mask_false():
    obs = _make_valid_obs()
    mod_reg = ModulationRegistry.from_profile("default")
    with pytest.raises(ev.ActionObservationConsistencyError):
        ev._validate_selected_r_action(obs, 0, (0, 0, 0), mod_reg)


def test_validate_selected_r_action_rejects_modulation_reach():
    obs = _make_valid_obs()
    mod_reg = ModulationRegistry.from_profile("default")
    obs["path_features"][0]["path_length_km"] = 1e6
    with pytest.raises(ev.ActionObservationConsistencyError, match="reach"):
        ev._validate_selected_r_action(obs, 1, (0, 0, 1), mod_reg)


def test_validate_selected_r_action_rejects_block_too_small():
    obs = _make_valid_obs()
    mod_reg = ModulationRegistry.from_profile("default")
    obs["candidate_blocks_per_path_mod"][0][0][1] = (5, 1)
    with pytest.raises(ev.ActionObservationConsistencyError, match="block_size"):
        ev._validate_selected_r_action(obs, 1, (0, 0, 1), mod_reg)


# ---------------------------------------------------------------------------
# KSP-FF highest action audit
# ---------------------------------------------------------------------------

def test_ksp_ff_highest_action_properties():
    from sa_hmarl.baselines.rmsa_baselines import ksp_ff_highest_mod_action

    mod_reg = ModulationRegistry.from_profile("default")
    num_paths = 3
    num_mods = mod_reg.num_formats
    num_blocks = 3
    feasible = [[True] * num_mods for _ in range(num_paths)]
    # BPSK needs 4, QPSK needs 2 on every path -> highest picks QPSK
    req_fs = [[4, 2, None, None] for _ in range(num_paths)]
    blocks = [
        [[(0, 4), (5, 4), (10, 4)], [(0, 2), (5, 2), (10, 2)], [], []]
        for _ in range(num_paths)
    ]
    from sa_hmarl.env.action_mask import build_agent_r_mask
    mask = build_agent_r_mask(num_paths, num_mods, num_blocks, list(mod_reg.names), feasible, req_fs, blocks)
    obs = {
        "agent_r_mask": mask,
        "candidate_paths": [[] for _ in range(num_paths)],
        "mod_names": list(mod_reg.names),
        "feasible_mask_per_path_mod": feasible,
        "required_fs_per_path_mod": req_fs,
        "candidate_blocks_per_path_mod": blocks,
    }
    action = ksp_ff_highest_mod_action(obs)
    path_idx, mod_idx, block_idx = ev.decode_agent_r_action(action, num_mods, num_blocks)
    assert 0 <= path_idx < num_paths
    assert mask[action]
    assert mod_idx == 1  # QPSK
    assert blocks[path_idx][mod_idx][block_idx][1] >= req_fs[path_idx][mod_idx]


# ---------------------------------------------------------------------------
# Failure decomposition conservation
# ---------------------------------------------------------------------------

def test_failure_decomposition_conservation():
    blocked = 100
    components = {
        "c_no_valid_action": 5,
        "r_no_valid_action": 10,
        "no_suitable_block": 20,
        "server_overload": 30,
        "deadline_failure": 25,
        "other_failure": 10,
    }
    assert sum(components.values()) == blocked


# ---------------------------------------------------------------------------
# Delta direction consistency
# ---------------------------------------------------------------------------

def test_delta_direction_strict_better():
    base = np.array([0.10, 0.12, 0.11, 0.13, 0.10])
    strict = np.array([0.05, 0.04, 0.06, 0.05, 0.05])
    mean, lo, hi = ana._paired_bootstrap_ci(base, strict)
    sign = ana._exact_sign_flip(base, strict)
    assert mean > 0
    assert lo > 0
    assert sign["one_sided_p"] < 0.05


def test_delta_direction_strict_worse():
    base = np.array([0.05, 0.04, 0.06, 0.05, 0.05])
    strict = np.array([0.10, 0.12, 0.11, 0.13, 0.10])
    mean, lo, hi = ana._paired_bootstrap_ci(base, strict)
    sign = ana._exact_sign_flip(base, strict)
    assert mean < 0
    assert hi < 0
    assert sign["one_sided_p"] > 0.95


# ---------------------------------------------------------------------------
# Phase output isolation
# ---------------------------------------------------------------------------

def test_phase_output_dirs_do_not_overlap():
    base = Path("sa_hmarl/experiments/v13_strict_multitopology_cside_verified_v2")
    micro_dir = runner._task_dir("t", 3030, "micro", base)
    smoke_dir = runner._task_dir("t", 3030, "smoke", base)
    full_dir = runner._task_dir("t", 3030, "full", base)
    assert micro_dir != smoke_dir != full_dir
    assert "micro" in str(micro_dir)
    assert "smoke" in str(smoke_dir)
    assert "full" in str(full_dir)


# ---------------------------------------------------------------------------
# Source manifest
# ---------------------------------------------------------------------------

def test_source_manifest_ksp_path_valid():
    root = Path(__file__).resolve().parents[2]
    manifest = runner._source_manifest(root)
    assert "sa_hmarl/sa_hmarl/network/ksp.py" in manifest
    assert manifest["sa_hmarl/sa_hmarl/network/ksp.py"] not in ("missing", "N/A", "")
    assert "sa_hmarl/sa_hmarl/baselines/ksp.py" not in manifest


def test_source_manifest_no_missing_or_na():
    root = Path(__file__).resolve().parents[2]
    manifest = runner._source_manifest(root)
    for rel, sha in manifest.items():
        assert sha not in ("missing", "N/A", ""), f"{rel} has invalid hash {sha}"


# ---------------------------------------------------------------------------
# Full cannot bypass smoke marker
# ---------------------------------------------------------------------------

def test_full_refuses_without_smoke_marker():
    root = Path(__file__).resolve().parents[1]
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        manifest = runner._source_manifest(root)
        ok, reason = runner._check_smoke_marker(
            base, manifest, runner.RANKER_CKPT, runner.OLD_RANKER_CKPT, runner.AGENT_R_CKPT,
            ["xlron_cost239_ptrnet_real"], [3030], ["ppo_c", "df_c"],
            "ppo_r_top1,ksp_ff_highest,strict_v13", 20, 50,
        )
        assert not ok
        assert "smoke marker missing" in reason


# ---------------------------------------------------------------------------
# Resume / hash validation
# ---------------------------------------------------------------------------

def test_micro_evaluator_output_hash_validation():
    root = Path(__file__).resolve().parents[2]
    with tempfile.TemporaryDirectory() as tmp:
        task_dir = Path(tmp)
        task = {
            "topology": "xlron_cost239_ptrnet_real",
            "seed": 3030,
            "c_modes": ["ppo_c", "df_c"],
            "c_checkpoints": [
                "sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt",
                "_heuristic_",
            ],
            "arrival_interval": 0.0625,
            "edge_cost_max": 2.2,
            "warmup_requests": 5,
            "requests_per_episode": 20,
            "task_dir": task_dir,
        }
        manifest = runner._source_manifest(root)
        expected = runner._expected_hashes(
            task, root, runner.RANKER_CKPT, runner.OLD_RANKER_CKPT,
            runner.AGENT_R_CKPT, "ppo_r_top1,ksp_ff_highest,strict_v13", manifest
        )
        payload = {
            "schema_version": runner.SCHEMA_VERSION,
            "config_hash": expected["config_hash"],
            "evaluator_code_hash": expected["evaluator_code_hash"],
            "runner_code_hash": expected["runner_code_hash"],
            "analyzer_code_hash": expected["analyzer_code_hash"],
            "source_manifest": expected["source_manifest"],
            "request_trace_hash": "trace_hash_abc",
            "checkpoint_sha256s": {
                "agent_r": expected["agent_r_sha256"],
                **expected["c_checkpoint_sha256s"],
                **expected["ranker_sha256s"],
            },
            "results": [{"topology": task["topology"], "c_mode": "ppo_c", "r_mode": "strict_v13", "seed": task["seed"], "request_trace_hash": "trace_hash_abc"}],
        }
        (task_dir / "results.json").write_text(json.dumps(payload))
        (task_dir / "done.marker").write_text("done")
        assert runner._is_done(task_dir, expected)
