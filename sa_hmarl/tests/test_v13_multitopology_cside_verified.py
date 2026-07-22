"""Unit tests for strict v1.3 verified multi-topology evaluator."""
from __future__ import annotations

from types import SimpleNamespace
import json
from pathlib import Path
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

from sa_hmarl.evaluation import eval_strict_v13_multitopology_cside_verified as ev
from sa_hmarl.evaluation import run_strict_v13_multitopology_cside_verified as runner
from sa_hmarl.evaluation import analyze_strict_v13_multitopology_cside_verified as ana


def _make_obs_c(mask_values):
    return {"agent_c_mask": np.asarray(mask_values, dtype=bool)}


def _make_obs_r(mask_values, num_mods=3):
    n = len(mask_values)
    return {
        "agent_r_mask": np.asarray(mask_values, dtype=bool),
        "mod_names": ["QPSK", "8QAM", "16QAM"][:num_mods],
        "path_features": [{"path_length_km": float(i * 100), "hop_count": float(i + 1)} for i in range(n)],
        "candidate_blocks_per_path_mod": [[[(j, j + 2) for j in range(5)] for _ in range(num_mods)] for _ in range(n)],
        "required_fs_per_path_mod": [[2] * num_mods for _ in range(n)],
    }


class TestSelectCAction:
    def test_baseline_returns_none_when_mask_empty(self):
        env = SimpleNamespace(net=SimpleNamespace(num_slots=320), mec=SimpleNamespace(servers=[0, 1, 2, 3]))
        obs_c = _make_obs_c([])
        with patch.object(ev, "select_offloading_action", return_value=0) as mock_select:
            flat, split, server, valid, info = ev._select_c_action("df_c", None, env, object(), obs_c, 4)
        assert not valid
        assert flat is None
        assert info["raw_mask_empty"]
        mock_select.assert_not_called()

    def test_ppo_c_returns_none_when_action_not_in_effective_mask(self):
        env = SimpleNamespace(net=SimpleNamespace(num_slots=320), mec=SimpleNamespace(servers=[0, 1, 2, 3]))
        obs_c = _make_obs_c([True] * 12)
        risk_mask = np.zeros(12)
        risk_mask[5] = 1
        with patch.object(ev, "_select_c_action_from_obs", return_value=(3, np.ones(12), risk_mask)) as mock:
            flat, split, server, valid, info = ev._select_c_action("ppo_c", object(), env, object(), obs_c, 4)
        assert not valid
        assert flat is None
        assert info["action_in_effective_mask"] is False


class TestSelectRAction:
    def test_all_r_modes_invalid_when_mask_empty(self):
        """Empty R mask must make every R mode invalid (no action=0 fallback)."""
        obs_r = _make_obs_r([])
        for r_mode in ["ppo_r_top1", "ksp_ff_highest", "strict_v13", "old_v13"]:
            idx, valid, info = ev._select_r_action(r_mode, None, None, None, None, None, obs_r, 0, 0, 0)
            assert not valid, f"{r_mode} must be invalid on empty mask"
            assert info["r_mask_empty"]

    def test_ppo_r_top1_returns_index_when_mask_nonempty(self):
        obs_r = _make_obs_r([False, True, False])
        idx, valid, info = ev._select_r_action("ppo_r_top1", None, None, None, None, None, obs_r, 42, 0, 0)
        assert valid
        assert idx == 42


class TestRecordFailure:
    def test_server_saturated_merged(self):
        res = ev.MethodResult("t", 1, "ppo_c", "ksp", "test")
        ev._record_failure(res, "server_saturated")
        ev._record_failure(res, "server_overload")
        assert res.server_overload == 2

    def test_r_no_valid_action_not_other(self):
        res = ev.MethodResult("t", 1, "ppo_c", "ksp", "test")
        ev._record_failure(res, "r_no_valid_action")
        assert res.other_failure == 0


class TestValidateResultRow:
    def test_conservation_passes(self):
        row = {
            "schema_version": ev.SCHEMA_VERSION,
            "topology": "t", "seed": 1, "c_mode": "ppo_c", "r_mode": "ksp_ff_highest", "method_name": "m",
            "evaluated_requests": 100, "admitted": 85, "blocked": 15,
            "r_reached": 98, "r_decisions": 95,
            "c_no_valid_action": 2, "r_no_valid_action": 3,
            "failure_reason_distribution": {"c_no_valid_action": 2, "r_no_valid_action": 3, "no_suitable_block": 10},
            "ppo_agreement_rate": 0.6, "ppo_agreement_numerator": 57, "ppo_agreement_denominator": 95,
            "request_trace_hash": "a", "selected_action_hash": "b", "config_hash": "c",
            "evaluator_code_hash": "d", "runner_code_hash": "e", "analyzer_code_hash": "f",
            "source_manifest": {}, "checkpoint_sha256s": {},
            "selected_path_idx_distribution": {"normalized": {"0": 0.5, "1": 0.5}},
            "selected_mod_distribution": {"normalized": {"QPSK": 1.0}},
            "admitted_path_idx_distribution": {"normalized": {"0": 1.0}},
            "admitted_mod_distribution": {"normalized": {"QPSK": 1.0}},
        }
        ev._validate_result_row(row)

    def test_ppo_r_top1_agreement(self):
        row = {
            "schema_version": ev.SCHEMA_VERSION,
            "topology": "t", "seed": 1, "c_mode": "ppo_c", "r_mode": "ppo_r_top1", "method_name": "m",
            "evaluated_requests": 100, "admitted": 90, "blocked": 10,
            "r_reached": 100, "r_decisions": 100,
            "c_no_valid_action": 0, "r_no_valid_action": 0,
            "failure_reason_distribution": {"server_overload": 10},
            "ppo_agreement_rate": 1.0, "ppo_agreement_numerator": 100, "ppo_agreement_denominator": 100,
            "request_trace_hash": "a", "selected_action_hash": "b", "config_hash": "c",
            "evaluator_code_hash": "d", "runner_code_hash": "e", "analyzer_code_hash": "f",
            "source_manifest": {}, "checkpoint_sha256s": {},
            "selected_path_idx_distribution": {"normalized": {"0": 1.0}},
            "selected_mod_distribution": {"normalized": {"QPSK": 1.0}},
            "admitted_path_idx_distribution": {"normalized": {"0": 1.0}},
            "admitted_mod_distribution": {"normalized": {"QPSK": 1.0}},
        }
        ev._validate_result_row(row)

    def test_agreement_null_when_no_decisions(self):
        row = {
            "schema_version": ev.SCHEMA_VERSION,
            "topology": "t", "seed": 1, "c_mode": "ppo_c", "r_mode": "strict_v13", "method_name": "m",
            "evaluated_requests": 10, "admitted": 0, "blocked": 10,
            "r_reached": 0, "r_decisions": 0,
            "c_no_valid_action": 10, "r_no_valid_action": 0,
            "failure_reason_distribution": {"c_no_valid_action": 10},
            "ppo_agreement_rate": None, "ppo_agreement_numerator": 0, "ppo_agreement_denominator": 0,
            "request_trace_hash": "a", "selected_action_hash": "b", "config_hash": "c",
            "evaluator_code_hash": "d", "runner_code_hash": "e", "analyzer_code_hash": "f",
            "source_manifest": {}, "checkpoint_sha256s": {},
        }
        ev._validate_result_row(row)


class TestHolm:
    def test_known_example(self):
        pvals = [0.01, 0.04, 0.05]
        adj = ana._holm(pvals)
        expected = [0.03, 0.08, 0.08]
        assert np.allclose(adj, expected), f"got {adj}"

    def test_monotonicity_and_clipping(self):
        pvals = [0.4, 0.4, 0.4]
        adj = ana._holm(pvals)
        assert all(a <= 1.0 for a in adj)
        assert adj[0] >= adj[1] >= adj[2]


class TestExactSignFlip:
    def test_known_direction(self):
        base = np.array([0.10, 0.12, 0.11, 0.13, 0.10])
        strict = np.array([0.05, 0.04, 0.06, 0.05, 0.05])
        res = ana._exact_sign_flip(base, strict)
        assert res["n"] == 5
        assert 0.0 <= res["one_sided_p"] <= 0.05
        assert res["two_sided_p"] >= res["one_sided_p"]

    def test_reverse_direction(self):
        base = np.array([0.05, 0.04, 0.06, 0.05, 0.05])
        strict = np.array([0.10, 0.12, 0.11, 0.13, 0.10])
        res = ana._exact_sign_flip(base, strict)
        assert res["one_sided_p"] >= 0.95


class TestRunnerExpectedHashes:
    def test_resume_validation_detects_config_change(self):
        import tempfile
        task = {
            "topology": "xlron_cost239_ptrnet_real",
            "seed": 3030,
            "c_modes": ["ppo_c", "df_c"],
            "c_checkpoints": ["sa_hmarl/checkpoints/agent_c_cost239_r_feasibility_safe_last.pt", "_heuristic_"],
            "ranker_specs": [
                f"strict_v13={Path(__file__).resolve().parents[2] / 'sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt'}",
                f"old_v13={Path(__file__).resolve().parents[2] / 'sa_hmarl/checkpoints/v13_kpath50_hops_medium_regret_selected/seed_42/ranking_model.pt'}",
            ],
            "arrival_interval": 0.0625,
            "edge_cost_max": 2.2,
            "warmup_requests": 20,
            "requests_per_episode": 50,
            "task_dir": Path(tempfile.mkdtemp()),
        }
        root = Path(__file__).resolve().parents[2]
        manifest = runner._source_manifest(root)
        expected = runner._expected_hashes(
            task, root, task["ranker_specs"], runner.AGENT_R_CKPT,
            "ppo_r_top1,ksp_ff_highest,strict_v13", manifest
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
        (task["task_dir"] / "results.json").write_text(json.dumps(payload))
        (task["task_dir"] / "done.marker").write_text("done")
        assert runner._is_done(task["task_dir"], expected)
        bad = expected.copy()
        bad["config_hash"] = "wrong"
        assert not runner._is_done(task["task_dir"], bad)
