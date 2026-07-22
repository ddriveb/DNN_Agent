"""Unit tests for strict v1.3 multi-topology C-side evaluator hard gating."""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Dict
from unittest.mock import patch

import numpy as np
import pytest

from sa_hmarl.evaluation import eval_strict_v13_multitopology_cside_fair as ev
from sa_hmarl.evaluation import analyze_strict_v13_multitopology_cside as ana


@dataclass
class _FakeEnv:
    net: Any
    mec: Any
    mod_reg: Any
    max_blocks: int = 10


def _make_obs_c(mask_values):
    return {"agent_c_mask": np.asarray(mask_values, dtype=bool)}


def _make_obs_r(mask_values, num_mods=3):
    return {
        "agent_r_mask": np.asarray(mask_values, dtype=bool),
        "mod_names": ["QPSK", "8QAM", "16QAM"][:num_mods],
        "path_features": [{"path_length_km": float(i * 100), "hop_count": float(i + 1)} for i in range(len(mask_values))],
        "candidate_blocks_per_path_mod": [[[ (j, j + 2) for j in range(5) ] for _ in range(num_mods)] for _ in range(len(mask_values))],
        "required_fs_per_path_mod": [[2] * num_mods for _ in range(len(mask_values))],
    }


class TestSelectCAction:
    def test_baseline_returns_none_when_mask_empty(self):
        env = SimpleNamespace(net=SimpleNamespace(num_slots=320), mec=SimpleNamespace(servers=[0, 1, 2, 3]))
        req = object()
        obs_c = _make_obs_c([])
        with patch.object(ev, "select_offloading_action", return_value=0) as mock_select:
            flat, split, server, valid, info = ev._select_c_action("df_c", None, env, req, obs_c, 4)
        assert not valid
        assert flat is None
        assert info["raw_mask_empty"] is True
        assert info["effective_mask_empty"] is True
        assert info["action_in_effective_mask"] is False
        mock_select.assert_not_called()

    def test_baseline_uses_mask_and_decodes(self):
        env = SimpleNamespace(net=SimpleNamespace(num_slots=320), mec=SimpleNamespace(servers=[0, 1, 2, 3]))
        req = object()
        obs_c = _make_obs_c([True] * 12)
        with patch.object(ev, "select_offloading_action", return_value=7) as mock_select:
            flat, split, server, valid, info = ev._select_c_action("df_c", None, env, req, obs_c, 4)
        assert valid
        assert flat == 7
        assert split == 7 // 4
        assert server == 7 % 4
        assert info["action_in_effective_mask"] is True
        mock_select.assert_called_once()

    def test_ppo_c_returns_none_when_risk_mask_empty(self):
        env = SimpleNamespace(net=SimpleNamespace(num_slots=320), mec=SimpleNamespace(servers=[0, 1, 2, 3]))
        req = object()
        obs_c = _make_obs_c([True] * 12)
        fake_agent = object()
        with patch.object(ev, "_select_c_action_from_obs", return_value=(0, np.ones(12), np.zeros(12))) as mock:
            flat, split, server, valid, info = ev._select_c_action("ppo_c", fake_agent, env, req, obs_c, 4)
        assert not valid
        assert flat is None
        assert info["raw_mask_empty"] is False
        assert info["effective_mask_empty"] is True
        assert info["action_in_effective_mask"] is False
        mock.assert_called_once()

    def test_ppo_c_decodes_when_risk_mask_nonempty(self):
        env = SimpleNamespace(net=SimpleNamespace(num_slots=320), mec=SimpleNamespace(servers=[0, 1, 2, 3]))
        req = object()
        obs_c = _make_obs_c([True] * 12)
        fake_agent = object()
        risk_mask = np.zeros(12)
        risk_mask[5] = 1
        with patch.object(ev, "_select_c_action_from_obs", return_value=(5, np.ones(12), risk_mask)) as mock:
            flat, split, server, valid, info = ev._select_c_action("ppo_c", fake_agent, env, req, obs_c, 4)
        assert valid
        assert flat == 5
        assert split == 5 // 4
        assert server == 5 % 4
        assert info["action_in_effective_mask"] is True

    def test_ppo_c_action_not_in_effective_mask_is_invalid(self):
        env = SimpleNamespace(net=SimpleNamespace(num_slots=320), mec=SimpleNamespace(servers=[0, 1, 2, 3]))
        req = object()
        obs_c = _make_obs_c([True] * 12)
        fake_agent = object()
        risk_mask = np.zeros(12)
        risk_mask[5] = 1
        with patch.object(ev, "_select_c_action_from_obs", return_value=(3, np.ones(12), risk_mask)) as mock:
            flat, split, server, valid, info = ev._select_c_action("ppo_c", fake_agent, env, req, obs_c, 4)
        assert not valid
        assert info["action_in_effective_mask"] is False


class TestSelectRAction:
    def test_ksp_invalid_when_mask_empty(self):
        obs_r = _make_obs_r([])
        idx, valid, info, timing = ev._select_r_action("ksp_ff_highest", None, None, None, None, None, obs_r, 0, 0, 0)
        assert not valid
        assert idx == 0
        assert info is None
        assert timing["ranker_or_ksp_ms"] == 0.0

    def test_ksp_invalid_when_ksp_returns_none(self):
        obs_r = _make_obs_r([True] * 6)
        with patch.object(ev, "ksp_ff_highest_mod_action", return_value=None) as mock:
            idx, valid, info, timing = ev._select_r_action("ksp_ff_highest", None, None, None, None, None, obs_r, 0, 0, 0)
        assert not valid
        assert idx == 0
        assert info is None
        assert timing["ranker_or_ksp_ms"] >= 0.0
        mock.assert_called_once_with(obs_r)

    def test_ppo_r_top1_returns_index(self):
        idx, valid, info, timing = ev._select_r_action("ppo_r_top1", None, None, None, None, None, None, 42, 0, 0)
        assert valid
        assert idx == 42
        assert info is None
        assert timing == {"proposer_ms": 0.0, "ranker_or_ksp_ms": 0.0}

    def test_strict_fallback_when_no_candidates(self):
        obs_r = _make_obs_r([True] * 6)
        fake_agent = SimpleNamespace()
        with patch.object(ev, "_ppo_r_topk_actions", return_value=np.array([], dtype=np.int64)) as mock:
            idx, valid, info, timing = ev._select_r_action("strict_v13", fake_agent, None, None, None, None, obs_r, 7, 1, 2)
        # Fallback uses PPO-R top-1, which is a legal decision.
        assert valid
        assert idx == 7
        assert info["fallback"] is True
        assert info["candidate_count"] == 0
        assert timing["proposer_ms"] >= 0.0


class TestRecordFailure:
    def test_server_failures_merged(self):
        res = ev.MethodResult("topo", 1, "ppo_c", "ksp_ff_highest", "test")
        ev._record_failure(res, "server_saturated")
        ev._record_failure(res, "server_overload")
        assert res.server_overload == 2
        assert res.failure_reason_counts["server_saturated"] == 1
        assert res.failure_reason_counts["server_overload"] == 1
        assert res.no_suitable_block == 0
        assert res.deadline_failure == 0
        assert res.other_failure == 0

    def test_no_suitable_block(self):
        res = ev.MethodResult("topo", 1, "ppo_c", "ksp_ff_highest", "test")
        ev._record_failure(res, "no_suitable_block")
        assert res.server_overload == 0
        assert res.no_suitable_block == 1
        assert res.deadline_failure == 0
        assert res.other_failure == 0

    def test_deadline_failure(self):
        res = ev.MethodResult("topo", 1, "ppo_c", "ksp_ff_highest", "test")
        ev._record_failure(res, "deadline_infeasible")
        assert res.server_overload == 0
        assert res.no_suitable_block == 0
        assert res.deadline_failure == 1
        assert res.other_failure == 0

    def test_r_no_valid_action_not_other(self):
        res = ev.MethodResult("topo", 1, "ppo_c", "ksp_ff_highest", "test")
        ev._record_failure(res, "r_no_valid_action")
        assert res.server_overload == 0
        assert res.no_suitable_block == 0
        assert res.deadline_failure == 0
        assert res.other_failure == 0
        assert res.failure_reason_counts["r_no_valid_action"] == 1

    def test_other_reasons(self):
        res = ev.MethodResult("topo", 1, "ppo_c", "ksp_ff_highest", "test")
        ev._record_failure(res, "modulation_reach")
        assert res.server_overload == 0
        assert res.no_suitable_block == 0
        assert res.deadline_failure == 0
        assert res.other_failure == 1


class TestValidateResultRow:
    def _base_row(self):
        return {
            "schema_version": ev.SCHEMA_VERSION,
            "topology": "t", "seed": 1, "c_mode": "ppo_c", "r_mode": "ksp_ff_highest", "method_name": "m",
            "total": 100, "admitted": 85, "blocked": 15,
            "r_reached": 100, "r_decisions": 100,
            "c_no_valid_action": 0, "r_no_valid_action": 0,
            "c_raw_mask_empty": 0, "c_effective_mask_empty": 0,
            "c_action_in_effective_mask": 100, "r_mask_empty": 0,
            "deadline_failure": 0,
            "failure_reason_distribution": {"no_suitable_block": 10, "server_overload": 5},
            "ppo_agreement_rate": 0.6,
            "ppo_agreement_numerator": 60,
            "ppo_agreement_denominator": 100,
            "server_overload": 5, "no_suitable_block": 10, "other_failure": 0,
        }

    def test_conservation_passes(self):
        row = self._base_row()
        row["path_idx_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 85, 85)
        row["block_start_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 85, 85)
        row["required_fs_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([2] * 85, 85)
        row["hop_count_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([1.0] * 85, 85)
        row["path_length_km_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0.0] * 85, 85)
        row["selected_path_idx_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 100, 100)
        row["selected_mod_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution(["QPSK"] * 100, 100)
        row["selected_block_start_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 100, 100)
        row["selected_required_fs_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([2] * 100, 100)
        row["selected_hop_count_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([1.0] * 100, 100)
        row["selected_path_length_km_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0.0] * 100, 100)
        row["selected_split_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 100, 100)
        row["selected_server_distribution"] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 100, 100)
        ev._validate_result_row(row)

    def test_ppo_r_top1_agreement_must_be_one(self):
        row = self._base_row()
        row["r_mode"] = "ppo_r_top1"
        row["ppo_agreement_rate"] = 1.0
        row["ppo_agreement_numerator"] = 100
        row["failure_reason_distribution"] = {"server_overload": 15}
        row["server_overload"] = 15
        row["no_suitable_block"] = 0
        row["other_failure"] = 0
        for key in [
            "path_idx_distribution", "block_start_distribution", "required_fs_distribution",
            "hop_count_distribution", "path_length_km_distribution",
        ]:
            row[key] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 85, 85)
        for key in [
            "selected_path_idx_distribution", "selected_mod_distribution",
            "selected_block_start_distribution", "selected_required_fs_distribution",
            "selected_hop_count_distribution", "selected_path_length_km_distribution",
            "selected_split_distribution", "selected_server_distribution",
        ]:
            row[key] = ev.MethodResult("", 0, "", "", "")._distribution([0] * 100, 100)
        ev._validate_result_row(row)

        row2 = row.copy()
        row2["ppo_agreement_rate"] = 0.99
        with pytest.raises(AssertionError):
            ev._validate_result_row(row2)

    def test_conservation_fails(self):
        row = self._base_row()
        row["total"] = 100
        row["admitted"] = 80
        row["blocked"] = 15
        with pytest.raises(AssertionError):
            ev._validate_result_row(row)


class TestEDiagnostics:
    def test_e_stratum(self):
        assert ev._e_stratum(0, 3) == "E=0"
        assert ev._e_stratum(1, 3) == "E=0"
        assert ev._e_stratum(2, 3) == "E=1"
        assert ev._e_stratum(0, 1) == "E=0"

    def test_compute_e_diagnostics(self):
        actions = [
            {"r_valid": True, "admitted": True, "r_idx": 0, "ppo_r_idx": 0,
             "path_idx": 0, "mod_name": "QPSK", "block_start": 0, "required_fs": 2,
             "hop_count": 1.0, "path_length_km": 100.0, "split_id": 0, "server_id": 0, "e_stratum": "E=0"},
            {"r_valid": True, "admitted": False, "r_idx": 1, "ppo_r_idx": 0,
             "path_idx": 1, "mod_name": "8QAM", "block_start": 1, "required_fs": 3,
             "hop_count": 2.0, "path_length_km": 200.0, "split_id": 2, "server_id": 1, "e_stratum": "E=1"},
        ]
        diag = ev._compute_e_diagnostics(actions)
        assert diag["E=0"]["total"] == 1
        assert diag["E=0"]["admitted"] == 1
        assert diag["E=1"]["total"] == 1
        assert diag["E=1"]["blocked"] == 1
        assert diag["E=1"]["ppo_r_agreement_rate"] == 0.0


class TestSourceManifest:
    def test_manifest_contains_key_files(self):
        root = ev.Path(__file__).resolve().parents[2]
        manifest = ev._source_manifest(root)
        for rel in [
            "sa_hmarl/sa_hmarl/evaluation/eval_strict_v13_multitopology_cside_fair.py",
            "sa_hmarl/sa_hmarl/network/ksp.py",
            "sa_hmarl/sa_hmarl/env/action_mask.py",
        ]:
            assert rel in manifest
            assert len(manifest[rel]) == 64 or manifest[rel] == "N/A"


class TestAnalyzeUtilities:
    def test_holm_known_input(self):
        pvals = [0.01, 0.03, 0.06]
        adj = ana._holm(pvals)
        expected = [0.03, 0.06, 0.06]
        assert np.allclose(adj, expected), f"got {adj}"

    def test_holm_monotonicity_and_clipping(self):
        pvals = [0.4, 0.4, 0.4]
        adj = ana._holm(pvals)
        assert all(a <= 1.0 for a in adj)
        assert adj[0] >= adj[1] >= adj[2]

    def test_permutation_direction(self):
        base = np.array([0.10, 0.12, 0.11, 0.13, 0.10])
        strict = np.array([0.05, 0.04, 0.06, 0.05, 0.05])
        p = ana._permutation_test(base, strict, n_perm=5000, seed=42)
        assert 0.0 <= p <= 0.05

        p_rev = ana._permutation_test(strict, base, n_perm=5000, seed=42)
        assert p_rev >= 0.95
