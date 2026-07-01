"""Unit + smoke tests for the v1.2 residual blocking floor diagnostic."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from sa_hmarl.evaluation.diagnose_v12_residual_blocking_floor import (
    BlockedRecord,
    EpisodeResult,
    _classify_from_outcomes,
    _aggregate_results,
)


ROOT = Path(__file__).resolve().parents[2]


def _make_blocked_record(category: str, successful_c: int = 0, successful_r_orig: int = 0) -> BlockedRecord:
    return BlockedRecord(
        seed=1, episode=0, request_index=0, req_id=0,
        selected_split=0, selected_server=0, selected_r_idx=0,
        category=category, reason="other",
        total_c_actions=12, total_r_actions_evaluated=50,
        successful_c_actions=successful_c,
        successful_r_actions_under_original_c=successful_r_orig,
    )


def test_classify_unavoidable_no_future():
    success = np.zeros((3, 4), dtype=bool)
    cat, *_ = _classify_from_outcomes(0, 0, success)
    assert cat == "unavoidable_block"


def test_classify_r_avoidable():
    success = np.zeros((3, 4), dtype=bool)
    success[0, 2] = True  # original C has a successful R action
    cat, *_ = _classify_from_outcomes(0, 0, success)
    assert cat == "R_avoidable"


def test_classify_c_avoidable():
    success = np.zeros((3, 4), dtype=bool)
    success[2, 1] = True  # a different C succeeds, original C row all False
    cat, *_ = _classify_from_outcomes(0, 0, success)
    assert cat == "C_avoidable"


def test_classify_trajectory_avoidable():
    success = np.zeros((3, 4), dtype=bool)
    future = np.array([[5, 4, 6, 7], [3, 3, 3, 3], [8, 8, 8, 8]], dtype=int)
    cat, chosen, best, rng = _classify_from_outcomes(0, 0, success, future)
    assert cat == "trajectory_avoidable"
    assert chosen == 5
    assert best == 3
    assert rng == 2


def test_classify_unavoidable_with_future():
    success = np.zeros((3, 4), dtype=bool)
    future = np.full((3, 4), 4, dtype=int)
    cat, chosen, best, rng = _classify_from_outcomes(0, 0, success, future)
    assert cat == "unavoidable_block"
    assert chosen == 4
    assert best == 4
    assert rng == 0


def test_aggregate_categories_sum_to_blocked():
    records = [
        _make_blocked_record("unavoidable_block"),
        _make_blocked_record("unavoidable_block"),
        _make_blocked_record("R_avoidable"),
        _make_blocked_record("C_avoidable"),
        _make_blocked_record("trajectory_avoidable"),
    ]
    episode_results = [
        EpisodeResult(
            seed=1, episode=0, total_requests=10,
            blocked_count=len(records), raw_mask_empty_count=0,
            blocked_records=records,
        )
    ]
    summary = _aggregate_results(episode_results)
    total_cats = sum(summary["category_counts"].values())
    assert total_cats == len(records)
    assert summary["category_counts"]["unavoidable_block"] == 2
    assert summary["category_counts"]["R_avoidable"] == 1
    assert summary["category_counts"]["C_avoidable"] == 1
    assert summary["category_counts"]["trajectory_avoidable"] == 1


def test_smoke_run():
    """Run the diagnostic as a script on a tiny setting and verify outputs."""
    output_json = ROOT / "sa_hmarl" / "experiments" / "v12_residual_blocking_floor_smoke_test.json"
    output_md = ROOT / "sa_hmarl" / "experiments" / "v12_residual_blocking_floor_smoke_test.md"
    cmd = [
        sys.executable,
        "-m", "sa_hmarl.evaluation.diagnose_v12_residual_blocking_floor",
        "--seeds", "3030",
        "--episodes", "1",
        "--requests_per_episode", "20",
        "--horizon", "3",
        "--output_json", str(output_json),
        "--output_md", str(output_md),
    ]
    env = {"PYTHONPATH": str(ROOT / "sa_hmarl")}
    env.update({k: v for k, v in subprocess.os.environ.items() if k != "PYTHONPATH"})
    result = subprocess.run(cmd, cwd=ROOT, env=env, check=True, capture_output=True, text=True)
    assert output_json.exists()
    assert output_md.exists()
    data = json.loads(output_json.read_text(encoding="utf-8"))
    required_keys = {
        "total_requests", "total_blocked", "blocking_rate",
        "category_counts", "category_rates", "reason_counts",
        "seed_summary", "episode_results",
    }
    assert required_keys.issubset(data.keys())
    assert sum(data["category_counts"].values()) == data["total_blocked"]
    # Clean up.
    output_json.unlink(missing_ok=True)
    output_md.unlink(missing_ok=True)
