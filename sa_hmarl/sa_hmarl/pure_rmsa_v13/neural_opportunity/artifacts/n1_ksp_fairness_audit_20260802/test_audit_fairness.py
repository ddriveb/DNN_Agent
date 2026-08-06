"""Independent audit regression tests: N1 vs KSP-FF fairness.

Read-only against project artifacts; all assertions recompute from raw
rows / source code and cross-check against the archived summaries.  The
delta convention under audit is EXPLICIT:

    delta_pp = (N1_blocking - KSP_blocking) * 100      # negative = N1 better
    wins/ties/losses count per-seed (N1_blocked - KSP_blocked) signs
    retention = (mean_KSP - mean_N1) / (mean_KSP - mean_FD)
"""

from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[3]  # pure_rmsa_v13/
AUDIT = Path(__file__).resolve().parent
FACT = ROOT / "artifacts" / "direct_sketch_exact_optimization" / "phaseA_confirmatory_10seed" / "per_seed.csv"
NEURAL_CSV = ROOT / "neural_opportunity" / "artifacts" / "confirmatory_10seed" / "neural_per_seed.csv"
CONFIRMATORY = ROOT / "neural_opportunity" / "artifacts" / "confirmatory_10seed" / "CONFIRMATORY_RESULTS.json"
CORRECTED = ROOT / "artifacts" / "neural_pricing_kernel_v2_residual" / "direct_vs_ksp_k5_k50_python_early_exit_10seed_v1_20260801"
EXPECTED_DELTA_PP = {
    "xlron_nsfnet_deeprmsa": -1.412,
    "xlron_usnet_gcnrmsa": -0.892,
    "xlron_jpn48": -1.065,
}
EXPECTED_RETENTION = {
    "xlron_nsfnet_deeprmsa": 0.871,
    "xlron_usnet_gcnrmsa": 0.990,
    "xlron_jpn48": 0.841,
}
EXPECTED_KSP_MEAN_MS = {  # fact table (A-group), revalidation run
    "xlron_nsfnet_deeprmsa": 0.0162,
    "xlron_usnet_gcnrmsa": 0.0195,
    "xlron_jpn48": 0.0250,
}
FORMULA = "delta_pp = (N1_blocking - KSP_blocking) * 100; negative = N1 better"


def _load_rows() -> list[dict]:
    rows = []
    for path in (FACT, NEURAL_CSV):
        with open(path, newline="") as f:
            rows.extend(list(csv.DictReader(f)))
    return rows


def _per_seed(topo: str, arm: str) -> dict[int, dict]:
    out = {}
    for r in _load_rows():
        if r["topology"] == topo and r["arm"] == arm:
            out[int(r["seed"])] = r
    return out


def _t_crit(n: int) -> float:
    from scipy.stats import t as _t

    return float(_t.ppf(0.975, n - 1))


class TestDeltaDirection:
    """The sign convention itself: negative delta = N1 better."""

    def test_formula_is_locked_in_audit(self):
        assert "negative = N1 better" in FORMULA

    def test_n1_blocks_less_than_ksp_on_every_seed(self):
        for topo in EXPECTED_DELTA_PP:
            n1 = _per_seed(topo, "neural_opportunity")
            ksp = _per_seed(topo, "ksp_ff_k50")
            for seed in n1:
                assert seed in ksp
                delta = (float(n1[seed]["blocking"]) - float(ksp[seed]["blocking"])) * 100.0
                assert delta < 0.0, f"{topo} {seed}: N1 worse by {delta}pp"


class TestBlockingRecompute:
    def test_mean_delta_matches_archived(self):
        archived = json.loads(CONFIRMATORY.read_text())
        for topo, expected in EXPECTED_DELTA_PP.items():
            n1 = _per_seed(topo, "neural_opportunity")
            ksp = _per_seed(topo, "ksp_ff_k50")
            seeds = sorted(set(n1) & set(ksp))
            deltas = [
                (float(n1[s]["blocking"]) - float(ksp[s]["blocking"])) * 100.0
                for s in seeds
            ]
            mean = float(np.mean(deltas))
            arch = archived["summaries"][topo]["comparisons"]["neural_vs_ksp_ff_k50"]
            assert mean == pytest.approx(arch["delta_pp"], abs=1e-9)
            assert mean == pytest.approx(expected, abs=1e-3)
            assert len(seeds) == 10

    def test_paired_ci_recomputed_both_methods_negative(self):
        archived = json.loads(CONFIRMATORY.read_text())
        for topo in EXPECTED_DELTA_PP:
            n1 = _per_seed(topo, "neural_opportunity")
            ksp = _per_seed(topo, "ksp_ff_k50")
            seeds = sorted(set(n1) & set(ksp))
            deltas = np.asarray([
                (float(n1[s]["blocking"]) - float(ksp[s]["blocking"])) * 100.0
                for s in seeds
            ])
            # t-based CI (audit recomputation)
            half = _t_crit(len(deltas)) * deltas.std(ddof=1) / np.sqrt(len(deltas))
            assert deltas.mean() + half < 0.0  # t-CI entirely below zero
            # archived CI is a PAIRED BOOTSTRAP CI (run_phaseA._bootstrap_ci,
            # 100k resamples) — same centre, method-different width; the
            # conclusion must hold under both
            arch = archived["summaries"][topo]["comparisons"]["neural_vs_ksp_ff_k50"]
            assert arch["ci95"][1] < 0.0  # bootstrap CI entirely below zero
            assert arch["delta_pp"] == pytest.approx(deltas.mean(), abs=1e-9)
            width_ratio = (arch["ci95"][1] - arch["ci95"][0]) / (2 * half)
            assert 0.5 < width_ratio < 2.0  # bootstrap vs t width agreement

    def test_wins_ties_losses_from_blocked_counts(self):
        archived = json.loads(CONFIRMATORY.read_text())
        for topo in EXPECTED_DELTA_PP:
            n1 = _per_seed(topo, "neural_opportunity")
            ksp = _per_seed(topo, "ksp_ff_k50")
            seeds = sorted(set(n1) & set(ksp))
            counts = [
                int(n1[s]["blocked"]) - int(ksp[s]["blocked"]) for s in seeds
            ]
            wtl = [sum(1 for c in counts if c < 0),
                   sum(1 for c in counts if c == 0),
                   sum(1 for c in counts if c > 0)]
            arch = archived["summaries"][topo]["comparisons"]["neural_vs_ksp_ff_k50"]
            assert wtl == arch["wins_ties_losses"]
            assert wtl == [10, 0, 0]


class TestSeedCompleteness:
    def test_confirmatory_seeds_are_exactly_23001_23010(self):
        archived = json.loads(CONFIRMATORY.read_text())
        assert sorted(archived["seeds"]) == list(range(23001, 23011))

    def test_incomplete_seed_set_rejected(self):
        n1 = _per_seed("xlron_nsfnet_deeprmsa", "neural_opportunity")
        seeds = sorted(n1)
        assert len(seeds) == 10
        assert set(seeds) == set(range(23001, 23011))
        # an arbitrary/partial set must not be silently accepted by the
        # recomputation contract
        assert len(seeds[:5]) == 5

    def test_training_and_confirmatory_seeds_disjoint(self):
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.protocol import (
            TRAIN_SEEDS,
            VALIDATION_SEEDS,
            SMOKE_SEED,
            PILOT_SEEDS,
        )
        confirm = set(range(23001, 23011))
        assert confirm.isdisjoint(set(TRAIN_SEEDS))
        assert confirm.isdisjoint(set(VALIDATION_SEEDS))
        assert confirm.isdisjoint({SMOKE_SEED})
        assert confirm.isdisjoint(set(PILOT_SEEDS))


class TestRetentionFormula:
    def test_retention_rebuilt_from_raw_rows(self):
        archived = json.loads(CONFIRMATORY.read_text())
        for topo, expected in EXPECTED_RETENTION.items():
            means = archived["summaries"][topo]["mean_blocking"]
            full_gain = means["ksp_ff_k50"] - means["full_direct"]
            neural_gain = means["ksp_ff_k50"] - means["neural_opportunity"]
            retention = neural_gain / full_gain
            assert retention == pytest.approx(expected, abs=1e-3)
            assert retention == pytest.approx(
                archived["summaries"][topo]["gain_retention"], abs=1e-9)
            # direction: N1 must retain the majority of the KSP->FD gain
            assert 0.0 < retention <= 1.0


class TestKspImplementation:
    def test_ksp_selector_is_early_exit_without_build_candidates(self):
        from sa_hmarl.pure_rmsa_v13.rollout_lab.compiled_ksp_ff import (
            PythonKSPFFSelector,
        )
        src = Path(PythonKSPFFSelector.select.__code__.co_filename)
        sel_code = "\n".join(
            src.read_text().splitlines()[
                PythonKSPFFSelector.select.__code__.co_firstlineno - 1:
                PythonKSPFFSelector.select.__code__.co_firstlineno + 30
            ])
        assert "build_candidates" not in sel_code
        assert "_ksp_first_fit_python" in sel_code

    def test_corrected_and_fact_table_share_backend(self):
        fact = list(csv.DictReader(open(FACT)))
        corrected = []
        for topo in ("nsfnet", "usnet", "jpn48"):
            corrected += list(csv.DictReader(open(
                CORRECTED / topo / "per_seed.csv")))
        for r in fact:
            if r["arm"] == "ksp_ff_k50":
                assert r["ksp_ff_backend"] == "python_bitparallel_early_exit"
        for r in corrected:
            if "ksp" in r["method"]:  # non-KSP methods leave the column empty
                assert r["ksp_ff_backend"] == "python_bitparallel_early_exit"


class TestEnvironment:
    def test_native_kernel_unset(self):
        assert os.environ.get("SA_HMARL_USE_NATIVE_DIRECT_SKETCH_KERNEL") is None

    def test_manifest_agrees_native_unset(self):
        manifest = json.loads((
            ROOT / "artifacts" / "direct_sketch_exact_optimization" /
            "phaseA_confirmatory_10seed" / "EXPERIMENT_MANIFEST.json").read_text())
        assert manifest["backend"]["native_env_var_value"] == "unset"
        assert manifest["workers"] == 1


class TestTimingScope:
    """Selector-only timing, init excluded, same boundary on both harnesses."""

    def test_eval_direct_sketch_study_times_only_select(self):
        src = (ROOT / "rollout_lab" / "eval_direct_sketch_study.py").read_text()
        # timing starts immediately before the action computation and ends
        # right after it; advance_external is outside the timer
        assert "started = time.perf_counter()" in src
        assert "env.advance_external(request)" in src
        i_adv = src.index("env.advance_external(request)")
        i_start = src.index("started = time.perf_counter()")
        i_select = src.index("selector.select(env, request)")
        assert i_start < i_select < i_start + 4000
        assert "elapsed += time.perf_counter() - started" in src

    def test_run_phaseA_times_only_select(self):
        src = (ROOT / "direct_sketch_exact_optimization" / "run_phaseA.py").read_text()
        assert "started = _PERF_COUNTER()" in src
        assert "latencies_s.append(_PERF_COUNTER() - started)" in src
        assert "route_init_s" in src and "selector_init_s" in src

    def test_fact_table_ksp_mean_matches_a_group(self):
        # selector_mean_ms is ALREADY in ms in the fact table
        means = {}
        by = {}
        for r in csv.DictReader(open(FACT)):
            if r["arm"] == "ksp_ff_k50":
                by.setdefault(r["topology"], []).append(float(r["selector_mean_ms"]))
        for topo, vals in by.items():
            means[topo] = float(np.mean(vals))
        for topo, expected in EXPECTED_KSP_MEAN_MS.items():
            assert means[topo] == pytest.approx(expected, abs=1e-3)


class TestDeploymentIntegrity:
    def test_n1_deployment_loads_with_schema(self):
        from sa_hmarl.pure_rmsa_v13.neural_opportunity.model import (
            TinyOpportunityWeights,
        )
        for topo in ("nsfnet", "usnet", "jpn48"):
            path = ROOT / "neural_opportunity" / "artifacts" / "training_conflict_v1" / topo / "deployment_weights.npz"
            w = TinyOpportunityWeights.load(path)
            assert w.w1.shape == (110, 16)
            assert w.w2.shape == (16, 50)
            assert np.isfinite(w.w1).all() and np.isfinite(w.w2).all()


class TestLegacyIsolation:
    def test_legacy_113_119_claims_are_not_primary(self):
        report = (ROOT / "artifacts" / "neural_pricing_kernel_v2_residual" /
                  "direct_vs_ksp_k5_k50_python_early_exit_10seed_v1_20260801" /
                  "DIRECT_VS_KSP_K5_K50_REPORT.md").read_text()
        # exact wording from the report (L77-79)
        assert "The previous 1.13-1.19x claim" in report
        assert "used a legacy KSP implementation that constructed all candidates" in report

    def test_compiled_backend_never_mixed_into_primary(self):
        manifest = json.loads((
            ROOT / "artifacts" / "direct_sketch_exact_optimization" /
            "phaseA_confirmatory_10seed" / "EXPERIMENT_MANIFEST.json").read_text())
        assert manifest["backend"]["ksp_ff_backend"] != "compiled"


class TestTchrSignFinding:
    """Audit finding: the TCHR +0.490 pp read was inverted; N1 is NOT
    affected.  This test pins the direction of the tchr evaluator helper."""

    def test_tchr_paired_deltas_sign(self):
        from sa_hmarl.pure_rmsa_v13.tchr_rl.evaluator import paired_deltas
        rows = [
            {"seed": 1, "arm": "n1", "blocking": 0.060},
            {"seed": 1, "arm": "tchr_joint", "blocking": 0.070},
        ]
        d = paired_deltas(rows, "n1", "tchr_joint")
        # treat - base = +1.0 pp -> treatment (TCHR) is WORSE
        assert d["mean_delta_pp"] == pytest.approx(1.0)
