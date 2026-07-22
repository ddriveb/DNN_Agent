"""Strict v1.3 multi-topology fair closed-loop evaluation pipeline (verified).

Phases:
1. Micro E2E (COST239 only, seed 3030, tiny workload) -- gate.
2. Smoke (4 topologies, seed 3030) -- writes smoke_validated.json.
3. Formal (4 topologies x 5 seeds, full workload) -- requires matching smoke marker.
4. Training-seed robustness: seed43 and seed44 (strict_v13 only under ppo_c/df_c).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow importing the sa_hmarl package.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sa_hmarl"))

from sa_hmarl.evaluation.run_strict_v13_multitopology_cside import run_experiment


RANKER_SEED42 = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt"
RANKER_SEED43 = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_43/ranking_model.pt"
RANKER_SEED44 = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_44/ranking_model.pt"
BASE = "sa_hmarl/experiments/v13_strict_multitopology_cside_verified"

MAIN_C_MODES = "ppo_c,df_c"
MAIN_R_MODES = "ppo_r_top1,ksp_ff_highest,strict_v13"
TOPOLOGIES = [
    "xlron_cost239_ptrnet_real",
    "xlron_german17",
    "xlron_nsfnet_deeprmsa",
    "xlron_jpn48",
]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_workers", type=int, default=6, help="Worker processes for each phase")
    args = parser.parse_args()

    # 1. Micro E2E (COST239 only, tiny workload, main matrix).
    print("=" * 60)
    print("[pipeline] Phase 1/4: micro E2E (COST239 only)")
    print("=" * 60)
    rc = run_experiment(
        smoke=False,
        max_workers=1,
        ranker_ckpt=RANKER_SEED42,
        output_dir=BASE,
        r_modes=MAIN_R_MODES,
        c_modes_override=MAIN_C_MODES,
        topologies=["xlron_cost239_ptrnet_real"],
        seeds=[3030],
        warmup_requests=5,
        requests_per_episode=20,
        require_smoke_marker=False,
    )
    if rc != 0:
        print("[pipeline] MICRO E2E FAILED -- stopping.")
        return rc
    print("[pipeline] Micro E2E passed.\n")

    # 2. Smoke test (4 topologies, single seed, small eval).
    print("=" * 60)
    print("[pipeline] Phase 2/4: smoke test")
    print("=" * 60)
    rc = run_experiment(
        smoke=True,
        max_workers=args.max_workers,
        ranker_ckpt=RANKER_SEED42,
        output_dir=BASE,
        r_modes=MAIN_R_MODES,
        c_modes_override=MAIN_C_MODES,
        require_smoke_marker=False,
    )
    if rc != 0:
        print("[pipeline] SMOKE FAILED -- stopping.")
        return rc
    print("[pipeline] Smoke passed and marker validated.\n")

    # 3. Formal evaluation (4 topologies x 5 seeds, full workload).
    print("=" * 60)
    print("[pipeline] Phase 3/4: formal evaluation")
    print("=" * 60)
    rc = run_experiment(
        smoke=False,
        max_workers=args.max_workers,
        ranker_ckpt=RANKER_SEED42,
        output_dir=BASE,
        r_modes=MAIN_R_MODES,
        c_modes_override=MAIN_C_MODES,
        topologies=TOPOLOGIES,
        seeds=[3030, 4040, 5050, 6060, 7070],
        warmup_requests=1000,
        requests_per_episode=5000,
        require_smoke_marker=True,
    )
    if rc != 0:
        print("[pipeline] FORMAL FAILED -- stopping.")
        return rc
    print("[pipeline] Formal passed.\n")

    # 4a. Training-seed robustness: seed43, only strict_v13 under ppo_c/df_c.
    print("=" * 60)
    print("[pipeline] Phase 4a/4: seed43 robustness (strict_v13 only)")
    print("=" * 60)
    rc = run_experiment(
        smoke=False,
        max_workers=args.max_workers,
        ranker_ckpt=RANKER_SEED43,
        output_dir=f"{BASE}_seed43",
        r_modes="strict_v13",
        c_modes_override="ppo_c,df_c",
        topologies=TOPOLOGIES,
        seeds=[3030, 4040, 5050, 6060, 7070],
        warmup_requests=1000,
        requests_per_episode=5000,
        require_smoke_marker=False,
    )
    if rc != 0:
        print("[pipeline] SEED43 ROBUSTNESS FAILED -- stopping.")
        return rc
    print("[pipeline] Seed43 robustness passed.\n")

    # 4b. Training-seed robustness: seed44.
    print("=" * 60)
    print("[pipeline] Phase 4b/4: seed44 robustness (strict_v13 only)")
    print("=" * 60)
    rc = run_experiment(
        smoke=False,
        max_workers=args.max_workers,
        ranker_ckpt=RANKER_SEED44,
        output_dir=f"{BASE}_seed44",
        r_modes="strict_v13",
        c_modes_override="ppo_c,df_c",
        topologies=TOPOLOGIES,
        seeds=[3030, 4040, 5050, 6060, 7070],
        warmup_requests=1000,
        requests_per_episode=5000,
        require_smoke_marker=False,
    )
    if rc != 0:
        print("[pipeline] SEED44 ROBUSTNESS FAILED -- stopping.")
        return rc
    print("[pipeline] Seed44 robustness passed.\n")

    print("[pipeline] ALL PHASES COMPLETED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
