"""Verified pipeline: micro E2E -> smoke -> full -> robustness seed43/44."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "sa_hmarl"))

from sa_hmarl.evaluation.run_strict_v13_multitopology_cside_verified import run_experiment


RANKER_SEED42 = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_42/ranking_model.pt"
RANKER_SEED43 = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_43/ranking_model.pt"
RANKER_SEED44 = "sa_hmarl/experiments/v13_strict_fixed/strict_loss_rerun/checkpoints/full_state_uniform_strict/seed_44/ranking_model.pt"
BASE = "sa_hmarl/experiments/v13_strict_multitopology_cside_verified"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max_workers", type=int, default=6)
    parser.add_argument("--skip_micro", action="store_true")
    parser.add_argument("--skip_smoke", action="store_true")
    parser.add_argument("--skip_full", action="store_true")
    parser.add_argument("--skip_robustness", action="store_true")
    args = parser.parse_args()

    # Phase 1: micro E2E on COST239, max_workers=1.
    if not args.skip_micro:
        print("=" * 60)
        print("[pipeline] Phase 1/5: micro E2E (COST239, seed 3030)")
        print("=" * 60)
        rc = run_experiment(
            phase="micro",
            max_workers=1,
            topologies=["xlron_cost239_ptrnet_real"],
            seeds=[3030],
            c_modes=["ppo_c", "df_c"],
            r_modes="ppo_r_top1,ksp_ff_highest,strict_v13",
            base_dir=Path(BASE),
            ranker_ckpt=RANKER_SEED42,
        )
        if rc != 0:
            print("[pipeline] MICRO E2E FAILED -- stopping.")
            return rc
        print("[pipeline] Micro E2E passed.\n")

    # Phase 2: smoke.
    if not args.skip_smoke:
        print("=" * 60)
        print("[pipeline] Phase 2/5: smoke")
        print("=" * 60)
        rc = run_experiment(
            phase="smoke",
            max_workers=args.max_workers,
            c_modes=["ppo_c", "df_c"],
            r_modes="ppo_r_top1,ksp_ff_highest,strict_v13",
            base_dir=Path(BASE),
            ranker_ckpt=RANKER_SEED42,
        )
        if rc != 0:
            print("[pipeline] SMOKE FAILED -- stopping.")
            return rc
        print("[pipeline] Smoke passed.\n")

    # Phase 3: full seed42.
    if not args.skip_full:
        print("=" * 60)
        print("[pipeline] Phase 3/5: seed42 full")
        print("=" * 60)
        rc = run_experiment(
            phase="full",
            max_workers=args.max_workers,
            c_modes=["ppo_c", "df_c"],
            r_modes="ppo_r_top1,ksp_ff_highest,strict_v13",
            base_dir=Path(BASE),
            ranker_ckpt=RANKER_SEED42,
        )
        if rc != 0:
            print("[pipeline] SEED42 FULL FAILED -- stopping.")
            return rc
        print("[pipeline] Seed42 full passed.\n")

    # Phase 4: seed43 robustness (strict only, main C modes).
    if not args.skip_robustness:
        print("=" * 60)
        print("[pipeline] Phase 4/5: seed43 robustness")
        print("=" * 60)
        rc = run_experiment(
            phase="full",
            max_workers=args.max_workers,
            c_modes=["ppo_c", "df_c"],
            r_modes="strict_v13",
            base_dir=Path(f"{BASE}_seed43"),
            ranker_ckpt=RANKER_SEED43,
        )
        if rc != 0:
            print("[pipeline] SEED43 ROBUSTNESS FAILED -- stopping.")
            return rc
        print("[pipeline] Seed43 robustness passed.\n")

        print("=" * 60)
        print("[pipeline] Phase 5/5: seed44 robustness")
        print("=" * 60)
        rc = run_experiment(
            phase="full",
            max_workers=args.max_workers,
            c_modes=["ppo_c", "df_c"],
            r_modes="strict_v13",
            base_dir=Path(f"{BASE}_seed44"),
            ranker_ckpt=RANKER_SEED44,
        )
        if rc != 0:
            print("[pipeline] SEED44 ROBUSTNESS FAILED -- stopping.")
            return rc
        print("[pipeline] Seed44 robustness passed.\n")

    print("[pipeline] ALL PHASES COMPLETED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
