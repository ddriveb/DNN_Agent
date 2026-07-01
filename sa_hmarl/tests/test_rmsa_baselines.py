"""Tests for RMSA rule-based baselines (KSP-FF, KSP-BF)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sa_hmarl.baselines.rmsa_baselines import ksp_ff_action, ksp_bf_action


# ------------------------------------------------------------------
# 1. KSP-FF selects first valid action
# ------------------------------------------------------------------

def test_ksp_ff_first_valid():
    obs = {
        "agent_r_mask": np.array([False, True, False, True, True], dtype=bool),
    }
    assert ksp_ff_action(obs) == 1
    print("test_ksp_ff_first_valid PASSED")


# ------------------------------------------------------------------
# 2. KSP-BF selects minimum-waste valid action
# ------------------------------------------------------------------

def test_ksp_bf_min_waste():
    """Construct an obs where action 5 has waste=0.0 (best) and action 4 has
    waste=0.5; KSP-BF should pick action 5."""
    obs = {
        "candidate_paths": [[0, 1], [0, 2], [0, 1, 3]],
        "mod_names": ["BPSK", "QPSK"],
        "agent_r_mask": np.array([
            # path0, mod0, block0-1
            True, True,
            # path0, mod1, block0-1
            False, False,
            # path1, mod0, block0-1
            True, True,
            # path1, mod1, block0-1
            False, False,
            # path2, mod0, block0-1
            True, False,
            # path2, mod1, block0-1
            False, False,
        ], dtype=bool),
        "required_fs_per_path_mod": [
            [3, 0],
            [3, 0],
            [3, 0],
        ],
        "candidate_blocks_per_path_mod": [
            # path0: mod0 has blocks (0,5), (5,4) → waste 0.4, 0.25
            [[(0, 5), (5, 4)], []],
            # path1: mod0 has blocks (0,6), (6,3) → waste 0.5, 0.0
            [[(0, 6), (6, 3)], []],
            # path2: mod0 has block (0,5) → waste 0.4
            [[(0, 5)], []],
        ],
    }
    # Action layout (3 paths × 2 mods × 2 blocks = 12):
    # 0: p0,m0,b0  waste=0.4
    # 1: p0,m0,b1  waste=0.25
    # 2: p0,m1,b0  (masked)
    # 3: p0,m1,b1  (masked)
    # 4: p1,m0,b0  waste=0.5
    # 5: p1,m0,b1  waste=0.0   <- best
    # 6: p1,m1,b0  (masked)
    # 7: p1,m1,b1  (masked)
    # 8: p2,m0,b0  waste=0.4
    # 9: p2,m0,b1  (masked)
    # 10: p2,m1,b0 (masked)
    # 11: p2,m1,b1 (masked)
    assert ksp_bf_action(obs) == 5, f"Expected action 5, got {ksp_bf_action(obs)}"
    print("test_ksp_bf_min_waste PASSED")


# ------------------------------------------------------------------
# 3. All-False mask returns None
# ------------------------------------------------------------------

def test_all_false_returns_none():
    obs = {"agent_r_mask": np.array([False, False, False], dtype=bool)}
    assert ksp_ff_action(obs) is None
    assert ksp_bf_action(obs) is None
    print("test_all_false_returns_none PASSED")


if __name__ == "__main__":
    test_ksp_ff_first_valid()
    test_ksp_bf_min_waste()
    test_all_false_returns_none()
    print("\n=== All RMSA baseline tests PASSED ===")
