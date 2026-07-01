"""Test action mask mechanisms for Agent-C and Agent-R."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from sa_hmarl.env.action_mask import build_agent_c_mask, build_agent_r_mask, apply_mask


# ------------------------------------------------------------------
# Agent-C tests
# ------------------------------------------------------------------

def test_agent_c_mask():
    util = [0.5, 0.96, 0.3]
    mask = build_agent_c_mask(2, 3, util, util_threshold=0.95)
    # Server 1 (idx 1) is over threshold → both splits masked
    assert mask[1] == False   # split0, server1
    assert mask[4] == False   # split1, server1
    assert mask[0] == True
    assert mask[2] == True
    print("test_agent_c_mask PASSED")


def test_agent_c_mask_with_feasible():
    util = [0.5, 0.5, 0.5]
    feas = [[1, 0, 2], [2, 3, 0]]  # 2 splits x 3 servers
    mask = build_agent_c_mask(2, 3, util, feasible_counts=feas)
    # split0, server1 has 0 feasible combos → masked
    assert mask[1] == False
    # split1, server2 has 0 feasible combos → masked
    assert mask[5] == False
    print("test_agent_c_mask_with_feasible PASSED")


# ------------------------------------------------------------------
# Agent-R tests
# ------------------------------------------------------------------

def test_agent_r_mask_mod_infeasible():
    """If modulation is infeasible for a path, all blocks are masked."""
    num_paths, num_mods, num_blocks = 2, 3, 4
    mod_names = ["BPSK", "QPSK", "16QAM"]

    # Path 0: only BPSK feasible
    # Path 1: BPSK and QPSK feasible
    feasible_mask = [
        [True, False, False],
        [True, True, False],
    ]

    # required_fs: same for all
    req_fs = [
        [2, 2, 2],
        [2, 2, 2],
    ]

    # candidate blocks: plenty for all
    blocks = [
        [[(0, 4), (4, 4), (8, 4), (12, 4)],
         [(0, 4), (4, 4), (8, 4), (12, 4)],
         [(0, 4), (4, 4), (8, 4), (12, 4)]],
        [[(0, 4), (4, 4), (8, 4), (12, 4)],
         [(0, 4), (4, 4), (8, 4), (12, 4)],
         [(0, 4), (4, 4), (8, 4), (12, 4)]],
    ]

    mask = build_agent_r_mask(num_paths, num_mods, num_blocks, mod_names,
                              feasible_mask, req_fs, blocks)

    total = num_paths * num_mods * num_blocks  # 24
    assert len(mask) == total

    # Path 0, QPSK (m=1): all 4 blocks should be masked
    p0_qpsk_base = 0 * (3 * 4) + 1 * 4
    for b in range(4):
        assert mask[p0_qpsk_base + b] == False

    # Path 0, 16QAM (m=2): all 4 blocks should be masked
    p0_16qam_base = 0 * (3 * 4) + 2 * 4
    for b in range(4):
        assert mask[p0_16qam_base + b] == False

    # Path 1, 16QAM (m=2): all 4 blocks should be masked
    p1_16qam_base = 1 * (3 * 4) + 2 * 4
    for b in range(4):
        assert mask[p1_16qam_base + b] == False

    # Path 0, BPSK (m=0): all 4 blocks should be allowed
    p0_bpsk_base = 0 * (3 * 4) + 0 * 4
    for b in range(4):
        assert mask[p0_bpsk_base + b] == True

    print("test_agent_r_mask_mod_infeasible PASSED")


def test_agent_r_mask_block_size_insufficient():
    """If block size < required_fs, that block is masked."""
    num_paths, num_mods, num_blocks = 1, 1, 4
    mod_names = ["QPSK"]

    feasible_mask = [[True]]
    req_fs = [[3]]  # need 3 slots

    # blocks: sizes 2, 3, 4, 1  → only b=1 (size=3) and b=2 (size=4) are enough
    blocks = [
        [[(0, 2), (2, 3), (5, 4), (9, 1)]],
    ]

    mask = build_agent_r_mask(num_paths, num_mods, num_blocks, mod_names,
                              feasible_mask, req_fs, blocks)

    assert mask[0] == False  # size=2 < 3
    assert mask[1] == True   # size=3 >= 3
    assert mask[2] == True   # size=4 >= 3
    assert mask[3] == False  # size=1 < 3
    print("test_agent_r_mask_block_size_insufficient PASSED")


def test_agent_r_mask_all_feasible():
    """When modulation feasible, block exists, and size sufficient → allow."""
    num_paths, num_mods, num_blocks = 1, 2, 3
    mod_names = ["BPSK", "QPSK"]

    feasible_mask = [[True, True]]
    req_fs = [[1, 2]]

    blocks = [
        [[(0, 5), (5, 5), (10, 5)],   # BPSK: 3 blocks, all size=5 >= 1
         [(0, 3), (3, 3), (6, 3)]],   # QPSK: 3 blocks, all size=3 >= 2
    ]

    mask = build_agent_r_mask(num_paths, num_mods, num_blocks, mod_names,
                              feasible_mask, req_fs, blocks)

    # All 6 slots should be True
    assert np.all(mask), f"Expected all True, got {mask}"
    print("test_agent_r_mask_all_feasible PASSED")


def test_agent_r_mask_req_fs_none():
    """If required_fs is None or 0, all blocks are masked."""
    num_paths, num_mods, num_blocks = 1, 2, 3
    mod_names = ["BPSK", "QPSK"]

    feasible_mask = [[True, True]]
    req_fs = [[None, 0]]  # None and 0

    blocks = [
        [[(0, 5), (5, 5), (10, 5)],
         [(0, 5), (5, 5), (10, 5)]],
    ]

    mask = build_agent_r_mask(num_paths, num_mods, num_blocks, mod_names,
                              feasible_mask, req_fs, blocks)

    assert not np.any(mask), f"Expected all False, got {mask}"
    print("test_agent_r_mask_req_fs_none PASSED")


def test_agent_r_mask_no_blocks():
    """If candidate_blocks list is empty, all blocks are masked."""
    num_paths, num_mods, num_blocks = 1, 1, 3
    mod_names = ["BPSK"]

    feasible_mask = [[True]]
    req_fs = [[2]]

    blocks = [
        [[]],  # empty blocks
    ]

    mask = build_agent_r_mask(num_paths, num_mods, num_blocks, mod_names,
                              feasible_mask, req_fs, blocks)

    assert not np.any(mask), f"Expected all False, got {mask}"
    print("test_agent_r_mask_no_blocks PASSED")


def test_agent_r_mask_partial_blocks():
    """Fewer candidate blocks than num_blocks → extra slots stay masked."""
    num_paths, num_mods, num_blocks = 1, 1, 5
    mod_names = ["BPSK"]

    feasible_mask = [[True]]
    req_fs = [[2]]

    # Only 2 candidate blocks provided
    blocks = [
        [[(0, 4), (4, 4)]],
    ]

    mask = build_agent_r_mask(num_paths, num_mods, num_blocks, mod_names,
                              feasible_mask, req_fs, blocks)

    assert mask[0] == True   # size=4 >= 2
    assert mask[1] == True   # size=4 >= 2
    assert mask[2] == False  # no block 2
    assert mask[3] == False  # no block 3
    assert mask[4] == False  # no block 4
    print("test_agent_r_mask_partial_blocks PASSED")


# ------------------------------------------------------------------
# apply_mask tests
# ------------------------------------------------------------------

def test_apply_mask():
    logits = np.array([1.0, 2.0, 3.0, 4.0])
    mask = np.array([True, False, True, False])
    probs = apply_mask(logits, mask)
    assert np.isclose(np.sum(probs), 1.0)
    assert probs[1] == 0.0
    assert probs[3] == 0.0
    assert probs[2] > probs[0]
    print("test_apply_mask PASSED")


# ------------------------------------------------------------------
# Shape validation tests
# ------------------------------------------------------------------

def _make_valid_agent_r_args():
    """Helper to build minimal valid args for build_agent_r_mask."""
    return {
        "num_paths": 1,
        "num_modulations": 2,
        "num_blocks": 3,
        "mod_names": ["BPSK", "QPSK"],
        "feasible_mask_per_path_mod": [[True, True]],
        "required_fs_per_path_mod": [[1, 1]],
        "candidate_blocks_per_path_mod": [
            [[(0, 5), (5, 5), (10, 5)],
             [(0, 5), (5, 5), (10, 5)]],
        ],
    }


def test_agent_r_mask_mod_names_mismatch():
    """mod_names length != num_modulations raises ValueError."""
    args = _make_valid_agent_r_args()
    args["mod_names"] = ["BPSK"]  # only 1, but num_modulations=2
    try:
        build_agent_r_mask(**args)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "mod_names length" in str(e)
        print(f"test_agent_r_mask_mod_names_mismatch: {e}")


def test_agent_r_mask_feasible_mask_too_short():
    """feasible_mask outer length < num_paths raises ValueError."""
    args = _make_valid_agent_r_args()
    args["num_paths"] = 2
    args["feasible_mask_per_path_mod"] = [[True, True]]  # only 1 path
    try:
        build_agent_r_mask(**args)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "feasible_mask_per_path_mod outer length" in str(e)
        print(f"test_agent_r_mask_feasible_mask_too_short: {e}")


def test_agent_r_mask_required_fs_too_short():
    """required_fs outer length < num_paths raises ValueError."""
    args = _make_valid_agent_r_args()
    args["num_paths"] = 2
    # feasible_mask has 2 paths (so it passes its own check)
    args["feasible_mask_per_path_mod"] = [[True, True], [True, True]]
    args["required_fs_per_path_mod"] = [[1, 1]]  # only 1 path
    try:
        build_agent_r_mask(**args)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "required_fs_per_path_mod outer length" in str(e)
        print(f"test_agent_r_mask_required_fs_too_short: {e}")


def test_agent_r_mask_inner_feasible_too_short():
    """feasible_mask inner length < num_modulations raises ValueError."""
    args = _make_valid_agent_r_args()
    args["feasible_mask_per_path_mod"] = [[True]]  # only 1 mod, need 2
    try:
        build_agent_r_mask(**args)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "feasible_mask_per_path_mod[0] length" in str(e)
        print(f"test_agent_r_mask_inner_feasible_too_short: {e}")


def test_agent_r_mask_inner_required_fs_too_short():
    """required_fs inner length < num_modulations raises ValueError."""
    args = _make_valid_agent_r_args()
    args["required_fs_per_path_mod"] = [[1]]  # only 1 mod, need 2
    try:
        build_agent_r_mask(**args)
        assert False, "Expected ValueError"
    except ValueError as e:
        assert "required_fs_per_path_mod[0] length" in str(e)
        print(f"test_agent_r_mask_inner_required_fs_too_short: {e}")


if __name__ == "__main__":
    test_agent_c_mask()
    test_agent_c_mask_with_feasible()
    test_agent_r_mask_mod_infeasible()
    test_agent_r_mask_block_size_insufficient()
    test_agent_r_mask_all_feasible()
    test_agent_r_mask_req_fs_none()
    test_agent_r_mask_no_blocks()
    test_agent_r_mask_partial_blocks()
    test_apply_mask()
    test_agent_r_mask_mod_names_mismatch()
    test_agent_r_mask_feasible_mask_too_short()
    test_agent_r_mask_required_fs_too_short()
    test_agent_r_mask_inner_feasible_too_short()
    test_agent_r_mask_inner_required_fs_too_short()
    print("\n=== All action mask tests PASSED ===")
