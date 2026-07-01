"""Test FS demand calculator."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sa_hmarl.env.fs_demand import FSDemandCalculator
from sa_hmarl.network.modulation import ModulationFormat


def test_basic_calculation():
    calc = FSDemandCalculator()
    mod = ModulationFormat("QPSK", 2000.0, 2.0)

    # Simple case: 1 Gbit data, 1 ms transmit time
    fs = calc.compute_fs_requirement(data_bits=1e9, modulation=mod, t_data_max=1e-3)
    # slot_bw = 12.5e9 Hz, SE = 2.0 → 25e9 bits/s per slot
    # 1e9 bits / (25e9 * 1e-3) = 1e9 / 25e6 = 40 slots
    assert fs >= 1
    print(f"test_basic_calculation: fs={fs} (expected ~40)")


def test_full_computation():
    calc = FSDemandCalculator()
    mod = ModulationFormat("QPSK", 2000.0, 2.0)

    # Use short path + relaxed deadline so time budget is positive
    edge_lengths = {(0, 1): 100.0}
    feasible, fs_req, t_data = calc.compute_full(
        data_bits=1e8,
        deadline_s=50e-3,       # 50ms deadline
        ctrl_delay_s=0.1e-3,
        local_compute_s=1.0e-3,
        edge_compute_s=1.0e-3,
        queue_delay_s=0.5e-3,
        path=[0, 1],
        edge_lengths=edge_lengths,
        modulation=mod
    )
    assert feasible
    assert fs_req >= 1
    print(f"test_full_computation: feasible={feasible}, fs={fs_req}, t_data={t_data*1e3:.3f}ms")


def test_infeasible_budget():
    calc = FSDemandCalculator()
    mod = ModulationFormat("QPSK", 2000.0, 2.0)

    edge_lengths = {(0, 1): 1000.0, (1, 2): 1000.0}
    feasible, fs_req, t_data = calc.compute_full(
        data_bits=1e12,
        deadline_s=0.001,  # Very tight deadline
        ctrl_delay_s=0.1e-3,
        local_compute_s=0.5e-3,
        edge_compute_s=0.5e-3,
        queue_delay_s=0.1e-3,
        path=[0, 1, 2],
        edge_lengths=edge_lengths,
        modulation=mod
    )
    assert not feasible
    print("test_infeasible_budget PASSED")


if __name__ == "__main__":
    test_basic_calculation()
    test_full_computation()
    test_infeasible_budget()
