"""Test modulation format module."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sa_hmarl.network.modulation import (
    ModulationFormat, DEFAULT_MODULATIONS,
    get_feasible_modulations, ModulationRegistry
)


def test_feasible_modulations():
    # Short path: all modulations feasible
    feas = get_feasible_modulations(300.0)
    assert len(feas) == 4
    assert feas[0].name == "16QAM"  # Highest SE first

    # Medium path: 16QAM excluded
    feas = get_feasible_modulations(600.0)
    names = [m.name for m in feas]
    assert "16QAM" not in names
    assert "8QAM" in names

    # Long path: only BPSK
    feas = get_feasible_modulations(3000.0)
    assert len(feas) == 1
    assert feas[0].name == "BPSK"

    print("test_feasible_modulations PASSED")


def test_registry():
    reg = ModulationRegistry()
    assert reg.num_formats == 4
    m = reg.by_name("QPSK")
    assert m is not None
    assert m.reach_km == 2000.0
    print("test_registry PASSED")


if __name__ == "__main__":
    test_feasible_modulations()
    test_registry()
