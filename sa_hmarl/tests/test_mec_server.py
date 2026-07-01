"""Test MEC server with consistent compute-unit semantics (GFLOPS)."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sa_hmarl.mec.server import MECServer


def test_unit_definitions():
    """Verify units are self-consistent."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    assert srv.compute_capacity == 50.0  # GFLOPS
    assert srv.current_load == 0.0       # GFLOPS
    assert srv.utilization == 0.0        # dimensionless
    print("test_unit_definitions PASSED")


def test_allocate_task_returns_bool():
    """allocate_task now returns bool."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    ok = srv.allocate_task(1.0)
    assert ok is True
    assert srv.current_load == 1.0
    print("test_allocate_task_returns_bool PASSED")


def test_small_task_small_utilization():
    """A small task (1 GFLOPS) on a 50 GFLOPS server → 2% utilization."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    assert srv.allocate_task(1.0) is True
    assert srv.current_load == 1.0
    assert srv.utilization == 0.02
    assert srv.available_compute == 49.0
    assert srv.active_tasks == 1
    print("test_small_task_small_utilization PASSED")


def test_multiple_tasks_accumulate():
    """Multiple 5-GFLOPS tasks accumulate linearly."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    for i in range(5):
        assert srv.allocate_task(5.0) is True
    assert srv.current_load == 25.0
    assert srv.utilization == 0.5
    assert srv.available_compute == 25.0
    assert srv.active_tasks == 5
    print("test_multiple_tasks_accumulate PASSED")


def test_release_reduces_load():
    """Release correctly subtracts the same cost."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    assert srv.allocate_task(10.0) is True
    assert srv.allocate_task(10.0) is True
    assert srv.current_load == 20.0
    assert srv.utilization == 0.4

    srv.release_task(10.0)
    assert srv.current_load == 10.0
    assert srv.utilization == 0.2
    assert srv.active_tasks == 1

    srv.release_task(10.0)
    assert srv.current_load == 0.0
    assert srv.utilization == 0.0
    assert srv.active_tasks == 0
    print("test_release_reduces_load PASSED")


def test_release_never_negative():
    """Releasing more than allocated clamps to zero."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    srv.allocate_task(5.0)
    srv.release_task(10.0)  # over-release
    assert srv.current_load == 0.0
    assert srv.utilization == 0.0
    assert srv.active_tasks == 0
    print("test_release_never_negative PASSED")


def test_allocate_rejects_over_total_capacity():
    """Task exceeding total capacity is rejected (not silently capped)."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    ok = srv.allocate_task(60.0)  # > 50 GFLOPS
    assert ok is False
    assert srv.current_load == 0.0
    assert srv.utilization == 0.0
    assert srv.active_tasks == 0
    print("test_allocate_rejects_over_total_capacity PASSED")


def test_allocate_rejects_over_available():
    """Task exceeding available capacity is rejected."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    assert srv.allocate_task(30.0) is True   # load=30
    assert srv.allocate_task(25.0) is False  # 25 > 20 available
    assert srv.current_load == 30.0          # unchanged
    assert srv.utilization == 0.6
    assert srv.active_tasks == 1
    print("test_allocate_rejects_over_available PASSED")


def test_delay_increases_with_load():
    """Higher load → higher compute_delay_ms."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0, scale_ms=20.0)
    cost = 10.0  # GFLOPS

    delay_low = srv.compute_delay_ms(cost)
    expected_low = (10.0 / 50.0) * 20.0  # = 4.0 ms
    assert abs(delay_low - expected_low) < 1e-6

    assert srv.allocate_task(40.0) is True  # 80% loaded
    delay_high = srv.compute_delay_ms(cost)
    expected_high = (10.0 / 10.0) * 20.0  # = 20.0 ms
    assert abs(delay_high - expected_high) < 1e-6

    assert delay_high > delay_low
    print(f"test_delay_increases_with_load: low={delay_low:.2f}ms, high={delay_high:.2f}ms")


def test_delay_inf_when_saturated():
    """Fully saturated server returns infinite delay."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    assert srv.allocate_task(50.0) is True
    assert srv.available_compute == 0.0
    delay = srv.compute_delay_ms(10.0)
    assert delay == float('inf')
    print("test_delay_inf_when_saturated PASSED")


def test_delay_inf_when_cost_exceeds_capacity():
    """Task demand > total capacity → delay = inf."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    delay = srv.compute_delay_ms(60.0)  # 60 > 50
    assert delay == float('inf')
    print("test_delay_inf_when_cost_exceeds_capacity PASSED")


def test_delay_with_memory_overhead():
    """Memory overhead adds linearly to delay."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0, scale_ms=20.0)
    delay_no_mem = srv.compute_delay_ms(10.0, data_size_mb=0.0)
    delay_with_mem = srv.compute_delay_ms(10.0, data_size_mb=10.0)
    assert delay_with_mem == delay_no_mem + 5.0  # 10 MB * 0.5 ms/MB
    print("test_delay_with_memory_overhead PASSED")


def test_queue_delay_ema():
    """Queue delay EMA updates correctly."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    srv.update_queue_delay(100.0)
    assert abs(srv._queue_delay_ema - 30.0) < 1e-6  # 0.3 * 100
    srv.update_queue_delay(0.0)
    assert abs(srv._queue_delay_ema - 21.0) < 1e-6  # 0.7 * 30 + 0.3 * 0
    print("test_queue_delay_ema PASSED")


def test_reset():
    """Reset clears all state."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0)
    srv.allocate_task(10.0)
    srv.update_queue_delay(50.0)
    srv.reset()
    assert srv.current_load == 0.0
    assert srv.utilization == 0.0
    assert srv.active_tasks == 0
    assert srv._queue_delay_ema == 0.0
    print("test_reset PASSED")


def test_delay_scale_ms_custom():
    """Custom scale_ms is respected."""
    srv = MECServer(0, 0, compute_capacity_gflops=50.0, scale_ms=100.0)
    delay = srv.compute_delay_ms(10.0)
    expected = (10.0 / 50.0) * 100.0  # = 20.0 ms
    assert abs(delay - expected) < 1e-6
    print("test_delay_scale_ms_custom PASSED")


if __name__ == "__main__":
    test_unit_definitions()
    test_allocate_task_returns_bool()
    test_small_task_small_utilization()
    test_multiple_tasks_accumulate()
    test_release_reduces_load()
    test_release_never_negative()
    test_allocate_rejects_over_total_capacity()
    test_allocate_rejects_over_available()
    test_delay_increases_with_load()
    test_delay_inf_when_saturated()
    test_delay_inf_when_cost_exceeds_capacity()
    test_delay_with_memory_overhead()
    test_queue_delay_ema()
    test_reset()
    test_delay_scale_ms_custom()
    print("\n=== All MEC server tests PASSED ===")
