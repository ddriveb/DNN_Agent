"""MEC server model with consistent compute-unit semantics.

Unit conventions (all compute quantities use GFLOPS / GFLOP):
  - compute_capacity : GFLOPS  (server max sustained compute rate)
  - current_load     : GFLOPS  (sum of active tasks' compute rate demands)
  - compute_cost     : GFLOPS  (per-task compute rate demand)

This means:
  - allocate_task(10.0) on a 50 GFLOPS server → utilization = 10/50 = 0.20
  - release_task(10.0)                      → utilization drops back
  - compute_delay_ms uses  (cost / available) * scale_ms  where the ratio
    is dimensionless and scale_ms is a calibrated baseline delay (ms).
"""
import numpy as np
from typing import Optional


class MECServer:
    """A single MEC/edge server with compute resources and queue tracking."""

    # Default baseline delay factor (ms) used when available_compute == compute_cost.
    # Calibrated so that a task whose demand equals the server's full capacity
    # would take ~20 ms to complete in this simplified model.
    DEFAULT_SCALE_MS = 20.0

    def __init__(self, server_id: int, node_id: int,
                 compute_capacity_gflops: float = 50.0,
                 memory_capacity_gb: float = 16.0,
                 scale_ms: Optional[float] = None):
        self.server_id = server_id
        self.node_id = node_id
        # compute_capacity: GFLOPS — maximum sustained compute rate of this server
        self.compute_capacity = float(compute_capacity_gflops)
        self.memory_capacity = float(memory_capacity_gb)
        # current_load: GFLOPS — total compute rate currently occupied by active tasks
        self.current_load = 0.0
        self.active_tasks = 0
        self._scale_ms = scale_ms if scale_ms is not None else self.DEFAULT_SCALE_MS
        # Queue delay estimation (exponential moving average)
        self._queue_delay_ema = 0.0
        self._queue_alpha = 0.3

    @property
    def available_compute(self) -> float:
        """Remaining compute capacity in GFLOPS."""
        return max(self.compute_capacity - self.current_load, 0.0)

    @property
    def utilization(self) -> float:
        """Dimensionless utilization in [0, 1]."""
        return self.current_load / self.compute_capacity if self.compute_capacity > 0 else 1.0

    def compute_delay_ms(self, compute_cost: float, data_size_mb: float = 0.0) -> float:
        """Estimate processing delay (ms) for a task.

        Args:
            compute_cost: GFLOPS — the task's compute rate demand.
            data_size_mb: MB — optional memory transfer overhead.

        Returns:
            Estimated delay in milliseconds.  Infinite if task is infeasible.
        """
        # Task demand exceeds total capacity → infeasible
        if compute_cost > self.compute_capacity:
            return float('inf')
        # No remaining capacity → infeasible
        if self.available_compute <= 0:
            return float('inf')
        # Dimensionless ratio: how much of the remaining capacity this task needs
        demand_ratio = compute_cost / self.available_compute
        base_ms = demand_ratio * self._scale_ms
        mem_ms = data_size_mb * 0.5
        return base_ms + mem_ms + self._queue_delay_ema

    def allocate_task(self, compute_cost: float) -> bool:
        """Book compute resources for an incoming task.

        Args:
            compute_cost: GFLOPS — the task's compute rate demand.

        Returns:
            True if the task is admitted, False if it exceeds capacity.
        """
        # Infeasible: task demand exceeds total server capacity
        if compute_cost > self.compute_capacity:
            return False
        # Infeasible: not enough remaining capacity
        if compute_cost > self.available_compute:
            return False
        self.current_load += compute_cost
        self.active_tasks += 1
        return True

    def release_task(self, compute_cost: float):
        """Release compute resources when a task finishes.

        Args:
            compute_cost: GFLOPS — the same value passed to allocate_task().
        """
        self.current_load -= compute_cost
        self.current_load = max(self.current_load, 0.0)
        self.active_tasks = max(self.active_tasks - 1, 0)

    def update_queue_delay(self, actual_queue_delay_ms: float):
        """Update queue delay EMA with an observed value."""
        self._queue_delay_ema = ((1 - self._queue_alpha) * self._queue_delay_ema
                                  + self._queue_alpha * actual_queue_delay_ms)

    def reset(self):
        self.current_load = 0.0
        self.active_tasks = 0
        self._queue_delay_ema = 0.0
