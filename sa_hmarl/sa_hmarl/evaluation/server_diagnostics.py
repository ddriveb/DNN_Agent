"""Light-weight server-level diagnostics for SMDP evaluation runs."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np


@dataclass
class ServerDiagnostics:
    """Accumulates per-server statistics across one or more episodes."""

    num_servers: int = 4
    num_splits: int = 3
    high_util_threshold: float = 0.90

    selected_count: np.ndarray = field(init=False)
    util_samples: List[List[float]] = field(init=False)
    queue_delay_samples: List[List[float]] = field(init=False)
    overload_by_server: np.ndarray = field(init=False)
    overload_by_split_server: np.ndarray = field(init=False)

    first_high_util_time: Dict[int, float] = field(default_factory=dict)
    high_util_samples: np.ndarray = field(init=False)
    total_samples: int = 0

    def __post_init__(self):
        self.selected_count = np.zeros(self.num_servers, dtype=int)
        self.util_samples = [[] for _ in range(self.num_servers)]
        self.queue_delay_samples = [[] for _ in range(self.num_servers)]
        self.overload_by_server = np.zeros(self.num_servers, dtype=int)
        self.overload_by_split_server = np.zeros((self.num_splits, self.num_servers), dtype=int)
        self.high_util_samples = np.zeros(self.num_servers, dtype=int)

    def record_step(
        self,
        env,
        split_id: int,
        server_id: int,
        info: Dict[str, Any],
    ) -> None:
        """Record one decision step."""
        if server_id < 0 or server_id >= self.num_servers:
            return

        self.selected_count[server_id] += 1

        # Per-server utilization / queue snapshots.
        servers = getattr(getattr(env, "mec", None), "servers", [])
        current_time = float(getattr(env, "time", 0.0))
        for s in range(min(self.num_servers, len(servers))):
            srv = servers[s]
            util = float(getattr(srv, "utilization", 0.0))
            qd = float(getattr(srv, "_queue_delay_ema", 0.0))
            self.util_samples[s].append(util)
            self.queue_delay_samples[s].append(qd)
            if util >= self.high_util_threshold:
                self.high_util_samples[s] += 1
                if s not in self.first_high_util_time:
                    self.first_high_util_time[s] = current_time

        # Overload attribution.
        if not info.get("success", True):
            reason = info.get("reason", "")
            if "server" in reason.lower() or reason in ("server_overload", "server_saturated"):
                self.overload_by_server[server_id] += 1
                split_idx = max(0, min(split_id, self.num_splits - 1))
                self.overload_by_split_server[split_idx, server_id] += 1

        self.total_samples += 1

    def aggregate(self) -> Dict[str, Any]:
        """Return aggregated per-server statistics."""
        n = self.num_servers
        total_selected = int(self.selected_count.sum())
        total_overloads = int(self.overload_by_server.sum())

        selection_pct = (
            (self.selected_count / max(total_selected, 1) * 100).tolist()
            if total_selected > 0
            else [0.0] * n
        )

        util_stats = []
        queue_stats = []
        high_util_frac = []
        for s in range(n):
            u = np.asarray(self.util_samples[s], dtype=float)
            q = np.asarray(self.queue_delay_samples[s], dtype=float)
            util_stats.append({
                "mean": float(u.mean()) if u.size else 0.0,
                "p95": float(np.percentile(u, 95)) if u.size else 0.0,
                "max": float(u.max()) if u.size else 0.0,
            })
            queue_stats.append({
                "mean": float(q.mean()) if q.size else 0.0,
                "p95": float(np.percentile(q, 95)) if q.size else 0.0,
                "max": float(q.max()) if q.size else 0.0,
            })
            high_util_frac.append(
                float(self.high_util_samples[s]) / max(self.total_samples, 1)
            )

        overload_by_server_pct = (
            (self.overload_by_server / max(total_overloads, 1) * 100).tolist()
            if total_overloads > 0
            else [0.0] * n
        )

        return {
            "num_servers": n,
            "num_splits": self.num_splits,
            "high_util_threshold": self.high_util_threshold,
            "total_samples": self.total_samples,
            "total_overloads": total_overloads,
            "selected_count": self.selected_count.tolist(),
            "selected_percentage": selection_pct,
            "utilization": util_stats,
            "queue_delay_ms": queue_stats,
            "high_util_fraction": high_util_frac,
            "first_high_util_time": self.first_high_util_time,
            "overload_by_server": self.overload_by_server.tolist(),
            "overload_by_server_percentage": overload_by_server_pct,
            "overload_by_split_server": self.overload_by_split_server.tolist(),
        }
