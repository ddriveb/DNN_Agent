"""K-Shortest-Path + First-Fit Spectrum Allocation (cached KSP, fast)"""
import numpy as np
from ksp_fast import k_shortest_paths


class KSPMapper:
    def __init__(self, network, k=3):
        self.net = network
        self.k = k
        self._path_cache = {}
        self._precompute_ksp()

    def _precompute_ksp(self):
        n = self.net.NUM_NODES
        for src in range(n):
            for dst in range(n):
                if src == dst:
                    self._path_cache[(src, dst)] = [[src]]
                else:
                    self._path_cache[(src, dst)] = k_shortest_paths(
                        self.net.G, src, dst, k=self.k, weight="weight"
                    )

    def map(self, src, dst, bandwidth_slots):
        if src == dst:
            return True, [src], 0, 0.0

        candidates = self._path_cache.get((src, dst), [])
        for path in candidates:
            avail = self.net.get_available_slots(path)
            start_slot = self._first_fit(avail, bandwidth_slots)
            if start_slot is not None:
                success = self.net.allocate(path, start_slot, bandwidth_slots)
                if success:
                    delay = self._compute_delay(path)
                    return True, path, start_slot, delay
        return False, None, None, None

    @staticmethod
    def _first_fit(avail, need):
        if need == 0:
            return 0
        count = 0
        for i, free in enumerate(avail):
            if free:
                count += 1
                if count >= need:
                    return i - need + 1
            else:
                count = 0
        return None

    def execute_path(self, src, dst, bandwidth_slots, path_id):
        """
        Execute a specific path chosen by Selector.
        Returns detailed result dict including frag_change and resource_cost.
        """
        if src == dst:
            return {
                "success": True, "path": [src], "start_slot": 0,
                "delay": 0.0, "resource_cost": 0, "frag_change": 0.0,
            }

        paths = self._path_cache.get((src, dst), [])
        if path_id >= len(paths):
            return {
                "success": False, "path": None, "start_slot": None,
                "delay": None, "resource_cost": 0, "frag_change": 0.0,
            }

        path = paths[path_id]
        frag_before = self._compute_path_frag(path)
        avail = self.net.get_available_slots(path)
        start_slot = self._first_fit(avail, bandwidth_slots)

        if start_slot is not None:
            success = self.net.allocate(path, start_slot, bandwidth_slots)
            if success:
                delay = self._compute_delay(path)
                resource_cost = bandwidth_slots * (len(path) - 1)
                frag_after = self._compute_path_frag(path)
                frag_change = frag_after - frag_before
                return {
                    "success": True, "path": path, "start_slot": start_slot,
                    "delay": delay, "resource_cost": resource_cost,
                    "frag_change": frag_change,
                }

        return {
            "success": False, "path": None, "start_slot": None,
            "delay": None, "resource_cost": 0, "frag_change": 0.0,
        }

    def _compute_path_frag(self, path):
        """Compute average fragmentation index over all links in path."""
        frags = []
        for u, v in zip(path[:-1], path[1:]):
            link = (min(u, v), max(u, v))
            slots = self.net.link_states[link]
            free = ~slots
            total_free = int(np.sum(free))
            if total_free == 0 or total_free == self.net.num_slots:
                continue
            max_free = self._max_consecutive(free)
            frags.append(1.0 - max_free / total_free)
        return float(np.mean(frags)) if frags else 0.0

    @staticmethod
    def _max_consecutive(arr):
        if not np.any(arr):
            return 0
        max_len = curr = 0
        for v in arr:
            if v:
                curr += 1
                max_len = max(max_len, curr)
            else:
                curr = 0
        return max_len

    def _compute_delay(self, path):
        prop_speed = 2e5
        proc_per_hop = 1e-3
        total_dist = sum(self.net.G[u][v]["length"] for u, v in zip(path[:-1], path[1:]))
        return total_dist / prop_speed + (len(path) - 1) * proc_per_hop
