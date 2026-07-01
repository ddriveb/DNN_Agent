"""K-Shortest-Path + First-Fit Spectrum Allocation (cached KSP, fast)"""
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

    def preview(self, src, dst, bandwidth_slots):
        """Preview mapper execution without changing persistent state."""
        global_before = self.net.get_global_spectrum_stats()
        if src == dst:
            same_node = {
                "feasible": True,
                "path": [src],
                "start_slot": 0,
                "delay_s": 0.0,
                "path_stats_before": self.net.summarize_availability(self.net.get_available_slots([src])),
                "path_stats_after": self.net.summarize_availability(self.net.get_available_slots([src])),
                "delta_path_frag": 0.0,
                "delta_path_lfb_ratio": 0.0,
                "future_blocking_risk": 0.0,
                "global_before": global_before,
                "global_after": global_before,
                "delta_global_frag": 0.0,
            }
            return same_node

        candidates = self._path_cache.get((src, dst), [])
        preview_candidates = []
        chosen = None

        for path in candidates:
            path_before = self.net.get_path_spectrum_stats(path)
            avail = self.net.get_available_slots(path)
            start_slot = self._first_fit(avail, bandwidth_slots)
            candidate = {
                "path": path,
                "path_length": max(len(path) - 1, 0),
                "path_stats_before": path_before,
                "start_slot": start_slot,
                "feasible": start_slot is not None,
            }

            if start_slot is not None:
                allocated = self.net.allocate(path, start_slot, bandwidth_slots)
                if allocated:
                    path_after = self.net.get_path_spectrum_stats(path)
                    global_after = self.net.get_global_spectrum_stats()
                    self.net.release(path, start_slot, bandwidth_slots)

                    delta_frag = path_after["frag_index"] - path_before["frag_index"]
                    delta_lfb_ratio = (
                        path_before["largest_free_block_ratio"] - path_after["largest_free_block_ratio"]
                    )
                    future_risk = min(
                        1.0,
                        bandwidth_slots / max(float(path_after["max_free_slots"]), 1.0),
                    )

                    candidate.update({
                        "delay_s": self._compute_delay(path),
                        "path_stats_after": path_after,
                        "delta_path_frag": delta_frag,
                        "delta_path_lfb_ratio": delta_lfb_ratio,
                        "future_blocking_risk": future_risk,
                        "global_before": global_before,
                        "global_after": global_after,
                        "delta_global_frag": global_after["avg_frag_index"] - global_before["avg_frag_index"],
                    })

                    if chosen is None:
                        chosen = candidate
                else:
                    candidate["feasible"] = False

            if "path_stats_after" not in candidate:
                candidate.update({
                    "delay_s": None,
                    "path_stats_after": path_before,
                    "delta_path_frag": 0.0,
                    "delta_path_lfb_ratio": 0.0,
                    "future_blocking_risk": min(
                        1.0,
                        bandwidth_slots / max(float(path_before["max_free_slots"]), 1.0),
                    ),
                    "global_before": global_before,
                    "global_after": global_before,
                    "delta_global_frag": 0.0,
                })

            preview_candidates.append(candidate)

        if chosen is None:
            return {
                "feasible": False,
                "path": None,
                "start_slot": None,
                "delay_s": None,
                "path_stats_before": None,
                "path_stats_after": None,
                "delta_path_frag": 0.0,
                "delta_path_lfb_ratio": 0.0,
                "future_blocking_risk": 1.0,
                "global_before": global_before,
                "global_after": global_before,
                "delta_global_frag": 0.0,
                "candidates": preview_candidates,
            }

        chosen = dict(chosen)
        chosen["candidates"] = preview_candidates
        return chosen

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

    def _compute_delay(self, path):
        prop_speed = 2e5
        proc_per_hop = 1e-3
        total_dist = sum(self.net.G[u][v]["length"] for u, v in zip(path[:-1], path[1:]))
        return total_dist / prop_speed + (len(path) - 1) * proc_per_hop
