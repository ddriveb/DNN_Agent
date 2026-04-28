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
