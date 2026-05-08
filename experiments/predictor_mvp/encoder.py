"""State encoder z_t with ablation variants + coarse binning + noise + topology conditioning."""
import numpy as np
import networkx as nx
from ksp_fast import k_shortest_paths


class Encoder:
    def __init__(self, network, k=3):
        self.net = network
        self.k = k
        self._path_cache = {}
        self._precompute_ksp()
        self._topo_stats = None

    def compute_topology_stats(self):
        """Compute normalized graph-level topology statistics."""
        G = self.net.G
        n = G.number_of_nodes()
        m = G.number_of_edges()
        avg_degree = 2 * m / n if n > 0 else 0.0
        density = nx.density(G)
        avg_clustering = nx.average_clustering(G)
        try:
            diameter = nx.diameter(G)
            avg_path_len = nx.average_shortest_path_length(G)
        except nx.NetworkXError:
            diameter = -1
            avg_path_len = -1
        stats = [
            n / 128.0,
            m / 500.0,
            avg_degree / 10.0,
            diameter / 20.0 if diameter > 0 else 0.0,
            avg_clustering,
            avg_path_len / 10.0 if avg_path_len > 0 else 0.0,
            density,
        ]
        self._topo_stats = np.array(stats, dtype=np.float32)
        return self._topo_stats

    def get_topology_stats(self):
        if self._topo_stats is None:
            self.compute_topology_stats()
        return self._topo_stats

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

    def encode_v0(self, src, dst):
        return np.array([self._global_frag_index()], dtype=np.float32)

    def encode_v1(self, src, dst):
        return np.array(
            [self._global_frag_index(), self._global_max_block() / self.net.num_slots],
            dtype=np.float32,
        )

    def encode_v2(self, src, dst):
        global_feats = [
            self._global_frag_index(),
            self._global_max_block() / self.net.num_slots,
        ]
        path_feats = self._path_histogram(src, dst, mode="exact")
        return np.array(global_feats + path_feats, dtype=np.float32)

    def encode_v2b(self, src, dst, num_bins=4):
        global_feats = [
            self._global_frag_index(),
            self._global_max_block() / self.net.num_slots,
        ]
        path_feats = self._path_histogram(src, dst, mode="binned", num_bins=num_bins)
        return np.array(global_feats + path_feats, dtype=np.float32)

    def encode_v2n(self, src, dst, noise_std=0.05):
        z = self.encode_v2(src, dst)
        noise = np.random.normal(0, noise_std, size=z.shape)
        return np.clip(z + noise, 0, 1).astype(np.float32)

    def encode_v2s(self, src, dst):
        global_feats = [
            self._global_frag_index(),
            self._global_max_block() / self.net.num_slots,
        ]
        path_feats = self._path_histogram(src, dst, mode="summary")
        return np.array(global_feats + path_feats, dtype=np.float32)

    def encode_v2t(self, src, dst, num_bins=4):
        """v2b + topology-conditioned: append graph-level topology statistics."""
        z_v2b = self.encode_v2b(src, dst, num_bins=num_bins)
        topo = self.get_topology_stats()
        return np.concatenate([z_v2b, topo]).astype(np.float32)

    def _global_frag_index(self):
        frags = []
        for link, slots in self.net.link_states.items():
            free = ~slots
            total_free = int(np.sum(free))
            if total_free == 0 or total_free == self.net.num_slots:
                continue
            max_free = self._max_consecutive(free)
            frags.append(1.0 - max_free / total_free)
        return float(np.mean(frags)) if frags else 0.0

    def _global_max_block(self):
        vals = []
        for link, slots in self.net.link_states.items():
            vals.append(self._max_consecutive(~slots))
        return float(np.mean(vals)) if vals else 0.0

    def _path_histogram(self, src, dst, mode="exact", num_bins=4):
        paths = self._path_cache.get((src, dst), [])
        if not paths:
            if mode == "binned":
                return [0.0] * (num_bins * 2)
            if mode == "summary":
                return [0.0] * 3
            return [0.0] * 5

        avails, blocks = [], []
        for path in paths:
            avail = self.net.get_available_slots(path)
            avails.append(float(np.sum(avail)))
            blocks.append(float(self._max_consecutive(avail)))
        avails = np.array(avails)
        blocks = np.array(blocks)
        N = self.net.num_slots

        if mode == "exact":
            return [
                float(np.mean(avails)) / N, float(np.max(avails)) / N,
                float(np.min(avails)) / N, float(np.mean(blocks)) / N,
                float(np.max(blocks)) / N,
            ]

        if mode == "summary":
            return [
                float(np.mean(avails)) / N,
                float(np.mean(blocks)) / N,
                float(np.max(blocks)) / N,
            ]

        if mode == "binned":
            avail_fracs = avails / N
            block_fracs = blocks / N
            bin_edges = np.linspace(0, 1, num_bins + 1)
            avail_hist = np.zeros(num_bins, dtype=np.float32)
            block_hist = np.zeros(num_bins, dtype=np.float32)
            for f in avail_fracs:
                idx = min(int(f / (1.0 / num_bins)), num_bins - 1)
                avail_hist[idx] += 1
            for f in block_fracs:
                idx = min(int(f / (1.0 / num_bins)), num_bins - 1)
                block_hist[idx] += 1
            if len(paths) > 0:
                avail_hist /= len(paths)
                block_hist /= len(paths)
            return avail_hist.tolist() + block_hist.tolist()

        return [0.0] * 5

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
