"""Event-driven dataset generation."""
import numpy as np
from link_as_node_gnn_predictor import build_converted_topology, compute_link_node_features


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


class DatasetGenerator:
    def __init__(self, network, mapper, encoder, num_partitions=3, seed=None):
        self.net = network
        self.mapper = mapper
        self.encoder = encoder
        self.num_partitions = num_partitions
        self.num_servers = network.NUM_NODES
        self.rng = np.random.RandomState(seed)
        self.bw_choices = [1, 2, 4, 8, 16, 32]
        self._edge_index = None
        self._link_index_map = None
        self._converted_edge_index = None
        self._path_masks = None
        # PINN structures
        self._pinn_link_index_map = None
        self._pinn_path_link_masks = None

    def _ensure_pinn_structure(self):
        """Precompute link index map and path link masks for PINN."""
        if self._pinn_link_index_map is None:
            links = sorted(self.net.link_states.keys())
            self._pinn_link_index_map = {link: i for i, link in enumerate(links)}
            self._num_links = len(links)
            # Precompute path link masks
            self._pinn_path_link_masks = {}
            for src in range(self.num_servers):
                for dst in range(self.num_servers):
                    if src == dst:
                        continue
                    paths = self.mapper._path_cache.get((src, dst), [])
                    for pid, path in enumerate(paths):
                        mask = np.zeros(self._num_links, dtype=np.float32)
                        for j in range(len(path) - 1):
                            link = (min(path[j], path[j + 1]), max(path[j], path[j + 1]))
                            if link in self._pinn_link_index_map:
                                mask[self._pinn_link_index_map[link]] = 1.0
                        self._pinn_path_link_masks[(src, dst, pid)] = mask

    def _get_S_current(self):
        """Compute current spectrum occupancy matrix [num_links, num_slots]."""
        num_links = len(self._pinn_link_index_map)
        S = np.zeros((num_links, self.net.num_slots), dtype=np.float32)
        for link, idx in sorted(self._pinn_link_index_map.items(), key=lambda x: x[1]):
            S[idx] = self.net.link_states[link].astype(np.float32)
        return S

    def _ensure_link_as_node_structure(self):
        """Precompute converted topology and path masks (static)."""
        if self._link_index_map is None:
            self._link_index_map, self._converted_edge_index = build_converted_topology(self.net)
            # Precompute path masks for all (src, dst, path_id)
            self._path_masks = {}
            for src in range(self.num_servers):
                for dst in range(self.num_servers):
                    if src == dst:
                        continue
                    paths = self.mapper._path_cache.get((src, dst), [])
                    for pid, path in enumerate(paths):
                        mask = np.zeros(len(self._link_index_map), dtype=np.float32)
                        for idx in range(len(path) - 1):
                            link = (min(path[idx], path[idx + 1]), max(path[idx], path[idx + 1]))
                            if link in self._link_index_map:
                                mask[self._link_index_map[link]] = 1.0
                        self._path_masks[(src, dst, pid)] = mask

    def _get_edge_index(self):
        if self._edge_index is None:
            edges = []
            for (u, v) in self.net.link_states.keys():
                edges.append([u, v])
            self._edge_index = np.array(edges, dtype=np.int64)
        return self._edge_index

    def _get_edge_features(self):
        """Compute per-edge spectrum features."""
        N = self.net.num_slots
        features = []
        for (u, v), slots in self.net.link_states.items():
            avail = ~slots
            total_free = int(np.sum(avail))
            max_free = _max_consecutive(avail)
            frag = 1.0 - max_free / total_free if total_free > 0 and total_free < N else 0.0
            avail_frac = total_free / N
            block_frac = max_free / N
            features.append([avail_frac, block_frac, frag])
        return np.array(features, dtype=np.float32)

    def _random_request(self):
        src = self.rng.randint(0, self.num_servers)
        dst = self.rng.randint(0, self.num_servers)
        while dst == src:
            dst = self.rng.randint(0, self.num_servers)
        path_id = self.rng.randint(0, self.num_partitions)
        # bw is independent of path_id: sample from valid choices
        valid_bw = [b for b in self.bw_choices if b <= self.net.num_slots]
        bw = valid_bw[self.rng.randint(0, len(valid_bw))]
        return src, dst, path_id, bw

    def generate(self, num_samples, arrival_rate=5.0, avg_holding_time=5.0, preload=500,
                 include_edge_features=False, include_link_as_node=False, include_pinn=False):
        self.net.reset()
        active = []
        t = 0.0

        # Pre-load network with random connections to create realistic fragmentation
        for _ in range(preload):
            src, dst, part, bw = self._random_request()
            success, path, start_slot, delay = self.mapper.map(src, dst, bw)
            if success:
                ht = self.rng.exponential(avg_holding_time * 2)
                active.append((path, start_slot, bw, t + ht))

        # Precompute edge index (topology is static)
        edge_index = self._get_edge_index() if include_edge_features else None

        # Precompute link-as-node structures if needed
        if include_link_as_node:
            self._ensure_link_as_node_structure()

        # Precompute PINN structures if needed
        if include_pinn:
            self._ensure_pinn_structure()

        # Now advance time and sample
        samples = []
        while len(samples) < num_samples:
            t += self.rng.exponential(1.0 / arrival_rate)
            new_active = []
            for path, start, bw, rt in active:
                if rt <= t:
                    self.net.release(path, start, bw)
                else:
                    new_active.append((path, start, bw, rt))
            active = new_active

            src, dst, path_id, bw = self._random_request()
            z = self.encoder.encode(src, dst)

            # FIX: test ONLY the specified path_id for correct per-path labels
            result = self.mapper.execute_path(src, dst, bw, path_id)
            success = result["success"]
            path = result["path"] if success else None
            start_slot = result["start_slot"] if success else None
            delay = result["delay"] if success else 0.0

            sample = {
                "src": src, "dst": dst, "path_id": path_id, "bw": float(bw),
                "z": z, "success": 1.0 if success else 0.0,
                "delay": delay,
            }
            if include_edge_features:
                sample["edge_features"] = self._get_edge_features()
                sample["edge_index"] = edge_index
                sample["num_nodes"] = self.num_servers
            if include_link_as_node:
                sample["link_node_features"] = compute_link_node_features(self.net, self._link_index_map)
                sample["converted_edge_index"] = self._converted_edge_index
                sample["path_mask"] = self._path_masks.get((src, dst, path_id),
                                                           np.zeros(len(self._link_index_map), dtype=np.float32))
            if include_pinn:
                sample["S_current"] = self._get_S_current()
                sample["path_link_mask"] = self._pinn_path_link_masks.get(
                    (src, dst, path_id),
                    np.zeros(self._num_links, dtype=np.float32)
                )
                sample["start_slot"] = int(start_slot) if start_slot is not None else -1
            samples.append(sample)

            if success:
                ht = self.rng.exponential(avg_holding_time)
                active.append((path, start_slot, bw, t + ht))

        for path, start, bw, _ in active:
            self.net.release(path, start, bw)
        return samples
