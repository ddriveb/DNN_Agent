"""Hybrid Encoder: outputs both v2b histogram and GAT structural inputs."""
import numpy as np
from encoder import Encoder
from link_as_node_gnn_predictor import build_converted_topology, compute_link_node_features


class HybridEncoder(Encoder):
    """
    Extends base Encoder to also precompute and output GAT structural data.

    Usage:
        encoder = HybridEncoder(network, k=3)
        out = encoder.encode_full(src, dst)
        # out["z"] → v2b histogram (10-dim)
        # out["link_node_features"] → (num_links, 6)
        # out["converted_edge_index"] → (num_converted_edges, 2)
        # out["path_masks"] → list of 3 binary masks
    """

    def __init__(self, network, k=3):
        super().__init__(network, k)
        self._link_index_map, self._converted_edge_index = build_converted_topology(network)
        self._num_links = len(self._link_index_map)
        self._precompute_path_masks()
        # Default encode method for DatasetGenerator compatibility
        self.encode = self.encode_v2b

    def _precompute_path_masks(self):
        """Precompute binary path masks for all (src, dst, path_id)."""
        self._path_masks = {}
        for src in range(self.net.NUM_NODES):
            for dst in range(self.net.NUM_NODES):
                if src == dst:
                    continue
                paths = self._path_cache.get((src, dst), [])
                for pid, path in enumerate(paths):
                    mask = np.zeros(self._num_links, dtype=np.float32)
                    for j in range(len(path) - 1):
                        link = (min(path[j], path[j + 1]), max(path[j], path[j + 1]))
                        if link in self._link_index_map:
                            mask[self._link_index_map[link]] = 1.0
                    self._path_masks[(src, dst, pid)] = mask

    def encode_full(self, src, dst):
        """
        Encode full state for Hybrid Predictor.
        Returns dict with keys: z, link_node_features, converted_edge_index, path_masks
        """
        z = self.encode_v2b(src, dst)
        link_feats = compute_link_node_features(self.net, self._link_index_map)
        path_masks = [
            self._path_masks.get((src, dst, i),
                                 np.zeros(self._num_links, dtype=np.float32))
            for i in range(self.k)
        ]
        return {
            "z": z,
            "link_node_features": link_feats,
            "converted_edge_index": self._converted_edge_index,
            "path_masks": path_masks,
        }
