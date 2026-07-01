"""
Optical Network Environment (MVP)
Supports multiple standard topologies + large-scale random graphs.
"""
import numpy as np
import networkx as nx


class OpticalNetwork:
    """Elastic Optical Network with FS allocation."""

    NUM_SLOTS = 128

    # NSFNET 14-node, 21-link
    NSFNET_EDGES = [
        (0, 1), (0, 2), (0, 5), (1, 2), (1, 3), (2, 4), (2, 8),
        (3, 4), (3, 10), (4, 5), (4, 6), (5, 7), (6, 7), (6, 13),
        (7, 8), (8, 9), (9, 10), (9, 12), (10, 11), (11, 12), (12, 13),
    ]

    # USNET 28-node, 45-link
    USNET_EDGES = [
        (0,1),(0,2),(1,2),(1,3),(2,4),(3,4),(3,5),(4,6),(5,6),(5,7),
        (6,8),(7,8),(7,9),(8,10),(9,10),(9,11),(10,12),(11,12),(11,13),
        (12,14),(13,14),(13,15),(14,16),(15,16),(15,17),(16,18),(17,18),
        (17,19),(18,20),(19,20),(19,21),(20,22),(21,22),(21,23),(22,24),
        (23,24),(23,25),(24,26),(25,26),(25,27),(26,27),(0,24),(3,20),(5,22),(7,26)
    ]

    # COST266 European network (28 nodes, 41 links)
    COST266_EDGES = [
        (0,1),(0,16),(1,2),(1,16),(2,3),(2,20),(3,4),(3,6),(4,5),(4,20),
        (5,8),(5,20),(6,7),(6,26),(7,21),(8,9),(8,10),(9,12),(10,11),(10,12),
        (11,13),(11,17),(12,13),(13,22),(14,15),(14,16),(15,24),(16,17),(17,18),
        (18,19),(18,20),(19,20),(21,22),(21,23),(22,23),(23,24),(25,26),(25,27),
        (26,27),(3,25),(7,25)
    ]

    # Germany50 (50 nodes, 88 links)
    GERMANY50_EDGES = [
        (0,1),(0,5),(1,2),(1,5),(2,3),(2,7),(3,4),(3,8),(4,9),(5,6),
        (5,10),(6,7),(6,11),(7,8),(7,12),(8,9),(8,13),(9,14),(10,11),(10,15),
        (11,12),(11,16),(12,13),(12,17),(13,14),(13,18),(14,19),(15,16),(15,20),
        (16,17),(16,21),(17,18),(17,22),(18,19),(18,23),(19,24),(20,21),(20,25),
        (21,22),(21,26),(22,23),(22,27),(23,24),(23,28),(24,29),(25,26),(25,30),
        (26,27),(26,31),(27,28),(27,32),(28,29),(28,33),(29,34),(30,31),(30,35),
        (31,32),(31,36),(32,33),(32,37),(33,34),(33,38),(34,39),(35,36),(35,40),
        (36,37),(36,41),(37,38),(37,42),(38,39),(38,43),(39,44),(40,41),(40,45),
        (41,42),(41,46),(42,43),(42,47),(43,44),(43,48),(44,49),(45,46),(46,47),
        (47,48),(48,49)
    ]

    def __init__(self, topology="nsfnet", num_slots=32, seed=None):
        self.topology = topology
        self.num_slots = num_slots
        self.rng = np.random.RandomState(seed)
        self.G = nx.Graph()
        self._build_topology()
        self.NUM_NODES = self.G.number_of_nodes()
        self.reset()

    def _build_waxman(self, n, alpha=0.4, beta=0.1):
        """Generate connected Waxman random graph."""
        pos = {i: (self.rng.rand(), self.rng.rand()) for i in range(n)}
        edges = []
        for i in range(n):
            for j in range(i+1, n):
                d = np.sqrt((pos[i][0]-pos[j][0])**2 + (pos[i][1]-pos[j][1])**2)
                prob = beta * np.exp(-d / (alpha * np.sqrt(2)))
                if self.rng.rand() < prob:
                    edges.append((i, j))
        G_tmp = nx.Graph()
        G_tmp.add_nodes_from(range(n))
        G_tmp.add_edges_from(edges)
        if not nx.is_connected(G_tmp):
            comps = list(nx.connected_components(G_tmp))
            for i in range(len(comps)-1):
                u = list(comps[i])[0]
                v = list(comps[i+1])[0]
                edges.append((u, v))
        return edges

    def _build_topology(self):
        if self.topology == "nsfnet":
            edges = self.NSFNET_EDGES
        elif self.topology == "usnet":
            edges = self.USNET_EDGES
        elif self.topology == "cost266":
            edges = self.COST266_EDGES
        elif self.topology == "germany50":
            edges = self.GERMANY50_EDGES
        elif self.topology == "random50":
            edges = self._build_waxman(50)
        elif self.topology == "random80":
            edges = self._build_waxman(80, alpha=0.35, beta=0.08)
        elif self.topology == "random100":
            edges = self._build_waxman(100, alpha=0.3, beta=0.06)
        else:
            raise ValueError(f"Unknown topology: {self.topology}")

        for u, v in edges:
            dist = float(self.rng.uniform(500, 3000))
            self.G.add_edge(u, v, length=dist, weight=1.0)

    def reset(self):
        self.link_states = {}
        for u, v in self.G.edges():
            key = (min(u, v), max(u, v))
            self.link_states[key] = np.zeros(self.num_slots, dtype=bool)

    def allocate(self, path, start_slot, num_slots):
        if start_slot < 0 or start_slot + num_slots > self.num_slots:
            return False
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            if np.any(self.link_states[link][start_slot:start_slot + num_slots]):
                return False
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            self.link_states[link][start_slot:start_slot + num_slots] = True
        return True

    def release(self, path, start_slot, num_slots):
        if path is None or start_slot is None:
            return
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            self.link_states[link][start_slot:start_slot + num_slots] = False

    def get_available_slots(self, path):
        avail = np.ones(self.num_slots, dtype=bool)
        for i in range(len(path) - 1):
            link = (min(path[i], path[i + 1]), max(path[i], path[i + 1]))
            avail &= ~self.link_states[link]
        return avail

    @staticmethod
    def _max_consecutive(arr) -> int:
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

    @staticmethod
    def _count_blocks(arr) -> int:
        count = 0
        in_block = False
        for v in arr:
            if v and not in_block:
                count += 1
                in_block = True
            elif not v:
                in_block = False
        return count

    def summarize_availability(self, free_arr):
        total_free = int(np.sum(free_arr))
        max_free = self._max_consecutive(free_arr)
        free_blocks = self._count_blocks(free_arr)
        if total_free == 0:
            frag = 1.0
        elif total_free == self.num_slots:
            frag = 0.0
        else:
            frag = 1.0 - (max_free / total_free)
        return {
            "total_free_slots": total_free,
            "max_free_slots": max_free,
            "largest_free_block_ratio": max_free / max(self.num_slots, 1),
            "free_block_count": free_blocks,
            "frag_index": float(frag),
        }

    def get_path_spectrum_stats(self, path):
        free_arr = self.get_available_slots(path)
        stats = self.summarize_availability(free_arr)
        stats["path_length"] = max(len(path) - 1, 0)
        return stats

    def get_global_spectrum_stats(self):
        total_slots = len(self.link_states) * self.num_slots
        occupied = 0
        frag_vals = []
        max_free_vals = []
        free_block_vals = []

        for slots in self.link_states.values():
            occupied += int(np.sum(slots))
            free_arr = ~slots
            stats = self.summarize_availability(free_arr)
            frag_vals.append(stats["frag_index"])
            max_free_vals.append(stats["max_free_slots"])
            free_block_vals.append(stats["free_block_count"])

        return {
            "spectrum_utilization": occupied / max(total_slots, 1),
            "avg_frag_index": float(np.mean(frag_vals)) if frag_vals else 0.0,
            "largest_free_block_ratio": (max(max_free_vals) / max(self.num_slots, 1)) if max_free_vals else 0.0,
            "avg_free_block_count": float(np.mean(free_block_vals)) if free_block_vals else 0.0,
            "num_links": len(self.link_states),
        }
