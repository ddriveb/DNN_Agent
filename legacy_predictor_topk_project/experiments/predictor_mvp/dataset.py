"""Event-driven dataset generation."""
import numpy as np


class DatasetGenerator:
    def __init__(self, network, mapper, encoder, num_partitions=3, seed=None):
        self.net = network
        self.mapper = mapper
        self.encoder = encoder
        self.num_partitions = num_partitions
        self.num_servers = network.NUM_NODES
        self.rng = np.random.RandomState(seed)
        self.bandwidth_map = {0: 1, 1: 4, 2: 8}

    def _random_request(self):
        src = self.rng.randint(0, self.num_servers)
        dst = self.rng.randint(0, self.num_servers)
        while dst == src:
            dst = self.rng.randint(0, self.num_servers)
        partition = self.rng.randint(0, self.num_partitions)
        return src, dst, partition, self.bandwidth_map[partition]

    def generate(self, num_samples, arrival_rate=5.0, avg_holding_time=5.0, preload=500):
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

            src, dst, partition, bw = self._random_request()
            z = self.encoder.encode(src, dst)
            success, path, start_slot, delay = self.mapper.map(src, dst, bw)

            samples.append({
                "src": src, "dst": dst, "partition": partition,
                "z": z, "success": 1.0 if success else 0.0,
                "delay": delay if success else 0.0,
            })

            if success:
                ht = self.rng.exponential(avg_holding_time)
                active.append((path, start_slot, bw, t + ht))

        for path, start, bw, _ in active:
            self.net.release(path, start, bw)
        return samples
