"""
AgentInterface: Bridge between DNN offloading decisions and optical feasibility prediction.

Translates high-level Agent actions a^{high} = (split_point, target_server, collab_mode)
into low-level predictor inputs (path_id, src, dst, bw, z).
"""
from selector import Selector


class AgentInterface:
    """
    Interface that wraps the Predictor + Selector for DNN offloading decisions.

    Usage:
        iface = AgentInterface(predictor, encoder)
        result = iface.evaluate_action(
            split_point=2, target_server=10, source_server=0
        )
        # result contains: best_path_id, success_prob, delay, all path candidates
    """

    def __init__(self, predictor, encoder, selector_strategy="max_prob"):
        self.predictor = predictor
        self.encoder = encoder
        self.selector = Selector(predictor, encoder, strategy=selector_strategy)

    # ------------------------------------------------------------------
    # DNN → Optical mapping functions
    # ------------------------------------------------------------------

    @staticmethod
    def data_size_after_split(split_point):
        """
        Map DNN split point to required bandwidth (slots).
        More layers offloaded → larger intermediate activations.
        """
        # Example: split after layer N → N feature maps traverse network
        mapping = {
            0: 1,   # No offload (local inference)
            1: 2,   # Split after layer 1
            2: 4,   # Split after layer 2
            3: 8,   # Split after layer 3
            4: 16,  # Split after layer 4
            5: 32,  # Split after layer 5 (heavy offload)
        }
        return mapping.get(split_point, 4)

    @staticmethod
    def get_src_dst(target_server, source_server=0):
        """
        In DNN offloading, src is the current inference node,
        dst is the target server where remaining layers execute.
        """
        return source_server, target_server

    # ------------------------------------------------------------------
    # Action evaluation
    # ------------------------------------------------------------------

    def evaluate_action(self, split_point, target_server, source_server=0,
                        strategy=None):
        """
        Evaluate a single DNN offload candidate action.

        Args:
            split_point: int, where to split the DNN (0=local, 5=all offloaded)
            target_server: int, destination server ID
            source_server: int, source node ID (default 0)
            strategy: override selector strategy for this call

        Returns:
            dict with keys:
                - split_point, target_server, source_server
                - bw_slots: bandwidth requirement
                - best_path_id: selected KSP path
                - success_prob: P(success | best_path)
                - delay: estimated delay (ms)
                - all_paths: list of evaluations for all 3 paths
        """
        bw = self.data_size_after_split(split_point)
        src, dst = self.get_src_dst(target_server, source_server)

        if strategy is not None:
            old_strategy = self.selector.strategy
            self.selector.strategy = strategy
            best, candidates = self.selector.select(src, dst, bw)
            self.selector.strategy = old_strategy
        else:
            best, candidates = self.selector.select(src, dst, bw)

        return {
            "split_point": split_point,
            "target_server": target_server,
            "source_server": source_server,
            "bw_slots": bw,
            "best_path_id": best["path_id"],
            "success_prob": best["success_prob"],
            "delay": best["delay"],
            "all_paths": candidates,
        }

    def batch_evaluate(self, actions):
        """
        Evaluate multiple actions in batch.

        Args:
            actions: list of dicts
                [{"split_point": 2, "target_server": 10}, ...]

        Returns:
            list of result dicts
        """
        return [
            self.evaluate_action(
                a.get("split_point", 0),
                a.get("target_server", 0),
                a.get("source_server", 0),
                a.get("strategy", None)
            )
            for a in actions
        ]

    # ------------------------------------------------------------------
    # Decision helpers
    # ------------------------------------------------------------------

    def select_best_action(self, candidate_actions):
        """
        Given multiple candidate (split_point, target_server) pairs,
        select the one with highest success probability.

        Args:
            candidate_actions: list of dicts
                [{"split_point": 2, "target_server": 10}, ...]

        Returns:
            best_result dict
        """
        results = self.batch_evaluate(candidate_actions)
        best = max(results, key=lambda r: r["success_prob"])
        return best
