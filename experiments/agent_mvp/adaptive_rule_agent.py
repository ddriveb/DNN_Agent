"""Adaptive Rule Agent: load-aware strategy selection.

Selects predictor + score weights + pruning strategy based on real-time network load.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "predictor_mvp"))

import numpy as np
import torch
from typing import Tuple, Dict, Optional

from rule_agent import RuleAgent
from pruned_rule_agent import PrunedRuleAgent
from traffic_generator import DNNRequest


class AdaptiveRuleAgent:
    """Selects the best RuleAgent strategy based on load level.

    Load level can be inferred from:
      - recent acceptance rate (sliding window)
      - average link utilization
      - arrival rate (if known)
    """

    def __init__(self,
                 predictors: Dict[str, object],
                 encoders: Dict[str, object],
                 mec_cluster,
                 history_window: int = 100,
                 device: str = "cpu"):
        """
        Args:
            predictors: dict of {name: predictor_model}
            encoders: dict of {name: encoder_instance} (one per predictor)
            mec_cluster: MECCluster instance
            history_window: number of recent requests to track for load estimation
        """
        self.predictors = predictors
        self.encoders = encoders
        self.mec = mec_cluster
        self.history_window = history_window
        self.device = device

        # Recent outcome history: list of (success_bool, blocking_reason_or_none)
        self.outcome_history = []

        # Sub-agents for each strategy
        self._sub_agents = {}
        self._build_sub_agents()

    def _build_sub_agents(self):
        """Pre-build all strategy variants."""
        strategies = [
            ("default+mixed", "mixed-v2b", 1.0, 0.5, 0.3, 0.1, False),
            ("default+nsfnet", "nsfnet-v2b", 1.0, 0.5, 0.3, 0.1, False),
            ("tuned+mixed", "mixed-v2b", 2.0, 0.3, 0.2, 0.05, False),
            ("tuned+nsfnet", "nsfnet-v2b", 2.0, 0.3, 0.2, 0.05, False),
            ("pruned+mixed", "mixed-v2b", 2.0, 0.3, 0.2, 0.05, True),
            ("pruned+nsfnet", "nsfnet-v2b", 2.0, 0.3, 0.2, 0.05, True),
        ]

        for name, pname, alpha, beta, gamma, delta, use_prune in strategies:
            predictor = self.predictors.get(pname)
            encoder = self.encoders.get(pname)
            if predictor is None or encoder is None:
                continue
            if use_prune:
                agent = PrunedRuleAgent(
                    predictor, encoder, self.mec,
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                    prune_deadline_ms=60, prune_server_util=0.95,
                    prune_bw_threshold=8, prune_min_p_success=0.3,
                    device=self.device,
                )
            else:
                agent = RuleAgent(
                    predictor, encoder, self.mec,
                    alpha=alpha, beta=beta, gamma=gamma, delta=delta,
                    device=self.device,
                )
            self._sub_agents[name] = agent

    def _estimate_load_level(self) -> str:
        """Estimate current load level from recent history.

        Returns one of: 'low', 'medium', 'high'
        """
        if len(self.outcome_history) < 20:
            return "medium"  # Not enough data yet

        recent = self.outcome_history[-self.history_window:]
        recent_accept_rate = np.mean([1.0 if success else 0.0 for success, _ in recent])

        # Also estimate from link utilization if available
        # (not directly accessible here, but acceptance rate is a good proxy)

        if recent_accept_rate >= 0.85:
            return "low"
        elif recent_accept_rate >= 0.70:
            return "medium"
        else:
            return "high"

    def select_strategy(self, load_level: str) -> str:
        """Select strategy name based on load level.

        Based on experimental results:
          - low load:    default + nsfnet (sharp predictor, avoid over-pruning)
          - medium load: pruned + mixed (balance between predictor confidence and action space)
          - high load:   default + nsfnet (don't prune, need all options)
        """
        if load_level == "low":
            return "default+nsfnet"
        elif load_level == "medium":
            return "pruned+mixed"
        else:  # high
            return "default+nsfnet"

    def decide(self, request: DNNRequest, network_state) -> Tuple[int, int, float, Dict]:
        """Select action using the load-adaptive strategy."""
        load_level = self._estimate_load_level()
        strategy_name = self.select_strategy(load_level)
        agent = self._sub_agents.get(strategy_name)

        if agent is None:
            # Fallback to first available
            strategy_name = list(self._sub_agents.keys())[0]
            agent = self._sub_agents[strategy_name]

        split_id, server_id, score, info = agent.decide(request, network_state)

        info.update({
            "adaptive_strategy": strategy_name,
            "load_level": load_level,
            "recent_accept_rate": np.mean([1.0 if s else 0.0 for s, _ in self.outcome_history[-self.history_window:]]) if self.outcome_history else 1.0,
        })
        return split_id, server_id, score, info

    def update_history(self, success: bool, blocking_reason: Optional[str] = None):
        """Call after each request to update load estimation."""
        self.outcome_history.append((success, blocking_reason))
        if len(self.outcome_history) > self.history_window * 2:
            # Trim to avoid unbounded growth
            self.outcome_history = self.outcome_history[-self.history_window:]

    def reset(self):
        """Reset history for a new episode."""
        self.outcome_history = []
