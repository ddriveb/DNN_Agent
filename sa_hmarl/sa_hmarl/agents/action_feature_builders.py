"""Neutral feature-builder mixins for PPO actors and diagnostics.

These aliases let active PPO code depend on feature-construction names that do
not carry the old DQN policy semantics.
"""
from __future__ import annotations

from sa_hmarl.agents.c_agent import AgentC as _LegacyAgentC
from sa_hmarl.agents.r_agent import AgentR as _LegacyAgentR


class AgentCFeatureBuilder:
    """Compatibility mixin exposing Agent-C candidate feature construction."""

    build_action_features = _LegacyAgentC.build_action_features
    compute_r_feasibility_diagnostics = staticmethod(
        _LegacyAgentC.compute_r_feasibility_diagnostics
    )
    _build_candidate_mean_field_features = staticmethod(
        _LegacyAgentC._build_candidate_mean_field_features
    )
    _build_r_feasibility_features = staticmethod(
        _LegacyAgentC._build_r_feasibility_features
    )
    _build_spectrum_impact_features = staticmethod(
        _LegacyAgentC._build_spectrum_impact_features
    )
    _build_server_margin_features = staticmethod(
        _LegacyAgentC._build_server_margin_features
    )
    _build_mr_feasibility_rule_features = staticmethod(
        _LegacyAgentC._build_mr_feasibility_rule_features
    )
    _build_cross_pressure_features = staticmethod(
        _LegacyAgentC._build_cross_pressure_features
    )
    _build_pressure_aware_features = staticmethod(
        _LegacyAgentC._build_pressure_aware_features
    )
    _build_overload_aware_features = staticmethod(
        _LegacyAgentC._build_overload_aware_features
    )
    _build_enhanced_features = staticmethod(
        _LegacyAgentC._build_enhanced_features
    )


class AgentRFeatureBuilder:
    """Compatibility mixin exposing Agent-R candidate feature construction."""

    build_action_features = _LegacyAgentR.build_action_features
    _build_c_aware_tail = staticmethod(_LegacyAgentR._build_c_aware_tail)
