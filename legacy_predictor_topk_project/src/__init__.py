"""Predictor-Guided Top-K Correction Agent for DNN Offloading.

This package provides the paper-version implementation.
Core simulation modules are in experiments/predictor_mvp/ and experiments/agent_mvp/.
This __init__ ensures they are discoverable.
"""
import sys
from pathlib import Path

_PROJECT_ROOT = Path(__file__).parent.parent
_PREDICTOR_MVP = _PROJECT_ROOT / "experiments" / "predictor_mvp"
_AGENT_MVP = _PROJECT_ROOT / "experiments" / "agent_mvp"

if str(_PREDICTOR_MVP) not in sys.path:
    sys.path.insert(0, str(_PREDICTOR_MVP))
if str(_AGENT_MVP) not in sys.path:
    sys.path.insert(0, str(_AGENT_MVP))
