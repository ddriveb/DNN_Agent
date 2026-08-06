"""K-shortest-path first-fit baseline for RMSA."""
from __future__ import annotations

from typing import Union

from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, PDSRMSAEnv, RMSAAction
from sa_hmarl.pds_rmsa.protocol import Request


def ksp_ff_action(env: PDSRMSAEnv, request: Request) -> Union[RMSAAction, BlockedAction]:
    """Return the first available RMSAAction, or BlockedAction if none exist."""
    candidates = env.build_candidates(request)
    for action in candidates:
        if isinstance(action, RMSAAction):
            return action
    return candidates[0]
