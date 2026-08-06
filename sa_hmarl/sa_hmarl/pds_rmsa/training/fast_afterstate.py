"""Functional afterstate computation without modifying a live environment."""
from __future__ import annotations

from typing import Tuple, Union

import numpy as np

from sa_hmarl.pds_rmsa.env.reservations import PostDecisionState, Reservation
from sa_hmarl.pds_rmsa.env.topology import Topology, load_topology
from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, RMSAAction
from sa_hmarl.pds_rmsa.features.afterstate import _directed_link_ids
from sa_hmarl.pds_rmsa.protocol import Request


def fast_afterstate(
    pre_bitmap: np.ndarray,
    pre_reservations: Tuple[Reservation, ...],
    current_time: float,
    request: Request,
    action: Union[RMSAAction, BlockedAction],
    topology: Union[str, Topology],
) -> Tuple[np.ndarray, Tuple[Reservation, ...], float]:
    """Compute the post-decision-state (bitmap, reservations, time) from a pre-state.

    This mirrors the effect of ``env.advance_external(request)`` followed by
    ``env.step(action)`` without mutating any live environment.

    Parameters
    ----------
    pre_bitmap
        Occupancy bitmap of shape (num_directed_links, num_slots). The caller
        must pass a state already advanced to the request's arrival time (i.e.
        expired reservations released).
    pre_reservations
        Sorted tuple of active reservations at ``current_time``.
    current_time
        The time at which the request arrives (after any releases).
    request
        The request to serve.
    action
        The RMSA action to apply, or ``BlockedAction``.
    topology
        Topology name or a ``Topology`` instance used to map path arcs to bitmap
        rows.

    Returns
    -------
    post_bitmap, post_reservations, post_time
    """
    if isinstance(topology, str):
        topology = load_topology(topology)

    if isinstance(action, BlockedAction):
        return pre_bitmap.copy(), tuple(pre_reservations), float(current_time)

    link_ids = _directed_link_ids(topology)
    post_bitmap = np.asarray(pre_bitmap, dtype=bool).copy()
    arcs = [(action.path[i], action.path[i + 1]) for i in range(len(action.path) - 1)]
    start = int(action.start_slot)
    end = start + int(action.required_fs)

    for arc in arcs:
        row = link_ids.index(arc)
        if np.any(post_bitmap[row, start:end]):
            raise ValueError(
                f"Illegal action: slot range [{start},{end}) is not free on arc {arc}"
            )

    for arc in arcs:
        row = link_ids.index(arc)
        post_bitmap[row, start:end] = True

    release_time = float(current_time) + float(request.holding_time)
    new_reservation = Reservation(
        request_id=int(request.request_id),
        path=tuple(int(n) for n in action.path),
        start_slot=int(action.start_slot),
        required_fs=int(action.required_fs),
        release_time=release_time,
    )
    post_reservations = tuple(sorted(
        tuple(pre_reservations) + (new_reservation,),
        key=lambda r: (r.release_time, r.request_id, r.start_slot, r.path),
    ))
    return post_bitmap, post_reservations, float(current_time)


def make_post_state(
    pre_bitmap: np.ndarray,
    pre_reservations: Tuple[Reservation, ...],
    current_time: float,
    request: Request,
    action: Union[RMSAAction, BlockedAction],
    topology: Union[str, Topology],
) -> PostDecisionState:
    """Build a ``PostDecisionState`` from a pre-state and action via fast_afterstate."""
    post_bitmap, post_reservations, post_time = fast_afterstate(
        pre_bitmap, pre_reservations, current_time, request, action, topology
    )
    admitted = isinstance(action, RMSAAction)
    return PostDecisionState(
        time=post_time,
        bitmap=post_bitmap,
        reservations=post_reservations,
        admitted_this_step=admitted,
        allocated_action=action if admitted else None,
    )
