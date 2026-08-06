"""Compact replay buffer for PDS/PreD RMSA training."""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import numpy as np

from sa_hmarl.pds_rmsa.env.reservations import Reservation, TrueState
from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, RMSAAction
from sa_hmarl.pds_rmsa.protocol import Request


def _pack_bitmap(bitmap: np.ndarray) -> np.ndarray:
    flat = np.asarray(bitmap, dtype=bool).flatten()
    return np.packbits(flat)


def _unpack_bitmap(
    packed: np.ndarray, num_directed_links: int, num_slots: int
) -> np.ndarray:
    flat = np.unpackbits(np.asarray(packed, dtype=np.uint8))
    needed = num_directed_links * num_slots
    flat = flat[:needed]
    return flat.reshape(num_directed_links, num_slots).astype(bool)


def _reservations_to_tuple(
    reservations: Union[Tuple[Reservation, ...], List[Reservation]],
) -> Tuple[Tuple[int, Tuple[int, ...], int, int, float], ...]:
    return tuple(
        (r.request_id, tuple(r.path), r.start_slot, r.required_fs, float(r.release_time))
        for r in reservations
    )


def _tuple_to_reservations(
    tuples: Tuple[Tuple[int, Tuple[int, ...], int, int, float], ...]
) -> Tuple[Reservation, ...]:
    return tuple(
        Reservation(
            request_id=r[0],
            path=r[1],
            start_slot=r[2],
            required_fs=r[3],
            release_time=r[4],
        )
        for r in tuples
    )


def _request_to_tuple(request: Request) -> Tuple[int, int, int, int, float, float]:
    return (
        request.request_id,
        request.src_node,
        request.dst_node,
        request.bitrate_gbps,
        float(request.arrival_time),
        float(request.holding_time),
    )


def _tuple_to_request(t: Tuple[int, int, int, int, float, float]) -> Request:
    return Request(
        request_id=t[0],
        src_node=t[1],
        dst_node=t[2],
        bitrate_gbps=t[3],
        arrival_time=t[4],
        holding_time=t[5],
    )


def _action_to_tuple(action: Union[RMSAAction, BlockedAction]) -> Optional[Tuple[int, str, int, int, Tuple[int, ...]]]:
    if isinstance(action, RMSAAction):
        return (action.path_rank, action.modulation, action.start_slot, action.required_fs, tuple(action.path))
    return None


def _tuple_to_action(
    t: Optional[Tuple[int, str, int, int, Tuple[int, ...]]]
) -> Union[RMSAAction, BlockedAction]:
    if t is None:
        return BlockedAction()
    return RMSAAction(
        action_id=t[:4],
        path_rank=t[0],
        path=t[4],
        modulation=t[1],
        start_slot=t[2],
        required_fs=t[3],
    )


@dataclass
class CompactState:
    """Compact pre-state or next-pre-state."""
    bitmap_packed: np.ndarray
    reservations_tuple: Tuple[Tuple[int, Tuple[int, ...], int, int, float], ...]
    time: float

    def to_true_state(self, num_directed_links: int, num_slots: int) -> TrueState:
        bitmap = _unpack_bitmap(self.bitmap_packed, num_directed_links, num_slots)
        reservations = _tuple_to_reservations(self.reservations_tuple)
        return TrueState(time=self.time, bitmap=bitmap, reservations=reservations)


@dataclass
class CompactTransition:
    pre: CompactState
    action_tuple: Optional[Tuple[int, str, int, int, Tuple[int, ...]]]
    reward: float
    next_pre: CompactState
    done: bool
    request_tuple: Tuple[int, int, int, int, float, float]
    next_request_tuple: Tuple[int, int, int, int, float, float]

    def action(self) -> Union[RMSAAction, BlockedAction]:
        return _tuple_to_action(self.action_tuple)

    def request(self) -> Request:
        return _tuple_to_request(self.request_tuple)

    def next_request(self) -> Request:
        return _tuple_to_request(self.next_request_tuple)


class ReplayBuffer:
    """Replay buffer with compact bitmap storage and full reconstruction on sample."""

    def __init__(
        self,
        capacity: int = 100000,
        num_directed_links: int = 52,
        num_slots: int = 50,
    ):
        self.capacity = int(capacity)
        self.num_directed_links = int(num_directed_links)
        self.num_slots = int(num_slots)

        self._pre_bitmaps: List[np.ndarray] = []
        self._pre_reservations: List[Tuple] = []
        self._pre_times: List[float] = []
        self._actions: List[Optional[Tuple[int, str, int, int, Tuple[int, ...]]]] = []
        self._rewards: List[float] = []
        self._next_bitmaps: List[np.ndarray] = []
        self._next_reservations: List[Tuple] = []
        self._next_times: List[float] = []
        self._dones: List[bool] = []
        self._requests: List[Tuple] = []
        self._next_requests: List[Tuple] = []

        self._size = 0
        self._pos = 0

    def add(
        self,
        pre_state: TrueState,
        action: Union[RMSAAction, BlockedAction],
        reward: float,
        next_pre_state: TrueState,
        done: bool,
        request: Request,
        next_request: Request,
    ) -> None:
        """Add a compact transition."""
        pre = CompactState(
            bitmap_packed=_pack_bitmap(pre_state.bitmap),
            reservations_tuple=_reservations_to_tuple(pre_state.reservations),
            time=float(pre_state.time),
        )
        next_pre = CompactState(
            bitmap_packed=_pack_bitmap(next_pre_state.bitmap),
            reservations_tuple=_reservations_to_tuple(next_pre_state.reservations),
            time=float(next_pre_state.time),
        )
        transition = CompactTransition(
            pre=pre,
            action_tuple=_action_to_tuple(action),
            reward=float(reward),
            next_pre=next_pre,
            done=bool(done),
            request_tuple=_request_to_tuple(request),
            next_request_tuple=_request_to_tuple(next_request),
        )

        if self._size < self.capacity:
            self._pre_bitmaps.append(transition.pre.bitmap_packed)
            self._pre_reservations.append(transition.pre.reservations_tuple)
            self._pre_times.append(transition.pre.time)
            self._actions.append(transition.action_tuple)
            self._rewards.append(transition.reward)
            self._next_bitmaps.append(transition.next_pre.bitmap_packed)
            self._next_reservations.append(transition.next_pre.reservations_tuple)
            self._next_times.append(transition.next_pre.time)
            self._dones.append(transition.done)
            self._requests.append(transition.request_tuple)
            self._next_requests.append(transition.next_request_tuple)
        else:
            idx = self._pos
            self._pre_bitmaps[idx] = transition.pre.bitmap_packed
            self._pre_reservations[idx] = transition.pre.reservations_tuple
            self._pre_times[idx] = transition.pre.time
            self._actions[idx] = transition.action_tuple
            self._rewards[idx] = transition.reward
            self._next_bitmaps[idx] = transition.next_pre.bitmap_packed
            self._next_reservations[idx] = transition.next_pre.reservations_tuple
            self._next_times[idx] = transition.next_pre.time
            self._dones[idx] = transition.done
            self._requests[idx] = transition.request_tuple
            self._next_requests[idx] = transition.next_request_tuple

        self._pos = (self._pos + 1) % self.capacity
        self._size = min(self._size + 1, self.capacity)

    def __len__(self) -> int:
        return self._size

    def _get(self, idx: int) -> CompactTransition:
        return CompactTransition(
            pre=CompactState(
                bitmap_packed=self._pre_bitmaps[idx],
                reservations_tuple=self._pre_reservations[idx],
                time=self._pre_times[idx],
            ),
            action_tuple=self._actions[idx],
            reward=self._rewards[idx],
            next_pre=CompactState(
                bitmap_packed=self._next_bitmaps[idx],
                reservations_tuple=self._next_reservations[idx],
                time=self._next_times[idx],
            ),
            done=self._dones[idx],
            request_tuple=self._requests[idx],
            next_request_tuple=self._next_requests[idx],
        )

    def sample(self, batch_size: int) -> List[CompactTransition]:
        """Sample ``batch_size`` transitions uniformly at random."""
        if self._size == 0:
            return []
        batch_size = min(int(batch_size), self._size)
        indices = np.random.randint(0, self._size, size=batch_size)
        return [self._get(int(i)) for i in indices]

    def sample_to_states(self, batch_size: int):
        """Convenience: sample and return reconstructed TrueState objects."""
        transitions = self.sample(batch_size)
        return [
            {
                "pre": t.pre.to_true_state(self.num_directed_links, self.num_slots),
                "action": t.action(),
                "reward": t.reward,
                "next_pre": t.next_pre.to_true_state(self.num_directed_links, self.num_slots),
                "done": t.done,
                "request": t.request(),
                "next_request": t.next_request(),
            }
            for t in transitions
        ]
