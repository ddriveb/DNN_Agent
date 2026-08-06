"""PDS-MLP and PreD-DQN agents."""
from __future__ import annotations

from typing import List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn.functional as F

from sa_hmarl.pds_rmsa.env.rmsa_env import BlockedAction, PDSRMSAEnv, RMSAAction
from sa_hmarl.pds_rmsa.env.reservations import TrueState
from sa_hmarl.pds_rmsa.env.topology import Topology, load_topology
from sa_hmarl.pds_rmsa.features.afterstate import _directed_link_ids
from sa_hmarl.pds_rmsa.networks.mlp import PDSMLP, PreDMLP, count_parameters
from sa_hmarl.pds_rmsa.protocol import (
    K_PATHS,
    MAX_BLOCKS_PER_PATH,
    MODULATION_TABLE,
    Request,
    highest_modulation_for_distance,
    required_fs,
)
from sa_hmarl.pds_rmsa.training.fast_afterstate import fast_afterstate, make_post_state
from sa_hmarl.pds_rmsa.training.pred_features import (
    build_pds_features,
    build_pds_features_batch,
    build_pred_features,
    build_pred_features_batch,
    build_pred_features_for_actions,
    build_pred_features_grouped,
)


def action_reward(action: Union[RMSAAction, BlockedAction]) -> float:
    """Immediate reward: 1 if admitted, 0 if blocked."""
    return 1.0 if isinstance(action, RMSAAction) else 0.0


def _group_argmax(
    values: torch.Tensor, counts: List[int]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return valid group ids and selected flat indices without CPU synchronization."""
    device = values.device
    counts_t = torch.as_tensor(counts, dtype=torch.long, device=device)
    valid_groups = torch.nonzero(counts_t > 0, as_tuple=False).view(-1)
    if valid_groups.numel() == 0:
        return valid_groups, torch.empty(0, dtype=torch.long, device=device)

    max_count = int(counts_t.max().item())
    offsets = torch.cumsum(counts_t, dim=0) - counts_t
    group_ids = torch.repeat_interleave(
        torch.arange(len(counts), device=device), counts_t
    )
    positions = (
        torch.arange(values.numel(), device=device)
        - torch.repeat_interleave(offsets, counts_t)
    )
    padded = torch.full(
        (len(counts), max_count),
        -torch.inf,
        dtype=values.dtype,
        device=device,
    )
    padded[group_ids, positions] = values
    best_offsets = torch.argmax(padded, dim=1)
    selected = offsets[valid_groups] + best_offsets[valid_groups]
    return valid_groups, selected


class CandidateGenerator:
    """Lightweight candidate generation that mirrors ``PDSRMSAEnv.build_candidates``."""

    def __init__(
        self,
        topology: Union[str, Topology],
        num_slots: int,
        k_paths: int = K_PATHS,
        path_sort_strategy: str = "hops",
    ):
        if isinstance(topology, str):
            self.topology = load_topology(topology)
            self.topology_name = topology
        else:
            self.topology = topology
            self.topology_name = topology.name
        self.num_slots = int(num_slots)
        self.k_paths = int(k_paths)
        self.path_sort_strategy = str(path_sort_strategy)
        if self.path_sort_strategy not in {"km", "hops"}:
            raise ValueError("path_sort_strategy must be 'km' or 'hops'")
        self._path_cache: dict[Tuple[int, int, int, str], List[Tuple[int, ...]]] = {}
        self._link_ids = _directed_link_ids(self.topology)

    def _cached_paths(self, src: int, dst: int) -> List[Tuple[int, ...]]:
        key = (int(src), int(dst), self.k_paths, self.path_sort_strategy)
        if key not in self._path_cache:
            self._path_cache[key] = self.topology.k_shortest_paths(
                int(src), int(dst), self.k_paths, sort_by=self.path_sort_strategy
            )
        return self._path_cache[key]

    @staticmethod
    def _free_blocks(available: np.ndarray, required: int) -> List[Tuple[int, int]]:
        blocks: List[Tuple[int, int]] = []
        start = None
        for i, free in enumerate(available):
            if free and start is None:
                start = i
            if (not free) and start is not None:
                if i - start >= required:
                    blocks.append((start, i - start))
                start = None
        if start is not None and len(available) - start >= required:
            blocks.append((start, len(available) - start))
        return blocks

    def generate(
        self, bitmap: np.ndarray, request: Request
    ) -> List[Union[RMSAAction, BlockedAction]]:
        """Return candidates for ``request`` in the state encoded by ``bitmap``."""
        raw_paths = self._cached_paths(request.src_node, request.dst_node)
        candidates: List[RMSAAction] = []
        for rank, path in enumerate(raw_paths):
            path_len = self.topology.path_length_km(path)
            mod = highest_modulation_for_distance(path_len)
            if mod is None:
                continue
            mod_name, se = mod
            req_fs = required_fs(request.bitrate_gbps, se)
            arcs = [(path[i], path[i + 1]) for i in range(len(path) - 1)]
            indices = [self._link_ids.index(arc) for arc in arcs]
            available = ~np.any(np.asarray(bitmap, dtype=bool)[indices, :], axis=0)
            blocks = self._free_blocks(available, req_fs)
            blocks = sorted(blocks, key=lambda b: b[0])[:MAX_BLOCKS_PER_PATH]
            for start, _size in blocks:
                action_id = (rank, mod_name, start, req_fs)
                candidates.append(
                    RMSAAction(
                        action_id=action_id,
                        path_rank=rank,
                        path=path,
                        modulation=mod_name,
                        start_slot=start,
                        required_fs=req_fs,
                    )
                )
        if not candidates:
            return [BlockedAction(reason="insufficient_spectrum")]
        return candidates


def _extract_state(env: PDSRMSAEnv) -> TrueState:
    return env.snapshot_true_state()


class PDSAgent:
    """PDS-MLP agent: scores actions as C_now + gamma * V_pds(Ω+_a)."""

    def __init__(
        self,
        input_dim: int,
        feature_config: dict,
        topology: Union[str, Topology],
        num_slots: int,
        gamma: float = 0.99,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        k_paths: int = K_PATHS,
        path_sort_strategy: str = "hops",
        eta: float = 0.0,
    ):
        self.input_dim = int(input_dim)
        self.feature_config = feature_config
        self.gamma = float(gamma)
        self.eta = float(eta)
        self.device = device or torch.device("cpu")
        self.num_slots = int(num_slots)
        self.k_paths = int(k_paths)
        self.path_sort_strategy = str(path_sort_strategy)

        self.online_net = PDSMLP(self.input_dim).to(self.device)
        self.target_net = PDSMLP(self.input_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.candidate_generator = CandidateGenerator(
            topology, num_slots, k_paths=k_paths, path_sort_strategy=path_sort_strategy
        )
        self.epsilon = 1.0

    def count_parameters(self) -> int:
        return count_parameters(self.online_net)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(epsilon)

    def select_action(
        self, env: PDSRMSAEnv, request: Request
    ) -> Union[RMSAAction, BlockedAction]:
        state = _extract_state(env)
        candidates = self.candidate_generator.generate(state.bitmap, request)
        if len(candidates) == 1 and isinstance(candidates[0], BlockedAction):
            return candidates[0]

        legal = [a for a in candidates if isinstance(a, RMSAAction)]
        if np.random.rand() < self.epsilon:
            return legal[np.random.randint(len(legal))]

        from sa_hmarl.pds_rmsa.env.reservations import PostDecisionState

        post_states = []
        for action in legal:
            post_bitmap, post_reservations, post_time = fast_afterstate(
                state.bitmap,
                state.reservations,
                state.time,
                request,
                action,
                self.candidate_generator.topology,
            )
            post_states.append(
                PostDecisionState(
                    time=post_time,
                    bitmap=post_bitmap,
                    reservations=post_reservations,
                    admitted_this_step=True,
                    allocated_action=action,
                )
            )
        features = build_pds_features_batch(post_states, self.feature_config)

        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            values = self.online_net(tensor).view(-1)
        if self.eta == 0.0:
            scores = 1.0 + self.gamma * values  # C_now=1 for legal actions
        else:
            from sa_hmarl.pds_rmsa.training.potential_shaping import (
                phi_batch,
                shaped_selector_scores,
            )

            bitmaps = np.stack([np.asarray(ps.bitmap, dtype=bool) for ps in post_states])
            next_phis = torch.tensor(
                phi_batch(bitmaps), dtype=torch.float32, device=self.device
            )
            rewards = torch.ones(len(legal), dtype=torch.float32, device=self.device)
            scores = shaped_selector_scores(rewards, values, next_phis, self.eta, self.gamma)
        best_idx = int(torch.argmax(scores).item())
        return legal[best_idx]

    def _build_current_post_states(self, batch: List[dict]):
        from sa_hmarl.pds_rmsa.env.reservations import PostDecisionState

        post_states = []
        for transition in batch:
            pre = transition["pre"]
            action = transition["action"]
            request = transition["request"]
            if isinstance(action, BlockedAction):
                post_state = PostDecisionState(
                    time=pre.time,
                    bitmap=pre.bitmap,
                    reservations=pre.reservations,
                    admitted_this_step=False,
                    allocated_action=None,
                )
            else:
                post_bitmap, post_reservations, post_time = fast_afterstate(
                    pre.bitmap,
                    pre.reservations,
                    pre.time,
                    request,
                    action,
                    self.candidate_generator.topology,
                )
                post_state = PostDecisionState(
                    time=post_time,
                    bitmap=post_bitmap,
                    reservations=post_reservations,
                    admitted_this_step=True,
                    allocated_action=action,
                )
            post_states.append(post_state)
        return post_states

    def _build_current_features(self, batch: List[dict]) -> np.ndarray:
        post_states = self._build_current_post_states(batch)
        return build_pds_features_batch(post_states, self.feature_config)

    def _build_current_phis(self, batch: List[dict]) -> torch.Tensor:
        from sa_hmarl.pds_rmsa.training.potential_shaping import phi_batch

        post_states = self._build_current_post_states(batch)
        bitmaps = np.stack([np.asarray(ps.bitmap, dtype=bool) for ps in post_states])
        return torch.tensor(phi_batch(bitmaps), dtype=torch.float32, device=self.device)

    def _compute_next_pds_target(self, next_pre: TrueState, next_request: Request) -> float:
        """Convenience wrapper for the batched target computation on a single transition."""
        if self.eta != 0.0:
            raise NotImplementedError(
                "_compute_next_pds_target convenience wrapper does not support eta != 0"
            )
        targets = self._compute_next_pds_targets([
            {"next_pre": next_pre, "next_request": next_request, "done": False}
        ])
        return float(targets[0])

    def _compute_next_pds_targets(
        self,
        batch: List[dict],
        current_phis: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Batched PDS Double-TD target values for a list of transitions.

        For each transition, enumerate next candidates, build their afterstate
        features in a single batched pass, run the online network once to select
        a* for every transition, and evaluate the selected afterstates with the
        target network.
        """
        from sa_hmarl.pds_rmsa.env.reservations import PostDecisionState
        from sa_hmarl.pds_rmsa.training.potential_shaping import (
            phi_batch,
            shaped_full_targets,
            shaped_selector_scores,
        )

        if self.eta != 0.0 and current_phis is None:
            raise ValueError(
                "eta != 0 requires current_phis to be passed to _compute_next_pds_targets"
            )

        # Collect all candidate post-states and remember how many belong to each transition.
        all_post_states: List[PostDecisionState] = []
        all_rewards: List[float] = []
        transition_counts: List[int] = []
        for transition in batch:
            if transition["done"]:
                transition_counts.append(0)
                continue
            next_pre = transition["next_pre"]
            next_request = transition["next_request"]
            candidates = self.candidate_generator.generate(next_pre.bitmap, next_request)
            count = 0
            for action in candidates:
                if isinstance(action, RMSAAction):
                    post_bitmap, post_reservations, post_time = fast_afterstate(
                        next_pre.bitmap,
                        next_pre.reservations,
                        next_pre.time,
                        next_request,
                        action,
                        self.candidate_generator.topology,
                    )
                    admitted = True
                else:
                    # A rejected request does not terminate the continuing RMSA
                    # process; its post-state keeps the same resource state.
                    post_bitmap = next_pre.bitmap
                    post_reservations = next_pre.reservations
                    post_time = next_pre.time
                    admitted = False
                all_post_states.append(
                    PostDecisionState(
                        time=post_time,
                        bitmap=post_bitmap,
                        reservations=post_reservations,
                        admitted_this_step=admitted,
                        allocated_action=action if admitted else None,
                    )
                )
                all_rewards.append(action_reward(action))
                count += 1
            transition_counts.append(count)

        if not all_post_states:
            return torch.zeros(len(batch), dtype=torch.float32, device=self.device)

        features = build_pds_features_batch(all_post_states, self.feature_config)
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        rewards = torch.tensor(
            all_rewards, dtype=torch.float32, device=self.device
        )
        with torch.no_grad():
            online_values = self.online_net(tensor).view(-1)
            if self.eta != 0.0:
                bitmaps = np.stack([np.asarray(ps.bitmap, dtype=bool) for ps in all_post_states])
                next_phis = torch.tensor(
                    phi_batch(bitmaps), dtype=torch.float32, device=self.device
                )
                selector_scores = shaped_selector_scores(
                    rewards, online_values, next_phis, self.eta, self.gamma
                )
            else:
                selector_scores = rewards + self.gamma * online_values
            valid_groups, selected_indices = _group_argmax(selector_scores, transition_counts)
            target_values = self.target_net(
                tensor.index_select(0, selected_indices)
            ).view(-1)
            if self.eta != 0.0:
                selected_next_phis = next_phis.index_select(0, selected_indices)
                selected_rewards = rewards.index_select(0, selected_indices)
                selected_current_phis = current_phis[valid_groups]
                selected_targets = shaped_full_targets(
                    selected_rewards,
                    target_values,
                    selected_next_phis,
                    selected_current_phis,
                    self.eta,
                    self.gamma,
                )
            else:
                selected_targets = (
                    rewards.index_select(0, selected_indices)
                    + self.gamma * target_values
                )
            targets = torch.zeros(
                len(batch), dtype=torch.float32, device=self.device
            )
            targets[valid_groups] = selected_targets
        return targets

    def update(self, batch: List[dict]) -> float:
        """Update using SmoothL1 loss and the PDS Double-TD target."""
        if not batch:
            return 0.0

        current_features = self._build_current_features(batch)
        if self.eta != 0.0:
            current_phis = self._build_current_phis(batch)
            targets = self._compute_next_pds_targets(batch, current_phis=current_phis)
        else:
            targets = self._compute_next_pds_targets(batch)

        current_tensor = torch.tensor(current_features, dtype=torch.float32, device=self.device)
        targets_t = torch.as_tensor(targets, dtype=torch.float32, device=self.device)
        bootstrap_mask = torch.tensor(
            [not transition["done"] for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        targets_t = targets_t * bootstrap_mask

        online_values = self.online_net(current_tensor).view(-1)
        loss = F.smooth_l1_loss(online_values, targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())


class PreDAgent:
    """PreD-DQN agent: Q_pred(S,a) with Double-DQN target."""

    def __init__(
        self,
        input_dim: int,
        feature_config: dict,
        topology: Union[str, Topology],
        num_slots: int,
        gamma: float = 0.99,
        lr: float = 1e-4,
        device: Optional[torch.device] = None,
        k_paths: int = K_PATHS,
        path_sort_strategy: str = "hops",
    ):
        self.input_dim = int(input_dim)
        self.feature_config = feature_config
        self.gamma = float(gamma)
        self.device = device or torch.device("cpu")
        self.num_slots = int(num_slots)
        self.k_paths = int(k_paths)
        self.path_sort_strategy = str(path_sort_strategy)

        self.online_net = PreDMLP(self.input_dim).to(self.device)
        self.target_net = PreDMLP(self.input_dim).to(self.device)
        self.target_net.load_state_dict(self.online_net.state_dict())
        self.target_net.eval()

        self.optimizer = torch.optim.Adam(self.online_net.parameters(), lr=lr)
        self.candidate_generator = CandidateGenerator(
            topology, num_slots, k_paths=k_paths, path_sort_strategy=path_sort_strategy
        )
        self.epsilon = 1.0

    def count_parameters(self) -> int:
        return count_parameters(self.online_net)

    def sync_target(self) -> None:
        self.target_net.load_state_dict(self.online_net.state_dict())

    def set_epsilon(self, epsilon: float) -> None:
        self.epsilon = float(epsilon)

    def select_action(
        self, env: PDSRMSAEnv, request: Request
    ) -> Union[RMSAAction, BlockedAction]:
        state = _extract_state(env)
        candidates = self.candidate_generator.generate(state.bitmap, request)
        if len(candidates) == 1 and isinstance(candidates[0], BlockedAction):
            return candidates[0]

        legal = [a for a in candidates if isinstance(a, RMSAAction)]
        if np.random.rand() < self.epsilon:
            return legal[np.random.randint(len(legal))]

        features = build_pred_features_for_actions(
            state, request, legal, self.feature_config
        )

        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            q_values = self.online_net(tensor).view(-1).cpu().numpy()
        best_idx = int(np.argmax(q_values))
        return legal[best_idx]

    def _build_current_features(self, batch: List[dict]) -> np.ndarray:
        return build_pred_features_batch(
            [transition["pre"] for transition in batch],
            [transition["request"] for transition in batch],
            [transition["action"] for transition in batch],
            self.feature_config,
        )

    def _compute_next_target(
        self, next_pre: TrueState, next_request: Request
    ) -> float:
        """PreD Double-DQN target value for the next state.

        a* = argmax_a Q_online(S_next, a)
        return Q_target(S_next, a*)

        If no RMSA allocation exists, BlockedAction is the only available action.
        Its immediate reward is 0, but its Q value still includes the continuing
        process after that rejected request.
        """
        value = self._compute_next_q_values([{
            "next_pre": next_pre,
            "next_request": next_request,
            "done": False,
        }])
        return float(value[0].item())

    def _compute_next_q_values(self, batch: List[dict]) -> torch.Tensor:
        """Compute all Double-DQN bootstrap values with two batched forwards."""
        pre_states = [transition["next_pre"] for transition in batch]
        requests = [transition["next_request"] for transition in batch]
        action_groups: List[List[Union[RMSAAction, BlockedAction]]] = []
        for transition in batch:
            if transition["done"]:
                action_groups.append([])
                continue
            candidates = self.candidate_generator.generate(
                transition["next_pre"].bitmap, transition["next_request"]
            )
            action_groups.append(list(candidates))

        features, counts = build_pred_features_grouped(
            pre_states, requests, action_groups, self.feature_config
        )
        values = torch.zeros(len(batch), dtype=torch.float32, device=self.device)
        if features.shape[0] == 0:
            return values

        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        with torch.no_grad():
            online_q = self.online_net(tensor).view(-1)
            valid_groups, selected_indices = _group_argmax(online_q, counts)
            selected_q = self.target_net(
                tensor.index_select(0, selected_indices)
            ).view(-1)
            values[valid_groups] = selected_q
        return values

    def update(self, batch: List[dict]) -> float:
        """Update using SmoothL1 loss and the PreD Double-DQN target."""
        if not batch:
            return 0.0

        current_features = self._build_current_features(batch)

        rewards = torch.tensor(
            [action_reward(transition["action"]) for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        next_values = self._compute_next_q_values(batch)
        bootstrap_mask = torch.tensor(
            [not transition["done"] for transition in batch],
            dtype=torch.float32,
            device=self.device,
        )
        targets_t = rewards + self.gamma * next_values * bootstrap_mask

        current_tensor = torch.tensor(current_features, dtype=torch.float32, device=self.device)

        online_q = self.online_net(current_tensor).view(-1)
        loss = F.smooth_l1_loss(online_q, targets_t)

        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.online_net.parameters(), max_norm=10.0)
        self.optimizer.step()

        return float(loss.item())


def linear_epsilon(
    step: int, total_decay_steps: int, eps_start: float = 1.0, eps_end: float = 0.05
) -> float:
    """Linear epsilon decay from ``eps_start`` to ``eps_end`` over ``total_decay_steps``."""
    if step >= total_decay_steps:
        return float(eps_end)
    return eps_start + (eps_end - eps_start) * (step / total_decay_steps)
