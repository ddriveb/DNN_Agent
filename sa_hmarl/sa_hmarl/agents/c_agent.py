"""Legacy-compatible Agent-C feature helper.

The old DQN Agent-C policy has been removed. This module is kept only as a
feature-builder shim for diagnostics and backward-compatible imports.
"""
import numpy as np
from typing import Optional, Dict, Any, Tuple


class AgentC:
    """Feature-only Agent-C helper kept for compatibility.

    It no longer contains a learnable control policy. Active training and
    evaluation code should use ``PPOAgentC`` or the neutral feature-builder
    aliases in :mod:`sa_hmarl.agents.action_feature_builders`.
    """

    def __init__(self,
                 input_dim: int = 17,
                 hidden_dims=(128, 64),
                 gamma: float = 0.95,
                 epsilon: float = 0.1,
                 lr: float = 1e-3,
                 device: str = 'cpu',
                 ablation: bool = False,
                 zero_spectrum: bool = False,
                 feature_mode: str = "default",
                 num_servers: Optional[int] = None,
                 fixed_blend_alpha: float = 0.5):
        if ablation and zero_spectrum:
            raise ValueError("Cannot use both ablation and zero_spectrum")
        if ablation and feature_mode in ("mean_field", "typed_mean_field", "gated_typed_mean_field", "fixed_blend_typed_mean_field", "candidate_mean_field", "candidate_mean_field_count_only"):
            raise ValueError(
                "mean_field/typed_mean_field/gated_typed_mean_field/fixed_blend_typed_mean_field/candidate_mean_field feature_modes are not supported with ablation"
            )
        if feature_mode not in (
            "default",
            "enhanced",
            "pressure_aware",
            "overload_aware",
            "cross_pressure",
            "r_feasibility",
            "r_feasibility_safe",
            "r_feasibility_spectrum_impact",
            "r_feasibility_mr_spec_compute",
            "mr_feasibility_rule",
            "mean_field",
            "typed_mean_field",
            "gated_typed_mean_field",
            "fixed_blend_typed_mean_field",
            "candidate_mean_field",
            "candidate_mean_field_count_only",
        ):
            raise ValueError(f"Unknown Agent-C feature_mode: {feature_mode}")
        actual_dim = 7 if ablation else input_dim
        if actual_dim <= 17:
            if feature_mode == "enhanced":
                actual_dim = 17 + 7
            elif feature_mode == "pressure_aware":
                actual_dim = 17 + 8
            elif feature_mode == "overload_aware":
                actual_dim = 17 + 9
            elif feature_mode == "r_feasibility":
                actual_dim = 17 + 10
            elif feature_mode == "r_feasibility_safe":
                actual_dim = 17 + 10 + 2
            elif feature_mode == "r_feasibility_spectrum_impact":
                actual_dim = 17 + 10 + 5
            elif feature_mode == "r_feasibility_mr_spec_compute":
                actual_dim = 17 + 10 + 6 + 5
            elif feature_mode == "mr_feasibility_rule":
                actual_dim = 17 + 10 + 6
            elif feature_mode == "cross_pressure":
                actual_dim = 29  # 29-dimensional cross_pressure features
            elif feature_mode == "mean_field":
                if num_servers is None:
                    raise ValueError(
                        "mean_field feature_mode requires num_servers to compute input_dim"
                    )
                actual_dim = 17 + num_servers + 2
            elif feature_mode in ("typed_mean_field", "gated_typed_mean_field", "fixed_blend_typed_mean_field"):
                if num_servers is None:
                    raise ValueError(
                        f"{feature_mode} feature_mode requires num_servers to compute input_dim"
                    )
                # 3 types * (server distribution + demand + release)
                actual_dim = 17 + 3 * (num_servers + 2)
            elif feature_mode == "candidate_mean_field":
                # global: 2 dims + 3 types * 4 dims = 14
                actual_dim = 17 + 14
            elif feature_mode == "candidate_mean_field_count_only":
                # global count + 3 types * (count_norm + ratio) = 7
                actual_dim = 17 + 7
        self.input_dim = actual_dim
        self.fixed_blend_alpha = float(fixed_blend_alpha)
        self.device = device
        self.ablation = ablation
        self.zero_spectrum = zero_spectrum
        self.feature_mode = feature_mode
        # Legacy DQN args are accepted for caller compatibility but unused.
        self.hidden_dims = tuple(hidden_dims)
        self.gamma = gamma
        self.epsilon = epsilon
        self.lr = lr

    # ------------------------------------------------------------------
    # Action-feature construction
    # ------------------------------------------------------------------

    def build_action_features(self, obs: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
        """Construct per-action feature vectors from Agent-C observation.

        Returns:
            action_features: (num_actions, input_dim) float32 array
            mask:            (num_actions,) bool array (alias of obs["agent_c_mask"])
        """
        mask = obs["agent_c_mask"]
        features = []
        for idx, feat_dict in enumerate(obs["candidate_features"]):
            local_ms = feat_dict["local_compute_ms"]
            edge_ms = feat_dict["edge_compute_ms"]
            best_fs = feat_dict["best_fs_estimate"]
            safe_fs = feat_dict["safe_fs_estimate"]

            # Use large sentinel values so infeasible candidates look BAD to the NN
            _INF_SENTINEL = 1e6

            feat = [
                feat_dict["intermediate_size_mb"],
                local_ms if local_ms != float('inf') else _INF_SENTINEL,
                edge_ms if edge_ms != float('inf') else _INF_SENTINEL,
                feat_dict["server_utilization"],
                best_fs if best_fs is not None else _INF_SENTINEL,
                safe_fs if safe_fs is not None else _INF_SENTINEL,
                feat_dict["feasible_count"],
            ]

            if not self.ablation:
                spec = feat_dict["spectrum_summary"]
                if isinstance(spec, list):
                    spec = np.array(spec, dtype=np.float32)
                if self.zero_spectrum:
                    spec = np.zeros_like(spec)
                feat.extend(spec.tolist())

                if getattr(self, "feature_mode", "default") == "enhanced":
                    feat.extend(AgentC._build_enhanced_features(self, obs, feat_dict, spec))
                elif getattr(self, "feature_mode", "default") == "pressure_aware":
                    feat.extend(AgentC._build_pressure_aware_features(self, feat_dict, spec))
                elif getattr(self, "feature_mode", "default") == "overload_aware":
                    feat.extend(AgentC._build_overload_aware_features(self, obs, feat_dict, idx))
                elif getattr(self, "feature_mode", "default") == "r_feasibility":
                    feat.extend(AgentC._build_r_feasibility_features(self, obs, feat_dict))
                elif getattr(self, "feature_mode", "default") == "r_feasibility_safe":
                    feat.extend(AgentC._build_r_feasibility_features(self, obs, feat_dict))
                    feat.extend(AgentC._build_server_margin_features(self, feat_dict))
                elif getattr(self, "feature_mode", "default") == "r_feasibility_spectrum_impact":
                    feat.extend(AgentC._build_r_feasibility_features(self, obs, feat_dict))
                    feat.extend(AgentC._build_spectrum_impact_features(self, obs, feat_dict))
                elif getattr(self, "feature_mode", "default") == "r_feasibility_mr_spec_compute":
                    feat.extend(AgentC._build_r_feasibility_features(self, obs, feat_dict))
                    mu_spec = obs.get("mu_res_spec")
                    if mu_spec is not None:
                        feat.extend(np.asarray(mu_spec, dtype=np.float32).tolist())
                    mu_compute = obs.get("mu_res_compute")
                    if mu_compute is not None:
                        feat.extend(np.asarray(mu_compute, dtype=np.float32).tolist())
                elif getattr(self, "feature_mode", "default") == "mr_feasibility_rule":
                    feat.extend(AgentC._build_r_feasibility_features(self, obs, feat_dict))
                    feat.extend(AgentC._build_mr_feasibility_rule_features(self, obs, feat_dict))
                elif getattr(self, "feature_mode", "default") == "cross_pressure":
                    feat = AgentC._build_cross_pressure_features(self, obs, feat_dict)
                    # cross_pressure replaces the entire feature vector
                    if getattr(self, "feature_mode", "default") == "mean_field":
                        raise ValueError(
                            "mean_field cannot be combined with cross_pressure feature_mode"
                        )
                    features.append(feat)
                    continue

            if getattr(self, "feature_mode", "default") == "mean_field":
                mf_vec = obs.get("mean_field")
                if mf_vec is not None:
                    feat.extend(np.asarray(mf_vec, dtype=np.float32).tolist())
            elif getattr(self, "feature_mode", "default") in ("typed_mean_field", "gated_typed_mean_field"):
                mf_vec = obs.get("typed_mean_field")
                if mf_vec is not None:
                    feat.extend(np.asarray(mf_vec, dtype=np.float32).tolist())
            elif getattr(self, "feature_mode", "default") == "fixed_blend_typed_mean_field":
                mf_vec = obs.get("typed_mean_field")
                if mf_vec is not None:
                    feat.extend((self.fixed_blend_alpha * np.asarray(mf_vec, dtype=np.float32)).tolist())
            elif getattr(self, "feature_mode", "default") in ("candidate_mean_field", "candidate_mean_field_count_only"):
                server_id = idx % max(len(obs.get("server_utilizations", [])), 1)
                count_only = getattr(self, "feature_mode", "default") == "candidate_mean_field_count_only"
                feat.extend(AgentC._build_candidate_mean_field_features(
                    self, obs, server_id, count_only=count_only
                ))

            features.append(feat)

        return np.array(features, dtype=np.float32), mask

    @staticmethod
    def _build_candidate_mean_field_features(
        self,
        obs: Dict[str, Any],
        server_id: int,
        count_only: bool = False,
        max_active_norm: float = 40.0,
        holding_norm: float = 10.0,
    ) -> list:
        """Build per-candidate server-conditioned mean-field features.

        Full version (14 dims):
            [global_srv_load_j, global_active_count_norm,
             light_srv_load_j, light_count_norm, light_dem_norm, light_rel_norm,
             medium_srv_load_j, medium_count_norm, medium_dem_norm, medium_rel_norm,
             heavy_srv_load_j, heavy_count_norm, heavy_dem_norm, heavy_rel_norm]

        Count-only ablation (7 dims):
            [global_active_count_norm,
             light_count_norm, light_type_ratio,
             medium_count_norm, medium_type_ratio,
             heavy_count_norm, heavy_type_ratio]
        """
        from sa_hmarl.env.mean_field import TYPED_MEAN_FIELD_TYPES

        global_mf = obs.get("global_mean_field_dict")
        typed_mf = obs.get("typed_mean_field_dict")
        num_slots = float(obs.get("num_slots", 1.0))
        norm_denom = max(num_slots, 1.0)
        rel_denom = max(float(holding_norm), 1.0)

        if global_mf is None or typed_mf is None:
            # Fallback for observations without raw dicts (should not happen in training)
            dim = 7 if count_only else 14
            return [0.0] * dim

        total_active = float(global_mf.get("count", 0))

        # Global features
        global_srv_load = float(global_mf["srv"][server_id])
        global_count_norm = float(global_mf.get("count", 0)) / max(float(max_active_norm), 1.0)

        if count_only:
            out = [global_count_norm]
            for t in TYPED_MEAN_FIELD_TYPES:
                tdata = typed_mf.get(t, {})
                count = float(tdata.get("count", 0))
                count_norm = count / max(float(max_active_norm), 1.0)
                type_ratio = count / max(total_active, 1.0)
                out.extend([count_norm, type_ratio])
            return out

        out = [global_srv_load, global_count_norm]
        for t in TYPED_MEAN_FIELD_TYPES:
            tdata = typed_mf.get(t, {})
            srv_load = float(tdata["srv"][server_id])
            count_norm = float(tdata.get("count", 0)) / max(float(max_active_norm), 1.0)
            dem_norm = float(tdata.get("dem", 0.0)) / norm_denom
            rel_norm = float(tdata.get("rel", 0.0)) / rel_denom
            out.extend([srv_load, count_norm, dem_norm, rel_norm])
        return out

    # ------------------------------------------------------------------
    # r_feasibility diagnostic helper (debug only, does not affect training)
    # ------------------------------------------------------------------

    @staticmethod
    def compute_r_feasibility_diagnostics(
        obs: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Compute raw counts behind the existing r_feasibility features.

        This is a debug-only helper for consistency auditing.  It mirrors
        :meth:`_build_r_feasibility_features` but returns the un-normalized
        raw quantities so they can be compared against the real Agent-R mask.

        Returns:
            dict with keys:
                - valid_r_actions (int): total number of valid (path,mod,block)
                  actions counted by the existing feature logic.
                - valid_path_mods (int): number of (path,mod) pairs with at
                  least one valid block.
                - valid_paths (set[int]): set of path indices with >=1 valid
                  (path,mod,block).
                - valid_mods (set[int]): set of modulation indices with >=1
                  valid (path,mod,block).
                - min_required_fs (float): smallest required FS among feasible
                  path-mod pairs.
                - safe_required_fs (float): largest required FS among feasible
                  path-mod pairs.
                - best_block_size (float): largest block size seen among valid
                  blocks.
                - avg_block_waste (float): average waste ratio among valid
                  blocks.
                - max_blocks_used (int): max number of candidate blocks seen
                  across path-mod pairs (used for normalization in features).
                - k_paths (int): k_paths value used for normalization.
                - num_mods (int): number of modulation formats used.
        """
        k_paths = int(obs.get("_k_paths", 3))
        num_mods = int(feat_dict.get("_mod_formats", 4))

        feasible_pm = feat_dict.get("_feasible_mask_per_path_mod", [])
        required_pm = feat_dict.get("_required_fs_per_path_mod", [])
        blocks_pm = feat_dict.get("_candidate_blocks_per_path_mod", [])

        max_blocks = 0
        for path_blocks in blocks_pm:
            for mod_blocks in path_blocks:
                max_blocks = max(max_blocks, len(mod_blocks))
        max_blocks = max(max_blocks, 1)

        valid_r_actions = 0
        valid_path_mods = 0
        valid_paths = set()
        valid_mods = set()
        required_fs_values = []
        block_wastes = []
        best_block_size = 0.0

        for p_idx, path_mask in enumerate(feasible_pm):
            for m_idx, is_feasible_pm in enumerate(path_mask):
                if not is_feasible_pm:
                    continue
                try:
                    req_fs = required_pm[p_idx][m_idx]
                except (IndexError, TypeError):
                    continue
                if req_fs is None or req_fs <= 0:
                    continue

                mod_blocks = []
                if p_idx < len(blocks_pm) and m_idx < len(blocks_pm[p_idx]):
                    mod_blocks = blocks_pm[p_idx][m_idx]

                pm_valid_blocks = 0
                for block in mod_blocks[:max_blocks]:
                    if not isinstance(block, (tuple, list)) or len(block) < 2:
                        continue
                    block_size = float(block[1])
                    if block_size >= req_fs:
                        valid_r_actions += 1
                        pm_valid_blocks += 1
                        best_block_size = max(best_block_size, block_size)
                        block_wastes.append((block_size - float(req_fs)) / max(block_size, 1.0))

                if pm_valid_blocks > 0:
                    valid_path_mods += 1
                    valid_paths.add(p_idx)
                    valid_mods.add(m_idx)
                    required_fs_values.append(float(req_fs))

        return {
            "valid_r_actions": valid_r_actions,
            "valid_path_mods": valid_path_mods,
            "valid_paths": valid_paths,
            "valid_mods": valid_mods,
            "min_required_fs": min(required_fs_values) if required_fs_values else None,
            "safe_required_fs": max(required_fs_values) if required_fs_values else None,
            "best_block_size": best_block_size,
            "avg_block_waste": float(np.mean(block_wastes)) if block_wastes else 1.0,
            "max_blocks_used": max_blocks,
            "k_paths": k_paths,
            "num_mods": num_mods,
        }

    # ------------------------------------------------------------------
    # r_feasibility features (base 17 + 10 dims)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_r_feasibility_features(
        self,
        obs: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> list:
        """Build direct R-side feasibility features for one C candidate.

        These features expose the immediate downstream R action-space size
        induced by choosing this (split, server). They do not look ahead to
        future requests; they summarize the current valid path/mod/block set.
        """
        num_slots = float(obs.get("num_slots", getattr(self, "_num_slots_override", 24)))
        num_slots = max(num_slots, 1.0)
        k_paths = int(obs.get("_k_paths", 3))
        num_mods = int(feat_dict.get("_mod_formats", 4))

        feasible_pm = feat_dict.get("_feasible_mask_per_path_mod", [])
        required_pm = feat_dict.get("_required_fs_per_path_mod", [])
        blocks_pm = feat_dict.get("_candidate_blocks_per_path_mod", [])

        max_blocks = 0
        for path_blocks in blocks_pm:
            for mod_blocks in path_blocks:
                max_blocks = max(max_blocks, len(mod_blocks))
        max_blocks = max(max_blocks, 1)
        max_actions = max(k_paths * num_mods * max_blocks, 1)
        max_path_mods = max(k_paths * num_mods, 1)

        valid_r_actions = 0
        valid_path_mods = 0
        valid_paths = set()
        valid_mods = set()
        required_fs_values = []
        block_wastes = []
        best_block_size = 0.0

        for p_idx, path_mask in enumerate(feasible_pm):
            for m_idx, is_feasible_pm in enumerate(path_mask):
                if not is_feasible_pm:
                    continue
                try:
                    req_fs = required_pm[p_idx][m_idx]
                except (IndexError, TypeError):
                    continue
                if req_fs is None or req_fs <= 0:
                    continue

                mod_blocks = []
                if p_idx < len(blocks_pm) and m_idx < len(blocks_pm[p_idx]):
                    mod_blocks = blocks_pm[p_idx][m_idx]

                pm_valid_blocks = 0
                for block in mod_blocks[:max_blocks]:
                    if not isinstance(block, (tuple, list)) or len(block) < 2:
                        continue
                    block_size = float(block[1])
                    if block_size >= req_fs:
                        valid_r_actions += 1
                        pm_valid_blocks += 1
                        best_block_size = max(best_block_size, block_size)
                        block_wastes.append((block_size - float(req_fs)) / max(block_size, 1.0))

                if pm_valid_blocks > 0:
                    valid_path_mods += 1
                    valid_paths.add(p_idx)
                    valid_mods.add(m_idx)
                    required_fs_values.append(float(req_fs))

        min_required_fs = min(required_fs_values) if required_fs_values else num_slots
        safe_required_fs = max(required_fs_values) if required_fs_values else num_slots
        avg_block_waste = float(np.mean(block_wastes)) if block_wastes else 1.0
        best_block_fit_pressure = (
            min_required_fs / max(best_block_size, 1.0)
            if valid_r_actions > 0 else 1.0
        )

        return [
            float(valid_r_actions) / max_actions,
            np.log1p(float(valid_r_actions)) / np.log1p(float(max_actions)),
            float(valid_path_mods) / max_path_mods,
            float(len(valid_paths)) / max(k_paths, 1),
            float(len(valid_mods)) / max(num_mods, 1),
            1.0 if valid_r_actions == 0 else 0.0,
            min_required_fs / num_slots,
            safe_required_fs / num_slots,
            float(np.clip(best_block_fit_pressure, 0.0, 1.0)),
            float(np.clip(avg_block_waste, 0.0, 1.0)),
        ]

    # ------------------------------------------------------------------
    # spectrum impact features (5-dim, appended to r_feasibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_spectrum_impact_features(
        self,
        obs: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> list:
        """Build non-leaky candidate-level spectrum impact features.

        These features estimate how much future spectrum feasibility a candidate
        is likely to leave, using only current-state information (no next-request
        lookahead).  They are the learnable proxies identified by the resource
        impact diagnostic.

        Returns 5 features:
            1. composite_congestion  = (1 - free_mean) * (safe_fs / lfb_max)
            2. lfb_pressure          = safe_fs / lfb_max
            3. free_mean             = average free slot ratio across paths
            4. lfb_max_norm          = largest free block / num_slots
            5. feasible_count_pct    = percentile of candidate feasible_count
                                       within the request (0=lowest, 1=highest)
        """
        # Cache feasible-count percentiles per observation to avoid recomputation
        cache_key = "_spectrum_impact_percentiles"
        if cache_key not in obs:
            counts = [float(f.get("feasible_count", 0.0)) for f in obs["candidate_features"]]
            n = len(counts)
            if n <= 1:
                percentiles = [0.5] * n
            else:
                sorted_idx = np.argsort(counts)
                percentiles = [0.0] * n
                for rank, idx in enumerate(sorted_idx):
                    percentiles[idx] = rank / (n - 1.0)
            obs[cache_key] = percentiles

        idx = obs["candidate_features"].index(feat_dict)
        feasible_count_pct = obs[cache_key][idx]

        spec_vec = np.asarray(feat_dict.get("spectrum_summary", []), dtype=float)

        def _vec(i, default=0.0):
            return float(spec_vec[i]) if len(spec_vec) > i else default

        num_slots = float(obs.get("num_slots", 24.0))
        num_slots = max(num_slots, 1.0)

        lfb_max = _vec(0, num_slots)
        free_mean = _vec(6, 1.0)

        safe_fs = feat_dict.get("safe_fs_estimate")
        safe_fs = float(safe_fs) if safe_fs is not None else 0.0

        lfb_pressure = safe_fs / max(lfb_max, 1.0)
        spectrum_congestion = 1.0 - free_mean
        composite_congestion = spectrum_congestion * lfb_pressure

        return [
            float(np.clip(composite_congestion, 0.0, 1.0)),
            float(np.clip(lfb_pressure, 0.0, 1.0)),
            float(np.clip(free_mean, 0.0, 1.0)),
            float(np.clip(lfb_max / num_slots, 0.0, 1.0)),
            float(np.clip(feasible_count_pct, 0.0, 1.0)),
        ]

    # ------------------------------------------------------------------
    # server margin features (2-dim, appended to r_feasibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_server_margin_features(
        self,
        feat_dict: Dict[str, Any],
    ) -> list:
        """Build compute-side safety margin features for one C candidate.

        These features complement r_feasibility by telling Agent-C how much
        compute headroom the target server still has, both before and after
        accepting this split.  They discourage candidates that look good for
        the R agent but would overload the server.

        Returns 2 features:
            1. server_available_ratio = available / capacity
            2. server_margin_after   = (available - edge_cost) / capacity
        """
        available = float(feat_dict.get("server_available_compute", 0.0))
        capacity = float(feat_dict.get("server_capacity", 1.0))
        edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
        capacity = max(capacity, 1e-6)

        server_available_ratio = available / capacity
        server_margin_after = (available - edge_cost) / capacity

        return [
            float(np.clip(server_available_ratio, 0.0, 1.0)),
            float(np.clip(server_margin_after, -1.0, 1.0)),
        ]

    # ------------------------------------------------------------------
    # overload-aware features (9-dim, appended to base 17)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_overload_aware_features(
        self,
        obs: Dict[str, Any],
        feat_dict: Dict[str, Any],
        idx: int,
    ) -> list:
        """Build C-side features that make split-server overload risk explicit.

        These features are designed for topologies like COST239 where blocking is
        dominated by C-side server overload, not R-side spectrum shortage.  They
        expose per-candidate post-action compute pressure and the relative
        attractiveness of the other servers for the same split.

        Returns 9 features:
            1. server_available_ratio     = available / capacity
            2. server_margin_after        = (available - edge_cost) / capacity
            3. server_util_after          = projected utilization after assignment
            4. projected_utilization      = current util + edge_cost / capacity
            5. high_util_risk_90          = 1 if projected_utilization > 0.90
            6. high_util_risk_95          = 1 if projected_utilization > 0.95
            7. server_relative_load       = this server util / mean util
            8. best_margin_server         = 1 if this server has max margin for this split
            9. min_other_server_avail_ratio = min available ratio among other servers for this split
        """
        available = float(feat_dict.get("server_available_compute", 0.0))
        capacity = float(feat_dict.get("server_capacity", 1.0))
        edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
        current_util = float(feat_dict.get("server_utilization", 0.0))
        capacity = max(capacity, 1e-6)

        server_available_ratio = available / capacity
        margin_after = (available - edge_cost) / capacity
        util_after = (capacity - (available - edge_cost)) / capacity
        projected_util = current_util + edge_cost / capacity

        high_util_risk_90 = 1.0 if projected_util > 0.90 else 0.0
        high_util_risk_95 = 1.0 if projected_util > 0.95 else 0.0

        server_utils = obs.get("server_utilizations")
        num_servers = len(server_utils) if server_utils else 1
        num_servers = max(num_servers, 1)
        server_id = idx % num_servers

        server_relative_load = 1.0
        if server_utils and len(server_utils) > 0:
            mean_util = float(np.mean(server_utils))
            if mean_util > 1e-6:
                server_relative_load = current_util / mean_util

        candidate_features = obs.get("candidate_features", [])
        avail_ratios = []
        margins = []
        for c in candidate_features:
            c_available = float(c.get("server_available_compute", 0.0))
            c_capacity = float(c.get("server_capacity", 1.0))
            c_edge_cost = float(c.get("edge_compute_cost", 0.0))
            c_capacity = max(c_capacity, 1e-6)
            avail_ratios.append(c_available / c_capacity)
            margins.append((c_available - c_edge_cost) / c_capacity)

        split_id = idx // num_servers
        start = split_id * num_servers
        end = start + num_servers
        split_margins = margins[start:end] if len(margins) >= end else margins[start:]
        split_avail_ratios = avail_ratios[start:end] if len(avail_ratios) >= end else avail_ratios[start:]

        best_margin_server = 0.0
        if split_margins:
            best_server = int(np.argmax(split_margins))
            if best_server == server_id:
                best_margin_server = 1.0

        min_other_server_avail_ratio = 1.0
        if split_avail_ratios:
            others = [r for i, r in enumerate(split_avail_ratios) if i != server_id]
            if others:
                min_other_server_avail_ratio = float(np.min(others))

        return [
            float(np.clip(server_available_ratio, 0.0, 1.0)),
            float(np.clip(margin_after, -1.0, 1.0)),
            float(np.clip(util_after, 0.0, 1.0)),
            float(np.clip(projected_util, 0.0, 2.0)),
            high_util_risk_90,
            high_util_risk_95,
            float(np.clip(server_relative_load, 0.0, 2.0)),
            best_margin_server,
            float(np.clip(min_other_server_avail_ratio, 0.0, 1.0)),
        ]

    # ------------------------------------------------------------------
    # MR-feasibility rule features (6-dim, appended to r_feasibility)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_mr_feasibility_rule_features(
        self,
        obs: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> list:
        """Build candidate-level multi-resource feasibility rule features.

        These 6 dimensions encode simple physical safety margins for the
        chosen (split, server) candidate. They are intentionally rule-based
        (not learned) and do not use global mean-field statistics.

        Returns 6 features in [0, 1]:
            1. server_available_ratio  = available / capacity
            2. server_margin_after     = normalized post-action compute margin
            3. server_util_after       = normalized post-action compute utilization
            4. deadline_margin         = normalized deadline slack
            5. spectrum_fit_margin     = 1 - best_block_fit_pressure
            6. multi_resource_safe_indicator
        """
        from sa_hmarl.env.fs_demand import DEFAULT_PROP_SPEED_KM_S, DEFAULT_SETUP_TIME_S

        # --- Compute side -------------------------------------------------
        available = float(feat_dict.get("server_available_compute", 0.0))
        capacity = float(feat_dict.get("server_capacity", 1.0))
        edge_cost = float(feat_dict.get("edge_compute_cost", 0.0))
        capacity = max(capacity, 1e-6)

        server_available_ratio = float(np.clip(available / capacity, 0.0, 1.0))

        margin_raw = (available - edge_cost) / capacity
        server_margin_after = float(0.5 * (np.clip(margin_raw, -1.0, 1.0) + 1.0))

        util_after_raw = (capacity - (available - edge_cost)) / capacity
        server_util_after = float(np.clip(util_after_raw, 0.0, 1.0))

        # --- Deadline side ------------------------------------------------
        deadline_ms = float(obs.get("request_features", {}).get("deadline_ms", 0.0))
        local_ms = float(feat_dict.get("local_compute_ms", 0.0))
        edge_ms = float(feat_dict.get("edge_compute_ms", 0.0))
        min_path_km = float(feat_dict.get("server_min_path_km", 0.0))

        # Lower-bound path delay estimate (propagation + setup; no hop count).
        path_delay_est = 0.0
        if np.isfinite(min_path_km) and DEFAULT_PROP_SPEED_KM_S > 0:
            path_delay_est += (min_path_km / DEFAULT_PROP_SPEED_KM_S) * 1000.0
        path_delay_est += DEFAULT_SETUP_TIME_S * 1000.0

        est_total_delay_ms = local_ms + edge_ms + path_delay_est
        deadline_raw = (
            (deadline_ms - est_total_delay_ms) / max(deadline_ms, 1e-6)
            if deadline_ms > 0 else -1.0
        )
        deadline_margin = float(0.5 * (np.clip(deadline_raw, -1.0, 1.0) + 1.0))

        # --- Spectrum side ------------------------------------------------
        diag = AgentC.compute_r_feasibility_diagnostics(obs, feat_dict)
        min_required_fs = diag.get("min_required_fs")
        best_block_size = diag.get("best_block_size", 0.0)
        valid_r_actions = diag.get("valid_r_actions", 0)

        if valid_r_actions > 0 and min_required_fs is not None and min_required_fs > 0:
            fit_pressure = min_required_fs / max(best_block_size, 1.0)
            spectrum_fit_margin = float(np.clip(1.0 - fit_pressure, 0.0, 1.0))
        else:
            spectrum_fit_margin = 0.0

        # --- Multi-resource safe indicator --------------------------------
        server_safe = margin_raw >= 0.0
        deadline_safe = deadline_raw >= 0.0
        r_safe = valid_r_actions > 0
        safe_indicator = 1.0 if (server_safe and deadline_safe and r_safe) else 0.0

        features = [
            server_available_ratio,
            server_margin_after,
            server_util_after,
            deadline_margin,
            spectrum_fit_margin,
            safe_indicator,
        ]
        # Sanitize: finite and clipped to [0, 1].
        return [float(np.clip(np.nan_to_num(v, nan=0.0, posinf=1.0, neginf=0.0), 0.0, 1.0))
                for v in features]

    # ------------------------------------------------------------------
    # cross_pressure features (29-dim)
    # ------------------------------------------------------------------

    @staticmethod
    def _build_cross_pressure_features(
        self,
        obs: Dict[str, Any],
        feat_dict: Dict[str, Any],
    ) -> list:
        """Build 29-dim cross_pressure feature vector for one C candidate.

        Features capture (spectrum × split × server) conditionality via
        valid-R-action analysis, not coarse server-direction summaries.
        """
        _INF = 1e6

        def _fin(v, default=_INF):
            try:
                v = float(v)
            except (TypeError, ValueError):
                return default
            if not np.isfinite(v):
                return default
            return v

        def _clip(v, lo=-1e6, hi=1e6):
            return float(np.clip(_fin(v), lo, hi))

        # --- Base request / candidate info ------------------------------------
        deadline_ms = _fin(obs.get("request_features", {}).get("deadline_ms", 100.0), 100.0)
        local_ms = _fin(feat_dict.get("local_compute_ms"))
        edge_ms = _fin(feat_dict.get("edge_compute_ms"))
        server_util = _fin(feat_dict.get("server_utilization"), 0.0)
        interm_size = _fin(feat_dict.get("intermediate_size_mb"), 0.0)
        num_slots = obs.get("num_slots", getattr(self, "_num_slots_override", 16))

        # --- Raw path×mod info ------------------------------------------------
        fmask_pm = feat_dict.get("_feasible_mask_per_path_mod", [])
        reqfs_pm = feat_dict.get("_required_fs_per_path_mod", [])
        blocks_pm = feat_dict.get("_candidate_blocks_per_path_mod", [])
        n_mod_formats = int(feat_dict.get("_mod_formats", 4))
        k_paths = int(obs.get("_k_paths", 3))

        # --- Decode valid R actions -------------------------------------------
        max_blocks = 0
        for p_blocks in blocks_pm:
            for m_blocks in p_blocks:
                max_blocks = max(max_blocks, len(m_blocks))
        if max_blocks == 0:
            max_blocks = 5

        valid_fs_vals: List[float] = []
        valid_block_sizes: List[float] = []
        valid_paths: set = set()
        valid_mods: set = set()
        seen_pm: set = set()

        for p in range(min(len(fmask_pm), k_paths)):
            for m in range(min(len(fmask_pm[p]) if p < len(fmask_pm) else 0, n_mod_formats)):
                if not fmask_pm[p][m]:
                    continue
                try:
                    req_fs = reqfs_pm[p][m]
                except IndexError:
                    continue
                if req_fs is None or req_fs <= 0:
                    continue
                blocks = blocks_pm[p][m] if p < len(blocks_pm) and m < len(blocks_pm[p]) else []
                has_valid = False
                for b_idx, blk in enumerate(blocks[:max_blocks]):
                    if isinstance(blk, (tuple, list)) and len(blk) >= 2:
                        if int(blk[1]) >= req_fs:
                            has_valid = True
                            valid_block_sizes.append(float(int(blk[1])))
                if has_valid:
                    valid_paths.add(p)
                    valid_mods.add(m)
                    if (p, m) not in seen_pm:
                        seen_pm.add((p, m))
                        valid_fs_vals.append(float(req_fs))

        n_valid_r = len(valid_fs_vals)  # unique (path,mod) pairs — a lower bound
        n_feas_pm = int(_fin(feat_dict.get("n_feas_path_mod", 0), 0))
        n_valid_paths = len(valid_paths)
        n_valid_mods = len(valid_mods)

        # --- Per-path stats for this server -----------------------------------
        server_id = -1
        for i, fd in enumerate(obs.get("candidate_features", [])):
            if fd is feat_dict:
                server_id = i % len(obs.get("server_utilizations", [1]))
                break

        per_server_stats = obs.get("_per_server_path_stats", [])
        path_stats_list = (
            per_server_stats[server_id]
            if 0 <= server_id < len(per_server_stats) else []
        )

        valid_path_frags, valid_path_frees, valid_path_lfbs, valid_path_delays = [], [], [], []
        for p in valid_paths:
            if p < len(path_stats_list):
                ps = path_stats_list[p]
                valid_path_frags.append(float(ps.get("frag_index", 0.0)))
                valid_path_frees.append(float(ps.get("free_ratio", 0.0)))
                valid_path_lfbs.append(float(ps.get("lfb", 0)))
                path_km = float(ps.get("path_dist_km", 0.0))
                hops = int(ps.get("path_length", 0))
                from sa_hmarl.env.fs_demand import (
                    DEFAULT_PROP_SPEED_KM_S, DEFAULT_PROC_PER_HOP_S, DEFAULT_SETUP_TIME_S,
                )
                prop_ms = (path_km / DEFAULT_PROP_SPEED_KM_S) * 1000.0
                proc_ms = hops * DEFAULT_PROC_PER_HOP_S * 1000.0
                setup_ms = DEFAULT_SETUP_TIME_S * 1000.0
                valid_path_delays.append(prop_ms + proc_ms + setup_ms)

        # --- Compute FS / block metrics ---------------------------------------
        if n_valid_r > 0 and valid_block_sizes:
            min_fs = min(valid_fs_vals)
            safe_fs = max(valid_fs_vals)
            best_lfb = max(valid_block_sizes)
            fs_lfb_ratio = min_fs / max(best_lfb, 1.0)
            p25_idx = max(0, int(len(valid_fs_vals) * 0.25))
            p25_fs_ratio = (
                sorted(valid_fs_vals)[p25_idx] / max(best_lfb, 1.0)
                if len(valid_fs_vals) > p25_idx else fs_lfb_ratio
            )
            avg_lfb_val = float(np.mean(valid_path_lfbs)) if valid_path_lfbs else 0.0
        else:
            min_fs = float(num_slots)
            safe_fs = float(num_slots)
            best_lfb = 0.0
            fs_lfb_ratio = 1.0
            p25_fs_ratio = 1.0
            avg_lfb_val = 0.0

        # --- Compute path pressure metrics ------------------------------------
        if valid_path_frags:
            avg_frag = float(np.mean(valid_path_frags))
            max_frag = float(np.max(valid_path_frags))
            avg_free = float(np.mean(valid_path_frees))
            min_free = float(np.min(valid_path_frees))
        else:
            avg_frag = max_frag = 0.0
            avg_free = min_free = 0.0

        # --- Compute delay metrics --------------------------------------------
        if valid_path_delays:
            min_delay = min(valid_path_delays)
            avg_delay = float(np.mean(valid_path_delays))
        else:
            min_delay = deadline_ms
            avg_delay = deadline_ms

        total_ms_min = min_delay + (local_ms if local_ms < _INF else 0) + (edge_ms if edge_ms < _INF else 0)
        total_ms_avg = avg_delay + (local_ms if local_ms < _INF else 0) + (edge_ms if edge_ms < _INF else 0)

        # --- 29-dim feature vector --------------------------------------------
        return [
            _clip(interm_size / 100.0),                                         #  1
            _clip(local_ms / 100.0),                                            #  2
            _clip(edge_ms / 100.0),                                             #  3
            _clip(server_util),                                                 #  4
            _clip((deadline_ms - local_ms - edge_ms) / 100.0),                  #  5
            _clip(float(n_valid_r) / max(k_paths * n_mod_formats * max_blocks, 1)),  #  6
            _clip(float(n_feas_pm) / max(k_paths * n_mod_formats, 1)),          #  7
            _clip(float(n_valid_mods) / max(n_mod_formats, 1)),                 #  8
            _clip(float(n_valid_paths) / max(k_paths, 1)),                      #  9
            _clip(min_fs / max(float(num_slots), 1)),                           # 10
            _clip(safe_fs / max(float(num_slots), 1)),                          # 11
            _clip(fs_lfb_ratio),                                                # 12
            _clip(p25_fs_ratio),                                                # 13
            _clip(best_lfb / max(float(num_slots), 1)),                         # 14
            _clip(avg_lfb_val / max(float(num_slots), 1)),                      # 15
            _clip(avg_frag),                                                    # 16
            _clip(max_frag),                                                    # 17
            _clip(avg_free),                                                    # 18
            _clip(min_free),                                                    # 19
            _clip(1.0 - min_free if valid_path_frees else 1.0),                 # 20
            _clip(1.0 - float(n_valid_paths) / max(k_paths, 1)),                # 21
            _clip(min_delay / 100.0),                                           # 22
            _clip(avg_delay / 100.0),                                           # 23
            _clip((deadline_ms - total_ms_min) / 100.0),                        # 24
            _clip((deadline_ms - total_ms_avg) / 100.0),                        # 25
            1.0 if n_valid_r == 0 else 0.0,                                     # 26
            1.0 if (n_valid_r > 0 and fs_lfb_ratio > 0.8) else 0.0,             # 27
            1.0 if (deadline_ms - total_ms_min < 0) else 0.0,                    # 28
            1.0 if server_util > 0.85 else 0.0,                                 # 29
        ]

    def _build_pressure_aware_features(
        self,
        feat_dict: Dict[str, Any],
        spec: np.ndarray,
    ) -> list:
        """Pressure-aware features for split-server-level decision making.

        Adds 8 dimensions:
          - n_feas_path_mod normalized
          - split_edge_ratio
          - lfb_pressure
          - spectrum_congestion
          - path_diversity
          - frag_mean
          - server_queue_pressure
          - compute_headroom (available / capacity)
        """
        _INF_SENTINEL = 1e6

        def _finite(value, default=_INF_SENTINEL):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            if not np.isfinite(value):
                return default
            return value

        def _clip(value, low=-_INF_SENTINEL, high=_INF_SENTINEL):
            return float(np.clip(_finite(value), low, high))

        n_feas = _finite(feat_dict.get("n_feas_path_mod", 0), 0.0)
        # Normalize: max possible = k paths * num_mods; typical k=3, mods=4 → 12
        max_feas_pm = 12.0
        norm_n_feas = min(n_feas / max_feas_pm, 1.0)

        split_edge_ratio = _clip(feat_dict.get("split_edge_ratio", 0.0), 0.0, 1.0)
        lfb_pressure = _clip(feat_dict.get("lfb_pressure", 0.0), 0.0, 10.0)
        spectrum_congestion = _clip(feat_dict.get("spectrum_congestion", 0.0), 0.0, 1.0)
        path_diversity = _clip(feat_dict.get("path_diversity", 0.0), 0.0, 1.0)
        frag_mean = _clip(feat_dict.get("frag_mean", 0.0), 0.0, 1.0)
        queue_pressure = _clip(feat_dict.get("server_queue_pressure", 0.0), 0.0, 10.0)

        available = _finite(feat_dict.get("server_available_compute", 0.0), 0.0)
        capacity = max(_finite(feat_dict.get("server_capacity", 1.0), 1.0), 1e-6)
        compute_headroom = available / capacity

        return [
            norm_n_feas,
            split_edge_ratio,
            lfb_pressure,
            spectrum_congestion,
            path_diversity,
            frag_mean,
            queue_pressure,
            compute_headroom,
        ]

    def _build_enhanced_features(
        self,
        obs: Dict[str, Any],
        feat_dict: Dict[str, Any],
        spec: np.ndarray,
    ) -> list:
        """Additional delay/resource pressure features for Agent-C.

        These features keep execution decentralized: they are computed from the
        same candidate summary already exposed to Agent-C, not from Agent-R
        private decisions.
        """
        _INF_SENTINEL = 1e6

        def _finite(value, default=_INF_SENTINEL):
            try:
                value = float(value)
            except (TypeError, ValueError):
                return default
            if not np.isfinite(value):
                return default
            return value

        def _clip(value, low=-_INF_SENTINEL, high=_INF_SENTINEL):
            return float(np.clip(_finite(value), low, high))

        local_ms = _finite(feat_dict.get("local_compute_ms"))
        edge_ms = _finite(feat_dict.get("edge_compute_ms"))
        best_fs = _finite(feat_dict.get("best_fs_estimate"))
        safe_fs = _finite(feat_dict.get("safe_fs_estimate"))
        deadline_ms = max(_finite(obs.get("request_features", {}).get("deadline_ms"), 1.0), 1.0)

        lfb_max = max(_finite(spec[0] if len(spec) > 0 else 0.0, 0.0), 1.0)
        lfb_min = max(_finite(spec[2] if len(spec) > 2 else 0.0, 0.0), 1.0)
        lfb_p25 = max(_finite(spec[3] if len(spec) > 3 else lfb_min, lfb_min), 1.0)
        delay_min_s = _finite(spec[9] if len(spec) > 9 else 0.0, 0.0)

        estimated_total_delay_ms = local_ms + edge_ms + delay_min_s * 1000.0
        delay_slack_ms = deadline_ms - estimated_total_delay_ms
        normalized_delay = estimated_total_delay_ms / deadline_ms

        spectrum_pressure = best_fs / lfb_max
        safe_spectrum_pressure = safe_fs / max(lfb_p25, lfb_min, 1.0)

        edge_cost = _finite(feat_dict.get("edge_compute_cost"), 0.0)
        available_compute = _finite(feat_dict.get("server_available_compute"), 0.0)
        server_capacity = max(_finite(feat_dict.get("server_capacity"), 1.0), 1e-6)
        if available_compute > 1e-6:
            server_compute_pressure = edge_cost / available_compute
        else:
            server_compute_pressure = _INF_SENTINEL
        server_capacity_pressure = edge_cost / server_capacity

        # Ratio > 1 means compute-side risk dominates spectrum-side risk.
        risk_balance = server_compute_pressure / max(spectrum_pressure, 1e-6)

        return [
            _clip(estimated_total_delay_ms),
            _clip(delay_slack_ms),
            _clip(normalized_delay, -100.0, 100.0),
            _clip(spectrum_pressure, 0.0, 100.0),
            _clip(safe_spectrum_pressure, 0.0, 100.0),
            _clip(server_compute_pressure, 0.0, 100.0),
            _clip(risk_balance, 0.0, 100.0),
        ]

    # ------------------------------------------------------------------
    # Action selection
    # ------------------------------------------------------------------

    def select_action(self, obs: Dict[str, Any], epsilon: Optional[float] = None) -> Optional[int]:
        raise RuntimeError(
            "Legacy DQN Agent-C has been removed. Use PPOAgentC for policy "
            "selection; AgentC now only provides feature construction helpers."
        )

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def optimize(self, batch: Tuple, batch_size: int) -> Optional[float]:
        raise RuntimeError(
            "Legacy DQN Agent-C optimization has been removed from the active codebase."
        )

    def update_target(self):
        raise RuntimeError(
            "Legacy DQN Agent-C target updates have been removed from the active codebase."
        )
