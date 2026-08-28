"""
Non-equilibrium cell fate landscape analysis.

Corrected approximations
------------------------
- U is treated as quasi -log P when density-calibrated; learned model U may differ.
- Default LAP uses flow-space Hamiltonian ODE rollout (see flow_space_lap.py).
- Gene-space predicted_delta is NOT used as embedding velocity.
- LAP score is flow-space action integral (not discrete graph shortest path).
- Transition state = interior potential maximum along path.
- Flux-logP alignment / EPR helpers are approximate diagnostics.
- Paths on UMAP are candidate embedding paths, not strict physical transitions.
- potential_transform controls whether external U is rescaled (train_model: prefer "none").
"""

import warnings

import numpy as np
import scanpy as sc
from sklearn.neighbors import NearestNeighbors

try:
    from VectorField import VectorFieldAnalyzer
except ImportError:
    VectorFieldAnalyzer = None

from landscape_core import (
    adaptive_path_point_count,
    batch_force_field,
    build_safe_scalar_field,
    build_safe_vector_field,
    calibrate_external_potential,
    compute_entropy_production_approx_along_path,
    compute_flux_logp_alignment_along_path,
    estimate_potential_from_density,
    identify_attractors_from_clusters,
    identify_cluster_endpoints,
    identify_transition_state,
    mean_neighbor_spacing,
    numerical_gradient,
    resolve_scalar_diffusion,
)

VALID_POTENTIAL_TRANSFORMS = frozenset({"none", "zscore", "neg_log_p"})
VALID_LAP_FORCE_MODES = frozenset({"gradient", "total"})
_EMBEDDING_BASIS_TOKENS = ("umap", "pca", "tsne", "latent")


class NonEquilibriumCellFateLandscape:
    """Landscape–flux analysis with least-action paths and transition states."""

    def __init__(
        self,
        adata,
        potential_key="potential_energy",
        embedding_2d_key="X_umap",
        embedding_3d_key="X_umap_3d",
        velocity_key="velocity",
        embedding_velocity_key=None,
        field_method="safe",
        diffusion_coefficient=None,
        potential_transform="none",
        calibrate_potential=False,
        kde_bandwidth=None,
        use_embedding_velocity=False,
        lap_force_mode="gradient",
    ):
        if lap_force_mode not in VALID_LAP_FORCE_MODES:
            raise ValueError(
                f"lap_force_mode 必须是 {sorted(VALID_LAP_FORCE_MODES)} 之一，"
                f"收到: {lap_force_mode!r}"
            )

        if calibrate_potential:
            if potential_transform != "none":
                warnings.warn(
                    "calibrate_potential 已弃用；同时设置了 potential_transform，"
                    "将以 potential_transform 为准。",
                    DeprecationWarning,
                    stacklevel=2,
                )
            else:
                potential_transform = "neg_log_p"
                warnings.warn(
                    "calibrate_potential=True 已弃用，请改用 potential_transform='neg_log_p'。",
                    DeprecationWarning,
                    stacklevel=2,
                )

        if potential_transform not in VALID_POTENTIAL_TRANSFORMS:
            raise ValueError(
                f"potential_transform 必须是 {sorted(VALID_POTENTIAL_TRANSFORMS)} 之一，"
                f"收到: {potential_transform!r}"
            )

        if field_method not in ("safe", "dynamo"):
            raise ValueError(
                f"field_method 必须是 'safe' 或 'dynamo'，收到: {field_method!r}"
            )
        if field_method == "dynamo":
            warnings.warn(
                "field_method='dynamo' 为实验性可视化路径；生产分析请优先使用 "
                "field_method='safe' + use_embedding_velocity=True（如有可靠 embedding velocity）。",
                UserWarning,
                stacklevel=2,
            )

        self.adata = adata
        self.potential_key = potential_key
        self.embedding_2d_key = embedding_2d_key
        self.embedding_3d_key = embedding_3d_key
        self.velocity_key = velocity_key
        self.embedding_velocity_key = embedding_velocity_key
        self.field_method = field_method
        self.use_embedding_velocity = bool(use_embedding_velocity)
        self.lap_force_mode = lap_force_mode
        self.potential_transform = potential_transform
        self.calibrate_potential = calibrate_potential

        self.cell_positions_2d = self._get_embedding(embedding_2d_key)
        self.cell_positions_3d = (
            self._get_embedding(embedding_3d_key)
            if embedding_3d_key in self.adata.obsm
            else None
        )
        self.embedding_velocity = self._get_embedding_velocity_data()

        self.potential_energy, self.log_prob = self._resolve_potential(kde_bandwidth)
        self.potential_is_density_calibrated = (
            self.potential_key not in self.adata.obs
            or self.potential_transform == "neg_log_p"
        )
        # P ∝ exp(-U) 仅在 U 已通过密度一致性校准时才有概率解释；否则为形式化 proxy。
        self.quasi_prob = np.exp(-self.potential_energy + np.min(self.potential_energy))

        self.neighbor_spacing_2d = mean_neighbor_spacing(self.cell_positions_2d)
        self.neighbor_spacing_3d = (
            mean_neighbor_spacing(self.cell_positions_3d)
            if self.cell_positions_3d is not None
            else None
        )

        self.diffusion_coefficient, self._diffusion_source = resolve_scalar_diffusion(
            adata, diffusion=diffusion_coefficient, default=0.1
        )

        if VectorFieldAnalyzer is not None:
            self.field_analyzer = VectorFieldAnalyzer(
                n_neighbors=30,
                grid_points=100,
                grid_points_3d=30,
            )
        else:
            self.field_analyzer = None
            if self.field_method == "dynamo":
                warnings.warn(
                    "VectorFieldAnalyzer 不可用，field_method 已回退为 'safe'。",
                    UserWarning,
                    stacklevel=2,
                )
                self.field_method = "safe"

        self.U_func_2d = None
        self.U_func_3d = None
        self._create_continuous_fields()

        print(f"系统初始化完成: {self.cell_positions_2d.shape[0]} 个细胞")
        print(f"2D embedding: {self.embedding_2d_key}")
        if self.cell_positions_3d is not None:
            print(f"3D embedding: {self.embedding_3d_key}")
        else:
            print(f"3D embedding 不可用 ({embedding_3d_key})，仅支持 2D 分析")
        print(f"场计算方法: {self.field_method}")
        print(
            f"LAP 扩散系数 D={self.diffusion_coefficient:.6g} "
            f"(source={self._diffusion_source}; scalar approx, not learned σ(z,t))"
        )
        print(
            "LAP action / total_action = heuristic path cost "
            "(relative score, not strict transition probability)"
        )
        print(f"Potential transform: {self.potential_transform}")
        if not self.potential_is_density_calibrated:
            print(
                "quasi_prob 仅为形式化 proxy（learned potential 未做 density calibration）；"
                "勿解释为真实稳态概率。"
            )
        if "umap" in self.embedding_2d_key.lower():
            print(
                "提示: 当前为 UMAP embedding 上的 candidate transition path，"
                "非高维表达/latent 空间的严格动力学路径。"
                "更严谨时可改用 train_model latent 的 PCA 坐标（如 embedding_2d_key='X_latent_pca'）。"
            )
        if self.lap_force_mode == "gradient":
            print("LAP 力场: F = -∇U（保守势能梯度，推荐用于 train_model potential）")
        elif self.embedding_velocity is None:
            print("提示: 无 embedding velocity，通量项退化为保守梯度场")
        elif not self.use_embedding_velocity:
            print("提示: use_embedding_velocity=False，忽略 embedding velocity")
        elif self.lap_force_mode == "total":
            print("LAP 力场: total force（gradient + embedding flux residual）")

    @staticmethod
    def _embedding_basis_tokens(embedding_key: str) -> set:
        emb = embedding_key.replace("X_", "").lower()
        tokens = {t for t in _EMBEDDING_BASIS_TOKENS if t in emb}
        if not tokens:
            tokens = {emb}
        return tokens

    @classmethod
    def _velocity_key_matches_embedding(cls, velocity_key: str, embedding_key: str) -> bool:
        """Ensure velocity obsm key refers to the same embedding basis as positions."""
        vel = velocity_key.lower()
        emb_tokens = cls._embedding_basis_tokens(embedding_key)
        vel_tokens = {t for t in _EMBEDDING_BASIS_TOKENS if t in vel}
        if not vel_tokens:
            return any(token in vel for token in emb_tokens)
        if not emb_tokens.intersection(vel_tokens):
            return False
        return vel_tokens.issubset(emb_tokens) or emb_tokens.issubset(vel_tokens)

    @property
    def steady_state_prob(self):
        warnings.warn(
            "steady_state_prob 已重命名为 quasi_prob；"
            "仅在 potential 已密度校准时才可解释为稳态概率 proxy。",
            DeprecationWarning,
            stacklevel=2,
        )
        return self.quasi_prob

    # ------------------------------------------------------------------ data
    def _get_embedding(self, embedding_key):
        if embedding_key in self.adata.obsm:
            return np.asarray(self.adata.obsm[embedding_key], dtype=float)
        raise ValueError(f"嵌入 {embedding_key} 未找到")

    def _ensure_3d_available(self, use_3d: bool) -> None:
        if use_3d and self.cell_positions_3d is None:
            raise ValueError(
                f"3D embedding '{self.embedding_3d_key}' 不在 adata.obsm 中。"
                "请先计算 3D 坐标，或保持 use_3d=False。"
            )

    def _resolve_potential(self, kde_bandwidth):
        kde_u, log_p = estimate_potential_from_density(
            self.cell_positions_2d, bandwidth=kde_bandwidth
        )

        if self.potential_key in self.adata.obs:
            external = np.asarray(self.adata.obs[self.potential_key].values, dtype=float)
            if self.potential_transform == "none":
                potential = external.copy()
            elif self.potential_transform == "zscore":
                potential = calibrate_external_potential(external, method="zscore")
                warnings.warn(
                    "potential_transform='zscore' 会改变 U 的绝对尺度；"
                    "跨训练 run / 数据集比较 barrier 或 action 时慎用。",
                    UserWarning,
                    stacklevel=2,
                )
            elif self.potential_transform == "neg_log_p":
                potential = calibrate_external_potential(external, log_p, method="neg_log_p")
                warnings.warn(
                    "外部势能已线性校准到 embedding KDE quasi -log P；"
                    "train_model 学到的 U(z,t) 通常应使用 potential_transform='none'。",
                    UserWarning,
                    stacklevel=2,
                )
        else:
            potential = kde_u
            print("未找到外部势能，已用 embedding 空间 KDE 估计 U = -log P")

        return potential, log_p

    def _get_embedding_velocity_data(self):
        """
        Embedding-space velocity only (e.g. obsm['velocity_umap']).

        Gene-space layers such as predicted_delta / velocity are intentionally
        excluded: they live in expression space and cannot be used directly as
        UMAP/PCA drift without projection.

        Requires explicit embedding_velocity_key matching embedding_2d_key
        (e.g. X_umap + velocity_umap, X_pca2 + velocity_pca2).
        """
        if not self.use_embedding_velocity:
            if "predicted_delta" in self.adata.layers or "velocity" in self.adata.layers:
                warnings.warn(
                    "检测到 gene-space predicted_delta/velocity layer，但默认不用于 embedding LAP；"
                    "请设置 use_embedding_velocity=True 并显式传入匹配的 embedding_velocity_key，"
                    "或保持 lap_force_mode='gradient' 仅使用 -∇U。",
                    UserWarning,
                    stacklevel=2,
                )
            return None

        if not self.embedding_velocity_key:
            raise ValueError(
                "use_embedding_velocity=True 必须显式设置 embedding_velocity_key，"
                "且需与 embedding_2d_key 对应，例如："
                "embedding_2d_key='X_umap', embedding_velocity_key='velocity_umap'。"
            )

        key = self.embedding_velocity_key
        if key not in self.adata.obsm:
            raise ValueError(f"embedding velocity '{key}' 不在 adata.obsm 中")

        if not self._velocity_key_matches_embedding(key, self.embedding_2d_key):
            raise ValueError(
                f"embedding_velocity_key='{key}' 与 embedding_2d_key="
                f"'{self.embedding_2d_key}' 命名空间不一致；"
                "请勿混用 UMAP position 与 PCA velocity 等。"
            )

        vectors = np.asarray(self.adata.obsm[key], dtype=float)
        if vectors.shape[0] != self.adata.n_obs:
            raise ValueError(
                f"embedding velocity '{key}' 细胞数 {vectors.shape[0]} "
                f"与 adata.n_obs={self.adata.n_obs} 不一致"
            )
        if vectors.shape[1] < self.cell_positions_2d.shape[1]:
            raise ValueError(
                f"embedding velocity '{key}' 维度 {vectors.shape[1]} "
                f"小于 embedding 维度 {self.cell_positions_2d.shape[1]}"
            )
        return vectors

    # ----------------------------------------------------------- field builders
    def _create_continuous_fields(self):
        if (
            self.field_method == "dynamo"
            and self.field_analyzer is not None
            and self.embedding_velocity is not None
            and self.use_embedding_velocity
        ):
            self._create_dynamo_fields()
        else:
            self._create_safe_nd_fields()

    def _create_dynamo_fields(self):
        print("使用 Dynamo 风格场计算（embedding velocity + 凸包裁剪）...")
        try:
            if self.embedding_velocity.shape[1] >= 2:
                self.field_analyzer.compute_vector_field_dynamo_style(
                    self.cell_positions_2d,
                    self.embedding_velocity[:, :2],
                    self.potential_energy.reshape(-1, 1),
                    n_dims=2,
                )
                self.v_func_2d = self._wrap_with_hull(
                    self.field_analyzer.v_interp_2d, self.cell_positions_2d
                )
                if self.field_analyzer.U_interp_2d is not None:
                    self.U_func_2d = self._wrap_with_hull(
                        self.field_analyzer.U_interp_2d, self.cell_positions_2d
                    )
                else:
                    self.U_func_2d, _ = build_safe_scalar_field(
                        self.cell_positions_2d, self.potential_energy
                    )
            if (
                self.embedding_velocity.shape[1] >= 3
                and self.cell_positions_3d is not None
            ):
                self.U_func_3d, _ = build_safe_scalar_field(
                    self.cell_positions_3d, self.potential_energy
                )
                self.v_func_3d, _ = build_safe_vector_field(
                    self.cell_positions_3d, self.embedding_velocity[:, :3]
                )
        except Exception as exc:
            print(f"Dynamo 场计算失败 ({exc})，回退到安全插值。")
            self._create_safe_nd_fields()

    def _create_safe_nd_fields(self):
        print("使用安全插值场（LinearND + 凸包投影）...")
        self.U_func_2d, _ = build_safe_scalar_field(
            self.cell_positions_2d, self.potential_energy
        )
        if self.cell_positions_3d is not None:
            self.U_func_3d, _ = build_safe_scalar_field(
                self.cell_positions_3d, self.potential_energy
            )
        if self.embedding_velocity is not None and self.use_embedding_velocity:
            self.v_func_2d, _ = build_safe_vector_field(
                self.cell_positions_2d, self.embedding_velocity[:, :2]
            )
            if (
                self.embedding_velocity.shape[1] >= 3
                and self.cell_positions_3d is not None
            ):
                self.v_func_3d, _ = build_safe_vector_field(
                    self.cell_positions_3d, self.embedding_velocity[:, :3]
                )

    def _wrap_with_hull(self, func, positions):
        nbrs = NearestNeighbors(n_neighbors=1).fit(positions)

        def wrapped(query):
            q = np.asarray(query, dtype=float)
            single = q.ndim == 1
            if single:
                q = q.reshape(1, -1)
            try:
                out = func(q)
                if single and np.ndim(out) == 1 and out.shape[0] == q.shape[1]:
                    pass
                elif single:
                    out = np.atleast_1d(out)
            except Exception:
                _, idx = nbrs.kneighbors(q)
                if func is self.field_analyzer.U_interp_2d:
                    out = self.potential_energy[idx.flatten()]
                else:
                    out = self.embedding_velocity[idx.flatten(), :2]
            return out[0] if single else out

        return wrapped

    def _lap_force_at(self, position, use_3d=False):
        self._ensure_3d_available(use_3d)
        pos = np.asarray(position, dtype=float).reshape(-1)
        U_func = self.U_func_3d if use_3d else self.U_func_2d
        spacing = self.neighbor_spacing_3d if use_3d else self.neighbor_spacing_2d

        if self.lap_force_mode == "gradient":
            return -numerical_gradient(U_func, pos, spacing_hint=spacing)

        if self.lap_force_mode == "total":
            return self.compute_force_field(pos.reshape(1, -1), use_3d=use_3d)[0]

        raise ValueError(f"未知 lap_force_mode: {self.lap_force_mode!r}")

    # -------------------------------------------------------------- force field
    def compute_force_field(self, positions, use_3d=False, return_components=False):
        """
        F = -∇U + (v_embed - (-∇U)) when embedding velocity is enabled.

        Embedding velocity must live in the same space as positions (obsm),
        not gene-space predicted_delta.
        """
        self._ensure_3d_available(use_3d)
        positions = np.asarray(positions, dtype=float)
        if positions.ndim == 1:
            positions = positions.reshape(1, -1)

        U_func = self.U_func_3d if use_3d else self.U_func_2d
        v_func = getattr(self, "v_func_3d" if use_3d else "v_func_2d", None)
        spacing = self.neighbor_spacing_3d if use_3d else self.neighbor_spacing_2d

        total, grad_part, flux_part = batch_force_field(
            positions, U_func, v_func, spacing_hint=spacing
        )

        if return_components:
            return total, grad_part, flux_part
        return total

    # ----------------------------------------------------------- LAP / TS / EPR
    def compute_least_action_path(
        self,
        start_pos,
        end_pos,
        n_points=50,
        use_3d=False,
        project_to_manifold=False,
        max_iter=500,
        use_hamiltonian_lap=True,
        use_ensemble=False,
    ):
        """Flow-space Hamiltonian action path via ODE rollout (no graph / discrete LAP)."""
        self._ensure_3d_available(use_3d)
        U_func = self.U_func_3d if use_3d else self.U_func_2d
        gamma = float(getattr(self, "hamiltonian_damping_gamma", 0.1))

        from flow_space_lap import compute_flow_space_lap_path

        flow = compute_flow_space_lap_path(
            np.asarray(start_pos, dtype=float),
            np.asarray(end_pos, dtype=float),
            U_func,
            n_points=n_points,
            gamma=gamma,
            use_ensemble=use_ensemble,
        )
        path = flow["path"]
        action_along_path = flow["action"]
        total_action = flow["total_action"]
        path_potential = flow.get("potential", np.array([U_func(p) for p in path]))
        transition_state_idx = identify_transition_state(path, U_func, action_along_path)

        return {
            "path": path,
            "potential": path_potential,
            "force": np.array([self._lap_force_at(p, use_3d=use_3d) for p in path]),
            "momentum": flow.get("momentum"),
            "action": action_along_path,
            "transition_state_idx": transition_state_idx,
            "transition_state_idx_potential": transition_state_idx,
            "total_action": total_action,
            "success": flow.get("success", False),
            "path_degenerate": flow.get("path_degenerate", False),
            "is_degenerate": flow.get("is_degenerate", False),
            "flow_degeneracy": flow.get("flow_degeneracy"),
            "flow_mismatch": flow.get("flow_mismatch"),
            "action_method": "flow_space_hamiltonian",
            "lap_method": "flow_space_ode",
            "path_method_used": flow.get("path_method_used", "flow_space_ode"),
            "path_description": (
                "Flow-space Hamiltonian ODE trajectory in model latent PCA compute space "
                f"({self.embedding_2d_key}); UMAP is display-only"
            ),
        }

    def compute_least_action_path_adaptive(
        self,
        start_pos,
        end_pos,
        target_spacing_factor=0.5,
        max_points=200,
        min_points=10,
        use_3d=False,
        **kwargs,
    ):
        self._ensure_3d_available(use_3d)
        positions = self.cell_positions_3d if use_3d else self.cell_positions_2d
        n_points = adaptive_path_point_count(
            start_pos,
            end_pos,
            positions,
            target_spacing_factor=target_spacing_factor,
            min_points=min_points,
            max_points=max_points,
        )
        return self.compute_least_action_path(
            start_pos, end_pos, n_points=n_points, use_3d=use_3d, **kwargs
        )

    def compute_flux_logp_alignment(self, path, use_3d=False):
        self._ensure_3d_available(use_3d)
        U_func = self.U_func_3d if use_3d else self.U_func_2d
        v_func = getattr(self, "v_func_3d" if use_3d else "v_func_2d", None)
        spacing = self.neighbor_spacing_3d if use_3d else self.neighbor_spacing_2d
        return compute_flux_logp_alignment_along_path(
            path,
            U_func,
            v_func if self.use_embedding_velocity else None,
            diffusion=self.diffusion_coefficient,
            spacing_hint=spacing,
        )

    def compute_entropy_production_rate(self, path, use_3d=False, non_negative=True):
        self._ensure_3d_available(use_3d)
        U_func = self.U_func_3d if use_3d else self.U_func_2d
        v_func = getattr(self, "v_func_3d" if use_3d else "v_func_2d", None)
        spacing = self.neighbor_spacing_3d if use_3d else self.neighbor_spacing_2d
        if non_negative:
            return compute_entropy_production_approx_along_path(
                path,
                U_func,
                v_func if self.use_embedding_velocity else None,
                diffusion=self.diffusion_coefficient,
                spacing_hint=spacing,
            )
        return compute_flux_logp_alignment_along_path(
            path,
            U_func,
            v_func if self.use_embedding_velocity else None,
            diffusion=self.diffusion_coefficient,
            spacing_hint=spacing,
        )

    def _evaluate_potential(self, position, use_3d=False):
        self._ensure_3d_available(use_3d)
        func = self.U_func_3d if use_3d else self.U_func_2d
        return func(np.asarray(position, dtype=float))

    # ---------------------------------------------------------- cell states
    def identify_cell_states(
        self,
        clustering_key=None,
        max_pot=False,
        use_3d=False,
        endpoint_mode=None,
        start_state=None,
        end_state=None,
        core_fraction=0.5,
    ):
        self._ensure_3d_available(use_3d)
        if clustering_key and clustering_key in self.adata.obs:
            clusters = self.adata.obs[clustering_key].astype("category")
            labels = clusters.values
        else:
            if "neighbors" not in self.adata.uns:
                sc.pp.neighbors(self.adata)
            sc.tl.leiden(self.adata, resolution=1.0)
            labels = self.adata.obs["leiden"].values

        positions = self.cell_positions_3d if use_3d else self.cell_positions_2d
        if endpoint_mode is None:
            endpoint_mode = "max_potential" if max_pot else "min_potential"

        pseudotime = None
        if endpoint_mode == "pseudotime_quantile" and "pseudotime" in self.adata.obs:
            pseudotime = np.asarray(self.adata.obs["pseudotime"].values, dtype=float)

        cell_states = identify_cluster_endpoints(
            positions,
            self.potential_energy,
            labels,
            endpoint_mode=endpoint_mode,
            start_state=start_state,
            end_state=end_state,
            pseudotime=pseudotime,
            core_fraction=core_fraction,
        )

        print(
            f"识别到 {len(cell_states)} 个细胞状态 ({'3D' if use_3d else '2D'}, "
            f"endpoint_mode={endpoint_mode})"
        )
        if use_3d:
            self.cell_states_3d = cell_states
        else:
            self.cell_states_2d = cell_states
        return cell_states

    def compute_multiple_paths(self, path_pairs, use_3d=False, n_points=100, **kwargs):
        self._ensure_3d_available(use_3d)
        cell_states = getattr(self, "cell_states_3d" if use_3d else "cell_states_2d", None)
        if cell_states is None:
            raise ValueError("请先调用 identify_cell_states()")

        all_path_results = {}
        for start_state, end_state in path_pairs:
            if start_state not in cell_states or end_state not in cell_states:
                print(f"警告: 跳过 {start_state}->{end_state}，状态未找到")
                continue

            path_key = f"{start_state}->{end_state}"
            print(f"计算路径: {path_key}")
            result = self.compute_least_action_path(
                cell_states[start_state]["position"],
                cell_states[end_state]["position"],
                n_points=n_points,
                use_3d=use_3d,
                **kwargs,
            )
            result.update(
                {
                    "start_state": start_state,
                    "end_state": end_state,
                    "path_key": path_key,
                }
            )
            all_path_results[path_key] = result
        return all_path_results
