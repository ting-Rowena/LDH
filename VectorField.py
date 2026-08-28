"""
Optional embedding velocity smoothing / visualization helper.

This module is NOT the core mathematical dependency of the landscape pipeline.
Prefer train_model learned potential + landscape_core.build_safe_vector_field()
for production LAP / force-field analysis.

When field_method='dynamo' is explicitly requested, VectorFieldAnalyzer provides
grid-based velocity smoothing and optional streamline plots. It does NOT infer
potential from vector-field divergence (non-conservative flows make that unreliable).
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np
from scipy import interpolate
from scipy.interpolate import RegularGridInterpolator
from sklearn.neighbors import NearestNeighbors


class VectorFieldAnalyzer:
    """Embedding-space vector field smoothing (experimental visualization helper)."""

    DEFAULT_GRID_POINTS_2D = 100
    DEFAULT_GRID_POINTS_3D = 30
    DEFAULT_MAX_INTERPOLATION_POINTS = 5000
    DEFAULT_RBF_NEIGHBORS = 50

    def __init__(
        self,
        n_neighbors: int = 30,
        smoothness: float = 0.5,
        grid_points: int = 100,
        grid_points_3d: Optional[int] = None,
        max_interpolation_points: int = DEFAULT_MAX_INTERPOLATION_POINTS,
        rbf_neighbors: int = DEFAULT_RBF_NEIGHBORS,
    ):
        """
        Parameters
        ----------
        n_neighbors : int
            Neighbors for local velocity smoothing.
        smoothness : float
            Kept for API compatibility (unused by local RBF path).
        grid_points : int
            Grid resolution for 2D field evaluation.
        grid_points_3d : int, optional
            Grid resolution for 3D mode (capped separately to limit memory).
        max_interpolation_points : int
            Downsample cells beyond this count before RBF fitting.
        rbf_neighbors : int
            Local RBF neighbor count (scipy RBFInterpolator).
        """
        self.n_neighbors = n_neighbors
        self.smoothness = smoothness
        self.grid_points = grid_points
        self.grid_points_3d = (
            grid_points_3d
            if grid_points_3d is not None
            else min(grid_points, self.DEFAULT_GRID_POINTS_3D)
        )
        self.max_interpolation_points = max_interpolation_points
        self.rbf_neighbors = rbf_neighbors

        self.vector_field_2d = None
        self.vector_field_3d = None
        self.potential_field_2d = None
        self.potential_field_3d = None
        self.grid_x_2d = None
        self.grid_y_2d = None
        self.grid_x_3d = None
        self.grid_y_3d = None
        self.grid_z_3d = None
        self.positions_2d = None
        self.positions_3d = None

        self.v_interp_2d = None
        self.v_interp_3d = None
        self.U_interp_2d = None
        self.U_interp_3d = None

    @staticmethod
    def _require_minimum_cells(n_cells: int, min_cells: int = 3) -> None:
        if n_cells < min_cells:
            raise ValueError(
                f"Need at least {min_cells} cells to estimate vector field; got {n_cells}."
            )

    @staticmethod
    def _subsample(
        positions: np.ndarray,
        values: np.ndarray,
        max_points: int,
        rng: Optional[np.random.Generator] = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        n = positions.shape[0]
        if n <= max_points:
            return positions, values
        rng = rng or np.random.default_rng(0)
        idx = rng.choice(n, size=max_points, replace=False)
        return positions[idx], values[idx]

    def _smooth_velocities_knn(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
    ) -> np.ndarray:
        n_cells = positions.shape[0]
        self._require_minimum_cells(n_cells)

        k = min(self.n_neighbors, n_cells - 1)
        knn = NearestNeighbors(n_neighbors=k)
        knn.fit(positions)
        distances, indices = knn.kneighbors(positions)

        if k >= 2:
            sigma = float(np.median(distances[:, 1:])) + 1e-6
        else:
            sigma = float(np.median(distances)) + 1e-6

        weights = np.exp(-(distances ** 2) / (2.0 * sigma ** 2))
        weights /= weights.sum(axis=1, keepdims=True) + 1e-10

        smoothed = np.zeros_like(velocities)
        for i in range(n_cells):
            neighbor_idx = indices[i]
            smoothed[i] = np.average(
                velocities[neighbor_idx],
                axis=0,
                weights=weights[i, : len(neighbor_idx)],
            )
        return smoothed

    def _build_rbf_interpolator(
        self,
        points: np.ndarray,
        values: np.ndarray,
    ):
        pts, vals = self._subsample(points, values, self.max_interpolation_points)
        n_pts = pts.shape[0]
        if points.shape[0] > n_pts:
            warnings.warn(
                f"Downsampling {points.shape[0]} cells to {n_pts} for RBF interpolation.",
                UserWarning,
                stacklevel=3,
            )
        neighbors = min(self.rbf_neighbors, n_pts)
        return interpolate.RBFInterpolator(
            pts,
            vals,
            kernel="thin_plate_spline",
            neighbors=neighbors,
        )

    @staticmethod
    def _create_potential_interpolator_2d(
        grid_x: np.ndarray,
        grid_y: np.ndarray,
        potential: np.ndarray,
    ):
        """Bilinear-style lookup via RegularGridInterpolator on a regular mesh."""
        y_coords = grid_y[:, 0]
        x_coords = grid_x[0, :]
        reg = RegularGridInterpolator(
            (y_coords, x_coords),
            potential,
            bounds_error=False,
            fill_value=np.nan,
        )

        def interpolator(x):
            x = np.asarray(x, dtype=float)
            single = x.ndim == 1
            if single:
                x = x.reshape(1, -1)
            query = np.column_stack([x[:, 1], x[:, 0]])
            out = reg(query)
            return float(out[0]) if single else out

        return interpolator

    def compute_vector_field_dynamo_style(
        self,
        positions: np.ndarray,
        velocities: np.ndarray,
        potentials: Optional[np.ndarray] = None,
        n_dims: int = 2,
    ) -> dict:
        """
        Smooth embedding velocities and evaluate them on a regular grid.

        Notes
        -----
        - Potential is NOT inferred from divergence (experimental Poisson path removed).
        - Pass external ``potentials`` (e.g. train_model output) when a scalar field is needed.
        """
        positions = np.asarray(positions, dtype=float)
        velocities = np.asarray(velocities, dtype=float)
        n_cells, n_comp = velocities.shape
        self._require_minimum_cells(n_cells)

        smoothed_velocities = self._smooth_velocities_knn(positions, velocities)

        if n_dims == 2:
            self.positions_2d = positions.copy()
            grid_n = self.grid_points
            grid_x = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_n)
            grid_y = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_n)
            grid_x_mesh, grid_y_mesh = np.meshgrid(grid_x, grid_y)
            grid_points_2d = np.column_stack([grid_x_mesh.ravel(), grid_y_mesh.ravel()])

            self.v_interp_2d = self._build_rbf_interpolator(
                positions, smoothed_velocities[:, :2]
            )
            vector_field_2d = self.v_interp_2d(grid_points_2d).reshape(grid_n, grid_n, 2)

            potential_field = None
            self.U_interp_2d = None
            if potentials is not None:
                pot = np.asarray(potentials, dtype=float).reshape(-1)
                if pot.shape[0] != n_cells:
                    raise ValueError(
                        f"potentials length {pot.shape[0]} != n_cells {n_cells}"
                    )
                pot_interp = self._build_rbf_interpolator(
                    positions, pot.reshape(-1, 1)
                )
                potential_field = pot_interp(grid_points_2d).reshape(grid_n, grid_n)
                self.U_interp_2d = self._create_potential_interpolator_2d(
                    grid_x_mesh, grid_y_mesh, potential_field
                )
            else:
                warnings.warn(
                    "No potential provided; skip potential field estimation from vector field. "
                    "Use train_model / KDE potential externally.",
                    UserWarning,
                    stacklevel=2,
                )

            self.grid_x_2d = grid_x_mesh
            self.grid_y_2d = grid_y_mesh
            self.vector_field_2d = vector_field_2d
            self.potential_field_2d = potential_field

            return {
                "grid_x": grid_x_mesh,
                "grid_y": grid_y_mesh,
                "vector_field": vector_field_2d,
                "potential_field": potential_field,
                "v_interp": self.v_interp_2d,
                "U_interp": self.U_interp_2d,
            }

        if n_dims == 3:
            self.positions_3d = positions.copy()
            grid_n = self.grid_points_3d
            if self.grid_points > self.DEFAULT_GRID_POINTS_3D:
                warnings.warn(
                    f"3D grid capped at {grid_n} (requested grid_points={self.grid_points}) "
                    "to avoid excessive memory use.",
                    UserWarning,
                    stacklevel=2,
                )

            grid_x = np.linspace(positions[:, 0].min(), positions[:, 0].max(), grid_n)
            grid_y = np.linspace(positions[:, 1].min(), positions[:, 1].max(), grid_n)
            grid_z = np.linspace(positions[:, 2].min(), positions[:, 2].max(), grid_n)
            grid_x_mesh, grid_y_mesh, grid_z_mesh = np.meshgrid(grid_x, grid_y, grid_z)
            grid_points_3d = np.column_stack(
                [grid_x_mesh.ravel(), grid_y_mesh.ravel(), grid_z_mesh.ravel()]
            )

            vel3 = smoothed_velocities[:, :3] if n_comp >= 3 else np.pad(
                smoothed_velocities[:, :2],
                ((0, 0), (0, 1)),
                mode="constant",
            )
            self.v_interp_3d = self._build_rbf_interpolator(positions, vel3)
            vector_field_3d = self.v_interp_3d(grid_points_3d).reshape(
                grid_n, grid_n, grid_n, 3
            )

            self.grid_x_3d = grid_x_mesh
            self.grid_y_3d = grid_y_mesh
            self.grid_z_3d = grid_z_mesh
            self.vector_field_3d = vector_field_3d

            return {
                "grid_x": grid_x_mesh,
                "grid_y": grid_y_mesh,
                "grid_z": grid_z_mesh,
                "vector_field": vector_field_3d,
                "v_interp": self.v_interp_3d,
            }

        raise ValueError(f"Unsupported n_dims={n_dims}; use 2 or 3.")

    def compute_streamlines(self, n_streamlines: int = 20, max_length: int = 500):
        """Integrate streamlines; seed points sampled from cell density."""
        if self.vector_field_2d is None or self.v_interp_2d is None:
            raise ValueError("需要先计算 2D 向量场 (compute_vector_field_dynamo_style).")
        if self.positions_2d is None:
            raise ValueError("缺少细胞坐标 positions_2d。")
        if self.grid_x_2d is None or self.grid_y_2d is None:
            raise ValueError("缺少网格坐标 grid_x_2d / grid_y_2d。")

        positions = self.positions_2d
        x_min, x_max = float(self.grid_x_2d.min()), float(self.grid_x_2d.max())
        y_min, y_max = float(self.grid_y_2d.min()), float(self.grid_y_2d.max())

        density, x_edges, y_edges = np.histogram2d(
            positions[:, 0],
            positions[:, 1],
            bins=20,
            range=[[x_min, x_max], [y_min, y_max]],
        )
        density = density + 1e-12
        prob = density.flatten() / density.sum()

        seed_points = []
        for _ in range(n_streamlines):
            idx = np.random.choice(len(prob), p=prob)
            i, j = np.unravel_index(idx, density.shape)
            x = (x_edges[i] + x_edges[i + 1]) / 2.0
            y = (y_edges[j] + y_edges[j + 1]) / 2.0
            seed_points.append([x, y])
        seed_points = np.asarray(seed_points)

        streamlines = []
        dt = 0.1
        for seed in seed_points:
            streamline = [seed.copy()]
            current_pos = seed.copy()
            for _ in range(max_length):
                # RBFInterpolator expects shape (n_query, n_dim)
                v = np.asarray(
                    self.v_interp_2d(np.asarray(current_pos, dtype=float).reshape(1, -1)),
                    dtype=float,
                ).reshape(-1)
                if v.shape[0] < 2 or np.linalg.norm(v[:2]) < 1e-3:
                    break
                current_pos = current_pos + v[:2] * dt
                if (
                    current_pos[0] < x_min
                    or current_pos[0] > x_max
                    or current_pos[1] < y_min
                    or current_pos[1] > y_max
                ):
                    break
                streamline.append(current_pos.copy())
            if len(streamline) > 2:
                streamlines.append(np.asarray(streamline))
        return streamlines

    @staticmethod
    def grid_divergence(grid_x: np.ndarray, grid_y: np.ndarray, vector_field: np.ndarray) -> np.ndarray:
        """Central-difference divergence ∇·v on a regular 2D mesh (ny, nx, 2)."""
        vx = np.asarray(vector_field[..., 0], dtype=float)
        vy = np.asarray(vector_field[..., 1], dtype=float)
        xs = np.asarray(grid_x[0, :], dtype=float)
        ys = np.asarray(grid_y[:, 0], dtype=float)
        dx = float(np.median(np.diff(xs))) if len(xs) > 1 else 1.0
        dy = float(np.median(np.diff(ys))) if len(ys) > 1 else 1.0
        dvx_dx = np.gradient(vx, dx, axis=1)
        dvy_dy = np.gradient(vy, dy, axis=0)
        return dvx_dx + dvy_dy

    def sink_strength_at_points(
        self,
        query_points: np.ndarray,
        *,
        radius: Optional[float] = None,
    ) -> dict:
        """
        Quantify local sink intensity near query points using grid divergence.

        Returns mean/median/min divergence in a neighborhood (more negative = stronger sink)
        and the fraction of nearby grid cells with div < 0 (inflow fraction).
        """
        if self.vector_field_2d is None or self.grid_x_2d is None or self.grid_y_2d is None:
            raise ValueError("Call compute_vector_field_dynamo_style first.")
        query_points = np.asarray(query_points, dtype=float).reshape(-1, 2)
        if query_points.size == 0:
            return {
                "mean_divergence": float("nan"),
                "median_divergence": float("nan"),
                "min_divergence": float("nan"),
                "inflow_fraction": float("nan"),
                "n_grid_cells": 0,
                "radius": float("nan"),
            }

        div = self.grid_divergence(self.grid_x_2d, self.grid_y_2d, self.vector_field_2d)
        gx = self.grid_x_2d.ravel()
        gy = self.grid_y_2d.ravel()
        div_flat = div.ravel()
        if radius is None:
            spacing = float(
                np.median(
                    [
                        np.median(np.diff(self.grid_x_2d[0, :])),
                        np.median(np.diff(self.grid_y_2d[:, 0])),
                    ]
                )
            )
            radius = max(3.0 * spacing, 1e-6)
        mask = np.zeros(div_flat.shape[0], dtype=bool)
        for p in query_points:
            mask |= (gx - p[0]) ** 2 + (gy - p[1]) ** 2 <= radius ** 2
        if not mask.any():
            # fall back to nearest grid cells
            d2 = (gx[:, None] - query_points[None, :, 0]) ** 2 + (
                gy[:, None] - query_points[None, :, 1]
            ) ** 2
            nearest = np.unique(np.argmin(d2, axis=0))
            mask[nearest] = True
        vals = div_flat[mask]
        return {
            "mean_divergence": float(np.nanmean(vals)),
            "median_divergence": float(np.nanmedian(vals)),
            "min_divergence": float(np.nanmin(vals)),
            "inflow_fraction": float(np.mean(vals < 0)),
            "n_grid_cells": int(mask.sum()),
            "radius": float(radius),
            "sink_strength": float(-np.nanmean(vals)),  # positive => net inflow
        }
