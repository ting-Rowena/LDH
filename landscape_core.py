"""
Shared landscape / LAP utilities with corrected non-equilibrium approximations.

Key conventions
---------------
- Potential: U ≈ -log P (normalized quasi-density).
- Drift decomposition on embedding: F = F_grad + F_flux, with
  F_grad = -∇U and F_flux = v_embed - F_grad (NOT RNA velocity / P).
- LAP / graph paths are candidate generators; arc-length least-action is a heuristic path proposal score.
- Transition state: potential barrier (saddle) along path, not cumulative-action maxima.
- Flux–logP alignment is NOT strict entropy production unless using the approx helper.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Optional, Tuple, Union

import numpy as np
from scipy import interpolate, ndimage, optimize, signal
from scipy.spatial import Delaunay
from sklearn.neighbors import KernelDensity, NearestNeighbors


Array = np.ndarray
FieldFunc = Callable[[Array], Array]


@dataclass
class FieldBundle:
    """Continuous scalar / vector fields on a low-dimensional embedding."""

    positions: Array
    potential: Array
    U_func: FieldFunc
    v_func: Optional[FieldFunc] = None
    log_prob: Optional[Array] = None
    neighbor_spacing: float = 1.0
    hull: Optional[Delaunay] = None


def estimate_potential_from_density(
    positions: Array,
    bandwidth: Optional[float] = None,
    normalize: bool = True,
) -> Tuple[Array, Array]:
    """
  Estimate U = -log P from embedding coordinates via KDE.

  Returns (potential, log_prob) where log_prob is log-density up to a constant.
  """
    positions = np.asarray(positions, dtype=float)
    if bandwidth is None:
        nbrs = NearestNeighbors(n_neighbors=min(30, len(positions))).fit(positions)
        dists, _ = nbrs.kneighbors(positions)
        bandwidth = float(np.median(dists[:, 1:]))

    kde = KernelDensity(bandwidth=max(bandwidth, 1e-6), kernel="gaussian")
    kde.fit(positions)
    log_prob = kde.score_samples(positions)

    if normalize:
        log_prob = log_prob - np.max(log_prob)

    potential = -log_prob
    return potential, log_prob


def calibrate_external_potential(
    potential: Array,
    log_prob: Optional[Array] = None,
    method: str = "zscore",
) -> Array:
    """
  Align an external potential with quasi -log P when possible.

  If log_prob is unavailable, only center/scale for numerical stability.
  """
    potential = np.asarray(potential, dtype=float)
    if method == "neg_log_p" and log_prob is not None:
        # Linear least-squares fit: potential ≈ a * (-log_prob) + b
        y = -np.asarray(log_prob, dtype=float)
        x = potential
        A = np.vstack([x, np.ones_like(x)]).T
        coef, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
        return coef[0] * potential + coef[1]

    potential = potential - np.min(potential)
    std = np.std(potential)
    if std > 1e-12:
        potential = potential / std
    return potential


def _build_hull(positions: Array) -> Optional[Delaunay]:
    if positions.shape[1] != 2 or len(positions) < 4:
        return None
    try:
        return Delaunay(positions)
    except Exception:
        return None


def _clip_to_hull(points: Array, hull: Optional[Delaunay], positions: Array) -> Array:
    points = np.asarray(points, dtype=float)
    if points.ndim == 1:
        points = points.reshape(1, -1)

    if hull is None:
        return points

    inside = hull.find_simplex(points) >= 0
    # Project exterior points to nearest observed cell location
    if not np.all(inside):
        nbrs = NearestNeighbors(n_neighbors=1).fit(positions)
        _, idx = nbrs.kneighbors(points[~inside])
        points[~inside] = positions[idx.flatten()]
    return points


def build_knn_scalar_field(
    positions: Array,
    values: Array,
    n_neighbors: int = 30,
) -> FieldFunc:
    """Fast IDW scalar field for high-dimensional LAP (avoids Delaunay/RBF cost)."""
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    k = min(n_neighbors, len(positions))
    nbrs = NearestNeighbors(n_neighbors=k).fit(positions)

    def U_func(query: Array) -> Array:
        q = np.asarray(query, dtype=float)
        single = q.ndim == 1
        if single:
            q = q.reshape(1, -1)
        dist, idx = nbrs.kneighbors(q)
        w = 1.0 / (dist + 1e-8)
        w = w / w.sum(axis=1, keepdims=True)
        out = np.sum(w * values[idx], axis=1)
        return float(out[0]) if single else out

    return U_func


def build_safe_scalar_field(
    positions: Array,
    values: Array,
    method: str = "linear_nd",
    hull_clip: bool = True,
    max_fit_points: int = 2500,
    random_state: int = 0,
) -> Tuple[FieldFunc, Optional[Delaunay]]:
    """Create a scalar interpolator with convex-hull clipping fallback."""
    positions = np.asarray(positions, dtype=float)
    values = np.asarray(values, dtype=float).reshape(-1)
    if positions.shape[1] > 3:
        return build_knn_scalar_field(positions, values), None

    hull = _build_hull(positions) if hull_clip and positions.shape[1] == 2 else None
    nbrs = NearestNeighbors(n_neighbors=1).fit(positions)

    fit_pos = positions
    fit_vals = values

    if method == "rbf":
        interp = interpolate.RBFInterpolator(fit_pos, fit_vals, kernel="linear")
        method = "rbf"
    elif positions.shape[1] == 2:
        try:
            interp = interpolate.CloughTocher2DInterpolator(positions, values)
            method = "ct"
        except Exception:
            interp = interpolate.RBFInterpolator(positions, values, kernel="linear")
            method = "rbf"
    else:
        interp = interpolate.LinearNDInterpolator(positions, values)
        method = "linear_nd"

    def U_func(query: Array) -> Array:
        q = np.asarray(query, dtype=float)
        single = q.ndim == 1
        if single:
            q = q.reshape(1, -1)
        q = _clip_to_hull(q, hull, positions)

        try:
            out = interp(q)
        except Exception:
            out = np.full(q.shape[0], np.nan)

        nan_mask = ~np.isfinite(out)
        if np.any(nan_mask):
            _, idx = nbrs.kneighbors(q[nan_mask])
            out[nan_mask] = values[idx.flatten()]

        return float(out[0]) if single else out

    return U_func, hull


def build_safe_vector_field(
    positions: Array,
    vectors: Array,
    method: str = "linear_nd",
    hull_clip: bool = True,
) -> Tuple[FieldFunc, Optional[Delaunay]]:
    """Vector field interpolator with the same safety guarantees as scalar fields."""
    positions = np.asarray(positions, dtype=float)
    vectors = np.asarray(vectors, dtype=float)
    hull = _build_hull(positions) if hull_clip and positions.shape[1] == 2 else None
    nbrs = NearestNeighbors(n_neighbors=1).fit(positions)

    if method == "rbf" or positions.shape[1] > 3:
        interp = interpolate.RBFInterpolator(positions, vectors, kernel="linear")
    else:
        interp = interpolate.LinearNDInterpolator(positions, vectors)

    def v_func(query: Array) -> Array:
        q = np.asarray(query, dtype=float)
        single = q.ndim == 1
        if single:
            q = q.reshape(1, -1)
        q = _clip_to_hull(q, hull, positions)

        try:
            out = interp(q)
        except Exception:
            out = np.full((q.shape[0], vectors.shape[1]), np.nan)

        nan_rows = ~np.all(np.isfinite(out), axis=1)
        if np.any(nan_rows):
            _, idx = nbrs.kneighbors(q[nan_rows])
            out[nan_rows] = vectors[idx.flatten()]

        return out[0] if single else out

    return v_func, hull


def mean_neighbor_spacing(positions: Array, k: int = 2) -> float:
    nbrs = NearestNeighbors(n_neighbors=min(k + 1, len(positions))).fit(positions)
    dists, _ = nbrs.kneighbors(positions)
    return float(np.median(dists[:, 1]))


def numerical_gradient(
    U_func: FieldFunc,
    position: Array,
    epsilon: Optional[float] = None,
    spacing_hint: float = 1.0,
) -> Array:
    """Central-difference gradient with scale-aware step size."""
    pos = np.asarray(position, dtype=float).reshape(-1)
    dim = pos.shape[0]
    if epsilon is None:
        epsilon = max(spacing_hint * 0.25, 1e-4)

    grad = np.zeros(dim, dtype=float)
    for d in range(dim):
        p_plus, p_minus = pos.copy(), pos.copy()
        p_plus[d] += epsilon
        p_minus[d] -= epsilon
        grad[d] = (U_func(p_plus) - U_func(p_minus)) / (2.0 * epsilon)
    return grad


def compute_force_decomposition(
    position: Array,
    U_func: FieldFunc,
    v_func: Optional[FieldFunc] = None,
    spacing_hint: float = 1.0,
) -> Tuple[Array, Array, Array]:
    """
  Decompose embedding drift into gradient + non-conservative residual.

  Returns (total_force, gradient_force, flux_force).
  """
    pos = np.asarray(position, dtype=float).reshape(-1)
    grad_u = numerical_gradient(U_func, pos, spacing_hint=spacing_hint)
    f_grad = -grad_u

    if v_func is None:
        return f_grad.copy(), f_grad, np.zeros_like(f_grad)

    v_embed = np.asarray(v_func(pos), dtype=float).reshape(-1)
    f_flux = v_embed - f_grad
    f_total = f_grad + f_flux
    return f_total, f_grad, f_flux


def batch_force_field(
    positions: Array,
    U_func: FieldFunc,
    v_func: Optional[FieldFunc] = None,
    spacing_hint: float = 1.0,
) -> Tuple[Array, Array, Array]:
    positions = np.asarray(positions, dtype=float)
    n = len(positions)
    dim = positions.shape[1]
    total = np.zeros((n, dim))
    grad_part = np.zeros((n, dim))
    flux_part = np.zeros((n, dim))
    for i in range(n):
        total[i], grad_part[i], flux_part[i] = compute_force_decomposition(
            positions[i], U_func, v_func, spacing_hint=spacing_hint
        )
    return total, grad_part, flux_part


def _reparameterize_by_arclength(path: Array) -> Tuple[Array, Array]:
    segments = np.linalg.norm(np.diff(path, axis=0), axis=1)
    s = np.concatenate([[0.0], np.cumsum(segments)])
    if s[-1] < 1e-12:
        return path.copy(), np.linspace(0.0, 1.0, len(path))
    uniform = np.linspace(0.0, s[-1], len(path))
    new_path = np.zeros_like(path)
    for d in range(path.shape[1]):
        new_path[:, d] = np.interp(uniform, s, path[:, d])
    return new_path, uniform


def _project_path_to_manifold(path: Array, positions: Array, k: int = 1) -> Array:
    nbrs = NearestNeighbors(n_neighbors=k).fit(positions)
    _, idx = nbrs.kneighbors(path)
    if k == 1:
        return positions[idx.flatten()]
    # Soft projection: weighted average of k neighbors
    dists, idx = nbrs.kneighbors(path)
    weights = 1.0 / (dists + 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    projected = np.zeros_like(path)
    for i in range(len(path)):
        projected[i] = np.average(positions[idx[i]], axis=0, weights=weights[i])
    return projected


def effective_scalar_diffusion(
    sigma: Array,
    default: float = 0.1,
) -> float:
    """
    Map diagonal diffusion σ to a scalar D_eff ≈ 0.5 * mean(σ²).

    train_model.py learns state-dependent σ(z, t); LAP here uses one scalar D
    as an external approximation unless a per-cell effective value is supplied.
    """
    sigma = np.asarray(sigma, dtype=float).reshape(-1)
    if sigma.size == 0 or not np.all(np.isfinite(sigma)):
        return float(default)
    return float(0.5 * np.mean(sigma ** 2))


def resolve_scalar_diffusion(
    adata=None,
    diffusion: Optional[float] = None,
    default: float = 0.1,
) -> Tuple[float, str]:
    """
    Choose scalar diffusion D for LAP.

    Priority: explicit user value > median adata.obs['diffusion_eff'] > default.
    """
    if diffusion is not None:
        return float(diffusion), "user"
    if adata is not None and "diffusion_eff" in getattr(adata, "obs", {}):
        values = np.asarray(adata.obs["diffusion_eff"], dtype=float)
        values = values[np.isfinite(values)]
        if values.size > 0:
            return float(np.median(values)), "model_median"
    return float(default), "default_fixed_approx"


def arc_length_least_action_score(
    path: Array,
    force_func: Callable[[Array], Array],
    diffusion: float,
) -> float:
    """
    Heuristic arc-length least-action path proposal score.

    - Heuristic relative score for LAP/graph candidate path generation and visualization.
    - Path proposal score along arc-length parameterization (unit tangent velocity).
    - Not a strict Freidlin-Wentzell transition probability or exact large-deviation rate.

    Accumulates S ≈ ∫ |ẋ - F(x)|² / (4D) ds with ẋ ≈ Δx/|Δx|. Suitable for comparing
    candidate paths; do not interpret S as a rigorous transition rate.
    """
    path = np.asarray(path, dtype=float)
    D = max(float(diffusion), 1e-12)
    action = 0.0
    for i in range(len(path) - 1):
        dx = path[i + 1] - path[i]
        ds = np.linalg.norm(dx)
        if ds < 1e-12:
            continue
        vel = dx / ds
        midpoint = 0.5 * (path[i] + path[i + 1])
        force = np.asarray(force_func(midpoint), dtype=float).reshape(-1)
        action += np.sum((vel - force) ** 2) / (4.0 * D) * ds
    return action


def freidlin_wentzell_action(
    path: Array,
    force_func: Callable[[Array], Array],
    diffusion: float,
) -> float:
    """Deprecated alias for :func:`arc_length_least_action_score`."""
    warnings.warn(
        "freidlin_wentzell_action is a heuristic arc-length path proposal score, not strict "
        "FW action. Use diffusion_weighted_action_score for model-based dynamics diagnostics.",
        DeprecationWarning,
        stacklevel=2,
    )
    return arc_length_least_action_score(path, force_func, diffusion)


def action_profile_along_path(
    path: Array,
    force_func: Callable[[Array], Array],
    diffusion: float,
) -> Array:
    """Cumulative arc-length least-action profile along a path."""
    path = np.asarray(path, dtype=float)
    D = max(float(diffusion), 1e-12)
    profile = np.zeros(len(path))
    cumulative = 0.0
    for i in range(1, len(path)):
        dx = path[i] - path[i - 1]
        ds = np.linalg.norm(dx)
        if ds < 1e-12:
            profile[i] = cumulative
            continue
        vel = dx / ds
        midpoint = 0.5 * (path[i] + path[i - 1])
        force = np.asarray(force_func(midpoint), dtype=float).reshape(-1)
        cumulative += np.sum((vel - force) ** 2) / (4.0 * D) * ds
        profile[i] = cumulative
    return profile


def identify_transition_state(
    path: Array,
    U_func: FieldFunc,
    action_profile: Optional[Array] = None,
    exclude_fraction: float = 0.1,
) -> int:
    """
  Transition state = potential barrier along LAP (interior maximum of U).

  Cumulative action is monotonic and MUST NOT be used for TS detection.
  """
    path = np.asarray(path, dtype=float)
    n = len(path)
    potentials = np.array([U_func(p) for p in path])

    lo = int(n * exclude_fraction)
    hi = max(lo + 1, int(n * (1.0 - exclude_fraction)))
    interior = potentials[lo:hi]
    if len(interior) == 0:
        return int(np.argmax(potentials))

    rel_idx = int(np.argmax(interior))
    ts_idx = lo + rel_idx

    # Tie-break: if action profile provided, prefer highest U among near-maxima
    if action_profile is not None and n > 5:
        thresh = np.max(interior) - 0.05 * (np.max(interior) - np.min(interior))
        candidates = np.where(potentials[lo:hi] >= thresh)[0] + lo
        if len(candidates) > 0:
            ts_idx = int(candidates[np.argmax(potentials[candidates])])

    return min(max(int(ts_idx), 0), n - 1)


def compute_flux_logp_alignment_along_path(
    path: Array,
    U_func: FieldFunc,
    v_func: Optional[FieldFunc],
    log_prob_values: Optional[Array] = None,
    positions: Optional[Array] = None,
    diffusion: float = 0.1,
    spacing_hint: float = 1.0,
) -> Tuple[float, Array]:
    """
    Signed alignment score flux · ∇(log P) with flux ≈ v - D ∇(log P).

    This is a diagnostic alignment measure and may be negative; it is NOT a
    guaranteed non-negative entropy production rate.
    """
    path = np.asarray(path, dtype=float)
    D = max(float(diffusion), 1e-12)
    alignment = np.zeros(len(path))

    for i, pos in enumerate(path):
        grad_u = numerical_gradient(U_func, pos, spacing_hint=spacing_hint)
        grad_log_p = -grad_u  # when U = -log P

        if v_func is not None:
            v = np.asarray(v_func(pos), dtype=float).reshape(-1)
        else:
            v = -grad_u

        flux = v - D * grad_log_p
        alignment[i] = float(np.dot(flux, grad_log_p))

    return float(np.mean(alignment)), alignment


def compute_entropy_production_approx_along_path(
    path: Array,
    U_func: FieldFunc,
    v_func: Optional[FieldFunc],
    diffusion: float = 0.1,
    spacing_hint: float = 1.0,
) -> Tuple[float, Array]:
    """
    Non-negative entropy production proxy: ||J/P||² / D with J/P ≈ v - D ∇(log P).

    Still an approximation; uses scalar D and embedding drift when provided.
    """
    path = np.asarray(path, dtype=float)
    D = max(float(diffusion), 1e-12)
    epr = np.zeros(len(path))

    for i, pos in enumerate(path):
        grad_u = numerical_gradient(U_func, pos, spacing_hint=spacing_hint)
        grad_log_p = -grad_u

        if v_func is not None:
            v = np.asarray(v_func(pos), dtype=float).reshape(-1)
        else:
            v = -grad_u

        flux_over_p = v - D * grad_log_p
        epr[i] = float(np.sum(flux_over_p ** 2) / D)

    return float(np.mean(epr)), epr


def compute_entropy_production_along_path(
    path: Array,
    U_func: FieldFunc,
    v_func: Optional[FieldFunc],
    log_prob_values: Optional[Array] = None,
    positions: Optional[Array] = None,
    diffusion: float = 0.1,
    spacing_hint: float = 1.0,
    non_negative: bool = True,
) -> Tuple[float, Array]:
    """
    Backward-compatible wrapper.

    By default returns the non-negative approximation; set non_negative=False
    for the signed flux-logP alignment score.
    """
    if non_negative:
        return compute_entropy_production_approx_along_path(
            path, U_func, v_func, diffusion=diffusion, spacing_hint=spacing_hint
        )
    return compute_flux_logp_alignment_along_path(
        path,
        U_func,
        v_func,
        log_prob_values=log_prob_values,
        positions=positions,
        diffusion=diffusion,
        spacing_hint=spacing_hint,
    )


def _pin_path_endpoints(path: Array, start: Array, end: Array) -> Array:
    path = np.asarray(path, dtype=float).copy()
    path[0] = start
    path[-1] = end
    return path


def optimize_least_action_path(
    start: Array,
    end: Array,
    force_func: Callable[[Array], Array],
    positions: Array,
    n_points: int = 50,
    diffusion: float = 0.1,
    project_to_manifold: bool = True,
    max_iter: int = 500,
) -> Tuple[Array, float, bool]:
    """Optimize LAP with arc-length score and optional interior manifold projection."""
    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    dim = start.shape[0]
    if dim > 3:
        max_iter = min(max_iter, 120)
        n_points = min(n_points, 40)

    init = np.linspace(start, end, n_points)
    bounds_lo = positions.min(axis=0)
    bounds_hi = positions.max(axis=0)

    def objective(flat):
        path = flat.reshape(n_points, dim)
        path = _pin_path_endpoints(path, start, end)
        if project_to_manifold and n_points > 2:
            path[1:-1] = _project_path_to_manifold(path[1:-1], positions)
        path, _ = _reparameterize_by_arclength(path)
        path = _pin_path_endpoints(path, start, end)
        return arc_length_least_action_score(path, force_func, diffusion)

    x0 = init.flatten()
    bnds = []
    for i in range(n_points):
        for d in range(dim):
            if i == 0:
                bnds.append((start[d], start[d]))
            elif i == n_points - 1:
                bnds.append((end[d], end[d]))
            else:
                bnds.append((bounds_lo[d], bounds_hi[d]))

    result = optimize.minimize(objective, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": max_iter})
    path = result.x.reshape(n_points, dim)
    path = _pin_path_endpoints(path, start, end)
    if project_to_manifold and n_points > 2:
        path[1:-1] = _project_path_to_manifold(path[1:-1], positions)
    path, _ = _reparameterize_by_arclength(path)
    path = _pin_path_endpoints(path, start, end)
    if project_to_manifold and n_points > 2:
        path[1:-1] = _project_path_to_manifold(path[1:-1], positions)
    path = _pin_path_endpoints(path, start, end)
    action = arc_length_least_action_score(path, force_func, diffusion)
    return path, action, bool(result.success)


def adaptive_path_point_count(
    start: Array,
    end: Array,
    positions: Array,
    target_spacing_factor: float = 0.5,
    min_points: int = 10,
    max_points: int = 200,
) -> int:
    dist = np.linalg.norm(end - start)
    spacing = mean_neighbor_spacing(positions) * target_spacing_factor
    return int(np.clip(dist / max(spacing, 1e-6), min_points, max_points))


def identify_attractors_from_clusters(
    positions: Array,
    potential: Array,
    cluster_labels: Array,
    use_min_potential: bool = True,
) -> dict:
    """Per-cluster attractor = minimum-potential cell (default) in embedding."""
    mode = "min_potential" if use_min_potential else "max_potential"
    return identify_cluster_endpoints(
        positions,
        potential,
        cluster_labels,
        endpoint_mode=mode,
    )


VALID_ENDPOINT_MODES = frozenset(
    {
        "min_potential",
        "max_potential",
        "medoid",
        "pseudotime_quantile",
        "min_potential_core",
    }
)


def _cluster_role(cluster_id, start_state, end_state) -> str:
    if start_state is not None and str(cluster_id) == str(start_state):
        return "start"
    if end_state is not None and str(cluster_id) == str(end_state):
        return "end"
    return "any"


def pick_cluster_representative_index(
    positions: Array,
    potential: Array,
    endpoint_mode: str,
    *,
    pseudotime: Optional[Array] = None,
    role: str = "any",
    core_fraction: float = 0.5,
    start_quantile: float = 0.25,
    end_quantile: float = 0.75,
) -> int:
    """Return index within the masked cluster subset (local index)."""
    if endpoint_mode not in VALID_ENDPOINT_MODES:
        raise ValueError(
            f"endpoint_mode must be one of {sorted(VALID_ENDPOINT_MODES)}, got {endpoint_mode!r}"
        )

    positions = np.asarray(positions, dtype=float)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    n = len(positions)
    if n == 0:
        raise ValueError("Cannot pick representative from empty cluster")

    if endpoint_mode == "min_potential":
        return int(np.argmin(potential))
    if endpoint_mode == "max_potential":
        return int(np.argmax(potential))

    centroid = positions.mean(axis=0)
    dist = np.linalg.norm(positions - centroid, axis=1)

    if endpoint_mode == "medoid":
        return int(np.argmin(dist))

    if endpoint_mode == "min_potential_core":
        thresh = np.quantile(dist, core_fraction) if n > 1 else dist.max()
        core = potential[dist <= thresh]
        core_idx = np.where(dist <= thresh)[0]
        if len(core_idx) == 0:
            return int(np.argmin(potential))
        local = core_idx[int(np.argmin(potential[dist <= thresh]))]
        return int(local)

    if endpoint_mode == "pseudotime_quantile":
        if pseudotime is None:
            raise ValueError("pseudotime_quantile requires pseudotime values")
        pt = np.asarray(pseudotime, dtype=float).reshape(-1)
        if role == "start":
            target = float(np.quantile(pt, start_quantile))
        elif role == "end":
            target = float(np.quantile(pt, end_quantile))
        else:
            target = float(np.median(pt))
        return int(np.argmin(np.abs(pt - target)))

    raise ValueError(f"Unhandled endpoint_mode: {endpoint_mode!r}")


def identify_cluster_endpoints(
    positions: Array,
    potential: Array,
    cluster_labels: Array,
    endpoint_mode: str = "min_potential",
    *,
    start_state=None,
    end_state=None,
    pseudotime: Optional[Array] = None,
    core_fraction: float = 0.5,
    start_quantile: float = 0.25,
    end_quantile: float = 0.75,
) -> dict:
    """Pick one representative cell per cluster for LAP endpoints."""
    positions = np.asarray(positions, dtype=float)
    potential = np.asarray(potential, dtype=float).reshape(-1)
    cluster_labels = np.asarray(cluster_labels)
    states = {}

    for cluster_id in np.unique(cluster_labels):
        mask = cluster_labels == cluster_id
        role = _cluster_role(cluster_id, start_state, end_state)
        idx_local = pick_cluster_representative_index(
            positions[mask],
            potential[mask],
            endpoint_mode,
            pseudotime=pseudotime[mask] if pseudotime is not None else None,
            role=role,
            core_fraction=core_fraction,
            start_quantile=start_quantile,
            end_quantile=end_quantile,
        )
        global_idx = int(np.where(mask)[0][idx_local])
        states[cluster_id] = {
            "position": positions[global_idx],
            "cell_index": global_idx,
            "potential": float(potential[global_idx]),
            "size": int(mask.sum()),
            "endpoint_mode": endpoint_mode,
            "role": role,
        }
    return states


def stage_core_cell_indices(
    positions: Array,
    cluster_labels: Array,
    stage,
    *,
    pseudotime: Optional[Array] = None,
    core_fraction: float = 0.5,
    by: str = "medoid",
) -> Array:
    """Global cell indices in the stage core used for endpoint bootstrap."""
    cluster_labels = np.asarray(cluster_labels)
    mask = cluster_labels == stage
    if not np.any(mask):
        return np.array([], dtype=int)

    pos = np.asarray(positions[mask], dtype=float)
    global_idx = np.where(mask)[0]

    if by == "pseudotime" and pseudotime is not None:
        pt = np.asarray(pseudotime[mask], dtype=float).reshape(-1)
        lo, hi = np.quantile(pt, [0.2, 0.8])
        keep = (pt >= lo) & (pt <= hi)
    else:
        centroid = pos.mean(axis=0)
        dist = np.linalg.norm(pos - centroid, axis=1)
        thresh = np.quantile(dist, core_fraction) if len(dist) > 1 else dist.max()
        keep = dist <= thresh

    if not np.any(keep):
        keep = np.ones(len(global_idx), dtype=bool)
    return global_idx[keep]
