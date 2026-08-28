"""
Flow-space Hamiltonian Action Integral — continuous dynamical path analysis.

Replaces graph-based / discrete-edge LAP with ODE rollout trajectories.

H(z, p, t) = U(z, t) + 0.5 * p^T M^{-1} p   (M = I  =>  f = ∂H/∂p = p)

Continuous flow field:
    f(z, t) = ∂H/∂p = p   (momentum state along rollout)

ODE:
    dz/dt = f(z, t) = p
    dp/dt = -∇U(z) - γ p

Flow-space action:
    S(γ) = ∫_0^T [ ||dz/dt - f(z, t)||^2 + λ U(z) ] dt

path_degenerate = True when:
    - numerical integration diverges
    - mean flow mismatch exceeds threshold
    - ODE ensemble trajectory variance exceeds threshold
"""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
from scipy import optimize
from scipy.integrate import solve_ivp

Array = np.ndarray
FieldFunc = Callable[[Array], float]

DEFAULT_FLOW_MISMATCH_THRESHOLD = 0.5
DEFAULT_ENSEMBLE_VARIANCE_THRESHOLD = 0.25
DEFAULT_DIVERGENCE_NORM = 1e3
DEFAULT_LAMBDA_U = 1.0


def numerical_grad_U(U_func: FieldFunc, z: Array, eps: float = 1e-5) -> Array:
    z = np.asarray(z, dtype=float).reshape(-1)
    g = np.zeros_like(z)
    u0 = float(np.asarray(U_func(z)).reshape(-1)[0])
    for i in range(len(z)):
        zp = z.copy()
        zp[i] += eps
        g[i] = (float(np.asarray(U_func(zp)).reshape(-1)[0]) - u0) / eps
    return g


def flow_field_hamiltonian(p: Array) -> Array:
    """f(z, t) = ∂H/∂p = M^{-1} p with M = I."""
    return np.asarray(p, dtype=float)


def _coupled_rhs(t: float, y: Array, U_func: FieldFunc, gamma: float, dim: int) -> Array:
    z = y[:dim]
    p = y[dim:]
    grad_u = numerical_grad_U(U_func, z)
    dz = flow_field_hamiltonian(p)
    dp = -grad_u - gamma * p
    return np.concatenate([dz, dp])


def rollout_ode_trajectory(
    z_start: Array,
    z_end: Array,
    U_func: FieldFunc,
    *,
    n_points: int = 50,
    t_span: Tuple[float, float] = (0.0, 1.0),
    gamma: float = 0.1,
    p0: Optional[Array] = None,
    shoot_endpoints: bool = True,
) -> Tuple[Optional[Array], Optional[Array], Array, Dict[str, Any]]:
    """
    Integrate dz/dt = p, dp/dt = -∇U - γp from z_start toward z_end.

    Returns path (n_points, dim), momentum (n_points, dim), times, metadata.
    """
    z_start = np.asarray(z_start, dtype=float).reshape(-1)
    z_end = np.asarray(z_end, dtype=float).reshape(-1)
    dim = z_start.shape[0]
    t0, t1 = t_span
    times = np.linspace(t0, t1, n_points)
    meta: Dict[str, Any] = {
        "action_method": "flow_space_ode",
        "gamma": float(gamma),
        "diverged": False,
    }

    if p0 is None:
        p0 = (z_end - z_start) / max(t1 - t0, 1e-6)

    def _integrate(p_init: Array):
        y0 = np.concatenate([z_start, p_init])
        try:
            sol = solve_ivp(
                lambda t, y: _coupled_rhs(t, y, U_func, gamma, dim),
                (t0, t1),
                y0,
                t_eval=times,
                method="RK45",
                rtol=1e-5,
                atol=1e-7,
            )
        except Exception as exc:
            return None, None, str(exc)
        if not sol.success or sol.y is None:
            return None, None, sol.message if sol.message else "integration failed"
        if not np.all(np.isfinite(sol.y)):
            return None, None, "non-finite trajectory"
        if np.max(np.linalg.norm(sol.y[:dim], axis=0)) > DEFAULT_DIVERGENCE_NORM:
            return None, None, "trajectory norm divergence"
        path = np.asarray(sol.y[:dim].T, dtype=float)
        mom = np.asarray(sol.y[dim:].T, dtype=float)
        return path, mom, None

    if shoot_endpoints:

        def objective(p_flat):
            path, _, err = _integrate(p_flat)
            if path is None:
                return 1e12
            return float(np.sum((path[-1] - z_end) ** 2))

        res = optimize.minimize(objective, p0, method="L-BFGS-B")
        p0 = res.x
        meta["shooting_success"] = bool(res.success)

    path, momentum, err = _integrate(p0)
    if path is None:
        meta["diverged"] = True
        meta["integration_error"] = err
        return None, None, times, meta

    meta["endpoint_error"] = float(np.linalg.norm(path[-1] - z_end))
    path[0] = z_start
    path[-1] = z_end
    meta["p0"] = p0.copy()
    return path, momentum, times, meta


def flow_space_action_integral(
    path: Array,
    times: Array,
    U_func: FieldFunc,
    momentum: Array,
    *,
    lambda_u: float = DEFAULT_LAMBDA_U,
) -> Tuple[float, Array]:
    """
    S(γ) = ∫ [ ||dz/dt - f(z,t)||^2 + λ U(z) ] dt   with f = p.
    """
    path = np.asarray(path, dtype=float)
    times = np.asarray(times, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    profile = np.zeros(len(path))
    total = 0.0
    for i in range(len(path) - 1):
        dt = float(times[i + 1] - times[i])
        if dt <= 0:
            continue
        dz_dt = (path[i + 1] - path[i]) / dt
        f = flow_field_hamiltonian(momentum[i])
        mismatch = float(np.sum((dz_dt - f) ** 2))
        u = float(np.asarray(U_func(path[i])).reshape(-1)[0])
        seg = (mismatch + lambda_u * u) * dt
        total += seg
        profile[i + 1] = total
    return float(total), profile


def mean_flow_mismatch(
    path: Array,
    times: Array,
    momentum: Array,
) -> float:
    path = np.asarray(path, dtype=float)
    times = np.asarray(times, dtype=float)
    momentum = np.asarray(momentum, dtype=float)
    vals = []
    for i in range(len(path) - 1):
        dt = float(times[i + 1] - times[i])
        if dt <= 0:
            continue
        dz_dt = (path[i + 1] - path[i]) / dt
        f = flow_field_hamiltonian(momentum[i])
        vals.append(float(np.mean((dz_dt - f) ** 2)))
    return float(np.mean(vals)) if vals else 0.0


def diagnose_flow_path_degeneracy(
    path: Optional[Array],
    momentum: Optional[Array],
    times: Array,
    meta: Dict[str, Any],
    *,
    ensemble_paths: Optional[Sequence[Array]] = None,
    flow_mismatch_threshold: float = DEFAULT_FLOW_MISMATCH_THRESHOLD,
    ensemble_variance_threshold: float = DEFAULT_ENSEMBLE_VARIANCE_THRESHOLD,
) -> Dict[str, Any]:
    """Continuous-dynamics path_degenerate (replaces graph/kNN geometry tests)."""
    warnings: List[str] = []
    diverged = bool(meta.get("diverged", False))
    integration_error = meta.get("integration_error")
    if integration_error:
        warnings.append(str(integration_error))

    flow_mismatch = float("nan")
    ensemble_variance = float("nan")
    ode_unstable = False

    if path is not None and momentum is not None and len(path) >= 2:
        flow_mismatch = mean_flow_mismatch(path, times, momentum)
        ode_unstable = flow_mismatch > flow_mismatch_threshold

    if ensemble_paths and len(ensemble_paths) >= 2:
        lengths = [len(p) for p in ensemble_paths]
        n = min(lengths)
        if n >= 2:
            stack = np.stack([np.asarray(p[:n], dtype=float) for p in ensemble_paths], axis=0)
            ensemble_variance = float(np.mean(np.var(stack, axis=0)))
            if ensemble_variance > ensemble_variance_threshold:
                ode_unstable = True

    is_degenerate = bool(diverged or ode_unstable or not np.isfinite(flow_mismatch))

    if diverged:
        rec = "ode_integration_diverged"
    elif ode_unstable:
        rec = "high_flow_mismatch_or_ensemble_variance"
    else:
        rec = "flow_trajectory_candidate"

    return {
        "is_degenerate": is_degenerate,
        "path_degenerate": is_degenerate,
        "diverged": diverged,
        "flow_mismatch": flow_mismatch,
        "ensemble_variance": ensemble_variance,
        "ode_unstable": ode_unstable,
        "flow_mismatch_threshold": flow_mismatch_threshold,
        "ensemble_variance_threshold": ensemble_variance_threshold,
        "endpoint_error": meta.get("endpoint_error"),
        "integration_error": integration_error,
        "degeneracy_reason": rec,
        "recommended_interpretation": rec,
        "warnings": warnings,
        "action_method": "flow_space_hamiltonian",
    }


def ode_ensemble_rollout(
    z_start: Array,
    z_end: Array,
    U_func: FieldFunc,
    *,
    n_ensemble: int = 5,
    perturb_scale: float = 0.02,
    random_state: int = 42,
    **rollout_kw,
) -> Tuple[Optional[Array], Optional[Array], Array, Dict[str, Any], List[Array]]:
    """Multi initial-condition ODE ensemble; return best-action trajectory."""
    rng = np.random.default_rng(random_state)
    z_start = np.asarray(z_start, dtype=float)
    z_end = np.asarray(z_end, dtype=float)
    dim = z_start.shape[0]
    spacing = max(float(np.linalg.norm(z_end - z_start)), 1e-6)

    paths: List[Array] = []
    best = None
    best_S = float("inf")
    best_mom = None
    best_times = np.linspace(0.0, 1.0, rollout_kw.get("n_points", 50))
    best_meta: Dict[str, Any] = {}

    for k in range(n_ensemble):
        if k == 0:
            zs = z_start.copy()
            p0 = None
        else:
            zs = z_start + rng.normal(0, perturb_scale * spacing, size=dim)
            p0 = (z_end - zs) / max(
                rollout_kw.get("t_span", (0.0, 1.0))[1]
                - rollout_kw.get("t_span", (0.0, 1.0))[0],
                1e-6,
            ) + rng.normal(0, perturb_scale * spacing, size=dim)

        path, mom, times, meta = rollout_ode_trajectory(
            zs, z_end, U_func, p0=p0, **rollout_kw
        )
        if path is None:
            continue
        paths.append(path)
        S, _ = flow_space_action_integral(path, times, U_func, mom)
        if S < best_S:
            best_S = S
            best = path
            best_mom = mom
            best_times = times
            best_meta = dict(meta)
            best_meta["ensemble_index"] = k

    meta_out = dict(best_meta)
    meta_out["n_ensemble"] = n_ensemble
    meta_out["n_success"] = len(paths)
    meta_out["ensemble_paths"] = paths
    meta_out["total_action"] = best_S if np.isfinite(best_S) else float("nan")
    return best, best_mom, best_times, meta_out, paths


def compute_flow_space_lap_path(
    start: Array,
    end: Array,
    U_func: FieldFunc,
    *,
    n_points: int = 50,
    gamma: float = 0.1,
    lambda_u: float = DEFAULT_LAMBDA_U,
    use_ensemble: bool = False,
    n_ensemble: int = 5,
    random_state: int = 42,
) -> Dict[str, Any]:
    """
    Primary flow-space LAP: ODE rollout + action integral + degeneracy diagnostics.
    """
    if use_ensemble:
        path, momentum, times, meta, ensemble = ode_ensemble_rollout(
            start,
            end,
            U_func,
            n_points=n_points,
            gamma=gamma,
            n_ensemble=n_ensemble,
            random_state=random_state,
        )
        path_method_used = "flow_space_ode_ensemble"
    else:
        path, momentum, times, meta = rollout_ode_trajectory(
            start, end, U_func, n_points=n_points, gamma=gamma
        )
        ensemble = [path] if path is not None else []
        path_method_used = "flow_space_ode"

    if path is None:
        deg = diagnose_flow_path_degeneracy(None, None, times, meta)
        return {
            "path": np.linspace(start, end, n_points),
            "success": False,
            "total_action": float("nan"),
            "action": np.zeros(n_points),
            "flow_degeneracy": deg,
            "path_degenerate": True,
            "is_degenerate": True,
            **meta,
        }

    total_action, action_profile = flow_space_action_integral(
        path, times, U_func, momentum, lambda_u=lambda_u
    )
    deg = diagnose_flow_path_degeneracy(
        path, momentum, times, meta, ensemble_paths=ensemble
    )

    return {
        "path": path,
        "momentum": momentum,
        "times": times,
        "potential": np.array([float(np.asarray(U_func(p)).reshape(-1)[0]) for p in path]),
        "total_action": total_action,
        "action": action_profile,
        "flow_mismatch": deg.get("flow_mismatch"),
        "flow_degeneracy": deg,
        "path_degenerate": bool(deg.get("path_degenerate", False)),
        "is_degenerate": bool(deg.get("is_degenerate", False)),
        "ensemble_paths": ensemble,
        "success": not bool(meta.get("diverged", False)),
        "action_method": "flow_space_hamiltonian",
        "lap_method": "flow_space_ode",
        "path_method_used": path_method_used,
        "lambda_u": float(lambda_u),
        **meta,
    }
