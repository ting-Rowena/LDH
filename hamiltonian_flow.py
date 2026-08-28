"""
Hamiltonian Biological Flow Model — unified (z, p) latent dynamics.

H(z, p, t) = U(z, t) + 0.5 * p^T M^{-1} p   (M = I by default)

Dynamics (damped Hamiltonian with diagonal noise on p):
    dz/dt = ∂H/∂p = p + r_theta(z,t)   (default residual mode)
    dp/dt = -∂H/∂z - γ p + noise

Training losses (beyond L_pred):
    L_energy   = |dU/dt + 0.5 ||∇U||^2|
    L_energy_total = |dU/dt + 0.5 ||-∇U + r_theta||^2|
    L_momentum_velocity = ||p_final - (z_pred - z_curr) / dt||^2
    L_momentum_force = ||(p_final - p_init) / dt + ∇U + γ p_init - r_force||^2

LAP action (legacy manifold optimizer — superseded by flow_space_lap.py):

    Flow-space action integral:
        S(γ) = ∫ [ ||dz/dt - f(z,t)||^2 + λ U(z) ] dt   with   dz/dt = f = ∂H/∂p = p

    Old Hamiltonian form (deprecated for path analysis):
        S(γ) = ∫ [ ||p||^2/2 + U(z) ] dt   with   dz/dt = p
"""

from __future__ import annotations

from typing import Callable, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy import optimize

Array = np.ndarray
FieldFunc = Callable[[Array], float]


def grad_potential(potential_net: nn.Module, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Return ∇_z U (positive gradient of potential)."""
    if t.dim() == 1:
        t = t.unsqueeze(1)
    z = z.requires_grad_(True)
    pot = potential_net(z, t)
    grad_z = torch.autograd.grad(
        pot, z, torch.ones_like(pot), create_graph=True, retain_graph=True
    )[0]
    return grad_z


def energy_regularizer(potential_net: nn.Module, z: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """
    L_energy = |dU/dt + 0.5 ||∇U||^2|

    Refactored Hamilton-Jacobi-inspired term on the learned potential only.
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)
    z = z.requires_grad_(True)
    t = t.requires_grad_(True)
    pot = potential_net(z, t)
    grad_z = torch.autograd.grad(
        pot, z, torch.ones_like(pot), create_graph=True, retain_graph=True
    )[0]
    grad_t = torch.autograd.grad(
        pot, t, torch.ones_like(pot), create_graph=True, retain_graph=True
    )[0]
    residual = grad_t + 0.5 * torch.sum(grad_z ** 2, dim=1, keepdim=True)
    return residual.abs()


def total_drift_energy_regularizer(
    potential_net: nn.Module,
    residual_drift_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]],
    z: torch.Tensor,
    t: torch.Tensor,
    *,
    residual_drift_mode: str = "velocity",
) -> torch.Tensor:
    """
    Total-drift-aware HJ term:
        |dU/dt + 0.5 || -∇U + r_theta ||^2|

    If residual drift is disabled or mode is "none", this reduces to the
    gradient-only energy regularizer.
    """
    if t.dim() == 1:
        t = t.unsqueeze(1)
    z = z.requires_grad_(True)
    t = t.requires_grad_(True)
    pot = potential_net(z, t)
    grad_z = torch.autograd.grad(
        pot, z, torch.ones_like(pot), create_graph=True, retain_graph=True
    )[0]
    grad_t = torch.autograd.grad(
        pot, t, torch.ones_like(pot), create_graph=True, retain_graph=True
    )[0]
    mode = (residual_drift_mode or "velocity").lower()
    if mode == "none" or residual_drift_fn is None:
        residual = torch.zeros_like(z)
    else:
        residual = residual_drift_fn(z, t)
    f_total = -grad_z + residual
    hj_residual = grad_t + 0.5 * torch.sum(f_total ** 2, dim=1, keepdim=True)
    return hj_residual.abs()


def momentum_regularizer(
    p: torch.Tensor,
    dz_dt: torch.Tensor,
    grad_u: torch.Tensor,
) -> torch.Tensor:
    """L_momentum = || p - (dz/dt + ∇U) ||^2 per batch, mean over cells."""
    target = dz_dt + grad_u
    return torch.mean((p - target) ** 2)


class HamiltonianFlowFunc(nn.Module):
    """
    Damped Hamiltonian flow in latent space.

    State: z (latent cell state), p (biological momentum / trajectory memory).
    Mass matrix M = I  =>  ∂H/∂p = p.
    """

    noise_type = "diagonal"
    sde_type = "ito"

    def __init__(self, potential_net, latent_dim, cfg, residual_net=None):
        super().__init__()
        self.potential_net = potential_net
        self.residual_net = residual_net
        self.latent_dim = latent_dim
        self.config = cfg
        self.gamma = float(getattr(cfg, "hamiltonian_damping_gamma", 0.1))
        if cfg.sigma_type == "Mlp":
            self.sigma = nn.Sequential(
                nn.Linear(latent_dim + 1, latent_dim),
                nn.Sigmoid(),
            )
        else:
            self.sigma = nn.Parameter(torch.full((1, latent_dim), 0.1))

    def grad_potential(self, z, t):
        return grad_potential(self.potential_net, z, t)

    def gradient_drift(self, z, t):
        """Legacy alias: -∇U (conservative drift)."""
        return -self.grad_potential(z, t)

    def residual_drift(self, z, t):
        mode = getattr(self.config, "residual_drift_mode", "velocity")
        if not getattr(self.config, "use_residual_drift", True):
            mode = "none"
        if self.residual_net is None or mode == "none":
            return torch.zeros_like(z)
        return self.residual_net(z, t)

    def total_drift(self, z, t):
        """Legacy total drift for diagnostics / graph fallback (-∇U + r)."""
        if t.dim() == 1:
            t = t.unsqueeze(1)
        return self.gradient_drift(z, t) + self.residual_drift(z, t)

    def hamiltonian_drift(
        self,
        z,
        p,
        t,
        detach_potential=False,
        z_curr=None,
        z_target=None,
    ):
        """Return (dz/dt, dp/dt) without noise.

        ``detach_potential`` stop-grads the potential force ``-∇U`` inside the
        (z, p) update. Forward values are unchanged, but no gradient flows into
        the potential network (U0 in particular) through this integration.
        """
        del z_curr, z_target
        if t.dim() == 1:
            t = t.unsqueeze(1)
        grad_u = self.grad_potential(z, t)
        if detach_potential:
            grad_u = grad_u.detach()
        mode = getattr(self.config, "residual_drift_mode", "velocity")
        if not getattr(self.config, "use_residual_drift", True):
            mode = "none"
        mode = (mode or "velocity").lower()
        if mode not in {"velocity", "force", "none"}:
            raise ValueError(f"Unknown residual_drift_mode={mode!r}")

        residual = self.residual_drift(z, t)
        if mode == "velocity":
            dz_dt = p + residual
            dp_dt = -grad_u - self.gamma * p
        elif mode == "force":
            dz_dt = p
            dp_dt = -grad_u + residual - self.gamma * p
        else:
            dz_dt = p
            dp_dt = -grad_u - self.gamma * p
        return dz_dt, dp_dt

    def f(self, t, z):
        """SDE-style drift on z only (legacy fallback): dz = p with p=0 => -∇U."""
        t_batch = torch.full((z.shape[0], 1), float(t), device=z.device, dtype=z.dtype)
        return self.gradient_drift(z, t_batch) + self.residual_drift(z, t_batch)

    def g(self, t, z):
        if self.config.sigma_type == "Mlp":
            t_batch = torch.full((z.shape[0], 1), float(t), device=z.device, dtype=z.dtype)
            raw = self.sigma(torch.cat([z, t_batch], dim=-1))
            return self.config.sigma_min + self.config.sigma_scale * raw
        raw = self.sigma.expand(z.shape[0], -1)
        return self.config.sigma_min + self.config.sigma_scale * raw

    def diffusion_on_p(self, t, z):
        return self.g(t, z)


def integrate_hamiltonian_flow(
    flow_func: HamiltonianFlowFunc,
    z0: torch.Tensor,
    p0: torch.Tensor,
    ts: torch.Tensor,
    *,
    dt: float = 0.1,
    add_noise: bool = True,
    detach_potential: bool = False,
    z_curr: Optional[torch.Tensor] = None,
    z_target: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Euler–Maruyama integration of coupled (z, p) Hamiltonian flow.

    Returns
    -------
    trajectory : (T, B, D)
    p_final : (B, D)
    """
    device, dtype = z0.device, z0.dtype
    ts = ts.to(device=device, dtype=dtype)
    if ts.numel() < 2:
        return z0.unsqueeze(0), p0

    z = z0
    p = p0
    traj = [z0]
    for i in range(ts.numel() - 1):
        t_val = float(ts[i].item())
        dt_step = float((ts[i + 1] - ts[i]).item())
        if dt_step <= 0:
            continue
        n_steps = max(1, int(np.ceil(abs(dt_step) / max(dt, 1e-6))))
        sub_dt = dt_step / n_steps
        t_batch = torch.full((z.shape[0], 1), t_val, device=device, dtype=dtype)
        for _ in range(n_steps):
            dz_dt, dp_dt = flow_func.hamiltonian_drift(
                z,
                p,
                t_batch,
                detach_potential=detach_potential,
                z_curr=z_curr,
                z_target=z_target,
            )
            z = z + dz_dt * sub_dt
            p = p + dp_dt * sub_dt
            z_lim = float(getattr(flow_func.config, "latent_integrate_clamp", 25.0) or 25.0)
            p_lim = float(getattr(flow_func.config, "momentum_integrate_clamp", 25.0) or 25.0)
            z = torch.nan_to_num(z, nan=0.0, posinf=z_lim, neginf=-z_lim).clamp(-z_lim, z_lim)
            p = torch.nan_to_num(p, nan=0.0, posinf=p_lim, neginf=-p_lim).clamp(-p_lim, p_lim)
            if add_noise and flow_func.training:
                sigma = flow_func.diffusion_on_p(t_val, z)
                noise = torch.randn_like(p) * sigma * (abs(sub_dt) ** 0.5)
                p = p + noise
        traj.append(z)

    return torch.stack(traj, dim=0), p


def hamiltonian_action_score(path: Array, U_func: FieldFunc) -> float:
    """
    S = ∫ [ ||p||^2/2 + U(z) ] dt  with  p = dz/dt  (M=I, dz/dt = ∂H/∂p).
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 2:
        return 0.0
    action = 0.0
    for i in range(len(path) - 1):
        dz = path[i + 1] - path[i]
        dt_seg = float(np.linalg.norm(dz))
        if dt_seg < 1e-12:
            continue
        p = dz / dt_seg
        u = float(np.asarray(U_func(path[i])).reshape(-1)[0])
        action += (0.5 * np.sum(p ** 2) + u) * dt_seg
    return float(action)


def hamiltonian_action_profile(path: Array, U_func: FieldFunc) -> Array:
    """Cumulative Hamiltonian action along a discretized path."""
    path = np.asarray(path, dtype=float)
    profile = np.zeros(len(path))
    cumulative = 0.0
    for i in range(1, len(path)):
        dz = path[i] - path[i - 1]
        dt_seg = float(np.linalg.norm(dz))
        if dt_seg < 1e-12:
            profile[i] = cumulative
            continue
        p = dz / dt_seg
        u = float(np.asarray(U_func(path[i - 1])).reshape(-1)[0])
        cumulative += (0.5 * np.sum(p ** 2) + u) * dt_seg
        profile[i] = cumulative
    return profile


def hamiltonian_constraint_violation(path: Array) -> float:
    """
    Mean squared deviation from dz/dt = p when p is inferred from consecutive segments.
    Used as a stability diagnostic (not added to training loss).
    """
    path = np.asarray(path, dtype=float)
    if len(path) < 3:
        return 0.0
    violations = []
    for i in range(len(path) - 2):
        dz1 = path[i + 1] - path[i]
        dz2 = path[i + 2] - path[i + 1]
        dt1 = max(float(np.linalg.norm(dz1)), 1e-12)
        dt2 = max(float(np.linalg.norm(dz2)), 1e-12)
        p1 = dz1 / dt1
        p2 = dz2 / dt2
        violations.append(float(np.mean((p2 - p1) ** 2)))
    return float(np.mean(violations)) if violations else 0.0


def _pin_path_endpoints(path: Array, start: Array, end: Array) -> Array:
    path = np.asarray(path, dtype=float).copy()
    path[0] = start
    path[-1] = end
    return path


def optimize_hamiltonian_action_path(
    start: Array,
    end: Array,
    U_func: FieldFunc,
    positions: Array,
    n_points: int = 50,
    project_to_manifold: bool = True,
    max_iter: int = 500,
    constraint_tol: float = 1.0,
    *,
    project_fn=None,
    reparameterize_fn=None,
) -> Tuple[Array, float, bool, dict]:
    """
    Minimize Hamiltonian action S with dz/dt = p enforced by path tangent parameterization.

    Returns path, total_action, success, metadata (includes hamiltonian_path_unstable).
    """
    from landscape_core import (
        _project_path_to_manifold,
        _reparameterize_by_arclength,
    )

    if project_fn is None:
        project_fn = _project_path_to_manifold
    if reparameterize_fn is None:
        reparameterize_fn = _reparameterize_by_arclength

    start = np.asarray(start, dtype=float)
    end = np.asarray(end, dtype=float)
    dim = start.shape[0]
    if dim > 3:
        max_iter = min(max_iter, 120)
        n_points = min(n_points, 40)

    init = np.linspace(start, end, n_points)
    bounds_lo = positions.min(axis=0)
    bounds_hi = positions.max(axis=0)

    def _prepare(path):
        path = _pin_path_endpoints(path, start, end)
        if project_to_manifold and n_points > 2:
            path[1:-1] = project_fn(path[1:-1], positions)
        path, _ = reparameterize_fn(path)
        return _pin_path_endpoints(path, start, end)

    def objective(flat):
        path = flat.reshape(n_points, dim)
        path = _prepare(path)
        return hamiltonian_action_score(path, U_func)

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

    result = optimize.minimize(
        objective, x0, method="L-BFGS-B", bounds=bnds, options={"maxiter": max_iter}
    )
    path = result.x.reshape(n_points, dim)
    path = _prepare(path)
    action = hamiltonian_action_score(path, U_func)
    violation = hamiltonian_constraint_violation(path)
    unstable = (
        not bool(result.success)
        or not np.isfinite(action)
        or violation > constraint_tol
    )
    meta = {
        "hamiltonian_path_unstable": bool(unstable),
        "hamiltonian_constraint_violation": float(violation),
        "optimizer_success": bool(result.success),
        "action_method": "hamiltonian",
    }
    success = bool(result.success) and np.isfinite(action)
    return path, action, success, meta
