"""Model-based drift, diffusion, and potential evaluation for FW-like action."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import torch


class ModelDynamicsEvaluator:
    """Evaluate U_theta, grad U, residual drift, total drift, and diagonal diffusion."""

    def __init__(self, model, device="cuda", eps=1e-6):
        self.model = model.to(device if torch.cuda.is_available() else "cpu")
        self.device = next(self.model.parameters()).device
        self.eps = eps
        self.model.eval()

    def _to_tensor(self, x):
        if isinstance(x, torch.Tensor):
            return x.to(self.device).float()
        return torch.tensor(x, dtype=torch.float32, device=self.device)

    def _format_time(self, t, n):
        t = self._to_tensor(t)
        if t.ndim == 0:
            t = t.repeat(n)
        if t.ndim == 1:
            t = t[:, None]
        return t

    def potential(self, z, t):
        z = self._to_tensor(z)
        t = self._format_time(t, z.shape[0])
        if hasattr(self.model, "potential_net"):
            return self.model.potential_net(z, t)
        if hasattr(self.model, "potential"):
            return self.model.potential(z, t)
        raise AttributeError("Model does not expose potential_net or potential.")

    def grad_potential(self, z, t):
        z = self._to_tensor(z).detach().clone().requires_grad_(True)
        t = self._format_time(t, z.shape[0])
        u = self.potential(z, t)
        u_scalar = u.sum() if u.ndim > 1 else u.sum()
        grad = torch.autograd.grad(u_scalar, z, create_graph=False, retain_graph=False)[0]
        return grad.detach().cpu().numpy()

    def residual_drift(self, z, t):
        z_t = self._to_tensor(z)
        t_t = self._format_time(t, z_t.shape[0])
        if hasattr(self.model, "residual_drift"):
            out = self.model.residual_drift(z_t, t_t)
        elif hasattr(self.model, "sde_func") and hasattr(self.model.sde_func, "residual_drift"):
            out = self.model.sde_func.residual_drift(z_t, t_t)
        else:
            out = torch.zeros_like(z_t)
        return out.detach().cpu().numpy()

    def total_drift(self, z, t):
        """Hamiltonian momentum drift p ≈ dz/dt; legacy fallback uses -∇U + r."""
        if hasattr(self.model, "flow_func") and getattr(
            getattr(self.model, "config", None), "use_hamiltonian_flow", True
        ):
            z_t = self._to_tensor(z)
            t_t = self._format_time(t, z_t.shape[0])
            if hasattr(self.model.flow_func, "hamiltonian_drift"):
                p = torch.zeros_like(z_t)
                dz, _ = self.model.flow_func.hamiltonian_drift(z_t, p, t_t)
                return dz.detach().cpu().numpy()
        grad_u = self.grad_potential(z, t)
        r = self.residual_drift(z, t)
        return -grad_u + r

    def hamiltonian_momentum(self, z, t):
        """Biological momentum p at (z, t).

        Prefers the state-dependent momentum network p_theta(z, t); falls back to the
        legacy global EMA momentum vector when the network is unavailable.
        """
        if getattr(self.model, "momentum_net", None) is not None or hasattr(
            self.model, "initial_momentum"
        ):
            z_t = self._to_tensor(z)
            single = z_t.ndim == 1
            if single:
                z_t = z_t[None, :]
            t_t = self._format_time(t, z_t.shape[0])
            p = self.model.initial_momentum(z_t, t_t)
            p_np = p.detach().cpu().numpy()
            return p_np.reshape(-1) if single else p_np
        if hasattr(self.model, "_momentum_ema"):
            ema = self.model._momentum_ema.detach().cpu().numpy()
            z_arr = np.asarray(z, dtype=float)
            if z_arr.ndim == 1:
                return ema.reshape(-1)
            return np.repeat(ema, z_arr.shape[0], axis=0)
        return np.zeros_like(np.asarray(z, dtype=float))

    def diffusion(self, z, t):
        """Diagonal diffusion variance sigma^2 (alias for diffusion_diag)."""
        return self.diffusion_diag(z, t)

    def diffusion_diag(self, z, t):
        z_t = self._to_tensor(z)
        t_t = self._format_time(t, z_t.shape[0])
        if hasattr(self.model, "diffusion_diag"):
            sigma = self.model.diffusion_diag(z_t, t_t)
        elif hasattr(self.model, "sde_func") and hasattr(self.model.sde_func, "g"):
            sigma_rows = []
            for i in range(z_t.shape[0]):
                sigma_rows.append(
                    self.model.sde_func.g(float(t_t[i].item()), z_t[i : i + 1])
                )
            sigma = torch.cat(sigma_rows, dim=0)
        else:
            return np.ones(z_t.shape, dtype=float)

        sigma_np = sigma.detach().cpu().numpy()
        if sigma_np.ndim == 1:
            sigma_np = np.repeat(sigma_np[:, None], z_t.shape[1], axis=1)
        elif sigma_np.ndim == 2:
            pass
        else:
            sigma_np = np.reshape(sigma_np, (z_t.shape[0], -1))
            if sigma_np.shape[1] != z_t.shape[1]:
                sigma_np = np.repeat(
                    np.mean(sigma_np, axis=1, keepdims=True), z_t.shape[1], axis=1
                )
        return np.maximum(sigma_np ** 2, self.eps)


def load_model_dynamics_from_checkpoint(
    checkpoint_dir: str,
    adata_full=None,
    device: str = "cuda",
) -> Optional[ModelDynamicsEvaluator]:
    """
    Load TemporalSDENetwork from checkpoint if adata_full is provided for architecture.
    Returns None when checkpoint or adata is unavailable.
    """
    ckpt = Path(checkpoint_dir)
    model_path = ckpt / "best_model.pth"
    if not model_path.is_file() or adata_full is None:
        return None
    try:
        from train_model import TemporalSDENetwork, Config
        from dataset_pipeline import apply_train_config

        # Minimal config — architecture comes from adata + saved weights
        config = Config()
        if hasattr(adata_full, "uns") and "dataset_spec" in adata_full.uns:
            pass
        model = TemporalSDENetwork(config, adata_full)
        state = torch.load(
            model_path, map_location=device if torch.cuda.is_available() else "cpu"
        )
        try:
            model.load_state_dict(state)
        except RuntimeError:
            model.load_state_dict(state, strict=False)
        return ModelDynamicsEvaluator(model, device=device)
    except Exception:
        return None
