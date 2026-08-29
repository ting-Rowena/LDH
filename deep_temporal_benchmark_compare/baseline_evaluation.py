#!/usr/bin/env python
"""Unified baseline comparison harness for temporal trajectory prediction.

Evaluates the trained latent-SDE model against reference baselines on the *same*
held-out validation split and the *same* distribution-level metrics, so numbers are
directly comparable for a paper's main table.

Protocol (designed for ``val_mode in {time, time_extrapolate}``):
  for each held-out transition (t_curr -> t_next):
    - source population  = cells at t_curr on the training mask
    - target population  = observed cells at t_next on the validation mask
    - each method maps the source population forward to a predicted t_next population
    - we score predicted vs. observed populations with OT / MMD / energy distance /
      mean per-gene 1-Wasserstein / mean-shift L2.

Self-contained baselines (no external dependencies):
  - ``persistence``          : x_next = x_curr (no dynamics)
  - ``global_linear_drift``  : x_next = x_curr + rate * dt, rate from prior train step
  - ``wot_barycentric``      : Waddington-OT-inspired entropic-OT barycentric velocity
                               estimated on the last train step, extrapolated forward

External-family baselines (trained on the *same* observed marginals, then rolled
forward on the held-out transitions):
  - ``prescient_potential_flow`` : PRESCIENT-family first-order potential-driven flow,
                                   dx = -grad Psi(x) dt (+ noise), Psi an MLP trained by
                                   entropic OT between consecutive population marginals.
  - ``mioflow_neural_ode``       : MIOFlow-family neural-ODE velocity field v(x, t),
                                   Euler-integrated, trained by entropic OT between
                                   consecutive population marginals.

These are self-contained in-harness reimplementations of each method family's core
transport objective, not executions of the official PRESCIENT, MIOFlow, or WOT
pipelines. The report records that distinction explicitly.

The main table is ranked by a *non-training-objective* metric (default: energy distance)
so that no method is scored on the exact loss it was optimised for. Sinkhorn OT is still
reported but flagged as a training-objective reference column.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
import torch
from torch import autograd, nn

# Repo root (parent of this package directory) for shared training / data modules.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from train_model import (
    Config,
    TemporalDataProcessor,
    TemporalSDENetwork,
    ValidationSplit,
    _load_state_dict_compat,
    gaussian_mmd_loss,
)

# ------------------------------------------------------------------ metrics


def _subsample(arr: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    if arr.shape[0] <= n:
        return arr
    idx = rng.choice(arr.shape[0], size=n, replace=False)
    return arr[idx]


def _energy_distance(x: np.ndarray, y: np.ndarray) -> float:
    """Székely energy distance between two samples (Euclidean)."""
    from scipy.spatial.distance import cdist

    d_xy = cdist(x, y).mean()
    d_xx = cdist(x, x).mean()
    d_yy = cdist(y, y).mean()
    return float(2.0 * d_xy - d_xx - d_yy)


def _mean_marginal_w1(x: np.ndarray, y: np.ndarray, max_genes: int = 200) -> float:
    """Mean per-gene 1-Wasserstein distance (subset of genes for speed)."""
    from scipy.stats import wasserstein_distance

    g = x.shape[1]
    cols = range(g) if g <= max_genes else np.linspace(0, g - 1, max_genes).astype(int)
    vals = [wasserstein_distance(x[:, j], y[:, j]) for j in cols]
    return float(np.mean(vals)) if vals else float("nan")


def population_distribution_metrics(
    pred: np.ndarray,
    target: np.ndarray,
    *,
    ot_loss=None,
    max_cells: int = 1000,
    seed: int = 0,
) -> Dict[str, float]:
    """Distribution-level distances between a predicted and observed population."""
    rng = np.random.RandomState(seed)
    p = _subsample(np.asarray(pred, dtype=np.float32), max_cells, rng)
    t = _subsample(np.asarray(target, dtype=np.float32), max_cells, rng)
    metrics: Dict[str, float] = {}

    if ot_loss is None:
        from geomloss import SamplesLoss

        ot_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True)
    with torch.no_grad():
        metrics["ot_sinkhorn"] = float(
            ot_loss(torch.from_numpy(p), torch.from_numpy(t)).item()
        )
        metrics["mmd"] = float(
            gaussian_mmd_loss(torch.from_numpy(p), torch.from_numpy(t)).item()
        )
    metrics["energy_distance"] = _energy_distance(p, t)
    metrics["mean_marginal_w1"] = _mean_marginal_w1(p, t)
    metrics["mean_shift_l2"] = float(np.linalg.norm(p.mean(0) - t.mean(0)))
    return metrics


# ------------------------------------------------------------------ baselines


def baseline_persistence(x_curr, x_prev_step, dt, dt_prev):
    return np.asarray(x_curr, dtype=np.float32)


def baseline_global_linear_drift(x_curr, x_prev_step, dt, dt_prev):
    """Extrapolate the mean drift rate observed over the previous train step."""
    if x_prev_step is None or dt_prev is None or dt_prev <= 0:
        return np.asarray(x_curr, dtype=np.float32)
    rate = (x_curr.mean(0) - x_prev_step.mean(0)) / dt_prev
    return np.asarray(x_curr + rate[None, :] * dt, dtype=np.float32)


def _entropic_sinkhorn_plan(a_pts, b_pts, blur=0.5, n_iter=200):
    """Compact entropic-OT plan (row-normalized) between two point clouds."""
    from scipy.spatial.distance import cdist

    cost = cdist(a_pts, b_pts, metric="sqeuclidean")
    eps = blur * (cost.mean() + 1e-9)
    K = np.exp(-cost / eps)
    n, m = cost.shape
    u = np.ones(n) / n
    v = np.ones(m) / m
    r = np.ones(n) / n
    c = np.ones(m) / m
    for _ in range(n_iter):
        u = r / (K @ v + 1e-12)
        v = c / (K.T @ u + 1e-12)
    plan = u[:, None] * K * v[None, :]
    row_sums = plan.sum(1, keepdims=True) + 1e-12
    return plan / row_sums


def baseline_wot_barycentric(x_curr, x_prev_step, dt, dt_prev, *, max_pts=600, seed=0):
    """Waddington-OT-inspired local barycentric displacement, extrapolated forward.

    Estimate an entropic-OT barycentric map from the previous train step
    (x_prev_step -> x_curr), interpolate its local displacement field at every
    current cell, and scale by dt / dt_prev to predict the next population.

    This reproduces the core unbalanced-population transport idea used by WOT but
    does not include WOT's growth-rate model; it is therefore labelled
    ``WOT-inspired`` in generated reports.
    """
    if x_prev_step is None or dt_prev is None or dt_prev <= 0:
        return np.asarray(x_curr, dtype=np.float32)
    rng = np.random.RandomState(seed)
    src = _subsample(np.asarray(x_prev_step, dtype=np.float64), max_pts, rng)
    dst = _subsample(np.asarray(x_curr, dtype=np.float64), max_pts, rng)
    plan = _entropic_sinkhorn_plan(src, dst)
    src_image = plan @ dst  # barycentric image of each src point
    displacement = src_image - src

    # Interpolate the transport displacement locally instead of applying one global
    # mean shift. Inverse-distance kNN is deterministic and avoids an extra model fit.
    from scipy.spatial.distance import cdist

    query = np.asarray(x_curr, dtype=np.float64)
    distances = cdist(query, src, metric="euclidean")
    k = min(15, src.shape[0])
    nn = np.argpartition(distances, kth=k - 1, axis=1)[:, :k]
    nn_dist = np.take_along_axis(distances, nn, axis=1)
    weights = 1.0 / np.maximum(nn_dist, 1e-8)
    weights /= weights.sum(axis=1, keepdims=True)
    local_disp = (displacement[nn] * weights[..., None]).sum(axis=1)
    return np.asarray(query + (dt / dt_prev) * local_disp, dtype=np.float32)


SELF_CONTAINED_BASELINES: Dict[str, Callable] = {
    "persistence": baseline_persistence,
    "global_linear_drift": baseline_global_linear_drift,
    "wot_barycentric": baseline_wot_barycentric,
}


# ---------------------------------------------- external-family trainable baselines
#
# In-harness reimplementations of the *core transport objective* of PRESCIENT and
# MIOFlow. Both learn a global (time-homogeneous or time-conditioned) field from all
# consecutive training marginals by entropic-OT matching, then roll a source population
# forward on the held-out transition. This mirrors how the original methods are fit
# (observed marginals -> forward prediction) while keeping the evaluation fully
# reproducible and dependency-safe.


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 128, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers += [nn.Linear(d, out_dim)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _FieldFlow(nn.Module):
    """First-order flow trained by OT marginal matching.

    ``kind='potential'`` -> velocity = -grad Psi(x)  (PRESCIENT family)
    ``kind='velocity'``  -> velocity = v(x, t)        (MIOFlow / neural-ODE family)
    """

    def __init__(self, dim: int, kind: str, hidden: int = 128):
        super().__init__()
        self.kind = kind
        if kind == "potential":
            self.net = _MLP(dim, 1, hidden=hidden)
        elif kind == "velocity":
            self.net = _MLP(dim + 1, dim, hidden=hidden)
        else:
            raise ValueError(f"unknown flow kind {kind!r}")

    def velocity(self, x, t_norm: float):
        if self.kind == "velocity":
            tt = torch.full((x.shape[0], 1), float(t_norm), device=x.device, dtype=x.dtype)
            return self.net(torch.cat([x, tt], dim=1))
        # potential: v = -grad_x Psi(x)
        xg = x if x.requires_grad else x.clone().requires_grad_(True)
        psi = self.net(xg).sum()
        (grad,) = autograd.grad(psi, xg, create_graph=self.training)
        return -grad

    def roll(self, x, dt_norm: float, t0_norm: float, n_sub: int = 5):
        h = dt_norm / n_sub
        cur = x
        for i in range(n_sub):
            cur = cur + h * self.velocity(cur, t0_norm + i * h)
        return cur


_FLOW_CACHE: Dict[tuple, tuple] = {}


def _get_X(adata):
    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    return X.astype(np.float32)


def _train_transitions(Z, t_all, train_mask, time_scale, tmin, max_pts, device, rng):
    """List of (src_tensor, tgt_tensor, dt_norm, t0_norm) over consecutive train steps."""
    times = sorted({float(t) for t in t_all[train_mask]})
    out = []
    for a, b in zip(times[:-1], times[1:]):
        src = Z[train_mask & np.isclose(t_all, a)]
        tgt = Z[train_mask & np.isclose(t_all, b)]
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            continue
        src = _subsample(src, max_pts, rng)
        tgt = _subsample(tgt, max_pts, rng)
        out.append(
            (
                torch.tensor(src, dtype=torch.float32, device=device),
                torch.tensor(tgt, dtype=torch.float32, device=device),
                (b - a) / time_scale,
                (a - tmin) / time_scale,
            )
        )
    return out


def _fit_field_flow(adata, config, split, kind, *, n_iter=120, hidden=128,
                    max_pts=160, n_sub=2, lr=2e-3, n_pca=50, seed=0):
    """Fit a PRESCIENT-/MIOFlow-family flow in a PCA-reduced space.

    Both reference methods learn dynamics in a reduced representation (PRESCIENT on
    PCA, MIOFlow on an autoencoder latent). We mirror that with a PCA fit on the
    *training* cells, train the flow in PCA space, and inverse-transform predictions
    back to gene space so scoring happens in the same space as every other method.
    """
    key = (id(adata), id(split), kind, int(seed))
    if key in _FLOW_CACHE:
        return _FLOW_CACHE[key]
    from geomloss import SamplesLoss
    from sklearn.decomposition import PCA

    # Flows are tiny (PCA space, small MLP, few hundred points). Train on CPU so the
    # double-backprop potential flow does not contend with (or get starved by) other
    # jobs on a shared GPU; the model's own prediction still uses config.device.
    device = "cpu"
    X = _get_X(adata)
    t_all = adata.obs[config.time_key].astype(float).values
    train_mask = split.train_mask
    train_times = sorted({float(t) for t in t_all[train_mask]})
    tmin = train_times[0]
    time_scale = max(train_times[-1] - tmin, 1e-6)

    k = int(min(n_pca, X.shape[1], max(2, int(train_mask.sum()) - 1)))
    pca = PCA(n_components=k, random_state=seed).fit(X[train_mask])
    Z = pca.transform(X).astype(np.float32)

    rng = np.random.RandomState(seed)
    trans = _train_transitions(Z, t_all, train_mask, time_scale, tmin, max_pts, device, rng)
    if not trans:
        raise RuntimeError("no consecutive training transitions to fit external baseline")

    torch.manual_seed(seed)
    flow = _FieldFlow(k, kind, hidden=hidden).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=lr)
    # Train with a closed-form kernel loss (Gaussian MMD): its second derivatives are
    # cheap, so the potential flow's double-backprop stays tractable. Sinkhorn OT /
    # energy / W1 are reserved for *scoring* so the flows are not fit on the ranking
    # metric (energy_distance).
    ot = SamplesLoss("gaussian", blur=0.5)

    # The double-backprop potential flow runs on tiny (PCA-space) tensors; multi-threaded
    # BLAS/OpenMP oversubscription causes pathological futex contention here, so pin to a
    # single thread for the fit and restore afterwards.
    import os

    _prev_threads = torch.get_num_threads()
    _env_keys = ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")
    _prev_env = {ek: os.environ.get(ek) for ek in _env_keys}
    for ek in _env_keys:
        os.environ[ek] = "1"
    torch.set_num_threads(1)
    try:
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass  # already set after first parallel work
        print(f"  fitting {kind} flow (n_iter={n_iter}, max_pts={max_pts}, n_pca={k}) ...", flush=True)
        flow.train()
        for it in range(n_iter):
            opt.zero_grad()
            total = 0.0
            for src, tgt, dt_norm, t0_norm in trans:
                rolled = flow.roll(src, dt_norm, t0_norm, n_sub=n_sub)
                total = total + ot(rolled, tgt)
            total = total / len(trans)
            total.backward()
            opt.step()
            if it == 0 or (it + 1) % 40 == 0 or it + 1 == n_iter:
                print(f"    {kind} iter {it+1}/{n_iter} loss={float(total):.4f}", flush=True)
        flow.eval()
    finally:
        torch.set_num_threads(_prev_threads)
        for k, v in _prev_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
    _FLOW_CACHE[key] = (flow, pca, time_scale, tmin, n_sub)
    return _FLOW_CACHE[key]


def _field_flow_predict(
    adata,
    config,
    split,
    transition,
    kind,
    *,
    source_population=None,
    seed=0,
):
    flow, pca, time_scale, tmin, n_sub = _fit_field_flow(
        adata, config, split, kind, seed=seed
    )
    device = "cpu"  # matches the CPU-trained flow (see _fit_field_flow)
    X = _get_X(adata)
    t_curr, t_next = float(transition[0]), float(transition[1])
    if source_population is None:
        t_all = adata.obs[config.time_key].astype(float).values
        src = X[split.train_mask & np.isclose(t_all, t_curr)]
    else:
        src = np.asarray(source_population, dtype=np.float32)
    if src.shape[0] == 0:
        return np.zeros((0, X.shape[1]), dtype=np.float32)
    z = torch.tensor(pca.transform(src).astype(np.float32), device=device)
    dt_norm = (t_next - t_curr) / time_scale
    t0_norm = (t_curr - tmin) / time_scale
    with torch.enable_grad():  # potential flow needs grad; velocity flow ignores it
        rolled = flow.roll(z, dt_norm, t0_norm, n_sub=n_sub)
    z_pred = rolled.detach().cpu().numpy().astype(np.float32)
    return pca.inverse_transform(z_pred).astype(np.float32)


def baseline_prescient_potential_flow(
    adata, config, split, transition, *, source_population=None, seed=0
):
    return _field_flow_predict(
        adata,
        config,
        split,
        transition,
        "potential",
        source_population=source_population,
        seed=seed,
    )


def baseline_mioflow_neural_ode(
    adata, config, split, transition, *, source_population=None, seed=0
):
    return _field_flow_predict(
        adata,
        config,
        split,
        transition,
        "velocity",
        source_population=source_population,
        seed=seed,
    )


# External baseline adapters. Register a callable with signature
#   fn(adata, config, split, transition, *, source_population, seed) -> np.ndarray
# and it is included automatically. The two entries below are dependency-safe in-harness
# reimplementations of the PRESCIENT / MIOFlow transport objectives.
EXTERNAL_BASELINES: Dict[str, Callable] = {
    "prescient_potential_flow": baseline_prescient_potential_flow,
    "mioflow_neural_ode": baseline_mioflow_neural_ode,
}


# ------------------------------------------------------------------ model predictor


def _predict_chunk(model, x_curr, ct_curr, t_curr, t_next, device, *, return_latent=False):
    x = torch.tensor(x_curr, dtype=torch.float32, device=device)
    ct = torch.tensor(ct_curr, dtype=torch.long, device=device)
    z = model.encode(x, ct)
    ts = torch.tensor([float(t_curr), float(t_next)], dtype=torch.float32, device=device)
    traj = model.integrate_latent(z, ts)
    z_pred = traj[-1]
    if return_latent:
        return z_pred.detach().cpu().numpy().astype(np.float32)
    t_col = torch.full((z_pred.shape[0], 1), float(t_next), device=device)
    expr = model.predict_expression(z_pred, t_col)
    return expr.detach().cpu().numpy().astype(np.float32)


def _model_predict_population(
    model, x_curr, ct_curr, t_curr, t_next, device, batch_size=512, *, return_latent=False
):
    # Integration needs autograd for -grad U(z, t); disable dropout/noise via eval().
    model.eval()
    try:
        outs = []
        for s in range(0, len(x_curr), batch_size):
            e = min(s + batch_size, len(x_curr))
            outs.append(
                _predict_chunk(
                    model,
                    x_curr[s:e],
                    ct_curr[s:e],
                    t_curr,
                    t_next,
                    device,
                    return_latent=return_latent,
                )
            )
        return np.concatenate(outs, axis=0)
    except torch.cuda.OutOfMemoryError:
        # GPU contention fallback: retry on CPU (slower but robust).
        print("WARNING: CUDA OOM in model prediction; falling back to CPU.", flush=True)
        torch.cuda.empty_cache()
        model_cpu = model.to("cpu")
        outs = []
        for s in range(0, len(x_curr), batch_size):
            e = min(s + batch_size, len(x_curr))
            outs.append(
                _predict_chunk(
                    model_cpu,
                    x_curr[s:e],
                    ct_curr[s:e],
                    t_curr,
                    t_next,
                    "cpu",
                    return_latent=return_latent,
                )
            )
        return np.concatenate(outs, axis=0)


def _encode_population(model, x, ct, device, batch_size=512):
    """Map gene-space populations into LDH latent space via the trained encoder."""
    model.eval()
    outs = []
    with torch.no_grad():
        for s in range(0, len(x), batch_size):
            e = min(s + batch_size, len(x))
            xb = torch.tensor(x[s:e], dtype=torch.float32, device=device)
            ctb = torch.tensor(ct[s:e], dtype=torch.long, device=device)
            zb = model.encode(xb, ctb)
            outs.append(zb.detach().cpu().numpy().astype(np.float32))
    return np.concatenate(outs, axis=0) if outs else np.zeros((0, 0), dtype=np.float32)


def _fit_score_pca(X: np.ndarray, train_mask: np.ndarray, n_pca: int = 50, seed: int = 0):
    """PCA on training cells for optional reduced-space scoring (avoids AE gene-space floor)."""
    from sklearn.decomposition import PCA

    k = int(min(n_pca, X.shape[1], max(2, int(train_mask.sum()) - 1)))
    return PCA(n_components=k, random_state=seed).fit(X[train_mask])


# ------------------------------------------------------------------ driver


def _sync_config_to_checkpoint(config: Config, checkpoint_dir: str) -> None:
    """Align architecture-defining config fields with the trained checkpoint.

    The model architecture (potential parameterization, momentum head) must match the
    weights being loaded, otherwise a partial strict=False load silently leaves those
    sub-networks randomly initialized. These fields are persisted in training_summary.json.
    """
    import json

    summary_path = Path(checkpoint_dir) / "training_summary.json"
    if not summary_path.is_file():
        print(
            f"WARNING: {summary_path} missing; using default architecture flags "
            "(model weights may not load correctly).",
            flush=True,
        )
        return
    with open(summary_path) as fh:
        summary = json.load(fh)
    for field in (
        "potential_time_mode",
        "potential_time_correction_scale",
        "use_state_momentum",
        "residual_drift_mode",
        "use_residual_drift",
        "use_cell_type_embedding",
        "hamiltonian_damping_gamma",
    ):
        if field in summary and summary[field] is not None:
            setattr(config, field, summary[field])
    print(
        f"Model architecture from checkpoint: potential_time_mode="
        f"{getattr(config, 'potential_time_mode', None)}, use_state_momentum="
        f"{getattr(config, 'use_state_momentum', None)}",
        flush=True,
    )


def _prev_train_time(train_times: List[float], t_curr: float) -> Optional[float]:
    earlier = [t for t in sorted(train_times) if t < t_curr - 1e-9]
    return earlier[-1] if earlier else None


def evaluate_baselines(
    adata,
    config: Config,
    *,
    checkpoint_dir: Optional[str] = None,
    methods: Optional[List[str]] = None,
    max_cells: int = 1000,
    max_source_cells: int = 512,
    seed: int = 0,
    primary_metric: str = "energy_distance",
    score_space: str = "gene",
    n_pca: int = 50,
) -> pd.DataFrame:
    from geomloss import SamplesLoss

    if score_space not in ("gene", "pca", "latent"):
        raise ValueError(f"score_space must be 'gene', 'pca', or 'latent', got {score_space!r}")

    ot_loss = SamplesLoss("sinkhorn", p=2, blur=0.05, scaling=0.5, debias=True)
    device = config.device

    split = ValidationSplit(adata, config)
    transitions = split.val_transitions or split.split_info.get("val_transition_pairs", [])
    train_times = split.split_info.get("train_times", [])
    if not transitions:
        raise SystemExit(
            "No held-out transitions found; use --val-mode time_extrapolate (or time)."
        )

    X = adata.X.toarray() if hasattr(adata.X, "toarray") else np.asarray(adata.X)
    X = X.astype(np.float32)
    t_all = adata.obs[config.time_key].astype(float).values
    ct_all = adata.obs[config.cell_type_key].values.astype(int)

    score_pca = None
    if score_space == "pca":
        score_pca = _fit_score_pca(X, split.train_mask, n_pca=n_pca, seed=seed)
        print(
            f"Scoring in PCA space (n_components={score_pca.n_components_}) to avoid "
            "gene-space AE reconstruction floor.",
            flush=True,
        )

    model = None
    if checkpoint_dir is not None:
        ckpt = Path(checkpoint_dir) / "best_model.pth"
        if ckpt.is_file():
            _sync_config_to_checkpoint(config, checkpoint_dir)
            model = TemporalSDENetwork(config, adata).to(device)
            _load_state_dict_compat(model, torch.load(ckpt, map_location=device))
        else:
            print(f"WARNING: checkpoint {ckpt} not found; skipping model.", flush=True)

    if score_space == "latent":
        if model is None:
            raise SystemExit(
                "score_space=latent requires a loaded LDH checkpoint (encoder + dynamics)."
            )
        print(
            "Scoring in LDH latent space: LDH uses native integrated z; other methods' "
            "gene-space predictions are encoded with the same LDH encoder.",
            flush=True,
        )

    requested = methods or (
        ["our_model"] + list(SELF_CONTAINED_BASELINES) + list(EXTERNAL_BASELINES)
    )
    rows: List[Dict] = []

    rng_src = np.random.RandomState(seed)
    for (t_curr, t_next) in transitions:
        t_curr, t_next = float(t_curr), float(t_next)
        dt = t_next - t_curr
        src_mask = split.train_mask & np.isclose(t_all, t_curr)
        tgt_mask = split.val_mask & np.isclose(t_all, t_next)
        if src_mask.sum() == 0 or tgt_mask.sum() == 0:
            continue
        x_curr = X[src_mask]
        ct_curr = ct_all[src_mask]
        x_target = X[tgt_mask]
        ct_target = ct_all[tgt_mask]

        # Subsample the source population uniformly ONCE per transition and reuse the
        # exact same cells for every method (fair) to keep model integration + flow
        # rollouts tractable. Metrics further subsample to max_cells.
        if max_source_cells and x_curr.shape[0] > max_source_cells:
            sel = rng_src.choice(x_curr.shape[0], size=max_source_cells, replace=False)
            x_curr = x_curr[sel]
            ct_curr = ct_curr[sel]

        t_prev = _prev_train_time(train_times, t_curr)
        x_prev_step = X[split.train_mask & np.isclose(t_all, t_prev)] if t_prev is not None else None
        dt_prev = (t_curr - t_prev) if t_prev is not None else None

        # Precompute latent target once per transition when scoring in LDH latent space.
        z_target = None
        if score_space == "latent":
            z_target = _encode_population(model, x_target, ct_target, device)

        print(
            f"[transition {t_curr}->{t_next}] source={x_curr.shape[0]} target={x_target.shape[0]} "
            f"score_space={score_space}",
            flush=True,
        )
        for method in requested:
            _t0 = __import__("time").time()
            if method == "our_model":
                if model is None:
                    continue
                pred = _model_predict_population(
                    model,
                    x_curr,
                    ct_curr,
                    t_curr,
                    t_next,
                    device,
                    return_latent=(score_space == "latent"),
                )
            elif method in SELF_CONTAINED_BASELINES:
                pred = SELF_CONTAINED_BASELINES[method](x_curr, x_prev_step, dt, dt_prev)
            elif method in EXTERNAL_BASELINES:
                try:
                    pred = EXTERNAL_BASELINES[method](
                        adata,
                        config,
                        split,
                        (t_curr, t_next),
                        source_population=x_curr,
                        seed=seed,
                    )
                except Exception as exc:  # noqa: BLE001
                    print(f"WARNING: external baseline {method} failed: {exc}", flush=True)
                    continue
            else:
                print(f"WARNING: unknown method {method}; skipping.", flush=True)
                continue

            if score_space == "latent":
                if method == "our_model":
                    pred_score = pred  # already integrated latent
                else:
                    # Same encoder; use source cell-type codes (cells rolled forward).
                    pred_score = _encode_population(model, pred, ct_curr, device)
                tgt_score = z_target
            elif score_pca is not None:
                pred_score = score_pca.transform(pred).astype(np.float32)
                tgt_score = score_pca.transform(x_target).astype(np.float32)
            else:
                pred_score, tgt_score = pred, x_target

            m = population_distribution_metrics(
                pred_score, tgt_score, ot_loss=ot_loss, max_cells=max_cells, seed=seed
            )
            m.update(
                {
                    "method": method,
                    "t_curr": t_curr,
                    "t_next": t_next,
                    "n_source": int(x_curr.shape[0]),
                    "n_target": int(tgt_mask.sum()),
                    "score_space": score_space,
                    "seed": int(seed),
                    "implementation": (
                        "method_family_core_reimplementation"
                        if method in EXTERNAL_BASELINES
                        else (
                            "wot_inspired_barycentric_transport"
                            if method == "wot_barycentric"
                            else "native"
                        )
                    ),
                }
            )
            rows.append(m)
            print(
                f"  {method:24s} energy={m['energy_distance']:.4f} "
                f"w1={m['mean_marginal_w1']:.4f} ot={m['ot_sinkhorn']:.2f} "
                f"({__import__('time').time() - _t0:.1f}s)",
                flush=True,
            )

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    metric_cols = ["ot_sinkhorn", "mmd", "energy_distance", "mean_marginal_w1", "mean_shift_l2"]
    if primary_metric not in metric_cols:
        raise ValueError(f"primary_metric must be one of {metric_cols}, got {primary_metric!r}")
    agg = df.groupby("method")[metric_cols].mean().reset_index()
    # Rank by a NON-training-objective metric so no method is judged on its own loss.
    agg = agg.sort_values(primary_metric).reset_index(drop=True)
    # Reorder columns so the primary metric is up front and OT (training objective) trails.
    ordered = ["method", primary_metric] + [c for c in metric_cols if c != primary_metric]
    agg = agg[ordered]
    df.attrs["aggregate"] = agg
    df.attrs["primary_metric"] = primary_metric
    df.attrs["metric_cols"] = metric_cols
    df.attrs["score_space"] = score_space
    return df


def _write_report(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "baseline_per_transition.csv", index=False)
    agg = df.attrs.get("aggregate")
    if agg is None or agg.empty:
        return
    agg.to_csv(out_dir / "baseline_comparison.csv", index=False)
    try:
        table = agg.to_markdown(index=False)
    except ImportError:
        table = agg.to_string(index=False)

    primary = df.attrs.get("primary_metric", "energy_distance")
    metric_cols = df.attrs.get("metric_cols", list(agg.columns[1:]))
    score_space = df.attrs.get("score_space", "gene")
    lines = [
        "# Baseline comparison (mean over held-out transitions)",
        "",
        "Lower is better for all metrics.",
        "",
        f"**Score space: `{score_space}`**"
        + (
            " (PCA on training cells; avoids gene-space AE reconstruction floor)."
            if score_space == "pca"
            else (
                " (LDH latent z; LDH uses native integrated latent, other methods "
                "are gene-predicted then encoded by the LDH encoder)."
                if score_space == "latent"
                else " (raw gene expression)."
            )
        ),
        "",
        f"**Primary ranking metric: `{primary}`** (a non-training objective; the table is "
        "sorted by it). `ot_sinkhorn` is the model's *training* objective and is shown for "
        "reference only — comparing methods on it would be biased toward the model.",
        "",
        "**Implementation fidelity:** `prescient_potential_flow` and "
        "`mioflow_neural_ode` are controlled reimplementations of the corresponding "
        "method-family objectives, not official package runs. `wot_barycentric` is a "
        "WOT-inspired entropic-transport baseline without WOT's growth-rate model.",
        "",
        table,
        "",
        "## Winner per metric",
        "",
    ]
    for col in metric_cols:
        winner = agg.loc[agg[col].idxmin(), "method"]
        tag = " _(training objective — reference only)_" if col == "ot_sinkhorn" else ""
        lines.append(f"- `{col}`: **{winner}**{tag}")

    best = agg.iloc[0]["method"]
    lines += [
        "",
        f"Best method by the primary metric (`{primary}`): **{best}**.",
    ]
    (out_dir / "baseline_comparison.md").write_text("\n".join(lines), encoding="utf-8")


def main(argv=None):
    from run_training import DATASETS
    from dataset_pipeline import apply_train_config, build_training_checkpoint_dir, resolve_data_path

    parser = argparse.ArgumentParser(description="Baseline comparison for trajectory prediction")
    parser.add_argument("--dataset", required=True, choices=sorted(DATASETS.keys()))
    parser.add_argument("--checkpoint-dir", default=None, help="Dir with best_model.pth (default: auto-resolve)")
    parser.add_argument("--save-dir", required=True)
    parser.add_argument("--val-mode", default="time_extrapolate",
                        choices=["time", "time_extrapolate", "patients", "cells"])
    parser.add_argument("--val-ratio", type=float, default=None)
    parser.add_argument("--methods", nargs="*", default=None,
                        help="Subset of: our_model persistence global_linear_drift "
                             "wot_barycentric prescient_potential_flow "
                             "mioflow_neural_ode")
    parser.add_argument("--max-cells", type=int, default=1000)
    parser.add_argument("--max-source-cells", type=int, default=512,
                        help="Uniformly subsample the source population per transition "
                             "(same cells for all methods) to keep prediction tractable. "
                             "0 disables subsampling.")
    parser.add_argument("--processed-cache", default=None,
                        help="Path to a cached processed .h5ad. If it exists it is loaded "
                             "(skipping the ~minutes-long preprocessing); otherwise the "
                             "processed AnnData is written there for reuse.")
    parser.add_argument("--primary-metric", default="energy_distance",
                        choices=["energy_distance", "mean_marginal_w1", "mmd",
                                 "mean_shift_l2", "ot_sinkhorn"],
                        help="Non-training objective used to rank the main table "
                             "(default: energy_distance).")
    parser.add_argument(
        "--score-space",
        default="gene",
        choices=["gene", "pca", "latent"],
        help="Where to score predictions: gene (expression), pca (training-fit PCA), "
             "or latent (LDH encoder latent; LDH uses native z, baselines are encoded).",
    )
    parser.add_argument("--n-pca", type=int, default=50, help="PCA dims when --score-space pca")
    parser.add_argument("--device", default=None, help="Override model device (e.g. cpu or cuda)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args(argv)

    spec, prepare_fn = DATASETS[args.dataset]
    config = apply_train_config(spec)
    config.val_mode = args.val_mode
    if args.device is not None:
        config.device = str(args.device)
    if args.val_ratio is not None:
        config.val_ratio = float(args.val_ratio)
    config.data_path = resolve_data_path(spec)

    cache = args.processed_cache
    if cache and Path(cache).is_file():
        print(f"Loading cached processed AnnData from {cache}", flush=True)
        adata = sc.read(cache)
    else:
        adata = sc.read(config.data_path)
        adata = prepare_fn(adata, config)
        adata = TemporalDataProcessor(adata).process()
        if cache:
            Path(cache).parent.mkdir(parents=True, exist_ok=True)
            print(f"Writing processed AnnData cache to {cache}", flush=True)
            adata.write(cache)

    # prepare_fn (e.g. HGSOC NACT-pair) may reset val_mode; re-apply the CLI choice last.
    config.val_mode = args.val_mode

    checkpoint_dir = args.checkpoint_dir
    if checkpoint_dir is None:
        checkpoint_dir = build_training_checkpoint_dir(spec, config)
        print(f"Auto-resolved checkpoint dir: {checkpoint_dir}", flush=True)

    df = evaluate_baselines(
        adata,
        config,
        checkpoint_dir=checkpoint_dir,
        methods=args.methods,
        max_cells=args.max_cells,
        max_source_cells=args.max_source_cells,
        seed=args.seed,
        primary_metric=args.primary_metric,
        score_space=args.score_space,
        n_pca=args.n_pca,
    )
    if df.empty:
        print("No comparable transitions produced results.", flush=True)
        return
    _write_report(df, Path(args.save_dir))
    print(df.attrs["aggregate"].to_string(index=False), flush=True)
    print(f"\nReport written to {args.save_dir}/baseline_comparison.md", flush=True)


if __name__ == "__main__":
    sys.exit(main())
