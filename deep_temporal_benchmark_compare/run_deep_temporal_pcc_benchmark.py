#!/usr/bin/env python3
"""Trajectory–time PCC for PRESCIENT / MIOFlow / WOT-inspired vs LDH-scRNA.

This is the Fig. 2d–aligned evaluation axis:

    velocity field → absorbing-Markov hitting order → PCC(order, biological time)

Population OT / Energy / MMD metrics are intentionally *not* used here: those
score marginal transport, which is the native objective of first-order OT-flow
methods and is not the manuscript's primary claim for LDH-scRNA.

Fidelity note: PRESCIENT / MIOFlow / WOT entries are controlled core-objective
reimplementations, not official package runs.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch import autograd, nn

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))

from methods_enhancement_utils import trajectory_time_pcc  # noqa: E402
from run_sota_velocity_benchmark import (  # noqa: E402
    _biological_time,
    _embedding_coords,
    _graph_pseudotime_from_velocity,
    _momentum_velocity,
    _run_cellrank,
    _subset_for_benchmark,
    _terminal_mask,
)
from celltype_analysis import DATASET_REGISTRY, load_annotated_adata  # noqa: E402
from latent_embeddings import ensure_latent_embeddings  # noqa: E402
import scanpy as sc  # noqa: E402

sys.path.insert(0, str(REPO_ROOT / "output_file"))
from _adopted import ADOPTED  # noqa: E402


class _MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 64, depth: int = 2):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for _ in range(depth):
            layers += [nn.Linear(d, hidden), nn.SiLU()]
            d = hidden
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class _FieldFlow(nn.Module):
    def __init__(self, dim: int, kind: str, hidden: int = 64):
        super().__init__()
        self.kind = kind
        if kind == "potential":
            self.net = _MLP(dim, 1, hidden=hidden)
        elif kind == "velocity":
            self.net = _MLP(dim + 1, dim, hidden=hidden)
        else:
            raise ValueError(kind)

    def velocity(self, x, t_norm: float):
        if self.kind == "velocity":
            tt = torch.full((x.shape[0], 1), float(t_norm), device=x.device, dtype=x.dtype)
            return self.net(torch.cat([x, tt], dim=1))
        xg = x if x.requires_grad else x.clone().requires_grad_(True)
        psi = self.net(xg).sum()
        (grad,) = autograd.grad(psi, xg, create_graph=self.training)
        return -grad


def _subsample(arr: np.ndarray, n: int, rng: np.random.RandomState) -> np.ndarray:
    if arr.shape[0] <= n:
        return arr
    return arr[rng.choice(arr.shape[0], size=n, replace=False)]


def _fit_field_on_embedding(
    coords: np.ndarray,
    bio_t: np.ndarray,
    kind: str,
    *,
    n_iter: int = 80,
    max_pts: int = 200,
    seed: int = 0,
) -> _FieldFlow:
    """Fit PRESCIENT-/MIOFlow-family flow in the *same* embedding used for PCC."""
    from geomloss import SamplesLoss

    times = sorted({float(t) for t in bio_t if np.isfinite(t)})
    if len(times) < 2:
        raise RuntimeError("need ≥2 biological time points to fit a temporal field")

    tmin, tmax = times[0], times[-1]
    scale = max(tmax - tmin, 1e-6)
    rng = np.random.RandomState(seed)
    device = "cpu"
    pairs = []
    for a, b in zip(times[:-1], times[1:]):
        src = coords[np.isclose(bio_t, a)]
        tgt = coords[np.isclose(bio_t, b)]
        if src.shape[0] == 0 or tgt.shape[0] == 0:
            continue
        pairs.append(
            (
                torch.tensor(_subsample(src, max_pts, rng), dtype=torch.float32, device=device),
                torch.tensor(_subsample(tgt, max_pts, rng), dtype=torch.float32, device=device),
                (b - a) / scale,
                (a - tmin) / scale,
            )
        )
    if not pairs:
        raise RuntimeError("no consecutive time marginals for field fit")

    torch.manual_seed(seed)
    flow = _FieldFlow(coords.shape[1], kind, hidden=64).to(device)
    opt = torch.optim.Adam(flow.parameters(), lr=2e-3)
    loss_fn = SamplesLoss("gaussian", blur=0.5)
    flow.train()
    for _ in range(n_iter):
        opt.zero_grad()
        total = 0.0
        for src, tgt, dt_norm, t0_norm in pairs:
            h = dt_norm / 2.0
            cur = src
            for i in range(2):
                cur = cur + h * flow.velocity(cur, t0_norm + i * h)
            total = total + loss_fn(cur, tgt)
        (total / len(pairs)).backward()
        opt.step()
    flow.eval()
    flow._time_scale = scale  # type: ignore[attr-defined]
    flow._tmin = tmin  # type: ignore[attr-defined]
    return flow


def _field_velocity(flow: _FieldFlow, coords: np.ndarray, bio_t: np.ndarray) -> np.ndarray:
    device = next(flow.parameters()).device
    x = torch.tensor(coords, dtype=torch.float32, device=device)
    tmin = float(getattr(flow, "_tmin", 0.0))
    scale = float(getattr(flow, "_time_scale", 1.0))
    t_norm = (np.asarray(bio_t, dtype=float) - tmin) / scale
    t_norm = np.nan_to_num(t_norm, nan=0.0)
    out = np.zeros_like(coords, dtype=np.float32)
    bs = 512
    for i in range(0, len(coords), bs):
        xb = x[i : i + bs]
        # evaluate at each cell's own time; potential flow ignores t
        tb = t_norm[i : i + bs]
        # batch by unique times for velocity net; potential uses one call
        if flow.kind == "potential":
            with torch.enable_grad():
                vb = flow.velocity(xb, 0.0)
            out[i : i + bs] = vb.detach().cpu().numpy()
        else:
            chunk = np.zeros((xb.shape[0], coords.shape[1]), dtype=np.float32)
            for u in np.unique(tb):
                m = np.isclose(tb, u)
                with torch.no_grad():
                    vb = flow.velocity(xb[m], float(u))
                chunk[m] = vb.detach().cpu().numpy()
            out[i : i + bs] = chunk
    return out.astype(np.float32)


def _wot_velocity(coords: np.ndarray, bio_t: np.ndarray, *, max_pts: int = 400, seed: int = 0) -> np.ndarray:
    """WOT-inspired local barycentric displacement as an embedding velocity."""
    from scipy.spatial.distance import cdist

    times = sorted({float(t) for t in bio_t if np.isfinite(t)})
    vel = np.zeros_like(coords, dtype=float)
    if len(times) < 2:
        return vel
    rng = np.random.RandomState(seed)

    def sinkhorn(a, b, blur=0.5, n_iter=80):
        cost = cdist(a, b, metric="sqeuclidean")
        eps = blur * (cost.mean() + 1e-9)
        K = np.exp(-cost / eps)
        u = np.ones(a.shape[0]) / a.shape[0]
        v = np.ones(b.shape[0]) / b.shape[0]
        r = u.copy()
        c = v.copy()
        for _ in range(n_iter):
            u = r / (K @ v + 1e-12)
            v = c / (K.T @ u + 1e-12)
        plan = u[:, None] * K * v[None, :]
        return plan / (plan.sum(1, keepdims=True) + 1e-12)

    for a, b in zip(times[:-1], times[1:]):
        src_idx = np.flatnonzero(np.isclose(bio_t, a))
        tgt_idx = np.flatnonzero(np.isclose(bio_t, b))
        if src_idx.size == 0 or tgt_idx.size == 0:
            continue
        src = coords[src_idx]
        tgt = coords[tgt_idx]
        src_s = _subsample(src, max_pts, rng)
        tgt_s = _subsample(tgt, max_pts, rng)
        plan = sinkhorn(src_s, tgt_s)
        disp = plan @ tgt_s - src_s
        # assign each source cell the nearest subsampled displacement
        d = cdist(src, src_s)
        nn = np.argmin(d, axis=1)
        vel[src_idx] = disp[nn] / max(b - a, 1e-6)
    # later times inherit last observed velocity via nearest earlier cell
    later = bio_t >= times[-1] - 1e-9
    earlier = ~later
    if later.any() and earlier.any():
        from sklearn.neighbors import NearestNeighbors

        nbrs = NearestNeighbors(n_neighbors=1).fit(coords[earlier])
        _, idx = nbrs.kneighbors(coords[later])
        vel[later] = vel[np.flatnonzero(earlier)[idx[:, 0]]]
    return vel.astype(np.float32)


def _row(
    dataset: str,
    method: str,
    protocol: str,
    pcc: float,
    markov_pcc: float,
    emb_key: str,
    n_cells: int,
    status: str = "ok",
) -> dict:
    return {
        "dataset": dataset,
        "method": method,
        "trajectory_time_pcc": float(pcc),
        "markov_hitting_pcc": float(markov_pcc),
        "pseudotime_protocol": protocol,
        "n_cells": int(n_cells),
        "embedding": emb_key,
        "status": status,
    }


def run_dataset(dataset: str, checkpoint: Path, device: str = "cuda", max_cells: int = 5000) -> pd.DataFrame:
    profile = DATASET_REGISTRY[dataset]
    adata = load_annotated_adata(profile, str(checkpoint))
    ensure_latent_embeddings(adata, checkpoint_dir=str(checkpoint), warn=False)
    adata = _subset_for_benchmark(adata, dataset)
    coords, emb_key = _embedding_coords(adata, prefer_latent_pca=True, n_dims=10)
    finite = np.isfinite(coords).all(axis=1)
    if finite.sum() < 100:
        raise ValueError(f"too few finite cells: {finite.sum()}")
    if not finite.all():
        adata = adata[finite].copy()
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0, copy=False)

    bio_t = _biological_time(adata, dataset)
    coords, emb_key = _embedding_coords(adata, prefer_latent_pca=True, n_dims=10)
    term = _terminal_mask(adata, dataset)
    rows: List[dict] = []

    # LDH-scRNA MomentumNetwork (same fair protocol as Fig. 2d)
    mom = _momentum_velocity(adata, checkpoint, device=device, emb_key=emb_key, n_dims=10)
    if mom is None:
        raise RuntimeError(f"MomentumNetwork velocity unavailable for {dataset}")
    pt_mom = _graph_pseudotime_from_velocity(coords, mom, bio_t)
    cr_mom = _run_cellrank(adata, mom, term, bio_t=bio_t)
    rows.append(
        _row(
            dataset,
            "LDH-scRNA",
            "velocity_graph_laplacian",
            trajectory_time_pcc(pt_mom, bio_t),
            float(cr_mom.get("trajectory_time_pcc", np.nan)),
            emb_key,
            adata.n_obs,
            status=str(cr_mom.get("cellrank_status", "ok")),
        )
    )

    # PRESCIENT-family
    try:
        flow_p = _fit_field_on_embedding(coords, bio_t, "potential")
        vel_p = _field_velocity(flow_p, coords, bio_t)
        pt_p = _graph_pseudotime_from_velocity(coords, vel_p, bio_t)
        cr_p = _run_cellrank(adata, vel_p, term, bio_t=bio_t)
        rows.append(
            _row(
                dataset,
                "PRESCIENT-family",
                "velocity_graph_laplacian",
                trajectory_time_pcc(pt_p, bio_t),
                float(cr_p.get("trajectory_time_pcc", np.nan)),
                emb_key,
                adata.n_obs,
                status=str(cr_p.get("cellrank_status", "ok")),
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_row(dataset, "PRESCIENT-family", "failed", np.nan, np.nan, emb_key, adata.n_obs, status=str(exc)))

    # MIOFlow-family
    try:
        flow_v = _fit_field_on_embedding(coords, bio_t, "velocity")
        vel_v = _field_velocity(flow_v, coords, bio_t)
        pt_v = _graph_pseudotime_from_velocity(coords, vel_v, bio_t)
        cr_v = _run_cellrank(adata, vel_v, term, bio_t=bio_t)
        rows.append(
            _row(
                dataset,
                "MIOFlow-family",
                "velocity_graph_laplacian",
                trajectory_time_pcc(pt_v, bio_t),
                float(cr_v.get("trajectory_time_pcc", np.nan)),
                emb_key,
                adata.n_obs,
                status=str(cr_v.get("cellrank_status", "ok")),
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_row(dataset, "MIOFlow-family", "failed", np.nan, np.nan, emb_key, adata.n_obs, status=str(exc)))

    # WOT-inspired
    try:
        vel_w = _wot_velocity(coords, bio_t)
        pt_w = _graph_pseudotime_from_velocity(coords, vel_w, bio_t)
        cr_w = _run_cellrank(adata, vel_w, term, bio_t=bio_t)
        rows.append(
            _row(
                dataset,
                "WOT-inspired",
                "velocity_graph_laplacian",
                trajectory_time_pcc(pt_w, bio_t),
                float(cr_w.get("trajectory_time_pcc", np.nan)),
                emb_key,
                adata.n_obs,
                status=str(cr_w.get("cellrank_status", "ok")),
            )
        )
    except Exception as exc:  # noqa: BLE001
        rows.append(_row(dataset, "WOT-inspired", "failed", np.nan, np.nan, emb_key, adata.n_obs, status=str(exc)))

    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["GSE155622", "GSE141259", "HGSOC"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-cells", type=int, default=5000)
    p.add_argument(
        "--save-dir",
        type=Path,
        default=HERE / "results" / "pcc",
    )
    args = p.parse_args(argv)

    args.save_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    for ds in args.datasets:
        ck = ADOPTED[ds]
        print(f"[pcc-benchmark] {ds} @ {ck.name}", flush=True)
        df = run_dataset(ds, ck, device=args.device, max_cells=args.max_cells)
        df.to_csv(args.save_dir / f"pcc_{ds}.csv", index=False)
        print(df.to_string(index=False), flush=True)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.save_dir / "deep_temporal_pcc_all.csv", index=False)

    # Primary manuscript-aligned metric: Markov hitting PCC (same as CellRank / LDH markov)
    pivot = all_df.pivot_table(
        index="dataset",
        columns="method",
        values="markov_hitting_pcc",
        aggfunc="first",
    )
    pivot.to_csv(args.save_dir / "deep_temporal_pcc_markov_summary.csv")
    graph = all_df.pivot_table(
        index="dataset",
        columns="method",
        values="trajectory_time_pcc",
        aggfunc="first",
    )
    graph.to_csv(args.save_dir / "deep_temporal_pcc_graph_summary.csv")

    note = (
        "# Deep temporal methods on the Fig. 2d axis\n\n"
        "Primary metric: **Markov hitting-time PCC** (velocity → absorbing Markov "
        "order → biological time). Graph-Laplacian PCC is reported as a secondary "
        "column in the detail CSV.\n\n"
        "These are method-family reimplementations, not official package runs.\n\n"
    )
    try:
        table = pivot.to_markdown()
    except Exception:
        table = pivot.to_string()
    (args.save_dir / "deep_temporal_pcc_summary.md").write_text(note + table + "\n", encoding="utf-8")
    print("\n=== Markov hitting-time PCC (primary) ===", flush=True)
    print(pivot.to_string(), flush=True)
    print(f"\nwrote {args.save_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
