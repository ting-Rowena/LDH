#!/usr/bin/env python3
"""Compare LDH vs PRESCIENT / MIOFlow / WOT on Fig.2–aligned axes.

1) Trajectory–time Markov hitting PCC (real labels)
2) Matched temporal null sensitivity on the *same* Markov PCC:
   fit / condition on shuffled time, evaluate PCC against clean bio_t.

LDH's official Fig.2 null (holdout predictive PCC + U₀–KDE Spearman after
full retrain) is also copied from existing summary JSONs for reference —
that protocol is LDH-native and not identically available for the baselines.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import spearmanr

HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "output_file"))

from _adopted import ADOPTED  # noqa: E402
from celltype_analysis import DATASET_REGISTRY, load_annotated_adata  # noqa: E402
from latent_embeddings import ensure_latent_embeddings  # noqa: E402
from methods_enhancement_utils import trajectory_time_pcc  # noqa: E402
from run_deep_temporal_pcc_benchmark import (  # noqa: E402
    _field_velocity,
    _fit_field_on_embedding,
    _wot_velocity,
)
from run_sota_velocity_benchmark import (  # noqa: E402
    _biological_time,
    _embedding_coords,
    _momentum_velocity,
    _run_cellrank,
    _subset_for_benchmark,
    _terminal_mask,
)

NULL_JSON = {
    "GSE155622": ADOPTED["GSE155622"]
    / "methods_enhancement"
    / "physical_retrain_controls_GSE155622_temporal_matched_e500_mc5000_bs128_summary.json",
    "GSE141259": ADOPTED["GSE141259"]
    / "methods_enhancement"
    / "physical_retrain_controls_GSE141259_temporal_matched_e500_mc5000_bs128_summary.json",
    "HGSOC": ADOPTED["HGSOC"]
    / "methods_enhancement"
    / "physical_retrain_controls_HGSOC_pairing_matched_e500_mc5000_bs128_summary.json",
}


def _shuffle_bio_t(bio_t: np.ndarray, dataset: str, adata, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = np.asarray(bio_t, dtype=float).copy()
    if dataset == "HGSOC" and "patient_id" in adata.obs.columns:
        for pid in adata.obs["patient_id"].astype(str).unique():
            m = np.asarray(adata.obs["patient_id"].astype(str) == pid)
            vals = out[m].copy()
            rng.shuffle(vals)
            out[m] = vals
        return out
    perm = rng.permutation(len(out))
    return out[perm]


def _potential_kde_spearman(coords: np.ndarray, potential: np.ndarray) -> float:
    """Spearman(U, -log KDE density) — higher means deeper valleys = denser regions."""
    from sklearn.neighbors import KernelDensity

    x = np.asarray(coords, dtype=float)
    u = np.asarray(potential, dtype=float).reshape(-1)
    m = np.isfinite(u) & np.isfinite(x).all(axis=1)
    if m.sum() < 50:
        return float("nan")
    kde = KernelDensity(bandwidth=0.5).fit(x[m])
    neglogp = -kde.score_samples(x[m])
    if np.std(u[m]) < 1e-12 or np.std(neglogp) < 1e-12:
        return 0.0
    return float(spearmanr(u[m], neglogp).correlation)


def _prescient_potential(flow, coords: np.ndarray) -> np.ndarray:
    import torch

    x = torch.tensor(coords, dtype=torch.float32)
    with torch.no_grad():
        # potential network outputs Psi; PRESCIENT velocity = -grad Psi
        psi = flow.net(x).squeeze(-1).cpu().numpy()
    return psi.astype(float)


def _markov_pcc(adata, vel, term, bio_t) -> float:
    cr = _run_cellrank(adata, vel, term, bio_t=bio_t)
    return float(cr.get("trajectory_time_pcc", np.nan))


def run_dataset(dataset: str, device: str, max_cells: int, n_null: int) -> pd.DataFrame:
    ck = ADOPTED[dataset]
    profile = DATASET_REGISTRY[dataset]
    adata = load_annotated_adata(profile, str(ck))
    ensure_latent_embeddings(adata, checkpoint_dir=str(ck), warn=False)
    adata = _subset_for_benchmark(adata, dataset)
    coords, emb = _embedding_coords(adata, prefer_latent_pca=True, n_dims=10)
    finite = np.isfinite(coords).all(axis=1)
    if not finite.all():
        adata = adata[finite].copy()
    if adata.n_obs > max_cells:
        sc.pp.subsample(adata, n_obs=max_cells, random_state=0, copy=False)
    bio_t = _biological_time(adata, dataset)
    coords, emb = _embedding_coords(adata, prefer_latent_pca=True, n_dims=10)
    term = _terminal_mask(adata, dataset)

    rows = []

    # --- LDH real ---
    mom = _momentum_velocity(adata, ck, device=device, emb_key=emb, n_dims=10)
    if mom is None:
        raise RuntimeError(f"no MomentumNetwork velocity for {dataset}")
    ldh_real = _markov_pcc(adata, mom, term, bio_t)
    rows.append(
        {
            "dataset": dataset,
            "method": "LDH-scRNA",
            "setting": "real",
            "markov_hitting_pcc": ldh_real,
            "U_neglogKDE_spearman": np.nan,  # filled from official JSON below
            "note": "momentum→Markov; U–KDE from official Fig.2 JSON",
        }
    )

    # LDH null proxy: condition momentum on shuffled time at inference (not full retrain)
    null_ldh = []
    for s in range(n_null):
        t_null = _shuffle_bio_t(bio_t, dataset, adata, seed=100 + s)
        # temporarily write shuffled time into a thin wrapper via bio_t only for momentum path:
        # _momentum_velocity reads adata.obs time/stage — patch obs copy
        ad_n = adata.copy()
        if "time" in ad_n.obs.columns:
            ad_n.obs["time"] = t_null
        if dataset == "GSE141259" and "stage" in ad_n.obs.columns:
            # keep stage strings; momentum uses stage map if no time — set numeric via time
            ad_n.obs["time"] = t_null
        if dataset == "GSE155622" and "condition" in ad_n.obs.columns:
            ad_n.obs["time"] = t_null
        mom_n = _momentum_velocity(ad_n, ck, device=device, emb_key=emb, n_dims=10)
        null_ldh.append(_markov_pcc(adata, mom_n, term, bio_t))
    rows.append(
        {
            "dataset": dataset,
            "method": "LDH-scRNA",
            "setting": "null_median",
            "markov_hitting_pcc": float(np.nanmedian(null_ldh)),
            "U_neglogKDE_spearman": np.nan,
            "note": f"inference-time shuffle proxy n={n_null}; NOT full Fig.2 retrain",
        }
    )

    # --- PRESCIENT / MIOFlow / WOT real + null ---
    for method, kind in (
        ("PRESCIENT-family", "potential"),
        ("MIOFlow-family", "velocity"),
        ("WOT-inspired", "wot"),
    ):
        if kind == "wot":
            vel = _wot_velocity(coords, bio_t)
            u_spear = np.nan
        else:
            flow = _fit_field_on_embedding(coords, bio_t, kind, seed=0)
            vel = _field_velocity(flow, coords, bio_t)
            u_spear = (
                _potential_kde_spearman(coords, _prescient_potential(flow, coords))
                if kind == "potential"
                else np.nan
            )
        real_pcc = _markov_pcc(adata, vel, term, bio_t)
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "setting": "real",
                "markov_hitting_pcc": real_pcc,
                "U_neglogKDE_spearman": u_spear,
                "note": "fit on clean bio_t",
            }
        )

        null_pccs = []
        null_us = []
        for s in range(n_null):
            t_null = _shuffle_bio_t(bio_t, dataset, adata, seed=200 + s)
            if kind == "wot":
                vel_n = _wot_velocity(coords, t_null, seed=s)
                null_us.append(np.nan)
            else:
                flow_n = _fit_field_on_embedding(coords, t_null, kind, seed=s)
                vel_n = _field_velocity(flow_n, coords, t_null)
                if kind == "potential":
                    null_us.append(
                        _potential_kde_spearman(coords, _prescient_potential(flow_n, coords))
                    )
                else:
                    null_us.append(np.nan)
            null_pccs.append(_markov_pcc(adata, vel_n, term, bio_t))
        rows.append(
            {
                "dataset": dataset,
                "method": method,
                "setting": "null_median",
                "markov_hitting_pcc": float(np.nanmedian(null_pccs)),
                "U_neglogKDE_spearman": float(np.nanmedian(null_us)) if np.isfinite(null_us).any() else np.nan,
                "note": f"fit on shuffled bio_t; eval Markov vs clean bio_t; n={n_null}",
            }
        )

    return pd.DataFrame(rows)


def load_ldh_fig2_null() -> pd.DataFrame:
    rows = []
    for ds, path in NULL_JSON.items():
        if not path.is_file():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": ds,
                "method": "LDH-scRNA",
                "real_holdout_pcc": rec.get("real_holdout_pcc"),
                "null_median_holdout_pcc": rec.get("null_median_holdout_pcc"),
                "holdout_pcc_collapse_ratio": rec.get("holdout_pcc_collapse_ratio"),
                "real_U_neglogKDE_spearman": rec.get("real_spearman"),
                "null_median_U_neglogKDE_spearman": rec.get("null_median_spearman"),
                "U_spearman_collapse_ratio": rec.get("collapse_ratio"),
                "shuffle_mode": rec.get("shuffle_mode"),
                "protocol": "Fig.2 full retrain null (LDH-native)",
            }
        )
    return pd.DataFrame(rows)


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--datasets", nargs="+", default=["GSE155622", "GSE141259", "HGSOC"])
    p.add_argument("--device", default="cuda")
    p.add_argument("--max-cells", type=int, default=5000)
    p.add_argument("--n-null", type=int, default=3)
    p.add_argument(
        "--save-dir",
        type=Path,
        default=HERE / "results" / "fig2_axes",
    )
    args = p.parse_args(argv)
    args.save_dir.mkdir(parents=True, exist_ok=True)

    fig2 = load_ldh_fig2_null()
    fig2.to_csv(args.save_dir / "ldh_fig2_official_null.csv", index=False)

    frames = []
    for ds in args.datasets:
        print(f"[fig2-axes] {ds}", flush=True)
        df = run_dataset(ds, args.device, args.max_cells, args.n_null)
        df.to_csv(args.save_dir / f"markov_null_{ds}.csv", index=False)
        print(df.to_string(index=False), flush=True)
        frames.append(df)

    all_df = pd.concat(frames, ignore_index=True)
    all_df.to_csv(args.save_dir / "markov_real_null_all.csv", index=False)

    # Wide summary: real PCC, null PCC, collapse
    parts = []
    for (ds, method), g in all_df.groupby(["dataset", "method"]):
        real = g.loc[g.setting == "real", "markov_hitting_pcc"]
        null = g.loc[g.setting == "null_median", "markov_hitting_pcc"]
        r = float(real.iloc[0]) if len(real) else np.nan
        n = float(null.iloc[0]) if len(null) else np.nan
        parts.append(
            {
                "dataset": ds,
                "method": method,
                "real_markov_pcc": r,
                "null_markov_pcc": n,
                "markov_collapse_ratio": (n / r) if abs(r) > 1e-6 else np.nan,
            }
        )
    summary = pd.DataFrame(parts).sort_values(["dataset", "real_markov_pcc"], ascending=[True, False])
    summary.to_csv(args.save_dir / "markov_pcc_real_vs_null_summary.csv", index=False)

    print("\n=== Markov hitting PCC: real vs matched-time null ===", flush=True)
    print(summary.to_string(index=False), flush=True)
    print("\n=== LDH official Fig.2 null (holdout PCC / U–KDE) ===", flush=True)
    print(fig2.to_string(index=False), flush=True)
    print(f"\nwrote {args.save_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
