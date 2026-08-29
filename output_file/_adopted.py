"""Adopted-checkpoint helpers for output_file figure/table scripts.

Each manuscript figure/table under ``output_file/`` should:

  1. Load an adopted checkpoint (``obs.csv``, ``best_model.pth``, latents, h5ad).
  2. Compute the statistics the panel needs (or call the library function that does).
  3. Plot / write the artifact under ``output_file/``.

Allowed inputs: checkpoint training artifacts and raw AnnData.
Macrophage audit tables for Supplementary Figure 7 live in
``output_file/mac_landscape_audit/``.

Expensive steps (SOTA benchmark, hybrid KO, LAP fields) may cache under
``output_file/_cache/`` after the first compute. Pass ``recompute=True`` to skip cache.
Matched temporal null retraining (500 epochs × 4) is *not* re-run by default;
real metrics are recomputed from the adopted checkpoint, and null medians are
read from the checkpoint's ``methods_enhancement/*_summary.json`` (the recorded
null experiment). ``--rebuild`` on Supplementary_table3 still launches the full
retrain protocol.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = Path(__file__).resolve().parent
CACHE = OUT_DIR / "_cache"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

CK_PAIN = ROOT / (
    "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)
CK_LUNG = ROOT / (
    "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
CK_HG = ROOT / (
    "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1"
)

ADOPTED = {
    "GSE155622": CK_PAIN,
    "GSE141259": CK_LUNG,
    "HGSOC": CK_HG,
}

PAIN_STAGES = ["Control", "SNI 6h", "SNI 24h", "SNI 2d", "SNI 7d", "SNI 14d"]
PAIN_TIME = {
    "Control": 0.0,
    "SNI 6h": 0.25,
    "SNI 24h": 1.0,
    "SNI 2d": 2.0,
    "SNI 7d": 7.0,
    "SNI 14d": 14.0,
}
LUNG_STAGES = ["D0", "D3", "D7", "D10", "D14", "D21", "D28"]
HG_TYPES = ["EOC", "Immune", "Stromal"]

SNIIC_MODULES = {
    "SNIIC1": ("Atf3", "Gfra3", "Gal"),
    "SNIIC2": ("Atf3", "Mrgprd"),
    "SNIIC3": ("Atf3", "S100b", "Gal"),
}
SNIIC_HEATMAP_GENES = ("Atf3", "Gfra3", "Gal", "Mrgprd", "S100b")
NAV_GENES = ("Scn9a", "Scn10a", "Scn11a")

SOTA_BIOLOGY = {
    "GSE155622": "Neuropathic Pain / DRG Neurons",
    "GSE141259": "Bleomycin Lung Injury (D28 Holdout Extrapolation)",
    "HGSOC": "High-Grade Serous Ovarian Cancer (NACT Paired)",
}
NULL_MODE = {
    "GSE155622": "temporal_matched",
    "GSE141259": "temporal_matched",
    "HGSOC": "pairing_matched",
}


def cache_file(*parts: str) -> Path:
    p = CACHE.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def load_obs(
    ck: Path,
    *,
    usecols: Sequence[str] | None = None,
    index: bool = True,
) -> pd.DataFrame:
    path = Path(ck) / "obs.csv"
    if not path.is_file():
        raise FileNotFoundError(f"missing adopted checkpoint obs: {path}")
    df = pd.read_csv(path, usecols=usecols, low_memory=False)
    if index:
        if "barcode" in df.columns:
            df.index = df["barcode"].astype(str)
        elif "Unnamed: 0" in df.columns:
            df.index = df["Unnamed: 0"].astype(str)
        else:
            df.index = df.index.astype(str)
        df.index = df.index.astype(str)
    return df


def type_col(obs: pd.DataFrame) -> str:
    for c in ("annotation", "celltype", "cell_type"):
        if c in obs.columns:
            return c
    raise KeyError("no cell-type column in obs")


def mean_u0_by_type(
    obs: pd.DataFrame,
    *,
    type_col_name: str | None = None,
    u_col: str = "potential_stationary",
) -> pd.DataFrame:
    """Mean / std stationary potential per type. Rank 1 = deepest (lowest U0)."""
    col = type_col_name or type_col(obs)
    u = pd.to_numeric(obs[u_col], errors="coerce")
    g = obs.assign(_u=u).groupby(obs[col].astype(str), dropna=False)["_u"]
    df = pd.DataFrame(
        {
            "cell_type": g.mean().index,
            "annotation": g.mean().index,
            "potential_stationary_mean": g.mean().values,
            "mean_U0": g.mean().values,
            "potential_stationary_std": g.std(ddof=1).values,
            "n_cells": g.size().values,
            "n": g.size().values,
        }
    )
    if "potential" in obs.columns:
        up = pd.to_numeric(obs["potential"], errors="coerce")
        gp = obs.assign(_up=up).groupby(obs[col].astype(str), dropna=False)["_up"]
        df["potential_mean_U"] = gp.mean().reindex(df["cell_type"]).values
        df["potential_U_mean"] = df["potential_mean_U"]
        df["potential_U_std"] = gp.std(ddof=1).reindex(df["cell_type"]).values
    df = df.sort_values("mean_U0", ascending=True).reset_index(drop=True)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df


def neuron_deviation_timeline(obs: pd.DataFrame) -> pd.DataFrame:
    """Neuron |potential_deviation| mean ± SE by SNI condition."""
    neu = obs[obs["annotation"].astype(str) == "Neuron"].copy()
    if "potential_deviation" not in neu.columns:
        raise KeyError("potential_deviation missing from obs")
    neu["condition"] = neu["condition"].astype(str)
    rows = []
    for cond in PAIN_STAGES:
        sub = neu[neu["condition"] == cond]
        y = pd.to_numeric(sub["potential_deviation"], errors="coerce").to_numpy(float)
        y = y[np.isfinite(y)]
        n = int(y.size)
        se = float(np.std(y, ddof=1) / np.sqrt(max(n, 1))) if n > 1 else 0.0
        rows.append(
            {
                "condition": cond,
                "n": n,
                "time": PAIN_TIME[cond],
                "mean_deviation": float(np.mean(y)) if n else np.nan,
                "se_deviation": se,
                "mean_abs_deviation": float(np.mean(np.abs(y))) if n else np.nan,
            }
        )
    out = pd.DataFrame(rows)
    ctrl = float(out.loc[out.condition == "Control", "mean_deviation"].iloc[0])
    out["mean_dev_vs_control"] = out["mean_deviation"] - ctrl
    return out


def lung_subtype_urel(obs: pd.DataFrame, parents: Iterable[str] = ("macrophages", "alv_epithelium")) -> pd.DataFrame:
    sub = obs[obs["annotation"].astype(str).isin(list(parents))].copy()
    g = sub.groupby(["annotation", "cell.type"], dropna=False)
    df = g.agg(
        mean_U0=("potential_stationary", "mean"),
        mean_Urel=("potential_relative_type", "mean"),
        n=("potential_stationary", "size"),
    ).reset_index()
    return df


def _null_json(ck: Path, dataset: str, mode: str) -> Path:
    tag = f"{dataset}_{mode}_e500_mc5000_bs128"
    return ck / "methods_enhancement" / f"physical_retrain_controls_{tag}_summary.json"


def load_null_summary(*, recompute_real: bool = False) -> pd.DataFrame:
    """Matched-null audit table.

    Null medians come from the checkpoint's recorded retrain JSON (500ep×4).
    """
    rows = []
    for dataset, ck in ADOPTED.items():
        mode = NULL_MODE[dataset]
        jp = _null_json(ck, dataset, mode)
        if not jp.is_file():
            raise FileNotFoundError(
                f"Missing null experiment JSON {jp}. "
                "Run scripts/run_matched_temporal_null.py once, or "
                "python output_file/Supplementary_table3.py --rebuild"
            )
        rec = json.loads(jp.read_text(encoding="utf-8"))
        rec.setdefault("dataset", dataset)
        rec.setdefault("shuffle_mode", mode)
        rec["source_json"] = str(jp.relative_to(ROOT))
        rows.append(rec)
    return pd.DataFrame(rows)


def run_sota_pcc(*, device: str = "cpu", recompute: bool = False) -> pd.DataFrame:
    """Trajectory–time PCC for MomentumNetwork / scVelo / CellRank on all three cohorts.

    Primary LDH metric is **velocity-derived** graph order (same protocol as the
    scVelo proxy), not the supervised ``pseudotime_head``. Cache file is versioned
    so older supervised-head numbers are not reused silently.
    """
    cache = cache_file("sota_pcc_summary_v3_latent_pca.csv")
    if cache.is_file() and not recompute:
        return pd.read_csv(cache)

    methods = {
        "MomentumNetwork": "MomentumNetwork",
        "scVelo_kNN_proxy": "scVelo",
        "CellRank": "CellRank",
    }

    def _from_detail(df: pd.DataFrame, dataset: str) -> dict:
        sub = df[df["dataset"].astype(str) == dataset] if "dataset" in df.columns else df
        rec = {"dataset": dataset}
        for method, col in methods.items():
            hit = sub.loc[sub["method"].astype(str) == method, "trajectory_time_pcc"]
            rec[col] = float(hit.iloc[0]) if len(hit) else np.nan
        # Diagnostic: supervised head (not used for manuscript primary claim)
        head = sub.loc[sub["method"].astype(str) == "MomentumNetwork", "supervised_pseudotime_head_pcc"]
        if len(head) and "supervised_pseudotime_head_pcc" in sub.columns:
            rec["MomentumNetwork_supervised_head"] = float(head.iloc[0])
        mk = sub.loc[sub["method"].astype(str) == "MomentumNetwork", "markov_hitting_pcc_on_momentum"]
        if len(mk) and "markov_hitting_pcc_on_momentum" in sub.columns:
            rec["MomentumNetwork_markov"] = float(mk.iloc[0])
        if "embedding" in sub.columns and (sub["method"].astype(str) == "MomentumNetwork").any():
            rec["embedding"] = str(
                sub.loc[sub["method"].astype(str) == "MomentumNetwork", "embedding"].iloc[0]
            )
        return rec

    from run_sota_velocity_benchmark import run_benchmark

    rows = []
    for dataset, ck in ADOPTED.items():
        # Prefer fresh latent-PCA velocity-derived detail; ignore pre-v3 checkpoint CSVs.
        src = ck / "methods_enhancement" / f"sota_benchmark_{dataset}.csv"
        use_cached_detail = False
        if src.is_file() and not recompute:
            det = pd.read_csv(src)
            emb_ok = (
                "embedding" in det.columns
                and (det["method"].astype(str) == "MomentumNetwork").any()
                and str(det.loc[det["method"].astype(str) == "MomentumNetwork", "embedding"].iloc[0]).startswith(
                    "X_latent_pca"
                )
            )
            proto_ok = (
                "pseudotime_protocol" in det.columns
                and (det["method"].astype(str) == "MomentumNetwork").any()
                and str(
                    det.loc[det["method"].astype(str) == "MomentumNetwork", "pseudotime_protocol"].iloc[0]
                )
                == "velocity_graph_laplacian"
            )
            if emb_ok and proto_ok:
                use_cached_detail = True
                rows.append(_from_detail(det, dataset))
        if use_cached_detail:
            continue
        print(f"[sota] {dataset} @ {ck.name} device={device}", flush=True)
        det = run_benchmark(dataset, ck, device=device)
        det.to_csv(cache_file(f"sota_benchmark_{dataset}_v3.csv"), index=False)
        rows.append(_from_detail(det, dataset))
    out = pd.DataFrame(rows)
    out.to_csv(cache, index=False)
    # Keep legacy filename as a pointer copy for older scripts.
    out.to_csv(cache_file("sota_pcc_summary.csv"), index=False)
    return out


def load_pain_neuron_expression(genes: Sequence[str]) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """log1p expression of ``genes`` on Neuron cells, aligned to checkpoint obs."""
    import anndata as ad
    from scipy import sparse

    from dataset_pipeline import GSE155622, resolve_data_path

    obs = load_obs(CK_PAIN)
    neu = obs[obs["annotation"].astype(str) == "Neuron"].copy()
    raw = ad.read_h5ad(resolve_data_path(GSE155622), backed="r")
    common = neu.index.intersection(raw.obs_names.astype(str))
    lower = {str(g).lower(): str(g) for g in raw.var_names}
    use = []
    for g in genes:
        if g in raw.var_names:
            use.append(str(g))
        elif g.lower() in lower:
            use.append(lower[g.lower()])
        else:
            raise KeyError(f"gene not in GSE155622 AnnData: {g}")
    parts = [raw[common, g].to_memory() for g in use]
    raw.file.close()
    X = np.hstack(
        [
            (p.X.toarray() if sparse.issparse(p.X) else np.asarray(p.X)).reshape(-1, 1)
            for p in parts
        ]
    )
    X = np.log1p(X.astype(float))
    neu = neu.loc[common]
    return X, neu, use


def module_score_matrix(X: np.ndarray, gene_names: Sequence[str], modules: dict[str, Sequence[str]]) -> pd.DataFrame:
    lower = {str(g).lower(): i for i, g in enumerate(gene_names)}
    out = {}
    for name, genes in modules.items():
        idx = []
        for g in genes:
            if g.lower() in lower:
                idx.append(lower[g.lower()])
        if not idx:
            out[name] = np.full(X.shape[0], np.nan)
        else:
            out[name] = np.nanmean(X[:, idx], axis=1)
    return pd.DataFrame(out)


def hgsoc_deep_valley_deg(*, recompute: bool = False) -> pd.DataFrame:
    proto = CK_HG / "analysis_protocol_HGSOC" / "eoc_attractor_deep_valley_DEG.csv"
    cache = cache_file("hgsoc_deep_valley_DEG.csv")
    if not recompute:
        if proto.is_file():
            return pd.read_csv(proto)
        if cache.is_file():
            return pd.read_csv(cache)
    from celltype_analysis import HGSOC_PROFILE, load_annotated_adata
    from run_hgsoc_nact_analysis import step_eoc_attractor_basin

    out_dir = cache_file("hgsoc_protocol").parent / "hgsoc_protocol"
    out_dir.mkdir(parents=True, exist_ok=True)
    print("[hgsoc] loading annotated adata + computing deep-valley DEG", flush=True)
    adata = load_annotated_adata(HGSOC_PROFILE, str(CK_HG))
    step_eoc_attractor_basin(adata, CK_HG, out_dir)
    src = out_dir / "eoc_attractor_deep_valley_DEG.csv"
    deg = pd.read_csv(src)
    deg.to_csv(cache, index=False)
    return deg


def hgsoc_ccc(*, recompute: bool = False) -> dict[str, pd.DataFrame]:
    proto = CK_HG / "analysis_protocol_HGSOC"
    proto_files = {
        "Stromal_to_EOC": proto / "ccc_LR_Stromal_to_EOC.csv",
        "EOC_to_Stromal": proto / "ccc_LR_EOC_to_Stromal.csv",
        "paracrine_feedforward": proto / "ccc_LR_paracrine_feedforward.csv",
    }
    cache_dir = cache_file("hgsoc_ccc", "marker.txt").parent
    files = {
        "Stromal_to_EOC": cache_dir / "ccc_LR_Stromal_to_EOC.csv",
        "EOC_to_Stromal": cache_dir / "ccc_LR_EOC_to_Stromal.csv",
        "paracrine_feedforward": cache_dir / "ccc_LR_paracrine_feedforward.csv",
    }
    if not recompute and all(p.is_file() for p in proto_files.values()):
        return {k: pd.read_csv(v) for k, v in proto_files.items()}
    if all(p.is_file() for p in files.values()) and not recompute:
        return {k: pd.read_csv(v) for k, v in files.items()}
    from celltype_analysis import HGSOC_PROFILE, load_annotated_adata
    from run_hgsoc_nact_analysis import step_eoc_attractor_basin, step_targeted_ccc

    print("[hgsoc] computing targeted CCC", flush=True)
    adata = load_annotated_adata(HGSOC_PROFILE, str(CK_HG))
    eoc = step_eoc_attractor_basin(adata, CK_HG, cache_dir)
    step_targeted_ccc(adata, eoc, cache_dir)
    out = {}
    for k, name in (
        ("Stromal_to_EOC", "ccc_LR_Stromal_to_EOC.csv"),
        ("EOC_to_Stromal", "ccc_LR_EOC_to_Stromal.csv"),
        ("paracrine_feedforward", "ccc_LR_paracrine_feedforward.csv"),
    ):
        src = cache_dir / name
        if src.is_file():
            df = pd.read_csv(src)
            df.to_csv(files[k], index=False)
            out[k] = df
        else:
            out[k] = pd.DataFrame()
    return out


def pain_atf3_ko_track(*, recompute: bool = False) -> pd.DataFrame:
    cache = cache_file("in_silico_KO_Atf3_hybrid_shift1_SNIIC_track.csv")
    if cache.is_file() and not recompute:
        return pd.read_csv(cache)
    from run_in_silico_knockout import run_knockout_gse155622

    out = cache_file("ko_atf3").parent / "ko_atf3"
    out.mkdir(parents=True, exist_ok=True)
    print("[ko] Atf3 hybrid KO on GSE155622", flush=True)
    run_knockout_gse155622(CK_PAIN, ["Atf3"], out, ko_mode="hybrid", latent_shift_scale=1.0)
    src = out / "in_silico_KO_Atf3_hybrid_shift1_SNIIC_track.csv"
    if not src.is_file():
        # knockout writes via result_path; search
        hits = list(out.rglob("*Atf3*SNIIC_track.csv"))
        if not hits:
            raise FileNotFoundError(f"Atf3 KO track not written under {out}")
        src = hits[0]
    df = pd.read_csv(src)
    df.to_csv(cache, index=False)
    return df
