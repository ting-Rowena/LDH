#!/usr/bin/env python3
"""
Unsupervised trajectory & intermediate-state discovery (no literature priors).

Logic (within each parent compartment that has multiple subtypes):
  1) Auto endpoints: subtype with max abundance at t_min → start;
     subtype with max abundance at t_max → end.
  2) Bidirectional LAP + force decomposition → differentiation vs reversible plasticity.
  3) Remaining subtypes scored as intermediate candidates
     (saddle proximity, temporal inverted-U, bidirectional DEG, plasticity peak).

Also runs metacelltype-level disease-remodeling screening (pre-LAP).

Outputs under --out-dir:
  01_celltype_screening_ranking.csv
  02_parent_trajectory_classification.csv
  03_intermediate_state_scores.csv
  trajectory_discovery_report.md
"""

from __future__ import annotations

import argparse
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from anndata import AnnData
from sklearn.decomposition import PCA

_ROOT = Path(__file__).resolve().parent
import sys

if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from CellFateLandscape import NonEquilibriumCellFateLandscape  # noqa: E402
from celltype_analysis import DATASET_REGISTRY, load_annotated_adata  # noqa: E402
from celltype_screening import screen_disease_remodeling_celltypes  # noqa: E402
from dataset_pipeline import recommended_checkpoint_dir  # noqa: E402
from run_gse141259_analysis import estimate_embedding_velocity  # noqa: E402


def _stage_order(obs: pd.DataFrame, stage_key: str) -> list[str]:
    stages = obs[stage_key].astype(str)
    # Prefer numeric time metadata when available and finite.
    if "time" in obs.columns:
        t = pd.to_numeric(obs["time"], errors="coerce")
        if np.isfinite(t).any():
            med = (
                obs.assign(_t=t)
                .groupby(stage_key, observed=False)["_t"]
                .median()
                .dropna()
                .sort_values()
            )
            if len(med):
                return med.index.astype(str).tolist()
    # Parse day-like labels: D0, D10, day28, …
    import re

    def _day_key(s: str) -> float:
        m = re.search(r"(\d+(?:\.\d+)?)", str(s))
        return float(m.group(1)) if m else float("inf")

    uniq = stages.unique().tolist()
    return sorted(uniq, key=_day_key)


def _abundance_table(obs: pd.DataFrame, parent_key: str, subtype_key: str, stage_key: str) -> pd.DataFrame:
    g = (
        obs.groupby([parent_key, stage_key, subtype_key], observed=False)
        .size()
        .rename("n")
        .reset_index()
    )
    tot = g.groupby([parent_key, stage_key], observed=False)["n"].transform("sum")
    g["frac"] = g["n"] / tot.replace(0, np.nan)
    return g


def _pick_start_end(
    ab: pd.DataFrame,
    parent: str,
    parent_key: str,
    subtype_key: str,
    stage_key: str,
    start_s: str,
    end_s: str,
    min_n: int = 20,
) -> tuple[str, str] | None:
    a0 = ab[(ab[parent_key] == parent) & (ab[stage_key] == start_s) & (ab["n"] >= min_n)]
    a1 = ab[(ab[parent_key] == parent) & (ab[stage_key] == end_s) & (ab["n"] >= min_n)]
    if a0.empty or a1.empty:
        # relax min_n
        a0 = ab[(ab[parent_key] == parent) & (ab[stage_key] == start_s)]
        a1 = ab[(ab[parent_key] == parent) & (ab[stage_key] == end_s)]
    if a0.empty or a1.empty:
        return None
    start = str(a0.sort_values("frac", ascending=False).iloc[0][subtype_key])
    end = str(a1.sort_values("frac", ascending=False).iloc[0][subtype_key])
    if start == end:
        # choose second-best end if possible
        if len(a1) >= 2:
            end = str(a1.sort_values("frac", ascending=False).iloc[1][subtype_key])
        else:
            return None
    return start, end


def _core_centroid(coords: np.ndarray, pot: np.ndarray, mask: np.ndarray, k: int = 40) -> np.ndarray | None:
    ix = np.where(mask)[0]
    if len(ix) < 5:
        return None
    order = np.argsort(pot[ix])[: min(k, len(ix))]
    return np.median(coords[ix[order]], axis=0)


def _temporal_inverted_u(
    frac_by_stage: pd.Series,
    stages: list[str],
) -> tuple[bool, float, str]:
    """Return (is_transient, score, peak_stage)."""
    vals = np.array([float(frac_by_stage.get(s, 0.0)) for s in stages], dtype=float)
    if len(vals) < 3:
        return False, 0.0, stages[0]
    i_peak = int(np.argmax(vals))
    peak = float(vals[i_peak])
    edge = float(vals[0] + vals[-1]) / 2.0
    mid_ok = 0 < i_peak < len(vals) - 1
    # inverted-U: mid peak and peak >> edge mean
    score = peak / (edge + 1e-8)
    is_transient = bool(mid_ok and peak >= 0.15 and score >= 1.5)
    return is_transient, float(score), stages[i_peak]


def _bidir_deg_counts(
    adata: AnnData,
    subtype_key: str,
    label_c: str,
    label_a: str,
    label_b: str,
    *,
    min_cells: int = 30,
    padj: float = 0.05,
    logfc: float = 0.25,
) -> dict:
    import scanpy as sc

    out = {
        "deg_C_vs_start": np.nan,
        "deg_C_vs_end": np.nan,
        "bidirectional_deg": False,
        "deg_status": "skipped",
    }
    labels = adata.obs[subtype_key].astype(str)

    def _count(g1: str, g2: str) -> int | float:
        m = labels.isin([g1, g2])
        if int((labels == g1).sum()) < min_cells or int((labels == g2).sum()) < min_cells:
            return np.nan
        sub = adata[m].copy()
        try:
            sc.tl.rank_genes_groups(
                sub,
                groupby=subtype_key,
                groups=[g1],
                reference=g2,
                method="wilcoxon",
                use_raw=False,
            )
            res = sub.uns["rank_genes_groups"]
            names = res["names"][g1]
            lfc = res["logfoldchanges"][g1]
            pa = res["pvals_adj"][g1]
            n = 0
            for _, lf, p in zip(names, lfc, pa):
                if p is None or lf is None:
                    continue
                if np.isfinite(p) and np.isfinite(lf) and float(p) < padj and abs(float(lf)) > logfc:
                    n += 1
            return int(n)
        except Exception:
            return np.nan

    d_cs = _count(label_c, label_a)
    d_ce = _count(label_c, label_b)
    out["deg_C_vs_start"] = d_cs
    out["deg_C_vs_end"] = d_ce
    if np.isfinite(d_cs) and np.isfinite(d_ce):
        out["bidirectional_deg"] = bool(d_cs >= 10 and d_ce >= 10)
        out["deg_status"] = "ok"
    else:
        out["deg_status"] = "underpowered_or_failed"
    return out


def _classify_process(action_asym: float, flux_frac: float) -> str:
    """flux_frac = ||F_flux|| / (||F_grad|| + ||F_flux||) in [0,1]."""
    if np.isfinite(action_asym) and np.isfinite(flux_frac):
        if action_asym >= 1.7 and flux_frac < 0.55:
            return "Directional Differentiation"
        if action_asym < 1.4 or flux_frac >= 0.55:
            return "Reversible Plasticity"
    return "Mixed / Indeterminate"


def analyze_parent(
    adata: AnnData,
    *,
    parent: str,
    parent_key: str,
    subtype_key: str,
    stage_key: str,
    stages: list[str],
    ab: pd.DataFrame,
    do_deg: bool,
) -> tuple[dict | None, pd.DataFrame]:
    sub = adata[adata.obs[parent_key].astype(str) == parent].copy()
    if sub.n_obs < 50:
        return None, pd.DataFrame()
    types = sub.obs[subtype_key].astype(str).value_counts()
    types = types[types >= 15].index.tolist()
    if len(types) < 2:
        return None, pd.DataFrame()

    start_s, end_s = stages[0], stages[-1]
    ends = _pick_start_end(ab, parent, parent_key, subtype_key, stage_key, start_s, end_s, min_n=15)
    if ends is None:
        return None, pd.DataFrame()
    start_ct, end_ct = ends

    # parent-internal PCA2
    if "X_latent" in sub.obsm:
        Z = np.asarray(sub.obsm["X_latent"], float)
    else:
        Z = np.asarray(sub.obsm["X_latent_pca"], float)
        if Z.shape[1] > 2:
            pass
    if Z.shape[1] > 2:
        coords = PCA(n_components=2, random_state=0, svd_solver="randomized").fit_transform(Z)
    else:
        coords = Z[:, :2]
    sub.obsm["_lap_compute_slice"] = coords

    pot = pd.to_numeric(sub.obs["potential"], errors="coerce").to_numpy(float)
    pt = pd.to_numeric(sub.obs["pseudotime"], errors="coerce").to_numpy(float)
    labels = sub.obs[subtype_key].astype(str).to_numpy()
    stage = sub.obs[stage_key].astype(str).to_numpy()

    # velocity for flux decomposition
    vel = estimate_embedding_velocity(coords, pot, pt, n_neighbors=min(30, max(5, sub.n_obs // 20)))
    sub.obsm["_lap_compute_slice_velocity"] = vel

    # subsample large parents for field fit
    rng = np.random.default_rng(0)
    n = sub.n_obs
    take = rng.choice(n, 1500, replace=False) if n > 1500 else np.arange(n)
    ad = AnnData(X=np.zeros((len(take), 1)))
    ad.obs["potential"] = pot[take]
    ad.obsm["_lap_compute_slice"] = coords[take]
    ad.obsm["_lap_compute_slice_velocity"] = vel[take]

    analyzer = NonEquilibriumCellFateLandscape(
        ad,
        potential_key="potential",
        embedding_2d_key="_lap_compute_slice",
        embedding_velocity_key="_lap_compute_slice_velocity",
        potential_transform="none",
        use_embedding_velocity=True,
        lap_force_mode="total",
    )

    m_start = labels == start_ct
    m_end = labels == end_ct
    # prefer early cells of start and late cells of end when available
    m_start_early = m_start & (stage == start_s)
    m_end_late = m_end & (stage == end_s)
    if m_start_early.sum() >= 10:
        m_start_use = m_start_early
    else:
        m_start_use = m_start
    if m_end_late.sum() >= 10:
        m_end_use = m_end_late
    else:
        m_end_use = m_end

    pos_a = _core_centroid(coords, pot, m_start_use)
    pos_b = _core_centroid(coords, pot, m_end_use)
    if pos_a is None or pos_b is None:
        return None, pd.DataFrame()

    fwd = analyzer.compute_least_action_path(pos_a, pos_b, n_points=20)
    bwd = analyzer.compute_least_action_path(pos_b, pos_a, n_points=20)
    s_fwd = float(fwd["total_action"])
    s_bwd = float(bwd["total_action"])
    # magnitude asymmetry (Hamiltonian actions can be signed)
    action_asym = float(abs(s_bwd) / (abs(s_fwd) + 1e-12))

    path = np.asarray(fwd["path"], float)
    tot_f, grad_f, flux_f = analyzer.compute_force_field(path, return_components=True)
    ng = np.linalg.norm(grad_f, axis=1)
    nf = np.linalg.norm(flux_f, axis=1)
    flux_frac = float(np.nanmean(nf) / (np.nanmean(ng) + np.nanmean(nf) + 1e-12))
    process = _classify_process(action_asym, flux_frac)

    ts_idx = int(fwd.get("transition_state_idx", len(path) // 2) or len(path) // 2)
    ts_idx = int(np.clip(ts_idx, 0, len(path) - 1))
    saddle = path[ts_idx]
    U_path = np.array([float(analyzer.U_func_2d(p)) for p in path])
    barrier = float(np.nanmax(U_path) - U_path[0])

    traj = {
        "parent": parent,
        "start_subtype": start_ct,
        "end_subtype": end_ct,
        "start_stage": start_s,
        "end_stage": end_s,
        "n_cells_parent": int(sub.n_obs),
        "S_forward": s_fwd,
        "S_backward": s_bwd,
        "action_asymmetry_ratio": action_asym,
        "non_conservative_flux_fraction": flux_frac,
        "barrier_height": barrier,
        "process_type": process,
        "transition_state_idx": ts_idx,
        "fwd_success": bool(fwd.get("success", False)),
        "bwd_success": bool(bwd.get("success", False)),
    }

    # abundance fractions for temporal tests
    ab_p = ab[ab[parent_key] == parent]
    plast = (
        pd.to_numeric(sub.obs["plasticity_score"], errors="coerce")
        if "plasticity_score" in sub.obs
        else pd.Series(np.nan, index=sub.obs_names)
    )

    inter_rows = []
    # saddle distance scale: median pairwise subtype centroid distance
    cents = {}
    for t in types:
        mm = labels == t
        if mm.sum() >= 5:
            cents[t] = np.median(coords[mm], axis=0)
    if len(cents) >= 2:
        keys = list(cents.keys())
        dists = [
            np.linalg.norm(cents[keys[i]] - cents[keys[j]])
            for i in range(len(keys))
            for j in range(i + 1, len(keys))
        ]
        scale = float(np.median(dists)) + 1e-8
    else:
        scale = 1.0
    saddle_thresh = 0.35 * scale  # relative threshold

    plast_by_type = {
        t: float(np.nanmean(plast.to_numpy()[labels == t])) if (labels == t).any() else np.nan
        for t in types
    }
    plast_peak = max(plast_by_type.values()) if plast_by_type else np.nan

    for t in types:
        if t in (start_ct, end_ct):
            role = "endpoint"
        else:
            role = "candidate"
        cent = cents.get(t)
        if cent is None:
            continue
        d_saddle = float(np.linalg.norm(cent - saddle))
        near_saddle = d_saddle <= saddle_thresh

        # temporal frac within parent
        fr = {}
        for s in stages:
            row = ab_p[(ab_p[stage_key] == s) & (ab_p[subtype_key] == t)]
            fr[s] = float(row["frac"].iloc[0]) if len(row) else 0.0
        is_transient, tscore, peak_s = _temporal_inverted_u(pd.Series(fr), stages)

        # plasticity peak among candidates
        pmean = plast_by_type.get(t, np.nan)
        plast_is_peak = bool(
            np.isfinite(pmean) and np.isfinite(plast_peak) and pmean >= 0.95 * plast_peak and role == "candidate"
        )

        deg_info = {
            "deg_C_vs_start": np.nan,
            "deg_C_vs_end": np.nan,
            "bidirectional_deg": False,
            "deg_status": "not_run",
        }
        if do_deg and role == "candidate" and (near_saddle or is_transient):
            deg_info = _bidir_deg_counts(sub, subtype_key, t, start_ct, end_ct)

        # intermediate call only if parent process is differentiation-like
        is_inter = False
        if process == "Directional Differentiation" and role == "candidate":
            geo_time = bool(near_saddle and is_transient)
            if deg_info["deg_status"] == "ok":
                is_inter = bool(geo_time and deg_info["bidirectional_deg"])
            else:
                # geometry + time only when DEG underpowered / skipped
                is_inter = geo_time

        # For plasticity parents, mark "window state" not differentiation intermediate
        window_state = bool(process == "Reversible Plasticity" and is_transient and role == "candidate")

        inter_rows.append(
            {
                "parent": parent,
                "subtype": t,
                "role": role,
                "process_type_parent": process,
                "start_subtype": start_ct,
                "end_subtype": end_ct,
                "distance_to_saddle": d_saddle,
                "saddle_distance_threshold": saddle_thresh,
                "near_saddle": near_saddle,
                "is_transient_peak": is_transient,
                "transient_score": tscore,
                "peak_stage": peak_s,
                "plasticity_mean": pmean,
                "plasticity_is_local_peak": plast_is_peak,
                "deg_C_vs_start": deg_info["deg_C_vs_start"],
                "deg_C_vs_end": deg_info["deg_C_vs_end"],
                "bidirectional_deg": deg_info["bidirectional_deg"],
                "deg_status": deg_info["deg_status"],
                "is_differentiation_intermediate": is_inter,
                "is_injury_window_state": window_state,
            }
        )

    return traj, pd.DataFrame(inter_rows)


def run_unsupervised_discovery(
    dataset_name: str,
    checkpoint_dir: str,
    parent_key: str,
    subtype_key: str,
    stage_key: str = "stage",
    out_dir: str = "./unsupervised_analysis_results",
    do_deg: bool = True,
    parents: list[str] | None = None,
) -> Path:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    profile = DATASET_REGISTRY[dataset_name]
    print(f"Loading {dataset_name} …", flush=True)
    adata = load_annotated_adata(profile, checkpoint_dir)
    adata.obs[stage_key] = adata.obs[stage_key].astype(str)
    adata.obs[parent_key] = adata.obs[parent_key].astype(str)
    adata.obs[subtype_key] = adata.obs[subtype_key].astype(str)

    # attach latent
    ck = Path(checkpoint_dir)
    lat = np.load(ck / "latent_embeddings.npz", allow_pickle=True)
    idx = pd.Index(np.asarray(lat["index"]).astype(str))
    Z = np.asarray(lat["X_latent"], float)
    mapper = {b: i for i, b in enumerate(idx.astype(str))}
    keep = []
    rows = []
    for i, b in enumerate(adata.obs_names.astype(str)):
        j = mapper.get(b)
        if j is not None:
            keep.append(i)
            rows.append(Z[j])
    adata = adata[np.asarray(keep)].copy()
    adata.obsm["X_latent"] = np.vstack(rows)
    adata.obsm["X_latent_pca"] = PCA(n_components=2, random_state=0, svd_solver="randomized").fit_transform(
        adata.obsm["X_latent"]
    )

    stages = _stage_order(adata.obs, stage_key)
    print(f"stages: {stages}", flush=True)

    print("=== [1/4] Metacelltype remodeling screen ===", flush=True)
    screen_key = parent_key if parent_key in adata.obs else profile.cell_type_column
    screen_df = screen_disease_remodeling_celltypes(
        adata,
        cell_type_key=screen_key,
        stage_key=stage_key,
        start_state=stages[0],
        end_state=stages[-1],
        potential_key="potential",
        pseudotime_key="pseudotime",
        min_cells=50,
        top_k=5,
    )
    screen_df.to_csv(out_path / "01_celltype_screening_ranking.csv", index=False)

    ab = _abundance_table(adata.obs, parent_key, subtype_key, stage_key)
    ab.to_csv(out_path / "00_subtype_stage_abundance.csv", index=False)

    parent_list = parents or sorted(adata.obs[parent_key].astype(str).unique())
    traj_rows = []
    inter_all = []

    print("=== [2–3/4] Parent-wise unsupervised trajectories ===", flush=True)
    for parent in parent_list:
        n_sub = adata.obs.loc[adata.obs[parent_key].astype(str) == parent, subtype_key].nunique()
        if n_sub < 2:
            continue
        print(f"  parent={parent} (n_subtypes={n_sub}) …", flush=True)
        try:
            traj, inter = analyze_parent(
                adata,
                parent=parent,
                parent_key=parent_key,
                subtype_key=subtype_key,
                stage_key=stage_key,
                stages=stages,
                ab=ab,
                do_deg=do_deg,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"    FAILED: {exc}", flush=True)
            continue
        if traj is not None:
            traj_rows.append(traj)
            print(
                f"    {traj['start_subtype']} → {traj['end_subtype']}: "
                f"{traj['process_type']}  R_action={traj['action_asymmetry_ratio']:.3f}  "
                f"R_flux={traj['non_conservative_flux_fraction']:.3f}",
                flush=True,
            )
        if inter is not None and len(inter):
            inter_all.append(inter)

    traj_df = pd.DataFrame(traj_rows)
    inter_df = pd.concat(inter_all, ignore_index=True) if inter_all else pd.DataFrame()
    traj_df.to_csv(out_path / "02_parent_trajectory_classification.csv", index=False)
    inter_df.to_csv(out_path / "03_intermediate_state_scores.csv", index=False)
    if dataset_name == "GSE141259" and len(traj_df):
        audit = _ROOT / "output_file" / "mac_landscape_audit"
        audit.mkdir(parents=True, exist_ok=True)
        traj_df.to_csv(audit / "02_parent_trajectory_classification.csv", index=False)

    # report
    lines = [
        "# Unsupervised trajectory discovery report",
        f"- Dataset: {dataset_name}",
        f"- Checkpoint: {checkpoint_dir}",
        f"- Parent key: `{parent_key}`; subtype key: `{subtype_key}`",
        f"- Auto stages: `{stages[0]}` → `{stages[-1]}`",
        "",
        "## Process classification (no literature priors)",
        "",
        "Criteria:",
        "- **Directional Differentiation**: `|S_bwd|/|S_fwd| ≥ 1.7` AND flux fraction `< 0.55`",
        "- **Reversible Plasticity**: `|S_bwd|/|S_fwd| < 1.4` OR flux fraction `≥ 0.55`",
        "- Else: Mixed / Indeterminate",
        "",
    ]
    if len(traj_df):
        lines.append(traj_df.to_string(index=False))
    else:
        lines.append("(no parent trajectories)")
    lines += [
        "",
        "## Intermediate / window calls",
        "",
        "Differentiation intermediate: parent=Directional Differentiation + near saddle + temporal inverted-U "
        "(+ bidirectional DEG when powered).",
        "Injury window state: parent=Reversible Plasticity + temporal inverted-U (not claimed as lineage intermediate).",
        "",
    ]
    if len(inter_df):
        show = inter_df[
            inter_df["is_differentiation_intermediate"]
            | inter_df["is_injury_window_state"]
            | (inter_df["role"] == "endpoint")
        ]
        lines.append(show.to_string(index=False))
    (out_path / "trajectory_discovery_report.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote report → {out_path / 'trajectory_discovery_report.md'}", flush=True)
    return out_path


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", default="GSE141259")
    p.add_argument("--checkpoint-dir", default=None)
    p.add_argument("--parent-key", default="annotation", help="Parent compartment column")
    p.add_argument("--subtype-key", default="cell.type", help="Subtype column for trajectories")
    p.add_argument("--stage-key", default="stage")
    p.add_argument(
        "--out-dir",
        default=None,
        help="Output directory (default: <checkpoint>/analysis_protocol_<dataset>/unsupervised_trajectory)",
    )
    p.add_argument("--no-deg", action="store_true")
    p.add_argument(
        "--parents",
        default="alv_epithelium,macrophages",
        help="Comma-separated parents to analyze (default: Alv+Mac). Use 'all' for every parent.",
    )
    args = p.parse_args(argv)
    ckpt = args.checkpoint_dir or str(recommended_checkpoint_dir(args.dataset))
    out_dir = args.out_dir or str(
        Path(ckpt) / f"analysis_protocol_{args.dataset}" / "unsupervised_trajectory"
    )
    parents = None if args.parents.strip().lower() == "all" else [x.strip() for x in args.parents.split(",") if x.strip()]
    run_unsupervised_discovery(
        args.dataset,
        ckpt,
        parent_key=args.parent_key,
        subtype_key=args.subtype_key,
        stage_key=args.stage_key,
        out_dir=out_dir,
        do_deg=not args.no_deg,
        parents=parents,
    )


if __name__ == "__main__":
    main()
