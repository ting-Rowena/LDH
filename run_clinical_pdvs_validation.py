#!/usr/bin/env python
"""Patient-level PDVS overall-survival analysis on TCGA-OV and GSE26712 (AOCS proxy).

PDVS5 (HVG-restricted): BBC3, SOD2, WFDC2, FTL, CEBPD.
Primary endpoint is OS only. PFS is not reported.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from typing import List, Optional, Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from methods_enhancement_utils import cache_path, fig_path, methods_outdir, result_path, write_output_file_index
from plot_utils import PALETTE, configure_headless, style_axis

configure_headless()

PDVS5 = ["BBC3", "SOD2", "WFDC2", "FTL", "CEBPD"]
RISK_TIMES = (12.0, 24.0, 36.0, 60.0)
PDVS_ENSEMBL = {
    "ENSG00000105327": "BBC3",
    "ENSG00000112096": "SOD2",
    "ENSG00000163874": "ZC3H12A",
    "ENSG00000101443": "WFDC2",
    "ENSG00000087086": "FTL",
    "ENSG00000221866": "CEBPD",
    "ENSG00000125347": "IRF1",
    "ENSG00000165949": "IFI27",
}


def _load_pdvs_genes(protocol_dir: Path, genes: Optional[Sequence[str]] = None) -> List[str]:
    if genes:
        return [str(g).upper() for g in genes]
    var_path = protocol_dir.parent / "training_var_names.json"
    if var_path.is_file():
        try:
            panel = {str(g).upper() for g in json.loads(var_path.read_text())}
            present = [g for g in PDVS5 if g in panel]
            if len(present) >= 3:
                return PDVS5
        except Exception:
            pass
    return list(PDVS5)


def _download_file(url: str, dest: Path) -> bool:
    import urllib.request

    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.is_file() and dest.stat().st_size > 1000:
        return True
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=180) as resp, open(dest, "wb") as out:
            out.write(resp.read())
        return dest.is_file() and dest.stat().st_size > 1000
    except Exception as exc:
        warnings.warn(f"Download failed {url}: {exc}", UserWarning)
        return False


def _ensembl_to_symbol(expr: pd.DataFrame) -> pd.DataFrame:
    ens_ids = expr.index.astype(str).str.split(".").str[0].str.upper().unique().tolist()
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        mapping = mg.querymany(
            ens_ids,
            scopes="ensembl.gene",
            fields="symbol",
            species="human",
            as_dataframe=True,
            verbose=False,
        )
        sym = mapping["symbol"].dropna().astype(str).str.upper()
        sym_map = sym.groupby(sym.index).first().to_dict()
    except Exception as exc:
        warnings.warn(f"Ensembl→symbol mapping failed: {exc}", UserWarning)
        sym_map = {}
    for ens, symbol in PDVS_ENSEMBL.items():
        sym_map.setdefault(ens, symbol)
    rows = []
    for ens, row in expr.iterrows():
        key = str(ens).split(".")[0].upper()
        rows.append((str(sym_map.get(key, key)).upper(), row))
    if not rows:
        return expr
    out = pd.DataFrame({s: r for s, r in rows}).T
    return out.groupby(out.index).mean()


def _maybe_symbol_index(expr: pd.DataFrame) -> pd.DataFrame:
    idx = expr.index.astype(str)
    if idx.str.startswith("ENSG").mean() > 0.5:
        return _ensembl_to_symbol(expr)
    expr = expr.copy()
    expr.index = idx.str.upper()
    return expr.groupby(expr.index).mean()


def _download_tcga_ov_xena(cache_dir: Path) -> Optional[pd.DataFrame]:
    import gzip

    cache_dir.mkdir(parents=True, exist_ok=True)
    expr_cache = cache_dir / "TCGA-OV.star_tpm.tsv.gz"
    sym_cache = cache_dir / "TCGA-OV.symbol_tpm.parquet"
    csv_cache = cache_dir / "TCGA-OV.symbol_tpm.csv"
    if sym_cache.is_file():
        expr = pd.read_parquet(sym_cache)
        return _maybe_symbol_index(expr)
    if csv_cache.is_file() and csv_cache.stat().st_size > 1000:
        expr = pd.read_csv(csv_cache, index_col=0)
        return _maybe_symbol_index(expr)
    url = "https://gdc-hub.s3.us-east-1.amazonaws.com/download/TCGA-OV.star_tpm.tsv.gz"
    if not _download_file(url, expr_cache):
        return None
    with gzip.open(expr_cache, "rb") as fh:
        expr = pd.read_csv(fh, sep="\t", index_col=0)
    expr.index = expr.index.astype(str).str.split(".").str[0].str.upper()
    expr = expr.groupby(expr.index).mean()
    expr = _ensembl_to_symbol(expr)
    try:
        expr.to_parquet(sym_cache)
    except Exception:
        expr.to_csv(csv_cache)
    return expr


def _gdc_cases(fields: str) -> list[dict]:
    import urllib.request

    hits_all: list[dict] = []
    offset = 0
    page = 500
    while True:
        body = {
            "filters": {
                "op": "and",
                "content": [{"op": "=", "content": {"field": "project.project_id", "value": "TCGA-OV"}}],
            },
            "fields": fields,
            "size": page,
            "from": offset,
        }
        req = urllib.request.Request(
            "https://api.gdc.cancer.gov/cases",
            data=json.dumps(body).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=180) as resp:
            payload = json.load(resp)
        hits = payload.get("data", {}).get("hits", [])
        if not hits:
            break
        hits_all.extend(hits)
        total = payload.get("data", {}).get("pagination", {}).get("total", 0)
        offset += page
        if offset >= total:
            break
    return hits_all


def _stage_group(raw: object) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return None
    s = str(raw).upper().replace("FIGO", " ").replace("STAGE", " ")
    s = " ".join(s.split())
    if not s or s in {"NOT REPORTED", "UNKNOWN", "NAN", "NONE"}:
        return None
    if "IV" in s:
        return "IV"
    if "III" in s:
        return "III"
    if "II" in s:
        return "II"
    if "I" in s.split() or s.startswith("I"):
        return "I"
    return None


def _residual_group(raw: object) -> Optional[str]:
    if raw is None or (isinstance(raw, float) and not np.isfinite(raw)):
        return None
    s = str(raw).lower()
    if not s or s in {"not reported", "unknown", "nan", "none"}:
        return None
    if "no macroscopic" in s or s in {"r0", "complete", "optimal"}:
        return "no_macroscopic"
    if "macroscopic" in s or "residual" in s or "suboptimal" in s:
        return "residual"
    return s.replace(" ", "_")[:40]


def _download_tcga_clinical(cache_dir: Path, *, force: bool = False) -> Optional[pd.DataFrame]:
    """One row per TCGA-OV patient: OS plus available covariates. No PFS."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    clin_cache = cache_dir / "TCGA_OV_clinical_patient.csv"
    if clin_cache.is_file() and not force:
        return pd.read_csv(clin_cache)

    fields = (
        "submitter_id,samples.submitter_id,samples.sample_type,"
        "demographic.vital_status,demographic.days_to_death,demographic.age_at_index,"
        "diagnoses.days_to_last_follow_up,diagnoses.age_at_diagnosis,"
        "diagnoses.figo_stage,diagnoses.ajcc_pathologic_stage,diagnoses.tumor_stage,"
        "diagnoses.residual_disease"
    )
    try:
        hits = _gdc_cases(fields)
    except Exception as exc:
        warnings.warn(f"GDC clinical download failed: {exc}", UserWarning)
        return None
    rows = []
    for case in hits:
        patient = str(case.get("submitter_id") or "")
        if not patient:
            continue
        demo = case.get("demographic") or {}
        vital = str(demo.get("vital_status") or "").lower()
        days_death = demo.get("days_to_death")
        diagnoses = case.get("diagnoses") or [{}]
        diag = diagnoses[0] if diagnoses else {}
        days_fu = diag.get("days_to_last_follow_up")
        if vital == "dead" and days_death is not None:
            os_time, os_event = float(days_death) / 30.44, 1
        elif days_fu is not None:
            os_time, os_event = float(days_fu) / 30.44, 0
        else:
            continue
        age_days = diag.get("age_at_diagnosis")
        age_years = float(age_days) / 365.25 if age_days is not None else (
            float(demo["age_at_index"]) if demo.get("age_at_index") is not None else np.nan
        )
        stage_raw = diag.get("figo_stage") or diag.get("ajcc_pathologic_stage") or diag.get("tumor_stage")
        rows.append(
            {
                "patient": patient,
                "os_months": os_time,
                "os_event": int(os_event),
                "age_years": age_years,
                "stage_raw": stage_raw,
                "stage": _stage_group(stage_raw),
                "residual_raw": diag.get("residual_disease"),
                "residual": _residual_group(diag.get("residual_disease")),
            }
        )
    if not rows:
        return None
    clin = pd.DataFrame(rows).drop_duplicates(subset=["patient"], keep="first")
    clin.to_csv(clin_cache, index=False)
    return clin


def _download_aocs_geo(cache_dir: Path) -> Optional[pd.DataFrame]:
    import gzip
    import io

    cache_dir.mkdir(parents=True, exist_ok=True)
    series_cache = cache_dir / "GSE26712_series_matrix.txt.gz"
    parquet_cache = cache_dir / "GSE26712.symbol_expr.parquet"
    csv_cache = cache_dir / "GSE26712.symbol_expr.csv"
    if parquet_cache.is_file():
        return _maybe_symbol_index(pd.read_parquet(parquet_cache))
    if csv_cache.is_file() and csv_cache.stat().st_size > 1000:
        return _maybe_symbol_index(pd.read_csv(csv_cache, index_col=0))
    url = "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE26nnn/GSE26712/matrix/GSE26712_series_matrix.txt.gz"
    if not _download_file(url, series_cache):
        return None
    with gzip.open(series_cache, "rb") as fh:
        raw = fh.read().decode("utf-8", errors="replace")
    lines = raw.splitlines()
    beg = next(i for i, ln in enumerate(lines) if ln.startswith("!series_matrix_table_begin"))
    end = next(i for i, ln in enumerate(lines) if ln.startswith("!series_matrix_table_end"))
    table = pd.read_csv(io.StringIO("\n".join(lines[beg + 1 : end])), sep="\t", index_col=0)
    table.index = table.index.astype(str)
    try:
        import mygene

        mg = mygene.MyGeneInfo()
        mapping = mg.querymany(
            table.index.tolist(),
            scopes="reporter",
            fields="symbol",
            species="human",
            as_dataframe=True,
            verbose=False,
        )
        sym = mapping["symbol"].dropna().astype(str).str.upper()
        sym_map = sym.groupby(sym.index).first().to_dict()
    except Exception as exc:
        warnings.warn(f"Probe→symbol mapping failed: {exc}", UserWarning)
        sym_map = {}
    rows = [(sym_map.get(p, p).upper(), row) for p, row in table.iterrows()]
    expr = pd.DataFrame({s: r for s, r in rows}).T
    expr = expr.groupby(expr.index).mean()
    try:
        expr.to_parquet(parquet_cache)
    except Exception:
        expr.to_csv(csv_cache)
    return expr


def _parse_geo_characteristic_map(raw: str) -> dict[str, list[str]]:
    gsm_ids: Optional[list[str]] = None
    char_rows: dict[str, list[str]] = {}
    for ln in raw.splitlines():
        if ln.startswith("!Sample_geo_accession"):
            gsm_ids = [v.strip('"') for v in ln.split("\t")[1:]]
        elif ln.startswith("!Sample_title"):
            char_rows["title"] = [v.strip('"') for v in ln.split("\t")[1:]]
        elif ln.startswith("!Sample_characteristics_ch1"):
            vals = [v.strip('"') for v in ln.split("\t")[1:]]
            keys = [v.split(":", 1)[0].strip().lower() for v in vals if v and ":" in v]
            if not keys:
                continue
            key = pd.Series(keys).value_counts().index[0]
            char_rows[key] = vals
    if gsm_ids is None:
        return {}
    char_rows["gsm"] = gsm_ids
    return char_rows


def _aocs_clinical_from_geo(cache_dir: Path, *, force: bool = False) -> Optional[pd.DataFrame]:
    import gzip
    import re

    series_cache = cache_dir / "GSE26712_series_matrix.txt.gz"
    if not series_cache.is_file():
        return None
    clin_cache = cache_dir / "GSE26712_clinical_patient.csv"
    if clin_cache.is_file() and not force:
        return pd.read_csv(clin_cache)
    with gzip.open(series_cache, "rb") as fh:
        raw = fh.read().decode("utf-8", errors="replace")
    cmap = _parse_geo_characteristic_map(raw)
    gsm_ids = cmap.get("gsm")
    if not gsm_ids:
        return None
    n = len(gsm_ids)
    titles = cmap.get("title", [""] * n)
    tissues = cmap.get("tissue", [""] * n)
    statuses = cmap.get("status", [""] * n)
    surgeries = cmap.get("surgery outcome", cmap.get("surgery", [""] * n))
    years_raw = cmap.get("survival years", [""] * n)

    def _field(vals: list[str], i: int, prefix: str) -> str:
        if i >= len(vals):
            return ""
        v = str(vals[i])
        m = re.search(rf"{re.escape(prefix)}\s*:\s*(.*)$", v, re.I)
        return (m.group(1).strip() if m else v).strip()

    rows = []
    for i, gsm in enumerate(gsm_ids):
        title = titles[i] if i < len(titles) else ""
        tissue = _field(tissues, i, "tissue")
        status = _field(statuses, i, "status").upper()
        surgery = _field(surgeries, i, "surgery outcome")
        year_txt = _field(years_raw, i, "survival years")
        m = re.search(r"([0-9.]+)", year_txt)
        is_normal = "normal" in str(title).lower() or "hose" in str(title).lower() or "normal" in tissue.lower()
        if is_normal or m is None:
            continue
        os_months = float(m.group(1)) * 12.0
        token = status.split()[0].strip("()") if status else ""
        if token == "DOD" or status.startswith("DOD"):
            event = 1
        elif token in {"NED", "AWD"} or status.startswith("NED") or status.startswith("AWD"):
            event = 0
        else:
            continue
        surg = surgery.strip()
        surg_grp = None
        if surg.lower().startswith("opt"):
            surg_grp = "Optimal"
        elif surg.lower().startswith("sub"):
            surg_grp = "Suboptimal"
        rows.append(
            {
                "patient": gsm,
                "sample": gsm,
                "os_months": os_months,
                "os_event": event,
                "status": status,
                "surgery_outcome": surg_grp,
                "tissue": tissue,
            }
        )
    if not rows:
        return None
    clin = pd.DataFrame(rows)
    clin.to_csv(clin_cache, index=False)
    return clin


def _score_pdvs(expr: pd.DataFrame, genes: Sequence[str]) -> tuple[pd.Series, list[str]]:
    g_upper = [g.upper() for g in genes]
    present = [g for g in g_upper if g in expr.index]
    if len(present) < 3:
        raise ValueError(f"Too few PDVS genes in expression matrix: {present}")
    sub = expr.loc[present].astype(float)
    z = (sub - sub.mean(axis=1).values[:, None]) / (sub.std(axis=1).values[:, None] + 1e-8)
    return z.mean(axis=0), present


def _tcga_primary_columns(columns: Sequence[str]) -> pd.DataFrame:
    rows = []
    for col in columns:
        s = str(col)
        patient = s[:12]
        suffix = s[13:15] if len(s) >= 15 else ""
        is_primary = suffix.startswith("01") or "-01" in s[12:16]
        is_normal_or_blood = suffix.startswith("10") or suffix.startswith("11")
        rows.append(
            {
                "sample": s,
                "patient": patient,
                "is_primary": bool(is_primary and not is_normal_or_blood),
                "is_normal_or_blood": bool(is_normal_or_blood),
            }
        )
    meta = pd.DataFrame(rows)
    prim = meta.loc[meta["is_primary"]].copy()
    if prim.empty:
        prim = meta.loc[~meta["is_normal_or_blood"]].copy()
    prim = prim.sort_values(["patient", "sample"])
    return prim.drop_duplicates(subset=["patient"], keep="first")


def _n_at_risk(times: np.ndarray, t_points: Sequence[float]) -> list[int]:
    return [int(np.sum(times >= t)) for t in t_points]


def _fmt_risk(times: np.ndarray, groups: np.ndarray, label: str) -> str:
    m = groups == label
    vals = _n_at_risk(times[m], RISK_TIMES)
    return ";".join(f"{int(t)}m={n}" for t, n in zip(RISK_TIMES, vals))


def _kaplan_meier_plot(
    times: np.ndarray,
    events: np.ndarray,
    groups: np.ndarray,
    *,
    title: str,
    out_path: Path,
    n: int,
    n_events: int,
    hr_text: str = "",
) -> float:
    try:
        from lifelines import KaplanMeierFitter
        from lifelines.plotting import add_at_risk_counts
        from lifelines.statistics import logrank_test
    except ImportError:
        warnings.warn("lifelines not installed; skip KM plot", UserWarning)
        return float("nan")

    fig, ax = plt.subplots(figsize=(5.8, 5.0))
    kmf = KaplanMeierFitter()
    fitted = []
    order = ["high_PDVS", "low_PDVS"]
    labels = [g for g in order if g in set(groups)] or sorted(np.unique(groups))
    for i, lab in enumerate(labels):
        m = groups == lab
        kmf_i = KaplanMeierFitter()
        kmf_i.fit(times[m], events[m], label=str(lab).replace("_", " "))
        kmf_i.plot_survival_function(ax=ax, color=PALETTE[i % len(PALETTE)], lw=2.2)
        fitted.append(kmf_i)
    pval = float("nan")
    if len(labels) == 2:
        m0, m1 = groups == labels[0], groups == labels[1]
        lr = logrank_test(times[m0], times[m1], events[m0], events[m1])
        pval = float(lr.p_value)
    note = f"n={n:,}; events={n_events:,}\nlog-rank P={pval:.2e}"
    if hr_text:
        note += f"\n{hr_text}"
    ax.text(0.03, 0.04, note, transform=ax.transAxes, fontsize=8, va="bottom")
    ax.set_xlabel("Months")
    ax.set_ylabel("Overall survival probability")
    ax.set_title(title, fontsize=11)
    ax.set_ylim(0, 1.02)
    try:
        add_at_risk_counts(*fitted, ax=ax, rows_to_show=["At risk"])
    except Exception:
        pass
    style_axis(ax, grid_axis="y")
    fig.tight_layout()
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    return pval


def _cox_fit(df: pd.DataFrame, formula_cols: list[str], label: str) -> pd.DataFrame:
    from lifelines import CoxPHFitter

    use = df[["os_months", "os_event", *formula_cols]].dropna().copy()
    dummy_cols = [c for c in formula_cols if use[c].dtype == object or str(use[c].dtype) == "category"]
    if dummy_cols:
        use = pd.get_dummies(use, columns=dummy_cols, drop_first=True)
    covs = [c for c in use.columns if c not in {"os_months", "os_event"}]
    if use["os_event"].sum() < 8 or len(use) < 20 or not covs:
        return pd.DataFrame(
            [{"model": label, "covariate": "NA", "n": int(len(use)), "n_events": int(use["os_event"].sum()),
              "hr": np.nan, "hr_lo": np.nan, "hr_hi": np.nan, "p": np.nan, "status": "insufficient"}]
        )
    cph = CoxPHFitter()
    cph.fit(use, duration_col="os_months", event_col="os_event")
    summ = cph.summary
    rows = []
    for cov in summ.index:
        rows.append(
            {
                "model": label,
                "covariate": cov,
                "n": int(len(use)),
                "n_events": int(use["os_event"].sum()),
                "hr": float(summ.loc[cov, "exp(coef)"]),
                "hr_lo": float(summ.loc[cov, "exp(coef) lower 95%"]),
                "hr_hi": float(summ.loc[cov, "exp(coef) upper 95%"]),
                "p": float(summ.loc[cov, "p"]),
                "status": "ok",
            }
        )
    return pd.DataFrame(rows)


def _cox_block(patient_df: pd.DataFrame, extra_covs: list[str], cohort: str) -> pd.DataFrame:
    out = [_cox_fit(patient_df, ["PDVS_z"], f"{cohort}_uni_PDVS_z")]
    out.append(_cox_fit(patient_df, ["PDVS_high"], f"{cohort}_uni_PDVS_high"))
    avail = [c for c in extra_covs if c in patient_df.columns and patient_df[c].notna().sum() >= 20]
    if avail:
        out.append(_cox_fit(patient_df, ["PDVS_z", *avail], f"{cohort}_multi_PDVS_z"))
        out.append(_cox_fit(patient_df, ["PDVS_high", *avail], f"{cohort}_multi_PDVS_high"))
    else:
        out.append(
            pd.DataFrame(
                [{
                    "model": f"{cohort}_multi_PDVS_z",
                    "covariate": "NA",
                    "n": int(len(patient_df)),
                    "n_events": int(patient_df["os_event"].sum()),
                    "hr": np.nan,
                    "hr_lo": np.nan,
                    "hr_hi": np.nan,
                    "p": np.nan,
                    "status": "no_covariates",
                }]
            )
        )
    block = pd.concat(out, ignore_index=True)
    block.insert(0, "cohort", cohort)
    return block


def _hr_note(cox: pd.DataFrame, model: str, covariate: str) -> str:
    sub = cox[(cox["model"] == model) & (cox["covariate"] == covariate) & (cox["status"] == "ok")]
    if sub.empty:
        return ""
    r = sub.iloc[0]
    return f"uni HR={r['hr']:.2f} (95% CI {r['hr_lo']:.2f}–{r['hr_hi']:.2f})"


def _summarize_cohort(
    *,
    cohort: str,
    patient_df: pd.DataFrame,
    cox: pd.DataFrame,
    genes_present: Sequence[str],
    genes_requested: Sequence[str],
    covariates: Sequence[str],
    logrank_p: float,
) -> dict:
    high = patient_df["PDVS_group"] == "high_PDVS"
    low = patient_df["PDVS_group"] == "low_PDVS"
    uni = cox[(cox["model"] == f"{cohort}_uni_PDVS_z") & (cox["covariate"] == "PDVS_z")]
    uni_h = cox[(cox["model"] == f"{cohort}_uni_PDVS_high") & (cox["covariate"].astype(str).str.contains("PDVS_high"))]
    multi = cox[(cox["model"] == f"{cohort}_multi_PDVS_z") & (cox["covariate"] == "PDVS_z")]
    def _pick(df: pd.DataFrame, col: str) -> float:
        return float(df.iloc[0][col]) if len(df) and df.iloc[0]["status"] == "ok" else np.nan

    times = patient_df["os_months"].to_numpy(float)
    groups = patient_df["PDVS_group"].to_numpy()
    return {
        "cohort": cohort,
        "endpoint": "OS",
        "n_patients": int(len(patient_df)),
        "n_events": int(patient_df["os_event"].sum()),
        "n_high": int(high.sum()),
        "n_low": int(low.sum()),
        "n_events_high": int(patient_df.loc[high, "os_event"].sum()),
        "n_events_low": int(patient_df.loc[low, "os_event"].sum()),
        "pdvs_median": float(patient_df["PDVS"].median()),
        "genes_requested": ",".join(genes_requested),
        "genes_present": ",".join(genes_present),
        "n_genes_present": int(len(genes_present)),
        "covariates": ",".join(covariates) if covariates else "",
        "km_logrank_p": logrank_p,
        "uni_PDVS_z_hr": _pick(uni, "hr"),
        "uni_PDVS_z_hr_lo": _pick(uni, "hr_lo"),
        "uni_PDVS_z_hr_hi": _pick(uni, "hr_hi"),
        "uni_PDVS_z_p": _pick(uni, "p"),
        "uni_PDVS_high_hr": _pick(uni_h, "hr"),
        "uni_PDVS_high_hr_lo": _pick(uni_h, "hr_lo"),
        "uni_PDVS_high_hr_hi": _pick(uni_h, "hr_hi"),
        "uni_PDVS_high_p": _pick(uni_h, "p"),
        "multi_PDVS_z_hr": _pick(multi, "hr"),
        "multi_PDVS_z_hr_lo": _pick(multi, "hr_lo"),
        "multi_PDVS_z_hr_hi": _pick(multi, "hr_hi"),
        "multi_PDVS_z_p": _pick(multi, "p"),
        "at_risk_high": _fmt_risk(times, groups, "high_PDVS"),
        "at_risk_low": _fmt_risk(times, groups, "low_PDVS"),
        "independent_prognostic_claim": "no",
        "notes": "Exploratory OS association; not an independent prognostic claim.",
    }


def _analyse_tcga(expr: pd.DataFrame, clin: pd.DataFrame, genes: Sequence[str], out: Path) -> dict:
    scores, present = _score_pdvs(expr, genes)
    score_meta = _tcga_primary_columns(scores.index.astype(str))
    score_meta["PDVS"] = score_meta["sample"].map(scores.to_dict())
    merged = clin.merge(score_meta[["patient", "sample", "PDVS"]], on="patient", how="inner")
    merged = merged.dropna(subset=["PDVS", "os_months", "os_event"])
    merged["os_event"] = merged["os_event"].astype(int)
    sd = float(merged["PDVS"].std(ddof=0) or 1.0)
    merged["PDVS_z"] = (merged["PDVS"] - merged["PDVS"].mean()) / sd
    med = float(merged["PDVS"].median())
    merged["PDVS_high"] = (merged["PDVS"] >= med).astype(int)
    merged["PDVS_group"] = np.where(merged["PDVS_high"] == 1, "high_PDVS", "low_PDVS")
    extra = [c for c in ("age_years", "stage", "residual") if merged[c].notna().sum() >= 20]
    cox = _cox_block(merged, extra, "TCGA_OV")
    cox.to_csv(result_path(out, "pdvs_cox_TCGA_OV.csv"), index=False)
    pval = _kaplan_meier_plot(
        merged["os_months"].to_numpy(float),
        merged["os_event"].to_numpy(int),
        merged["PDVS_group"].to_numpy(),
        title="Overall survival (TCGA-OV, patient-level)",
        out_path=fig_path(out, "KM_OS_TCGA_OV_PDVS.png"),
        n=len(merged),
        n_events=int(merged["os_event"].sum()),
        hr_text=_hr_note(cox, "TCGA_OV_uni_PDVS_z", "PDVS_z"),
    )
    keep = [
        "patient", "sample", "PDVS", "PDVS_z", "PDVS_group", "os_months", "os_event",
        "age_years", "stage", "residual",
    ]
    patient_tbl = merged[keep].sort_values("patient")
    patient_tbl.to_csv(result_path(out, "pdvs_TCGA_OV_patient_table.csv"), index=False)
    patient_tbl[["sample", "PDVS"]].rename(columns={"sample": "sample"}).to_csv(
        result_path(out, "pdvs_TCGA_OV_scores.csv"), index=False
    )
    summary = _summarize_cohort(
        cohort="TCGA_OV",
        patient_df=merged,
        cox=cox,
        genes_present=present,
        genes_requested=genes,
        covariates=extra,
        logrank_p=pval,
    )
    return {"summary": summary, "cox": cox, "n": len(merged)}


def _analyse_aocs(expr: pd.DataFrame, clin: pd.DataFrame, genes: Sequence[str], out: Path) -> dict:
    scores, present = _score_pdvs(expr, genes)
    scores.index = scores.index.astype(str)
    merged = clin.copy()
    merged["PDVS"] = merged["sample"].astype(str).map(scores.to_dict())
    merged = merged.dropna(subset=["PDVS", "os_months", "os_event"])
    merged["os_event"] = merged["os_event"].astype(int)
    sd = float(merged["PDVS"].std(ddof=0) or 1.0)
    merged["PDVS_z"] = (merged["PDVS"] - merged["PDVS"].mean()) / sd
    med = float(merged["PDVS"].median())
    merged["PDVS_high"] = (merged["PDVS"] >= med).astype(int)
    merged["PDVS_group"] = np.where(merged["PDVS_high"] == 1, "high_PDVS", "low_PDVS")
    extra = [c for c in ("surgery_outcome",) if merged[c].notna().sum() >= 20]
    cox = _cox_block(merged, extra, "AOCS_GSE26712")
    cox.to_csv(result_path(out, "pdvs_cox_AOCS_GSE26712.csv"), index=False)
    pval = _kaplan_meier_plot(
        merged["os_months"].to_numpy(float),
        merged["os_event"].to_numpy(int),
        merged["PDVS_group"].to_numpy(),
        title="Overall survival (GSE26712 / AOCS proxy, tumors)",
        out_path=fig_path(out, "KM_OS_AOCS_GSE26712_PDVS.png"),
        n=len(merged),
        n_events=int(merged["os_event"].sum()),
        hr_text=_hr_note(cox, "AOCS_GSE26712_uni_PDVS_z", "PDVS_z"),
    )
    keep = [
        "patient", "sample", "PDVS", "PDVS_z", "PDVS_group", "os_months", "os_event",
        "status", "surgery_outcome",
    ]
    patient_tbl = merged[keep].sort_values("patient")
    patient_tbl.to_csv(result_path(out, "pdvs_AOCS_GSE26712_patient_table.csv"), index=False)
    patient_tbl[["sample", "PDVS"]].to_csv(result_path(out, "pdvs_AOCS_GSE26712_scores.csv"), index=False)
    summary = _summarize_cohort(
        cohort="AOCS_GSE26712",
        patient_df=merged,
        cox=cox,
        genes_present=present,
        genes_requested=genes,
        covariates=extra,
        logrank_p=pval,
    )
    summary["notes"] = (
        "GSE26712 late-stage HGSOC (AOCS-linked GEO series). "
        "OS event = DOD; NED/AWD censored. Exploratory; not an independent prognostic claim."
    )
    return {"summary": summary, "cox": cox, "n": len(merged)}


def run_clinical_pdvs_validation(
    checkpoint_dir: Path,
    protocol_dir: Optional[Path] = None,
    genes: Optional[Sequence[str]] = None,
) -> dict:
    protocol_dir = protocol_dir or (checkpoint_dir / "analysis_protocol_HGSOC")
    out = methods_outdir(checkpoint_dir)
    gene_list = _load_pdvs_genes(protocol_dir, genes)
    results_all = []

    tcga_cache = cache_path(out, "tcga")
    expr = _download_tcga_ov_xena(tcga_cache)
    clin = _download_tcga_clinical(tcga_cache, force=True)
    if expr is None:
        results_all.append({"cohort": "TCGA_OV", "status": "expression_download_failed", "genes_requested": ",".join(gene_list)})
    elif clin is None or clin.empty:
        results_all.append({"cohort": "TCGA_OV", "status": "clinical_download_failed", "genes_requested": ",".join(gene_list)})
    else:
        try:
            tcga = _analyse_tcga(expr, clin, gene_list, out)
            results_all.append(tcga["summary"])
        except Exception as exc:
            warnings.warn(f"TCGA analysis failed: {exc}", UserWarning)
            results_all.append({"cohort": "TCGA_OV", "status": f"failed:{exc}", "genes_requested": ",".join(gene_list)})

    aocs_cache = cache_path(out, "aocs")
    aocs_expr = _download_aocs_geo(aocs_cache)
    aocs_clin = _aocs_clinical_from_geo(aocs_cache, force=True)
    if aocs_expr is None:
        results_all.append({"cohort": "AOCS_GSE26712", "status": "expression_download_failed"})
    elif aocs_clin is None or aocs_clin.empty:
        results_all.append({"cohort": "AOCS_GSE26712", "status": "clinical_parse_failed"})
    else:
        try:
            aocs = _analyse_aocs(aocs_expr, aocs_clin, gene_list, out)
            results_all.append(aocs["summary"])
        except Exception as exc:
            warnings.warn(f"AOCS analysis failed: {exc}", UserWarning)
            results_all.append({"cohort": "AOCS_GSE26712", "status": f"failed:{exc}"})

    summary_df = pd.DataFrame(results_all)
    summary_df.to_csv(result_path(out, "pdvs_clinical_summary.csv"), index=False)
    write_output_file_index(out, dataset_key="HGSOC")
    return {"results": results_all, "genes": gene_list}


def main(argv=None):
    p = argparse.ArgumentParser(description="Patient-level PDVS OS validation (TCGA-OV and GSE26712)")
    p.add_argument("--checkpoint-dir", type=str, required=True)
    p.add_argument("--genes", nargs="*", default=None)
    args = p.parse_args(argv)
    rep = run_clinical_pdvs_validation(Path(args.checkpoint_dir), genes=args.genes)
    print(json.dumps(rep, default=str, indent=2))


if __name__ == "__main__":
    main()
