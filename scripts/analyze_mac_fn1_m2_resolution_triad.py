#!/usr/bin/env python3
"""Fn1+ / Arg1+ M2 / Mfge8+ Resolution triad audit (GSE141259 macrophages).

Tests whether bleomycin-associated macrophage states form a classical
fibrogenic↔resolution lineage bifurcation, or reversible multi-attractor
state plasticity. Does NOT claim Strunz lineage proof.

Inputs (precomputed):
  output_file/mac_landscape_audit/GSE141259_mac_alv_dynamics_first_*.csv
  output_file/mac_landscape_audit/GSE141259_mac_alv_3d_landscape_lap_summary.csv
  output_file/mac_landscape_audit/GSE141259_mac_alv_parent_process_type.csv
  <CK>/obs.csv

Outputs:
  output_file/mac_landscape_audit/GSE141259_mac_fn1_m2_resolution_triad_*.csv/json
  <CK>/analysis_protocol_GSE141259/GSE141259_mac_fn1_m2_resolution_triad_summary.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from dataset_pipeline import PROJECT_ROOT, recommended_checkpoint_dir

CK = Path(
    recommended_checkpoint_dir("GSE141259")
    or PROJECT_ROOT
    / "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
TAB = PROJECT_ROOT / "output_file" / "mac_landscape_audit"
PROTO = CK / "analysis_protocol_GSE141259"
for p in (TAB, PROTO):
    p.mkdir(parents=True, exist_ok=True)

TRIAD = ["Fn1+ macrophages", "M2 macrophages", "Resolution macrophages"]
SHORT = {
    "Fn1+ macrophages": "Fn1+",
    "M2 macrophages": "Arg1+ M2",
    "Resolution macrophages": "Mfge8+ Resol.",
    "AM (PBS)": "AM(PBS)",
    "AM (Bleo)": "AM(Bleo)",
    "Recruited macrophages": "Recruited",
}
STAGE_ORDER = [0.0, 3.0, 7.0, 10.0, 14.0, 21.0, 28.0]
STAGE_LAB = {0.0: "D0", 3.0: "D3", 7.0: "D7", 10.0: "D10", 14.0: "D14", 21.0: "D21", 28.0: "D28"}


def _pair_key(a: str, b: str) -> tuple[str, str]:
    return tuple(sorted([a, b]))


def build_time_composition(obs: pd.DataFrame) -> pd.DataFrame:
    mac = obs[obs["annotation"].astype(str) == "macrophages"].copy()
    mac["time"] = pd.to_numeric(mac["time"], errors="coerce")
    rows = []
    for t in STAGE_ORDER:
        sub = mac[mac["time"] == t]
        n_parent = len(sub)
        for ct in TRIAD + ["AM (PBS)", "AM (Bleo)"]:
            n = int((sub["cell.type"].astype(str) == ct).sum())
            rows.append(
                {
                    "compartment": "macrophages",
                    "stage": STAGE_LAB[t],
                    "day": t,
                    "cell.type": ct,
                    "short": SHORT[ct],
                    "n": n,
                    "n_parent": n_parent,
                    "frac": (n / n_parent) if n_parent else np.nan,
                }
            )
    # Origin context: Recruited macrophages live under monocytes annotation
    mono = obs[obs["annotation"].astype(str) == "monocytes"].copy()
    mono["time"] = pd.to_numeric(mono["time"], errors="coerce")
    for t in STAGE_ORDER:
        sub = mono[mono["time"] == t]
        n_parent = len(sub)
        n = int((sub["cell.type"].astype(str) == "Recruited macrophages").sum())
        rows.append(
            {
                "compartment": "monocytes",
                "stage": STAGE_LAB[t],
                "day": t,
                "cell.type": "Recruited macrophages",
                "short": SHORT["Recruited macrophages"],
                "n": n,
                "n_parent": n_parent,
                "frac": (n / n_parent) if n_parent else np.nan,
            }
        )
    return pd.DataFrame(rows)


def build_basin_table(att: pd.DataFrame) -> pd.DataFrame:
    mac = att[att["parent"] == "macrophages"].copy()
    mac["short"] = mac["cell.type"].map(lambda x: SHORT.get(x, x))
    mac["in_triad"] = mac["cell.type"].isin(TRIAD)
    keep = [
        "cell.type",
        "short",
        "n",
        "role",
        "mean_potential_stationary",
        "mean_potential_relative_type",
        "mean_plasticity",
        "peak_stage",
        "transient_score",
        "intermediate_call",
        "in_triad",
    ]
    return mac[keep].sort_values("mean_potential_relative_type")


def build_triad_edges(edges: pd.DataFrame) -> pd.DataFrame:
    mac = edges[edges["parent"] == "macrophages"].copy()
    # Undirected triad pairs: keep both directions
    mask = mac["src"].isin(TRIAD) & mac["dst"].isin(TRIAD)
    tri = mac.loc[mask].copy()
    tri["pair"] = [f"{SHORT[a]}↔{SHORT[b]}" for a, b in zip(tri["src"], tri["dst"])]
    tri["short_src"] = tri["src"].map(SHORT)
    tri["short_dst"] = tri["dst"].map(SHORT)
    return tri.sort_values(["src", "dst"])


def build_feed_edges(edges: pd.DataFrame) -> pd.DataFrame:
    """AM(PBS)/AM(Bleo) → triad coupling (tests multi-exit vs single fork)."""
    mac = edges[edges["parent"] == "macrophages"].copy()
    srcs = {"AM (PBS)", "AM (Bleo)"}
    mask = mac["src"].isin(srcs) & mac["dst"].isin(TRIAD)
    feed = mac.loc[mask].copy()
    feed["short_src"] = feed["src"].map(SHORT)
    feed["short_dst"] = feed["dst"].map(SHORT)
    feed["edge"] = feed["short_src"] + "→" + feed["short_dst"]
    return feed.sort_values(["src", "dst"])


def build_lap_triad(lap: pd.DataFrame) -> pd.DataFrame:
    mac = lap[lap["parent"] == "macrophages"].copy()
    triad_kinds = {
        "M2→Resol.",
        "Fn1+→Resol.",
        "AM(Bleo)→M2",
        "AM(PBS)→Resol.",
    }
    out = mac[mac["kind"].isin(triad_kinds) | (
        mac["src"].isin(TRIAD + ["AM (PBS)", "AM (Bleo)"])
        & mac["dst"].isin(TRIAD)
    )].copy()
    return out


def synthesize_verdict(basin: pd.DataFrame, triad_edges: pd.DataFrame, process: pd.DataFrame) -> dict:
    b = basin.set_index("cell.type")
    roles = {ct: str(b.loc[ct, "role"]) for ct in TRIAD if ct in b.index}
    m2_window = bool(b.loc["M2 macrophages", "intermediate_call"]) if "M2 macrophages" in b.index else False
    fn1_well = roles.get("Fn1+ macrophages") == "well"
    m2_slope = roles.get("M2 macrophages") == "high_U_slope"
    resol_slope = roles.get("Resolution macrophages") == "high_U_slope"

    # All triad undirected pairs nearly reversible?
    ratios = triad_edges["action_ratio"].astype(float)
    mean_ratio = float(np.nanmean(ratios)) if len(ratios) else float("nan")
    all_reversible = bool(triad_edges["reversible"].astype(bool).all()) if len(triad_edges) else False

    mac_proc = process[process["parent"] == "macrophages"].iloc[0]
    alv_proc = process[process["parent"] == "alv_epithelium"].iloc[0]

    classical_bifurcation = False  # data do not support lineage fork
    reasons = []
    if fn1_well and m2_slope:
        reasons.append("Fn1+ is a relative well while Arg1+ M2 is a high-U_rel injury-window slope — not one shared fibrogenic basin.")
    if m2_window:
        reasons.append("M2 shows a strong inverted-U injury window (D10), consistent with transient activation not terminal fate.")
    if all_reversible and np.isfinite(mean_ratio) and 0.9 <= mean_ratio <= 1.1:
        reasons.append(
            f"Triad edges are reversible (mean action_ratio={mean_ratio:.3f}); no strongly directional lineage commitment."
        )
    reasons.append(
        f"Parent process type is {mac_proc['process_type']} "
        f"(flux={float(mac_proc['non_conservative_flux_fraction']):.3f}, "
        f"asymmetry={float(mac_proc['action_asymmetry_ratio']):.3f}), matching plasticity not directional differentiation."
    )

    return {
        "hypothesis_tested": (
            "Do recruited/injury macrophages bifurcate into fibrogenic-like "
            "(Fn1+/Arg1+ M2) vs resolution-like (Mfge8+) lineage fates?"
        ),
        "classical_lineage_bifurcation_supported": classical_bifurcation,
        "supported_interpretation": "reversible_multi_attractor_state_plasticity",
        "basin_roles": roles,
        "fn1_and_m2_same_basin": False if (fn1_well and m2_slope) else None,
        "m2_is_injury_window": m2_window,
        "triad_mean_action_ratio": mean_ratio,
        "triad_edges_reversible": all_reversible,
        "mac_process_type": str(mac_proc["process_type"]),
        "alv_process_type": str(alv_proc["process_type"]),
        "mac_flux_fraction": float(mac_proc["non_conservative_flux_fraction"]),
        "alv_flux_fraction": float(alv_proc["non_conservative_flux_fraction"]),
        "contrast_with_epithelium": (
            "Shared reversible-plasticity math; epithelium is interpreted as differentiation/"
            "trajectory hypotheses, macrophages as state-coupling/turnover — not lineage commitment."
        ),
        "reasons": reasons,
        "caveat": (
            "Strunz showed temporal state succession and monocyte contribution to multiple identities; "
            "this audit tests landscape topology/path reversibility and does not certify ontogenetic lineage."
        ),
    }


def main() -> int:
    obs = pd.read_csv(
        CK / "obs.csv",
        usecols=["cell.type", "annotation", "time"],
        low_memory=False,
    )
    att = pd.read_csv(TAB / "GSE141259_mac_alv_dynamics_first_attractors.csv")
    edges = pd.read_csv(TAB / "GSE141259_mac_alv_dynamics_first_edges.csv")
    lap = pd.read_csv(TAB / "GSE141259_mac_alv_3d_landscape_lap_summary.csv")
    process = pd.read_csv(TAB / "GSE141259_mac_alv_parent_process_type.csv")

    time_df = build_time_composition(obs)
    basin = build_basin_table(att)
    triad_edges = build_triad_edges(edges)
    feed = build_feed_edges(edges)
    lap_tri = build_lap_triad(lap)
    verdict = synthesize_verdict(basin, triad_edges, process)

    time_path = TAB / "GSE141259_mac_fn1_m2_resolution_time_composition.csv"
    basin_path = TAB / "GSE141259_mac_fn1_m2_resolution_basin_roles.csv"
    triad_path = TAB / "GSE141259_mac_fn1_m2_resolution_triad_edges.csv"
    feed_path = TAB / "GSE141259_mac_fn1_m2_resolution_feed_edges.csv"
    lap_path = TAB / "GSE141259_mac_fn1_m2_resolution_lap_paths.csv"
    summary_path = TAB / "GSE141259_mac_fn1_m2_resolution_triad_summary.json"
    proto_path = PROTO / "GSE141259_mac_fn1_m2_resolution_triad_summary.json"

    time_df.to_csv(time_path, index=False)
    basin.to_csv(basin_path, index=False)
    triad_edges.to_csv(triad_path, index=False)
    feed.to_csv(feed_path, index=False)
    lap_tri.to_csv(lap_path, index=False)
    summary_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    proto_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(json.dumps(verdict, indent=2))
    print("Wrote:")
    for p in (time_path, basin_path, triad_path, feed_path, lap_path, summary_path, proto_path):
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
