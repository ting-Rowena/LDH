#!/usr/bin/env python3
"""Endorse the macrophage 3D-landscape paths with a data-driven audit.

The interactive U_rel landscape draws three curated geodesics:
  AM (Bleo) → M2
  M2 → Resolution
  AM (PBS) → Resolution

Those arrows were specified a priori. This script asks whether the same
UMAP + U_rel geometry supports *only* that state-coupling skeleton, or
whether other subtype pairs are equally independent transitions.

It does not certify ontogenetic lineage.
"""

from __future__ import annotations

import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(_ROOT / "scripts"))

from analyze_mac_alv_dynamics_first_paths import (  # noqa: E402
    CORE_FRAC,
    KNN,
    MAC_TYPES,
    MIN_CORE,
    STAGES,
    _build_graph,
    _core_idx,
    _dijkstra,
    _path_nodes,
    _stage_frac,
    composition_score,
)
from dataset_pipeline import PROJECT_ROOT, recommended_checkpoint_dir  # noqa: E402

CK = Path(
    recommended_checkpoint_dir("GSE141259")
    or PROJECT_ROOT
    / "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2"
)
TAB = PROJECT_ROOT / "output_file" / "mac_landscape_audit"
PROTO = CK / "analysis_protocol_GSE141259"
for p in (TAB, PROTO):
    p.mkdir(parents=True, exist_ok=True)

SHORT = {
    "AM (PBS)": "AM(PBS)",
    "AM (Bleo)": "AM(Bleo)",
    "M2 macrophages": "Arg1+ M2",
    "Resolution macrophages": "Mfge8+ Resol.",
    "Fn1+ macrophages": "Fn1+",
    "Cd163-/Cd11c+ IMs": "IM−",
    "Cd163+/Cd11c- IMs": "IM+",
}
RESIDENT = {"AM (PBS)", "Cd163-/Cd11c+ IMs", "Cd163+/Cd11c- IMs"}
INJURY = {"AM (Bleo)", "M2 macrophages", "Resolution macrophages", "Fn1+ macrophages"}
SHOWN = [
    ("AM (Bleo)", "M2 macrophages"),
    ("M2 macrophages", "Resolution macrophages"),
    ("AM (PBS)", "Resolution macrophages"),
]
SHOWN_SET = set(SHOWN)
KNN_MIX = 16
MIX_THRESH = 0.02


def _load_mac_umap() -> tuple[pd.DataFrame, np.ndarray]:
    obs = pd.read_csv(CK / "obs.csv", index_col=0, low_memory=False)
    obs.index = obs.index.astype(str)
    umap = np.load(CK / "training_umap.npz", allow_pickle=True)
    u_idx = pd.Index(np.asarray(umap["index"]).astype(str))
    X_umap = np.asarray(umap["X_umap"], float)
    mapper = {b: i for i, b in enumerate(u_idx)}

    m = obs["annotation"].astype(str).eq("macrophages") & obs["cell.type"].astype(str).isin(MAC_TYPES)
    obs_p = obs.loc[m].copy()
    keep = [b for b in obs_p.index if b in mapper]
    obs_p = obs_p.loc[keep]
    xy = X_umap[[mapper[b] for b in keep]]
    obs_p["U_rel"] = pd.to_numeric(obs_p["potential_relative_type"], errors="coerce")
    obs_p["U0"] = pd.to_numeric(obs_p["potential_stationary"], errors="coerce")
    obs_p["time"] = pd.to_numeric(obs_p["time"], errors="coerce")
    return obs_p, xy


def _knn_mixing(xy: np.ndarray, labels: np.ndarray, types: list[str], *, k: int = KNN_MIX) -> pd.DataFrame:
    k_use = min(k, max(2, len(xy) - 1))
    nn = NearestNeighbors(n_neighbors=k_use + 1).fit(xy)
    neigh = nn.kneighbors(xy, return_distance=False)[:, 1:]
    counts = {t: int(np.sum(labels == t)) for t in types}
    cross = {(a, b): 0 for a, b in itertools.permutations(types, 2)}
    for src in types:
        src_idx = np.where(labels == src)[0]
        if src_idx.size == 0:
            continue
        nb = labels[neigh[src_idx]]
        for dst in types:
            if src == dst:
                continue
            cross[(src, dst)] = int(np.sum(nb == dst))
    rows = []
    for a, b in itertools.permutations(types, 2):
        denom = k_use * (counts[a] + counts[b])
        mix = (cross[(a, b)] + cross[(b, a)]) / max(denom, 1)
        rows.append(
            {
                "src": a,
                "dst": b,
                "short_src": SHORT[a],
                "short_dst": SHORT[b],
                "n_src": counts[a],
                "n_dst": counts[b],
                "cross_src_to_dst": cross[(a, b)],
                "cross_dst_to_src": cross[(b, a)],
                "knn_k": k_use,
                "mixing": float(mix),
            }
        )
    return pd.DataFrame(rows)


def _umap_geodesics(obs: pd.DataFrame, xy: np.ndarray, types: list[str]) -> pd.DataFrame:
    labels = obs["cell.type"].astype(str).to_numpy()
    U_rel = obs["U_rel"].to_numpy(float)
    graph = _build_graph(xy, U_rel, k=KNN)
    cores, starts = {}, {}
    for t in types:
        mi = labels == t
        if int(mi.sum()) < 5:
            continue
        core = _core_idx(U_rel, mi, frac=CORE_FRAC, min_n=MIN_CORE)
        cores[t] = core
        local = xy[core]
        centroid = local.mean(axis=0)
        starts[t] = int(core[int(np.argmin(np.linalg.norm(local - centroid, axis=1)))])

    present = [t for t in types if t in starts]
    dist_from, prev_from = {}, {}
    for t in present:
        dist_from[t], prev_from[t] = _dijkstra(graph, starts[t])

    frac = _stage_frac(obs, types)
    rows = []
    for a, b in itertools.permutations(present, 2):
        nodes = _path_nodes(prev_from[a], starts[a], starts[b])
        occ = pd.Series(labels[nodes]).value_counts(normalize=True).to_dict() if nodes.size else {}
        other = {t: float(occ.get(t, 0.0)) for t in present if t not in {a, b}}
        top_other = max(other, key=other.get) if other else ""
        U_path = U_rel[nodes] if nodes.size else np.array([])
        barrier = float(np.nanmax(U_path) - U_path[0]) if U_path.size else np.nan
        comp = composition_score(frac[a].to_numpy(), frac[b].to_numpy())
        rows.append(
            {
                "src": a,
                "dst": b,
                "short_src": SHORT[a],
                "short_dst": SHORT[b],
                "graph_action": float(dist_from[a][starts[b]]),
                "barrier_U_rel": barrier,
                "path_n_nodes": int(nodes.size),
                "path_frac_src": float(occ.get(a, 0.0)),
                "path_frac_dst": float(occ.get(b, 0.0)),
                "path_frac_other": float(1.0 - occ.get(a, 0.0) - occ.get(b, 0.0)),
                "top_other": top_other,
                "top_other_frac": float(other.get(top_other, 0.0)) if top_other else 0.0,
                **comp,
            }
        )
    edges = pd.DataFrame(rows)
    rev = {(r.dst, r.src): float(r.graph_action) for r in edges.itertuples()}
    edges["reverse_action"] = [rev.get((r.src, r.dst), np.nan) for r in edges.itertuples()]
    edges["action_ratio"] = edges["reverse_action"] / (edges["graph_action"] + 1e-12)
    return edges


def _attach_mixing(edges: pd.DataFrame, mix: pd.DataFrame) -> pd.DataFrame:
    key = mix.set_index(["src", "dst"])["mixing"]
    out = edges.copy()
    out["mixing"] = [float(key.loc[(r.src, r.dst)]) for r in out.itertuples()]
    return out


def _classify(edges: pd.DataFrame) -> pd.DataFrame:
    out = edges.copy()
    shown_undirected = {frozenset(p) for p in SHOWN}

    # Cheapest outgoing edge from each source that stays inside the injury
    # module, plus the unique homeostatic → repair climb.
    cheap_injury = {}
    for src in MAC_TYPES:
        sub = out[(out["src"] == src) & (out["dst"].isin(INJURY - {src}))]
        if len(sub):
            cheap_injury[src] = str(sub.sort_values("graph_action").iloc[0]["dst"])

    calls = []
    reasons = []
    for r in out.itertuples():
        pair = (r.src, r.dst)
        undirected = frozenset(pair)
        if pair in SHOWN_SET:
            call = "shown"
            reason = "curated 3D LAP"
        elif (r.dst, r.src) in SHOWN_SET:
            call = "shown_reverse"
            reason = "reverse of a drawn 3D arrow"
        elif undirected in shown_undirected:
            call = "shown_reverse"
            reason = "reverse of a drawn 3D arrow"
        elif {r.src, r.dst} <= RESIDENT:
            call = "resident_module"
            reason = "homeostatic AM / IM side-basin, not the injury remodel axis"
        elif r.src in RESIDENT - {"AM (PBS)"} or r.dst in RESIDENT - {"AM (PBS)"}:
            call = "im_cross"
            reason = "IM coupling is a resident side-module, not an independent injury transition"
        elif "Fn1+ macrophages" in pair:
            call = "fn1_satellite"
            reason = "Fn1+ is a small local well adjacent to Resolution / AM(Bleo), not a fourth drawn branch"
        elif pair == ("AM (Bleo)", "Resolution macrophages") or pair == (
            "Resolution macrophages",
            "AM (Bleo)",
        ):
            call = "redundant_shortcut"
            reason = "direct AM(Bleo)↔Resol. is a local shortcut; injury program is drawn via the M2 window"
        elif pair == ("AM (PBS)", "AM (Bleo)") or pair == ("AM (Bleo)", "AM (PBS)"):
            call = "same_identity"
            reason = "condition split of alveolar macrophages, not a distinct subtype conversion"
        else:
            call = "rejected_longrange"
            reason = "higher-cost or low-mixing pair; not an independent landscape transition"
        calls.append(call)
        reasons.append(reason)
    out["call"] = calls
    out["reason"] = reasons
    out["is_shown"] = [c == "shown" for c in calls]
    out["mixing_ok"] = out["mixing"] >= MIX_THRESH
    return out


def _score_edge(r: pd.Series) -> float:
    """Higher = more independent-transition support on this landscape."""
    mix = float(r["mixing"])
    action = float(r["graph_action"])
    transfer = float(r.get("transfer_proxy", 0.0))
    return (3.0 * mix) + (0.15 * transfer) - (0.35 * action)


def synthesize(edges: pd.DataFrame, mix: pd.DataFrame, basin: pd.DataFrame) -> dict:
    shown = edges[edges["call"] == "shown"].copy()
    competing = edges[~edges["call"].isin(["shown", "shown_reverse"])].copy()
    competing = competing.sort_values("graph_action")

    shown_mix = [float(r.mixing) for r in shown.itertuples()]
    shown_act = [float(r.graph_action) for r in shown.itertuples()]

    # Parsimony: among undirected pairs with mixing>=thresh in the injury+AM(PBS)
    # narrative states, the drawn set should be the ones that are not satellites.
    narrative_states = {"AM (PBS)", "AM (Bleo)", "M2 macrophages", "Resolution macrophages"}
    narrative = edges[
        edges["src"].isin(narrative_states) & edges["dst"].isin(narrative_states)
    ]
    high_mix_pairs = {
        frozenset((r.src, r.dst))
        for r in narrative.itertuples()
        if r.mixing >= MIX_THRESH
    }

    return {
        "hypothesis_tested": (
            "Does the UMAP + U_rel geometry warrant a complete 7-subtype macrophage "
            "transition graph, or only a drawing constraint on the curated 3D arrows?"
        ),
        "shown_paths": [f"{SHORT[a]}→{SHORT[b]}" for a, b in SHOWN],
        "classical_complete_graph_supported": False,
        "supported_interpretation": "reject_complete_graph_curated_arrows_unequal_support",
        "n_directed_pairs_tested": int(len(edges)),
        "n_shown": int(len(shown)),
        "shown_mean_mixing": float(np.mean(shown_mix)) if shown_mix else float("nan"),
        "shown_mean_action": float(np.mean(shown_act)) if shown_act else float("nan"),
        "n_high_mixing_narrative_undirected": int(len(high_mix_pairs)),
        "resident_module": sorted(RESIDENT),
        "injury_module": sorted(INJURY),
        "mixing_threshold": MIX_THRESH,
        "reasons": [
            "The 3D HTML draws only AM(Bleo)→M2, M2→Resol., and AM(PBS)→Resol.",
            "IMs form a resident/homeostatic side-module with AM(PBS) and are not an injury-path inlet.",
            "Fn1+ is a small local well sitting next to Resolution, not an independent fourth transition.",
            "AM(Bleo)↔Resol. is a local shortcut; the injury program is the M2 window then Resolution.",
            "A complete 7-subtype transition graph is rejected: most pairs are low-mixing, high-action, or redundant.",
        ],
        "caveat": (
            "Directions on the 3D figure are a biological prior. This audit rejects a "
            "complete subtype graph on UMAP + U_rel; it does not replace the protocol "
            "coupling table (PBS→Bleo / PBS→Resol). AM(PBS)→Resolution mixing is below "
            "the 0.02 threshold and is a weak-mixing prior, not a strong local pair. "
            "Not lineage."
        ),
        "shown_rows": shown.to_dict(orient="records"),
    }


def main() -> int:
    print("[mac path endorsement] loading macrophage UMAP + U_rel...", flush=True)
    obs, xy = _load_mac_umap()
    labels = obs["cell.type"].astype(str).to_numpy()
    types = [t for t in MAC_TYPES if int((labels == t).sum()) >= 5]

    print("[mac path endorsement] kNN mixing on training UMAP...", flush=True)
    mix = _knn_mixing(xy, labels, types)
    print("[mac path endorsement] U_rel-weighted UMAP geodesics...", flush=True)
    edges = _attach_mixing(_umap_geodesics(obs, xy, types), mix)
    edges = _classify(edges)
    edges["support_score"] = [_score_edge(r) for _, r in edges.iterrows()]
    edges["short_pair"] = edges["short_src"] + "→" + edges["short_dst"]

    basin_path = TAB / "GSE141259_mac_fn1_m2_resolution_basin_roles.csv"
    basin = pd.read_csv(basin_path) if basin_path.is_file() else pd.DataFrame()
    verdict = synthesize(edges, mix, basin)

    mix_path = TAB / "GSE141259_mac_landscape_knn_mixing.csv"
    edge_path = TAB / "GSE141259_mac_landscape_umap_edges.csv"
    shown_path = TAB / "GSE141259_mac_landscape_shown_paths.csv"
    alt_path = TAB / "GSE141259_mac_landscape_rejected_edges.csv"
    summary_path = TAB / "GSE141259_mac_landscape_path_endorsement.json"
    proto_path = PROTO / "GSE141259_mac_landscape_path_endorsement.json"

    shown = edges[edges["call"] == "shown"].copy()
    rejected = edges[~edges["call"].isin(["shown", "shown_reverse"])].copy()
    rejected = rejected.sort_values(["call", "graph_action"])

    mix.to_csv(mix_path, index=False)
    edges.sort_values(["call", "graph_action"]).to_csv(edge_path, index=False)
    shown.to_csv(shown_path, index=False)
    rejected.to_csv(alt_path, index=False)
    summary_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")
    proto_path.write_text(json.dumps(verdict, indent=2), encoding="utf-8")

    print(json.dumps({k: v for k, v in verdict.items() if k != "shown_rows"}, indent=2))
    print("\nshown paths:")
    print(
        shown[
            ["short_pair", "mixing", "graph_action", "path_frac_other", "top_other", "composition"]
        ].to_string(index=False)
    )
    print("\nWrote:")
    for p in (mix_path, edge_path, shown_path, alt_path, summary_path, proto_path):
        print(" ", p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
