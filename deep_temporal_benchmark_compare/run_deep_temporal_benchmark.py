#!/usr/bin/env python3
"""Run the held-out temporal benchmark across datasets and random seeds.

This is the paper-facing launcher for LDH-scRNA and the available temporal
population baselines:

  * WOT-inspired local entropic barycentric transport
  * PRESCIENT-family potential flow
  * MIOFlow-family time-conditioned neural ODE

The latter three are controlled, dependency-safe core-objective
reimplementations. They must not be described as official package runs.

All results from this launcher default to this directory:
  deep_temporal_benchmark_compare/
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import pandas as pd


HERE = Path(__file__).resolve().parent
REPO_ROOT = HERE.parent
DEFAULT_METHODS = [
    "our_model",
    "wot_barycentric",
    "prescient_potential_flow",
    "mioflow_neural_ode",
]
ADOPTED_CHECKPOINTS = {
    "GSE155622": REPO_ROOT
    / "GSE155622_checkpoints_3000_3000_384_0.05_recon0.01_lossnorm_qp_d0p01_z0p5_k0p2_ld1",
    "GSE141259": REPO_ROOT
    / "GSE141259_checkpoints_5000_5000_512_0.01_recon0.1_valD28_timeX_lossnorm_qp_d0p01_z0p1_k0p2",
    "HGSOC": REPO_ROOT
    / "HGSOC_checkpoints_3000_3000_512_0.05_recon0.01_nactpair_lossnorm_qp_d0p01_z0p5_k0p2_ld1",
}
TRANSPORT_CHECKPOINTS = {
    "GSE155622": HERE / "checkpoints" / "transportOT_v1" / "GSE155622",
    "GSE141259": HERE / "checkpoints" / "transportOT_v1" / "GSE141259",
    "HGSOC": HERE / "checkpoints" / "transportOT_v1" / "HGSOC",
}
TRANSPORT_V2_CHECKPOINTS = {
    "GSE155622": HERE / "checkpoints" / "transportOT_v2" / "GSE155622",
    "GSE141259": HERE / "checkpoints" / "transportOT_v2" / "GSE141259",
    "HGSOC": HERE / "checkpoints" / "transportOT_v2" / "HGSOC",
}
HAMILTONIAN4_CHECKPOINTS = {
    "GSE155622": HERE / "Hamiltonian4_GSE155622",
    "GSE141259": HERE / "Hamiltonian4_GSE141259",
    "HGSOC": HERE / "Hamiltonian4_HGSOC",
}
CHECKPOINT_SETS = {
    "adopted": ADOPTED_CHECKPOINTS,
    "hamiltonian4": HAMILTONIAN4_CHECKPOINTS,
    "transportOT_v1": TRANSPORT_CHECKPOINTS,
    "transportOT_v2": TRANSPORT_V2_CHECKPOINTS,
}
# Match each cohort's training validation protocol.
DATASET_VAL_MODE = {
    "GSE155622": "time_extrapolate",
    "GSE141259": "time_extrapolate",
    "HGSOC": "patients",
    "GSE225948_Brain": "time_extrapolate",
}
METRICS = [
    "energy_distance",
    "mean_marginal_w1",
    "mmd",
    "mean_shift_l2",
    "ot_sinkhorn",
]


def _checkpoint_map(values: list[str]) -> dict[str, str]:
    result: dict[str, str] = {}
    for value in values:
        if "=" not in value:
            raise SystemExit(
                f"Invalid --checkpoint {value!r}; expected DATASET=/path/to/checkpoint_dir"
            )
        dataset, path = value.split("=", 1)
        result[dataset] = path
    return result


def _run_one(
    *,
    dataset: str,
    seed: int,
    save_dir: Path,
    cache_dir: Path,
    methods: list[str],
    checkpoint: str | None,
    device: str,
    score_space: str,
    max_cells: int,
    max_source_cells: int,
) -> pd.DataFrame:
    run_dir = save_dir / dataset / f"seed_{seed}"
    cache = cache_dir / f"{dataset}_processed.h5ad"
    val_mode = DATASET_VAL_MODE.get(dataset, "time_extrapolate")
    command = [
        sys.executable,
        str(HERE / "baseline_evaluation.py"),
        "--dataset",
        dataset,
        "--val-mode",
        val_mode,
        "--save-dir",
        str(run_dir),
        "--processed-cache",
        str(cache),
        "--device",
        device,
        "--score-space",
        score_space,
        "--max-cells",
        str(max_cells),
        "--max-source-cells",
        str(max_source_cells),
        "--seed",
        str(seed),
        "--methods",
        *methods,
    ]
    if checkpoint is not None:
        command[2:2] = ["--checkpoint-dir", checkpoint]

    print(f"[deep-baseline] {dataset} seed={seed}", flush=True)
    subprocess.run(command, cwd=REPO_ROOT, check=True)

    result_path = run_dir / "baseline_per_transition.csv"
    if not result_path.is_file():
        raise RuntimeError(f"Benchmark did not create {result_path}")
    frame = pd.read_csv(result_path)
    frame.insert(0, "dataset", dataset)
    if "our_model" in methods and "our_model" not in set(frame["method"]):
        raise RuntimeError(
            f"{dataset}: LDH-scRNA checkpoint was not loaded. Pass "
            f"--checkpoint {dataset}=/path/to/checkpoint_dir."
        )
    return frame


def _write_summary(per_transition: pd.DataFrame, save_dir: Path) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    per_transition.to_csv(save_dir / "deep_baseline_per_transition.csv", index=False)

    per_seed = (
        per_transition.groupby(["dataset", "seed", "method"], as_index=False)[METRICS]
        .mean()
        .sort_values(["dataset", "seed", "energy_distance"])
    )
    per_seed.to_csv(save_dir / "deep_baseline_per_seed.csv", index=False)

    summary = per_seed.groupby(["dataset", "method"])[METRICS].agg(["mean", "std"])
    summary.columns = [f"{metric}_{stat}" for metric, stat in summary.columns]
    summary = summary.reset_index().sort_values(["dataset", "energy_distance_mean"])
    summary.to_csv(save_dir / "deep_baseline_summary.csv", index=False)

    note = (
        "# Deep temporal baseline benchmark\n\n"
        "Lower is better for every metric. Results use identical held-out transitions, "
        "source-cell subsamples, target populations, and scoring spaces within each run.\n\n"
        "**Fidelity note:** `prescient_potential_flow` and `mioflow_neural_ode` are "
        "controlled core-objective reimplementations, not official package runs. "
        "`wot_barycentric` is WOT-inspired and omits WOT's growth-rate model.\n\n"
    )
    try:
        table = summary.to_markdown(index=False)
    except ImportError:
        table = summary.to_string(index=False)
    (save_dir / "deep_baseline_summary.md").write_text(note + table + "\n", encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["GSE155622", "GSE141259", "HGSOC"],
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--methods", nargs="+", default=DEFAULT_METHODS)
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=[],
        metavar="DATASET=DIR",
        help="Optional checkpoint override; repeat once per dataset.",
    )
    parser.add_argument(
        "--checkpoint-set",
        choices=sorted(CHECKPOINT_SETS.keys()),
        default="adopted",
        help="Which LDH checkpoint family to load when --checkpoint is omitted "
        "(adopted=paper landscape models; hamiltonian4=four-method transport table; "
        "transportOT_v1/v2=transport-priority retrains).",
    )
    parser.add_argument(
        "--save-dir",
        type=Path,
        default=HERE / "results",
        help="Output directory for summaries and per-dataset runs (default: this folder/results).",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=HERE / "_cache",
        help="Processed AnnData cache directory (default: this folder/_cache).",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--score-space", choices=["gene", "pca", "latent"], default="pca")
    parser.add_argument("--max-cells", type=int, default=1000)
    parser.add_argument("--max-source-cells", type=int, default=512)
    args = parser.parse_args(argv)

    # Always keep outputs under this package unless user passes an absolute override
    # outside it; relative paths are resolved against HERE.
    save_dir = args.save_dir if args.save_dir.is_absolute() else (HERE / args.save_dir)
    cache_dir = args.cache_dir if args.cache_dir.is_absolute() else (HERE / args.cache_dir)
    # If user passes --save-dir results/pca, keep under HERE; default is HERE/results.
    if save_dir.resolve() == (HERE / "results").resolve():
        # Organize by score space for subsequent analyses.
        save_dir = HERE / "results" / str(args.score_space)
    elif args.save_dir == Path("results"):
        save_dir = HERE / "results" / str(args.score_space)

    checkpoints = _checkpoint_map(args.checkpoint)
    default_set = CHECKPOINT_SETS[args.checkpoint_set]
    for dataset, path in default_set.items():
        if dataset not in checkpoints and (path / "best_model.pth").is_file():
            checkpoints[dataset] = str(path)

    # Keep transport-priority evals under results/<tag>/<score_space>/
    if args.checkpoint_set.startswith("transportOT_") and args.save_dir in {
        Path("results"),
        HERE / "results",
    }:
        save_dir = HERE / "results" / args.checkpoint_set / str(args.score_space)
    elif (
        args.checkpoint_set.startswith("transportOT_")
        and save_dir.resolve() == (HERE / "results" / str(args.score_space)).resolve()
    ):
        save_dir = HERE / "results" / args.checkpoint_set / str(args.score_space)

    frames = []
    for dataset in args.datasets:
        if "our_model" in args.methods and dataset not in checkpoints:
            raise SystemExit(
                f"No LDH checkpoint for {dataset} in set {args.checkpoint_set!r}. "
                f"Train first: python deep_temporal_benchmark_compare/train_transport_priority.py "
                f"--datasets {dataset}"
            )
        for seed in args.seeds:
            frames.append(
                _run_one(
                    dataset=dataset,
                    seed=seed,
                    save_dir=save_dir,
                    cache_dir=cache_dir,
                    methods=args.methods,
                    checkpoint=checkpoints.get(dataset),
                    device=args.device,
                    score_space=args.score_space,
                    max_cells=args.max_cells,
                    max_source_cells=args.max_source_cells,
                )
            )
    _write_summary(pd.concat(frames, ignore_index=True), save_dir)
    print(f"wrote {save_dir / 'deep_baseline_summary.csv'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
