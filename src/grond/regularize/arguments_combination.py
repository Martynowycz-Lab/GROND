"""Argument declarations for the combine native nuisance runs workflow.

The adapter supplies runtime defaults explicitly. Keep option order, numerical
defaults, choices, and help text identical to the established command contract.
"""
from __future__ import annotations

import argparse


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build the parser without reading data or starting a calculation."""

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--run", action="append", required=True, help="name=/path/to/native_nuisance_run")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--wght-a", type=float, default=None)
    ap.add_argument("--wght-b", type=float, default=None)
    ap.add_argument(
        "--endpoint-lambda",
        default="1.0",
        help="Diagnostic endpoint q exponent. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument(
        "--endpoint-source",
        choices=("native", "hkl"),
        default="native",
        help="Build corrected endpoint rows from native_dyn_calc.csv, or read normalized endpoint HKL rows from each run.",
    )
    ap.add_argument(
        "--interdataset-scale",
        choices=("overlap", "none"),
        default="overlap",
        help="Fit one scalar per dataset from overlapping corrected reflections before merging the multi-dataset endpoint.",
    )
    ap.add_argument(
        "--endpoint-merge-key",
        choices=("model_symmetry", "model_symmetry_friedel", "exact_signed_hkl"),
        default="model_symmetry",
        help=(
            "Reflection key used before the shared endpoint LS step. model_symmetry merges corrected observations related "
            "by the model space-group symmetry while preserving Friedel opposites in acentric groups; model_symmetry_friedel "
            "also merges Friedel mates; exact_signed_hkl is a diagnostic legacy mode."
        ),
    )
    ap.add_argument(
        "--dataset-sigma-policy",
        choices=("none", "uniform_z_rms"),
        default="none",
        help=(
            "Endpoint uncertainty policy applied independently to each dataset before interdataset scaling and merging. "
            "This keeps a weak crystal from setting the sigma scale for stronger crystals."
        ),
    )
    ap.add_argument("--dataset-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--dataset-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--dataset-sigma-max-scale", type=float, default=4.0)
    ap.add_argument(
        "--dataset-sigma-holdout-floor",
        action="store_true",
        help="For per-dataset uniform_z_rms, do not deflate sigmas below that dataset's held-out RMS(z).",
    )
    ap.add_argument(
        "--merge-sigma-policy",
        choices=("formal", "chisq_overdispersion"),
        default="formal",
        help=(
            "How to assign sigmas after merging corrected endpoint rows. formal uses propagated inverse-variance errors; "
            "chisq_overdispersion inflates a merged reflection when its contributors disagree more than their sigmas predict."
        ),
    )
    ap.add_argument("--shelxl", default="")
    ap.add_argument("--shelxl-iterations", type=int, default=5)
    ap.add_argument(
        "--endpoint-ls-cycles",
        type=int,
        default=1,
        help="Internal L.S./CGLS cycle count written to the combined corrected-HKL endpoint. Production keeps this at 1.",
    )
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=("freeze-zero", "inherit", "promote"),
        default="freeze-zero",
        help=(
            "SHELXL weight policy for the shared combined endpoint. freeze-zero is the production default after "
            "GROND sigma calibration; inherit preserves the source active WGHT; promote uses SHELXL recommendations on later passes."
        ),
    )
    ap.add_argument(
        "--post-bloch-sigma-policy",
        choices=("none", "uniform_z_rms"),
        default="none",
        help=(
            "Endpoint uncertainty policy applied after multi-dataset merging and before SHELXL. "
            "uniform_z_rms fits one residual RMS(z) scalar from the supplied model and scales endpoint sigmas only."
        ),
    )
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    ap.add_argument(
        "--min-d-spacing",
        "--endpoint-min-d-spacing",
        dest="min_d_spacing",
        type=float,
        default=0.0,
        help=(
            "Optional reporting/endpoint cutoff applied to existing native row tables before joint metrics, "
            "sigma calibration, merging, and SHELXL. Zero preserves every row in the supplied runs."
        ),
    )
    return ap
