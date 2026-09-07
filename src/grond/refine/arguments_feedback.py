"""Argument declarations for the shelxl feedback loop workflow.

The adapter supplies runtime defaults explicitly. Keep option order, numerical
defaults, choices, and help text identical to the established command contract.
"""
from __future__ import annotations

import argparse
import os
import sys


def build_parser(description: str) -> argparse.ArgumentParser:
    """Build the parser without reading data or starting a calculation."""

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--source-summary", required=True, help="Existing native_row_regularize summary whose selected nuisance values should be pinned.")
    ap.add_argument("--start-res", required=True, help="Initial structure for cycle 1.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--hkl", default="")
    ap.add_argument("--baseline-hkl", default="")
    ap.add_argument(
        "--ignore-source-baseline-hkl",
        action="store_true",
        help=(
            "Do not inherit baseline_hkl from the source summary. Use this for a single-crystal leave-one-out "
            "replay when the pinned nuisance summary came from a larger cohort."
        ),
    )
    ap.add_argument("--hkl-transform", default="")
    ap.add_argument("--cycles", type=int, default=3)
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--final-n-basis", type=int, default=0, help="Optional high-N final polish basis. 0 disables the final polish.")
    ap.add_argument("--final-n-points", type=int, default=20)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--shelxl", default=os.environ.get("SHELXL", "shelxl"))
    ap.add_argument("--shelxl-iterations", type=int, default=3)
    ap.add_argument("--endpoint-ls-cycles", type=int, default=1, help="Internal L.S./CGLS cycles used inside each generated SHELXL endpoint input.")
    ap.add_argument("--endpoint-wght-mode", choices=["freeze-zero", "inherit", "promote"], default="freeze-zero")
    ap.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="uniform_z_rms", help="Endpoint sigma calibration applied before each SHELXL feedback update.")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    ap.add_argument("--lambda-grid", default="1.0")
    ap.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    ap.add_argument("--cache-compression", choices=["compressed", "none"], default="none")
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy")
    ap.add_argument("--row-plan-cache", choices=["off", "exact", "readwrite", "rebuild"], default="off")
    ap.add_argument("--jax-batch-size", type=int, default=0)
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0)
    ap.add_argument("--bethe-strong-count", type=int, default=48)
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005)
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5)
    ap.add_argument("--min-d-spacing", type=float, default=0.0)
    ap.add_argument("--topup-d-min", type=float, default=0.0)
    ap.add_argument("--min-peak", type=float, default=-1.0)
    ap.add_argument("--geometry-percentile", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--make-maps-on-final", action="store_true", help="Generate maps only for the last requested cycle or final polish.")
    return ap
