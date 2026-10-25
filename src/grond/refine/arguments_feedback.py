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
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default=None)
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default=None)
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
    ap.add_argument("--endpoint-wght-mode", choices=["converge"], default="converge")
    ap.add_argument("--wght-converge-max-calls", type=int, default=20, help="Maximum SHELXL calls of one endpoint under --endpoint-wght-mode converge.")
    ap.add_argument("--error-model-refits", type=int, default=1, help="error_model only: refits of the error model against the refined endpoint model.")
    ap.add_argument("--error-model-dataset-weight", choices=["apply", "off"], default="apply", help="Combined endpoints only (a dataset-policy weight exists only there); accepted so that one handoff option list serves every engine.")
    ap.add_argument("--zerr-esds", default="", help="Six cell e.s.d.s a,b,c,alpha,beta,gamma for the ZERR record of generated endpoint inputs. Empty keeps the model's ZERR.")
    ap.add_argument("--strip-q-peaks", action="store_true", help="Drop previous difference-map peaks (Q1, Q2, ...) from generated endpoint inputs.")
    ap.add_argument("--endpoint-list-code", type=int, default=0, help="LIST code of generated endpoint inputs (AUTO 6). 0 keeps the model's LIST.")
    ap.add_argument("--shelxl-convergence-shift-esd", type=float, default=0.0, help="Stop the SHELXL calls of one macro-cycle once |max shift/esd| is below this value (AUTO 0.01). 0 keeps the fixed call count (legacy).")
    ap.add_argument("--shelxl-npd-recovery", choices=["damp", "isor"], default="damp", help="NPD recovery: legacy DAMP ladder or ISOR on the NPD atoms only (AUTO).")
    ap.add_argument("--endpoint-q-policy", choices=["exclude"], default="exclude", help="Pathological correction ratios: exclude the rows.")
    ap.add_argument("--shelxl-feedback-xyz-tol", type=float, default=0.0, help="Stop the macro-cycles once the max non-H coordinate shift between successive models is below this value in A (AUTO 0.002). 0 runs every requested cycle (legacy).")
    ap.add_argument("--shelxl-feedback-keep", choices=["last"], default="last", help="Macro-cycle model to keep: the last adopted cycle.")
    ap.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms", "error_model"], default="uniform_z_rms", help="Endpoint sigma calibration applied before each SHELXL feedback update.")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20)
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25)
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0)
    ap.add_argument("--lambda-grid", default="1.0")
    ap.add_argument("--rescale", choices=["none", "absolute"], default="absolute")
    ap.add_argument("--cache-compression", choices=["compressed", "none"], default="none")
    ap.add_argument("--row-plan-cache", choices=["off", "exact", "readwrite", "rebuild"], default="off")
    ap.add_argument("--min-d-spacing", type=float, default=0.0)
    ap.add_argument("--topup-d-min", type=float, default=0.0)
    ap.add_argument("--min-peak", type=float, default=-1.0)
    ap.add_argument("--geometry-percentile", type=float, default=0.0)
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--make-maps-on-final", action="store_true", help="Generate maps only for the last requested cycle or final polish.")
    ap.add_argument("--feedback-thickness-update", choices=["on", "off"], default="off", help="Re-select the nested thickness distribution and lambda on the spectra of every new structure (variant fixed) and correct each cycle with it (AUTO under --nuisance-selection nested). off pins the source summary's thickness (legacy).")
    ap.add_argument("--nested-nuisance-contract", default="", help="decisions/<ds>/nested_nuisance.json of the nested nuisance stage; required by --feedback-thickness-update on.")
    return ap
