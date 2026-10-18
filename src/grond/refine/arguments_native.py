"""Argument declarations for the native structure update workflow.

The adapter supplies runtime defaults explicitly. Keep option order, numerical
defaults, choices, and help text identical to the established command contract.
"""
from __future__ import annotations

import argparse


def build_parser(description: str, *, default_hkl, default_res, default_baseline_hkl, target_modes, max_support_sg) -> argparse.ArgumentParser:
    """Build the parser without reading data or starting a calculation."""

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--hkl", default=str(default_hkl))
    ap.add_argument("--res", default=str(default_res))
    ap.add_argument("--baseline-hkl", default=str(default_baseline_hkl))
    ap.add_argument("--hkl-transform", default="identity", help="Signed axis permutation mapping XDS HKLs into the model basis, e.g. 'l,-k,h'.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"], default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50)
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=0.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0, help="Keep this broad geometry percentile after PEAK/d-spacing filters.")
    ap.add_argument("--target-mode", choices=list(target_modes), default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true", help="Use alternate fractional xyz before native row-parameter regularization as an alternate-hand control.")
    ap.add_argument("--objective-rows", type=int, default=800)
    ap.add_argument("--objective-bins", type=int, default=10)
    ap.add_argument("--objective-group-width-deg", type=float, default=3.0, help="Coarse frame-angle group width used only while fitting coordinate steps.")
    ap.add_argument(
        "--objective-scale-mode",
        choices=["frozen_block", "refit_each_eval"],
        default="frozen_block",
        help="Scale policy inside coordinate/ADP microcycles. frozen_block fits objective scales once from the block-start model and reuses them for every finite-difference evaluation; refit_each_eval preserves the older, more fragile behavior.",
    )
    ap.add_argument("--scale-fit-split", choices=["work", "all"], default="work", help="Rows used to fit validation/final nuisance scales.")
    ap.add_argument("--detector-scale-bins", type=int, default=0, help="Optional NxN detector-position scales in objective and validation fits.")
    ap.add_argument("--block-schedule", default="xyz", help="Comma-separated model-update blocks: xyz, adp_ueq, or xyz_adp_ueq. Each block is validation-gated before the next starts.")
    ap.add_argument("--atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    ap.add_argument("--atom-labels", default="")
    ap.add_argument("--max-shift-frac", type=float, default=0.015)
    ap.add_argument("--xyz-restraint-sigma", type=float, default=0.008)
    ap.add_argument("--adp-shift", type=float, default=0.010, help="Maximum Ueq change allowed per ADP block.")
    ap.add_argument("--adp-restraint-sigma", type=float, default=0.004)
    ap.add_argument("--adp-min", type=float, default=0.002)
    ap.add_argument("--adp-max", type=float, default=0.080)
    ap.add_argument("--restraints", choices=["none", "start_geometry"], default="start_geometry")
    ap.add_argument("--bond-restraint-sigma", type=float, default=0.025)
    ap.add_argument("--adp-similarity-sigma", type=float, default=0.006)
    ap.add_argument("--bond-detection-tolerance", type=float, default=0.45)
    ap.add_argument("--bond-detection-max-distance", type=float, default=1.95)
    ap.add_argument("--max-nfev", type=int, default=80)
    ap.add_argument(
        "--diff-step",
        type=float,
        default=None,
        help="Deprecated compatibility name setting both absolute finite-difference steps in each parameter's native unit; use the group-specific options instead. It is not a SciPy relative multiplier.",
    )
    ap.add_argument("--xyz-diff-step-frac", type=float, default=1e-4, help="Absolute fractional-coordinate step used for native Bloch finite differences.")
    ap.add_argument("--adp-diff-step-a2", type=float, default=1e-4, help="Absolute Ueq step in A^2 used for native Bloch finite differences.")
    ap.add_argument("--thickness", type=float, default=2200.0)
    ap.add_argument("--thickness-grad-frac", type=float, default=0.8)
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--topup-d-min", type=float, default=0.8, help="Resolution limit for Bloch support top-up beams.")
    ap.add_argument(
        "--max-support-sg",
        type=float,
        default=max_support_sg,
        help="Maximum absolute excitation error for candidate support beams.",
    )
    ap.add_argument("--group-width-deg", type=float, default=1.0)
    ap.add_argument("--frame-offset", type=float, default=0.0)
    ap.add_argument("--row-resolution-b", type=float, default=0.0, help="Raw-row dynamical resolution-envelope B from nuisance regularization.")
    ap.add_argument("--ug-d-min", type=float, default=0.30, help="Model-dependent U_g cutoff used by the production kernel; raising it is an explicit speed/accuracy tradeoff.")
    ap.add_argument("--sg-jitter-mode", choices=["none", "gaussian", "student_t"], default="none", help="Accepted rocking-profile Sg jitter model from the production row regularizer.")
    ap.add_argument("--sg-jitter-sigma0", type=float, default=0.0, help="Base Sg jitter sigma in inverse Angstrom.")
    ap.add_argument("--sg-jitter-sigma-d-inv", type=float, default=0.0, help="Resolution-dependent Sg jitter coefficient.")
    ap.add_argument("--sg-jitter-sigma-frame", type=float, default=0.0, help="Frame-dependent Sg jitter coefficient.")
    ap.add_argument("--sg-jitter-points", type=int, default=1, help="Quadrature points used for the accepted Sg jitter convolution.")
    ap.add_argument("--sg-jitter-student-df", type=float, default=4.0, help="Student-t degrees of freedom for student_t Sg jitter.")
    ap.add_argument("--sg-jitter-width-factor", type=float, default=3.0, help="Finite support width in sigma units for the Sg jitter convolution.")
    ap.add_argument("--absorption-mode", choices=["none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"], default="none", help="Accepted absorptive Bloch Hamiltonian mode from the production row regularizer.")
    ap.add_argument("--absorption-mu", type=float, default=0.0, help="Uniform imaginary diagonal damping in inverse Angstrom.")
    ap.add_argument("--absorption-frac", type=float, default=0.0, help="Imaginary absorptive off-diagonal fraction.")
    ap.add_argument("--absorption-mu-g2", type=float, default=0.0, help="Beam-dependent diagonal damping slope versus |g|^2.")
    ap.add_argument("--absorption-mu-theta2", type=float, default=0.0, help="Beam-dependent diagonal damping slope versus Bragg-angle^2.")
    ap.add_argument("--absorption-tds-b", type=float, default=0.0, help="B_TDS parameter for offdiag_tds absorption.")
    ap.add_argument("--sigma-a", type=float, default=0.0)
    ap.add_argument("--sigma-b-corr", type=float, default=0.0)
    ap.add_argument("--sigma-c-peak", type=float, default=0.05)
    ap.add_argument("--row-plan-cache", choices=["off", "exact", "readwrite", "rebuild"], default="off", help="Cache row geometry. exact is model-keyed; readwrite explicitly freezes the builder basis across moving-model evaluations.")
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.")
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    from grond.regularize.arguments_native import add_split_scheme_args

    add_split_scheme_args(ap)
    ap.add_argument(
        "--split-contract",
        default="",
        help="Immutable candidate-union raw-row work/free split (orbit-hash runs; legacy keeps the per-dataset split).",
    )
    ap.add_argument(
        "--lambda-grid",
        default="1.0",
        help="Diagnostic q-exponent grid. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument("--acceptance-metric", choices=["all_wR2", "work_wR2", "free_wR2", "all_R1"], default="all_wR2")
    ap.add_argument("--acceptance-min-delta", type=float, default=0.0, help="Required absolute improvement in the acceptance metric before exporting a moved model.")
    ap.add_argument("--free-wR2-tolerance", type=float, default=0.0, help="Maximum allowed free wR2 worsening unless --allow-free-worse is set.")
    ap.add_argument("--allow-free-worse", action="store_true")
    ap.add_argument("--early-stop-window", type=int, default=96, help="Stop a structure block after this many residual calls without a meaningful objective improvement; 0 disables.")
    ap.add_argument("--early-stop-min-evals", type=int, default=0, help="Minimum residual calls before early stopping; 0 uses an automatic parameter-count-aware floor.")
    ap.add_argument("--early-stop-min-improvement", type=float, default=1e-5, help="Minimum RMS improvement counted as progress for structure-block early stopping.")
    ap.add_argument("--early-stop-metric", choices=["total_rms", "data_rms"], default="total_rms", help="RMS trace column used by the structure-block plateau stop.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--make-maps", action="store_true")
    return ap
