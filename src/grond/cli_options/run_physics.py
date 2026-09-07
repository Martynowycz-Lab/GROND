"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse

from grond.cli_support.values import DEFAULT_MAX_SUPPORT_SG


def add_backend(ap: argparse.ArgumentParser) -> None:
    """Add backend options in the established order."""

    ap.add_argument(
        "--prediction-cache-dir", default="", help="Optional root for prediction caches. Default is OUT/cache."
    )
    ap.add_argument(
        "--prediction-cache-scope",
        choices=["shared", "stage"],
        default="shared",
        help="Shared reuses safe hash-keyed predictions across adaptive stages; stage isolates every run.",
    )
    ap.add_argument(
        "--prediction-cache-compression",
        choices=["compressed", "none"],
        default="none",
        help="Use none for fastest local SSD/NVMe caches; compressed saves disk on slow or limited storage.",
    )
    ap.add_argument(
        "--compute-backend",
        choices=[
            "scipy",
            "scipy_bethe",
            "scipy_expm",
            "scipy_expm_multiply",
            "jax",
            "jax_auto",
            "jax_cpu",
            "jax_mps",
            "jax_cuda",
            "jax_padded",
            "jax_auto_padded",
            "jax_cpu_padded",
            "jax_mps_padded",
            "jax_cuda_padded",
            "jax_cpu_real_expm",
            "jax_mps_real_expm",
            "jax_mps_real_expm_padded",
        ],
        default="scipy",
        help="Bloch prediction backend. SciPy eigensolver is the production reference; scipy_bethe uses an opt-in promoted Bethe/Schur strong-beam reduction; SciPy expm/expm_multiply, real-embedded JAX expm, and JAX eigensolver backends are experimental agreement/speed probes. *_padded JAX backends pad matrix dimensions into small buckets to reduce compilation churn.",
    )
    ap.add_argument(
        "--row-plan-cache",
        choices=["auto", "off", "exact", "readwrite", "rebuild"],
        default="auto",
        help=(
            "Row-geometry cache policy. auto uses model-keyed exact reuse only "
            "for fixed-model grids with at least 16 predictions per geometry key "
            "and disables it for moving-model updates; readwrite explicitly "
            "freezes the builder basis across model updates."
        ),
    )
    ap.add_argument(
        "--jax-batch-size",
        type=int,
        default=0,
        help="JAX matrix batch size for accelerator backends. 0 chooses a conservative size from matrix dimension and device count.",
    )
    ap.add_argument(
        "--jax-matrix-bucket-size",
        type=int,
        default=0,
        help="Pad JAX matrix dimensions to this multiple. 0 keeps exact sizes unless a *_padded backend is selected, where 0 means 8.",
    )
    ap.add_argument(
        "--bethe-strong-count",
        type=int,
        default=48,
        help="Strong-beam block size for scipy_bethe. Values >= final N recover the dense SciPy matrix.",
    )
    ap.add_argument(
        "--bethe-variant",
        choices=["truncate", "schur0", "schur_target", "schur_mean"],
        default="schur_target",
        help="Strong/weak-beam reduction used by scipy_bethe.",
    )
    ap.add_argument(
        "--bethe-promote-denom",
        type=float,
        default=0.005,
        help="Promote weak beams whose diagonal denominator is this close to the Schur energy.",
    )
    ap.add_argument(
        "--bethe-denom-eps", type=float, default=1e-5, help="Small-denominator floor for scipy_bethe Schur corrections."
    )
    ap.add_argument(
        "--matrix-convention",
        choices=["j_minus_i", "i_minus_j"],
        default="j_minus_i",
        help="Off-diagonal U_g indexing convention. Keep j_minus_i for production; i_minus_j is diagnostic.",
    )


def add_basis(ap: argparse.ArgumentParser) -> None:
    """Add basis options in the established order."""

    ap.add_argument(
        "--screen-n-basis", type=int, default=24, help="Basis size for initial nuisance/geometry screening."
    )
    ap.add_argument(
        "--screen-n-points", type=int, default=10, help="Quadrature points for initial nuisance/geometry screening."
    )
    ap.add_argument(
        "--n-basis",
        type=int,
        default=0,
        help="Basis size for pinned final polish runs. Default 0 scans Sg support and uses max support plus margin.",
    )
    ap.add_argument(
        "--n-points",
        type=int,
        default=20,
        help="Quadrature points for pinned final polish runs; keep <=20 for production.",
    )
    ap.add_argument(
        "--final-basis-scan-limit",
        type=int,
        default=500,
        help="Upper limit used by the cheap support-only scan when --n-basis=0.",
    )
    ap.add_argument(
        "--final-basis-margin",
        type=int,
        default=10,
        help="Extra beams above the observed maximum support for final polish.",
    )
    ap.add_argument("--final-basis-min", type=int, default=20)
    ap.add_argument("--final-basis-max", type=int, default=500)
    ap.add_argument("--final-basis-fallback", type=int, default=128, help="Dry-run fallback when --n-basis=0.")


def add_geometry(ap: argparse.ArgumentParser) -> None:
    """Add geometry options in the established order."""

    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument(
        "--sg-window-mode",
        choices=["auto", "fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"],
        default="auto",
        help="Rocking-width source. Auto keeps the fixed Sg width; DIALS bbox z-ranges are available as an explicit diagnostic.",
    )
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50)
    ap.add_argument(
        "--sg-window-scout",
        choices=["auto", "on", "off"],
        default="off",
        help="Late held-out scout over simple fixed Sg rocking-window widths after final scale and resolution gates. Off is the production default; use auto/on when auditing rocking-width sensitivity.",
    )
    ap.add_argument(
        "--sg-window-scout-grid",
        default="0.003,0.004,0.006,0.008",
        help="Comma-separated fixed Sg widths tested by the late rocking-window scout.",
    )
    ap.add_argument("--sg-window-scout-n-basis", type=int, default=24, help="Basis size for the late Sg-window scout.")
    ap.add_argument(
        "--sg-window-scout-n-points", type=int, default=10, help="Quadrature points for the late Sg-window scout."
    )
    ap.add_argument(
        "--sg-window-scout-max-rows",
        type=int,
        default=0,
        help="Optional row cap for the late Sg-window scout; 0 uses the final row set.",
    )
    ap.add_argument(
        "--sg-window-scout-min-wR2-gain",
        type=float,
        default=0.002,
        help="Minimum all-row wR2 gain required before a non-baseline Sg width can be promoted.",
    )
    ap.add_argument(
        "--sg-window-scout-free-wR2-tolerance",
        type=float,
        default=0.003,
        help="Maximum held-out wR2 penalty allowed when promoting a non-baseline Sg width.",
    )
    ap.add_argument(
        "--topup-d-min",
        type=float,
        default=0.8,
        help="Nominal Bloch support top-up dmin. Auto never uses a shallower support limit than the selected data dmin.",
    )
    ap.add_argument(
        "--max-support-sg",
        type=float,
        default=DEFAULT_MAX_SUPPORT_SG,
        help="Maximum absolute excitation error for beam-basis support (default 0.01 A^-1).",
    )
    ap.add_argument(
        "--profile-quadrature",
        choices=["uniform", "trapezoid", "gauss_legendre"],
        default="uniform",
        help="Rocking-profile Sg quadrature rule used inside Bloch row predictions.",
    )
    ap.add_argument(
        "--profile-accumulation",
        choices=["sum", "mean"],
        default="sum",
        help="Accumulate rocking-profile samples as summed detector sub-exposures; mean is the legacy normalized diagnostic.",
    )
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument(
        "--selection-metric",
        choices=["work_wR2", "free_wR2", "all_wR2", "all_R1"],
        default="work_wR2",
        help="Metric used to choose nuisance trials. Work-row wR2 keeps free rows held out; all/free choices are diagnostics.",
    )


def add_nuisance(ap: argparse.ArgumentParser) -> None:
    """Add nuisance options in the established order."""

    ap.add_argument(
        "--row-resolution-b-grid",
        default="0.0,14,18",
        help="Raw-row dynamical resolution-envelope grid. Positive values are the dynamic-gradient counterpart of negative endpoint-B diagnostics.",
    )
    ap.add_argument("--thickness-grid", default="1000,1200,1400")
    ap.add_argument("--thickness-model-grid", default="uniform_gradient")
    ap.add_argument("--thickness-grad-grid", default="0.4,1.0")
    ap.add_argument(
        "--continuous-thickness-polish",
        choices=["auto", "on", "off"],
        default="auto",
        help="After the coarse thickness grid, polish continuous thickness/gradient by reusing cached Bloch eigensystems. Auto is on for GROND auto except --fast.",
    )
    ap.add_argument(
        "--continuous-thickness-metric",
        choices=["work_wR2", "all_wR2", "free_wR2", "all_R1", "all_R1gt"],
        default="work_wR2",
        help="Objective used by continuous thickness polish. Work-row wR2 keeps free rows held out.",
    )
    ap.add_argument(
        "--continuous-thickness-strong-rows",
        type=int,
        default=0,
        help="Rows used by continuous thickness polish; 0 uses all selected rows. Strong-row-only scouts are diagnostic and are not the default.",
    )
    ap.add_argument(
        "--continuous-thickness-max-rows",
        type=int,
        default=0,
        help="Hard row limit for continuous thickness polish; 0 uses all rows selected by the data filters.",
    )
    ap.add_argument(
        "--continuous-thickness-bracket-frac",
        type=float,
        default=0.35,
        help="Local fractional thickness bracket around the coarse-grid winner.",
    )
    ap.add_argument(
        "--continuous-thickness-grid-step", type=float, default=100.0, help="Guard-grid thickness step in Angstrom."
    )
    ap.add_argument(
        "--continuous-thickness-grad-step", type=float, default=0.2, help="Guard-grid thickness-gradient step."
    )
    ap.add_argument(
        "--continuous-thickness-powell-maxiter",
        type=int,
        default=18,
        help="Maximum Powell iterations for the final phase-only thickness/gradient polish.",
    )
    ap.add_argument(
        "--continuous-thickness-min-gain",
        type=float,
        default=5e-5,
        help="Minimum objective gain required to promote a non-grid continuous thickness.",
    )
    ap.add_argument(
        "--continuous-thickness-max-free-wr2-loss",
        type=float,
        default=0.002,
        help="Maximum tolerated held-out/free wR2 loss for promoted continuous thickness.",
    )
    ap.add_argument(
        "--continuous-thickness-max-all-r1-loss",
        type=float,
        default=0.001,
        help="Maximum tolerated all-row R1 loss for promoted continuous thickness.",
    )
    ap.add_argument(
        "--frame-offset-grid",
        default="auto",
        help=(
            "Profile-center offsets in XDS frames, or 'auto'. Auto reconciles the global XDS header "
            "orientation with row ZCAL positions before work-row nuisance selection."
        ),
    )
    ap.add_argument(
        "--group-width-grid",
        default="auto",
        help="Scale-group width grid. Numeric values are degrees; frame, 2frame, 3frames, and auto resolve from XDS OSCILLATION_RANGE. Auto never goes below one physical frame; explicit numeric sub-frame values are diagnostic opt-ins.",
    )
    ap.add_argument(
        "--detector-scale-bins",
        default="0",
        help="Detector-position field used during physical nuisance selection. The default is detector-free so a flexible field cannot choose thickness or other physics; 0x0, 2x2, 4x4, and diagnostic 8x8 fields are evaluated later on the same frozen predictions.",
    )
    ap.add_argument(
        "--detector-scale-ridge",
        type=float,
        default=0.05,
        help="Dimensionless shrinkage of detector factors toward unity.",
    )
    ap.add_argument(
        "--detector-scale-smoothness",
        type=float,
        default=0.10,
        help="Dimensionless nearest-neighbor detector-field smoothness penalty.",
    )
    ap.add_argument(
        "--detector-scale-min-rows",
        type=int,
        default=20,
        help="Minimum work rows required for every independently supported detector tile.",
    )
    ap.add_argument("--detector-scale-min-holdout-improvement", type=float, default=0.002)
    ap.add_argument("--detector-scale-required-improved-splits", type=int, default=2)
    ap.add_argument("--detector-scale-max-bloch-gain-loss", type=float, default=0.002)
    ap.add_argument("--detector-scale-min-field-correlation", type=float, default=0.50)
    ap.add_argument("--detector-scale-max-field-log-rms", type=float, default=0.10)
    ap.add_argument(
        "--detector-scale-independent-calibration",
        action="store_true",
        help="Permit 8x8 detector-scale promotion only when an independent detector calibration supports that field.",
    )
    ap.add_argument("--sigma-a-grid", default="0.0")
    ap.add_argument("--sigma-b-corr-grid", default="0.0")
    ap.add_argument("--sigma-c-peak-grid", default="0.0,0.15")
    ap.add_argument("--sg-jitter-mode-grid", default="none,gaussian")
    ap.add_argument("--sg-jitter-sigma0-grid", default="0.0,0.0015")
    ap.add_argument("--sg-jitter-sigma-d-inv-grid", default="0.0")
    ap.add_argument("--sg-jitter-sigma-frame-grid", default="0.0")
    ap.add_argument("--sg-jitter-points", type=int, default=3)
    ap.add_argument(
        "--pinned-sg-jitter-fallback",
        choices=["none", "off"],
        default="none",
        help=(
            "When a pinned nonzero Sg-jitter convolution is locally infeasible, also score an explicit no-jitter "
            "candidate. This preserves the local-root bound and records any selected nuisance change."
        ),
    )
    ap.add_argument("--absorption-mode-grid", default="none,midgley_like")
    ap.add_argument("--absorption-mu-grid", default="0.0,0.00005")
    ap.add_argument("--absorption-frac-grid", default="0.0,0.02")
    ap.add_argument("--absorption-mu-g2-grid", default="0.0")
    ap.add_argument("--absorption-mu-theta2-grid", default="0.0")
    ap.add_argument("--absorption-tds-b-grid", default="0.0")
    ap.add_argument("--dose-b-mode-grid", default="none")
    ap.add_argument("--dose-b-per-frame-grid", default="0.0")
    ap.add_argument(
        "--thickness-points", type=int, default=5, help="Thickness quadrature points for screen/regularization stages."
    )
    ap.add_argument(
        "--final-thickness-points",
        type=int,
        default=9,
        help="Thickness quadrature points for pinned final-basis polish runs.",
    )
