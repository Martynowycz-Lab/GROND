"""Argument declarations for the native row regularize workflow.

The adapter supplies runtime defaults explicitly. Keep option order, numerical
defaults, choices, and help text identical to the established command contract.
"""
from __future__ import annotations

import argparse


def add_split_scheme_args(ap: argparse.ArgumentParser) -> None:
    """Engine split-scheme options; the engine default stays the legacy per-dataset shuffle."""

    ap.add_argument(
        "--split-scheme",
        choices=["legacy", "orbit-hash"],
        default="legacy",
        help=(
            "Free-set definition: legacy shuffles this dataset's orbits with --split-seed; orbit-hash is "
            "crystal-independent (salted hash of the model Laue-orbit key, grond.validation.heldout)."
        ),
    )
    from grond.validation.heldout import DEFAULT_FREE_SALT

    ap.add_argument("--split-salt", default=DEFAULT_FREE_SALT, help="Salt of the orbit-hash free set.")
    ap.add_argument(
        "--exclude-orbits-json",
        default="",
        help="Lockbox file of orbits held back from every stage (orbit-hash only); those rows are dropped before any fit.",
    )


def build_parser(description: str, *, default_hkl, default_res, default_baseline_hkl, target_modes, max_support_sg, shelxl, runtime_cache_directory) -> argparse.ArgumentParser:
    """Build the parser without reading data or starting a calculation."""

    ap = argparse.ArgumentParser(description=description)
    ap.add_argument("--hkl", default=str(default_hkl))
    ap.add_argument("--res", default=str(default_res))
    ap.add_argument("--baseline-hkl", default=str(default_baseline_hkl))
    ap.add_argument("--hkl-transform", default="identity", help="Signed axis permutation mapping XDS HKLs into the model basis, e.g. 'l,-k,h'.")
    ap.add_argument(
        "--row-contract",
        default="",
        help=(
            "Immutable GROND native_dyn_calc.csv row-membership contract. When supplied, select exactly its "
            "row_index set after verifying HKL, target_I, xds_sigma, and d_A against the XDS input; do not "
            "rebuild membership from raw eligibility filters."
        ),
    )
    ap.add_argument(
        "--split-contract",
        default="",
        help=(
            "Immutable candidate-union raw-row work/free split shared by every "
            "transform candidate and subsequent hand comparison."
        ),
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"], default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50)
    ap.add_argument("--sg-jitter-mode-grid", default="none", help="Comma grid: none, gaussian, student_t.")
    ap.add_argument("--sg-jitter-sigma0-grid", default="0.0", help="Base target-Sg jitter sigma grid in inverse Angstrom.")
    ap.add_argument("--sg-jitter-sigma-d-inv-grid", default="0.0", help="Target-Sg jitter sigma slope grid versus 1/d.")
    ap.add_argument("--sg-jitter-sigma-frame-grid", default="0.0", help="Target-Sg jitter sigma slope grid versus normalized frame number.")
    ap.add_argument("--sg-jitter-points", type=int, default=5, help="Kernel quadrature points for jitter convolution.")
    ap.add_argument("--sg-jitter-student-df", type=float, default=4.0)
    ap.add_argument("--sg-jitter-width-factor", type=float, default=3.0)
    ap.add_argument("--mosaicity-mode", choices=["none", "gaussian_omega", "gaussian_cone", "transverse_target"], default="none", help="Fixed projected one-axis or incoherent two-axis Gaussian mosaic model; transverse_target is the re-centred transverse cone (target_sg profiles only).")
    ap.add_argument("--mosaicity-sigma-deg", type=float, default=0.0, help="Projected Gaussian mosaicity sigma in degrees; one nuisance parameter when active.")
    ap.add_argument("--mosaicity-mode-grid", default="", help="Comma grid of mosaicity modes searched as nuisance candidates. Empty uses --mosaicity-mode.")
    ap.add_argument("--mosaicity-sigma-deg-grid", default="", help="Comma grid of mosaicity sigmas in degrees for every active grid mode (mode none always uses 0). Empty uses --mosaicity-sigma-deg.")
    ap.add_argument("--mosaicity-max-sigma-deg", type=float, default=0.5, help="Hard upper bound for projected mosaicity sigma.")
    ap.add_argument("--mosaicity-points", type=int, default=3, help="Gauss-Hermite points for direct-angle mosaic quadrature.")
    ap.add_argument("--mosaicity-support-sigma-deg", type=float, default=0.0, help="Largest tested two-axis cone sigma used to construct one common beam basis for every candidate width.")
    ap.add_argument("--mosaicity-support-points", type=int, default=5, help="Points per scan/cone axis used only for common-basis construction.")
    ap.add_argument("--mosaicity-support-width-sigma", type=float, default=4.0, help="Half-width of the common basis cone in support-sigma units.")
    ap.add_argument("--geometry-path", default="", help="Fixed geometry-only slew/two-axis tilt contract fitted without intensities.")
    ap.add_argument("--profile-basis-mode", choices=["center", "common_profile"], default="center", help="Use the center-orientation basis or one fixed beam union over the complete rocking-profile support.")
    ap.add_argument("--profile-coordinate", choices=["target_sg", "direct_angle"], default="target_sg", help="Integrate through solved target-Sg points or directly in recorded scan angle.")
    ap.add_argument("--profile-normalization", choices=["legacy", "target_sg_equivalent_v1"], default="legacy", help="Explicit direct-angle measure; target_sg_equivalent_v1 preserves target-Sg row normalization and endpoints.")
    ap.add_argument("--profile-angle-max-half-width-deg", type=float, default=2.0, help="Hard half-width bound for direct-angle integration.")
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.")
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=0.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0, help="Keep this broad geometry percentile after PEAK/d-spacing filters; 99 removes only the worst 1 percent geometry tail.")
    ap.add_argument("--target-mode", choices=list(target_modes), default="iobs")
    ap.add_argument("--positive-only", action="store_true", default=True)
    ap.add_argument("--include-nonpositive", dest="positive_only", action="store_false")
    ap.add_argument("--thickness-grid", default="1200,1600,1800,2200")
    ap.add_argument("--thickness-model-grid", default="uniform_gradient", help="Comma grid: single, uniform_gradient, gaussian, gaussian_gradient, gamma.")
    ap.add_argument("--thickness-grad-grid", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--thickness-sigma-frac-grid", default="0.0", help="Gaussian thickness sigma/t0 grid. Used only by thickness_model=gaussian.")
    ap.add_argument("--thickness-gamma-shape-grid", default="1.0", help="Gamma thickness shape grid (the mean is --thickness-grid). Used only by thickness_model=gamma.")
    ap.add_argument(
        "--thickness-evaluator",
        choices=["quadrature", "table", "cf", "auto"],
        default="quadrature",
        help=(
            "Thickness average. quadrature: the legacy --thickness-points nodes (not for gamma). table: exact "
            "hat-function weights on a --thickness-table-dt grid over the support. cf: exact untruncated gamma "
            "average. auto: cf when the gamma mass outside the support is below 1e-6, else table."
        ),
    )
    ap.add_argument("--thickness-min-nm", type=float, default=10.0, help="Physical thickness support minimum (nm) for table/cf/auto.")
    ap.add_argument("--thickness-max-nm", type=float, default=1000.0, help="Physical thickness support maximum (nm) for table/cf/auto.")
    ap.add_argument("--thickness-table-dt", type=float, default=2.0, help="Thickness table step (A) for table/auto.")
    ap.add_argument(
        "--frame-offset-grid",
        default="auto",
        help=(
            "Comma grid in XDS frames, or 'auto'. Auto measures the global-header/ZCAL "
            "scan-datum mismatch without using intensities and tests a compact symmetric residual grid."
        ),
    )
    ap.add_argument("--absorption-mode-grid", default="none", help="Comma grid: none, mean, beam_dependent, offdiag_fraction, offdiag_tds, midgley_like.")
    ap.add_argument("--absorption-mu-grid", default="0.0", help="Uniform imaginary diagonal damping grid in inverse Angstrom.")
    ap.add_argument("--absorption-frac-grid", default="0.0", help="Imaginary absorptive off-diagonal fraction grid.")
    ap.add_argument("--absorption-mu-g2-grid", default="0.0", help="Beam-dependent diagonal damping slope grid versus |g|^2.")
    ap.add_argument("--absorption-mu-theta2-grid", default="0.0", help="Beam-dependent diagonal damping slope grid versus Bragg-angle^2.")
    ap.add_argument("--absorption-tds-b-grid", default="0.0", help="B_TDS grid for offdiag_tds eta*U_g*(1-exp(-B_TDS*|g|^2/4)).")
    ap.add_argument("--dose-b-mode-grid", default="none", help="Comma grid: none, linear_frame.")
    ap.add_argument("--dose-b-per-frame-grid", default="0.0", help="Extra isotropic B per XDS frame for dose_b_mode=linear_frame.")
    ap.add_argument("--dose-b-ref-frame", type=float, default=float("nan"), help="Reference ZCAL frame for dose-dependent B. Default uses the first frame in the data.")
    ap.add_argument("--row-resolution-b-grid", default="0.0", help="Raw-row dynamical resolution-envelope grid. Positive values boost high-resolution Idyn and are roughly the alternate sign of endpoint-B.")
    ap.add_argument(
        "--matrix-convention",
        choices=["j_minus_i", "i_minus_j"],
        default="j_minus_i",
        help="Off-diagonal U_g indexing convention. j_minus_i is the production signed-row default; i_minus_j is a diagnostic.",
    )
    ap.add_argument(
        "--q-control",
        choices=["real", "shuffle_global", "shuffle_shell", "shuffle_shell_scan"],
        default="real",
        help="Diagnostic null that permutes log(Idyn/Iborn) before normal scale fitting and endpoint export.",
    )
    ap.add_argument("--q-control-seed", type=int, default=12345)
    ap.add_argument("--q-control-shells", type=int, default=10, help="Resolution shells used by shell and shell+scan q-shuffle controls.")
    ap.add_argument(
        "--group-width-grid",
        default="auto",
        help="Comma grid of frame-group widths (the row table's group column). Values are degrees; also accepts frame, 2frame, 3frames, and auto.",
    )
    ap.add_argument("--sigma-a-grid", default="0.0,0.03,0.06")
    ap.add_argument("--sigma-b-corr-grid", default="0.0")
    ap.add_argument("--sigma-c-peak-grid", default="0.0")
    ap.add_argument("--wght-a-grid", default="", help="Native SHELX-style weight a grid. Empty uses the WGHT value parsed from the RES.")
    ap.add_argument("--wght-b-grid", default="", help="Native SHELX-style weight b grid. Empty uses the WGHT value parsed from the RES.")
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument(
        "--max-support-sg",
        type=float,
        default=max_support_sg,
        help="Maximum absolute excitation error for beam-basis support (default 0.01 A^-1).",
    )
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    add_split_scheme_args(ap)
    ap.add_argument("--scale-fit-split", choices=["work", "all"], default="work", help="Rows used to fit the P-spline frame scales. Use 'all' to reproduce final all-data model-update statistics; use 'work' for validation.")
    ap.add_argument(
        "--selection-metric",
        choices=["work_born_wR2", "work_wR2", "free_wR2", "all_wR2", "all_R1", "cv_Rall"],
        default="work_wR2",
        help=(
            "Metric used to choose the reported best nuisance trial. "
            "work_born_wR2 is reserved for hand-invariant transform scouting; "
            "cv_Rall is the inner orbit-CV R_all of the P-spline scale fit."
        ),
    )
    ap.add_argument(
        "--promotion-baseline-json",
        default="",
        help="Internal automatic-workflow contract: exact starting nuisance used for a post-ranking held-out no-harm gate.",
    )
    ap.add_argument("--promotion-min-gain", type=float, default=5e-5)
    ap.add_argument("--promotion-max-free-wr2-loss", type=float, default=0.002)
    ap.add_argument("--promotion-max-all-r1-loss", type=float, default=0.001)
    ap.add_argument(
        "--promotion-score-rows",
        choices=["all", "work"],
        default="all",
        help=(
            "Rows behind the promotion gate's composite score and R1 veto. all (default): all-row score plus "
            "0.25 free wR2 and the all-row R1 veto. work: work-row score (free weight 0) "
            "and the work-row R1 veto; with --promotion-max-free-wr2-loss inf the gate then reads no free row."
        ),
    )
    ap.add_argument("--fail-if-born-r1gt-above", type=float, default=0.0, help="Fail after the prediction grid if matched Born R1gt is above this value. Use as a basis/indexing sanity gate.")
    ap.add_argument(
        "--lambda-grid",
        default="1.0",
        help="Diagnostic q-exponent grid. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    add_scale_spline_arguments(ap)
    ap.add_argument("--endpoint-b-grid", default="0.0", help="Diagnostic endpoint resolution-scale grid in intensity B units: Icorr *= exp(B*(1/4d^2 - median(1/4d^2))). Nonzero values are not promoted over a neutral B=0 endpoint.")
    ap.add_argument("--endpoint-exti-grid", default="off", help="Comma grid of endpoint EXTI modes: off,on. Off removes EXTI from the corrected endpoint; on keeps/regularizes the source EXTI term.")
    ap.add_argument("--shelxl", default=shelxl, help="Explicit SHELXL executable used for corrected-HKL endpoints.")
    ap.add_argument("--shelxl-iterations", type=int, default=1, help="Number of SHELXL polish passes per corrected endpoint. Later passes feed .res back into .ins.")
    ap.add_argument("--endpoint-ls-cycles", type=int, default=1, help="Internal L.S./CGLS cycle count written into generated endpoint .ins/.res files. Production uses 1; larger values are diagnostic stress tests.")
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=["freeze-zero", "inherit", "promote", "promote-gold", "converge"],
        default="freeze-zero",
        help="SHELXL WGHT policy for corrected-HKL endpoints. freeze-zero is the production default after GROND sigma calibration; inherit keeps the source-model WGHT; promote keeps the source-model WGHT on the first call and uses SHELXL recommendations on later external passes; promote-gold starts at WGHT 0.1 0 and then promotes like promote; converge starts at WGHT 0 0 and promotes until the applied and suggested WGHT agree and max shift/esd < --shelxl-convergence-shift-esd (AUTO).",
    )
    ap.add_argument("--wght-converge-max-calls", type=int, default=20, help="Maximum SHELXL calls of one endpoint under --endpoint-wght-mode converge (replaces --shelxl-iterations there).")
    ap.add_argument("--zerr-esds", default="", help="Six cell e.s.d.s a,b,c,alpha,beta,gamma for the ZERR record of generated endpoint inputs (Z kept). Empty keeps the model's ZERR.")
    ap.add_argument("--strip-q-peaks", action="store_true", help="Drop difference-map peaks (Q1, Q2, ...) of a previous SHELXL run from generated endpoint inputs (AUTO).")
    ap.add_argument("--endpoint-list-code", type=int, default=0, help="LIST code of generated endpoint inputs (AUTO 6: an .fcf with phases for SHELXLE/Olex2 maps). 0 keeps the model's LIST.")
    ap.add_argument("--acta", action="store_true", help="Add ACTA to generated endpoint inputs so SHELXL writes a CIF.")
    ap.add_argument("--shelxl-convergence-shift-esd", type=float, default=0.0, help="Stop the SHELXL calls of an endpoint once the last L.S. cycle has |max shift/esd| below this value (AUTO 0.01). 0 disables the test (legacy).")
    ap.add_argument("--shelxl-npd-recovery", choices=["damp", "isor"], default="damp", help="Recovery for a non-positive-definite ADP step: legacy DAMP ladder, or ISOR on the NPD atoms only (AUTO).")
    ap.add_argument("--keep-damp", action="store_true", help="Legacy: carry an active DAMP instruction into later SHELXL calls and the written RES. Default strips it.")
    ap.add_argument("--endpoint-include-free-rows", action=argparse.BooleanOptionalAction, default=True, help="Write split=free rows into the corrected HKL (legacy default). AUTO passes --no-endpoint-include-free-rows.")
    ap.add_argument("--endpoint-q-policy", choices=["clip", "exclude"], default="clip", help="Rows whose correction ratio relative to the median lies outside (0.05, 20): clip q and keep them (legacy) or exclude and count them (AUTO).")
    ap.add_argument("--skip-endpoint-refinement", action="store_true", help="Run the fixed-model Bloch row calculation and row reports only. No corrected-HKL SHELXL endpoint is written or refined.")
    ap.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms", "error_model"], default="none", help="Endpoint uncertainty policy applied after Bloch correction and before SHELXL. uniform_z_rms fits one residual RMS(z) scalar from the current model and scales endpoint sigmas only; error_model fits sigma'^2 = k^2 (sigma^2 + (u P)^2) on this crystal's work rows against the current model (AUTO final endpoints).")
    ap.add_argument("--error-model-refits", type=int, default=1, help="error_model only: after SHELXL has refined the endpoint, refit the error model against the refined model (work rows only) and refine again, this many times.")
    ap.add_argument("--error-model-dataset-weight", choices=["apply", "off"], default="apply", help="Combined endpoints only (a dataset-policy weight exists only there); accepted so that one handoff option list serves every engine.")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20, help="Deterministic HKL holdout fraction reported for uniform_z_rms diagnostics.")
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25, help="Lower guardrail for the uniform_z_rms sigma multiplier.")
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0, help="Upper guardrail for the uniform_z_rms sigma multiplier.")
    ap.add_argument("--rescale", choices=["median-iobs", "none", "absolute"], default="median-iobs", help="Corrected-intensity scale: median-iobs (legacy), none, or absolute kappa*|F|^2 * Iobs/Icalc with the OSF reset to sqrt(kappa) (AUTO).")
    ap.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--prediction-cache-dir", default=str(runtime_cache_directory("native_row_regularize", "predictions")))
    ap.add_argument(
        "--ug-cache-dir",
        default=str(runtime_cache_directory("native_row_regularize", "ug")),
        help="Directory for model-dependent U_g cubes; use a run-local path on restricted systems.",
    )
    ap.add_argument("--cache-compression", choices=["compressed", "none"], default="compressed", help="Use none for faster local SSD caches; compressed saves disk on slow/limited storage.")
    ap.add_argument(
        "--row-plan-cache",
        choices=["auto", "off", "exact", "readwrite", "rebuild"],
        default="off",
        help=(
            "Row-geometry cache. auto enables the model-keyed exact cache only "
            "when enough fixed-model candidates share a geometry key to amortize "
            "its serial build; readwrite is the opt-in frozen-basis mode across "
            "model updates."
        ),
    )
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument(
        "--trial-rows-npz",
        default="",
        help=(
            "Also write every evaluated nuisance trial's raw row predictions with the row split, HKL, orbit key, "
            "target, sigma, frame group and scan angle to this .npz, for workflow-side held-out scoring."
        ),
    )
    ap.add_argument("--make-maps", action="store_true")
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true", help="Use alternate fractional xyz before building U_g for an alternate-hand control run.")
    ap.add_argument(
        "--paired-inverse-hand",
        action="store_true",
        help=(
            "Also calculate the elastic inverse hand by negative-thickness propagation from each "
            "already diagonalized Bloch matrix. Requires absorption off and the SciPy backend."
        ),
    )
    ap.add_argument(
        "--double",
        "--float64",
        dest="double_precision",
        action="store_true",
        default=False,
        help=(
            "Run the Bloch predictions in complex128/float64 double precision. Without it GROND_PRECISION decides "
            "(float64 or float32; default complex64 single precision). summary.json records the resolved "
            "precision under prediction_precision."
        ),
    )
    return ap


def add_scale_spline_arguments(ap: argparse.ArgumentParser) -> None:
    """P-spline frame-scale options of the native row engine.

    Every trial fits a smooth cubic P-spline in scan angle on the scale-fit rows, with n_basis and lambda chosen by
    inner 3-fold orbit CV over these grids.
    """

    ap.add_argument(
        "--scale-spline-n-basis-grid",
        default="32",
        help="P-spline basis sizes tried by the inner orbit CV; one value pins it.",
    )
    ap.add_argument(
        "--scale-spline-lambda-grid",
        default="0.3,1,3,10,30,100",
        help="P-spline second-difference penalties tried by the inner orbit CV; one value pins it.",
    )
