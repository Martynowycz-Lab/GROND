"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse

# AUTO beam-inclusion cutoff |Sg| in 1/A (the analogue of JANA2020's Sg(matrix) and diffBloch's sg_max, both
# 0.01); the engine default is the same.
AUTO_MAX_SUPPORT_SG = 0.01
# AUTO fixed rocking-window width in 1/A, used for rows without a DIALS-bbox header (XDS INTEGRATE.HKL, DIALS
# bridge files without the bbox columns).
AUTO_SG_WINDOW = 0.010


def add_profile_coordinates(ap: argparse.ArgumentParser, *, continuation: bool = False) -> None:
    """Expose the observation measure; continuations inherit saved values unless explicitly overridden."""

    ap.add_argument("--profile-coordinate", choices=["target_sg", "direct_angle"],
                    default=None if continuation else "target_sg")
    ap.add_argument("--profile-normalization", choices=["legacy", "target_sg_equivalent_v1"],
                    default=None if continuation else "legacy",
                    help="Explicit direct-angle measure. target_sg_equivalent_v1 preserves the target-Sg row convention.")
    ap.add_argument("--profile-angle-max-half-width-deg", type=float, default=None if continuation else 2.0,
                    help="Direct-angle bound; target_sg_equivalent_v1 refuses truncation at this bound.")
    ap.add_argument("--profile-basis-mode", choices=["center", "common_profile"],
                    default=None if continuation else "center")


def add_crystal_overrides(ap: argparse.ArgumentParser) -> None:
    """Use the same named controls for fresh runs and continuation."""

    ap.add_argument(
        "--dataset-n-basis", action="append", default=[], metavar="DATASET=N",
        help="Fix this crystal's beam limit (integer >= 2). Repeat per crystal.",
    )
    ap.add_argument(
        "--dataset-thickness-nm", action="append", default=[], metavar="DATASET=NM",
        help="Fix this crystal's thickness center in nm. Repeat per crystal; other thickness parameters are unchanged.",
    )


def add_backend(ap: argparse.ArgumentParser) -> None:
    """Add backend options in the established order."""

    ap.add_argument(
        "--prediction-cache-dir", default="", help="Directory for cached predictions (default OUT/cache)."
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
        "--shared-grid",
        choices=["off", "auto", "force"],
        default=None,
        help=(
            "off (default), auto or force. auto diagonalizes each matrix once per scan orientation and interpolates onto the rocking points of every reflection, when that is at least 3x cheaper. Used for SU-102."
        ),
    )
    ap.add_argument(
        "--basis-centring-coset",
        choices=["off", "on"],
        default=None,
        help="on: use only beams in the lattice-centring coset of the direct beam (exact). Used for SU-102.",
    )
    ap.add_argument(
        "--row-plan-cache",
        choices=["auto", "off", "exact", "readwrite", "rebuild"],
        default="auto",
        help=(
            "Reuse of per-observation geometry between stages: auto (default), off, exact, readwrite or rebuild."
        ),
    )
    ap.add_argument(
        "--matrix-convention",
        choices=["j_minus_i", "i_minus_j"],
        default="j_minus_i",
        help="Off-diagonal U_g indexing convention. Keep j_minus_i for production; i_minus_j is diagnostic.",
    )


def add_basis(ap: argparse.ArgumentParser) -> None:
    """Add basis options in the established order."""

    add_crystal_overrides(ap)
    ap.add_argument(
        "--crystal-parameters-from", default="", metavar="CHECKPOINT",
        help="Fresh run: copy per-crystal beam/rocking settings and thickness distributions from a completed run's checkpoint. Scout indexing and fit other nuisance terms again; do not copy coordinates or transforms.",
    )
    ap.add_argument(
        "--screen-n-basis", type=int, default=24, help="Basis size for initial nuisance/geometry screening."
    )
    ap.add_argument(
        "--screen-n-points",
        type=int,
        default=None,
        help=(
            "Quadrature points for screening, thickness and feature-gate stages. Default derives it per crystal from "
            "the rocking window as half the final count (fixed 0.010: 25; DIALS bbox: 30). Legacy: 10."
        ),
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
        default=None,
        help=(
            "Rocking points per observation in the final rounds (default: 60 for DIALS, 50 for XDS)."
        ),
    )
    ap.add_argument(
        "--final-basis-scan-limit",
        type=int,
        default=500,
        help="Largest beam count considered when the final beam count is chosen (default %(default)s).",
    )
    ap.add_argument(
        "--final-basis-margin",
        type=int,
        default=10,
        help="Extra beams above the observed maximum support for final polish.",
    )
    ap.add_argument("--final-basis-min", type=int, default=20)
    ap.add_argument("--final-basis-max", type=int, default=500, help="Most beams per observation in the final rounds (default %(default)s).")
    ap.add_argument("--final-basis-fallback", type=int, default=128, help="Dry-run fallback when --n-basis=0.")


def add_geometry(ap: argparse.ArgumentParser) -> None:
    """Add geometry options in the established order."""

    ap.add_argument(
        "--sg-window",
        type=float,
        default=AUTO_SG_WINDOW,
        help=(
            "Fixed rocking-window width in 1/A (default 0.010), used when rows carry no DIALS bbox; also the "
            "reference width of adaptive windows."
        ),
    )
    ap.add_argument(
        "--sg-window-mode",
        choices=["auto", "fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"],
        default="auto",
        help=(
            "Rocking-width source. Auto uses the per-row DIALS bbox z-range for rows from GROND's DIALS bridge "
            "(header DIALS_BBOX_Z_RANGE_AS_ALF0_BET0=TRUE) and the fixed Sg width otherwise. Legacy: fixed."
        ),
    )
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument(
        "--sg-window-max",
        type=float,
        default=None,
        help="Upper bound of per-row windows. Default 0.03 1/A for the DIALS bbox window and 0.012 otherwise. Legacy: 0.012.",
    )
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
        help="Comma-separated fixed Sg widths tested by the late rocking-window scout; the run's own width is always added.",
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
        help="Minimum wR2 gain required before a non-baseline Sg width can be promoted (all rows; work rows under --nuisance-score-rows work).",
    )
    ap.add_argument(
        "--sg-window-scout-free-wR2-tolerance",
        type=float,
        default=0.003,
        help="Maximum held-out wR2 penalty allowed when promoting a non-baseline Sg width (--nuisance-score-rows legacy only; work uses no free-row penalty).",
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
        default=AUTO_MAX_SUPPORT_SG,
        help=(
            "Beams with |S_g| up to this value (1/Å) at the observation centre enter the calculation (default %(default)s)."
        ),
    )
    add_profile_coordinates(ap)
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
    ap.add_argument(
        "--absorption-mode-grid",
        default="none",
        help="Absorption feature-gate grid. Default none (no absorption). Legacy: none,midgley_like.",
    )
    ap.add_argument("--absorption-mu-grid", default="0.0", help="Legacy: 0.0,0.00005.")
    ap.add_argument("--absorption-frac-grid", default="0.0", help="Legacy: 0.0,0.02.")
    ap.add_argument("--absorption-mu-g2-grid", default="0.0")
    ap.add_argument("--absorption-mu-theta2-grid", default="0.0")
    ap.add_argument("--absorption-tds-b-grid", default="0.0")
    ap.add_argument(
        "--dose-b-mode-grid",
        default="none,linear_frame",
        help="Dose-B feature-gate modes. Legacy: none (with --dose-b-per-frame-grid 0.0).",
    )
    ap.add_argument(
        "--dose-b-total-grid",
        default="0,0.5,1,2,3",
        help=(
            "Dose-B grid in A^2 accumulated over the whole scan; each crystal converts it to B per frame with "
            "its own ZCAL span. Ignored when --dose-b-per-frame-grid is given."
        ),
    )
    ap.add_argument(
        "--dose-b-per-frame-grid",
        default="",
        help="Explicit dose-B grid in A^2 per XDS frame (legacy units); overrides --dose-b-total-grid. Legacy: 0.0.",
    )
    ap.add_argument(
        "--mosaicity-mode",
        choices=["none", "transverse_target", "gaussian_cone"],
        default="transverse_target",
        help=(
            "Mosaic model: transverse_target (default; Gaussian tilt about the axis normal to the beam and the rotation axis) or none."
        ),
    )
    ap.add_argument(
        "--mosaicity-sigma-grid",
        default="0,0.1,0.2,0.3",
        help="Mosaic cone widths in degrees tested per crystal; 0 is the no-cone candidate.",
    )
    ap.add_argument(
        "--mosaicity-points", type=int, default=3, help="Gauss-Hermite tilt points of the mosaic cone quadrature."
    )
    ap.add_argument("--mosaicity-support-sigma-deg", type=float, default=0.0,
                    help="Declared largest two-axis cone width; must cover every tested sigma.")
    ap.add_argument("--mosaicity-support-points", type=int, default=5,
                    help="Points per axis in the common beam-support envelope.")
    ap.add_argument("--mosaicity-support-width-sigma", type=float, default=4.0,
                    help="Half-width of the common beam-support envelope in support-sigma units.")
    ap.add_argument(
        "--physics-gate-selection",
        choices=["orbit-cv", "in-sample"],
        default="orbit-cv",
        help=(
            "Score of the mosaicity and dose feature gates. orbit-cv compares every candidate on 3-fold "
            "orbit-grouped cross-validation over work rows (free rows unused) and accepts only a paired "
            "orbit-bootstrap improvement; in-sample is the legacy score of the other feature gates."
        ),
    )
    ap.add_argument(
        "--thickness-gate-split",
        choices=["matched", "legacy"],
        default="matched",
        help=(
            "Scale-fit split of the in-sample feature gates. matched fits every gate on its baseline's split, so "
            "each comparison uses one split; legacy follows --production-scales."
        ),
    )
    ap.add_argument(
        "--nuisance-selection",
        choices=["nested", "sequential"],
        default="nested",
        help=(
            "nested: per crystal, after the screen and the early hand decision, select mosaic cone x dose B x thickness "
            "distribution (--nested-families) x P-spline lambda jointly by inner orbit CV on work rows from Bloch "
            "spectra, and pin the pick for every later stage (the cone and dose gates do not run). sequential: the "
            "discrete/continuous thickness polish followed by the cone and dose gates. Legacy: sequential."
        ),
    )
    ap.add_argument(
        "--nested-n-basis",
        type=int,
        default=64,
        help=(
            "Most beams per observation while the settings are chosen (default %(default)s)."
        ),
    )
    ap.add_argument(
        "--nested-n-points",
        default="final",
        help=(
            "Rocking points while the settings are chosen: final (default), screen or a number."
        ),
    )
    ap.add_argument(
        "--nested-families",
        default="uniform_gradient,gamma",
        help="Thickness-distribution families of the nested selection: uniform_gradient (t0, gradient) and/or gamma (mean, shape).",
    )
    ap.add_argument(
        "--nested-response-tables",
        action="store_true",
        help=(
            "Keep only the thickness tables while the settings are chosen (less memory; needed with --shared-grid and for large cells)."
        ),
    )
    ap.add_argument(
        "--feedback-thickness-update",
        choices=["on", "off"],
        default=None,
        help=(
            "Re-select the thickness distribution and lambda (physics variant fixed) on the spectra of each new "
            "structure before every SHELXL macro-cycle and for the final model. Default: on with "
            "--nuisance-selection nested, not applicable with sequential. Legacy: off."
        ),
    )
    ap.add_argument(
        "--thickness-points", type=int, default=5, help="Thickness quadrature points for screen/regularization stages."
    )
    ap.add_argument(
        "--final-thickness-points",
        type=int,
        default=9,
        help="Thickness quadrature points for pinned final-basis polish runs.",
    )
