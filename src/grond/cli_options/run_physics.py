"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse

# AUTO beam-inclusion cutoff |Sg| in 1/A (the analogue of JANA2020's Sg(matrix) and diffBloch's sg_max, both
# 0.01); the engine default is the same.
AUTO_MAX_SUPPORT_SG = 0.01
# AUTO fixed rocking-window width in 1/A, used for rows without a DIALS-bbox header (XDS INTEGRATE.HKL, DIALS
# bridge files without the bbox columns).
AUTO_SG_WINDOW = 0.010


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
        "--shared-grid-gpus",
        default=None,
        help="CUDA devices for the shared grid's prediction passes, e.g. 0,1,2 (needs JAX with CUDA). Used only when the "
        "grid matrices have at least 2,000 beams; smaller ones solve faster on CPU cores.",
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
        help="Beams per observation in the final rounds. Default 0 scans Sg support and uses the maximum support plus margin.",
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
        help="Extra beams above the observed maximum support in the final rounds.",
    )
    ap.add_argument("--final-basis-min", type=int, default=20)
    ap.add_argument("--final-basis-max", type=int, default=500, help="Most beams per observation in the final rounds (default %(default)s).")


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
        choices=["auto", "fixed", "dials_bbox"],
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
        "--topup-d-min",
        type=float,
        default=0.8,
        help="Beam candidates are the reflections with d at least the finer of this value and the data's d_min, in Å "
        "(default %(default)s).",
    )
    ap.add_argument(
        "--max-support-sg",
        type=float,
        default=AUTO_MAX_SUPPORT_SG,
        help=(
            "Beams with |S_g| up to this value (1/Å) at the observation centre enter the calculation (default %(default)s)."
        ),
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

    ap.add_argument("--thickness-grid", default="1000,1200,1400")
    ap.add_argument("--thickness-grad-grid", default="0.4,1.0")
    ap.add_argument("--sg-jitter-points", type=int, default=3)
    ap.add_argument(
        "--dose-b-total-grid",
        default="0,0.5,1,2,3",
        help=(
            "Dose-B grid in A^2 accumulated over the whole scan; each crystal converts it to B per frame with "
            "its own ZCAL span."
        ),
    )
    ap.add_argument(
        "--mosaicity-mode",
        choices=["none", "transverse_target"],
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
    ap.add_argument(
        "--nuisance-selection",
        choices=["nested", "sequential"],
        default="nested",
        help=(
            "nested: per crystal, after the screen and the early hand decision, select mosaic cone x dose B x thickness "
            "distribution (uniform_gradient or gamma) x P-spline lambda jointly by inner orbit CV on work rows from Bloch "
            "spectra, and pin the pick for every later stage (the cone and dose gates do not run). sequential: the "
            "discrete thickness polish followed by the cone and dose gates. Legacy: sequential."
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
