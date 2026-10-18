"""Register options shared by automatic and expert run commands."""

from __future__ import annotations

import argparse

from grond.cli_support.tools import add_tool_config_args
from grond.cli_support.values import XDS_TARGET_MODES


def add_inputs(ap: argparse.ArgumentParser) -> None:
    """Add inputs options in the established order."""

    ap.add_argument("--prep", default="", help="Input file written by `grond prep`.")
    ap.add_argument("--hkl", default="")
    ap.add_argument("--xds-dir", default="")
    ap.add_argument(
        "--dials-refl",
        default="",
        help="DIALS integrated.refl/scaled.refl file. Auto-converted into a GROND row directory under OUT/prepared.",
    )
    ap.add_argument("--dials-expt", default="", help="DIALS integrated.expt/scaled.expt file matching --dials-refl.")
    ap.add_argument(
        "--dials-python",
        default="",
        help="Python executable that can import dials and dxtbx. Leave blank when the active Python already can.",
    )
    ap.add_argument(
        "--dials-intensity-mode",
        choices=["sum_complete", "sum_raw", "scale"],
        default="sum_complete",
        help="DIALS intensity mapping for the GROND row contract.",
    )
    ap.add_argument(
        "--dials-row-selection",
        choices=["integrated", "scaled"],
        default="integrated",
        help="Use all integrated rows or only rows accepted by DIALS scaling, independently of the selected intensity values.",
    )
    ap.add_argument(
        "--dials-sigma-mode",
        choices=["auto", "raw", "detector_gain"],
        default="raw",
        help="DIALS sigma bridge policy. raw keeps the detector-model variance; auto empirical calibration and detector_gain scaling are explicit diagnostics.",
    )
    ap.add_argument("--structure", default="")
    ap.add_argument("--baseline-hkl", default="")
    ap.add_argument("--dataset-name", default="dataset")
    ap.add_argument("--out-dir", required=True, help="Output directory. Run the same command again to resume.")


def add_execution(ap: argparse.ArgumentParser) -> None:
    """Add execution options in the established order."""

    ap.add_argument(
        "--preset",
        choices=["standard", "fast"],
        default="standard",
        help=(
            "standard (default) or fast: fewer rocking points, beams and candidate settings, every check kept."
        ),
    )
    ap.add_argument("--python", default="", help="Python for the calculation subprocesses (default: the one running grond).")
    add_tool_config_args(ap)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument(
        "--resume",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse the stages already finished in --out-dir (default on).",
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=0,
        help="Worker processes (default 0: 85%% of the cores).",
    )
    ap.add_argument(
        "--max-core-fraction", type=float, default=0.85, help="Fraction of detected CPU cores to use when --workers=0."
    )
    ap.add_argument(
        "--chunksize",
        type=int,
        default=0,
        help="Observations per worker task (default 0: chosen from --workers).",
    )


def add_observations_and_outputs(ap: argparse.ArgumentParser) -> None:
    """Add observations and outputs options in the established order."""

    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--min-d-spacing", type=float, default=0.70, help="Resolution limit of the observations in Å (default %(default)s).")
    ap.add_argument("--min-peak", type=float, default=90.0, help="Minimum XDS PEAK, or 100 x DIALS partiality, of an observation (default %(default)s).")
    ap.add_argument("--geometry-percentile", type=float, default=99.0)
    ap.add_argument("--target-mode", choices=list(XDS_TARGET_MODES), default="iobs", help=argparse.SUPPRESS)
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument(
        "--make-maps",
        dest="make_maps",
        action="store_true",
        default=False,
        help="Also write SHELXL Fo-Fc maps with shelx2map at the end of the run.",
    )
    ap.add_argument("--no-maps", dest="make_maps", action="store_false")
    ap.add_argument(
        "--h-omit",
        choices=["off", "on", "only", "both"],
        default="off",
        help=(
            "Hydrogen-omit maps of the final model of one crystal: off (default) or on."
        ),
    )
    ap.add_argument(
        "--hand-control",
        dest="hand_control",
        action="store_true",
        default=True,
        help="Run one final elastic negative-thickness hand decision when model symmetry makes it meaningful.",
    )
    ap.add_argument(
        "--no-hand-control", dest="hand_control", action="store_false", help="Skip the alternate-hand diagnostic."
    )
