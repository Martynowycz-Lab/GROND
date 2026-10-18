"""AUTO frame-scale policy: P-spline scales in every native stage.

Every native stage fits smooth P-spline frame scales in scan angle on the scale-fit rows, with (n_basis, lambda)
chosen per crystal by inner orbit CV over --scale-spline-n-basis-grid and --scale-spline-lambda-grid.
"""

from __future__ import annotations

import argparse

from grond.regularize.contracts import scale_model_from_args


def scale_model_cli_arguments(args: argparse.Namespace) -> list[str]:
    """Native-engine options of the P-spline scale grids."""

    return [
        "--scale-spline-n-basis-grid",
        str(getattr(args, "scale_spline_n_basis_grid", "32")),
        "--scale-spline-lambda-grid",
        str(getattr(args, "scale_spline_lambda_grid", "0.3,1,3,10,30,100")),
    ]


def validate_scale_model_policy(args: argparse.Namespace) -> dict[str, object]:
    """Fail closed on P-spline grids the engine cannot use; return the run_config record."""

    contract = scale_model_from_args(args)
    return {
        "scale_model": "pspline",
        "scale_fit_split": "work",
        "contract": contract.to_dict(),
        "hyperparameters": "chosen per crystal and stage by inner 3-fold orbit CV over the scale-fit rows",
    }
