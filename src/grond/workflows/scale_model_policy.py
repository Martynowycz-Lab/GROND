"""AUTO frame-scale model policy: stage forwarding and the post-hoc scale stages a P-spline run turns off.

``--scale-model pspline`` (the AUTO default) fits smooth P-spline frame scales on the scale-fit rows
inside every native stage, with (n_basis, lambda) chosen per crystal by inner orbit CV. The post-hoc
multi-dataset compact scale-contract selection would replace the P-spline model, so it resolves to off.
``--scale-model per-frame`` fits free per-frame scales.
"""

from __future__ import annotations

import argparse

from grond.regularize.contracts import SCALE_MODELS, parse_int_grid, scale_model_from_args

PSPLINE_OFF_REASON = "scale_model_pspline"


def resolved_scale_model(args: argparse.Namespace) -> str:
    """Return the requested frame-scale family; legacy namespaces without the option are per-frame."""

    kind = str(getattr(args, "scale_model", "per-frame") or "per-frame")
    if kind not in SCALE_MODELS:
        raise ValueError(f"unknown --scale-model {kind!r}; choices are {', '.join(SCALE_MODELS)}")
    return kind


def scale_model_cli_arguments(args: argparse.Namespace) -> list[str]:
    """Native-engine options for the scale model; empty for per-frame so legacy commands stay identical."""

    if resolved_scale_model(args) == "per-frame":
        return []
    return [
        "--scale-model",
        "pspline",
        "--scale-spline-n-basis-grid",
        str(getattr(args, "scale_spline_n_basis_grid", "32")),
        "--scale-spline-lambda-grid",
        str(getattr(args, "scale_spline_lambda_grid", "0.3,1,3,10,30,100")),
    ]


def resolved_multi_scale_contract_selection(args: argparse.Namespace) -> tuple[str, str]:
    """(mode, reason) for the multi-dataset compact scale-contract (BIC) selection."""

    mode = str(getattr(args, "multi_scale_contract_selection", "auto"))
    if resolved_scale_model(args) != "pspline":
        return mode, ""
    return "off", PSPLINE_OFF_REASON


def validate_scale_model_policy(args: argparse.Namespace) -> dict[str, object]:
    """Fail closed on combinations a P-spline run cannot honour; return the run_config record."""

    kind = resolved_scale_model(args)
    record: dict[str, object] = {
        "scale_model": kind,
        "scale_fit_split": "all" if bool(getattr(args, "production_scales", False)) else "work",
    }
    if kind != "pspline":
        return record
    contract = scale_model_from_args(args)
    if any(value > 1 for value in parse_int_grid(str(getattr(args, "detector_scale_bins", "0")))):
        raise ValueError("--scale-model pspline has no detector field; use --detector-scale-bins 0 or --scale-model per-frame")
    if abs(float(getattr(args, "scale_smoothness", 0.0))) > 0.0:
        raise ValueError("--scale-smoothness applies only to per-frame scales; set it to 0 with --scale-model pspline")
    record.update(
        {
            "contract": contract.to_dict(),
            "hyperparameters": "chosen per crystal and stage by inner 3-fold orbit CV over the scale-fit rows",
            "multi_scale_contract_selection": resolved_multi_scale_contract_selection(args)[0],
            "off_reason": PSPLINE_OFF_REASON,
        }
    )
    return record
