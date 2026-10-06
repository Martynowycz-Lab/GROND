"""AUTO frame-scale model policy: stage forwarding and the post-hoc scale stages a P-spline run turns off.

``--scale-model pspline`` (the AUTO default) fits smooth P-spline frame scales on the scale-fit rows
inside every native stage, with (n_basis, lambda) chosen per crystal by inner orbit CV. The post-hoc
per-frame compressions (scale-window polish, the multi-dataset compact scale-contract selection and the
single-dataset endpoint-calibration scale replays) refit free per-frame, window or adaptive scales from
the frozen row table and would replace the P-spline model, so they resolve to off. An explicit ``on``
request for one of them cannot be honoured and raises before any calculation starts.
``--scale-model per-frame`` fits free per-frame scales and leaves these compressions to their own settings.
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


def resolved_scale_polish_mode(args: argparse.Namespace) -> tuple[str, str]:
    """(mode, reason) for the post-hoc scale-window polish."""

    mode = str(getattr(args, "scale_polish", "auto"))
    if resolved_scale_model(args) != "pspline":
        return mode, ""
    if mode == "on":
        raise ValueError(
            "--scale-polish on refits compact per-frame scales and would replace the P-spline scale model; "
            "use --scale-model per-frame for the legacy scale compression"
        )
    return "off", PSPLINE_OFF_REASON


def resolved_multi_scale_contract_selection(args: argparse.Namespace) -> tuple[str, str]:
    """(mode, reason) for the multi-dataset compact scale-contract (BIC) selection."""

    mode = str(getattr(args, "multi_scale_contract_selection", "auto"))
    if resolved_scale_model(args) != "pspline":
        return mode, ""
    return "off", PSPLINE_OFF_REASON


def resolved_endpoint_calibration_mode(args: argparse.Namespace) -> tuple[str, str]:
    """(mode, reason) for the single-dataset endpoint-calibration scale replays."""

    mode = str(getattr(args, "endpoint_calibration", "auto"))
    if resolved_scale_model(args) != "pspline":
        return mode, ""
    if mode == "on":
        raise ValueError(
            "--endpoint-calibration on replays compact per-frame/window scale contracts and would replace the "
            "P-spline scale model; use --scale-model per-frame for the legacy calibration replay"
        )
    return "off", PSPLINE_OFF_REASON


def native_gradient_strategy_reason(args: argparse.Namespace) -> str:
    """The option that made resolve_structure_update_strategy choose the native_gradient update."""

    if str(getattr(args, "structure_update", "auto")) == "native_gradient":
        return "--structure-update native_gradient"
    if bool(getattr(args, "full_fast", False)):
        return "--full-fast (it selects the native_gradient update unless --structure-update is given)"
    if bool(getattr(args, "native_only", False)):
        return "--native-only (it selects the native_gradient update unless --structure-update is given)"
    return f"--structure-update auto with no usable SHELXL executable ({str(getattr(args, 'shelxl', 'shelxl'))!r})"


def native_gradient_startup_conflicts(args: argparse.Namespace, *, multi_dataset: bool, prep=None) -> list[str]:
    """Settings the native_gradient structure update cannot honour, each with the option that removes it.

    The structure engines fit their own per-frame scales and do not model the mosaic cone; the single-dataset
    engine has no dose B, and native_multi_structure_update applies one rocking window to every crystal.
    Without this check the refusal comes after the screen, thickness and every physics gate.
    """

    from grond.workflows.nuisance_policy import feature_is_enabled, structure_update_window_options

    conflicts = []
    if resolved_scale_model(args) == "pspline":
        conflicts.append(
            "--scale-model pspline (the native_gradient structure update fits its own per-frame scales and cannot "
            "honour --scale-model pspline; use --scale-model per-frame)"
        )
    if feature_is_enabled(args, "mosaicity"):
        conflicts.append(
            f"the mosaicity gate (--mosaicity-mode {getattr(args, 'mosaicity_mode', 'none')}: the structure update "
            "does not model the mosaic cone; use --mosaicity-mode none)"
        )
    if not multi_dataset and feature_is_enabled(args, "dose"):
        conflicts.append(
            f"the dose-B gate (--dose-b-mode-grid {getattr(args, 'dose_b_mode_grid', '')}: the single-dataset "
            "structure update has no dose B; use --dose-b-mode-grid none)"
        )
    if multi_dataset and prep is not None:
        windows = structure_update_window_options(prep, args)
        if len(set(windows.values())) > 1:
            described = "; ".join(f"{name}: {' '.join(option)}" for name, option in windows.items())
            conflicts.append(
                "crystals that resolve to different rocking windows (native_multi_structure_update applies one "
                f"window to every crystal; {described}; set one window explicitly, e.g. --sg-window-mode fixed "
                "with --sg-window and --sg-window-max)"
            )
    return conflicts


def refuse_native_gradient_startup_conflicts(args: argparse.Namespace, *, multi_dataset: bool, prep=None) -> None:
    """Refuse at startup, before any stage, a native_gradient run whose structure update would be refused."""

    from grond.workflows.model_selection import resolve_structure_update_strategy

    if (
        str(getattr(args, "mode", "auto")) not in {"regularize", "auto"}
        or bool(getattr(args, "fast", False))
        or resolve_structure_update_strategy(args, multi_dataset=multi_dataset) != "native_gradient"
    ):
        return
    conflicts = native_gradient_startup_conflicts(args, multi_dataset=multi_dataset, prep=prep)
    if conflicts:
        raise ValueError(
            f"the native_gradient structure update was chosen by {native_gradient_strategy_reason(args)} and "
            f"cannot run with: {'; '.join(conflicts)}. Either use --structure-update shelxl_feedback with a SHELXL "
            "executable, or remove each setting as stated"
        )


def validate_scale_model_policy(args: argparse.Namespace, *, multi_dataset: bool, prep=None) -> dict[str, object]:
    """Fail closed on combinations a P-spline or native-gradient run cannot honour; return the run_config record.

    ``prep`` (the run's datasets) lets the native-gradient check compare the crystals' rocking windows.
    """

    refuse_native_gradient_startup_conflicts(args, multi_dataset=multi_dataset, prep=prep)
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
            "scale_polish": resolved_scale_polish_mode(args)[0],
            "multi_scale_contract_selection": resolved_multi_scale_contract_selection(args)[0],
            "endpoint_calibration": resolved_endpoint_calibration_mode(args)[0],
            "off_reason": PSPLINE_OFF_REASON,
        }
    )
    return record
