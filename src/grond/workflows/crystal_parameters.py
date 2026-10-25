"""Resolve per-crystal continuation settings before starting any calculations."""

from __future__ import annotations

import argparse
import copy
import math
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.values import try_read_json
from grond.io.config import DatasetSpec
from grond.workflows.checkpoints import _checkpoint_path

# These control the numerical prediction, not the structural model or its scale fit.
PREDICTION_FIELDS = (
    "n_basis",
    "n_points",
    "thickness_points",
    "topup_d_min",
    "max_support_sg",
    "sg_window",
    "sg_window_mode",
    "sg_window_min",
    "sg_window_max",
    "sg_window_scale",
    "sg_window_frame_floor",
    "profile_quadrature",
    "profile_accumulation",
    "mosaicity_points",
)


def dataset_values(specifications: list[str], names: set[str], option: str, *, integer: bool = False) -> dict:
    """Reject misspelled, repeated and invalid overrides instead of ignoring them."""

    values = {}
    for specification in specifications:
        name, separator, raw = specification.partition("=")
        name, raw = name.strip(), raw.strip()
        if not separator or name not in names:
            raise SystemExit(
                f"{option}: expected DATASET=VALUE with a dataset name; got {specification!r}. "
                f"Available datasets: {', '.join(sorted(names))}"
            )
        if name in values:
            raise SystemExit(f"{option}: dataset {name!r} was specified more than once.")
        try:
            value = int(raw) if integer else float(raw)
        except (ValueError, OverflowError):
            value = float("nan")
        if (
            (not integer and not math.isfinite(value))
            or value < (2 if integer else 0)
            or (not integer and value == 0)
            or (integer and not isinstance(value, int))
        ):
            units = "an integer of at least 2 beams" if integer else "a finite positive thickness in nm"
            raise SystemExit(f"{option}: {name!r} requires {units}; got {raw!r}.")
        values[name] = value
    return values


def _saved_integer(value: object, name: str, field: str, minimum: int) -> int:
    """Return a saved checkpoint setting as an int; exit when it is not an integer >= ``minimum``.

    ``name`` (the dataset) and ``field`` (the setting) only label the error message.
    """

    try:
        result = int(value)
        valid = not isinstance(value, bool) and float(value) == result and result >= minimum
    except (TypeError, ValueError, OverflowError):
        valid = False
    if not valid:
        raise SystemExit(
            f"Checkpoint dataset {name!r} has invalid {field}={value!r}; expected an integer >= {minimum}."
        )
    return result


def resolve_crystal_parameters(
    checkpoint: Mapping,
    datasets: list[DatasetSpec],
    nuisance: dict[str, dict],
    args: argparse.Namespace,
    *,
    checkpoint_dir: Path,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Restore each crystal's prediction contract; explicit overrides take precedence."""

    names = {ds.name for ds in datasets}
    if len(names) != len(datasets):
        raise SystemExit("Continuation checkpoint contains duplicate dataset names.")
    beams = dataset_values(args.dataset_n_basis, names, "--dataset-n-basis", integer=True)
    thickness = dataset_values(args.dataset_thickness_nm, names, "--dataset-thickness-nm")
    rows = {row["name"]: row for row in checkpoint["datasets"]}
    parameters = {}
    pinned = copy.deepcopy(nuisance)
    for ds in datasets:
        row = rows[ds.name]
        source = _checkpoint_path(row.get("source_summary"), base=checkpoint_dir)
        summary = try_read_json(Path(source)) if source else {}
        config = summary.get("config") or {}
        saved = row.get("final_config") or {}
        if not isinstance(config, Mapping) or not isinstance(saved, Mapping):
            raise SystemExit(f"Checkpoint dataset {ds.name!r} has a malformed prediction config.")
        # Older checkpoints omitted quadrature and rocking settings. Recover them
        # from that crystal's source summary, not from another crystal or a default.
        config = {**config, **saved}
        settings = {key: config[key] for key in PREDICTION_FIELDS if config.get(key) is not None}
        for key in ("profile_quadrature", "profile_accumulation"):
            override = getattr(args, key, None)
            if override is not None:
                settings[key] = override
        defaulted = [key for key in PREDICTION_FIELDS if key not in settings]
        if args.n_points is not None and "n_points" in defaulted:
            defaulted.remove("n_points")
        if args.topup_d_min is not None and "topup_d_min" in defaulted:
            defaulted.remove("topup_d_min")
        # Beam count never falls back silently: it must be saved or explicit.
        if "n_basis" in defaulted:
            defaulted.remove("n_basis")
        basis = beams.get(ds.name, args.n_basis if args.n_basis is not None else settings.get("n_basis"))
        if basis is None:
            raise SystemExit(
                f"Checkpoint dataset {ds.name!r} has no saved n_basis. "
                f"Supply --dataset-n-basis {ds.name}=N or --n-basis N."
            )
        settings["n_basis"] = _saved_integer(basis, ds.name, "n_basis", 2)
        points = args.n_points if args.n_points is not None else settings.get("n_points")
        if points is None:
            raise SystemExit(f"Checkpoint dataset {ds.name!r} has no saved n_points. Supply --n-points N.")
        settings["n_points"] = _saved_integer(points, ds.name, "n_points", 1)
        if args.topup_d_min is not None:
            if not math.isfinite(args.topup_d_min) or args.topup_d_min <= 0:
                raise SystemExit("--topup-d-min must be finite and positive.")
            settings["topup_d_min"] = args.topup_d_min
        if "thickness_points" in settings:
            settings["thickness_points"] = _saved_integer(settings["thickness_points"], ds.name, "thickness_points", 1)
            settings["final_thickness_points"] = settings["thickness_points"]
        if ds.name in thickness:
            pinned[ds.name]["thickness_A"] = 10.0 * thickness[ds.name]
        center = pinned[ds.name].get("thickness_A")
        try:
            valid_center = not isinstance(center, bool) and math.isfinite(float(center)) and float(center) > 0
        except (TypeError, ValueError, OverflowError):
            valid_center = False
        if not valid_center:
            raise SystemExit(
                f"Checkpoint dataset {ds.name!r} has invalid thickness_A={center!r}. "
                f"Supply --dataset-thickness-nm {ds.name}=NM."
            )
        pinned[ds.name]["thickness_A"] = float(center)
        parameters[ds.name] = {
            "prediction": settings,
            "thickness_A": float(center),
            "thickness_nm": float(center) / 10.0,
            "n_basis_source": "dataset_override"
            if ds.name in beams
            else "global_override"
            if args.n_basis is not None
            else "checkpoint",
            "thickness_source": "dataset_override" if ds.name in thickness else "checkpoint",
            "source_summary": source,
            "defaulted_prediction_fields": defaulted,
        }
    return parameters, pinned


def crystal_arguments(args: argparse.Namespace, name: str) -> argparse.Namespace:
    """Copy shared options so one crystal cannot change another crystal's settings."""

    record = getattr(args, "crystal_parameters", {}).get(name, {})
    settings = record.get("prediction", {})
    if not record:
        return args
    resolved = copy.copy(args)
    # Window-derived values of configure_auto_physics (n_points, screen_n_points, sg_window_max) apply to
    # this crystal only and are never fixed: each stage still resolves its own screen or final Q.
    # Explicit or reference prediction values take precedence.
    for key, value in record.get("auto_physics", {}).items():
        if key not in settings:
            setattr(resolved, key, value)
    for key, value in settings.items():
        setattr(resolved, key, value)
    return resolved
