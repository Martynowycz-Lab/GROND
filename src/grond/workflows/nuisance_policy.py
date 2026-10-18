"""Resolve basis sizes, fidelity profiles and nuisance defaults before building stages."""

from __future__ import annotations

import argparse
import functools
import json
import math
from pathlib import Path

from grond.cli_support.values import read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.paths import engine_script
from grond.workflows.crystal_parameters import crystal_arguments
from grond.workflows.decision import safe_label

# ----------------------------------------------------------------------------- AUTO physics policy
# Every value below is an AUTO default only; the engine Config and the native engine CLI keep their own
# defaults.

# Rocking-window mode for rows from GROND's DIALS bridge (header DIALS_BBOX_Z_RANGE_AS_ALF0_BET0=TRUE).
# Everything DIALS-specific derives from it.
AUTO_DIALS_WINDOW_MODE = "dials_bbox"
# Final-fidelity profile spacing of the fixed window in 1/A (the AUTO default 0.010 1/A window gets 50 final
# and 25 screen points).
AUTO_FIXED_PROFILE_SPACING = 0.0002
# Per-row DIALS-bbox window cap in 1/A and its final point count.
AUTO_DIALS_BBOX_WINDOW_MAX = 0.03
AUTO_DIALS_BBOX_N_POINTS = 60
# Window cap for every other mode.
LEGACY_SG_WINDOW_MAX = 0.012
# Engine bound on the mosaic width (Config.mosaicity_max_sigma_deg default).
MOSAICITY_MAX_SIGMA_DEG = 0.5
# Sequential per-crystal feature gates.  The mosaic cone precedes the Sg-jitter proxy because the
# two profile broadenings exclude each other and the cone is the geometrically defined one.
FEATURE_GATE_FAMILIES = ("row_resolution", "absorption", "mosaicity", "sg_jitter", "sigma_error", "dose")


def error_model_sigma_active(args: argparse.Namespace) -> bool:
    """Whether the per-crystal error model sets the SHELXL/merge sigmas (--dataset-sigma-policy error_model)."""

    return str(getattr(args, "dataset_sigma_policy", "auto")) == "error_model"


def resolved_post_bloch_sigma_policy(
    args: argparse.Namespace, *, final_endpoint: bool = False, feedback_loop: bool = False, combined: bool = False
) -> str:
    """Resolve the endpoint sigma calibration policy for a workflow stage.

    Under the error model auto gives every final or feedback per-crystal endpoint
    its own error model and no second rescale of the merged combined HKL (``combined``), whose rows already
    carry per-crystal error-model sigmas.  Otherwise auto resolves to uniform_z_rms.
    """

    requested = getattr(args, "post_bloch_sigma_policy", "auto")
    if requested == "auto":
        if not (final_endpoint or feedback_loop):
            return "none"
        if error_model_sigma_active(args):
            return "none" if combined else "error_model"
        return "uniform_z_rms"
    return str(requested)


def resolved_dataset_sigma_policy(args: argparse.Namespace, *, multi_dataset: bool = False) -> str:
    """Resolve per-dataset endpoint sigma calibration for combination stages."""

    requested = getattr(args, "dataset_sigma_policy", "auto")
    if requested == "auto":
        return "uniform_z_rms" if multi_dataset else "none"
    return str(requested)


def resolved_feedback_wght_mode(args: argparse.Namespace) -> str:
    """SHELXL weighting policy of the structure-feedback macro-cycles (``endpoint`` follows the endpoints)."""

    requested = str(getattr(args, "shelxl_feedback_wght_mode", "endpoint") or "endpoint")
    if requested == "endpoint":
        return str(getattr(args, "endpoint_wght_mode", "freeze-zero"))
    return requested


def refuse_sigma_weight_startup_conflicts(args: argparse.Namespace) -> None:
    """Refuse at startup, before any stage, sigma/weight settings every corrected-HKL endpoint would refuse.

    The engines refuse these combinations at their first endpoint; a run that mixes the current defaults with
    part of the older flag set would otherwise stop only after the screening and nuisance stages.
    """

    if str(getattr(args, "mode", "auto")) not in {"auto", "regularize"}:
        return
    conflicts = []
    weight_modes = {str(getattr(args, "endpoint_wght_mode", "freeze-zero")), resolved_feedback_wght_mode(args)}
    if "converge" in weight_modes and not float(getattr(args, "shelxl_convergence_shift_esd", 0.0) or 0.0) > 0.0:
        conflicts.append(
            "the converge SHELXL weighting mode needs --shelxl-convergence-shift-esd > 0 (AUTO 0.01); with a zero "
            "tolerance use --endpoint-wght-mode freeze-zero or promote-gold"
        )
    error_model = error_model_sigma_active(args) or str(getattr(args, "post_bloch_sigma_policy", "auto")) == "error_model"
    if error_model and bool(getattr(args, "endpoint_include_free_rows", False)):
        conflicts.append(
            "the per-crystal error model (--dataset-sigma-policy error_model) is fitted on work rows only and cannot "
            "run with --endpoint-include-free-rows (free rows in every SHELXL HKL); add --dataset-sigma-policy auto "
            "for the legacy free-row HKL"
        )
    explicit = str(getattr(args, "cell_esd", "") or "").strip()
    if explicit:
        try:
            values = [float(value) for value in explicit.split(",")]
        except ValueError:
            values = []
        if len(values) != 6 or not all(math.isfinite(value) and value >= 0.0 for value in values):
            conflicts.append(f"--cell-esd needs six finite nonnegative values a,b,c,alpha,beta,gamma, got {explicit!r}")
    if conflicts:
        raise ValueError("cannot run with: " + "; ".join(conflicts))


def resolved_merge_sigma_policy(args: argparse.Namespace, *, multi_dataset: bool = False) -> str:
    """Resolve the sigma model used when corrected endpoint rows are merged."""

    requested = getattr(args, "merge_sigma_policy", "auto")
    if requested == "auto":
        return "chisq_overdispersion" if multi_dataset else "formal"
    return str(requested)


def resolve_final_n_basis(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    structure: str,
    stage: str,
    *,
    alternate_hand: bool = False,
) -> int:
    """Choose the final Bloch basis size from a cheap support scan."""

    from grond.cli_support.execution import run_logged

    args = crystal_arguments(args, ds.name)
    if args.fast:
        return 20
    if int(args.n_basis) > 0:
        return int(args.n_basis)
    if args.dry_run:
        return int(args.final_basis_fallback)
    scan = root / "decisions" / ds.name / f"{safe_label(stage)}_support_basis.json"
    cmd = [
        python,
        str(engine_script("support_basis_scan.py")),
        "--hkl",
        ds.hkl,
        "--res",
        structure,
        f"--hkl-transform={ds.hkl_transform}",
        "--out",
        str(scan),
        "--n-basis-limit",
        str(args.final_basis_scan_limit),
        "--margin",
        str(args.final_basis_margin),
        "--min-final",
        str(args.final_basis_min),
        "--max-final",
        str(args.final_basis_max),
        "--topup-d-min",
        f"{effective_topup_d_min(args, ds):g}",
        "--max-support-sg",
        f"{args.max_support_sg:g}",
        "--min-d-spacing",
        f"{ds.min_d_spacing:g}",
        "--min-peak",
        f"{ds.min_peak:g}",
        "--geometry-percentile",
        f"{ds.geometry_percentile:g}",
        "--target-mode",
        ds.target_mode,
        "--workers",
        str(args.workers),
        "--chunksize",
        str(args.chunksize),
    ]
    cmd.append("--include-nonpositive" if ds.include_nonpositive else "--positive-only")
    if args.max_rows:
        cmd += ["--max-rows", str(args.max_rows)]
    if alternate_hand:
        cmd.append("--alternate-hand")
    # The support basis is sized for the rocking window, profile sampling and mosaic cone the
    # stages actually use, not the engine defaults.
    cmd += [
        "--sg-window",
        f"{args.sg_window:g}",
        "--sg-window-mode",
        resolved_sg_window_mode(args, ds),
        "--sg-window-min",
        f"{args.sg_window_min:g}",
        "--sg-window-max",
        f"{resolved_sg_window_max(args, ds):g}",
        "--sg-window-scale",
        f"{args.sg_window_scale:g}",
        "--sg-window-frame-floor",
        f"{args.sg_window_frame_floor:g}",
        "--n-points",
        str(resolved_final_n_points(args, ds)),
    ]
    mosaicity_modes, mosaicity_sigmas = auto_mosaicity_grid(args)
    active_modes = [mode for mode in mosaicity_modes.split(",") if mode != "none"]
    if active_modes:
        cmd += [
            "--mosaicity-mode",
            active_modes[0],
            "--mosaicity-sigma-deg",
            f"{max(parse_strict_grid(mosaicity_sigmas, 'mosaicity sigma')):g}",
            "--mosaicity-points",
            str(int(args.mosaicity_points)),
        ]
    run_logged(cmd, logs, f"{ds.name}_{safe_label(stage)}_support_basis", dry_run=args.dry_run)
    payload = read_json(scan)
    n_basis = int(payload.get("recommended_n_basis") or args.final_basis_fallback)
    return max(int(args.final_basis_min), min(int(args.final_basis_max), n_basis))


def grid_from_nuisance(nui: dict, key: str, fallback: str) -> str:
    """Return a single-value grid string from a pinned nuisance parameter."""

    value = nui.get(key, fallback)
    try:
        return f"{float(value):.15g}"
    except (TypeError, ValueError):
        return str(value or fallback)


def grid_has_variation(text: str) -> bool:
    """Return true when a grid string contains more than one unique value."""

    parts = [p for p in str(text or "").replace(",", " ").split() if p]
    return len(dict.fromkeys(parts)) > 1


def compact_base_nuisance(args: argparse.Namespace) -> dict:
    """Build the starting nuisance grid for fast or full automatic runs."""

    if args.fast:
        return {
            "thickness_A": "1000,1200,1400",
            "thickness_model": "uniform_gradient",
            "thickness_grad_frac": "0.0,0.4",
            "frame_offset": "0.0",
            "group_width_deg": "frame,2frame",
            "detector_scale_bins": 0,
            "sigma_a": "0.0",
            "sigma_b_corr": "0.0",
            "sigma_c_peak": "0.0",
            "sg_jitter_mode": "none",
            "absorption_mode": "none",
            "dose_b_mode": "none",
            "mosaicity_mode": "none",
            "mosaicity_sigma_deg": "0.0",
            "row_resolution_b": "0.0",
        }
    return {
        "thickness_A": args.thickness_grid,
        "thickness_model": "uniform_gradient",
        "thickness_grad_frac": args.thickness_grad_grid,
        "thickness_sigma_frac": "0.0",
        "frame_offset": args.frame_offset_grid,
        "group_width_deg": args.group_width_grid,
        "detector_scale_bins": "0",  # recorded nuisance field; no detector scale field
        "sigma_a": "0.0",
        "sigma_b_corr": "0.0",
        "sigma_c_peak": "0.0",
        "sg_jitter_mode": "none",
        "sg_jitter_sigma0": "0.0",
        "sg_jitter_sigma_d_inv": "0.0",
        "sg_jitter_sigma_frame": "0.0",
        "absorption_mode": "none",
        "absorption_mu": "0.0",
        "absorption_frac": "0.0",
        "absorption_mu_g2": "0.0",
        "absorption_mu_theta2": "0.0",
        "absorption_tds_b": "0.0",
        "dose_b_mode": "none",
        "dose_b_per_frame": "0.0",
        "mosaicity_mode": "none",
        "mosaicity_sigma_deg": "0.0",
        "row_resolution_b": "0.0",
    }


def transform_scout_nuisance(args: argparse.Namespace) -> dict:
    """Return a single frozen nuisance contract for hand-invariant scouting.

    Transform ranking must not inherit a dynamical best-of-grid nuisance
    choice that can move when the input coordinates are inverted.
    """

    def first_numeric(value: object, fallback: float) -> str:
        """Return the first finite number in a comma or space separated value as text, else ``fallback``."""
        for token in str(value or "").replace(",", " ").split():
            try:
                parsed = float(token)
            except ValueError:
                continue
            if math.isfinite(parsed):
                return f"{parsed:.15g}"
        return f"{float(fallback):.15g}"

    base = compact_base_nuisance(args)
    # Every field that expands into an engine product grid is explicitly one
    # value.  In particular, "auto" is never allowed to survive for frame
    # offset or grouping because it would create a hand-sensitive best-of-grid
    # dynamical preselection before the matched-Born transform score is read.
    return {
        "thickness_A": first_numeric(base.get("thickness_A"), 1200.0),
        "thickness_model": "uniform_gradient",
        "thickness_grad_frac": first_numeric(base.get("thickness_grad_frac"), 0.0),
        "thickness_sigma_frac": "0.0",
        "frame_offset": "0.0",
        "group_width_deg": "frame",
        "detector_scale_bins": "0",
        "sigma_a": "0.0",
        "sigma_b_corr": "0.0",
        "sigma_c_peak": "0.0",
        "sg_jitter_mode": "none",
        "sg_jitter_sigma0": "0.0",
        "sg_jitter_sigma_d_inv": "0.0",
        "sg_jitter_sigma_frame": "0.0",
        "absorption_mode": "none",
        "absorption_mu": "0.0",
        "absorption_frac": "0.0",
        "absorption_mu_g2": "0.0",
        "absorption_mu_theta2": "0.0",
        "absorption_tds_b": "0.0",
        "dose_b_mode": "none",
        "dose_b_per_frame": "0.0",
        "mosaicity_mode": "none",
        "mosaicity_sigma_deg": "0.0",
        "row_resolution_b": "0.0",
    }


def nuisance_with_updates(nui: dict, **updates) -> dict:
    """Return a nuisance dictionary with only non-empty updates applied."""

    out = dict(nui)
    for key, value in updates.items():
        if value is not None and str(value).strip() != "":
            out[key] = value
    return out


def feature_grid(args: argparse.Namespace, family: str, nui: dict, ds: DatasetSpec | None = None) -> dict:
    """Expand one optional feature family into a candidate nuisance grid.

    The dose family needs the dataset when its grid is declared as total B over the scan.
    """

    if family == "row_resolution":
        return nuisance_with_updates(nui, row_resolution_b=args.row_resolution_b_grid)
    if family == "absorption":
        return nuisance_with_updates(
            nui,
            absorption_mode=args.absorption_mode_grid,
            absorption_mu=args.absorption_mu_grid,
            absorption_frac=args.absorption_frac_grid,
            absorption_mu_g2=args.absorption_mu_g2_grid,
            absorption_mu_theta2=args.absorption_mu_theta2_grid,
            absorption_tds_b=args.absorption_tds_b_grid,
        )
    if family == "sg_jitter":
        return nuisance_with_updates(
            nui,
            sg_jitter_mode=args.sg_jitter_mode_grid,
            sg_jitter_sigma0=args.sg_jitter_sigma0_grid,
            sg_jitter_sigma_d_inv=args.sg_jitter_sigma_d_inv_grid,
            sg_jitter_sigma_frame=args.sg_jitter_sigma_frame_grid,
            sg_jitter_points=args.sg_jitter_points,
        )
    if family == "sigma_error":
        return nuisance_with_updates(
            nui,
            sigma_a=args.sigma_a_grid,
            sigma_b_corr=args.sigma_b_corr_grid,
            sigma_c_peak=args.sigma_c_peak_grid,
        )
    if family == "dose":
        per_frame, _conversion = dose_b_per_frame_grid(args, ds)
        return nuisance_with_updates(
            nui,
            dose_b_mode=args.dose_b_mode_grid,
            dose_b_per_frame=per_frame,
        )
    if family == "mosaicity":
        modes, sigmas = auto_mosaicity_grid(args)
        return nuisance_with_updates(
            nui,
            mosaicity_mode=modes,
            mosaicity_sigma_deg=sigmas,
            mosaicity_points=int(args.mosaicity_points),
        )
    raise ValueError(f"unknown feature family {family!r}")


def feature_is_enabled(args: argparse.Namespace, family: str) -> bool:
    """Return true when an optional feature family has a real search grid."""

    if args.fast:
        return False
    if family == "row_resolution":
        return grid_has_variation(args.row_resolution_b_grid)
    if family == "absorption":
        return (
            grid_has_variation(args.absorption_mode_grid)
            or grid_has_variation(args.absorption_mu_grid)
            or grid_has_variation(args.absorption_frac_grid)
            or grid_has_variation(args.absorption_mu_g2_grid)
            or grid_has_variation(args.absorption_mu_theta2_grid)
            or grid_has_variation(args.absorption_tds_b_grid)
        )
    if family == "sg_jitter":
        return (
            grid_has_variation(args.sg_jitter_mode_grid)
            or grid_has_variation(args.sg_jitter_sigma0_grid)
            or grid_has_variation(args.sg_jitter_sigma_d_inv_grid)
            or grid_has_variation(args.sg_jitter_sigma_frame_grid)
        )
    if family == "sigma_error":
        return (
            grid_has_variation(args.sigma_a_grid)
            or grid_has_variation(args.sigma_b_corr_grid)
            or grid_has_variation(args.sigma_c_peak_grid)
        )
    if family == "dose":
        modes = [m for m in str(args.dose_b_mode_grid or "").replace(",", " ").split() if m]
        values = str(getattr(args, "dose_b_per_frame_grid", "") or "").strip() or str(
            getattr(args, "dose_b_total_grid", "") or ""
        )
        return "linear_frame" in modes and (grid_has_variation(args.dose_b_mode_grid) or grid_has_variation(values))
    if family == "mosaicity":
        modes, sigmas = auto_mosaicity_grid(args)
        return grid_has_variation(modes) or grid_has_variation(sigmas)
    return False


def prediction_cache_dir(args: argparse.Namespace, ds: DatasetSpec, stage: str) -> Path:
    """Choose the prediction-cache directory for a dataset and stage."""

    root = Path(args.prediction_cache_dir).expanduser() if args.prediction_cache_dir else Path(args.out_dir) / "cache"
    if args.prediction_cache_scope == "stage":
        return root / "stage_predictions" / stage / ds.name
    return root / "native_predictions" / ds.name


def effective_topup_d_min(
    args: argparse.Namespace, ds: DatasetSpec | None = None, min_d_spacing: float | None = None
) -> float:
    """Return the Bloch support depth, never shallower than selected data."""

    requested = float(getattr(args, "topup_d_min", 0.8))
    data_limit = float(
        min_d_spacing if min_d_spacing is not None else (ds.min_d_spacing if ds is not None else requested)
    )
    if data_limit > 0.0:
        return min(requested, data_limit)
    return requested


@functools.lru_cache(maxsize=64)
def _declares_dials_bbox(path: str) -> bool:
    """Return true when the leading header of a reflection file flags DIALS-bridge bbox z-ranges."""

    source = Path(path).expanduser()
    if not source.is_file():
        # Nothing can be computed from a missing file; the engine reports it.
        return False
    with source.open(errors="replace") as handle:
        for line in handle:
            if not line.startswith("!"):
                break
            key, separator, value = line[1:].partition("=")
            if separator and key.strip() == "DIALS_BBOX_Z_RANGE_AS_ALF0_BET0":
                return value.strip().upper() == "TRUE"
    return False


def resolved_sg_window_mode(args: argparse.Namespace, ds: DatasetSpec) -> str:
    """Choose the rocking-window source for this dataset.

    DIALS rows generated by GROND carry the reflection-table bbox z-range in
    ALF0/BET0 and declare it in the header.  Auto uses AUTO_DIALS_WINDOW_MODE
    for those rows and the fixed Sg window for everything else (in an XDS
    INTEGRATE.HKL, ALF0/BET0 are incident-beam angles, not a bbox).
    """

    requested = str(getattr(args, "sg_window_mode", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested
    return AUTO_DIALS_WINDOW_MODE if _declares_dials_bbox(str(ds.hkl)) else "fixed"


def resolved_sg_window_max(args: argparse.Namespace, ds: DatasetSpec) -> float:
    """Return the explicit --sg-window-max, or the cap belonging to the resolved window mode."""

    explicit = getattr(args, "sg_window_max", None)
    if explicit is not None:
        return float(explicit)
    return AUTO_DIALS_BBOX_WINDOW_MAX if resolved_sg_window_mode(args, ds) == "dials_bbox" else LEGACY_SG_WINDOW_MAX


def resolved_final_n_points(args: argparse.Namespace, ds: DatasetSpec) -> int:
    """Return the explicit --n-points, or the final profile point count derived from the window.

    Fixed (and other width-bounded) windows keep AUTO_FIXED_PROFILE_SPACING (0.010 -> 50, 0.004 -> 20);
    the DIALS-bbox window uses AUTO_DIALS_BBOX_N_POINTS.
    """

    explicit = getattr(args, "n_points", None)
    if explicit is not None:
        return int(explicit)
    if resolved_sg_window_mode(args, ds) == "dials_bbox":
        return int(AUTO_DIALS_BBOX_N_POINTS)
    width = float(args.sg_window) * float(getattr(args, "sg_window_scale", 1.0))
    return max(1, int(math.ceil(round(width / AUTO_FIXED_PROFILE_SPACING, 6))))


def resolved_screen_n_points(args: argparse.Namespace, ds: DatasetSpec) -> int:
    """Return the explicit --screen-n-points, or half the window-derived final count (0.010 -> 25, bbox -> 30)."""

    explicit = getattr(args, "screen_n_points", None)
    if explicit is not None:
        return int(explicit)
    derived = resolved_final_n_points(argparse.Namespace(**{**vars(args), "n_points": None}), ds)
    return max(1, int(math.ceil(derived / 2.0)))


def stage_n_points(args: argparse.Namespace, ds: DatasetSpec, *, screen: bool) -> int:
    """Profile points for one crystal's screening (screen, thickness, gates) or final-fidelity stage."""

    crystal = crystal_arguments(args, ds.name)
    return resolved_screen_n_points(crystal, ds) if screen else resolved_final_n_points(crystal, ds)


def parse_strict_grid(text: object, label: str) -> list[float]:
    """Parse a comma grid of finite numbers, refusing any token that is not one."""

    values: list[float] = []
    for token in str(text or "").replace(",", " ").split():
        try:
            value = float(token)
        except ValueError as exc:
            raise ValueError(f"{label} grid has a non-numeric value {token!r}") from exc
        if not math.isfinite(value):
            raise ValueError(f"{label} grid has a non-finite value {token!r}")
        values.append(value)
    return values


def auto_mosaicity_grid(args: argparse.Namespace) -> tuple[str, str]:
    """Translate --mosaicity-mode/--mosaicity-sigma-grid into engine mode and sigma grids.

    Sigma 0 means no cone and becomes the mode-none candidate; the positive widths use the
    requested mode.  Mode none (legacy) disables the cone family entirely.
    """

    mode = str(getattr(args, "mosaicity_mode", "none") or "none").strip().lower()
    if mode == "none":
        return "none", "0.0"
    sigmas = parse_strict_grid(getattr(args, "mosaicity_sigma_grid", ""), "mosaicity sigma")
    if not sigmas:
        raise ValueError("--mosaicity-sigma-grid is empty; use --mosaicity-mode none to disable the mosaic cone")
    if any(value < 0.0 or value > MOSAICITY_MAX_SIGMA_DEG for value in sigmas):
        raise ValueError(f"--mosaicity-sigma-grid values must lie in [0, {MOSAICITY_MAX_SIGMA_DEG:g}] degrees")
    positive = list(dict.fromkeys(value for value in sigmas if value > 0.0))
    if positive and not any(value == 0.0 for value in sigmas):
        # The gate compares every width with the active no-cone physics; without that candidate it
        # would be skipped as an invariant grid and the requested cone would silently never be tested.
        raise ValueError(
            "--mosaicity-sigma-grid must include 0, the no-cone reference every cone width is compared with"
        )
    if mode == "gaussian_cone":
        if getattr(args, "nuisance_selection", "nested") != "nested":
            raise ValueError("AUTO gaussian_cone currently requires nested selection so its zero-width reference "
                             "uses the same common beam envelope; fixed native/refine replays remain available")
        if (getattr(args, "profile_coordinate", "target_sg") != "direct_angle"
                or getattr(args, "profile_normalization", "legacy") != "target_sg_equivalent_v1"):
            raise ValueError("AUTO gaussian_cone needs --profile-coordinate direct_angle and "
                             "--profile-normalization target_sg_equivalent_v1 for comparable row intensities")
        support = float(getattr(args, "mosaicity_support_sigma_deg", 0.0))
        if support < max(sigmas):
            raise ValueError("--mosaicity-support-sigma-deg must cover the entire two-axis cone sigma grid")
        # Retain the cone selector at zero width so the reference uses the same declared maximum beam envelope.
        return "gaussian_cone", ",".join(f"{value:.15g}" for value in dict.fromkeys(sigmas))
    modes = (["none"] if any(value == 0.0 for value in sigmas) else []) + ([mode] if positive else [])
    return ",".join(modes), (",".join(f"{value:.15g}" for value in positive) or "0.0")


@functools.lru_cache(maxsize=64)
def dataset_frame_span(path: str) -> float:
    """ZCAL span of every row in a reflection file (the frames over which dose B accumulates).

    The engine applies dose B as dose_b_per_frame * (ZCAL - min ZCAL) over all finite ZCAL rows,
    so the total over the scan is dose_b_per_frame * (max ZCAL - min ZCAL).
    """

    zcal: list[float] = []
    with Path(path).expanduser().open(errors="replace") as handle:
        for line in handle:
            if line.startswith("!"):
                continue
            parts = line.split()
            if len(parts) < 11 or len(parts[:21]) not in {12, 21}:
                continue
            try:
                value = float(parts[7])
            except ValueError:
                continue
            if math.isfinite(value):
                zcal.append(value)
    if len(zcal) < 2:
        raise ValueError(f"cannot determine the frame span of {path}: fewer than two finite ZCAL rows")
    span = max(zcal) - min(zcal)
    if not span > 0.0:
        raise ValueError(f"reflection rows in {path} span no frames; a dose-B grid over the scan is undefined")
    return float(span)


def dose_b_per_frame_grid(args: argparse.Namespace, ds: DatasetSpec | None) -> tuple[str, dict]:
    """Return the engine per-frame dose grid and how it was obtained.

    An explicit --dose-b-per-frame-grid is used verbatim (legacy units).  Otherwise
    --dose-b-total-grid (A^2 over the scan) is divided by this dataset's ZCAL span.
    """

    explicit = str(getattr(args, "dose_b_per_frame_grid", "") or "").strip()
    if explicit:
        return explicit, {"source": "dose_b_per_frame_grid", "per_frame_grid": explicit}
    totals = parse_strict_grid(getattr(args, "dose_b_total_grid", ""), "dose total B")
    if not totals:
        raise ValueError("neither --dose-b-per-frame-grid nor --dose-b-total-grid gives a dose grid")
    if any(value < 0.0 for value in totals):
        raise ValueError("--dose-b-total-grid values must be nonnegative (A^2 accumulated over the scan)")
    if ds is None:
        raise ValueError("a dose-B grid given as total B over the scan needs the dataset to convert it per frame")
    span = dataset_frame_span(str(ds.hkl))
    per_frame = ",".join(f"{value / span:.15g}" for value in totals)
    return per_frame, {
        "source": "dose_b_total_grid",
        "total_b_grid_A2": totals,
        "zcal_span_frames": span,
        "per_frame_grid": per_frame,
        "definition": "dose_b_per_frame = total_B / (max ZCAL - min ZCAL); engine reference frame = min ZCAL",
    }


def _point_count_source(crystal: argparse.Namespace, key: str, derived_source: str) -> str:
    """Where a crystal's profile point count came from: derived, a named preset (--preset), or given explicitly."""

    from grond.workflows.run_presets import preset_source

    if getattr(crystal, key, None) is None:
        return derived_source
    return preset_source(crystal, key) or "explicit"


def configure_auto_physics(prep: PrepConfig, args: argparse.Namespace) -> dict:
    """Resolve the per-crystal AUTO rocking-window and profile contract before any stage is built.

    Unset --n-points/--screen-n-points/--sg-window-max are derived per crystal from the resolved
    window.  When all crystals agree the values become the run-wide settings; otherwise each crystal
    carries its own through crystal_arguments and the run-wide value stays unset, so no stage can use
    another crystal's value.  The derived values go into the crystal's ``auto_physics`` block, never into
    its ``prediction`` block: crystal_arguments turns a fixed-parameter crystal's prediction values into
    fixed_* settings that override every stage, which would run the screen-type stages at the final Q.
    The contract is written to decisions/auto_physics_contract.json.
    """

    # Validate the gate grids before any calculation starts.
    auto_mosaicity_grid(args)
    parse_strict_grid(getattr(args, "dose_b_per_frame_grid", ""), "dose per-frame B")
    if any(value < 0.0 for value in parse_strict_grid(getattr(args, "dose_b_total_grid", ""), "dose total B")):
        raise ValueError("--dose-b-total-grid values must be nonnegative (A^2 accumulated over the scan)")
    dose_by_total = feature_is_enabled(args, "dose") and not str(getattr(args, "dose_b_per_frame_grid", "") or "").strip()
    records: dict[str, dict] = {}
    for ds in prep.datasets:
        crystal = crystal_arguments(args, ds.name)
        mode = resolved_sg_window_mode(crystal, ds)
        if mode == "dials_bbox" and Path(ds.hkl).expanduser().is_file() and not _declares_dials_bbox(str(ds.hkl)):
            # The engine refuses these rows at the first prediction; refuse before any stage runs.
            raise ValueError(
                f"sg_window_mode dials_bbox for {ds.name}: {ds.hkl} does not declare "
                "DIALS_BBOX_Z_RANGE_AS_ALF0_BET0=TRUE (GROND DIALS bridge), so ALF0/BET0 are not a bbox z-range"
            )
        derived_source = "window"
        dose: dict = {}
        if dose_by_total and Path(ds.hkl).expanduser().is_file():
            try:
                dose = dose_b_per_frame_grid(crystal, ds)[1]
            except ValueError as exc:
                # The dose gate reports the same error as its failed decision for this crystal.
                dose = {"source": "dose_b_total_grid", "error": str(exc)}
                print(f"WARN dose-B gate for {ds.name} cannot convert total B per frame: {exc}", flush=True)
        records[ds.name] = {
            "hkl": str(ds.hkl),
            "dials_bbox_header": _declares_dials_bbox(str(ds.hkl)),
            "requested_sg_window_mode": str(getattr(crystal, "sg_window_mode", "auto")),
            "sg_window_mode": mode,
            "sg_window": float(crystal.sg_window),
            "sg_window_max": resolved_sg_window_max(crystal, ds),
            "n_points": resolved_final_n_points(crystal, ds),
            "screen_n_points": resolved_screen_n_points(crystal, ds),
            "n_points_source": _point_count_source(crystal, "n_points", derived_source),
            "screen_n_points_source": _point_count_source(crystal, "screen_n_points", derived_source),
            "max_support_sg": float(crystal.max_support_sg),
            "dose_b_grid": dose,
        }
    for key in ("n_points", "screen_n_points", "sg_window_max"):
        if getattr(args, key, None) is not None:
            continue
        values = {record[key] for record in records.values()}
        if len(values) == 1:
            setattr(args, key, values.pop())
            continue
        crystals = getattr(args, "crystal_parameters", None)
        if not isinstance(crystals, dict):
            crystals = {}
            args.crystal_parameters = crystals
        for name, record in records.items():
            derived = crystals.setdefault(name, {}).setdefault("auto_physics", {})
            derived.setdefault(key, record[key])
    modes, sigmas = auto_mosaicity_grid(args)
    contract = {
        "schema": "grond.auto-physics-contract.v1",
        "auto_dials_window_mode": AUTO_DIALS_WINDOW_MODE,
        "datasets": records,
        "mosaicity": {"mode_grid": modes, "sigma_deg_grid": sigmas, "points": int(args.mosaicity_points)},
        "dose": {
            "mode_grid": str(args.dose_b_mode_grid),
            "per_frame_grid": str(getattr(args, "dose_b_per_frame_grid", "") or ""),
            "total_b_grid_A2": str(getattr(args, "dose_b_total_grid", "") or ""),
        },
        "absorption": {
            "mode_grid": str(args.absorption_mode_grid),
            "mu_grid": str(args.absorption_mu_grid),
            "frac_grid": str(args.absorption_frac_grid),
        },
        "physics_gate_selection": str(getattr(args, "physics_gate_selection", "in-sample")),
        "thickness_gate_split": str(getattr(args, "thickness_gate_split", "legacy")),
    }
    out = Path(args.out_dir) / "decisions" / "auto_physics_contract.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(contract, indent=2, allow_nan=True) + "\n")
    return contract


def warn_missing_cell_esd_source(structure: str, datasets, args: argparse.Namespace) -> None:
    """Say at startup when the auto cell e.s.d. policy will fall back to the starting model's ZERR.

    The sources do not depend on the HKL transform (it only permutes the values), so their presence is
    checked on the input files: every crystal needs its UNIT_CELL axes, and one crystal also needs an XDS
    CORRECT.LP or a DIALS scan-varying cell.  Without them configure_cell_esd uses the model's ZERR e.s.d.s
    (or keeps its ZERR unchanged when they are zero) instead of stopping the run; this only warns early.
    """

    from grond.io.structure_contract import is_shelx_model
    from grond.refine.cell_esd import dataset_model_cell

    if str(getattr(args, "cell_esd_policy", "source") or "source") == "source" or bool(getattr(args, "dry_run", False)):
        return
    if str(getattr(args, "cell_esd", "") or "").strip() or not is_shelx_model(structure):
        return
    if not any(line.split()[:1] == ["ZERR"] for line in Path(structure).expanduser().read_text(errors="replace").upper().splitlines()):
        return
    hint = "pass --cell-esd a,b,c,alpha,beta,gamma to give measured values"
    records = []
    for ds in datasets:
        try:
            records.append(dataset_model_cell(ds.hkl, None))
        except (OSError, ValueError) as exc:
            print(f"WARN cell e.s.d.s: the cell of {ds.name} cannot be read ({exc}); the ZERR e.s.d.s will come from the "
                  f"starting model; {hint}", flush=True)
            return
    if len(records) == 1 and not (records[0].get("xds_correct_lp_esd") or records[0].get("scan_varying_cell_sd")):
        mismatch = records[0].get("xds_correct_lp_setting_mismatch")
        detail = (
            f" (its CORRECT.LP cell {mismatch['correct_lp_cell']} is in another setting than the row file's axes)"
            if mismatch
            else ""
        )
        print(
            f"WARN cell e.s.d.s: one crystal ({datasets[0].name}) without an XDS CORRECT.LP in the row file's setting or "
            f"a DIALS scan-varying cell{detail}; the ZERR e.s.d.s will come from the starting model; {hint}",
            flush=True,
        )


def configure_cell_esd(structure: str, datasets, args: argparse.Namespace) -> dict:
    """Resolve the ZERR cell e.s.d.s of every SHELXL endpoint once the HKL transforms are fixed.

    ``--cell-esd-policy auto`` (AUTO) takes --cell-esd, else the crystals' cell spread, else the one
    crystal's XDS CORRECT.LP or DIALS scan-varying spread (grond.refine.cell_esd); without a measured source it
    uses the model's ZERR e.s.d.s (or keeps its ZERR when they are zero) and warns; ``source`` (legacy, and every
    namespace without the option) keeps the model's ZERR.  The
    record is written to decisions/cell_esd.json and the value reaches the engines as --zerr-esds.
    """

    from grond.io.structure_contract import is_shelx_model
    from grond.refine.cell_esd import resolve_cell_esds

    policy = str(getattr(args, "cell_esd_policy", "source") or "source")
    args.resolved_zerr_esds = ""
    if policy == "source":
        record: dict = {"status": "source_policy", "policy": policy, "definition": "the model's ZERR is written unchanged"}
    elif not is_shelx_model(structure):
        record = {"status": "not_shelx_model", "policy": policy, "structure": str(structure)}
    elif bool(getattr(args, "dry_run", False)):
        record = {"status": "dry_run", "policy": policy}
    else:
        text = Path(structure).expanduser().read_text(errors="replace")
        if not any(line.split()[:1] == ["ZERR"] for line in text.upper().splitlines()):
            # Nothing to fill in: SHELXL itself refuses a model without ZERR ("ZERR MISSING") at the first endpoint.
            record = {"status": "no_zerr_in_model", "policy": policy, "structure": str(structure)}
            print(f"WARN {structure} has no ZERR record; SHELXL needs 'ZERR Z ...' after CELL", flush=True)
            out = Path(args.out_dir) / "decisions" / "cell_esd.json"
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(record, indent=2, allow_nan=True) + "\n")
            return record
        records = [{"name": ds.name, "hkl": str(ds.hkl), "hkl_transform": ds.hkl_transform} for ds in datasets]
        record = {"policy": policy, **resolve_cell_esds(structure, records, explicit=str(getattr(args, "cell_esd", "") or ""))}
        args.resolved_zerr_esds = str(record.get("zerr_esds_arg") or "")
        floored = record.get("floored_to_print_precision") or []
        if record.get("status") == "model_zerr_kept":
            print(f"WARN cell e.s.d.s: {record['source']}; SHELXL bond e.s.d.s will not include the cell uncertainty. "
                  "Pass --cell-esd a,b,c,alpha,beta,gamma to give measured values.", flush=True)
        else:
            prefix = "WARN " if record.get("status") == "ok_model_zerr_fallback" else ""
            print(
                f"{prefix}Cell e.s.d.s (ZERR): {', '.join(f'{v:g}' for v in record['esds'])} from {record['source']}"
                + (f"; below print precision, written as one unit of the last digit: {', '.join(floored)}" if floored else ""),
                flush=True,
            )
    out = Path(args.out_dir) / "decisions" / "cell_esd.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(record, indent=2, allow_nan=True) + "\n")
    return record


def resolved_row_plan_cache(args: argparse.Namespace, *, fixed_model: bool) -> str:
    """Resolve the workflow cache policy without freezing a moving-model basis."""

    mode = str(getattr(args, "row_plan_cache", "off") or "off").lower()
    if mode == "auto":
        # Fixed-model stages know the nuisance-grid shape and decide whether
        # enough candidates share one geometry key to amortize a cold cache
        # build.  Moving-model stages must not freeze a basis implicitly.
        return "auto" if fixed_model else "off"
    return mode
