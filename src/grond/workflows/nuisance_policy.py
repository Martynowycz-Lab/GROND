"""Resolve basis sizes, fidelity profiles and nuisance defaults before building stages."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

from grond.cli_support.values import read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.paths import engine_script
from grond.workflows.crystal_parameters import crystal_arguments
from grond.workflows.decision import safe_label


def resolved_post_bloch_sigma_policy(
    args: argparse.Namespace, *, final_endpoint: bool = False, feedback_loop: bool = False
) -> str:
    """Resolve the endpoint sigma calibration policy for a workflow stage."""

    requested = getattr(args, "post_bloch_sigma_policy", "auto")
    if requested == "auto":
        return "uniform_z_rms" if (final_endpoint or feedback_loop) else "none"
    return str(requested)


def resolved_dataset_sigma_policy(args: argparse.Namespace, *, multi_dataset: bool = False) -> str:
    """Resolve per-dataset endpoint sigma calibration for combination stages."""

    requested = getattr(args, "dataset_sigma_policy", "auto")
    if requested == "auto":
        return "uniform_z_rms" if multi_dataset else "none"
    return str(requested)


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
    if getattr(args, "full_fast", False):
        return int(args.full_fast_n_basis)
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
    run_logged(cmd, logs, f"{ds.name}_{safe_label(stage)}_support_basis", dry_run=args.dry_run)
    payload = read_json(scan)
    n_basis = int(payload.get("recommended_n_basis") or args.final_basis_fallback)
    return max(int(args.final_basis_min), min(int(args.final_basis_max), n_basis))


def first(value: str, fallback: str) -> str:
    """Return the first comma-separated grid value or a fallback."""

    text = str(value or "").strip()
    return text.split(",")[0] if text else fallback


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
            "row_resolution_b": "0.0",
        }
    pets_static = bool(getattr(args, "pets_static_auto", False))
    return {
        "thickness_A": args.thickness_grid,
        "thickness_model": (args.thickness_model_grid if pets_static else "uniform_gradient"),
        "thickness_grad_frac": args.thickness_grad_grid,
        "thickness_sigma_frac": "0.0",
        "frame_offset": args.frame_offset_grid,
        "group_width_deg": args.group_width_grid,
        "detector_scale_bins": args.detector_scale_bins,
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
        "row_resolution_b": "0.0",
    }


def transform_scout_nuisance(args: argparse.Namespace) -> dict:
    """Return a single frozen nuisance contract for hand-invariant scouting.

    Transform ranking must not inherit a dynamical best-of-grid nuisance
    choice that can move when the input coordinates are inverted.
    """

    def first_numeric(value: object, fallback: float) -> str:
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
        "row_resolution_b": "0.0",
    }


def apply_speed_profile(args: argparse.Namespace, structure: str | None = None) -> dict[str, object]:
    """Apply named run profiles after parsing and return a provenance record."""

    from grond.io.structure_contract import estimate_model_nonh_atoms

    if getattr(args, "fast", False) and getattr(args, "full_fast", False):
        raise SystemExit(
            "Choose either --fast or --full-fast. --fast is triage-only; --full-fast runs the automatic gates with cheap N/Q."
        )
    if not getattr(args, "full_fast", False):
        return {"profile": "standard"}

    changes: dict[str, object] = {
        "profile": "full_fast",
        "reason": "automatic gates with cheap Bloch settings and minimal endpoint polish",
    }

    def assign(name: str, value: object) -> None:
        old = getattr(args, name)
        setattr(args, name, value)
        if old != value:
            changes[name] = {"old": old, "new": value}

    n_basis = int(args.full_fast_n_basis)
    n_points = int(args.full_fast_n_points)
    assign("screen_n_basis", n_basis)
    assign("screen_n_points", n_points)
    assign("transform_rerank_n_basis", min(int(args.transform_rerank_n_basis), n_basis))
    assign("transform_rerank_n_points", min(int(args.transform_rerank_n_points), n_points))
    assign("hand_n_basis", min(int(args.hand_n_basis), n_basis))
    assign("hand_n_points", min(int(args.hand_n_points), n_points))
    assign("n_points", n_points)
    assign("structure_n_basis", min(int(args.structure_n_basis), n_basis))
    assign("structure_n_points", min(int(args.structure_n_points), max(3, n_points)))
    atom_count = estimate_model_nonh_atoms(structure or getattr(args, "structure", "") or "")
    auto_rows = 250
    auto_nfev = 2
    if atom_count is not None and atom_count > 80:
        auto_rows = 120
        auto_nfev = 1
    elif atom_count is not None and atom_count > 45:
        auto_rows = 160
        auto_nfev = 1
    row_limit = int(args.full_fast_structure_rows) if int(args.full_fast_structure_rows) > 0 else auto_rows
    eval_limit = int(args.full_fast_structure_max_nfev) if int(args.full_fast_structure_max_nfev) > 0 else auto_nfev
    changes["structure_budget"] = {
        "non_h_atoms": atom_count,
        "row_limit": row_limit,
        "max_eval_limit": eval_limit,
        "source": "explicit"
        if int(args.full_fast_structure_rows) > 0 or int(args.full_fast_structure_max_nfev) > 0
        else "atom_count_auto",
    }
    assign("structure_objective_rows", min(int(args.structure_objective_rows), row_limit))
    assign("structure_max_nfev", min(int(args.structure_max_nfev), eval_limit))
    assign("shelxl_iterations", min(int(args.shelxl_iterations), 1))
    if args.endpoint_feedback == "auto":
        assign("endpoint_feedback", "off")
    if args.observation_polish == "auto":
        assign("observation_polish", "off")
    if args.forensic_depth == "standard":
        assign("forensic_depth", "quick")
    return changes


def nuisance_with_updates(nui: dict, **updates) -> dict:
    """Return a nuisance dictionary with only non-empty updates applied."""

    out = dict(nui)
    for key, value in updates.items():
        if value is not None and str(value).strip() != "":
            out[key] = value
    return out


def feature_grid(args: argparse.Namespace, family: str, nui: dict) -> dict:
    """Expand one optional feature family into a candidate nuisance grid."""

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
        return nuisance_with_updates(
            nui,
            dose_b_mode=args.dose_b_mode_grid,
            dose_b_per_frame=args.dose_b_per_frame_grid,
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
        return grid_has_variation(args.dose_b_mode_grid) or grid_has_variation(args.dose_b_per_frame_grid)
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


def resolved_sg_window_mode(args: argparse.Namespace, ds: DatasetSpec) -> str:
    """Choose the rocking-window source for this dataset.

    DIALS rows generated by GROND carry the reflection-table bbox z-range in
    ALF0/BET0.  The bbox is kept for diagnostics, but auto mode keeps the
    reference fixed Sg window because the DIALS integration support is usually
    broader than the physical Bloch quadrature width.
    """

    requested = str(getattr(args, "sg_window_mode", "auto") or "auto").strip().lower()
    if requested != "auto":
        return requested
    return "fixed"


def resolved_row_plan_cache(args: argparse.Namespace, *, fixed_model: bool) -> str:
    """Resolve the workflow cache policy without freezing a moving-model basis."""

    mode = str(getattr(args, "row_plan_cache", "off") or "off").lower()
    if mode == "auto":
        # Fixed-model stages know the nuisance-grid shape and decide whether
        # enough candidates share one geometry key to amortize a cold cache
        # build.  Moving-model stages must not freeze a basis implicitly.
        return "auto" if fixed_model else "off"
    return mode
