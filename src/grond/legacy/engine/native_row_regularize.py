#!/usr/bin/env python3
"""Nuisance-only native Bloch regularization and corrected-HKL endpoint.

This is deliberately fixed-structure. It estimates and applies the row-level
regularization terms we can defend from the current XDS row contract:

* crystal thickness and thickness gradient;
* one global frame offset;
* per-frame/wedge group scales fitted on work reflections;
* simple sigma inflation terms used for weighting;
* corrected-HKL export from q = Idyn/Iborn, followed by SHELXL scoring.

It does not move coordinates, ADPs, occupancies, or restraints. Those changes are
processed by a separate least-squares model-update bridge after the row target is stable.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import asdict
from importlib.metadata import PackageNotFoundError, version as distribution_version
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

if not __package__:
    # Direct legacy execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
import scipy
from scipy.optimize import brentq

from grond.refine.shelx_input import (
    _SHELX_FLOAT_TOKEN_RE as _SHELX_FLOAT_TOKEN_RE,
    DEFAULT_ELECTRON_WAVELENGTH_A as DEFAULT_ELECTRON_WAVELENGTH_A,
    _format_wght_line as _format_wght_line,
    _hklf_order as _hklf_order,
    _iter_recommended_wght_terms as _iter_recommended_wght_terms,
    _unsupported_endpoint_reason as _unsupported_endpoint_reason,
    enforce_hkl_row_width as enforce_hkl_row_width,
    force_shelxl_cycle_count as force_shelxl_cycle_count,
    force_shelxl_one_cycle as force_shelxl_one_cycle,
    normalize_shelx_cell_wavelength as normalize_shelx_cell_wavelength,
    parse_shelx_wght_pair as parse_shelx_wght_pair,
    parse_shelx_wght_terms as parse_shelx_wght_terms,
    parse_wght as parse_wght,
    set_active_wght_lines as set_active_wght_lines,
    shelx_wavelength_needs_electron_default as shelx_wavelength_needs_electron_default,
    write_endpoint_ins as write_endpoint_ins,
    write_hkl as write_hkl,
    write_no_exti_ins as write_no_exti_ins,
)
from grond.refine.shelxl_diagnostics import (
    classify_shelxl_iteration_rejection as classify_shelxl_iteration_rejection,
    parse_lst as parse_lst,
    parse_recommended_wght as parse_recommended_wght,
    shelx_res_health as shelx_res_health,
    shelxl_iteration_safe as shelxl_iteration_safe,
    shelxl_statistics_complete as shelxl_statistics_complete,
)
from grond.refine.shelxl_iterations import (
    DEFAULT_NPD_DAMP_LIMSE as DEFAULT_NPD_DAMP_LIMSE,
    DEFAULT_NPD_DAMP_RETRIES as DEFAULT_NPD_DAMP_RETRIES,
    _active_wght_from_stats as _active_wght_from_stats,
    _shelxl_attempt_label as _shelxl_attempt_label,
    archive_shelxl_attempt as archive_shelxl_attempt,
    archived_shelxl_attempt_record as archived_shelxl_attempt_record,
    promote_recommended_wght as promote_recommended_wght,
    recommended_wght_from_stats as recommended_wght_from_stats,
    recover_npd_shelxl_attempt as recover_npd_shelxl_attempt,
    run_one_shelxl_attempt as run_one_shelxl_attempt,
    select_last_safe_shelxl_iteration as select_last_safe_shelxl_iteration,
    snapshot_safe_shelxl_iteration as snapshot_safe_shelxl_iteration,
    write_active_damp as write_active_damp,
    write_active_wght as write_active_wght,
)
from grond.regularize.arguments_native import build_parser as build_native_parser
from grond.regularize.contracts import (
    AUTO_EXACT_ROW_PLAN_MIN_REUSE as AUTO_EXACT_ROW_PLAN_MIN_REUSE,
    TARGET_MODE_LABELS as TARGET_MODE_LABELS,
    TARGET_MODES as TARGET_MODES,
    Nuisance as Nuisance,
    absorption_parameter_grid as absorption_parameter_grid,
    estimated_row_plan_reuse_per_geometry_key as estimated_row_plan_reuse_per_geometry_key,
    kernel_variance_terms as kernel_variance_terms,
    parse_float_grid as parse_float_grid,
    parse_int_grid as parse_int_grid,
    resolve_fixed_model_row_plan_cache as resolve_fixed_model_row_plan_cache,
)
from grond.regularize.evaluation import (
    apply_q_control as apply_q_control,
    evaluate_trial as evaluate_trial,
    paired_inverse_prediction as paired_inverse_prediction,
    resolution_shell_bins as resolution_shell_bins,
)
from grond.regularize.grid import populate_nuisance_trials
from grond.regularize.observations import (
    automatic_frame_offset_grid as automatic_frame_offset_grid,
    detector_bin_indices as detector_bin_indices,
    endpoint_observation_arrays as endpoint_observation_arrays,
    endpoint_resolution_modulation as endpoint_resolution_modulation,
    geometry_tail_mask as geometry_tail_mask,
    group_indices as group_indices,
    nominal_frame_width_deg as nominal_frame_width_deg,
    parse_group_width_grid as parse_group_width_grid,
    row_contract_selection as row_contract_selection,
    row_resolution_modulation as row_resolution_modulation,
    row_selection as row_selection,
    sigma_eff as sigma_eff,
    target_arrays as target_arrays,
    target_factor as target_factor,
)
from grond.regularize.prediction_cache import (
    cacheable_prediction_arrays as cacheable_prediction_arrays,
    canonical_prediction_cache_payload as canonical_prediction_cache_payload,
    external_geometry_fingerprints as external_geometry_fingerprints,
    prediction_cache_validation_errors as prediction_cache_validation_errors,
    read_prediction_cache as read_prediction_cache,
    write_prediction_cache as write_prediction_cache,
)
from grond.regularize.row_output import (
    native_row_table_artifact as native_row_table_artifact,
    write_row_table as write_row_table,
)
from grond.regularize.scales import (
    _fit_global_scale as _fit_global_scale,
    _scale_fit_mask as _scale_fit_mask,
    fit_frame_detector_scales as fit_frame_detector_scales,
    fit_group_scales as fit_group_scales,
    native_metrics as native_metrics,
    shelx_weights as shelx_weights,
    solve_regularized_scales as solve_regularized_scales,
)
from grond.regularize.selection import (
    _NUISANCE_CATEGORICAL_FIELDS as _NUISANCE_CATEGORICAL_FIELDS,
    _NUISANCE_NULLABLE_NUMERIC_FIELDS as _NUISANCE_NULLABLE_NUMERIC_FIELDS,
    _promotion_trial_record as _promotion_trial_record,
    _trial_full_row_score as _trial_full_row_score,
    _trial_metric_value as _trial_metric_value,
    _trial_split_count as _trial_split_count,
    _trial_valid_mask as _trial_valid_mask,
    _valid_mask_sha256 as _valid_mask_sha256,
    metric_row as metric_row,
    nuisance_reference_matches as nuisance_reference_matches,
    rank_trials as rank_trials,
    select_trial_with_promotion_gate as select_trial_with_promotion_gate,
    trial_metric as trial_metric,
)

try:
    from grond.paths import runtime_cache_dir
except ImportError:  # Direct script execution from the legacy engine directory.

    def runtime_cache_dir(*parts: str) -> Path:
        override = os.environ.get("GROND_CACHE_DIR", "").strip()
        root = Path(override).expanduser() if override else Path.home() / ".cache" / "grond"
        return root.joinpath(*parts)

try:  # Package import.
    from .bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        InfeasibleSgJitterProfileError,
        NonPassiveAbsorptionError as NonPassiveAbsorptionError,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        frozen_kernel_contract,
        load_transform_split_contract_masks,
        parse_model,
        sha256,
        split_masks,
    )
    from .detector_scale_regularization import (
        detector_field_diagnostics as detector_field_diagnostics,
        solve_regularized_detector_field as solve_regularized_detector_field,
    )
    from .endpoint_modulation import modulation_manifest_fields
    from .post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas
    from .shelx_diagnostics import (
        parse_shelx_lst_pathologies as parse_shelx_lst_pathologies,
        shelx_pathology_safe_for_feedback as shelx_pathology_safe_for_feedback,
    )
    from .shelx_hkl import (
        safe_fixed_width_factor as safe_fixed_width_factor,
        write_fixed_width_hkl as write_fixed_width_hkl,
    )
except ImportError:  # Direct script execution from this directory.
    from bloch_engine import (
        DEFAULT_MAX_SUPPORT_SG,
        Config,
        InfeasibleSgJitterProfileError,
        XDSBlochPort,
        XDSData,
        alternate_hand_model,
        frozen_kernel_contract,
        load_transform_split_contract_masks,
        parse_model,
        sha256,
        split_masks,
    )
    from endpoint_modulation import modulation_manifest_fields
    from post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_RES = Path("model.res")
DEFAULT_BASELINE_HKL = Path("model.hkl")
SHELXL = os.environ.get("SHELXL") or shutil.which("shelxl") or shutil.which("shelxl2019") or ""
SHELX2MAP = os.environ.get("SHELX2MAP") or shutil.which("shelx2map") or ""

PREDICTION_CACHE_SCHEMA = 7


def configure_shelxl_executable(executable: str) -> str:
    """Bind the explicit endpoint executable for this worker process."""

    global SHELXL
    SHELXL = str(executable or "").strip()
    return SHELXL


def shelxl_executable_is_available(executable: str) -> bool:
    """Return whether an endpoint executable exists and is launchable."""

    value = str(executable or "").strip()
    if not value:
        return False
    if "/" in value:
        path = Path(value).expanduser()
        return path.is_file() and os.access(path, os.X_OK)
    return shutil.which(value) is not None


def require_endpoint_shelxl(executable: str, *, skip_endpoint_refinement: bool) -> None:
    """Fail before native computation when an endpoint tool cannot launch."""

    if skip_endpoint_refinement or shelxl_executable_is_available(executable):
        return
    raise SystemExit(
        f"SHELXL executable is missing or not executable: {executable!r}. "
        "Supply --shelxl with a launchable executable, or add "
        "--skip-endpoint-refinement for a deliberate row-only run."
    )


def fail_if_shelxl_became_unavailable(results: list[dict], *, context: str) -> None:
    """Reject a completed endpoint stage containing a missing-tool record."""

    missing = [
        record
        for record in results
        if str((record.get("shelxl") or {}).get("status", "")).strip().lower()
        == "missing_shelxl"
    ]
    if missing:
        raise RuntimeError(
            f"SHELXL became unavailable during {context}; refusing to mark the "
            "endpoint-enabled native stage complete."
        )


def _prediction_engine_source_fingerprint() -> str:
    """Fingerprint every explicit implementation owner of cached predictions."""

    from grond.bloch.source import physics_source_files
    from grond.regularize.source import regularization_source_files

    package_root = Path(__file__).resolve().parents[2]
    paths = dict.fromkeys((*physics_source_files(), *regularization_source_files()))
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda item: item.relative_to(package_root).as_posix()):
        digest.update(path.relative_to(package_root).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def _installed_version(distribution: str) -> str:
    try:
        return distribution_version(distribution)
    except PackageNotFoundError:
        return "not-installed"


# These are intentionally computed once per process. A source or numerical
# runtime change produces a new content-addressed prediction cache key.
PREDICTION_ENGINE_SOURCE_SHA256 = _prediction_engine_source_fingerprint()
NUMERICAL_RUNTIME_VERSIONS = {
    "numpy": str(np.__version__),
    "scipy": str(scipy.__version__),
    "gemmi": _installed_version("gemmi"),
    "jax": _installed_version("jax"),
    "jaxlib": _installed_version("jaxlib"),
}


def _xds_header_sg(data: XDSData, hkl: np.ndarray, omega_deg: float) -> float:
    """Evaluate target-beam Sg using only the geometry recorded by XDS."""

    A = data.A_for_omega(float(omega_deg))
    theta = np.radians(float(omega_deg))
    rotation = data.axis_eye + np.sin(theta) * data.axis_cross + (1.0 - np.cos(theta)) * data.axis_cross2
    transform = data.setting_rotation @ rotation @ data.fixed_rotation @ A
    g = transform @ np.asarray(hkl, dtype=float)
    return float((data.k**2 - np.sum((data.s0 + g) ** 2)) / (2.0 * data.k))


def _xds_header_bragg_omega(data: XDSData, hkl: np.ndarray, omega0: float) -> float:
    """Find the nearest Sg=0 crossing for a row without using intensities."""

    oscillation = abs(float(data.oscillation_range))
    max_delta = max(2.0 * oscillation, min(2.0, 10.0 * oscillation))
    offsets = np.linspace(-max_delta, max_delta, 65)
    values = np.asarray([_xds_header_sg(data, hkl, omega0 + offset) for offset in offsets])
    close = np.flatnonzero(np.abs(values) <= 1e-10)
    if len(close):
        best = int(close[np.argmin(np.abs(offsets[close]))])
        return float(omega0 + offsets[best])
    brackets = []
    for i in range(len(offsets) - 1):
        if values[i] * values[i + 1] < 0.0:
            brackets.append((abs(0.5 * (offsets[i] + offsets[i + 1])), i))
    if not brackets:
        raise ValueError("no local Sg=0 crossing near XDS ZCAL")
    _, index = min(brackets)
    return float(
        brentq(
            lambda omega: _xds_header_sg(data, hkl, omega),
            omega0 + float(offsets[index]),
            omega0 + float(offsets[index + 1]),
            xtol=1e-12,
            rtol=1e-12,
            maxiter=100,
        )
    )


def xds_frame_datum_summary(data: XDSData, indices: np.ndarray, max_samples: int = 512) -> dict:
    """Measure the global-header versus row-ZCAL angular datum in frames."""

    selected = np.asarray(indices, dtype=int)
    if len(selected) > max_samples:
        positions = np.linspace(0, len(selected) - 1, max_samples).round().astype(int)
        selected = selected[np.unique(positions)]
    offsets = []
    failures = 0
    for row_idx in selected:
        omega_center = float(data.omega_from_z(data.zcal[int(row_idx)]))
        try:
            omega_bragg = _xds_header_bragg_omega(data, data.hkl[int(row_idx)], omega_center)
        except ValueError:
            failures += 1
            continue
        offsets.append((omega_bragg - omega_center) / float(data.oscillation_range))
    values = np.asarray(offsets, dtype=float)
    values = values[np.isfinite(values)]
    sampled_rows = int(len(selected))
    successful_rows = int(len(values))
    if not successful_rows:
        return {
            "status": "unavailable",
            "sampled_rows": sampled_rows,
            "successful_rows": 0,
            "failed_rows": int(failures),
        }
    median = float(np.median(values))
    p05 = float(np.quantile(values, 0.05))
    p95 = float(np.quantile(values, 0.95))
    success_fraction = successful_rows / max(1, sampled_rows)
    reliability_reasons = []
    if successful_rows < 16:
        reliability_reasons.append(f"only {successful_rows} successful geometry roots; at least 16 are required")
    if success_fraction < 0.75:
        reliability_reasons.append(f"geometry-root success fraction {success_fraction:.3f} is below 0.750")
    if p95 - p05 > 2.0:
        reliability_reasons.append(f"central offset span {p95 - p05:.3f} frames exceeds 2.000")
    if abs(median) > 5.0:
        reliability_reasons.append(f"median offset magnitude {abs(median):.3f} frames exceeds 5.000")
    return {
        "status": "unreliable" if reliability_reasons else "ok",
        "definition": "(nearest global-header Sg=0 omega - XDS ZCAL omega) / OSCILLATION_RANGE",
        "sampled_rows": sampled_rows,
        "successful_rows": successful_rows,
        "failed_rows": int(failures),
        "success_fraction": success_fraction,
        "median_offset_frames": median,
        "p05_offset_frames": p05,
        "p95_offset_frames": p95,
        "median_abs_deviation_frames": float(np.median(np.abs(values - median))),
        "reliability_reasons": reliability_reasons,
    }


def resolve_frame_offset_grid(text: str, data: XDSData, indices: np.ndarray) -> tuple[list[float], dict]:
    """Resolve explicit offsets or an XDS-geometry-informed automatic grid."""

    tokens = [token for token in str(text or "").replace(",", " ").split() if token]
    wants_auto = any(token.lower() == "auto" for token in tokens)
    explicit = [float(token) for token in tokens if token.lower() != "auto"]
    datum = xds_frame_datum_summary(data, indices) if wants_auto else {"status": "not_requested"}
    automatic = []
    if wants_auto:
        reliable_median = datum.get("median_offset_frames") if datum.get("status") == "ok" else float("nan")
        automatic = automatic_frame_offset_grid(float(reliable_median))
    values = []
    for value in [*automatic, *explicit]:
        if np.isfinite(value) and not any(abs(value - existing) <= 1e-12 for existing in values):
            values.append(float(value))
    if not values:
        values = [0.0, 0.2]
    datum["requested"] = str(text)
    datum["resolved_grid_frames"] = values
    return values, datum


def exact_hkl_merge(hkls, intensities, sigmas):
    bins: dict[tuple[int, int, int], dict[str, float]] = {}
    for hkl, I, sig in zip(hkls, intensities, sigmas, strict=True):
        h = tuple(int(v) for v in hkl)
        sig = max(abs(float(sig)), 1e-6)
        w = 1.0 / (sig * sig)
        b = bins.setdefault(h, {"w": 0.0, "wI": 0.0, "n": 0})
        b["w"] += w
        b["wI"] += w * float(I)
        b["n"] += 1
    rows = []
    for (h, k, l), b in bins.items():
        rows.append((h, k, l, b["wI"] / max(b["w"], 1e-30), (1.0 / max(b["w"], 1e-30)) ** 0.5))
    rows.sort(key=lambda r: (r[0], r[1], r[2]))
    return rows


def prediction_cache_key(data: XDSData, hkl: Path, res: Path, cfg: Config, indices, model_tag: str = "original"):
    scan_matrix_path = getattr(data, "a_scan_points_path", None)
    scan_matrix_sha256 = ""
    if scan_matrix_path:
        scan_matrix_path = Path(scan_matrix_path)
        if scan_matrix_path.exists():
            scan_matrix_sha256 = sha256(scan_matrix_path)
    payload = {
        "cache_schema": PREDICTION_CACHE_SCHEMA,
        "prediction_engine_source_sha256": PREDICTION_ENGINE_SOURCE_SHA256,
        "numerical_runtime_versions": NUMERICAL_RUNTIME_VERSIONS,
        "hkl_sha256": sha256(hkl),
        "scan_matrix_sha256": scan_matrix_sha256,
        "external_geometry": external_geometry_fingerprints(data, cfg),
        "res_sha256": sha256(res),
        "model_tag": str(model_tag),
        "hkl_transform": str(getattr(data, "hkl_transform_spec", "identity")),
        "indices_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
        "cfg": asdict(cfg),
    }
    text = canonical_prediction_cache_payload(payload)
    return hashlib.sha256(text.encode()).hexdigest()[:20], payload


def predict_with_cache(
    data,
    model,
    res_hash,
    hkl_path,
    res_path,
    cfg,
    indices,
    cache_dir: Path,
    workers: int,
    chunksize: int,
    model_tag: str = "original",
    cache_compression: str = "compressed",
    ug_cache_dir: Path | None = None,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key, payload = prediction_cache_key(data, hkl_path, res_path, cfg, indices, model_tag=model_tag)
    path = cache_dir / f"native_pred_{key}.npz"
    if path.exists():
        try:
            pred = read_prediction_cache(path)
            errors = prediction_cache_validation_errors(pred, payload, indices)
            if not errors:
                pred.pop("prediction_cache_meta", None)
                pred["cache_status"] = "hit"
                pred["cache_path"] = str(path)
                return pred
            print(f"Ignoring stale prediction cache {path}: {'; '.join(errors)}", flush=True)
        except (OSError, ValueError, TypeError, json.JSONDecodeError) as exc:
            print(f"Ignoring unreadable prediction cache {path}: {type(exc).__name__}: {exc}", flush=True)
    resolved_ug_cache_dir = (
        Path(ug_cache_dir)
        if ug_cache_dir is not None
        else runtime_cache_dir("native_row_regularize", "ug")
    )
    port = XDSBlochPort(
        data, model, cfg, cache_dir=resolved_ug_cache_dir, res_hash=res_hash
    )
    pred = port.predict_rows(indices, workers=workers, chunksize=chunksize)
    save = cacheable_prediction_arrays(pred)
    validation_view = {**save, "prediction_cache_meta": payload}
    errors = prediction_cache_validation_errors(validation_view, payload, indices)
    if errors:
        raise ValueError(f"new prediction cache failed validation: {'; '.join(errors)}")
    write_prediction_cache(path, save, payload, cache_compression)
    pred["cache_status"] = "built"
    pred["cache_path"] = str(path)
    return pred


def run_shelxl(work_dir: Path, stem: str, iterations: int = 1, wght_mode: str = "freeze-zero", endpoint_ls_cycles: int = 1):
    exe = str(SHELXL)
    if not shelxl_executable_is_available(exe):
        msg = f"SHELXL executable not found: {SHELXL}\n"
        (work_dir / "shelxl_stdout.log").write_text(msg)
        return {"returncode": None, "status": "missing_shelxl", "executable": str(SHELXL)}
    history = []
    last_safe = None
    last_proc = None
    n_iter = max(1, int(iterations))
    mode = str(wght_mode or "freeze-zero")
    freeze_zero = mode == "freeze-zero"
    if mode not in {"inherit", "freeze-zero", "promote"}:
        return {"returncode": None, "status": "invalid_wght_mode", "wght_mode": mode}
    if freeze_zero:
        write_active_wght(work_dir / f"{stem}.ins", 0.0, 0.0)
    for i in range(n_iter):
        if freeze_zero:
            write_active_wght(work_dir / f"{stem}.ins", 0.0, 0.0)
        ins = work_dir / f"{stem}.ins"
        base_ins_text = ins.read_text(errors="replace") if ins.exists() else ""
        proc, stats = run_one_shelxl_attempt(
            work_dir,
            stem,
            exe,
            iteration=i + 1,
            log_name=f"shelxl_iter_{i + 1}.log",
        )
        last_proc = proc
        if str(stats.get("status", "")) == "adp_pathology" and base_ins_text:
            recovered_proc, stats = recover_npd_shelxl_attempt(
                work_dir,
                stem,
                exe,
                iteration=i + 1,
                base_ins_text=base_ins_text,
                initial_stats=stats,
            )
            if recovered_proc is not None:
                last_proc = recovered_proc
        history.append(stats)
        if not shelxl_iteration_safe(stats):
            break
        last_safe = snapshot_safe_shelxl_iteration(work_dir, stem, i + 1, stats)
        stats["safe_snapshot_res"] = last_safe["res"]
        stats["safe_snapshot_lst"] = last_safe["lst"]
        stats["safe_snapshot_fcf"] = last_safe["fcf"]
        last_safe["stats"] = dict(stats)
        res = work_dir / f"{stem}.res"
        if i + 1 < n_iter and res.exists():
            active = (0.0, 0.0) if freeze_zero else _active_wght_from_stats(stats)
            write_endpoint_ins(res, ins, hkl_name=f"{stem}.hkl", keep_exti=True, wght=active, ls_cycles=endpoint_ls_cycles)
            if mode == "promote":
                stats["next_wght"] = promote_recommended_wght(
                    ins,
                    recommended_wght_from_stats(stats),
                )
            elif freeze_zero:
                stats["next_wght"] = {"promoted": False, "frozen": True, "active_wght_a": 0.0, "active_wght_b": 0.0}
            else:
                stats["next_wght"] = {"promoted": False, "inherited": True}
    if last_proc is not None:
        (work_dir / "shelxl_stdout.log").write_text(last_proc.stdout)
    final = select_last_safe_shelxl_iteration(work_dir, stem, history, last_safe)
    final.setdefault("res_health", shelx_res_health(work_dir / f"{stem}.res", work_dir / f"{stem}.lst"))
    if final.get("returncode") not in {None, 0}:
        final["status"] = "shelxl_failed"
        final["invalid_res_reason"] = f"SHELXL returned {final['returncode']}"
    elif not final["res_health"].get("ok"):
        final["status"] = "invalid_res_output"
        final["invalid_res_reason"] = final["res_health"].get("reason")
    else:
        active = (0.0, 0.0) if freeze_zero else _active_wght_from_stats(final)
        write_endpoint_ins(
            work_dir / f"{stem}.res",
            work_dir / f"{stem}.res",
            hkl_name=f"{stem}.hkl",
            # Keep the model that SHELXL actually refined. The initial input
            # writer decides whether this branch uses extinction.
            keep_exti=True,
            wght=active,
            ls_cycles=endpoint_ls_cycles,
        )
        final["res_normalized_for_reopen"] = True
        final["res_health"] = shelx_res_health(
            work_dir / f"{stem}.res",
            work_dir / f"{stem}.lst",
        )
        if not final["res_health"].get("ok"):
            final["status"] = "invalid_res_output"
            final["invalid_res_reason"] = final["res_health"].get("reason")
    final["iterations"] = n_iter
    final["iterations_completed"] = len(history)
    final["endpoint_ls_cycles"] = max(1, int(endpoint_ls_cycles))
    final["iteration_history"] = history
    final["wght_mode"] = mode
    pathology = final.get("res_health") or {}
    final["npd_count"] = int(pathology.get("npd_count", 0) or 0)
    final["split_count"] = int(pathology.get("split_count", 0) or 0)
    final["npd_atoms"] = pathology.get("npd_atoms", [])
    final["has_npd"] = bool(pathology.get("has_npd", False))
    final["has_split_warnings"] = bool(pathology.get("has_split_warnings", False))
    return final


def run_maps(work_dir: Path, stem: str):
    maps = {}
    exe = str(SHELX2MAP)
    if ("/" in exe and not Path(exe).exists()) or ("/" not in exe and shutil.which(exe) is None) or not (work_dir / f"{stem}.fcf").exists():
        return maps
    for label, mode in [("fofc", "0"), ("fo", "1"), ("2fofc", "2")]:
        out_map = work_dir / f"{label}.map"
        proc = subprocess.run([exe, "-m", mode, "-o", str(out_map.name), stem], cwd=str(work_dir), text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        (work_dir / f"shelx2map_{label}.log").write_text(proc.stdout)
        actual = out_map
        for line in proc.stdout.splitlines():
            m = re.search(r'Map data written to "([^"]+)"', line)
            if m:
                actual = work_dir / m.group(1)
        maps[label] = {"map": str(actual), "requested_map": str(out_map), "returncode": proc.returncode, "exists": actual.exists()}
    return maps


def export_corrected_endpoint(
    out_dir: Path,
    data: XDSData,
    target,
    sig,
    best_eval,
    masks,
    res_path: Path,
    lambda_q: float,
    hkl_format: str,
    rescale: str,
    make_maps: bool,
    endpoint_b: float = 0.0,
    keep_exti: bool = False,
    shelxl_iterations: int = 1,
    endpoint_ls_cycles: int = 1,
    post_bloch_sigma_policy: str = "none",
    post_bloch_sigma_holdout_fraction: float = 0.20,
    post_bloch_sigma_min_scale: float = 0.25,
    post_bloch_sigma_max_scale: float = 4.0,
    endpoint_wght_mode: str = "freeze-zero",
    target_mode: str = "iobs",
):
    out_dir.mkdir(parents=True, exist_ok=True)
    arrays = best_eval["arrays"]
    q = arrays["dyn_raw"] / np.maximum(arrays["born_raw"], 1e-30)
    q_clip = np.clip(q, 0.05, 20.0)
    groups = arrays["groups"]
    scales = np.full(len(target), np.nan, dtype=float)
    for g, scale in best_eval["dyn_scales"].items():
        scales[groups == int(g)] = float(scale)
    if best_eval.get("dyn_detector_scales"):
        det_bins = arrays.get("detector_bins")
        if det_bins is not None:
            for d, scale in best_eval["dyn_detector_scales"].items():
                scales[det_bins == int(d)] *= float(scale)
    endpoint_target, endpoint_sig, endpoint_restore_factor = endpoint_observation_arrays(
        data,
        target,
        sig,
        target_mode,
    )
    valid = (
        arrays["valid"]
        & np.isfinite(endpoint_target)
        & np.isfinite(endpoint_sig)
        & (np.asarray(endpoint_sig, dtype=float) > 0.0)
        & np.isfinite(arrays["dyn_scaled"])
        & (arrays["dyn_scaled"] > 0.0)
        & np.isfinite(q)
        & (q > 0.0)
        & np.isfinite(scales)
        & (scales > 0.0)
    )
    if not np.any(valid):
        raise ValueError("no finite rows with positive sigma, prediction, and fitted scale are available for endpoint export")
    denom = scales * (q_clip ** float(lambda_q))
    Icorr = endpoint_target / denom
    sigcorr = endpoint_sig / denom
    modulation, modulation_meta = endpoint_resolution_modulation(data, valid, endpoint_b)
    Icorr = Icorr * modulation
    sigcorr = sigcorr * modulation
    if rescale == "median-iobs":
        src = float(np.nanmedian(endpoint_target[valid]))
        dst = float(np.nanmedian(Icorr[valid]))
        factor = src / max(dst, 1e-30)
        Icorr *= factor
        sigcorr *= factor
    else:
        factor = 1.0
    modulation_audit = modulation_manifest_fields(
        float(factor) * modulation / np.maximum(denom, 1e-30),
        valid,
    )
    rows_unmerged = [(data.hkl[i, 0], data.hkl[i, 1], data.hkl[i, 2], Icorr[i], sigcorr[i]) for i in np.where(valid)[0]]
    rows_merged = exact_hkl_merge(data.hkl[valid], Icorr[valid], sigcorr[valid])
    hkl = out_dir / "dyn_nuisance_corrected.hkl"
    rows_merged, rows_unmerged, hkl_safe_factor = enforce_hkl_row_width(
        rows_merged, rows_unmerged, hkl_format
    )
    post_bloch_sigma = {"policy": str(post_bloch_sigma_policy), "status": "not_requested"}
    if str(post_bloch_sigma_policy) == "uniform_z_rms":
        try:
            current_model = parse_model(res_path)
            post_bloch_sigma = fit_uniform_z_rms_sigma_scale(
                rows_merged,
                current_model,
                holdout_fraction=float(post_bloch_sigma_holdout_fraction),
                min_scale=float(post_bloch_sigma_min_scale),
                max_scale=float(post_bloch_sigma_max_scale),
            )
            if str(post_bloch_sigma.get("status", "")).startswith("ok"):
                sigma_scale = float(post_bloch_sigma["sigma_scale"])
                rows_merged = scale_row_sigmas(rows_merged, sigma_scale)
                rows_unmerged = scale_row_sigmas(rows_unmerged, sigma_scale)
        except Exception as exc:
            post_bloch_sigma = {
                "policy": "uniform_z_rms",
                "status": "failed",
                "error": f"{type(exc).__name__}: {exc}",
            }
    elif str(post_bloch_sigma_policy) != "none":
        post_bloch_sigma = {
            "policy": str(post_bloch_sigma_policy),
            "status": "unknown_policy",
        }
    rows_merged, rows_unmerged, post_sigma_hkl_safe_factor = enforce_hkl_row_width(
        rows_merged, rows_unmerged, hkl_format
    )
    hkl_safe_factor *= post_sigma_hkl_safe_factor
    write_hkl(hkl, rows_merged, hkl_format)
    write_hkl(out_dir / "dyn_nuisance_corrected_unmerged.hkl", rows_unmerged, hkl_format)
    initial_wght = (0.0, 0.0) if str(endpoint_wght_mode) == "freeze-zero" else None
    write_endpoint_ins(
        res_path,
        out_dir / "dyn_nuisance_corrected.ins",
        hkl_name="dyn_nuisance_corrected.hkl",
        keep_exti=keep_exti,
        wght=initial_wght,
        ls_cycles=endpoint_ls_cycles,
    )
    shelxl = run_shelxl(
        out_dir,
        "dyn_nuisance_corrected",
        iterations=shelxl_iterations,
        wght_mode=endpoint_wght_mode,
        endpoint_ls_cycles=endpoint_ls_cycles,
    )
    maps = run_maps(out_dir, "dyn_nuisance_corrected") if make_maps else {}
    endpoint_res = out_dir / "dyn_nuisance_corrected.res"
    endpoint_lst = out_dir / "dyn_nuisance_corrected.lst"
    endpoint_hkl = out_dir / "dyn_nuisance_corrected.hkl"
    manifest = {
        "lambda_q": float(lambda_q),
        "q_min": 0.05,
        "q_max": 20.0,
        "rescale": rescale,
        "rescale_factor": factor,
        "hkl_safe_factor": hkl_safe_factor,
        "post_sigma_hkl_safe_factor": post_sigma_hkl_safe_factor,
        **modulation_audit,
        **modulation_meta,
        "keep_exti": bool(keep_exti),
        "shelxl_iterations": int(max(1, shelxl_iterations)),
        "endpoint_ls_cycles": int(max(1, endpoint_ls_cycles)),
        "endpoint_wght_mode": str(endpoint_wght_mode),
        "post_bloch_sigma": post_bloch_sigma,
        "fit_target_mode": str(target_mode),
        "endpoint_target_mode": "iobs",
        "endpoint_target_factor_restored": bool(str(target_mode) != "iobs"),
        "endpoint_target_factor_p05": float(np.nanpercentile(endpoint_restore_factor[valid], 5)),
        "endpoint_target_factor_median": float(np.nanmedian(endpoint_restore_factor[valid])),
        "endpoint_target_factor_p95": float(np.nanpercentile(endpoint_restore_factor[valid], 95)),
        "rows_unmerged": len(rows_unmerged),
        "rows_merged_exact_hkl": len(rows_merged),
        "q_median": float(np.nanmedian(q_clip[valid])),
        "q_p05": float(np.nanpercentile(q_clip[valid], 5)),
        "q_p95": float(np.nanpercentile(q_clip[valid], 95)),
        "shelxl": shelxl,
        "maps": maps,
        "artifacts": {
            "res": str(endpoint_res.resolve()),
            "res_sha256": sha256(endpoint_res) if endpoint_res.exists() else "",
            "hkl": str(endpoint_hkl.resolve()),
            "hkl_sha256": sha256(endpoint_hkl) if endpoint_hkl.exists() else "",
            "lst": str(endpoint_lst.resolve()),
            "lst_sha256": sha256(endpoint_lst) if endpoint_lst.exists() else "",
        },
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    return manifest


def copy_baseline(out_dir: Path, res_path: Path, hkl_path: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(hkl_path, out_dir / "temp_a.hkl")
    # This is a diagnostic replay of the user's conventional endpoint, not a
    # GROND corrected-HKL endpoint.  Keep the user's active WGHT and EXTI policy,
    # but normalize the copied input to a single LS/CGLS cycle so stale Olex2 or
    # SHELXLE directives such as ``L.S. 100`` cannot turn the baseline check into
    # a long re-refinement under different conditions.
    write_endpoint_ins(
        res_path,
        out_dir / "temp_a.ins",
        hkl_name="temp_a.hkl",
        keep_exti=True,
        wght=None,
        ls_cycles=1,
    )
    return run_shelxl(out_dir, "temp_a", iterations=1, wght_mode="inherit", endpoint_ls_cycles=1)


def endpoint_has_adp_pathology(endpoint: dict) -> bool:
    """Return true when a SHELXL endpoint has NPD pathology."""

    shelxl = endpoint.get("shelxl") or {}
    try:
        npd_count = int(shelxl.get("npd_count", 0) or 0)
    except (TypeError, ValueError):
        return bool(shelxl.get("has_npd"))
    return npd_count > 0 or bool(shelxl.get("has_npd"))


def sg_jitter_candidate_rejection(cfg: Config, exc: InfeasibleSgJitterProfileError) -> dict:
    """Return an auditable rejection record for one infeasible jitter candidate."""

    mode = str(cfg.sg_jitter_mode or "none").lower()
    active = bool(
        mode != "none"
        and (
            abs(float(cfg.sg_jitter_sigma0)) > 0.0
            or abs(float(cfg.sg_jitter_sigma_d_inv)) > 0.0
            or abs(float(cfg.sg_jitter_sigma_frame)) > 0.0
        )
    )
    if not active:
        raise RuntimeError(
            "an infeasible Sg-jitter profile error reached a candidate with no active jitter"
        ) from exc
    return {
        "status": "rejected",
        "reason": "infeasible_local_sg_jitter_profile",
        "message": str(exc),
        "config": asdict(cfg),
    }


def main():
    ap = build_native_parser(
        description=__doc__,
        default_hkl=DEFAULT_HKL,
        default_res=DEFAULT_RES,
        default_baseline_hkl=DEFAULT_BASELINE_HKL,
        target_modes=TARGET_MODES,
        max_support_sg=DEFAULT_MAX_SUPPORT_SG,
        shelxl=SHELXL,
        runtime_cache_directory=runtime_cache_dir,
    )
    args = ap.parse_args()
    if args.paired_inverse_hand and str(args.q_control) != "real":
        raise ValueError(
            "--paired-inverse-hand cannot be combined with a shuffled-q control"
        )
    configure_shelxl_executable(args.shelxl)
    require_endpoint_shelxl(
        args.shelxl,
        skip_endpoint_refinement=bool(args.skip_endpoint_refinement),
    )
    effective_topup_d_min = min(float(args.topup_d_min), float(args.min_d_spacing)) if args.min_d_spacing > 0 else float(args.topup_d_min)

    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    data = XDSData(Path(args.hkl), hkl_transform=args.hkl_transform)
    if getattr(data, "pets_exact_a_center", None) is not None:
        static_endpoint_validated = (
            str(data.header.get("PETS_STATIC_HKLF4_VALIDATED", ""))
            .strip()
            .upper()
            == "TRUE"
        )
        if not bool(args.skip_endpoint_refinement) and not static_endpoint_validated:
            raise ValueError(
                "native PETS rows do not have a validated row-preserving static SHELXL "
                "HKLF4 contract; add --skip-endpoint-refinement or re-import the ordinary "
                "cif_pets with PETS' matching static HKLF4 as --baseline-hkl"
            )
        if any(value > 1 for value in parse_int_grid(args.detector_scale_bins)):
            raise ValueError(
                "PETS cif_pets input has no detector coordinates; detector scale bins must be 0"
            )
        if any(
            abs(value) > 1e-12 for value in parse_float_grid(args.frame_offset_grid)
        ):
            raise ValueError(
                "PETS exact geometry owns the zone orientations; frame-offset grids are not applicable"
            )
        if str(args.profile_quadrature) != "uniform":
            raise ValueError(
                "PETS exact rocking wedges currently implement uniform direct-angle sampling only"
            )
        if str(args.mosaicity_mode) != "none" or abs(float(args.mosaicity_sigma_deg)) > 0.0:
            raise ValueError(
                "PETS exact-row mosaicity controls are not implemented and cannot be silently ignored"
            )
        if any(
            str(value).strip().lower() != "none"
            for value in str(args.sg_jitter_mode_grid).split(",")
            if str(value).strip()
        ):
            raise ValueError(
                "PETS exact-row Sg-jitter controls are not implemented and cannot be silently ignored"
            )
        if str(args.geometry_path).strip():
            raise ValueError(
                "PETS exact geometry cannot be combined with an XDS/DIALS geometry-path correction"
            )
    target, base_sig = target_arrays(data, args.target_mode)
    row_contract_summary = {}
    if str(args.row_contract).strip():
        if int(args.max_rows) > 0:
            raise ValueError("--max-rows cannot be combined with --row-contract; membership must remain exact")
        indices, row_contract_summary = row_contract_selection(data, target, Path(args.row_contract))
        datum_indices = indices
        geometry_summary = {
            "status": "not_reapplied_row_contract_is_authoritative",
            "percentile": float(args.geometry_percentile),
            "reason": (
                "The immutable row contract was already produced under the promoted geometry/quality policy. "
                "Reapplying a raw XDS filter could silently change membership."
            ),
        }
        (out / "row_contract_verification.json").write_text(
            json.dumps(row_contract_summary, indent=2, allow_nan=True) + "\n"
        )
    else:
        base_pre_geom = np.ones(len(target), dtype=bool)
        if args.positive_only:
            base_pre_geom &= target > 0.0
        if args.min_peak > 0:
            base_pre_geom &= data.peak >= args.min_peak
        if args.min_d_spacing > 0:
            dvals_for_geom = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
            base_pre_geom &= dvals_for_geom >= args.min_d_spacing
        _geom_keep, geometry_summary = geometry_tail_mask(data, base_pre_geom, args.geometry_percentile)
        datum_indices = row_selection(
            data,
            target,
            args.positive_only,
            args.min_d_spacing,
            args.min_peak,
            0,
            args.geometry_percentile,
        )
        indices = row_selection(data, target, args.positive_only, args.min_d_spacing, args.min_peak, args.max_rows, args.geometry_percentile)
    model = parse_model(Path(args.res))
    split_contract_summary = {}
    if str(args.split_contract).strip():
        masks, split_contract_summary = load_transform_split_contract_masks(
            Path(args.split_contract),
            data,
            model,
        )
        if not math.isclose(
            float(args.free_fraction),
            float(split_contract_summary["free_fraction"]),
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            raise ValueError(
                "--free-fraction does not match the immutable transform/hand split contract"
            )
        if int(args.split_seed) != int(split_contract_summary["split_seed"]):
            raise ValueError(
                "--split-seed does not match the immutable transform/hand split contract"
            )
    else:
        masks = split_masks(
            data.hkl,
            args.free_fraction,
            args.split_seed,
            symmetry_ops=model.get("symmetry_ops"),
        )
    selected_mask = np.zeros(len(target), dtype=bool)
    selected_mask[indices] = True
    masks = {k: v & selected_mask for k, v in masks.items()}
    model_tag = "alternate_fractional_xyz" if args.alternate_hand else "original"
    model_tag = f"{model_tag};hkl_transform={data.hkl_transform_spec}"
    if args.alternate_hand:
        model = alternate_hand_model(model)
    res_hash = sha256(Path(args.res))
    wght_a, wght_b = parse_wght(Path(args.res))
    wght_a_values = parse_float_grid(args.wght_a_grid) if str(args.wght_a_grid).strip() else [wght_a]
    wght_b_values = parse_float_grid(args.wght_b_grid) if str(args.wght_b_grid).strip() else [wght_b]
    geom_trials = []
    all_trials = []
    rejected_prediction_candidates = []
    print(f"selected {len(indices)} rows; WGHT=({wght_a:g}, {wght_b:g})", flush=True)

    thickness_models = [m.strip() for m in str(args.thickness_model_grid).replace(",", " ").split() if m.strip()]
    absorption_modes = [m.strip() for m in str(args.absorption_mode_grid).replace(",", " ").split() if m.strip()]
    dose_modes = [m.strip() for m in str(args.dose_b_mode_grid).replace(",", " ").split() if m.strip()]
    sg_jitter_modes = [m.strip() for m in str(args.sg_jitter_mode_grid).replace(",", " ").split() if m.strip()]
    group_width_values = parse_group_width_grid(args.group_width_grid, data)
    detector_scale_bins_values = parse_int_grid(args.detector_scale_bins)
    frame_offset_values, frame_datum = resolve_frame_offset_grid(args.frame_offset_grid, data, datum_indices)
    row_plan_cache_decision = resolve_fixed_model_row_plan_cache(
        args.row_plan_cache,
        estimated_reuse=estimated_row_plan_reuse_per_geometry_key(
            args,
            thickness_models,
            absorption_modes,
        ),
    )
    print(
        "row-plan cache: "
        f"requested={row_plan_cache_decision['requested']} "
        f"resolved={row_plan_cache_decision['resolved']} "
        "estimated_reuse_per_geometry_key="
        f"{row_plan_cache_decision['estimated_reuse_per_geometry_key']} "
        f"auto_threshold={row_plan_cache_decision['auto_exact_minimum_reuse']}",
        flush=True,
    )
    if str(frame_datum.get("status")) == "ok":
        print(
            "XDS frame datum: median="
            f"{float(frame_datum['median_offset_frames']):.4f} frames; "
            f"resolved grid={frame_offset_values}",
            flush=True,
        )
    populate_nuisance_trials(
        args=args,
        thickness_models=thickness_models,
        frame_offset_values=frame_offset_values,
        sg_jitter_modes=sg_jitter_modes,
        absorption_modes=absorption_modes,
        dose_modes=dose_modes,
        effective_topup_d_min=effective_topup_d_min,
        group_width_values=group_width_values,
        row_plan_cache_decision=row_plan_cache_decision,
        predict_with_cache=predict_with_cache,
        data=data,
        model=model,
        res_hash=res_hash,
        indices=indices,
        model_tag=model_tag,
        sg_jitter_candidate_rejection=sg_jitter_candidate_rejection,
        rejected_prediction_candidates=rejected_prediction_candidates,
        geom_trials=geom_trials,
        detector_scale_bins_values=detector_scale_bins_values,
        wght_a_values=wght_a_values,
        wght_b_values=wght_b_values,
        evaluate_trial=evaluate_trial,
        target=target,
        base_sig=base_sig,
        masks=masks,
        wght_a=wght_a,
        wght_b=wght_b,
        all_trials=all_trials,
    )

    if rejected_prediction_candidates:
        (out / "rejected_prediction_candidates.json").write_text(
            json.dumps(rejected_prediction_candidates, indent=2, allow_nan=True) + "\n"
        )
    if not all_trials:
        failure = {
            "reason": "no_feasible_prediction_candidates",
            "message": (
                "Every requested nuisance prediction candidate failed a physical contract. "
                "See rejected_prediction_candidates.json for each configuration and exact error: "
                + "; ".join(dict.fromkeys(str(r.get("message", "")) for r in rejected_prediction_candidates))
            ),
            "rejected_prediction_candidates": rejected_prediction_candidates,
        }
        (out / "prediction_grid_failure.json").write_text(
            json.dumps(failure, indent=2, allow_nan=True) + "\n"
        )
        raise RuntimeError(failure["message"])

    all_trials_sorted = rank_trials(all_trials, args.selection_metric)
    ranked_candidate = all_trials_sorted[0]
    promotion_decision = {
        "family": "discrete_thickness_polish",
        "status": "not_requested",
        "promoted": False,
    }
    best = ranked_candidate
    if str(args.promotion_baseline_json).strip():
        try:
            promotion_baseline = json.loads(args.promotion_baseline_json)
        except (TypeError, json.JSONDecodeError) as exc:
            raise RuntimeError("invalid --promotion-baseline-json contract") from exc
        if not isinstance(promotion_baseline, dict):
            raise RuntimeError("--promotion-baseline-json must decode to an object")
        best, promotion_decision = select_trial_with_promotion_gate(
            all_trials_sorted,
            promotion_baseline,
            selection_metric=args.selection_metric,
            min_gain=args.promotion_min_gain,
            max_free_wr2_loss=args.promotion_max_free_wr2_loss,
            max_all_r1_loss=args.promotion_max_all_r1_loss,
        )
        (out / "discrete_thickness_promotion.json").write_text(
            json.dumps(promotion_decision, indent=2, allow_nan=True) + "\n"
        )
    if args.fail_if_born_r1gt_above > 0.0 and best["metrics"]["all"]["born"]["R1_gt"] > float(args.fail_if_born_r1gt_above):
        failure = {
            "reason": "matched_born_gate_failed",
            "message": "Matched Born is too poor for a meaningful Bloch/nuisance search. Check HKL transform, hand, cell basis, and geometry before changing physics knobs.",
            "threshold_R1_gt": float(args.fail_if_born_r1gt_above),
            "born_R1_gt": float(best["metrics"]["all"]["born"]["R1_gt"]),
            "born_R1_all": float(best["metrics"]["all"]["born"]["R1_all"]),
            "born_wR2_all": float(best["metrics"]["all"]["born"]["wR2_all"]),
            "born_CC": float(best["metrics"]["all"]["born"]["CC"]),
            "hkl_transform": data.hkl_transform_spec,
            "raw_xds_cell": [data.raw_cell.a, data.raw_cell.b, data.raw_cell.c, data.raw_cell.alpha, data.raw_cell.beta, data.raw_cell.gamma],
            "axis_derived_source_cell": [
                data.axis_derived_source_cell.a,
                data.axis_derived_source_cell.b,
                data.axis_derived_source_cell.c,
                data.axis_derived_source_cell.alpha,
                data.axis_derived_source_cell.beta,
                data.axis_derived_source_cell.gamma,
            ],
            "axis_derived_source_cell_basis": data.axis_derived_source_cell_basis,
            "simulation_cell": [data.cell.a, data.cell.b, data.cell.c, data.cell.alpha, data.cell.beta, data.cell.gamma],
            "simulation_cell_basis": data.working_cell_basis,
        }
        (out / "gate_failure.json").write_text(json.dumps(failure, indent=2, allow_nan=True) + "\n")
        raise RuntimeError(
            f"matched Born gate failed: R1gt={failure['born_R1_gt']:.4f} > "
            f"{failure['threshold_R1_gt']:.4f}; hkl_transform={data.hkl_transform_spec}"
        )
    promotion_requested = bool(str(args.promotion_baseline_json).strip())
    rows = []
    for selection_rank, trial in enumerate(all_trials_sorted, start=1):
        roles = []
        if trial is ranked_candidate:
            roles.append("ranked_candidate")
        if promotion_requested and nuisance_reference_matches(
            promotion_baseline, trial.get("nuisance") or {}
        ):
            roles.append("promotion_baseline")
        if trial is best:
            roles.append("selected_output")
        rows.append(
            {
                "selection_rank": selection_rank,
                "promotion_role": ";".join(roles),
                "selected_for_output": int(trial is best),
                **metric_row(trial),
            }
        )
    with (out / "nuisance_trials.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    (out / "best_native_fit.json").write_text(json.dumps({k: v for k, v in best.items() if k != "arrays"}, indent=2, allow_nan=True) + "\n")
    native_row_table = out / "native_dyn_calc.csv"
    write_row_table(native_row_table, data, target, best, masks, target_mode=args.target_mode)
    native_row_artifact = native_row_table_artifact(native_row_table)
    paired_inverse_hand = {}
    if args.paired_inverse_hand:
        inverse_pred = paired_inverse_prediction(best)
        paired_indices = inverse_pred["indices"]
        inverse_eval = evaluate_trial(
            data,
            target,
            base_sig,
            inverse_pred,
            paired_indices,
            masks,
            Nuisance(**best["nuisance"]),
            wght_a,
            wght_b,
            scale_fit_split=args.scale_fit_split,
            q_control_args=args,
        )
        inverse_row_table = out / "native_inverse_hand_dyn_calc.csv"
        write_row_table(inverse_row_table, data, target, inverse_eval, masks, target_mode=args.target_mode)
        paired_inverse_hand = {
            "status": "complete",
            "method": "nakane_negative_thickness_elastic_pair_v1",
            "physics_contract": {
                "absorption": "disabled",
                "incident_wave": "real direct-beam basis vector",
                "same_rows_basis_geometry_nuisance": True,
                "same_positive_physical_thickness_distribution": True,
                "inverse_evaluation": "I_inverse(t)=I_original(-t)",
                "shared_eigendecomposition": True,
            },
            "selected_row_table": str(native_row_table.resolve()),
            "inverse_row_table": str(inverse_row_table.resolve()),
            "inverse_row_table_sha256": sha256(inverse_row_table),
            "selected_metrics": best["metrics"],
            "inverse_metrics": inverse_eval["metrics"],
            "selected_scale_count": int(best["scale_count"]),
            "inverse_scale_count": int(inverse_eval["scale_count"]),
        }
        (out / "paired_inverse_hand.json").write_text(
            json.dumps(paired_inverse_hand, indent=2, allow_nan=True) + "\n"
        )

    endpoint_results = []
    endpoint_exti_modes = [m.strip().lower() for m in str(args.endpoint_exti_grid).replace(",", " ").split() if m.strip()]
    if not endpoint_exti_modes:
        endpoint_exti_modes = ["off"]
    for mode in endpoint_exti_modes:
        if mode not in {"off", "on", "false", "true", "0", "1", "no", "yes"}:
            raise ValueError(f"unknown endpoint EXTI mode {mode!r}; use off,on")
    if not args.skip_endpoint_refinement:
        for lam in parse_float_grid(args.lambda_grid):
            for endpoint_b in parse_float_grid(args.endpoint_b_grid):
                for exti_mode in endpoint_exti_modes:
                    keep_exti = exti_mode in {"on", "true", "1", "yes"}
                    lam_tag = str(lam).replace(".", "p")
                    b_tag = str(endpoint_b).replace("-", "m").replace(".", "p")
                    exti_tag = "exti" if keep_exti else "noexti"
                    ep_dir = out / f"endpoint_lam_{lam_tag}_B_{b_tag}_{exti_tag}"
                    endpoint_results.append(
                        export_corrected_endpoint(
                            ep_dir,
                            data,
                            target,
                            best["arrays"]["sigma_eff"],
                            best,
                            masks,
                            Path(args.res),
                            lam,
                            args.hkl_float_format,
                            args.rescale,
                            args.make_maps,
                            endpoint_b=endpoint_b,
                            keep_exti=keep_exti,
                            shelxl_iterations=args.shelxl_iterations,
                            endpoint_ls_cycles=args.endpoint_ls_cycles,
                            post_bloch_sigma_policy=args.post_bloch_sigma_policy,
                            post_bloch_sigma_holdout_fraction=args.post_bloch_sigma_holdout_fraction,
                            post_bloch_sigma_min_scale=args.post_bloch_sigma_min_scale,
                            post_bloch_sigma_max_scale=args.post_bloch_sigma_max_scale,
                            endpoint_wght_mode=args.endpoint_wght_mode,
                            target_mode=args.target_mode,
                        )
                    )
    fail_if_shelxl_became_unavailable(
        endpoint_results,
        context="corrected-HKL endpoint refinement",
    )
    endpoint_results = sorted(endpoint_results, key=lambda r: (r["shelxl"].get("wR2_all", 999), r["shelxl"].get("R1_all", 999)))
    no_exti_endpoint_results = [ep for ep in endpoint_results if not bool(ep.get("keep_exti", False))]
    safe_no_exti_endpoint_results = [ep for ep in no_exti_endpoint_results if not endpoint_has_adp_pathology(ep)]
    neutral_endpoint_results = [
        ep for ep in no_exti_endpoint_results
        if abs(float(ep.get("endpoint_b", 0.0))) < 1e-12
    ]
    safe_neutral_endpoint_results = [
        ep for ep in safe_no_exti_endpoint_results
        if abs(float(ep.get("endpoint_b", 0.0))) < 1e-12
    ]
    final_endpoint_results = safe_neutral_endpoint_results or safe_no_exti_endpoint_results or neutral_endpoint_results or no_exti_endpoint_results

    baseline = {}
    if not args.skip_endpoint_refinement and Path(args.baseline_hkl).exists():
        baseline = copy_baseline(out / "baseline_exti", Path(args.res), Path(args.baseline_hkl))
        fail_if_shelxl_became_unavailable(
            [{"shelxl": baseline}],
            context="matched conventional baseline replay",
        )
    endpoint_factor_decision = {}
    if no_exti_endpoint_results:
        metric_best_ep = no_exti_endpoint_results[0]
        final_ep = final_endpoint_results[0] if final_endpoint_results else metric_best_ep
        best_r1_ep = min(
            no_exti_endpoint_results,
            key=lambda r: (
                r["shelxl"].get("R1_all", 999),
                r["shelxl"].get("wR2_all", 999),
            ),
        )
        neutral = min(neutral_endpoint_results, key=lambda r: (r["shelxl"].get("wR2_all", 999), r["shelxl"].get("R1_all", 999))) if neutral_endpoint_results else {}
        best_sh = metric_best_ep.get("shelxl", {})
        best_r1_sh = best_r1_ep.get("shelxl", {})
        final_sh = final_ep.get("shelxl", {})
        neutral_sh = neutral.get("shelxl", {}) if neutral else {}
        balanced_wR2_tolerance = 0.005
        minimum_hkl_safe_factor = 0.25
        best_wR2 = float(best_sh.get("wR2_all", float("nan")))
        balanced_candidates = [
            ep for ep in (safe_neutral_endpoint_results or neutral_endpoint_results)
            if (
                math.isfinite(float(ep.get("hkl_safe_factor", 1.0)))
                and float(ep.get("hkl_safe_factor", 1.0)) > 0.0
                and float(ep.get("shelxl", {}).get("wR2_all", 999)) <= best_wR2 + balanced_wR2_tolerance
            )
        ]
        balanced_ep = min(
            balanced_candidates,
            key=lambda r: (
                r["shelxl"].get("R1_all", 999),
                r["shelxl"].get("wR2_all", 999),
                abs(float(r.get("endpoint_b", 0.0))),
            ),
        ) if balanced_candidates else final_ep
        balanced_sh = balanced_ep.get("shelxl", {})
        best_r1_is_diagnostic = bool(
            best_r1_ep is not balanced_ep
            and (
                abs(float(best_r1_ep.get("endpoint_b", 0.0))) > 1e-12
                or not math.isfinite(float(best_r1_ep.get("hkl_safe_factor", 1.0)))
                or float(best_r1_ep.get("hkl_safe_factor", 1.0)) <= 0.0
                or float(best_r1_sh.get("wR2_all", 999)) > best_wR2 + balanced_wR2_tolerance
            )
        )
        r1_gain = safe_gain = float("nan")
        if neutral_sh:
            r1_gain = float(neutral_sh.get("R1_all", float("nan"))) - float(best_sh.get("R1_all", float("nan")))
            safe_gain = float(neutral_sh.get("wR2_all", float("nan"))) - float(best_sh.get("wR2_all", float("nan")))
        endpoint_b_diagnostic_best = abs(float(metric_best_ep.get("endpoint_b", 0.0))) > 1e-12
        endpoint_factor_decision = {
            "best_lambda_q": metric_best_ep.get("lambda_q"),
            "best_endpoint_b": metric_best_ep.get("endpoint_b", 0.0),
            "best_keep_exti": bool(metric_best_ep.get("keep_exti", False)),
            "final_lambda_q": final_ep.get("lambda_q"),
            "final_endpoint_b": final_ep.get("endpoint_b", 0.0),
            "final_R1_all": final_sh.get("R1_all"),
            "final_wR2_all": final_sh.get("wR2_all"),
            "final_endpoint_has_adp_pathology": endpoint_has_adp_pathology(final_ep),
            "safe_no_exti_endpoint_count": len(safe_no_exti_endpoint_results),
            "safe_neutral_endpoint_count": len(safe_neutral_endpoint_results),
            "endpoint_b_policy": "diagnostic_only_prefer_neutral",
            "balanced_lambda_q": balanced_ep.get("lambda_q"),
            "balanced_endpoint_b": balanced_ep.get("endpoint_b", 0.0),
            "balanced_R1_all": balanced_sh.get("R1_all"),
            "balanced_wR2_all": balanced_sh.get("wR2_all"),
            "best_R1_all_lambda_q": best_r1_ep.get("lambda_q"),
            "best_R1_all_endpoint_b": best_r1_ep.get("endpoint_b", 0.0),
            "best_R1_all": best_r1_sh.get("R1_all"),
            "best_R1_all_wR2_all": best_r1_sh.get("wR2_all"),
            "best_R1_all_hkl_safe_factor": best_r1_ep.get("hkl_safe_factor", 1.0),
            "best_R1_all_diagnostic_only": best_r1_is_diagnostic,
            "balanced_wR2_tolerance": balanced_wR2_tolerance,
            "minimum_hkl_safe_factor": minimum_hkl_safe_factor,
            "neutral_R1_all": neutral_sh.get("R1_all") if neutral_sh else None,
            "neutral_wR2_all": neutral_sh.get("wR2_all") if neutral_sh else None,
            "delta_R1_all_neutral_minus_best": r1_gain,
            "delta_wR2_all_neutral_minus_best": safe_gain,
            "nonzero_endpoint_b_improves_endpoint_metrics": endpoint_b_diagnostic_best,
            "promote_endpoint_resolution_scale": False,
            "recommendation": "use the neutral B=0 no-EXTI endpoint as final; nonzero endpoint-B branches are diagnostic only because they can improve R factors while distorting ADPs and maps" if neutral_sh else "nonzero endpoint-B was explicitly requested without a neutral B=0 endpoint; treat this output as diagnostic unless independently justified",
        }
    kernel_variance = kernel_variance_terms(best["nuisance"])
    continuation_used = np.asarray(
        best["arrays"].get(
            "sg_profile_same_branch_continuation_used",
            np.zeros(len(target), dtype=bool),
        ),
        dtype=bool,
    )
    continuation_segments = np.asarray(
        best["arrays"].get(
            "sg_profile_same_branch_continuation_segments",
            np.ones(len(target), dtype=int),
        ),
        dtype=int,
    )
    predicted_mask = np.isfinite(np.asarray(best["arrays"].get("dyn_raw", np.full(len(target), np.nan))))
    omega_bragg_root_available = np.asarray(
        best["arrays"].get(
            "omega_bragg_root_available",
            np.isfinite(
                np.asarray(
                    best["arrays"].get("omega_bragg", np.full(len(target), np.nan))
                )
            ),
        ),
        dtype=bool,
    )
    valid_continuation_mask = np.asarray(best["arrays"]["valid"], dtype=bool) & continuation_used
    profile_root_solver_audit = {
        "contract": "strict_bragg_same_branch_base_profile_v1",
        "bragg_crossing": "strict_local_bound",
        "base_profile": "exact_target_strict_step_continuation_on_one_derivative_sign_certified_branch",
        "active_jitter": "strict_local_bound_or_reject_candidate",
        "prediction_cache_schema": PREDICTION_CACHE_SCHEMA,
        "predicted_rows": int(np.count_nonzero(predicted_mask)),
        "predicted_rows_with_local_zero_sg_diagnostic": int(
            np.count_nonzero(predicted_mask & omega_bragg_root_available)
        ),
        "predicted_rows_without_local_zero_sg_diagnostic": int(
            np.count_nonzero(predicted_mask & ~omega_bragg_root_available)
        ),
        "unavailable_zero_sg_diagnostic_policy": (
            "retain row; report omega_bragg as NaN because propagation uses the "
            "recorded scan-centered Sg profile, not this diagnostic"
        ),
        "predicted_rows_using_same_branch_continuation": int(
            np.count_nonzero(predicted_mask & continuation_used)
        ),
        "valid_output_rows_using_same_branch_continuation": int(
            np.count_nonzero(valid_continuation_mask)
        ),
        "maximum_strict_local_segments_used": int(
            np.max(continuation_segments[continuation_used])
        )
        if np.any(continuation_used)
        else 1,
    }
    summary = {
        "artifacts": native_row_artifact,
        "paired_inverse_hand": paired_inverse_hand,
        "inputs": {
            "hkl": str(Path(args.hkl).resolve()),
            "hkl_sha256": sha256(Path(args.hkl)),
            "hkl_transform": data.hkl_transform_spec,
            "raw_xds_cell": [
                data.raw_cell.a,
                data.raw_cell.b,
                data.raw_cell.c,
                data.raw_cell.alpha,
                data.raw_cell.beta,
                data.raw_cell.gamma,
            ],
            "axis_derived_source_cell": [
                data.axis_derived_source_cell.a,
                data.axis_derived_source_cell.b,
                data.axis_derived_source_cell.c,
                data.axis_derived_source_cell.alpha,
                data.axis_derived_source_cell.beta,
                data.axis_derived_source_cell.gamma,
            ],
            "axis_derived_source_cell_basis": data.axis_derived_source_cell_basis,
            "simulation_cell": [
                data.cell.a,
                data.cell.b,
                data.cell.c,
                data.cell.alpha,
                data.cell.beta,
                data.cell.gamma,
            ],
            "simulation_cell_basis": data.working_cell_basis,
            "res": str(Path(args.res).resolve()),
            "res_sha256": sha256(Path(args.res)),
            "model_tag": model_tag,
            "baseline_hkl": str(Path(args.baseline_hkl).resolve()) if Path(args.baseline_hkl).exists() else "",
            "row_contract": row_contract_summary,
        },
        "config": vars(args),
        "resolved_grids": {
            "nominal_frame_width_deg": nominal_frame_width_deg(data),
            "group_width_deg": group_width_values,
            "frame_offset_frames": frame_offset_values,
            "xds_frame_datum": frame_datum,
            "row_plan_cache": row_plan_cache_decision,
        },
        "selection_metric": str(args.selection_metric),
        "nuisance_trial_count": int(len(all_trials)),
        "prediction_geometry_trial_count": int(len(geom_trials)),
        "shelx_weight": {
            "source_a": wght_a,
            "source_b": wght_b,
            "selected_native_a": best["native_weight"]["a"],
            "selected_native_b": best["native_weight"]["b"],
            "endpoint_wght_mode": args.endpoint_wght_mode,
        },
        "bloch_kernel_contract": frozen_kernel_contract(
            Config(
                n_basis=args.n_basis,
                n_points=args.n_points,
                sg_window=args.sg_window,
                sg_window_mode=args.sg_window_mode,
                sg_window_min=args.sg_window_min,
                sg_window_max=args.sg_window_max,
                sg_window_scale=args.sg_window_scale,
                sg_window_frame_floor=args.sg_window_frame_floor,
                sg_jitter_mode=best["nuisance"]["sg_jitter_mode"],
                sg_jitter_sigma0=best["nuisance"]["sg_jitter_sigma0"],
                sg_jitter_sigma_d_inv=best["nuisance"]["sg_jitter_sigma_d_inv"],
                sg_jitter_sigma_frame=best["nuisance"]["sg_jitter_sigma_frame"],
                sg_jitter_points=args.sg_jitter_points,
                sg_jitter_student_df=args.sg_jitter_student_df,
                sg_jitter_width_factor=args.sg_jitter_width_factor,
                mosaicity_mode=best["nuisance"].get("mosaicity_mode", "none"),
                mosaicity_sigma_deg=best["nuisance"].get("mosaicity_sigma_deg", 0.0),
                mosaicity_points=best["nuisance"].get("mosaicity_points", 1),
                mosaicity_support_sigma_deg=best["nuisance"].get(
                    "mosaicity_support_sigma_deg", 0.0
                ),
                mosaicity_support_points=best["nuisance"].get(
                    "mosaicity_support_points", 5
                ),
                mosaicity_support_width_sigma=best["nuisance"].get(
                    "mosaicity_support_width_sigma", 4.0
                ),
                geometry_path_file=best["nuisance"].get("geometry_path_file", ""),
                profile_basis_mode=args.profile_basis_mode,
                mosaicity_max_sigma_deg=args.mosaicity_max_sigma_deg,
                profile_coordinate=args.profile_coordinate,
                profile_angle_max_half_width_deg=args.profile_angle_max_half_width_deg,
                profile_quadrature=args.profile_quadrature,
                profile_accumulation=args.profile_accumulation,
                topup_d_min=effective_topup_d_min,
                max_support_sg=args.max_support_sg,
                thickness_A=best["nuisance"]["thickness_A"],
                thickness_model=best["nuisance"]["thickness_model"],
                thickness_grad_frac=best["nuisance"]["thickness_grad_frac"],
                thickness_sigma_frac=best["nuisance"]["thickness_sigma_frac"],
                thickness_points=args.thickness_points,
                group_width_deg=best["nuisance"]["group_width_deg"],
                free_fraction=args.free_fraction,
                split_seed=args.split_seed,
                variance_a=kernel_variance["variance_a"],
                variance_b_corr=kernel_variance["variance_b_corr"],
                variance_c_peak=kernel_variance["variance_c_peak"],
                frame_offset=best["nuisance"]["frame_offset"],
                absorption_mode=best["nuisance"]["absorption_mode"],
                absorption_mu=best["nuisance"]["absorption_mu"],
                absorption_frac=best["nuisance"]["absorption_frac"],
                absorption_mu_g2=best["nuisance"].get("absorption_mu_g2", 0.0),
                absorption_mu_theta2=best["nuisance"].get("absorption_mu_theta2", 0.0),
                absorption_tds_b=best["nuisance"].get("absorption_tds_b", 0.0),
                dose_b_mode=best["nuisance"]["dose_b_mode"],
                dose_b_per_frame=best["nuisance"]["dose_b_per_frame"],
                dose_b_ref_frame=best["nuisance"]["dose_b_ref_frame"],
                compute_backend=args.compute_backend,
                jax_batch_size=args.jax_batch_size,
                jax_matrix_bucket_size=args.jax_matrix_bucket_size,
                bethe_strong_count=args.bethe_strong_count,
                bethe_variant=args.bethe_variant,
                bethe_promote_denom=args.bethe_promote_denom,
                bethe_denom_eps=args.bethe_denom_eps,
                matrix_convention=args.matrix_convention,
                row_plan_cache=str(row_plan_cache_decision["resolved"]),
                paired_inverse_hand=args.paired_inverse_hand,
            )
        ),
        "profile_root_solver": profile_root_solver_audit,
        "row_selection": {
            "selected_rows": int(len(indices)),
            "work_rows": int(np.count_nonzero(masks["work"])),
            "free_rows": int(np.count_nonzero(masks["free"])),
            "geometry_filter": geometry_summary,
            "contract": row_contract_summary,
            "transform_hand_split_contract": split_contract_summary,
        },
        "best_native": {k: v for k, v in best.items() if k != "arrays"},
        "selection_ranked_best_native": {k: v for k, v in ranked_candidate.items() if k != "arrays"},
        "promotion_decision": promotion_decision,
        "rejected_prediction_candidates": rejected_prediction_candidates,
        "endpoint_results_sorted": endpoint_results,
        "best_endpoint": final_endpoint_results[0] if final_endpoint_results else {},
        "skip_endpoint_refinement": bool(args.skip_endpoint_refinement),
        "endpoint_factor_decision": endpoint_factor_decision,
        "baseline_exti": baseline,
        "wall_seconds": time.perf_counter() - t0,
        "caveat": "This is fixed-structure nuisance regularization. Coordinates, ADPs, occupancies, and restraints still require a cctbx/SHELXL gradient bridge.",
    }
    native_gate = {
        "bloch_beats_born_all_wR2": bool(best["metrics"]["all"]["dyn"]["wR2_all"] < best["metrics"]["all"]["born"]["wR2_all"]),
        "bloch_beats_born_free_wR2": bool(best["metrics"]["free"]["dyn"]["wR2_all"] < best["metrics"]["free"]["born"]["wR2_all"]),
        "native_dyn_wR2_all_below_0p20": bool(best["metrics"]["all"]["dyn"]["wR2_all"] < 0.20),
    }
    endpoint_gate = {}
    if baseline and final_endpoint_results:
        sh = final_endpoint_results[0]["shelxl"]
        endpoint_gate = {
            "endpoint_beats_baseline_R1gt": bool(sh.get("R1_gt", 999) < baseline.get("R1_gt", -999)),
            "endpoint_beats_baseline_R1all": bool(sh.get("R1_all", 999) < baseline.get("R1_all", -999)),
            "endpoint_beats_baseline_wR2all": bool(sh.get("wR2_all", 999) < baseline.get("wR2_all", -999)),
            "same_resolution_cutoff_not_same_row_set": True,
        }
    summary["gates"] = {"native": native_gate, "endpoint": endpoint_gate}
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    lines = [
        "# sample Native Nuisance Regularization",
        "",
        f"Rows selected: `{len(indices)}`. N/Q: `{args.n_basis}/{args.n_points}`. Target: `{args.target_mode}`. Resolution cutoff: `d >= {args.min_d_spacing}`.",
        "",
        f"The native fit optimizes fixed-structure nuisance parameters with scales fit on `{args.scale_fit_split}` rows and selects the reported trial by `{args.selection_metric}`. Corrected-HKL endpoint statistics are from SHELXL over the requested lambda, endpoint-B, and EXTI endpoint grid; no-EXTI B=0 endpoints are the normal promotable final outputs, while nonzero endpoint-B branches are diagnostic-only unless explicitly forced by omitting B=0 from the grid.",
        f"Native row weights use `w = 1/[sigma_eff^2 + (aP)^2 + bP]`; source RES WGHT was `a={wght_a:g}, b={wght_b:g}`, and the selected native row-weight values are `a={best['native_weight']['a']:g}, b={best['native_weight']['b']:g}`.",
        "",
        "## Best Native Row Fit",
        "",
        "| split | model | R1 all | R1 gt | R2 all | wR2 all | GooF | mwR | CC | n / n_gt |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("work", "free", "all"):
        for model_name in ("born", "dyn"):
            m = best["metrics"][split][model_name]
            lines.append(
                f"| {split} | {model_name} | {100*m['R1_all']:.2f} | {100*m['R1_gt']:.2f} | "
                f"{100*m['R2_all']:.2f} | {100*m['wR2_all']:.2f} | {m['GooF']:.3f} | "
                f"{m['mean_weighted_abs_residual']:.3f} | {m['CC']:.4f} | {m['n']} / {m['n_gt']} |"
            )
    lines.extend([
        "",
        "Best nuisance parameters:",
        "",
        "```json",
        json.dumps(best["nuisance"], indent=2),
        "```",
        "",
        "## Corrected-HKL Endpoint",
        "",
        "| lambda_q | endpoint B | EXTI | post-Bloch sigma | z RMS held-out | R1gt % | R1all % | wR2all % | GooF | Ngt/Nall | unique | q p05/median/p95 |",
        "|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for ep in endpoint_results:
        sh = ep["shelxl"]
        sig_meta = ep.get("post_bloch_sigma") or {}
        sig_scale = sig_meta.get("sigma_scale", 1.0 if sig_meta.get("status") == "not_requested" else float("nan"))
        heldout_z = sig_meta.get("heldout_rms_z_after", sig_meta.get("all_rms_z_after", float("nan")))
        lines.append(
            f"| {ep['lambda_q']:.2f} | {ep.get('endpoint_b', 0.0):.2f} | {'on' if ep.get('keep_exti') else 'off'} | "
            f"{sig_scale:.3f} | {heldout_z:.3f} | "
            f"{100*sh.get('R1_gt', float('nan')):.2f} | {100*sh.get('R1_all', float('nan')):.2f} | "
            f"{100*sh.get('wR2_all', float('nan')):.2f} | {sh.get('GooF', float('nan')):.3f} | "
            f"{sh.get('N_gt', '')}/{sh.get('N_all', '')} | {sh.get('unique', '')} | "
            f"{ep['q_p05']:.3f}/{ep['q_median']:.3f}/{ep['q_p95']:.3f} |"
        )
    def _percent_or_nan(value):
        try:
            if value is None:
                return float("nan")
            return 100.0 * float(value)
        except (TypeError, ValueError):
            return float("nan")

    if endpoint_factor_decision:
        lines.extend([
            "",
            "Endpoint factor decision:",
            "",
            f"`{endpoint_factor_decision.get('recommendation', '')}`. Neutral-to-best deltas: "
            f"`dR1all={_percent_or_nan(endpoint_factor_decision.get('delta_R1_all_neutral_minus_best')):.2f}%`, "
            f"`dwR2all={_percent_or_nan(endpoint_factor_decision.get('delta_wR2_all_neutral_minus_best')):.2f}%`.",
            f"Final no-EXTI endpoint under the B=0 policy and wR2/sigma-safety gate: "
            f"`lambda={endpoint_factor_decision.get('balanced_lambda_q', float('nan'))}`, "
            f"`B={endpoint_factor_decision.get('balanced_endpoint_b', float('nan'))}`, "
            f"`R1all={_percent_or_nan(endpoint_factor_decision.get('balanced_R1_all')):.2f}%`, "
            f"`wR2all={_percent_or_nan(endpoint_factor_decision.get('balanced_wR2_all')):.2f}%`. "
            f"Lowest-R1 endpoint is diagnostic-only: `{endpoint_factor_decision.get('best_R1_all_diagnostic_only', False)}`.",
        ])
    if baseline:
        best_sh = final_endpoint_results[0]["shelxl"] if final_endpoint_results else {}
        lines.extend([
            "",
            "Baseline EXTI rerun from the same `temp_a.res/temp_a.hkl`:",
            "",
            f"`R1gt={100*baseline.get('R1_gt', float('nan')):.2f}%`, `R1all={100*baseline.get('R1_all', float('nan')):.2f}%`, `wR2all={100*baseline.get('wR2_all', float('nan')):.2f}%`, `GooF={baseline.get('GooF', float('nan')):.3f}`, `N={baseline.get('N_all', '')}`.",
            "",
            "## Gate Verdict",
            "",
            f"Native Bloch beats matched Born on all-row wR2: `{native_gate['bloch_beats_born_all_wR2']}`. Native Bloch beats matched Born on free-row wR2: `{native_gate['bloch_beats_born_free_wR2']}`. Native all-row wR2 below 20%: `{native_gate['native_dyn_wR2_all_below_0p20']}`.",
            "",
            f"Corrected-HKL endpoint beats baseline R1gt/R1all/wR2all: `{endpoint_gate.get('endpoint_beats_baseline_R1gt')}/{endpoint_gate.get('endpoint_beats_baseline_R1all')}/{endpoint_gate.get('endpoint_beats_baseline_wR2all')}`. The comparison uses the same 0.70 Å cutoff, but the corrected endpoint has `{best_sh.get('N_all', '')}` SHELXL all-data reflections versus `{baseline.get('N_all', '')}` for the baseline, so it is not an identical row-set comparison.",
        ])
    if args.make_maps and final_endpoint_results:
        maps = final_endpoint_results[0].get("maps", {})
        lines.extend(["", "## Maps", ""])
        for name, rec in maps.items():
            lines.append(f"`{name}`: `{rec.get('map')}`")
    lines.extend([
        "",
        "## Next Implementation Boundary",
        "",
        "The next real structure-update layer should pass this native target into cctbx as an external least-squares target with analytic or finite-difference derivatives for coordinates, ADPs, occupancies, and restraint terms. This script deliberately keeps that out of the nuisance fit so the current endpoint remains auditable.",
    ])
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out.resolve()), "best_endpoint": summary["best_endpoint"], "report": str((out / "REPORT.md").resolve())}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
