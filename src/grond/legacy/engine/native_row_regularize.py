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

import argparse
import csv
import hashlib
import itertools
import json
import math
import os
import re
import shutil
import subprocess
import time
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as distribution_version
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
import scipy
from scipy.optimize import brentq

try:
    from grond.paths import runtime_cache_dir
except ImportError:  # Direct script execution from the legacy engine directory.

    def runtime_cache_dir(*parts: str) -> Path:
        override = os.environ.get("GROND_CACHE_DIR", "").strip()
        root = Path(override).expanduser() if override else Path.home() / ".cache" / "grond"
        return root.joinpath(*parts)

try:  # Package import.
    from .bloch_engine import (
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
    from .endpoint_modulation import modulation_manifest_fields
    from .post_bloch_sigma import fit_uniform_z_rms_sigma_scale, scale_row_sigmas
    from .shelx_diagnostics import parse_shelx_lst_pathologies, shelx_pathology_safe_for_feedback
    from .shelx_hkl import safe_fixed_width_factor, write_fixed_width_hkl
except ImportError:  # Direct script execution from this directory.
    from bloch_engine import (
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
    from shelx_diagnostics import parse_shelx_lst_pathologies, shelx_pathology_safe_for_feedback
    from shelx_hkl import safe_fixed_width_factor, write_fixed_width_hkl


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_RES = Path("model.res")
DEFAULT_BASELINE_HKL = Path("model.hkl")
SHELXL = os.environ.get("SHELXL") or shutil.which("shelxl") or shutil.which("shelxl2019") or ""
SHELX2MAP = os.environ.get("SHELX2MAP") or shutil.which("shelx2map") or ""
TARGET_MODES = ("iobs", "iobs_over_rlp", "iobs_times_peak", "recorded_proxy")
TARGET_MODE_LABELS = {
    "iobs": "IOBS",
    "iobs_over_rlp": "IOBS/RLP",
    "iobs_times_peak": "IOBS*PEAK/100",
    "recorded_proxy": "IOBS*PEAK/(100*RLP)",
}
DEFAULT_ELECTRON_WAVELENGTH_A = 0.025079
PREDICTION_CACHE_SCHEMA = 5


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
    """Fingerprint the source files that define cached row predictions."""

    digest = hashlib.sha256()
    for path in (Path(__file__).resolve(), Path(__file__).with_name("bloch_engine.py").resolve()):
        digest.update(path.name.encode())
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


@dataclass(frozen=True)
class Nuisance:
    thickness_A: float
    thickness_grad_frac: float
    frame_offset: float
    group_width_deg: float
    detector_scale_bins: int
    sigma_a: float
    sigma_b_corr: float
    sigma_c_peak: float
    thickness_model: str = "uniform_gradient"
    thickness_sigma_frac: float = 0.0
    sg_jitter_mode: str = "none"
    sg_jitter_sigma0: float = 0.0
    sg_jitter_sigma_d_inv: float = 0.0
    sg_jitter_sigma_frame: float = 0.0
    sg_jitter_points: int = 1
    sg_jitter_student_df: float = 4.0
    sg_jitter_width_factor: float = 3.0
    mosaicity_mode: str = "none"
    mosaicity_sigma_deg: float = 0.0
    mosaicity_points: int = 1
    geometry_path_file: str = ""
    absorption_mode: str = "none"
    absorption_mu: float = 0.0
    absorption_frac: float = 0.0
    absorption_mu_g2: float = 0.0
    absorption_mu_theta2: float = 0.0
    absorption_tds_b: float = 0.0
    dose_b_mode: str = "none"
    dose_b_per_frame: float = 0.0
    dose_b_ref_frame: float = float("nan")
    matrix_convention: str = "j_minus_i"
    row_resolution_b: float = 0.0
    native_wght_a: float = float("nan")
    native_wght_b: float = float("nan")
    scale_smoothness: float = 0.0


def kernel_variance_terms(nuisance: dict) -> dict[str, float]:
    """Translate fitted nuisance sigma names to the Bloch ``Config`` names."""

    return {
        "variance_a": float(nuisance["sigma_a"]),
        "variance_b_corr": float(nuisance["sigma_b_corr"]),
        "variance_c_peak": float(nuisance["sigma_c_peak"]),
    }


def parse_float_grid(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if x.strip()]


def parse_int_grid(text: str) -> list[int]:
    return [int(round(float(x))) for x in str(text).replace(",", " ").split() if x.strip()]


def nominal_frame_width_deg(data: XDSData) -> float:
    """Return the best available physical angular width of one XDS frame."""

    osc = abs(float(getattr(data, "oscillation_range", float("nan"))))
    if np.isfinite(osc) and osc > 0.0:
        return osc
    z = np.asarray(getattr(data, "zcal", []), dtype=float)
    z = z[np.isfinite(z)]
    if z.size >= 2:
        unique = np.unique(np.round(z, 6))
        if unique.size >= 2:
            dz = float(np.median(np.diff(unique)))
            if np.isfinite(dz) and dz > 0.0:
                o0 = float(data.omega_from_z(unique[0]))
                o1 = float(data.omega_from_z(unique[0] + dz))
                width = abs(o1 - o0)
                if np.isfinite(width) and width > 0.0:
                    return width
    return 1.0


def automatic_frame_offset_grid(datum_offset_frames: float) -> list[float]:
    """Return a compact profile-center grid around the XDS/GROND scan datum.

    The XDS header contains one global orientation while ``ZCAL`` may reflect
    locally refined integration geometry.  Center the grid on the nearest
    0.2-frame datum and retain symmetric residual probes instead of assuming
    that zero in the global header is also the row-profile center.
    """

    if not np.isfinite(datum_offset_frames):
        return [0.0, 0.2]
    center = round(float(datum_offset_frames) / 0.2) * 0.2
    return [round(center + residual, 10) for residual in (-0.4, 0.0, 0.4)]


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


def parse_group_width_grid(text: str, data: XDSData) -> list[float]:
    """Parse physical scale-group widths.

    In addition to numeric degree values, GROND accepts `frame`, `2frame`,
    `3frames`, and `auto`.  `auto` expands to one-, two-, and three-frame
    groups plus coarser one/two/three/five-degree checks that are not smaller
    than one physical frame.  This keeps the scale model portable across XDS
    runs with different oscillation widths without creating sub-frame scale
    factors unless the user asks for them explicitly.
    """

    frame_width = nominal_frame_width_deg(data)
    raw = [x for x in str(text).replace(",", " ").split() if x.strip()]
    if not raw:
        raw = ["auto"]
    values: list[float] = []
    for token in raw:
        t = token.strip().lower()
        if t == "auto":
            candidates = [frame_width, 2.0 * frame_width, 3.0 * frame_width, 1.0, 2.0, 3.0, 5.0]
            values.extend(v for v in candidates if v + 1e-9 >= frame_width)
            continue
        if t in {"frame", "frames"}:
            values.append(frame_width)
            continue
        m = re.fullmatch(r"([0-9]*\.?[0-9]+)(?:x)?frames?", t)
        if m:
            values.append(float(m.group(1)) * frame_width)
            continue
        if t.endswith("deg"):
            t = t[:-3]
        values.append(float(t))
    out: list[float] = []
    for value in values:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"group widths must be positive finite values; got {value!r}")
        if not any(abs(value - seen) <= max(1e-9, 1e-7 * abs(seen)) for seen in out):
            out.append(value)
    return out


def absorption_parameter_grid(args, mode: str) -> list[dict[str, float | str]]:
    valid_modes = {"none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"}
    if mode not in valid_modes:
        raise ValueError(f"unknown absorption mode {mode!r}")
    mu_values = parse_float_grid(args.absorption_mu_grid) if mode in {"mean", "beam_dependent", "midgley_like"} else [0.0]
    frac_values = parse_float_grid(args.absorption_frac_grid) if mode in {"offdiag_fraction", "offdiag_tds", "midgley_like"} else [0.0]
    mu_g2_values = parse_float_grid(args.absorption_mu_g2_grid) if mode == "beam_dependent" else [0.0]
    mu_theta2_values = parse_float_grid(args.absorption_mu_theta2_grid) if mode == "beam_dependent" else [0.0]
    tds_b_values = parse_float_grid(args.absorption_tds_b_grid) if mode == "offdiag_tds" else [0.0]
    out = []
    for mu in mu_values:
        for frac in frac_values:
            for mu_g2 in mu_g2_values:
                for mu_theta2 in mu_theta2_values:
                    for tds_b in tds_b_values:
                        out.append(
                            {
                                "absorption_mode": mode,
                                "absorption_mu": float(mu),
                                "absorption_frac": float(frac),
                                "absorption_mu_g2": float(mu_g2),
                                "absorption_mu_theta2": float(mu_theta2),
                                "absorption_tds_b": float(tds_b),
                            }
                        )
    return out


_SHELX_FLOAT_TOKEN_RE = re.compile(r"[-+]?(?:(?:\d+(?:\.\d*)?)|(?:\.\d+))(?:[Ee][-+]?\d+)?")


def parse_shelx_wght_terms(line: str) -> list[float]:
    """Return numeric WGHT terms from one SHELX line.

    SHELXL accepts a one-term WGHT record and treats the missing second term as
    zero.  A line-based parser is important here: using a whitespace regex can
    accidentally consume the ``E`` from a following EXTI line as a fake WGHT
    value.
    """

    parts = line.strip().split()
    if not parts or parts[0].upper() != "WGHT":
        return []
    terms: list[float] = []
    for token in parts[1:]:
        if not _SHELX_FLOAT_TOKEN_RE.fullmatch(token):
            break
        terms.append(float(token))
    return terms


def parse_shelx_wght_pair(line: str) -> tuple[float, float] | None:
    terms = parse_shelx_wght_terms(line)
    if not terms:
        return None
    return terms[0], terms[1] if len(terms) >= 2 else 0.0


def _iter_recommended_wght_terms(text: str):
    for line in text.splitlines():
        upper = line.upper()
        if "RECOMMENDED WEIGHTING SCHEME" not in upper or "WGHT" not in upper:
            continue
        terms = parse_shelx_wght_terms(line[upper.find("WGHT") :])
        if terms:
            yield terms


def parse_wght(res_path: Path) -> tuple[float, float]:
    for line in res_path.read_text(errors="replace").splitlines():
        pair = parse_shelx_wght_pair(line)
        if pair is not None:
            return pair
    return 0.05, 0.0


def write_hkl(path: Path, rows, float_format: str = "shelx_fixed") -> None:
    if float_format == "wide":
        fmt = "{h:4d}{k:4d}{l:4d}{I:14.4f}{sig:14.4f}\n"
    elif float_format == "shelx_fixed":
        write_fixed_width_hkl(path, rows)
        return
    else:
        raise ValueError(f"unknown hkl format {float_format!r}")
    with path.open("w") as f:
        for h, k, l, I, sig in rows:
            f.write(fmt.format(h=int(h), k=int(k), l=int(l), I=float(I), sig=float(sig)))
        f.write(fmt.format(h=0, k=0, l=0, I=0.0, sig=0.0))


def enforce_hkl_row_width(rows_merged, rows_unmerged, hkl_format: str):
    """Apply one common scale so both HKLF tables fit the requested format."""
    merged = list(rows_merged)
    unmerged = list(rows_unmerged)
    if hkl_format != "shelx_fixed":
        return merged, unmerged, 1.0
    factor = min(safe_fixed_width_factor(merged), safe_fixed_width_factor(unmerged))
    if factor < 1.0:
        merged = [
            (h, k, l, intensity * factor, sigma * factor)
            for h, k, l, intensity, sigma in merged
        ]
        unmerged = [
            (h, k, l, intensity * factor, sigma * factor)
            for h, k, l, intensity, sigma in unmerged
        ]
    return merged, unmerged, float(factor)


def shelx_wavelength_needs_electron_default(value: float) -> bool:
    """Return true when a SHELX CELL wavelength is clearly not an electron-diffraction value."""

    if not math.isfinite(float(value)) or float(value) <= 0.0:
        return True
    if abs(float(value) - 1.0) < 1e-6:
        return True
    # 200-300 keV electron wavelengths are near 0.025-0.020 A.  Values above
    # 0.1 A are almost always X-ray placeholders in small-molecule RES files.
    return float(value) >= 0.1


def normalize_shelx_cell_wavelength(line: str, default_wavelength: float = DEFAULT_ELECTRON_WAVELENGTH_A) -> str:
    """Normalize obvious placeholder CELL wavelengths for electron endpoints."""

    parts = line.split()
    if len(parts) < 8 or parts[0].upper() != "CELL":
        return line
    try:
        wavelength = float(parts[1])
    except ValueError:
        parts[1] = f"{default_wavelength:.6f}"
        return " ".join(parts)
    if shelx_wavelength_needs_electron_default(wavelength):
        parts[1] = f"{default_wavelength:.6f}"
        return " ".join(parts)
    return line


def force_shelxl_cycle_count(line: str, cycles: int = 1) -> tuple[str, bool]:
    """Force SHELXL LS/CGLS instruction lines to the requested internal cycle count."""

    match = re.match(r"^(\s*)(L\.S\.|LS|CGLS)(?=\s|$)(.*)$", line, flags=re.IGNORECASE)
    if not match:
        return line, False
    indent, command, tail = match.groups()
    count = max(1, int(cycles))
    rest = tail.split()
    if rest:
        rest[0] = str(count)
    else:
        rest = [str(count)]
    return f"{indent}{command} {' '.join(rest)}", True


def force_shelxl_one_cycle(line: str) -> tuple[str, bool]:
    """Force SHELXL LS/CGLS instruction lines to one internal cycle."""

    return force_shelxl_cycle_count(line, 1)


def _format_wght_line(a: float, b: float) -> str:
    return f"WGHT      {float(a):.4f}      {float(b):.4f}"


def set_active_wght_lines(lines: list[str], a: float, b: float) -> list[str]:
    """Return SHELX input lines with exactly one active pre-END WGHT line.

    SHELXL writes recommended WGHT records after END.  Those records are useful
    diagnostics, but they are not active input.  Endpoint files that users open
    in SHELXLE/Olex2 should therefore contain a single declared active WGHT in
    the input block and no stale tail after END.
    """

    out: list[str] = []
    inserted = False
    wght_line = _format_wght_line(a, b)
    for line in lines:
        upper = line.strip().upper()
        if upper.startswith("WGHT"):
            if not inserted:
                out.append(wght_line)
                inserted = True
            continue
        if not inserted and upper.startswith("FVAR"):
            out.append(wght_line)
            inserted = True
        out.append(line)
    if not inserted:
        insert_at = len(out)
        for i, line in enumerate(out):
            if line.strip().upper().startswith(("HKLF", "END")):
                insert_at = i
                break
        out.insert(insert_at, wght_line)
    return out


def _hklf_order(line: str) -> str:
    parts = line.split()
    return parts[1] if len(parts) > 1 else ""


def _unsupported_endpoint_reason(line: str) -> str:
    upper = line.strip().upper()
    head = upper.split()[0] if upper.split() else ""
    if head in {"BASF", "TWIN", "SWAT"}:
        return f"{head} is not supported by GROND corrected-HKLF4 endpoints yet"
    if head == "SQUEEZE" or "SQUEEZE" in upper:
        return "SQUEEZE/solvent-mask corrected SHELX models are not supported by GROND corrected-HKLF4 endpoints yet"
    if head == "HKLF":
        order = _hklf_order(line)
        if order not in {"3", "4"}:
            return f"HKLF {order or '?'} is not supported; GROND currently writes corrected intensity endpoints as HKLF 4"
    return ""


def write_endpoint_ins(
    src_res: Path,
    dst_ins: Path,
    hkl_name: str = "dyn_nuisance_corrected.hkl",
    keep_exti: bool = False,
    wght: tuple[float, float] | None = None,
    ls_cycles: int = 1,
) -> None:
    """Write the corrected-HKL endpoint input with GROND's endpoint contract."""

    out = []
    saw_cycle_instruction = False
    saw_hklf = False
    cycle_count = max(1, int(ls_cycles))
    for line in src_res.read_text(errors="replace").splitlines():
        stripped = line.strip()
        upper = stripped.upper()
        unsupported = _unsupported_endpoint_reason(line)
        if unsupported:
            raise ValueError(unsupported)
        if upper.startswith("ACTA"):
            continue
        if not keep_exti and upper.startswith("EXTI"):
            continue
        if upper.startswith("CELL "):
            line = normalize_shelx_cell_wavelength(line)
        line, forced_cycle = force_shelxl_cycle_count(line, cycle_count)
        saw_cycle_instruction = saw_cycle_instruction or forced_cycle
        if "<HklSrc" in line:
            line = f'REM <HklSrc "%./{hkl_name}">'
        if upper.startswith("HKLF"):
            line = "HKLF 4"
            saw_hklf = True
        out.append(line)
        if upper.startswith("END"):
            break
    if not saw_cycle_instruction:
        insert_at = len(out)
        for i, line in enumerate(out):
            if line.strip().upper().startswith(("HKLF", "END")):
                insert_at = i
                break
        out.insert(insert_at, f"L.S. {cycle_count}")
    if not saw_hklf:
        insert_at = len(out)
        for i, line in enumerate(out):
            if line.strip().upper().startswith("END"):
                insert_at = i
                break
        out.insert(insert_at, "HKLF 4")
    if wght is not None:
        out = set_active_wght_lines(out, float(wght[0]), float(wght[1]))
    dst_ins.write_text("\n".join(out) + "\n")


def write_no_exti_ins(src_res: Path, dst_ins: Path, hkl_name: str = "dyn_nuisance_corrected.hkl") -> None:
    write_endpoint_ins(src_res, dst_ins, hkl_name=hkl_name, keep_exti=False)


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


def target_arrays(data: XDSData, mode: str):
    """Return the reduced XDS row target and matching sigma convention."""

    if mode == "iobs":
        factor = np.ones_like(data.iobs, dtype=float)
    elif mode == "iobs_over_rlp":
        factor = 1.0 / np.maximum(data.rlp, 1e-12)
    elif mode == "iobs_times_peak":
        factor = np.maximum(data.peak, 0.0) / 100.0
    elif mode == "recorded_proxy":
        factor = np.maximum(data.peak, 0.0) / 100.0 / np.maximum(data.rlp, 1e-12)
    else:
        raise ValueError(f"unknown target mode {mode!r}")
    return data.iobs * factor, data.sigma * np.abs(factor)


def sigma_eff(data: XDSData, target, base_sigma, nui: Nuisance):
    corr_term = np.maximum(0.0, 1.0 - data.corr / 100.0)
    peak_term = np.maximum(0.0, 1.0 - np.maximum(data.peak, 0.0) / 100.0)
    return np.sqrt(
        np.maximum(base_sigma, 1e-12) ** 2
        + (nui.sigma_a * target) ** 2
        + (nui.sigma_b_corr * target * corr_term) ** 2
        + (nui.sigma_c_peak * target * peak_term) ** 2
    )


def group_indices(data: XDSData, group_width_deg: float, frame_offset: float):
    omegas = np.asarray([data.omega_from_z(float(z) + float(frame_offset)) for z in data.zcal], dtype=float)
    return ((omegas - float(np.min(omegas))) // float(group_width_deg)).astype(int)


def detector_bin_indices(data: XDSData, n_bins: int):
    if int(n_bins) <= 1:
        return np.zeros(len(data.hkl), dtype=int)
    nx_text = str(data.header.get("NX", ""))
    ny_text = str(data.header.get("NY", ""))
    nx_match = re.search(r"^\s*([0-9.]+)", nx_text)
    ny_match = re.search(r"(?:^|\s)NY\s*=\s*([0-9.]+)", nx_text) or re.search(r"^\s*([0-9.]+)", ny_text)
    x = np.asarray(data.xcal, dtype=float)
    y = np.asarray(data.ycal, dtype=float)
    nx = float(nx_match.group(1)) if nx_match else float("nan")
    ny = float(ny_match.group(1)) if ny_match else float("nan")
    if not np.isfinite(nx) or nx <= 0:
        nx = float(np.nanmax(x)) + 1.0
    if not np.isfinite(ny) or ny <= 0:
        ny = float(np.nanmax(y)) + 1.0
    xb = np.floor(np.clip(x, 0.0, nx - 1e-6) / nx * int(n_bins)).astype(int)
    yb = np.floor(np.clip(y, 0.0, ny - 1e-6) / ny * int(n_bins)).astype(int)
    return yb * int(n_bins) + xb


def shelx_weights(obs, sig, pred, a: float | np.ndarray, b: float | np.ndarray):
    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return 1.0 / np.maximum(sig * sig + (a * p) ** 2 + b * p, 1e-30)


def solve_regularized_scales(diag_sums, x_y_sums, unique_groups, lambda_smooth):
    n = len(unique_groups)
    if n == 0:
        return {}
    if n == 1:
        return {int(unique_groups[0]): max(float(x_y_sums[0] / max(diag_sums[0], 1e-30)), 1e-30)}
    d = np.array(diag_sums, dtype=float)
    d[0] += float(lambda_smooth)
    d[1:-1] += 2.0 * float(lambda_smooth)
    d[-1] += float(lambda_smooth)
    c = np.full(n - 1, -float(lambda_smooth), dtype=float)
    b = np.array(x_y_sums, dtype=float)
    c_prime = np.zeros(n - 1, dtype=float)
    d_prime = np.zeros(n, dtype=float)
    c_prime[0] = c[0] / max(d[0], 1e-30)
    d_prime[0] = b[0] / max(d[0], 1e-30)
    for i in range(1, n - 1):
        denom = d[i] - c[i-1] * c_prime[i-1]
        c_prime[i] = c[i] / max(denom, 1e-30)
        d_prime[i] = (b[i] - c[i-1] * d_prime[i-1]) / max(denom, 1e-30)
    denom = d[n-1] - c[n-2] * c_prime[n-2]
    d_prime[n-1] = (b[n-1] - c[n-2] * d_prime[n-2]) / max(denom, 1e-30)
    x = np.zeros(n, dtype=float)
    x[-1] = d_prime[n-1]
    for i in range(n - 2, -1, -1):
        x[i] = d_prime[i] - c_prime[i] * x[i+1]
    return {int(g): max(float(val), 1e-30) for g, val in zip(unique_groups, x, strict=True)}


def _scale_fit_mask(obs, sig, pred, fit_mask):
    return (
        np.asarray(fit_mask, dtype=bool)
        & np.isfinite(obs)
        & np.isfinite(sig)
        & (np.asarray(sig, dtype=float) > 0.0)
        & np.isfinite(pred)
        & (np.asarray(pred, dtype=float) > 0.0)
    )


def _fit_global_scale(obs, sig, pred, fit_mask, wght_a: float, wght_b: float, n_iter: int):
    fit = _scale_fit_mask(obs, sig, pred, fit_mask)
    if np.count_nonzero(fit) < 2:
        raise ValueError("at least two work rows with positive sigma and prediction are required to fit scales")
    x = np.asarray(pred[fit], dtype=float)
    y = np.asarray(obs[fit], dtype=float)
    sfit = np.asarray(sig[fit], dtype=float)
    w = 1.0 / (sfit * sfit)
    scale = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
    for _ in range(int(n_iter)):
        w = shelx_weights(y, sfit, scale * x, wght_a, wght_b)
        new_scale = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
        if abs(new_scale - scale) <= 1e-8 * max(1.0, abs(scale)):
            return new_scale
        scale = new_scale
    return scale


def fit_group_scales(obs, sig, pred, groups, fit_mask, wght_a: float, wght_b: float, n_iter: int = 6, scale_smoothness: float = 0.0):
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (len(obs) == len(sig) == len(pred) == len(groups) == len(fit_mask)):
        raise ValueError("observation, sigma, prediction, group, and fit-mask lengths differ")
    scaled = np.full_like(pred, np.nan, dtype=float)
    scales: dict[int, float] = {}
    unique_groups = np.unique(groups)
    if len(unique_groups) == 0:
        return scaled, scales
    global_scale = _fit_global_scale(obs, sig, pred, fit_mask, wght_a, wght_b, n_iter)

    if scale_smoothness > 0.0:
        diag_sums = np.zeros(len(unique_groups), dtype=float)
        x_y_sums = np.zeros(len(unique_groups), dtype=float)
        for j, g in enumerate(unique_groups):
            fit = (groups == g) & _scale_fit_mask(obs, sig, pred, fit_mask)
            if np.count_nonzero(fit) >= 2:
                x = pred[fit]
                y = obs[fit]
                sfit = sig[fit]
                w = 1.0 / np.maximum(sfit * sfit, 1e-30)
                diag_sums[j] = np.sum(w * x * x)
                x_y_sums[j] = np.sum(w * x * y)
        solved = solve_regularized_scales(diag_sums, x_y_sums, unique_groups, scale_smoothness)
        for g in unique_groups:
            value = float(solved[int(g)])
            scales[int(g)] = value if np.isfinite(value) and value > 1e-20 else global_scale

        for _ in range(int(n_iter)):
            diag_sums = np.zeros(len(unique_groups), dtype=float)
            x_y_sums = np.zeros(len(unique_groups), dtype=float)
            for j, g in enumerate(unique_groups):
                fit = (groups == g) & _scale_fit_mask(obs, sig, pred, fit_mask)
                if np.count_nonzero(fit) >= 2:
                    x = pred[fit]
                    y = obs[fit]
                    sfit = sig[fit]
                    pcalc = scales[int(g)] * x
                    w = shelx_weights(y, sfit, pcalc, wght_a, wght_b)
                    diag_sums[j] = np.sum(w * x * x)
                    x_y_sums[j] = np.sum(w * x * y)
            solved = solve_regularized_scales(diag_sums, x_y_sums, unique_groups, scale_smoothness)
            max_diff = 0.0
            for g in unique_groups:
                max_diff = max(max_diff, abs(solved[int(g)] - scales[int(g)]) / max(1.0, abs(scales[int(g)])))
            for g in unique_groups:
                value = float(solved[int(g)])
                scales[int(g)] = value if np.isfinite(value) and value > 1e-20 else global_scale
            if max_diff <= 1e-8:
                break

        for g in unique_groups:
            apply = (groups == g) & np.isfinite(pred)
            scaled[apply] = pred[apply] * scales[int(g)]
    else:
        for g in unique_groups:
            fit = (groups == g) & _scale_fit_mask(obs, sig, pred, fit_mask)
            apply = (groups == g) & np.isfinite(pred)
            if np.count_nonzero(fit) < 2:
                scales[int(g)] = global_scale
                scaled[apply] = pred[apply] * global_scale
                continue
            x = pred[fit]
            y = obs[fit]
            sfit = sig[fit]
            w = 1.0 / np.maximum(sfit * sfit, 1e-30)
            scale = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
            scale = scale if np.isfinite(scale) and scale > 1e-20 else global_scale
            for _ in range(int(n_iter)):
                pcalc = scale * x
                w = shelx_weights(y, sfit, pcalc, wght_a, wght_b)
                new_scale = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
                new_scale = new_scale if np.isfinite(new_scale) and new_scale > 1e-20 else global_scale
                if abs(new_scale - scale) <= 1e-8 * max(1.0, abs(scale)):
                    scale = new_scale
                    break
                scale = new_scale
            scales[int(g)] = scale
            scaled[apply] = pred[apply] * scale
    return scaled, scales


def fit_frame_detector_scales(obs, sig, pred, groups, det_bins, fit_mask, wght_a: float, wght_b: float, n_iter: int = 12, scale_smoothness: float = 0.0):
    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups)
    det_bins = np.asarray(det_bins)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (len(obs) == len(sig) == len(pred) == len(groups) == len(det_bins) == len(fit_mask)):
        raise ValueError("observation, sigma, prediction, scale-group, detector-bin, and fit-mask lengths differ")
    scaled = np.full_like(pred, np.nan, dtype=float)
    frame_scales: dict[int, float] = {}
    det_scales: dict[int, float] = {}
    good = _scale_fit_mask(obs, sig, pred, fit_mask)
    global_scale = _fit_global_scale(obs, sig, pred, fit_mask, wght_a, wght_b, n_iter)
    unique_groups = np.asarray(sorted(np.unique(groups[np.isfinite(groups)])), dtype=int)
    unique_det = np.asarray(sorted(np.unique(det_bins[np.isfinite(det_bins)])), dtype=int)
    if len(unique_groups) == 0 or len(unique_det) == 0:
        return scaled, frame_scales, det_scales
    gmap = {int(g): i for i, g in enumerate(unique_groups)}
    dmap = {int(d): i for i, d in enumerate(unique_det)}
    gi = np.asarray([gmap.get(int(g), -1) for g in groups], dtype=int)
    di = np.asarray([dmap.get(int(d), -1) for d in det_bins], dtype=int)
    frame = np.ones(len(unique_groups), dtype=float)
    det = np.ones(len(unique_det), dtype=float)
    frame[:] = global_scale
    det[:] = 1.0
    for _ in range(int(n_iter)):
        base = pred * det[np.maximum(di, 0)]
        if scale_smoothness > 0.0:
            diag_sums = np.zeros(len(unique_groups), dtype=float)
            x_y_sums = np.zeros(len(unique_groups), dtype=float)
            for j, _g in enumerate(unique_groups):
                m = good & (gi == j) & np.isfinite(base) & (base > 0)
                if np.count_nonzero(m) >= 2:
                    x = base[m]
                    y = obs[m]
                    sfit = sig[m]
                    w = shelx_weights(y, sfit, x * frame[j], wght_a, wght_b)
                    diag_sums[j] = np.sum(w * x * x)
                    x_y_sums[j] = np.sum(w * x * y)
            solved_scales = solve_regularized_scales(diag_sums, x_y_sums, unique_groups, scale_smoothness)
            for j, g in enumerate(unique_groups):
                value = float(solved_scales[int(g)])
                frame[j] = value if np.isfinite(value) and value > 1e-20 else global_scale
        else:
            for j, _g in enumerate(unique_groups):
                m = good & (gi == j) & np.isfinite(base) & (base > 0)
                if np.count_nonzero(m) < 2:
                    continue
                x = base[m]
                y = obs[m]
                sfit = sig[m]
                w = shelx_weights(y, sfit, x * frame[j], wght_a, wght_b)
                frame[j] = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
        base = pred * frame[np.maximum(gi, 0)]
        for j, _d in enumerate(unique_det):
            m = good & (di == j) & np.isfinite(base) & (base > 0)
            if np.count_nonzero(m) < 10:
                continue
            x = base[m]
            y = obs[m]
            sfit = sig[m]
            w = shelx_weights(y, sfit, x * det[j], wght_a, wght_b)
            det[j] = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
        median_det = float(np.median(det[np.isfinite(det) & (det > 0)]))
        if median_det > 0:
            det /= median_det
            frame *= median_det
    apply = np.isfinite(pred) & (gi >= 0) & (di >= 0)
    scaled[apply] = pred[apply] * frame[gi[apply]] * det[di[apply]]
    frame_scales = {str(int(g)): float(frame[j]) for j, g in enumerate(unique_groups)}
    det_scales = {str(int(d)): float(det[j]) for j, d in enumerate(unique_det)}
    return scaled, frame_scales, det_scales


def native_metrics(obs, sig, pred, mask, wght_a: float, wght_b: float, n_params: int = 0):
    m = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(obs)
        & np.isfinite(sig)
        & (np.asarray(sig, dtype=float) > 0.0)
        & np.isfinite(pred)
    )
    obs_m = obs[m]
    sig_m = sig[m]
    pred_m = pred[m]
    if len(obs_m) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"), "R2_all": float("nan"), "wR2_all": float("nan"), "GooF": float("nan"), "mean_weighted_abs_residual": float("nan"), "CC": float("nan")}
    gt = obs_m > 2.0 * sig_m
    fobs = np.sqrt(np.maximum(obs_m, 0.0))
    fpred = np.sqrt(np.maximum(pred_m, 0.0))
    eps = 1e-30

    def r1(local):
        if not np.any(local):
            return float("nan")
        return float(np.sum(np.abs(fobs[local] - fpred[local])) / max(np.sum(fobs[local]), eps))

    w = shelx_weights(obs_m, sig_m, pred_m, wght_a, wght_b)
    res = obs_m - pred_m
    wr2 = float(np.sqrt(np.sum(w * res * res) / max(np.sum(w * obs_m * obs_m), eps)))
    dof = max(1, len(obs_m) - int(n_params))
    goof = float(np.sqrt(np.sum(w * res * res) / dof))
    r2 = float(np.sum(np.abs(res)) / max(np.sum(np.abs(obs_m)), eps))
    cc = float(np.corrcoef(obs_m, pred_m)[0, 1]) if len(obs_m) > 2 and np.std(obs_m) > 0 and np.std(pred_m) > 0 else float("nan")
    mw_abs = float(np.mean(np.sqrt(w) * np.abs(res)))
    return {
        "n": int(len(obs_m)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "R2_all": r2,
        "wR2_all": wr2,
        "GooF": goof,
        "mean_weighted_abs_residual": mw_abs,
        "CC": cc,
        "dof": int(dof),
    }


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
        "res_sha256": sha256(res),
        "model_tag": str(model_tag),
        "hkl_transform": str(getattr(data, "hkl_transform_spec", "identity")),
        "indices_sha256": hashlib.sha256(np.asarray(indices, dtype=np.int64).tobytes()).hexdigest(),
        "cfg": asdict(cfg),
    }
    text = canonical_prediction_cache_payload(payload)
    return hashlib.sha256(text.encode()).hexdigest()[:20], payload


def canonical_prediction_cache_payload(payload: dict) -> str:
    """Return a deterministic representation that preserves NaN sentinels.

    ``Config.dose_b_ref_frame`` uses NaN to mean that no reference frame was
    supplied. JSON round-trips that sentinel correctly for our local NPZ
    metadata, but Python dictionary equality does not because NaN is unequal
    to itself. Comparing this canonical representation makes a valid cache
    round-trip stable without weakening any content checks.
    """

    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=True)


def write_prediction_cache(path: Path, save: dict, payload: dict, compression: str) -> None:
    tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
    if compression == "none":
        np.savez(tmp, **save, meta_json=json.dumps(payload, sort_keys=True))
    else:
        np.savez_compressed(tmp, **save, meta_json=json.dumps(payload, sort_keys=True))
    os.replace(tmp, path)


def cacheable_prediction_arrays(pred: dict) -> dict:
    """Return the numeric prediction arrays that are safe to store in NPZ.

    Prediction dictionaries can also contain live diagnostics such as row-cache
    status dictionaries.  NumPy stores those as object arrays, which cannot be
    reloaded with allow_pickle=False and should not be part of the portable
    prediction cache.
    """

    skip = {"support_hkls", "cache_status", "cache_path", "row_plan_cache_status"}
    save = {}
    for key, value in pred.items():
        if key in skip:
            continue
        arr = np.asarray(value)
        if arr.dtype.hasobject:
            continue
        if arr.dtype.kind not in {"b", "i", "u", "f", "c"}:
            continue
        save[key] = arr
    return save


def read_prediction_cache(path: Path) -> dict:
    """Read a prediction cache, ignoring stale object-array diagnostics."""

    pred = {}
    skipped = []
    with np.load(path, allow_pickle=False) as loaded:
        if "meta_json" not in loaded.files:
            raise ValueError(f"prediction cache {path} has no metadata")
        pred["prediction_cache_meta"] = json.loads(str(loaded["meta_json"]))
        for key in loaded.files:
            if key == "meta_json":
                continue
            try:
                value = loaded[key]
            except ValueError as exc:
                if "Object arrays cannot be loaded" in str(exc):
                    skipped.append(key)
                    continue
                raise
            if key == "seconds" and np.ndim(value) == 0:
                pred[key] = float(value)
            elif key == "row_plan_cache_unique_diffs" and np.ndim(value) == 0:
                pred[key] = int(value)
            else:
                pred[key] = value
    if skipped:
        pred["prediction_cache_skipped_keys"] = skipped
    return pred


def prediction_cache_validation_errors(pred: dict, payload: dict, indices) -> list[str]:
    """Return reasons a prediction cache cannot satisfy the requested calculation."""

    errors = []
    meta = pred.get("prediction_cache_meta")
    if not isinstance(meta, dict) or canonical_prediction_cache_payload(meta) != canonical_prediction_cache_payload(payload):
        errors.append("metadata does not match the requested prediction contract")
    expected_indices = np.asarray(indices, dtype=np.int64)
    cached_indices = np.asarray(pred.get("indices", []), dtype=np.int64)
    if not np.array_equal(cached_indices, expected_indices):
        errors.append("cached row indices differ")
    for key in ("dyn", "born"):
        values = np.asarray(pred.get(key, []))
        if values.shape != (len(expected_indices),):
            errors.append(f"{key} has shape {values.shape}, expected {(len(expected_indices),)}")
        elif not np.all(np.isfinite(values)) or np.any(values < 0.0):
            errors.append(f"{key} contains non-finite or negative predictions")
    if "basis_size" in pred:
        basis_size = np.asarray(pred["basis_size"])
        if basis_size.shape != (len(expected_indices),) or np.any(basis_size < 2):
            errors.append("basis_size is incomplete or invalid")
    if int(payload.get("cache_schema", -1)) >= 5:
        continuation_used = np.asarray(
            pred.get("sg_profile_same_branch_continuation_used", [])
        )
        continuation_segments = np.asarray(
            pred.get("sg_profile_same_branch_continuation_segments", [])
        )
        if continuation_used.shape != (len(expected_indices),):
            errors.append("profile continuation usage provenance is incomplete")
        elif (
            not np.all(np.isfinite(continuation_used))
            or np.any(~np.isin(continuation_used, (False, True)))
        ):
            errors.append("profile continuation usage provenance is invalid")
        if continuation_segments.shape != (len(expected_indices),):
            errors.append("profile continuation segment provenance is incomplete")
        elif (
            not np.all(np.isfinite(continuation_segments))
            or np.any(continuation_segments < 1)
            or np.any(continuation_segments != np.floor(continuation_segments))
        ):
            errors.append("profile continuation segment provenance is invalid")
        if (
            continuation_used.shape == (len(expected_indices),)
            and continuation_segments.shape == (len(expected_indices),)
        ):
            continuation_used = continuation_used.astype(bool)
            continuation_segments = continuation_segments.astype(np.int64)
            if np.any(continuation_used & (continuation_segments < 2)):
                errors.append("continued profile rows must record at least two local segments")
            if np.any(~continuation_used & (continuation_segments != 1)):
                errors.append("direct profile rows must record exactly one local segment")
    return errors


def predict_with_cache(data, model, res_hash, hkl_path, res_path, cfg, indices, cache_dir: Path, workers: int, chunksize: int, model_tag: str = "original", cache_compression: str = "compressed"):
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
    ug_cache_dir = runtime_cache_dir("native_row_regularize", "ug")
    port = XDSBlochPort(data, model, cfg, cache_dir=ug_cache_dir, res_hash=res_hash)
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


def resolution_shell_bins(data: XDSData, valid, shell_count: int) -> np.ndarray:
    """Assign rows to approximately equal-population resolution bins."""

    bins = np.full(len(data.hkl), -1, dtype=int)
    valid = np.asarray(valid, dtype=bool)
    n_shells = max(1, int(shell_count))
    if not np.any(valid):
        return bins
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    inv_d = np.full(len(dvals), np.nan, dtype=float)
    finite = valid & np.isfinite(dvals) & (dvals > 0.0)
    inv_d[finite] = 1.0 / dvals[finite]
    values = inv_d[finite]
    if values.size == 0:
        return bins
    if values.size < n_shells:
        n_shells = max(1, values.size)
    quantiles = np.linspace(0.0, 1.0, n_shells + 1)
    edges = np.quantile(values, quantiles)
    edges = np.maximum.accumulate(edges)
    for i, val in zip(np.where(finite)[0], inv_d[finite], strict=True):
        shell = int(np.searchsorted(edges, val, side="right") - 1)
        bins[i] = min(max(shell, 0), n_shells - 1)
    return bins


def apply_q_control(data: XDSData, dyn: np.ndarray, born: np.ndarray, valid, groups: np.ndarray, q_control_args=None) -> tuple[np.ndarray, dict]:
    """Apply an explicit row-space q null control before ordinary scale fitting."""

    mode = str(getattr(q_control_args, "q_control", "real") if q_control_args is not None else "real")
    meta = {
        "mode": mode,
        "description": "real row-wise Bloch/Born q" if mode == "real" else "row-space permutation of log(Idyn/Iborn)",
    }
    if mode == "real":
        return dyn, meta
    valid = np.asarray(valid, dtype=bool) & np.isfinite(dyn) & np.isfinite(born) & (dyn > 0.0) & (born > 0.0)
    out = np.array(dyn, dtype=float, copy=True)
    q = np.full(len(dyn), np.nan, dtype=float)
    q[valid] = dyn[valid] / np.maximum(born[valid], 1e-30)
    logq = np.full(len(dyn), np.nan, dtype=float)
    logq[valid] = np.log(np.clip(q[valid], 1e-30, np.inf))
    seed = int(getattr(q_control_args, "q_control_seed", 12345) if q_control_args is not None else 12345)
    rng = np.random.default_rng(seed)
    shell_count = int(getattr(q_control_args, "q_control_shells", 10) if q_control_args is not None else 10)
    shell_bins = resolution_shell_bins(data, valid, shell_count)
    if mode == "shuffle_global":
        keys = np.zeros(len(dyn), dtype=int)
    elif mode == "shuffle_shell":
        keys = shell_bins
    elif mode == "shuffle_shell_scan":
        scan_groups = np.asarray(groups, dtype=int)
        keys = shell_bins.astype(np.int64) * 1000003 + scan_groups.astype(np.int64)
    else:
        raise ValueError(f"unknown q_control {mode!r}")
    shuffled_logq = np.array(logq, dtype=float, copy=True)
    shuffled_rows = 0
    singleton_bins = 0
    active_bins = 0
    for key in np.unique(keys[valid]):
        idx = np.where(valid & (keys == key))[0]
        if idx.size < 2:
            singleton_bins += int(idx.size == 1)
            continue
        active_bins += 1
        shuffled_logq[idx] = rng.permutation(shuffled_logq[idx])
        shuffled_rows += int(idx.size)
    out[valid] = born[valid] * np.exp(shuffled_logq[valid])
    meta.update(
        {
            "seed": seed,
            "shell_count": shell_count,
            "valid_rows": int(np.count_nonzero(valid)),
            "shuffled_rows": int(shuffled_rows),
            "active_bins": int(active_bins),
            "singleton_bins": int(singleton_bins),
            "logq_corr_original_shuffled": float(np.corrcoef(logq[valid], shuffled_logq[valid])[0, 1]) if np.count_nonzero(valid) > 2 else float("nan"),
        }
    )
    return out, meta


def evaluate_trial(data, target, base_sig, pred, indices, masks, nui: Nuisance, wght_a, wght_b, scale_fit_split: str = "work", q_control_args=None):
    dyn = np.full(len(target), np.nan, dtype=float)
    dyn_inverse = np.full(len(target), np.nan, dtype=float)
    born = np.full(len(target), np.nan, dtype=float)
    basis_size = np.zeros(len(target), dtype=int)
    dyn[pred["indices"]] = pred["dyn"]
    if "dyn_inverse" in pred:
        dyn_inverse[pred["indices"]] = pred["dyn_inverse"]
    born[pred["indices"]] = pred["born"]
    dyn_unmodulated = dyn.copy()
    dyn_inverse_unmodulated = dyn_inverse.copy()
    if "basis_size" in pred:
        basis_size[pred["indices"]] = np.asarray(pred["basis_size"], dtype=int)
    pred_indices = np.asarray(pred["indices"], dtype=int)

    def expand_pred_array(name: str, *, dtype=float, default=np.nan):
        values = np.full(len(target), default, dtype=dtype)
        if name in pred:
            values[pred_indices] = np.asarray(pred[name], dtype=dtype)
        return values

    sg_arrays = {
        "center_sg": expand_pred_array("center_sg"),
        "omega_bragg": expand_pred_array("omega_bragg"),
        "omega_delta": expand_pred_array("omega_delta"),
        "sg_window": expand_pred_array("sg_window"),
        "sg_window_raw": expand_pred_array("sg_window_raw"),
        "sg_window_half_frames": expand_pred_array("sg_window_half_frames"),
        "sg_window_clipped": expand_pred_array("sg_window_clipped", dtype=bool, default=False),
        "sg_dSg_domega": expand_pred_array("sg_dSg_domega"),
        "sg_dSg_dframe": expand_pred_array("sg_dSg_dframe"),
        "sg_jitter_sigma": expand_pred_array("sg_jitter_sigma"),
        "sg_jitter_d_inv": expand_pred_array("sg_jitter_d_inv"),
        "sg_jitter_frame_fraction": expand_pred_array("sg_jitter_frame_fraction"),
        "sg_jitter_kernel_points": expand_pred_array("sg_jitter_kernel_points", dtype=int, default=1),
        "sg_profile_points_total": expand_pred_array("sg_profile_points_total", dtype=int, default=0),
        "sg_profile_weight_sum": expand_pred_array("sg_profile_weight_sum"),
        "sg_profile_same_branch_continuation_used": expand_pred_array(
            "sg_profile_same_branch_continuation_used",
            dtype=bool,
            default=False,
        ),
        "sg_profile_same_branch_continuation_segments": expand_pred_array(
            "sg_profile_same_branch_continuation_segments",
            dtype=int,
            default=1,
        ),
        "sg_residual": expand_pred_array("sg_residual"),
    }
    groups = group_indices(data, nui.group_width_deg, nui.frame_offset)
    det_bins = detector_bin_indices(data, nui.detector_scale_bins)
    sig = sigma_eff(data, target, base_sig, nui)
    use_wght_a = float(nui.native_wght_a) if np.isfinite(float(nui.native_wght_a)) else float(wght_a)
    use_wght_b = float(nui.native_wght_b) if np.isfinite(float(nui.native_wght_b)) else float(wght_b)
    q_control_meta = {}
    valid = (
        np.isfinite(target)
        & np.isfinite(sig)
        & (sig > 0.0)
        & np.isfinite(dyn)
        & np.isfinite(born)
        & (dyn > 0.0)
        & (born > 0.0)
    )
    dyn, q_control_meta = apply_q_control(data, dyn, born, valid, groups, q_control_args)
    row_mod, row_mod_meta = row_resolution_modulation(data, valid, nui.row_resolution_b)
    dyn = dyn * row_mod
    dyn_inverse = dyn_inverse * row_mod
    valid = (
        np.isfinite(target)
        & np.isfinite(sig)
        & (sig > 0.0)
        & np.isfinite(dyn)
        & np.isfinite(born)
        & (dyn > 0.0)
        & (born > 0.0)
    )
    if scale_fit_split not in masks:
        raise ValueError(f"unknown scale_fit_split={scale_fit_split!r}; choices are {sorted(masks)}")
    scale_fit = masks[scale_fit_split] & valid
    n_scales = len([g for g in np.unique(groups) if np.count_nonzero((groups == g) & scale_fit) >= 2])
    if int(nui.detector_scale_bins) > 1:
        n_scales += int(nui.detector_scale_bins) * int(nui.detector_scale_bins)
    n_abs = int(str(nui.absorption_mode) != "none" and (
        abs(float(nui.absorption_mu)) > 0.0
        or abs(float(nui.absorption_frac)) > 0.0
        or abs(float(nui.absorption_mu_g2)) > 0.0
        or abs(float(nui.absorption_mu_theta2)) > 0.0
    ))
    n_thick_shape = int(str(nui.thickness_model) != "uniform_gradient" or abs(float(nui.thickness_sigma_frac)) > 0.0)
    n_dose = int(str(nui.dose_b_mode) != "none" and abs(float(nui.dose_b_per_frame)) > 0.0)
    n_jitter = int(str(nui.sg_jitter_mode) != "none" and (
        abs(float(nui.sg_jitter_sigma0)) > 0.0
        or abs(float(nui.sg_jitter_sigma_d_inv)) > 0.0
        or abs(float(nui.sg_jitter_sigma_frame)) > 0.0
    ))
    n_mosaicity = int(
        str(nui.mosaicity_mode) != "none"
        and abs(float(nui.mosaicity_sigma_deg)) > 0.0
    )
    n_resolution = int(abs(float(nui.row_resolution_b)) > 0.0)
    n_weight = int(np.isfinite(float(nui.native_wght_a)) or np.isfinite(float(nui.native_wght_b)))
    n_params = (
        n_scales
        + 7
        + n_abs
        + n_thick_shape
        + n_dose
        + n_jitter
        + n_mosaicity
        + n_resolution
        + n_weight
    )
    if int(nui.detector_scale_bins) > 1:
        dyn_scaled, dyn_scales, dyn_detector_scales = fit_frame_detector_scales(target, sig, dyn, groups, det_bins, scale_fit, use_wght_a, use_wght_b, scale_smoothness=nui.scale_smoothness)
        born_scaled, born_scales, born_detector_scales = fit_frame_detector_scales(target, sig, born, groups, det_bins, scale_fit, use_wght_a, use_wght_b, scale_smoothness=nui.scale_smoothness)
    else:
        dyn_scaled, dyn_scales = fit_group_scales(target, sig, dyn, groups, scale_fit, use_wght_a, use_wght_b, scale_smoothness=nui.scale_smoothness)
        born_scaled, born_scales = fit_group_scales(target, sig, born, groups, scale_fit, use_wght_a, use_wght_b, scale_smoothness=nui.scale_smoothness)
        dyn_detector_scales = {}
        born_detector_scales = {}
    metrics = {}
    for split, mask in masks.items():
        metrics[split] = {
            "dyn": native_metrics(target, sig, dyn_scaled, mask & valid, use_wght_a, use_wght_b, n_params=n_params),
            "born": native_metrics(target, sig, born_scaled, mask & valid, use_wght_a, use_wght_b, n_params=n_params),
        }
    return {
        "nuisance": asdict(nui),
        "native_weight": {"a": use_wght_a, "b": use_wght_b, "source_a": float(wght_a), "source_b": float(wght_b)},
        "metrics": metrics,
        "valid_rows": int(np.count_nonzero(valid)),
        "scale_count": int(n_scales),
        "scale_fit_split": scale_fit_split,
        "row_resolution": row_mod_meta,
        "q_control": q_control_meta,
        "basis_size_stats": {
            "min": int(np.min(basis_size[valid])) if np.count_nonzero(valid) else 0,
            "median": float(np.median(basis_size[valid])) if np.count_nonzero(valid) else 0.0,
            "p95": float(np.percentile(basis_size[valid], 95)) if np.count_nonzero(valid) else 0.0,
            "p99": float(np.percentile(basis_size[valid], 99)) if np.count_nonzero(valid) else 0.0,
            "max": int(np.max(basis_size[valid])) if np.count_nonzero(valid) else 0,
        },
        "dyn_scales": {str(k): float(v) for k, v in dyn_scales.items()},
        "born_scales": {str(k): float(v) for k, v in born_scales.items()},
        "dyn_detector_scales": {str(k): float(v) for k, v in dyn_detector_scales.items()},
        "born_detector_scales": {str(k): float(v) for k, v in born_detector_scales.items()},
        "arrays": {
            "dyn_raw": dyn,
            "dyn_inverse_raw": dyn_inverse,
            "dyn_raw_unmodulated": dyn_unmodulated,
            "dyn_inverse_raw_unmodulated": dyn_inverse_unmodulated,
            "born_raw": born,
            "row_resolution_factor": row_mod,
            "dyn_scaled": dyn_scaled,
            "born_scaled": born_scaled,
            "sigma_eff": sig,
            "groups": groups,
            "detector_bins": det_bins,
            "basis_size": basis_size,
            **sg_arrays,
            "valid": valid,
        },
    }


def geometry_tail_mask(data: XDSData, base_mask, percentile: float = 100.0):
    """Keep the broad geometry body and exclude only the worst geometry tail."""
    if percentile >= 100.0:
        return np.ones(len(data.iobs), dtype=bool), {}
    base = np.asarray(base_mask, dtype=bool)
    dx = data.xobs - data.xcal
    dy = data.yobs - data.ycal
    dz = data.zobs - data.zcal
    dr = np.sqrt(dx * dx + dy * dy)
    finite_dr = np.isfinite(dr)
    finite_dz = np.isfinite(dz)
    geom_base = base & finite_dr & finite_dz
    if not np.any(geom_base):
        return np.ones(len(data.iobs), dtype=bool), {"status": "no_finite_xds_observed_geometry"}
    dr_cut = float(np.nanpercentile(dr[geom_base], percentile))
    dz_cut = float(np.nanpercentile(np.abs(dz[geom_base]), percentile))
    finite_rlp = np.isfinite(data.rlp) & (data.rlp > 0)
    log_rlp = np.full(len(data.iobs), np.nan, dtype=float)
    log_rlp[finite_rlp] = np.log(data.rlp[finite_rlp])
    rlp_base = geom_base & np.isfinite(log_rlp)
    if np.any(rlp_base):
        med = float(np.nanmedian(log_rlp[rlp_base]))
        mad = float(np.nanmedian(np.abs(log_rlp[rlp_base] - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(log_rlp[rlp_base]))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        rlp_z = np.abs((log_rlp - med) / sigma)
        rlp_cut = float(np.nanpercentile(rlp_z[rlp_base], percentile))
        rlp_keep = ~np.isfinite(rlp_z) | (rlp_z <= rlp_cut)
    else:
        rlp_cut = float("nan")
        rlp_keep = np.ones(len(data.iobs), dtype=bool)
    keep = (~finite_dr | (dr <= dr_cut)) & (~finite_dz | (np.abs(dz) <= dz_cut)) & rlp_keep
    return keep, {
        "status": "ok",
        "percentile": float(percentile),
        "dr_cut_px": dr_cut,
        "abs_dz_cut_frame": dz_cut,
        "rlp_log_robust_z_cut": rlp_cut,
        "base_rows": int(np.count_nonzero(base)),
        "kept_base_rows": int(np.count_nonzero(base & keep)),
        "rejected_base_rows": int(np.count_nonzero(base & ~keep)),
    }


def row_selection(data: XDSData, target, positive_only: bool, min_d: float, min_peak: float, max_rows: int = 0, geometry_percentile: float = 100.0):
    mask = (
        np.isfinite(target)
        & np.isfinite(data.sigma)
        & (np.asarray(data.sigma, dtype=float) > 0.0)
        & np.isfinite(data.zcal)
        & np.isfinite(data.rlp)
        & (np.asarray(data.rlp, dtype=float) > 0.0)
    )
    if positive_only:
        mask &= target > 0.0
    if min_peak > 0:
        mask &= data.peak >= min_peak
    if min_d > 0:
        dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
        mask &= dvals >= min_d
    geom_keep, _geom_summary = geometry_tail_mask(data, mask, geometry_percentile)
    mask &= geom_keep
    idx = np.where(mask)[0]
    if max_rows > 0 and len(idx) > max_rows:
        n_bins = max(1, min(10, int(math.sqrt(int(max_rows)))))
        dvals = np.asarray(
            [data.cell.calculate_d(tuple(int(v) for v in data.hkl[i])) for i in idx],
            dtype=float,
        )
        frames = np.asarray(data.zcal[idx], dtype=float)
        isigi = np.asarray(target[idx], dtype=float) / np.asarray(data.sigma[idx], dtype=float)

        def rank_bins(values):
            order = np.argsort(values, kind="mergesort")
            bins = np.empty(len(values), dtype=int)
            bins[order] = np.minimum(n_bins - 1, np.arange(len(values)) * n_bins // len(values))
            return bins

        frame_bins = rank_bins(frames)
        resolution_bins = rank_bins(1.0 / np.maximum(dvals, 1e-12))
        strata = {}
        for local, row_idx in enumerate(idx):
            strata.setdefault((int(frame_bins[local]), int(resolution_bins[local])), []).append(
                (float(isigi[local]), int(row_idx))
            )
        queues = [
            [row_idx for _score, row_idx in sorted(rows, key=lambda item: (item[0], -item[1]))]
            for _key, rows in sorted(strata.items())
        ]
        selected = []
        while len(selected) < int(max_rows) and any(queues):
            for queue in queues:
                if queue and len(selected) < int(max_rows):
                    selected.append(queue.pop())
        idx = np.asarray(selected, dtype=int)
    return idx


def row_contract_selection(data: XDSData, target, path: Path) -> tuple[np.ndarray, dict]:
    """Load one immutable row-membership contract and verify it against XDS.

    Continuation jobs normally rebuild their eligible observations from the raw
    XDS file. That is unsafe after a multi-dataset policy/scale promotion: the
    promoted result may intentionally retain a strict subset of otherwise
    eligible rows. A contract therefore selects *only* its recorded row
    indices and fails closed if it no longer describes the supplied XDS input.

    The source is a GROND ``native_dyn_calc.csv`` (policy annotations are
    allowed). ``row_index,h,k,l,target_I,xds_sigma,d_A`` bind both row identity
    and the raw observation rather than merely trusting a list of integers.
    """

    contract = Path(path).expanduser().resolve()
    if not contract.is_file():
        raise ValueError(f"row contract is missing: {contract}")
    required = ("row_index", "h", "k", "l", "target_I", "xds_sigma", "d_A")
    with contract.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in required if field not in fields]
        if missing:
            raise ValueError(
                f"row contract {contract} is missing required column(s): {', '.join(missing)}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"row contract {contract} contains no rows")

    seen: set[int] = set()
    errors: list[str] = []
    checked_target = checked_sigma = checked_d = 0
    for ordinal, row in enumerate(rows, start=2):
        try:
            idx = int(str(row["row_index"]).strip())
            hkl = tuple(int(str(row[field]).strip()) for field in ("h", "k", "l"))
            target_i = float(str(row["target_I"]).strip())
            xds_sigma = float(str(row["xds_sigma"]).strip())
            d_expected = float(str(row["d_A"]).strip())
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"line {ordinal}: invalid row identity/observation: {exc}")
            if len(errors) >= 10:
                break
            continue
        if idx < 0 or idx >= len(data.hkl):
            errors.append(f"line {ordinal}: row_index {idx} is outside XDS range [0, {len(data.hkl)})")
        elif idx in seen:
            errors.append(f"line {ordinal}: duplicate row_index {idx}")
        else:
            seen.add(idx)
            observed_hkl = tuple(int(value) for value in np.asarray(data.hkl[idx], dtype=int))
            if observed_hkl != hkl:
                errors.append(
                    f"line {ordinal}: row_index {idx} HKL mismatch; contract={hkl}, XDS={observed_hkl}"
                )
            else:
                if not (
                    math.isfinite(target_i)
                    and math.isclose(float(target[idx]), target_i, rel_tol=1e-11, abs_tol=1e-10)
                ):
                    errors.append(
                        f"line {ordinal}: row_index {idx} target_I mismatch; "
                        f"contract={target_i:.17g}, XDS={float(target[idx]):.17g}"
                    )
                else:
                    checked_target += 1
                if not (
                    math.isfinite(xds_sigma)
                    and math.isclose(float(data.sigma[idx]), xds_sigma, rel_tol=1e-11, abs_tol=1e-10)
                ):
                    errors.append(
                        f"line {ordinal}: row_index {idx} xds_sigma mismatch; "
                        f"contract={xds_sigma:.17g}, XDS={float(data.sigma[idx]):.17g}"
                    )
                else:
                    checked_sigma += 1
                d_actual = float(data.cell.calculate_d(observed_hkl))
                if not (
                    math.isfinite(d_expected)
                    and math.isclose(d_actual, d_expected, rel_tol=1e-10, abs_tol=1e-10)
                ):
                    errors.append(
                        f"line {ordinal}: row_index {idx} d_A mismatch; "
                        f"contract={d_expected:.17g}, XDS={d_actual:.17g}"
                    )
                else:
                    checked_d += 1
        if len(errors) >= 10:
            break
    if errors:
        raise ValueError(
            "row contract does not bind exactly to the supplied XDS observations; "
            "refusing to continue with changed row membership:\n- " + "\n- ".join(errors)
        )

    indices = np.asarray(sorted(seen), dtype=int)
    if len(indices) != len(rows):
        raise ValueError(
            f"row contract {contract} selected {len(indices)} unique indices from {len(rows)} rows"
        )
    return indices, {
        "status": "verified_exact_row_contract",
        "path": str(contract),
        "sha256": sha256(contract),
        "requested_rows": int(len(rows)),
        "selected_rows": int(len(indices)),
        "row_index_min": int(indices[0]),
        "row_index_max": int(indices[-1]),
        "identity_columns": list(required),
        "target_rows_verified": int(checked_target),
        "sigma_rows_verified": int(checked_sigma),
        "d_spacing_rows_verified": int(checked_d),
        "selection_rule": "contract row_index membership; no raw-XDS eligibility rows were added or removed",
    }


def parse_lst(path: Path):
    out = {"lst": str(path), "status": "missing"}
    if not path.exists():
        return out
    out["status"] = "ok"
    out.update(parse_shelx_lst_pathologies(path))
    text = path.read_text(errors="replace")
    for line in text.splitlines():
        terms = parse_shelx_wght_terms(line)
        if terms:
            out["active_wght_a"] = terms[0]
            out["active_wght_b"] = terms[1] if len(terms) >= 2 else 0.0
            out["active_wght_terms"] = terms
            break
    m = re.findall(r"(\d+)\s+Reflections read, of which\s+(\d+)\s+rejected", text)
    if m:
        out["reflections_read"] = int(m[-1][0])
        out["reflections_rejected"] = int(m[-1][1])
    m = re.findall(r"(\d+)\s+Unique reflections, of which\s+(\d+)\s+suppressed", text)
    if m:
        out["unique"] = int(m[-1][0])
        out["suppressed"] = int(m[-1][1])
    m = re.findall(r"R\(int\) =\s*([0-9.]+)\s+R\(sigma\) =\s*([0-9.]+)", text)
    if m:
        out["Rint"] = float(m[-1][0])
        out["Rsigma"] = float(m[-1][1])
    m = re.findall(r"wR2 =\s*([0-9.]+), GooF = S =\s*([0-9.]+).* for all data", text)
    if m:
        out["wR2_all"] = float(m[-1][0])
        out["GooF"] = float(m[-1][1])
    m = re.findall(r"R1 =\s*([0-9.]+) for\s+(\d+) Fo > 4sig\(Fo\) and\s+([0-9.]+) for all\s+(\d+) data", text)
    if m:
        out["R1_gt"] = float(m[-1][0])
        out["N_gt"] = int(m[-1][1])
        out["R1_all"] = float(m[-1][2])
        out["N_all"] = int(m[-1][3])
    if "wR2_all" not in out or "R1_all" not in out:
        if "HAS WRONG FORMAT" in text:
            out["status"] = "bad_hkl_format"
        else:
            out["status"] = "no_regularization_stats"
    return out


def parse_recommended_wght(lst_path: Path) -> tuple[float, float] | None:
    """Return SHELXL's recommended WGHT values from an LST file, if present."""
    if not lst_path.exists():
        return None
    text = lst_path.read_text(errors="replace")
    matches = list(_iter_recommended_wght_terms(text))
    if not matches:
        return None
    terms = matches[-1]
    return terms[0], terms[1] if len(terms) >= 2 else 0.0


def shelx_res_health(
    res_path: Path,
    lst_path: Path | None = None,
    *,
    require_lst: bool = True,
) -> dict:
    """Return whether a SHELXL RES file is usable as the next model.

    SHELXL can fail in ways that still leave a path on disk.  In particular,
    interrupted or failed app-bundle runs on macOS have produced zero-byte
    ``.res`` files.  Those must not be fed back into the next Bloch cycle.
    """

    if not res_path.exists():
        return {"ok": False, "reason": "missing_res", "path": str(res_path)}
    try:
        size = res_path.stat().st_size
    except OSError as exc:
        return {"ok": False, "reason": "stat_failed", "path": str(res_path), "error": str(exc)}
    if size <= 0:
        return {"ok": False, "reason": "empty_res", "path": str(res_path), "size": int(size)}
    try:
        lines = res_path.read_text(errors="replace").splitlines()
    except OSError as exc:
        return {"ok": False, "reason": "read_failed", "path": str(res_path), "size": int(size), "error": str(exc)}
    has_cell = any(line.strip().upper().startswith("CELL ") for line in lines)
    if not has_cell:
        return {"ok": False, "reason": "missing_cell", "path": str(res_path), "size": int(size)}
    try:
        model = parse_model(res_path)
    except Exception as exc:
        return {
            "ok": False,
            "reason": "invalid_structure_model",
            "path": str(res_path),
            "size": int(size),
            "error": f"{type(exc).__name__}: {exc}",
        }
    atom_count = len(model.get("atoms") or [])
    sfac_count = len(model.get("sfac") or [])
    if atom_count <= 0 or sfac_count <= 0:
        return {
            "ok": False,
            "reason": "missing_model_records",
            "path": str(res_path),
            "size": int(size),
            "atom_count": int(atom_count),
            "sfac_count": int(sfac_count),
        }
    if not require_lst and lst_path is None:
        # Structural-only validation is reserved for a caller that is
        # deliberately reusing an already selected input model without running
        # SHELXL.  Do not let an unrelated sibling LST alter that explicit
        # decision; generated endpoint models always require a conclusive LST.
        pathology = {
            "lst": "",
            "pathology_status": "not_applicable_structural_only",
            "npd_count": None,
            "split_count": None,
            "npd_atoms": [],
            "has_npd": None,
            "has_split_warnings": None,
        }
        pathology_safe = True
    else:
        resolved_lst = lst_path or res_path.with_suffix(".lst")
        pathology = parse_shelx_lst_pathologies(resolved_lst)
        pathology_safe = shelx_pathology_safe_for_feedback(pathology)
    return {
        "ok": True,
        "reason": "ok",
        "path": str(res_path),
        "size": int(size),
        "sha256": sha256(res_path),
        "atom_count": int(atom_count),
        "sfac_count": int(sfac_count),
        **pathology,
        "lst_required": bool(require_lst),
        "safe_for_feedback": pathology_safe,
    }


def shelxl_statistics_complete(stats: dict) -> bool:
    """Return whether SHELXL printed a complete, finite endpoint record."""

    try:
        r1_all = float(stats.get("R1_all"))
        wr2_all = float(stats.get("wR2_all"))
        n_all = int(stats.get("N_all", 0) or 0)
    except (TypeError, ValueError):
        return False
    return bool(
        stats.get("returncode") == 0
        and math.isfinite(r1_all)
        and math.isfinite(wr2_all)
        and n_all > 0
    )


def classify_shelxl_iteration_rejection(stats: dict) -> tuple[str, str] | None:
    """Classify a structurally valid, zero-return-code SHELXL iteration.

    ``safe_for_feedback=False`` deliberately covers both an explicit NPD and
    missing/truncated final pathology diagnostics.  Keep those cases distinct:
    an NPD is an ADP pathology, while absent statistics or an incomplete final
    diagnostic footer are separate output-integrity failures.
    """

    health = stats.get("res_health") if isinstance(stats.get("res_health"), dict) else {}
    if bool(health.get("has_npd")) or int(health.get("npd_count", 0) or 0) > 0:
        return "adp_pathology", "adp_pathology"
    if not shelxl_statistics_complete(stats):
        parsed_status = str(stats.get("status", "missing"))
        return (
            "invalid_shelxl_statistics",
            "SHELXL output did not contain a complete finite R1/wR2/N statistics record "
            f"(parse status={parsed_status})",
        )
    if health.get("safe_for_feedback") is False:
        pathology_status = str(health.get("pathology_status", "missing"))
        return (
            "incomplete_shelxl_diagnostics",
            "SHELXL output did not contain a conclusive final NPD/split pathology summary "
            f"(pathology status={pathology_status})",
        )
    return None


def shelxl_iteration_safe(stats: dict) -> bool:
    """Return whether one completed SHELXL iteration is safe to retain."""

    health = stats.get("res_health")
    return bool(
        shelxl_statistics_complete(stats)
        and isinstance(health, dict)
        and health.get("ok") is True
        and int(health.get("atom_count", 0) or 0) > 0
        and int(health.get("sfac_count", 0) or 0) > 0
        and health.get("safe_for_feedback") is not False
    )


def snapshot_safe_shelxl_iteration(work_dir: Path, stem: str, iteration: int, stats: dict) -> dict:
    """Snapshot matching safe SHELXL artifacts before a later LS1 pass can overwrite them."""

    if not shelxl_iteration_safe(stats):
        raise ValueError("cannot snapshot an unsafe SHELXL iteration")
    source_res = work_dir / f"{stem}.res"
    source_lst = work_dir / f"{stem}.lst"
    source_fcf = work_dir / f"{stem}.fcf"
    snapshot_res = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.res"
    snapshot_lst = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.lst"
    snapshot_fcf = work_dir / f"shelxl_safe_iter_{int(iteration):02d}.fcf"
    shutil.copy2(source_res, snapshot_res)
    if source_lst.exists():
        shutil.copy2(source_lst, snapshot_lst)
    else:
        snapshot_lst.unlink(missing_ok=True)
    if source_fcf.exists():
        shutil.copy2(source_fcf, snapshot_fcf)
    else:
        snapshot_fcf.unlink(missing_ok=True)
    return {
        "iteration": int(iteration),
        "res": str(snapshot_res),
        "lst": str(snapshot_lst) if snapshot_lst.exists() else "",
        "fcf": str(snapshot_fcf) if snapshot_fcf.exists() else "",
        "stats": dict(stats),
    }


def select_last_safe_shelxl_iteration(
    work_dir: Path,
    stem: str,
    history: list[dict],
    last_safe: dict | None,
) -> dict:
    """Select the final safe iteration, restoring it after a later rejected pass."""

    attempted = dict(history[-1]) if history else {"returncode": None, "status": "not_run"}
    attempted_safe = shelxl_iteration_safe(attempted)
    restored = bool(last_safe and not attempted_safe)
    rejected_snapshot_res = ""
    rejected_snapshot_lst = ""
    rejected_snapshot_fcf = ""
    if history and restored:
        try:
            rejected_iteration = int(attempted.get("iteration", len(history)))
        except (TypeError, ValueError):
            rejected_iteration = len(history)
        canonical_res = work_dir / f"{stem}.res"
        canonical_lst = work_dir / f"{stem}.lst"
        canonical_fcf = work_dir / f"{stem}.fcf"
        rejected_res = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.res"
        rejected_lst = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.lst"
        rejected_fcf = work_dir / f"shelxl_rejected_iter_{rejected_iteration:02d}.fcf"
        if canonical_res.exists():
            shutil.copy2(canonical_res, rejected_res)
            rejected_snapshot_res = str(rejected_res)
        else:
            rejected_res.unlink(missing_ok=True)
        if canonical_lst.exists():
            shutil.copy2(canonical_lst, rejected_lst)
            rejected_snapshot_lst = str(rejected_lst)
        else:
            rejected_lst.unlink(missing_ok=True)
        attempted["lst"] = rejected_snapshot_lst
        if canonical_fcf.exists():
            shutil.copy2(canonical_fcf, rejected_fcf)
            rejected_snapshot_fcf = str(rejected_fcf)
        else:
            rejected_fcf.unlink(missing_ok=True)
        attempted["rejected_snapshot_res"] = rejected_snapshot_res
        attempted["rejected_snapshot_lst"] = rejected_snapshot_lst
        attempted["rejected_snapshot_fcf"] = rejected_snapshot_fcf
        rejected_health = attempted.get("res_health")
        if isinstance(rejected_health, dict):
            rejected_health = dict(rejected_health)
            rejected_health["path"] = rejected_snapshot_res
            rejected_health["lst"] = rejected_snapshot_lst
            attempted["res_health"] = rejected_health
        history[-1].update(attempted)
    final = attempted
    if restored:
        canonical_res = work_dir / f"{stem}.res"
        canonical_lst = work_dir / f"{stem}.lst"
        canonical_fcf = work_dir / f"{stem}.fcf"
        shutil.copy2(Path(str(last_safe["res"])), canonical_res)
        snapshot_lst = str(last_safe.get("lst", ""))
        if snapshot_lst and Path(snapshot_lst).exists():
            shutil.copy2(Path(snapshot_lst), canonical_lst)
        else:
            canonical_lst.unlink(missing_ok=True)
        snapshot_fcf = str(last_safe.get("fcf", ""))
        if snapshot_fcf and Path(snapshot_fcf).exists():
            shutil.copy2(Path(snapshot_fcf), canonical_fcf)
        else:
            canonical_fcf.unlink(missing_ok=True)
        final = dict(last_safe["stats"])
        final["status"] = "ok"
        final.pop("invalid_res_reason", None)
        final["res_health"] = shelx_res_health(canonical_res, canonical_lst)

    selected_iteration = int(last_safe["iteration"]) if last_safe and (attempted_safe or restored) else None
    final["selected_iteration"] = selected_iteration
    final["restored_last_safe_iteration"] = restored
    if not attempted_safe:
        final["rejected_iteration"] = attempted.get("iteration")
        final["rejected_status"] = str(attempted.get("status", ""))
        final["rejected_invalid_res_reason"] = str(attempted.get("invalid_res_reason", ""))
        rejected_health = attempted.get("res_health")
        final["rejected_res_health"] = dict(rejected_health) if isinstance(rejected_health, dict) else {}
        final["rejected_snapshot_res"] = rejected_snapshot_res
        final["rejected_snapshot_lst"] = rejected_snapshot_lst
        final["rejected_snapshot_fcf"] = rejected_snapshot_fcf
    return final


def write_active_wght(path: Path, a: float, b: float) -> dict:
    """Force one active WGHT line in a SHELX input or output file."""

    if not path.exists():
        return {"written": False, "reason": "missing_file"}
    lines = []
    for line in path.read_text(errors="replace").splitlines():
        lines.append(line)
        if line.strip().upper().startswith("END"):
            break
    lines = set_active_wght_lines(lines, a, b)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return {"written": True, "active_wght_a": float(a), "active_wght_b": float(b)}


def promote_recommended_wght(ins_path: Path, recommended: tuple[float, float] | None) -> dict:
    """Move SHELXL's recommended WGHT into the active pre-END input block.

    SHELXL writes the suggested next WGHT after END in the RES file.  Copying RES
    to INS verbatim leaves that recommendation ignored on the next cycle, so the
    first active WGHT continues to control the weights.  GROND promotes the
    recommendation explicitly and discards the post-END tail in the working INS.
    """
    if recommended is None or not ins_path.exists():
        return {"promoted": False}
    a, b = recommended
    lines = ins_path.read_text(errors="replace").splitlines()
    active_lines: list[str] = []
    saw_end = False
    for line in lines:
        active_lines.append(line)
        if re.match(r"^\s*END\b", line, flags=re.IGNORECASE):
            saw_end = True
            break
    if not active_lines:
        return {"promoted": False}

    del saw_end
    active_lines = set_active_wght_lines(active_lines, a, b)
    ins_path.write_text("\n".join(active_lines).rstrip() + "\n")
    return {"promoted": True, "active_wght_a": float(a), "active_wght_b": float(b)}


def _active_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    try:
        a = float(stats["active_wght_a"])
        b = float(stats.get("active_wght_b", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


def recommended_wght_from_stats(stats: dict) -> tuple[float, float] | None:
    try:
        a = float(stats["recommended_wght_a"])
        b = float(stats.get("recommended_wght_b", 0.0))
    except (KeyError, TypeError, ValueError):
        return None
    if not math.isfinite(a) or not math.isfinite(b):
        return None
    return a, b


DEFAULT_NPD_DAMP_RETRIES = (100.0, 300.0, 1000.0, 3000.0, 10000.0)
DEFAULT_NPD_DAMP_LIMSE = 15.0


def write_active_damp(path: Path, damp: float, limse: float = DEFAULT_NPD_DAMP_LIMSE) -> dict:
    """Write one active pre-END SHELXL DAMP instruction.

    This controls only the size of the least-squares model step.  It does not
    add restraints, permit NPD atoms, or alter the corrected reflection data.
    """

    if not path.exists():
        return {"written": False, "reason": "missing_file"}
    lines = path.read_text(errors="replace").splitlines()
    active_end = next(
        (index for index, line in enumerate(lines) if line.strip().upper().startswith("END")),
        len(lines),
    )
    active_damp = [
        index
        for index, line in enumerate(lines[:active_end])
        if line.strip().upper().startswith("DAMP")
    ]
    damp_line = f"DAMP {float(damp):.8g} {float(limse):.8g}"
    if active_damp:
        lines[active_damp[0]] = damp_line
        for index in reversed(active_damp[1:]):
            del lines[index]
    else:
        insert_at = next(
            (
                index
                for index, line in enumerate(lines[:active_end])
                if line.strip().upper().startswith("FVAR")
            ),
            active_end,
        )
        lines.insert(insert_at, damp_line)
    path.write_text("\n".join(lines).rstrip() + "\n")
    return {
        "written": True,
        "active_damp": float(damp),
        "active_damp_limse": float(limse),
    }


def _shelxl_attempt_label(value: float | None) -> str:
    if value is None:
        return "undamped"
    return f"damp_{float(value):g}".replace(".", "p").replace("-", "m")


def archive_shelxl_attempt(
    work_dir: Path,
    stem: str,
    iteration: int,
    attempt: int,
    damp: float | None,
) -> dict:
    """Preserve one SHELXL attempt before a bounded recovery overwrites it."""

    label = _shelxl_attempt_label(damp)
    prefix = f"shelxl_attempt_iter_{int(iteration):02d}_{int(attempt):02d}_{label}"
    archived: dict[str, object] = {
        "attempt": int(attempt),
        "label": label,
        "damp": None if damp is None else float(damp),
    }
    for suffix in ("res", "lst", "fcf"):
        source = work_dir / f"{stem}.{suffix}"
        target = work_dir / f"{prefix}.{suffix}"
        if source.exists():
            shutil.copy2(source, target)
            archived[suffix] = str(target)
        else:
            target.unlink(missing_ok=True)
            archived[suffix] = ""
    return archived


def archived_shelxl_attempt_record(stats: dict, archived: dict) -> dict:
    """Bind an attempt's health paths to its immutable archived artifacts."""

    record = {**dict(stats), **dict(archived)}
    health = record.get("res_health")
    if isinstance(health, dict):
        health = dict(health)
        health["path"] = str(archived.get("res", ""))
        health["lst"] = str(archived.get("lst", ""))
        record["res_health"] = health
    return record


def run_one_shelxl_attempt(
    work_dir: Path,
    stem: str,
    executable: str,
    *,
    iteration: int,
    log_name: str,
) -> tuple[subprocess.CompletedProcess[str], dict]:
    """Run and classify one SHELXL attempt after clearing stale outputs."""

    res = work_dir / f"{stem}.res"
    lst = work_dir / f"{stem}.lst"
    fcf = work_dir / f"{stem}.fcf"
    res.unlink(missing_ok=True)
    lst.unlink(missing_ok=True)
    fcf.unlink(missing_ok=True)
    proc = subprocess.run(
        [executable, stem],
        cwd=str(work_dir),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    (work_dir / log_name).write_text(proc.stdout)
    stats = {"iteration": int(iteration), "returncode": proc.returncode, **parse_lst(lst)}
    recommended_wght = parse_recommended_wght(lst)
    if recommended_wght is not None:
        stats["recommended_wght_a"] = recommended_wght[0]
        stats["recommended_wght_b"] = recommended_wght[1]
    health = shelx_res_health(res, lst)
    stats["res_health"] = health
    if proc.returncode != 0:
        stats["status"] = "shelxl_failed"
        stats["invalid_res_reason"] = f"SHELXL returned {proc.returncode}"
    elif not health["ok"]:
        stats["status"] = "invalid_res_output"
        stats["invalid_res_reason"] = health["reason"]
    else:
        rejection = classify_shelxl_iteration_rejection(stats)
        if rejection is not None:
            stats["status"], stats["invalid_res_reason"] = rejection
    return proc, stats


def recover_npd_shelxl_attempt(
    work_dir: Path,
    stem: str,
    executable: str,
    *,
    iteration: int,
    base_ins_text: str,
    initial_stats: dict,
    damp_values: tuple[float, ...] = DEFAULT_NPD_DAMP_RETRIES,
    limse: float = DEFAULT_NPD_DAMP_LIMSE,
) -> tuple[subprocess.CompletedProcess[str] | None, dict]:
    """Retry one rejected NPD step from its unchanged input with bounded DAMP.

    The first zero-NPD, complete SHELXL result is accepted.  Every attempt is
    archived and all failed trials remain visible in the returned provenance.
    """

    attempts: list[dict] = []
    initial_archive = archive_shelxl_attempt(work_dir, stem, iteration, 0, None)
    attempts.append(archived_shelxl_attempt_record(initial_stats, initial_archive))
    last_proc: subprocess.CompletedProcess[str] | None = None
    last_stats = dict(initial_stats)
    ins = work_dir / f"{stem}.ins"
    for attempt_number, damp in enumerate(damp_values, start=1):
        ins.write_text(base_ins_text)
        damp_write = write_active_damp(ins, damp, limse)
        proc, stats = run_one_shelxl_attempt(
            work_dir,
            stem,
            executable,
            iteration=iteration,
            log_name=(
                f"shelxl_iter_{int(iteration):02d}_recovery_"
                f"{int(attempt_number):02d}_{_shelxl_attempt_label(damp)}.log"
            ),
        )
        stats["recovery_attempt"] = int(attempt_number)
        stats["active_damp"] = float(damp)
        stats["active_damp_limse"] = float(limse)
        stats["damp_write"] = damp_write
        archived = archive_shelxl_attempt(
            work_dir,
            stem,
            iteration,
            attempt_number,
            damp,
        )
        attempts.append(archived_shelxl_attempt_record(stats, archived))
        last_proc = proc
        last_stats = stats
        if shelxl_iteration_safe(stats):
            stats["status"] = "ok"
            stats["npd_recovery"] = {
                "trigger": "adp_pathology",
                "status": "accepted_first_safe_damped_step",
                "accepted_damp": float(damp),
                "limse": float(limse),
                "attempts": attempts,
            }
            return proc, stats
        if str(stats.get("status", "")) != "adp_pathology":
            break
    last_stats["npd_recovery"] = {
        "trigger": "adp_pathology",
        "status": "exhausted_without_safe_step",
        "accepted_damp": None,
        "limse": float(limse),
        "attempts": attempts,
    }
    return last_proc, last_stats


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
            write_endpoint_ins(res, ins, hkl_name=f"{stem}.hkl", keep_exti=False, wght=active, ls_cycles=endpoint_ls_cycles)
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
            keep_exti=False,
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


def endpoint_resolution_modulation(data: XDSData, valid, endpoint_b: float):
    if abs(float(endpoint_b)) <= 0.0:
        return np.ones(len(data.hkl), dtype=float), {"endpoint_b": float(endpoint_b), "s2_center": float("nan")}
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    s2 = 1.0 / np.maximum(4.0 * dvals * dvals, 1e-30)
    center_mask = np.asarray(valid, dtype=bool) & np.isfinite(s2)
    s2_center = float(np.nanmedian(s2[center_mask])) if np.any(center_mask) else float(np.nanmedian(s2))
    mod = np.exp(float(endpoint_b) * (s2 - s2_center))
    return mod, {"endpoint_b": float(endpoint_b), "s2_center": s2_center}


def row_resolution_modulation(data: XDSData, valid, row_resolution_b: float):
    """Resolution-envelope factor applied to raw dynamical row predictions.

    Positive values increase high-resolution dynamical intensities relative to
    low-resolution rows. In corrected-HKL diagnostics this is approximately the
    alternate sign of endpoint_b because the endpoint divides observations by
    the dynamical correction.
    """

    row_resolution_b = float(row_resolution_b)
    if abs(row_resolution_b) <= 0.0:
        return np.ones(len(data.hkl), dtype=float), {"row_resolution_b": row_resolution_b, "row_s2_center": float("nan")}
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    s2 = 1.0 / np.maximum(4.0 * dvals * dvals, 1e-30)
    center_mask = np.asarray(valid, dtype=bool) & np.isfinite(s2)
    s2_center = float(np.nanmedian(s2[center_mask])) if np.any(center_mask) else float(np.nanmedian(s2))
    exponent = np.clip(row_resolution_b * (s2 - s2_center), -20.0, 20.0)
    mod = np.exp(exponent)
    return mod, {"row_resolution_b": row_resolution_b, "row_s2_center": s2_center}


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
    valid = (
        arrays["valid"]
        & np.isfinite(target)
        & np.isfinite(sig)
        & (np.asarray(sig, dtype=float) > 0.0)
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
    Icorr = target / denom
    sigcorr = sig / denom
    modulation, modulation_meta = endpoint_resolution_modulation(data, valid, endpoint_b)
    Icorr = Icorr * modulation
    sigcorr = sigcorr * modulation
    if rescale == "median-iobs":
        src = float(np.nanmedian(target[valid]))
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


def write_row_table(path: Path, data: XDSData, target, best_eval, masks):
    arrays = best_eval["arrays"]
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [
            "row_index", "h", "k", "l", "d_A", "target_I", "sigma_eff", "xds_sigma", "peak", "corr", "rlp",
            "group", "detector_bin", "basis_size", "center_sg_inv_A", "omega_bragg_deg", "omega_bragg_minus_center_deg",
            "sg_window_inv_A", "sg_window_raw_inv_A", "sg_window_half_frames", "sg_window_clipped",
            "sg_dSg_domega_inv_A_per_deg", "sg_dSg_dframe_inv_A", "sg_jitter_sigma_inv_A",
            "sg_jitter_d_inv_A", "sg_jitter_frame_fraction", "sg_jitter_kernel_points",
            "sg_profile_points_total", "sg_profile_weight_sum",
            "sg_profile_same_branch_continuation_used",
            "sg_profile_same_branch_continuation_segments",
            "max_sg_solve_residual",
            "split", "row_resolution_factor", "dyn_raw_unmodulated", "dyn_raw", "born_raw", "q_dyn_over_born", "dyn_scaled", "born_scaled",
            "dyn_residual_over_sigma", "born_residual_over_sigma",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        default_float = np.full(len(target), np.nan, dtype=float)
        default_int = np.zeros(len(target), dtype=int)
        default_one_int = np.ones(len(target), dtype=int)
        default_bool = np.zeros(len(target), dtype=bool)
        for i in np.where(arrays["valid"])[0]:
            split = "free" if masks["free"][i] else "work"
            q = arrays["dyn_raw"][i] / max(arrays["born_raw"][i], 1e-30)
            w.writerow({
                "row_index": int(i),
                "h": int(data.hkl[i, 0]), "k": int(data.hkl[i, 1]), "l": int(data.hkl[i, 2]),
                "d_A": float(dvals[i]),
                "target_I": float(target[i]),
                "sigma_eff": float(arrays["sigma_eff"][i]),
                "xds_sigma": float(data.sigma[i]),
                "peak": float(data.peak[i]),
                "corr": float(data.corr[i]),
                "rlp": float(data.rlp[i]),
                "group": int(arrays["groups"][i]),
                "detector_bin": int(arrays.get("detector_bins", np.zeros(len(target), dtype=int))[i]),
                "basis_size": int(arrays.get("basis_size", np.zeros(len(target), dtype=int))[i]),
                "center_sg_inv_A": float(arrays.get("center_sg", default_float)[i]),
                "omega_bragg_deg": float(arrays.get("omega_bragg", default_float)[i]),
                "omega_bragg_minus_center_deg": float(arrays.get("omega_delta", default_float)[i]),
                "sg_window_inv_A": float(arrays.get("sg_window", default_float)[i]),
                "sg_window_raw_inv_A": float(arrays.get("sg_window_raw", default_float)[i]),
                "sg_window_half_frames": float(arrays.get("sg_window_half_frames", default_float)[i]),
                "sg_window_clipped": int(bool(arrays.get("sg_window_clipped", default_bool)[i])),
                "sg_dSg_domega_inv_A_per_deg": float(arrays.get("sg_dSg_domega", default_float)[i]),
                "sg_dSg_dframe_inv_A": float(arrays.get("sg_dSg_dframe", default_float)[i]),
                "sg_jitter_sigma_inv_A": float(arrays.get("sg_jitter_sigma", default_float)[i]),
                "sg_jitter_d_inv_A": float(arrays.get("sg_jitter_d_inv", default_float)[i]),
                "sg_jitter_frame_fraction": float(arrays.get("sg_jitter_frame_fraction", default_float)[i]),
                "sg_jitter_kernel_points": int(arrays.get("sg_jitter_kernel_points", default_int)[i]),
                "sg_profile_points_total": int(arrays.get("sg_profile_points_total", default_int)[i]),
                "sg_profile_weight_sum": float(arrays.get("sg_profile_weight_sum", default_float)[i]),
                "sg_profile_same_branch_continuation_used": int(
                    bool(arrays.get("sg_profile_same_branch_continuation_used", default_bool)[i])
                ),
                "sg_profile_same_branch_continuation_segments": int(
                    arrays.get("sg_profile_same_branch_continuation_segments", default_one_int)[i]
                ),
                "max_sg_solve_residual": float(arrays.get("sg_residual", default_float)[i]),
                "split": split,
                "row_resolution_factor": float(arrays.get("row_resolution_factor", np.ones(len(target), dtype=float))[i]),
                "dyn_raw_unmodulated": float(arrays.get("dyn_raw_unmodulated", arrays["dyn_raw"])[i]),
                "dyn_raw": float(arrays["dyn_raw"][i]),
                "born_raw": float(arrays["born_raw"][i]),
                "q_dyn_over_born": float(q),
                "dyn_scaled": float(arrays["dyn_scaled"][i]),
                "born_scaled": float(arrays["born_scaled"][i]),
                "dyn_residual_over_sigma": float((arrays["dyn_scaled"][i] - target[i]) / max(arrays["sigma_eff"][i], 1e-12)),
                "born_residual_over_sigma": float((arrays["born_scaled"][i] - target[i]) / max(arrays["sigma_eff"][i], 1e-12)),
            })


def native_row_table_artifact(path: Path) -> dict[str, str]:
    """Return the byte-bound native row-table artifact recorded in summary.json."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"native row table is missing: {resolved}")
    return {
        "native_row_table": str(resolved),
        "native_row_table_sha256": sha256(resolved),
    }


def metric_row(trial):
    m = trial["metrics"]
    n = trial["nuisance"]
    q_control = trial.get("q_control") or {}
    return {
        **n,
        "q_control": q_control.get("mode", "real"),
        "q_control_shuffled_rows": q_control.get("shuffled_rows", 0),
        "q_control_active_bins": q_control.get("active_bins", 0),
        "q_control_logq_corr": q_control.get("logq_corr_original_shuffled", float("nan")),
        "valid_rows": trial["valid_rows"],
        "scale_count": trial["scale_count"],
        "work_dyn_wR2": m["work"]["dyn"]["wR2_all"],
        "free_dyn_wR2": m["free"]["dyn"]["wR2_all"],
        "all_dyn_wR2": m["all"]["dyn"]["wR2_all"],
        "all_dyn_R1": m["all"]["dyn"]["R1_all"],
        "all_dyn_R1gt": m["all"]["dyn"]["R1_gt"],
        "all_born_wR2": m["all"]["born"]["wR2_all"],
        "all_born_R1": m["all"]["born"]["R1_all"],
        "all_dyn_minus_born_wR2": m["all"]["dyn"]["wR2_all"] - m["all"]["born"]["wR2_all"],
    }


def trial_metric(trial, metric_name: str) -> float:
    if metric_name == "work_born_wR2":
        return float(trial["metrics"]["work"]["born"]["wR2_all"])
    split, value = metric_name.split("_", 1)
    if split not in trial["metrics"]:
        raise ValueError(f"unknown selection split {split!r} in {metric_name!r}")
    key = "wR2_all" if value == "wR2" else "R1_all"
    return float(trial["metrics"][split]["dyn"][key])


def rank_trials(trials, metric_name: str):
    """Rank nuisance trials without using held-out rows as an implicit tie-breaker."""

    return sorted(trials, key=lambda trial: trial_metric(trial, metric_name))


_NUISANCE_CATEGORICAL_FIELDS = {
    "thickness_model",
    "sg_jitter_mode",
    "absorption_mode",
    "dose_b_mode",
    "matrix_convention",
}
_NUISANCE_NULLABLE_NUMERIC_FIELDS = {
    "dose_b_ref_frame",
    "native_wght_a",
    "native_wght_b",
}


def nuisance_reference_matches(reference: dict, nuisance: dict) -> bool:
    """Return whether a trial reproduces an explicitly supplied nuisance state.

    The reference comes from the preceding automatic stage and may contain
    additional summary metadata.  Only fields in the frozen ``Nuisance``
    contract participate.  Matching NaNs are accepted for intentionally
    disabled terms such as ``dose_b_ref_frame``; one-sided NaNs and infinities
    are never treated as equal.
    """

    checked = 0
    for key in Nuisance.__dataclass_fields__:
        if key not in reference:
            continue
        checked += 1
        if key not in nuisance:
            return False
        expected = reference[key]
        actual = nuisance[key]
        if key in _NUISANCE_CATEGORICAL_FIELDS:
            if str(expected) != str(actual):
                return False
            continue
        try:
            expected_value = float(expected)
            actual_value = float(actual)
        except (TypeError, ValueError):
            return False
        if math.isnan(expected_value) and math.isnan(actual_value):
            if key in _NUISANCE_NULLABLE_NUMERIC_FIELDS:
                continue
            return False
        if not math.isfinite(expected_value) or not math.isfinite(actual_value):
            return False
        if not math.isclose(expected_value, actual_value, rel_tol=1e-12, abs_tol=1e-12):
            return False
    return checked > 0


def _trial_metric_value(trial: dict, split: str, metric: str) -> float:
    try:
        return float(trial["metrics"][split]["dyn"][metric])
    except (KeyError, TypeError, ValueError):
        return float("nan")


def _trial_split_count(trial: dict, split: str) -> int:
    value = _trial_metric_value(trial, split, "n")
    return max(0, int(value)) if math.isfinite(value) else 0


def _trial_full_row_score(trial: dict) -> float:
    """Mirror the public native-summary score for an in-memory trial."""

    dyn_wr2 = _trial_metric_value(trial, "all", "wR2_all")
    free_wr2 = _trial_metric_value(trial, "free", "wR2_all")
    dyn_r1 = _trial_metric_value(trial, "all", "R1_all")
    try:
        born_r1gt = float(trial["metrics"]["all"]["born"]["R1_gt"])
        born_cc = float(trial["metrics"]["all"]["born"]["CC"])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    values = (dyn_wr2, free_wr2, dyn_r1, born_r1gt, born_cc)
    if not all(math.isfinite(value) for value in values):
        return float("nan")
    return (
        dyn_wr2
        + 0.25 * free_wr2
        + 0.10 * dyn_r1
        + max(0.0, born_r1gt - 0.35)
        + max(0.0, 0.5 - born_cc)
    )


def _trial_valid_mask(trial: dict) -> np.ndarray | None:
    arrays = trial.get("arrays")
    if not isinstance(arrays, dict) or "valid" not in arrays:
        return None
    mask = np.asarray(arrays["valid"])
    if mask.ndim != 1:
        return None
    return np.asarray(mask, dtype=bool)


def _valid_mask_sha256(mask: np.ndarray | None) -> str:
    if mask is None:
        return ""
    payload = np.asarray(mask, dtype=np.uint8)
    digest = hashlib.sha256()
    digest.update(str(payload.shape).encode())
    digest.update(b"\0")
    digest.update(payload.tobytes())
    return digest.hexdigest()


def _promotion_trial_record(trial: dict, selection_metric: str) -> dict:
    return {
        "nuisance": dict(trial.get("nuisance") or {}),
        "selection_metric": str(selection_metric),
        "selection_value": trial_metric(trial, selection_metric),
        "valid_rows": int(trial.get("valid_rows", 0)),
        "scale_count": int(trial.get("scale_count", 0)),
        "scale_fit_split": str(trial.get("scale_fit_split", "")),
        "metrics": trial.get("metrics") or {},
    }


def select_trial_with_promotion_gate(
    ranked_trials: list[dict],
    baseline_nuisance: dict,
    *,
    selection_metric: str,
    min_gain: float,
    max_free_wr2_loss: float,
    max_all_r1_loss: float,
) -> tuple[dict, dict]:
    """Gate one work-ranked nuisance winner against the exact starting trial.

    Ranking remains blind to held-out rows.  Only after the single work winner
    is known do free rows act as a no-harm validation gate.  Both candidates
    were already evaluated in the same grid with the same model, observations,
    split, and scale policy, so rejecting a non-generalizing winner adds no
    Bloch calculation and restores the actual baseline arrays.
    """

    if not ranked_trials:
        raise RuntimeError("held-out promotion gate received no nuisance trials")
    if str(selection_metric) != "work_wR2":
        raise RuntimeError(
            "held-out promotion gate requires work_wR2 ranking; refusing a selection that used free or all rows"
        )
    thresholds = {
        "min_gain": float(min_gain),
        "max_free_wr2_loss": float(max_free_wr2_loss),
        "max_all_r1_loss": float(max_all_r1_loss),
    }
    invalid_thresholds = [
        name for name, value in thresholds.items() if not math.isfinite(value) or value < 0.0
    ]
    if invalid_thresholds:
        raise RuntimeError(
            "held-out promotion thresholds must be finite and nonnegative: "
            + ", ".join(invalid_thresholds)
        )
    matches = [
        trial
        for trial in ranked_trials
        if nuisance_reference_matches(baseline_nuisance, trial.get("nuisance") or {})
    ]
    if len(matches) != 1:
        raise RuntimeError(
            "held-out promotion gate could not identify exactly one starting-nuisance trial "
            f"(matches={len(matches)}); refusing to promote a work-only winner"
        )

    candidate = ranked_trials[0]
    baseline = matches[0]
    nonwork_scale_trials = [
        label
        for label, trial in (("baseline", baseline), ("candidate", candidate))
        if str(trial.get("scale_fit_split", "")) != "work"
    ]
    if nonwork_scale_trials:
        raise RuntimeError(
            "held-out promotion gate requires work-only scale fitting for "
            + ", ".join(nonwork_scale_trials)
        )
    candidate_is_baseline = candidate is baseline
    split_names = ("all", "work", "free")
    baseline_counts = {split: _trial_split_count(baseline, split) for split in split_names}
    candidate_counts = {split: _trial_split_count(candidate, split) for split in split_names}
    baseline_value = trial_metric(baseline, selection_metric)
    candidate_value = trial_metric(candidate, selection_metric)
    baseline_free = _trial_metric_value(baseline, "free", "wR2_all")
    candidate_free = _trial_metric_value(candidate, "free", "wR2_all")
    baseline_all_r1 = _trial_metric_value(baseline, "all", "R1_all")
    candidate_all_r1 = _trial_metric_value(candidate, "all", "R1_all")
    baseline_all_wr2 = _trial_metric_value(baseline, "all", "wR2_all")
    candidate_all_wr2 = _trial_metric_value(candidate, "all", "wR2_all")
    baseline_full_row_score = _trial_full_row_score(baseline)
    candidate_full_row_score = _trial_full_row_score(candidate)
    baseline_valid_mask = _trial_valid_mask(baseline)
    candidate_valid_mask = _trial_valid_mask(candidate)
    gain = baseline_value - candidate_value
    free_wr2_loss = candidate_free - baseline_free
    all_r1_loss = candidate_all_r1 - baseline_all_r1
    all_wr2_loss = candidate_all_wr2 - baseline_all_wr2
    full_row_score_gain = baseline_full_row_score - candidate_full_row_score

    reasons: list[str] = []
    for split in split_names:
        if baseline_counts[split] <= 0:
            reasons.append(f"baseline_missing_positive_{split}_rows")
        if candidate_counts[split] <= 0:
            reasons.append(f"candidate_missing_positive_{split}_rows")
        if baseline_counts[split] != candidate_counts[split]:
            reasons.append(f"{split}_row_count_mismatch")
    finite_values = (
        baseline_value,
        candidate_value,
        baseline_free,
        candidate_free,
        baseline_all_r1,
        candidate_all_r1,
        baseline_all_wr2,
        candidate_all_wr2,
        baseline_full_row_score,
        candidate_full_row_score,
    )
    valid_row_mismatch_count: int | None = None
    if baseline_valid_mask is None:
        reasons.append("baseline_missing_valid_row_mask")
    if candidate_valid_mask is None:
        reasons.append("candidate_missing_valid_row_mask")
    if baseline_valid_mask is not None and candidate_valid_mask is not None:
        if baseline_valid_mask.shape != candidate_valid_mask.shape:
            reasons.append("valid_row_mask_shape_mismatch")
        else:
            valid_row_mismatch_count = int(np.count_nonzero(baseline_valid_mask != candidate_valid_mask))
            if valid_row_mismatch_count:
                reasons.append("valid_row_identity_mismatch")
    if not all(math.isfinite(value) for value in finite_values):
        reasons.append("missing_finite_promotion_metrics")
    elif not candidate_is_baseline:
        if gain < thresholds["min_gain"]:
            reasons.append("gain_below_minimum")
        if full_row_score_gain < thresholds["min_gain"]:
            reasons.append("full_row_score_gain_below_minimum")
        if free_wr2_loss > thresholds["max_free_wr2_loss"]:
            reasons.append("free_wR2_loss")
        if all_r1_loss > thresholds["max_all_r1_loss"]:
            reasons.append("all_R1_loss")

    promoted = bool(not candidate_is_baseline and not reasons)
    selected = candidate if promoted else baseline
    status = "accepted" if promoted else ("baseline_already_best" if candidate_is_baseline else "rejected")
    decision = {
        "family": "discrete_thickness_polish",
        "status": status,
        "promoted": promoted,
        "candidate_was_baseline": candidate_is_baseline,
        "selection_metric": str(selection_metric),
        "objective_gain": gain,
        "free_wR2_loss": free_wr2_loss,
        "all_R1_loss": all_r1_loss,
        "all_wR2_loss": all_wr2_loss,
        "baseline_full_row_score": baseline_full_row_score,
        "candidate_full_row_score": candidate_full_row_score,
        "full_row_score_gain": full_row_score_gain,
        "min_promotion_gain": thresholds["min_gain"],
        "max_free_wR2_loss": thresholds["max_free_wr2_loss"],
        "max_all_R1_loss": thresholds["max_all_r1_loss"],
        "baseline_row_counts": baseline_counts,
        "candidate_row_counts": candidate_counts,
        "baseline_valid_row_mask_sha256": _valid_mask_sha256(baseline_valid_mask),
        "candidate_valid_row_mask_sha256": _valid_mask_sha256(candidate_valid_mask),
        "valid_row_mismatch_count": valid_row_mismatch_count,
        "promotion_reasons": list(dict.fromkeys(reasons)),
        "ranked_candidate": _promotion_trial_record(candidate, selection_metric),
        "baseline": _promotion_trial_record(baseline, selection_metric),
        "selected": _promotion_trial_record(selected, selection_metric),
    }
    return selected, decision


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
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--baseline-hkl", default=str(DEFAULT_BASELINE_HKL))
    ap.add_argument("--hkl-transform", default="identity", help="Signed axis permutation mapping XDS HKLs into the model basis, e.g. 'l,-k,h'.")
    ap.add_argument(
        "--row-contract",
        default="",
        help=(
            "Immutable GROND native_dyn_calc.csv row-membership contract. When supplied, select exactly its "
            "row_index set after verifying HKL, target_I, xds_sigma, and d_A against the XDS input; do not "
            "rebuild membership from raw eligibility filters."
        ),
    )
    ap.add_argument(
        "--split-contract",
        default="",
        help=(
            "Immutable candidate-union raw-row work/free split shared by every "
            "transform candidate and subsequent hand comparison."
        ),
    )
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"], default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50)
    ap.add_argument("--sg-jitter-mode-grid", default="none", help="Comma grid: none, gaussian, student_t.")
    ap.add_argument("--sg-jitter-sigma0-grid", default="0.0", help="Base target-Sg jitter sigma grid in inverse Angstrom.")
    ap.add_argument("--sg-jitter-sigma-d-inv-grid", default="0.0", help="Target-Sg jitter sigma slope grid versus 1/d.")
    ap.add_argument("--sg-jitter-sigma-frame-grid", default="0.0", help="Target-Sg jitter sigma slope grid versus normalized frame number.")
    ap.add_argument("--sg-jitter-points", type=int, default=5, help="Kernel quadrature points for jitter convolution.")
    ap.add_argument("--sg-jitter-student-df", type=float, default=4.0)
    ap.add_argument("--sg-jitter-width-factor", type=float, default=3.0)
    ap.add_argument("--mosaicity-mode", choices=["none", "gaussian_omega"], default="none", help="Fixed direct-angle projected mosaic model for this run.")
    ap.add_argument("--mosaicity-sigma-deg", type=float, default=0.0, help="Projected Gaussian mosaicity sigma in degrees; one nuisance parameter when active.")
    ap.add_argument("--mosaicity-max-sigma-deg", type=float, default=0.5, help="Hard upper bound for projected mosaicity sigma.")
    ap.add_argument("--mosaicity-points", type=int, default=3, help="Gauss-Hermite points for direct-angle mosaic quadrature.")
    ap.add_argument("--geometry-path", default="", help="Fixed geometry-only slew/two-axis tilt contract fitted without intensities.")
    ap.add_argument("--profile-coordinate", choices=["target_sg", "direct_angle"], default="target_sg", help="Integrate through solved target-Sg points or directly in recorded scan angle.")
    ap.add_argument("--profile-angle-max-half-width-deg", type=float, default=2.0, help="Hard half-width bound for direct-angle integration.")
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.")
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=0.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0, help="Keep this broad geometry percentile after PEAK/d-spacing filters; 99 removes only the worst 1 percent geometry tail.")
    ap.add_argument("--target-mode", choices=list(TARGET_MODES), default="iobs")
    ap.add_argument("--positive-only", action="store_true", default=True)
    ap.add_argument("--include-nonpositive", dest="positive_only", action="store_false")
    ap.add_argument("--thickness-grid", default="1200,1600,1800,2200")
    ap.add_argument("--thickness-model-grid", default="uniform_gradient", help="Comma grid: single, uniform_gradient, gaussian, gaussian_gradient.")
    ap.add_argument("--thickness-grad-grid", default="0.0,0.2,0.4,0.6")
    ap.add_argument("--thickness-sigma-frac-grid", default="0.0", help="Gaussian thickness sigma/t0 grid. Used only by thickness_model=gaussian.")
    ap.add_argument(
        "--frame-offset-grid",
        default="auto",
        help=(
            "Comma grid in XDS frames, or 'auto'. Auto measures the global-header/ZCAL "
            "scan-datum mismatch without using intensities and tests a compact symmetric residual grid."
        ),
    )
    ap.add_argument("--absorption-mode-grid", default="none", help="Comma grid: none, mean, beam_dependent, offdiag_fraction, offdiag_tds, midgley_like.")
    ap.add_argument("--absorption-mu-grid", default="0.0", help="Uniform imaginary diagonal damping grid in inverse Angstrom.")
    ap.add_argument("--absorption-frac-grid", default="0.0", help="Imaginary absorptive off-diagonal fraction grid.")
    ap.add_argument("--absorption-mu-g2-grid", default="0.0", help="Beam-dependent diagonal damping slope grid versus |g|^2.")
    ap.add_argument("--absorption-mu-theta2-grid", default="0.0", help="Beam-dependent diagonal damping slope grid versus Bragg-angle^2.")
    ap.add_argument("--absorption-tds-b-grid", default="0.0", help="B_TDS grid for offdiag_tds eta*U_g*(1-exp(-B_TDS*|g|^2/4)).")
    ap.add_argument("--dose-b-mode-grid", default="none", help="Comma grid: none, linear_frame.")
    ap.add_argument("--dose-b-per-frame-grid", default="0.0", help="Extra isotropic B per XDS frame for dose_b_mode=linear_frame.")
    ap.add_argument("--dose-b-ref-frame", type=float, default=float("nan"), help="Reference ZCAL frame for dose-dependent B. Default uses the first frame in the data.")
    ap.add_argument("--row-resolution-b-grid", default="0.0", help="Raw-row dynamical resolution-envelope grid. Positive values boost high-resolution Idyn and are roughly the alternate sign of endpoint-B.")
    ap.add_argument(
        "--matrix-convention",
        choices=["j_minus_i", "i_minus_j"],
        default="j_minus_i",
        help="Off-diagonal U_g indexing convention. j_minus_i is the production signed-row default; i_minus_j is a diagnostic.",
    )
    ap.add_argument(
        "--q-control",
        choices=["real", "shuffle_global", "shuffle_shell", "shuffle_shell_scan"],
        default="real",
        help="Diagnostic null that permutes log(Idyn/Iborn) before normal scale fitting and endpoint export.",
    )
    ap.add_argument("--q-control-seed", type=int, default=12345)
    ap.add_argument("--q-control-shells", type=int, default=10, help="Resolution shells used by shell and shell+scan q-shuffle controls.")
    ap.add_argument(
        "--group-width-grid",
        default="auto",
        help="Comma grid of physical frame-scale group widths. Values are degrees; also accepts frame, 2frame, 3frames, and auto.",
    )
    ap.add_argument("--detector-scale-bins", default="0", help="Comma grid for optional NxN detector-position scale correction fitted with frame scales. Use 0 to disable.")
    ap.add_argument("--sigma-a-grid", default="0.0,0.03,0.06")
    ap.add_argument("--sigma-b-corr-grid", default="0.0")
    ap.add_argument("--sigma-c-peak-grid", default="0.0")
    ap.add_argument("--wght-a-grid", default="", help="Native SHELX-style weight a grid. Empty uses the WGHT value parsed from the RES.")
    ap.add_argument("--wght-b-grid", default="", help="Native SHELX-style weight b grid. Empty uses the WGHT value parsed from the RES.")
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--scale-fit-split", choices=["work", "all"], default="work", help="Rows used to fit nuisance group scales. Use 'all' to reproduce final all-data model-update statistics; use 'work' for validation.")
    ap.add_argument(
        "--selection-metric",
        choices=["work_born_wR2", "work_wR2", "free_wR2", "all_wR2", "all_R1"],
        default="work_wR2",
        help=(
            "Metric used to choose the reported best nuisance trial. "
            "work_born_wR2 is reserved for hand-invariant transform scouting."
        ),
    )
    ap.add_argument(
        "--promotion-baseline-json",
        default="",
        help="Internal automatic-workflow contract: exact starting nuisance used for a post-ranking held-out no-harm gate.",
    )
    ap.add_argument("--promotion-min-gain", type=float, default=5e-5)
    ap.add_argument("--promotion-max-free-wr2-loss", type=float, default=0.002)
    ap.add_argument("--promotion-max-all-r1-loss", type=float, default=0.001)
    ap.add_argument("--fail-if-born-r1gt-above", type=float, default=0.0, help="Fail after the prediction grid if matched Born R1gt is above this value. Use as a basis/indexing sanity gate.")
    ap.add_argument(
        "--lambda-grid",
        default="1.0",
        help="Diagnostic q-exponent grid. Production workflows use the full Bloch/Born correction with lambda_q=1.0.",
    )
    ap.add_argument(
        "--scale-smoothness",
        type=float,
        default=0.0,
        help="Experimental L2 frame-scale smoothness coefficient. Default 0 keeps per-group scales independent; use post-hoc scale compression for publication parameter reduction.",
    )
    ap.add_argument("--endpoint-b-grid", default="0.0", help="Diagnostic endpoint resolution-scale grid in intensity B units: Icorr *= exp(B*(1/4d^2 - median(1/4d^2))). Nonzero values are not promoted over a neutral B=0 endpoint.")
    ap.add_argument("--endpoint-exti-grid", default="off", help="Comma grid of endpoint EXTI modes: off,on. Off removes EXTI from the corrected endpoint; on keeps/regularizes the source EXTI term.")
    ap.add_argument("--shelxl", default=SHELXL, help="Explicit SHELXL executable used for corrected-HKL endpoints.")
    ap.add_argument("--shelxl-iterations", type=int, default=1, help="Number of SHELXL polish passes per corrected endpoint. Later passes feed .res back into .ins.")
    ap.add_argument("--endpoint-ls-cycles", type=int, default=1, help="Internal L.S./CGLS cycle count written into generated endpoint .ins/.res files. Production uses 1; larger values are diagnostic stress tests.")
    ap.add_argument(
        "--endpoint-wght-mode",
        choices=["freeze-zero", "inherit", "promote"],
        default="freeze-zero",
        help="SHELXL WGHT policy for corrected-HKL endpoints. freeze-zero is the production default after GROND sigma calibration; inherit keeps the source-model WGHT; promote uses SHELXL recommendations on later external passes.",
    )
    ap.add_argument("--skip-endpoint-refinement", action="store_true", help="Run the fixed-model Bloch row calculation and row reports only. No corrected-HKL SHELXL endpoint is written or refined.")
    ap.add_argument("--post-bloch-sigma-policy", choices=["none", "uniform_z_rms"], default="none", help="Endpoint uncertainty policy applied after Bloch correction and before SHELXL. uniform_z_rms fits one residual RMS(z) scalar from the current model and scales endpoint sigmas only.")
    ap.add_argument("--post-bloch-sigma-holdout-fraction", type=float, default=0.20, help="Deterministic HKL holdout fraction reported for uniform_z_rms diagnostics.")
    ap.add_argument("--post-bloch-sigma-min-scale", type=float, default=0.25, help="Lower guardrail for the uniform_z_rms sigma multiplier.")
    ap.add_argument("--post-bloch-sigma-max-scale", type=float, default=4.0, help="Upper guardrail for the uniform_z_rms sigma multiplier.")
    ap.add_argument("--rescale", choices=["median-iobs", "none"], default="median-iobs")
    ap.add_argument("--hkl-float-format", choices=["shelx_fixed", "wide"], default="shelx_fixed")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--chunksize", type=int, default=8)
    ap.add_argument("--prediction-cache-dir", default=str(runtime_cache_dir("native_row_regularize", "predictions")))
    ap.add_argument("--cache-compression", choices=["compressed", "none"], default="compressed", help="Use none for faster local SSD caches; compressed saves disk on slow/limited storage.")
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda", "jax_padded", "jax_auto_padded", "jax_cpu_padded", "jax_mps_padded", "jax_cuda_padded", "jax_cpu_real_expm", "jax_mps_real_expm", "jax_mps_real_expm_padded"], default="scipy", help="Prediction backend. SciPy eigensolver is the production reference; scipy_bethe uses an opt-in promoted Bethe/Schur strong-beam reduction; SciPy expm/expm_multiply, real-embedded JAX expm, and JAX eigensolver backends are experimental agreement/speed probes. *_padded JAX backends pad matrix dimensions into small buckets to reduce compilation churn.")
    ap.add_argument("--row-plan-cache", choices=["off", "readwrite", "rebuild"], default="off", help="Experimental frozen row-geometry cache. It reuses basis/Sg/difference-HKL plans across model updates while recomputing current U_g values.")
    ap.add_argument("--jax-batch-size", type=int, default=0, help="JAX matrix batch size. 0 chooses a conservative size from matrix dimension and device count.")
    ap.add_argument("--jax-matrix-bucket-size", type=int, default=0, help="Pad JAX matrix dimensions to this multiple. 0 keeps exact sizes unless a *_padded backend is selected, where 0 means 8.")
    ap.add_argument("--bethe-strong-count", type=int, default=48, help="Strong-beam block size for scipy_bethe. Values >= n_basis recover the dense SciPy matrix.")
    ap.add_argument("--bethe-variant", choices=["truncate", "schur0", "schur_target", "schur_mean"], default="schur_target", help="Strong/weak-beam reduction used by scipy_bethe.")
    ap.add_argument("--bethe-promote-denom", type=float, default=0.005, help="Promote weak beams whose diagonal denominator is this close to the Schur energy.")
    ap.add_argument("--bethe-denom-eps", type=float, default=1e-5, help="Small-denominator floor for scipy_bethe Schur corrections.")
    ap.add_argument("--max-rows", type=int, default=0)
    ap.add_argument("--make-maps", action="store_true")
    ap.add_argument("--alternate-hand", dest="alternate_hand", action="store_true", help="Use alternate fractional xyz before building U_g for an alternate-hand control run.")
    ap.add_argument(
        "--paired-inverse-hand",
        action="store_true",
        help=(
            "Also calculate the elastic inverse hand by negative-thickness propagation from each "
            "already diagonalized Bloch matrix. Requires absorption off and the SciPy backend."
        ),
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
    if str(frame_datum.get("status")) == "ok":
        print(
            "XDS frame datum: median="
            f"{float(frame_datum['median_offset_frames']):.4f} frames; "
            f"resolved grid={frame_offset_values}",
            flush=True,
        )
    for thickness in parse_float_grid(args.thickness_grid):
        for thickness_model in thickness_models:
            if thickness_model not in {"single", "uniform_gradient", "gaussian", "gaussian_gradient"}:
                raise ValueError(f"unknown thickness_model {thickness_model!r}")
            grad_values = parse_float_grid(args.thickness_grad_grid) if thickness_model in {"uniform_gradient", "gaussian_gradient"} else [0.0]
            sigma_values = parse_float_grid(args.thickness_sigma_frac_grid) if thickness_model in {"gaussian", "gaussian_gradient"} else [0.0]
            for grad in grad_values:
                for thickness_sigma_frac in sigma_values:
                    for frame_offset in frame_offset_values:
                        for sg_jitter_mode in sg_jitter_modes:
                            if sg_jitter_mode not in {"none", "gaussian", "student_t"}:
                                raise ValueError(f"unknown sg_jitter_mode {sg_jitter_mode!r}")
                            jitter_sigma0_values = parse_float_grid(args.sg_jitter_sigma0_grid) if sg_jitter_mode != "none" else [0.0]
                            jitter_d_inv_values = parse_float_grid(args.sg_jitter_sigma_d_inv_grid) if sg_jitter_mode != "none" else [0.0]
                            jitter_frame_values = parse_float_grid(args.sg_jitter_sigma_frame_grid) if sg_jitter_mode != "none" else [0.0]
                            for sg_jitter_sigma0 in jitter_sigma0_values:
                                for sg_jitter_sigma_d_inv in jitter_d_inv_values:
                                    for sg_jitter_sigma_frame in jitter_frame_values:
                                        for absorption_mode in absorption_modes:
                                            for abs_par in absorption_parameter_grid(args, absorption_mode):
                                                for dose_b_mode in dose_modes:
                                                    if dose_b_mode not in {"none", "linear_frame"}:
                                                        raise ValueError(f"unknown dose_b_mode {dose_b_mode!r}")
                                                    dose_b_values = parse_float_grid(args.dose_b_per_frame_grid) if dose_b_mode == "linear_frame" else [0.0]
                                                    for dose_b_per_frame in dose_b_values:
                                                        cfg = Config(
                                                            n_basis=args.n_basis,
                                                            n_points=args.n_points,
                                                            sg_window=args.sg_window,
                                                            sg_window_mode=args.sg_window_mode,
                                                            sg_window_min=args.sg_window_min,
                                                            sg_window_max=args.sg_window_max,
                                                            sg_window_scale=args.sg_window_scale,
                                                            sg_window_frame_floor=args.sg_window_frame_floor,
                                                            sg_jitter_mode=sg_jitter_mode,
                                                            sg_jitter_sigma0=sg_jitter_sigma0,
                                                            sg_jitter_sigma_d_inv=sg_jitter_sigma_d_inv,
                                                            sg_jitter_sigma_frame=sg_jitter_sigma_frame,
                                                            sg_jitter_points=args.sg_jitter_points,
                                                            sg_jitter_student_df=args.sg_jitter_student_df,
                                                            sg_jitter_width_factor=args.sg_jitter_width_factor,
                                                            mosaicity_mode=args.mosaicity_mode,
                                                            mosaicity_sigma_deg=args.mosaicity_sigma_deg,
                                                            mosaicity_max_sigma_deg=args.mosaicity_max_sigma_deg,
                                                            mosaicity_points=args.mosaicity_points,
                                                            geometry_path_file=args.geometry_path,
                                                            profile_coordinate=args.profile_coordinate,
                                                            profile_angle_max_half_width_deg=args.profile_angle_max_half_width_deg,
                                                            profile_quadrature=args.profile_quadrature,
                                                            profile_accumulation=args.profile_accumulation,
                                                            topup_d_min=effective_topup_d_min,
                                                            max_support_sg=args.max_support_sg,
                                                            thickness_A=thickness,
                                                            thickness_model=thickness_model,
                                                            thickness_grad_frac=grad,
                                                            thickness_sigma_frac=thickness_sigma_frac,
                                                            thickness_points=args.thickness_points,
                                                            group_width_deg=group_width_values[0],
                                                            frame_offset=frame_offset,
                                                            dose_b_mode=dose_b_mode,
                                                            dose_b_per_frame=dose_b_per_frame,
                                                            dose_b_ref_frame=args.dose_b_ref_frame,
                                                            compute_backend=args.compute_backend,
                                                            jax_batch_size=args.jax_batch_size,
                                                            jax_matrix_bucket_size=args.jax_matrix_bucket_size,
                                                            bethe_strong_count=args.bethe_strong_count,
                                                            bethe_variant=args.bethe_variant,
                                                            bethe_promote_denom=args.bethe_promote_denom,
                                                            bethe_denom_eps=args.bethe_denom_eps,
                                                            matrix_convention=args.matrix_convention,
                                                            row_plan_cache=args.row_plan_cache,
                                                            paired_inverse_hand=args.paired_inverse_hand,
                                                            free_fraction=args.free_fraction,
                                                            split_seed=args.split_seed,
                                                            **abs_par,
                                                        )
                                                        print(
                                                            f"predict t={thickness:g} thickness_model={thickness_model} grad={grad:g} "
                                                            f"sigma_frac={thickness_sigma_frac:g} frame_offset={frame_offset:g} "
                                                            f"sg_jitter={sg_jitter_mode} sigma0={sg_jitter_sigma0:g} "
                                                            f"sigma_d_inv={sg_jitter_sigma_d_inv:g} sigma_frame={sg_jitter_sigma_frame:g} "
                                                            f"abs={abs_par['absorption_mode']} mu={abs_par['absorption_mu']:g} "
                                                            f"frac={abs_par['absorption_frac']:g} mu_g2={abs_par['absorption_mu_g2']:g} "
                                                            f"mu_theta2={abs_par['absorption_mu_theta2']:g} tds_b={abs_par['absorption_tds_b']:g} "
                                                            f"dose_b={dose_b_mode} per_frame={dose_b_per_frame:g}",
                                                            flush=True,
                                                        )
                                                        try:
                                                            pred = predict_with_cache(
                                                                data,
                                                                model,
                                                                res_hash,
                                                                Path(args.hkl),
                                                                Path(args.res),
                                                                cfg,
                                                                indices,
                                                                Path(args.prediction_cache_dir),
                                                                args.workers,
                                                                args.chunksize,
                                                                model_tag=model_tag,
                                                                cache_compression=args.cache_compression,
                                                            )
                                                        except InfeasibleSgJitterProfileError as exc:
                                                            rejection = sg_jitter_candidate_rejection(cfg, exc)
                                                            rejected_prediction_candidates.append(rejection)
                                                            print(
                                                                "rejected nuisance candidate: "
                                                                f"{rejection['reason']}: {rejection['message']}",
                                                                flush=True,
                                                            )
                                                            continue
                                                        geom_trials.append(
                                                            {
                                                                "cfg": cfg,
                                                                "pred": pred,
                                                                "thickness": thickness,
                                                                "thickness_model": thickness_model,
                                                                "grad": grad,
                                                                "thickness_sigma_frac": thickness_sigma_frac,
                                                                "frame_offset": frame_offset,
                                                                "sg_jitter_mode": sg_jitter_mode,
                                                                "sg_jitter_sigma0": sg_jitter_sigma0,
                                                                "sg_jitter_sigma_d_inv": sg_jitter_sigma_d_inv,
                                                                "sg_jitter_sigma_frame": sg_jitter_sigma_frame,
                                                                "dose_b_mode": dose_b_mode,
                                                                "dose_b_per_frame": dose_b_per_frame,
                                                                **abs_par,
                                                            }
                                                        )
                                                        sigma_a_values = parse_float_grid(args.sigma_a_grid)
                                                        sigma_b_values = parse_float_grid(args.sigma_b_corr_grid)
                                                        sigma_c_values = parse_float_grid(args.sigma_c_peak_grid)
                                                        row_b_values = parse_float_grid(args.row_resolution_b_grid)
                                                        for grid_values in itertools.product(
                                                            group_width_values,
                                                            detector_scale_bins_values,
                                                            sigma_a_values,
                                                            sigma_b_values,
                                                            sigma_c_values,
                                                            row_b_values,
                                                            wght_a_values,
                                                            wght_b_values,
                                                        ):
                                                            group_width, detector_scale_bins, sig_a, sig_b, sig_c, row_resolution_b, native_wght_a, native_wght_b = grid_values
                                                            nui = Nuisance(
                                                                thickness,
                                                                grad,
                                                                frame_offset,
                                                                group_width,
                                                                detector_scale_bins,
                                                                sig_a,
                                                                sig_b,
                                                                sig_c,
                                                                thickness_model=thickness_model,
                                                                thickness_sigma_frac=thickness_sigma_frac,
                                                                sg_jitter_mode=sg_jitter_mode,
                                                                sg_jitter_sigma0=sg_jitter_sigma0,
                                                                sg_jitter_sigma_d_inv=sg_jitter_sigma_d_inv,
                                                                sg_jitter_sigma_frame=sg_jitter_sigma_frame,
                                                                sg_jitter_points=args.sg_jitter_points,
                                                                sg_jitter_student_df=args.sg_jitter_student_df,
                                                                sg_jitter_width_factor=args.sg_jitter_width_factor,
                                                                mosaicity_mode=args.mosaicity_mode,
                                                                mosaicity_sigma_deg=args.mosaicity_sigma_deg,
                                                                mosaicity_points=args.mosaicity_points,
                                                                geometry_path_file=args.geometry_path,
                                                                dose_b_mode=dose_b_mode,
                                                                dose_b_per_frame=dose_b_per_frame,
                                                                dose_b_ref_frame=args.dose_b_ref_frame,
                                                                matrix_convention=args.matrix_convention,
                                                                row_resolution_b=row_resolution_b,
                                                                native_wght_a=native_wght_a,
                                                                native_wght_b=native_wght_b,
                                                                scale_smoothness=args.scale_smoothness,
                                                                **abs_par,
                                                            )
                                                            ev = evaluate_trial(data, target, base_sig, pred, indices, masks, nui, wght_a, wght_b, scale_fit_split=args.scale_fit_split, q_control_args=args)
                                                            ev["prediction_cache"] = {"status": pred.get("cache_status"), "path": pred.get("cache_path")}
                                                            all_trials.append(ev)

    if rejected_prediction_candidates:
        (out / "rejected_prediction_candidates.json").write_text(
            json.dumps(rejected_prediction_candidates, indent=2, allow_nan=True) + "\n"
        )
    if not all_trials:
        failure = {
            "reason": "no_feasible_prediction_candidates",
            "message": "Every requested nuisance prediction candidate was rejected by the strict local-geometry contract.",
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
    write_row_table(native_row_table, data, target, best, masks)
    native_row_artifact = native_row_table_artifact(native_row_table)
    paired_inverse_hand = {}
    if args.paired_inverse_hand:
        inverse_raw = np.asarray(best["arrays"]["dyn_inverse_raw"], dtype=float)
        paired_indices = np.where(
            np.isfinite(inverse_raw)
            & np.isfinite(np.asarray(best["arrays"]["born_raw"], dtype=float))
        )[0]
        if len(paired_indices) == 0:
            raise RuntimeError(
                "paired inverse-hand calculation produced no finite inverse intensities"
            )
        inverse_pred = {
            "indices": paired_indices,
            "dyn": inverse_raw[paired_indices],
            "born": np.asarray(best["arrays"]["born_raw"], dtype=float)[paired_indices],
        }
        for key in (
            "basis_size",
            "center_sg",
            "omega_bragg",
            "omega_delta",
            "sg_window",
            "sg_window_raw",
            "sg_window_half_frames",
            "sg_window_clipped",
            "sg_dSg_domega",
            "sg_dSg_dframe",
            "sg_jitter_sigma",
            "sg_jitter_d_inv",
            "sg_jitter_frame_fraction",
            "sg_jitter_kernel_points",
            "sg_profile_points_total",
            "sg_profile_weight_sum",
            "sg_profile_same_branch_continuation_used",
            "sg_profile_same_branch_continuation_segments",
            "sg_residual",
        ):
            values = best["arrays"].get(key)
            if values is not None:
                inverse_pred[key] = np.asarray(values)[paired_indices]
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
        write_row_table(inverse_row_table, data, target, inverse_eval, masks)
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
                float(ep.get("hkl_safe_factor", 1.0)) >= minimum_hkl_safe_factor
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
                or float(best_r1_ep.get("hkl_safe_factor", 1.0)) < minimum_hkl_safe_factor
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
    valid_continuation_mask = np.asarray(best["arrays"]["valid"], dtype=bool) & continuation_used
    profile_root_solver_audit = {
        "contract": "strict_bragg_same_branch_base_profile_v1",
        "bragg_crossing": "strict_local_bound",
        "base_profile": "exact_target_strict_step_continuation_on_one_derivative_sign_certified_branch",
        "active_jitter": "strict_local_bound_or_reject_candidate",
        "prediction_cache_schema": PREDICTION_CACHE_SCHEMA,
        "predicted_rows": int(np.count_nonzero(predicted_mask)),
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
                geometry_path_file=best["nuisance"].get("geometry_path_file", ""),
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
                row_plan_cache=args.row_plan_cache,
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
