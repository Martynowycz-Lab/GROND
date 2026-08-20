"""Export DIALS or XDS observations to the PETS ``cif_pets`` contract.

The output is accepted by PETS/Jana2020/Dyngo and is also the experimental
data format consumed by diffBloch.  DIALS owns the geometry conversion.  For
XDS, GROND reconstructs the recorded summation observation from the documented
INTEGRATE.HKL relation before passing it through the same DIALS PETS writer.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import gemmi


@dataclass(frozen=True)
class PetsExportResult:
    """Materialized PETS export and its machine-readable provenance."""

    out_dir: Path
    files: tuple[Path, ...]
    manifest: Path
    readme: Path


def xds_recorded_observation(
    iobs: float,
    sigma: float,
    peak_percent: float,
    rlp: float,
) -> tuple[float, float]:
    """Undo XDS INTEGRATE's RLP/partiality correction for one observation.

    XDS documents ``IOBS = I_recorded * RLP / (PEAK/100)``.  This inverse is
    therefore exact for a summation-only INTEGRATE.HKL row.  The uncertainty is
    transformed by the same positive linear factor.
    """

    values = (iobs, sigma, peak_percent, rlp)
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("XDS observation terms must all be finite")
    if float(rlp) <= 0.0 or float(peak_percent) <= 0.0 or float(sigma) <= 0.0:
        raise ValueError("XDS RLP, PEAK, and SIGMA must be positive")
    factor = (float(peak_percent) / 100.0) / float(rlp)
    return float(iobs) * factor, float(sigma) * factor


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _required_float(block: gemmi.cif.Block, tag: str) -> float:
    value = block.find_value(tag)
    if value is None or not str(value).strip() or str(value).strip() in {"?", "."}:
        raise ValueError(f"PETS export is missing required value {tag}")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"PETS export has non-finite value {tag}={value!r}")
    return result


def _validate_and_repair_pets(path: Path) -> dict[str, Any]:
    """Validate PETS output and repair writer details rejected by consumers."""

    document = gemmi.cif.read_file(str(path))
    block = document.sole_block()
    cell_values = [
        _required_float(block, tag)
        for tag in (
            "_cell_length_a",
            "_cell_length_b",
            "_cell_length_c",
            "_cell_angle_alpha",
            "_cell_angle_beta",
            "_cell_angle_gamma",
        )
    ]
    if any(value <= 0.0 for value in cell_values[:3]):
        raise ValueError("PETS export has a non-positive unit-cell length")
    wavelength = _required_float(block, "_diffrn_radiation_wavelength")
    if wavelength <= 0.0:
        raise ValueError("PETS export has a non-positive wavelength")
    ub = [
        _required_float(block, f"_diffrn_orient_matrix_UB_{row}{column}")
        for row in range(1, 4)
        for column in range(1, 4)
    ]
    determinant = (
        ub[0] * (ub[4] * ub[8] - ub[5] * ub[7])
        - ub[1] * (ub[3] * ub[8] - ub[5] * ub[6])
        + ub[2] * (ub[3] * ub[7] - ub[4] * ub[6])
    )
    if not math.isfinite(determinant) or abs(determinant) < 1e-12:
        raise ValueError("PETS export has a singular UB matrix")

    h_values = list(block.find_values("_refln_index_h"))
    k_values = list(block.find_values("_refln_index_k"))
    l_values = list(block.find_values("_refln_index_l"))
    intensities = list(block.find_values("_refln_intensity_meas"))
    sigmas = list(block.find_values("_refln_intensity_sigma"))
    reflection_zone_ids = list(block.find_values("_refln_zone_axis_id"))
    row_count = len(h_values)
    if row_count == 0 or len({row_count, len(k_values), len(l_values), len(intensities), len(sigmas), len(reflection_zone_ids)}) != 1:
        raise ValueError("PETS reflection loop is empty or has inconsistent column lengths")
    zone_ids = {str(value) for value in block.find_values("_diffrn_zone_axis_id")}
    if not zone_ids:
        raise ValueError("PETS export has no zone-axis geometry loop")
    used_zone_ids = {str(value) for value in reflection_zone_ids}
    empty_zone_ids = zone_ids - used_zone_ids
    if empty_zone_ids:
        zone_column = block.find_loop("_diffrn_zone_axis_id")
        zone_loop = zone_column.get_loop()
        zone_tags = list(zone_loop.tags)
        zone_width = zone_loop.width()
        zone_id_column = zone_tags.index("_diffrn_zone_axis_id")
        zone_rows = [
            zone_loop.values[offset : offset + zone_width]
            for offset in range(0, len(zone_loop.values), zone_width)
        ]
        kept_zone_rows = [
            row for row in zone_rows if str(row[zone_id_column]) in used_zone_ids
        ]
        if not kept_zone_rows:
            raise ValueError("PETS export has no zone-axis geometry used by reflections")
        zone_loop.set_all_values(
            [
                [row[column] for row in kept_zone_rows]
                for column in range(zone_width)
            ]
        )
        document.write_file(str(path))
        zone_ids = used_zone_ids

    cell = gemmi.UnitCell(*cell_values)
    maximum_d_star = 0.0
    invalid_rows = 0
    for h_text, k_text, l_text, intensity_text, sigma_text, zone_id in zip(
        h_values,
        k_values,
        l_values,
        intensities,
        sigmas,
        reflection_zone_ids,
        strict=True,
    ):
        try:
            hkl = (int(h_text), int(k_text), int(l_text))
            intensity = float(intensity_text)
            sigma = float(sigma_text)
            d_spacing = float(cell.calculate_d(hkl))
        except (TypeError, ValueError):
            invalid_rows += 1
            continue
        if (
            not math.isfinite(intensity)
            or not math.isfinite(sigma)
            or sigma <= 0.0
            or not math.isfinite(d_spacing)
            or d_spacing <= 0.0
            or str(zone_id) not in zone_ids
        ):
            invalid_rows += 1
            continue
        maximum_d_star = max(maximum_d_star, 1.0 / d_spacing)
    if invalid_rows:
        raise ValueError(f"PETS export contains {invalid_rows} invalid reflection rows")
    if maximum_d_star <= 0.0:
        raise ValueError("PETS export contains no finite reciprocal-space support")

    text = path.read_text()
    replacement = f"dstarmax:  {maximum_d_star:.6f}"
    repaired, count = re.subn(r"(?m)^dstarmax:\s*[^\r\n]*$", replacement, text)
    if count != 1:
        raise ValueError(f"expected exactly one dstarmax line in {path}; found {count}")
    path.write_text(repaired)
    # Parse the materialized result again; this catches malformed semicolon text
    # or accidental loop damage caused by post-processing.
    gemmi.cif.read_file(str(path))
    return {
        "rows": row_count,
        "zone_axes": len(zone_ids),
        "zone_axes_removed_empty": len(empty_zone_ids),
        "wavelength_angstrom": wavelength,
        "unit_cell": cell_values,
        "ub_determinant": determinant,
        "dstarmax_inverse_angstrom": maximum_d_star,
        "d_min_angstrom": 1.0 / maximum_d_star,
    }


def _resolve_dials_python(value: str) -> str:
    if value:
        candidate = Path(value).expanduser()
        if candidate.is_file():
            return str(candidate.resolve())
        resolved = shutil.which(value)
        if resolved:
            return resolved
        raise ValueError(f"DIALS Python executable was not found: {value}")
    resolved = shutil.which("dials.python")
    return resolved or sys.executable


def _source_hashes(config: dict[str, Any]) -> dict[str, str]:
    paths: list[Path] = []
    if config["source_kind"] == "DIALS":
        paths.extend((Path(config["refl"]), Path(config["expt"])))
    else:
        xds_dir = Path(config["xds_dir"])
        paths.extend(path for path in (xds_dir / "XDS.INP", xds_dir / "INTEGRATE.HKL", xds_dir / "XDS_ASCII.HKL") if path.is_file())
    return {str(path.resolve()): _sha256(path) for path in paths}


def _write_readme(path: Path, manifest_name: str, files: list[Path], source_kind: str) -> None:
    names = "\n".join(f"- `{file.name}`" for file in files)
    xds_note = ""
    if source_kind == "XDS":
        xds_note = (
            "\nThe XDS route is a geometry-preserving interoperability conversion, not a claim "
            "that XDS and DIALS integrations are numerically interchangeable. GROND inverted "
            "XDS's documented RLP/PEAK correction and retained the CORRECT-accepted signed rows.\n"
        )
    path.write_text(
        "# GROND PETS interoperability export\n\n"
        "These `cif_pets` files contain signed observations, uncertainties, unit cell, "
        "wavelength, UB matrix, and virtual-frame geometry. They can be imported into "
        "PETS/Jana2020/Dyngo and are the experimental-data input used by diffBloch.\n\n"
        f"Files:\n{names}\n\n"
        f"The exact source hashes, row convention, filters, and transformations are in `{manifest_name}`.\n"
        f"{xds_note}\n"
        "GROND does not merge Friedel mates or choose an absolute hand during export. "
        "The source indexing basis is preserved through the recorded transform ledger.\n"
    )


def export_pets(
    *,
    out_dir: str | Path,
    dataset_name: str,
    dials_python: str = "",
    dials_refl: str | Path | None = None,
    dials_expt: str | Path | None = None,
    xds_dir: str | Path | None = None,
    experiment_ids: list[int] | None = None,
    intensity: str = "sum",
    row_selection: str = "integrated",
    xds_row_selection: str = "correct",
    xds_sigma_mode: str = "reported",
    min_partiality: float = 0.90,
    excitation_error_cutoff: float = 0.04,
    frame_ownership: str = "overlap",
    n_merged: int = 1,
    step: int = 1,
    scan_varying_cell: str = "project",
    max_cell_length_fraction: float = 0.03,
    max_cell_angle_degrees: float = 1.0,
    keep_bad_flags: bool = False,
    allow_profile_fitted_xds: bool = False,
    overwrite: bool = False,
) -> PetsExportResult:
    """Create one or more audited ``cif_pets`` interoperability files."""

    has_dials = dials_refl is not None or dials_expt is not None
    has_xds = xds_dir is not None
    if has_dials == has_xds:
        raise ValueError("provide exactly one source: --dials-refl/--dials-expt or --xds-dir")
    if has_dials and (dials_refl is None or dials_expt is None):
        raise ValueError("DIALS export requires both --dials-refl and --dials-expt")
    if not 0.0 < float(min_partiality) <= 1.0:
        raise ValueError("--min-partiality must be in (0, 1]")
    if int(n_merged) < 1 or int(step) < 1:
        raise ValueError("virtual-frame n-merged and step must be positive integers")
    if frame_ownership not in {"overlap", "nearest-excitation"}:
        raise ValueError("--frame-ownership must be overlap or nearest-excitation")
    if xds_sigma_mode not in {"reported", "raw"}:
        raise ValueError("--xds-sigma-mode must be reported or raw")
    if float(max_cell_length_fraction) < 0.0 or float(max_cell_angle_degrees) < 0.0:
        raise ValueError("scan-varying cell deviation limits must be non-negative")
    if not str(dataset_name).strip() or not re.fullmatch(r"[A-Za-z0-9_.-]+", str(dataset_name)):
        raise ValueError("--dataset-name may contain only letters, numbers, dot, underscore, and hyphen")

    destination = Path(out_dir).expanduser().resolve()
    if destination.exists() and any(destination.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {destination}; pass --overwrite to replace named artifacts")
    destination.mkdir(parents=True, exist_ok=True)
    helper = Path(__file__).with_name("_dials_pets_helper.py")
    python = _resolve_dials_python(dials_python)
    config: dict[str, Any] = {
        "source_kind": "DIALS" if has_dials else "XDS",
        "out_dir": str(destination),
        "dataset_name": str(dataset_name),
        "intensity": str(intensity),
        "row_selection": str(row_selection),
        "xds_row_selection": str(xds_row_selection),
        "xds_sigma_mode": str(xds_sigma_mode),
        "experiment_ids": experiment_ids,
        "min_partiality": float(min_partiality),
        "excitation_error_cutoff": float(excitation_error_cutoff),
        "frame_ownership": str(frame_ownership),
        "n_merged": int(n_merged),
        "step": int(step),
        "scan_varying_cell": str(scan_varying_cell),
        "max_cell_length_fraction": float(max_cell_length_fraction),
        "max_cell_angle_degrees": float(max_cell_angle_degrees),
        "keep_bad_flags": bool(keep_bad_flags),
        "allow_profile_fitted_xds": bool(allow_profile_fitted_xds),
    }
    if has_dials:
        config.update(
            {
                "refl": str(Path(dials_refl).expanduser().resolve()),
                "expt": str(Path(dials_expt).expanduser().resolve()),
            }
        )
    else:
        config["xds_dir"] = str(Path(xds_dir).expanduser().resolve())
    for key in ("refl", "expt") if has_dials else ("xds_dir",):
        if not Path(config[key]).exists():
            raise FileNotFoundError(config[key])

    with tempfile.TemporaryDirectory(prefix="grond_pets_export_") as temporary:
        config_path = Path(temporary) / "config.json"
        result_path = Path(temporary) / "result.json"
        config_path.write_text(json.dumps(config, indent=2) + "\n")
        command = [python, str(helper), "--config", str(config_path), "--result", str(result_path)]
        environment = os.environ.copy()
        environment.setdefault("MPLCONFIGDIR", str(Path(temporary) / "matplotlib"))
        process = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
            env=environment,
        )
        if process.returncode != 0:
            message = process.stderr.strip() or process.stdout.strip() or "DIALS PETS helper failed"
            raise RuntimeError(f"PETS export failed with {python}:\n{message}")
        if not result_path.is_file():
            raise RuntimeError("PETS export helper completed without writing its result contract")
        helper_result = json.loads(result_path.read_text())

    files = [Path(row["path"]).resolve() for row in helper_result.get("outputs", [])]
    if not files or any(not path.is_file() for path in files):
        raise RuntimeError("PETS export helper did not materialize every declared output")
    validations = []
    for path in files:
        validation = _validate_and_repair_pets(path)
        validation.update({"path": str(path), "sha256": _sha256(path)})
        validations.append(validation)

    manifest = destination / "pets_export_manifest.json"
    readme = destination / "README.md"
    payload = {
        "schema": "grond.pets-export.v1",
        "status": "complete",
        "source_kind": config["source_kind"],
        "contract": "PETS cif_pets as written by the DIALS PETS exporter",
        "command_contract": config,
        "source_sha256": _source_hashes(config),
        "dials_python": python,
        "helper_sha256": _sha256(helper),
        "helper_result": helper_result,
        "outputs": validations,
        "important_limits": [
            "No Friedel merging or absolute-hand selection is performed by this export.",
            "cif_pets stores one declared cell; any scan-varying DIALS cell projection is explicit in helper_result.",
            "The XDS conversion is an approximate XDS-to-DIALS geometry bridge and must retain its transform ledger.",
            (
                "nearest-excitation frame ownership emits each selected source observation once; "
                "overlap is the stock PETS/Jana candidate-frame convention."
            ),
        ],
    }
    manifest.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    _write_readme(readme, manifest.name, files, config["source_kind"])
    return PetsExportResult(destination, tuple(files), manifest, readme)
