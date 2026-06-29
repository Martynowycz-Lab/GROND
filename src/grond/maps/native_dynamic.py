"""Native row-level dynamical Fourier-map export.

The maps here are built from GROND's native row table, not from SHELXL FCF
files.  Each observed row is converted to an electron structure-factor
amplitude using the row-level dynamical prediction and the current model phase:

    |F_dyn_obs,row| = |F_model| * sqrt(I_obs / I_dyn_scaled)

Rows with the same signed HKL are merged in amplitude space.  The resulting
Fo, Fo-Fc, and 2Fo-Fc coefficients are then Fourier transformed onto a CCP4
grid.  This follows the usual dynamical-refinement map convention of using a
model phase with a dynamically estimated observed electron amplitude.
"""

from __future__ import annotations

import csv
import json
import math
import re
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import gemmi
import numpy as np

from grond.legacy.engine.bloch_engine import fcalc_res_grid, inv_d2_from_hkl_array, parse_model

DEFAULT_ELECTRON_WAVELENGTH_A = 0.025079


@dataclass(frozen=True)
class NativeDynamicMapOptions:
    """Controls for row-level dynamical map synthesis."""

    grid_spacing_A: float = 0.25
    d_min_A: float | None = None
    min_i_over_sigma: float = 0.0
    max_grid_points: int = 8_000_000


def _read_rows(path: Path) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def _as_float(row: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    try:
        return float(row.get(key, default))
    except (TypeError, ValueError):
        return float(default)


def _as_int(row: Mapping[str, object], key: str, default: int = 0) -> int:
    try:
        return int(float(row.get(key, default)))
    except (TypeError, ValueError):
        return int(default)


def _required_row_columns(rows: Sequence[Mapping[str, str]]) -> list[str]:
    if not rows:
        return ["<non-empty row table>"]
    required = {"h", "k", "l", "target_I", "sigma_eff", "dyn_scaled"}
    return sorted(required.difference(rows[0].keys()))


def _model_wavelength_A(model_path: Path) -> float:
    """Extract a SHELX/CIF wavelength when possible, else use 200 keV default."""

    def normalized(value: float) -> float:
        if not math.isfinite(float(value)) or float(value) <= 0.0:
            return DEFAULT_ELECTRON_WAVELENGTH_A
        if abs(float(value) - 1.0) < 1e-6:
            return DEFAULT_ELECTRON_WAVELENGTH_A
        if float(value) >= 0.1:
            return DEFAULT_ELECTRON_WAVELENGTH_A
        return float(value)

    path = Path(model_path)
    text = path.read_text(errors="replace") if path.exists() else ""
    if path.suffix.lower() in {".res", ".ins"}:
        for line in text.splitlines():
            tokens = line.strip().split()
            if len(tokens) >= 2 and tokens[0].upper() == "CELL":
                try:
                    return normalized(float(tokens[1]))
                except ValueError:
                    return DEFAULT_ELECTRON_WAVELENGTH_A
    if path.suffix.lower() == ".cif":
        match = re.search(r"_diffrn_radiation_wavelength\\s+([^\\s]+)", text)
        if match:
            try:
                return normalized(float(match.group(1).strip("'\"")))
            except ValueError:
                return DEFAULT_ELECTRON_WAVELENGTH_A
    return DEFAULT_ELECTRON_WAVELENGTH_A


def _next_fastish_size(value: int) -> int:
    """Return a small-prime-friendly FFT grid size at least as large as value."""

    n = max(1, int(value))
    while True:
        m = n
        for p in (2, 3, 5, 7):
            while m % p == 0:
                m //= p
        if m == 1:
            return n
        n += 1


def _grid_shape(cell: Sequence[float], hkls: np.ndarray, spacing_A: float) -> tuple[int, int, int]:
    lengths = [float(cell[0]), float(cell[1]), float(cell[2])]
    max_abs = np.max(np.abs(hkls), axis=0) if len(hkls) else np.zeros(3, dtype=int)
    dims = []
    for length, hmax in zip(lengths, max_abs, strict=True):
        by_spacing = int(math.ceil(length / max(float(spacing_A), 0.05)))
        by_hkl = int(2 * int(hmax) + 3)
        dims.append(_next_fastish_size(max(by_spacing, by_hkl, 4)))
    return int(dims[0]), int(dims[1]), int(dims[2])


def _merge_native_amplitudes(
    rows: Sequence[Mapping[str, str]],
    model: Mapping[str, object],
    options: NativeDynamicMapOptions,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    missing = _required_row_columns(rows)
    if missing:
        raise ValueError(f"native dynamical map row table is missing required columns: {', '.join(missing)}")

    h = np.asarray([_as_int(row, "h") for row in rows], dtype=int)
    k = np.asarray([_as_int(row, "k") for row in rows], dtype=int)
    ell = np.asarray([_as_int(row, "l") for row in rows], dtype=int)
    target = np.asarray([_as_float(row, "target_I") for row in rows], dtype=float)
    sigma = np.asarray([_as_float(row, "sigma_eff") for row in rows], dtype=float)
    dyn_scaled = np.asarray([_as_float(row, "dyn_scaled") for row in rows], dtype=float)
    dyn_raw = np.asarray([_as_float(row, "dyn_raw") for row in rows], dtype=float)
    born_scaled = np.asarray([_as_float(row, "born_scaled") for row in rows], dtype=float)
    q_dyn_over_born = np.asarray([_as_float(row, "q_dyn_over_born") for row in rows], dtype=float)
    d_A = np.asarray([
        _as_float(row, "d_A", float("nan")) for row in rows
    ], dtype=float)
    if not np.any(np.isfinite(d_A)):
        inv_d2 = inv_d2_from_hkl_array(h, k, ell, model["cell"])
        with np.errstate(divide="ignore", invalid="ignore"):
            d_A = 1.0 / np.sqrt(inv_d2)

    fc = fcalc_res_grid(model, h, k, ell)
    fc_abs = np.abs(fc)
    min_isig = float(options.min_i_over_sigma)
    isig = target / np.maximum(sigma, 1e-30)
    valid = (
        np.isfinite(target)
        & np.isfinite(sigma)
        & np.isfinite(dyn_scaled)
        & np.isfinite(fc_abs)
        & np.isfinite(d_A)
        & (target > 0.0)
        & (sigma > 0.0)
        & (dyn_scaled > 0.0)
        & (fc_abs > 1e-12)
        & (isig >= min_isig)
    )
    if options.d_min_A is not None and float(options.d_min_A) > 0.0:
        valid &= d_A >= float(options.d_min_A)

    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in np.where(valid)[0]:
        grouped[(int(h[idx]), int(k[idx]), int(ell[idx]))].append(int(idx))

    merged: list[dict[str, object]] = []
    for hkl, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=int)
        fc_ref = complex(fcalc_res_grid(model, np.asarray([hkl[0]]), np.asarray([hkl[1]]), np.asarray([hkl[2]]))[0])
        fc_ref_abs = abs(fc_ref)
        if fc_ref_abs <= 1e-12:
            continue
        row_fobs = fc_ref_abs * np.sqrt(np.maximum(target[idx], 0.0) / np.maximum(dyn_scaled[idx], 1e-30))
        sigma_f = 0.5 * fc_ref_abs * sigma[idx] / np.sqrt(np.maximum(target[idx] * dyn_scaled[idx], 1e-30))
        weights = 1.0 / np.maximum(sigma_f * sigma_f, 1e-30)
        fobs = float(np.sum(weights * row_fobs) / max(float(np.sum(weights)), 1e-30))
        sigf = float(1.0 / math.sqrt(max(float(np.sum(weights)), 1e-30)))
        phase = math.atan2(fc_ref.imag, fc_ref.real)
        phase_factor = complex(math.cos(phase), math.sin(phase))
        fo = fobs * phase_factor
        fofc = (fobs - fc_ref_abs) * phase_factor
        twofofc = (2.0 * fobs - fc_ref_abs) * phase_factor
        split_values = {str(rows[i].get("split", "")).strip() or "unknown" for i in indices}
        scale_values = np.divide(
            dyn_scaled[idx],
            dyn_raw[idx],
            out=np.full_like(dyn_scaled[idx], np.nan, dtype=float),
            where=np.isfinite(dyn_raw[idx]) & (np.abs(dyn_raw[idx]) > 1e-30),
        )
        merged.append(
            {
                "h": hkl[0],
                "k": hkl[1],
                "l": hkl[2],
                "d_A": float(np.nanmedian(d_A[idx])),
                "n_rows": int(len(idx)),
                "splits": ",".join(sorted(split_values)),
                "F_dyn_obs": fobs,
                "sigF_dyn_obs": sigf,
                "F_dyn_calc": fc_ref_abs,
                "phase_calc_deg": math.degrees(phase),
                "Fo_real": fo.real,
                "Fo_imag": fo.imag,
                "FoFc_real": fofc.real,
                "FoFc_imag": fofc.imag,
                "TwoFoFc_real": twofofc.real,
                "TwoFoFc_imag": twofofc.imag,
                "mean_Iobs": float(np.mean(target[idx])),
                "mean_dyn_scaled": float(np.mean(dyn_scaled[idx])),
                "mean_born_scaled": float(np.nanmean(born_scaled[idx])) if np.any(np.isfinite(born_scaled[idx])) else float("nan"),
                "mean_q_dyn_over_born": float(np.nanmean(q_dyn_over_born[idx])) if np.any(np.isfinite(q_dyn_over_born[idx])) else float("nan"),
                "mean_row_scale": float(np.nanmean(scale_values)) if np.any(np.isfinite(scale_values)) else float("nan"),
                "mean_I_over_sigma": float(np.mean(isig[idx])),
            }
        )

    summary = {
        "input_rows": int(len(rows)),
        "valid_positive_rows": int(np.count_nonzero(valid)),
        "merged_reflections": int(len(merged)),
        "rejected_rows": int(len(rows) - np.count_nonzero(valid)),
        "min_i_over_sigma": float(options.min_i_over_sigma),
        "d_min_A": None if options.d_min_A is None else float(options.d_min_A),
        "definition": "|F_dyn_obs,row| = |F_model| * sqrt(I_obs / I_dyn_scaled), merged by signed HKL with propagated amplitude weights.",
    }
    return merged, summary


def _coefficient_dict(coefficients: Sequence[Mapping[str, object]], kind: str) -> dict[tuple[int, int, int], complex]:
    real_key, imag_key = {
        "fo": ("Fo_real", "Fo_imag"),
        "fofc": ("FoFc_real", "FoFc_imag"),
        "2fofc": ("TwoFoFc_real", "TwoFoFc_imag"),
    }[kind]
    return {
        (int(row["h"]), int(row["k"]), int(row["l"])): complex(float(row[real_key]), float(row[imag_key]))
        for row in coefficients
    }


def _synthesize_map(
    coefficients: Sequence[Mapping[str, object]],
    model: Mapping[str, object],
    kind: str,
    spacing_A: float,
    max_grid_points: int,
) -> tuple[np.ndarray, dict[str, object]]:
    hkls = np.asarray([[int(row["h"]), int(row["k"]), int(row["l"])] for row in coefficients], dtype=int)
    shape = _grid_shape(model["cell"], hkls, spacing_A)
    grid_points = int(shape[0] * shape[1] * shape[2])
    if grid_points > int(max_grid_points):
        raise ValueError(
            f"native dynamical map grid {shape} has {grid_points} points, above --native-map-max-grid-points={max_grid_points}"
        )
    reciprocal = np.zeros(shape, dtype=np.complex128)
    direct = _coefficient_dict(coefficients, kind)
    for hkl, value in direct.items():
        h, k, ell = hkl
        reciprocal[h % shape[0], k % shape[1], ell % shape[2]] = value
    filled_conjugates = 0
    conjugate_conflicts = 0
    for hkl, value in list(direct.items()):
        neg = (-hkl[0], -hkl[1], -hkl[2])
        neg_index = (neg[0] % shape[0], neg[1] % shape[1], neg[2] % shape[2])
        if neg not in direct:
            reciprocal[neg_index] = np.conjugate(value)
            filled_conjugates += 1
        elif abs(reciprocal[neg_index] - np.conjugate(value)) > 1e-8 * max(abs(value), 1.0):
            conjugate_conflicts += 1
    complex_grid = np.fft.ifftn(reciprocal) * float(grid_points) / float(gemmi.UnitCell(*model["cell"]).volume)
    data = np.asarray(complex_grid.real, dtype=np.float32)
    meta = {
        "kind": kind,
        "grid_shape": list(shape),
        "grid_points": grid_points,
        "filled_friedel_conjugates": int(filled_conjugates),
        "friedel_conjugate_conflicts": int(conjugate_conflicts),
        "imaginary_rms_before_real_projection": float(np.sqrt(np.mean(np.square(complex_grid.imag)))),
        "real_rms": float(np.sqrt(np.mean(np.square(data)))),
        "normalization": "sum(F exp(2*pi*i*h.x)) / cell_volume via numpy ifftn",
    }
    return data, meta


def _write_ccp4(path: Path, data: np.ndarray, model: Mapping[str, object]) -> None:
    grid = gemmi.FloatGrid(int(data.shape[0]), int(data.shape[1]), int(data.shape[2]))
    grid.set_unit_cell(gemmi.UnitCell(*model["cell"]))
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    grid.array[:] = data
    ccp4 = gemmi.Ccp4Map()
    ccp4.grid = grid
    ccp4.update_ccp4_header()
    path.parent.mkdir(parents=True, exist_ok=True)
    ccp4.write_ccp4_map(str(path))


def _write_coefficients_csv(path: Path, coefficients: Sequence[Mapping[str, object]]) -> None:
    fieldnames = [
        "h",
        "k",
        "l",
        "d_A",
        "n_rows",
        "splits",
        "F_dyn_obs",
        "sigF_dyn_obs",
        "F_dyn_calc",
        "phase_calc_deg",
        "Fo_real",
        "Fo_imag",
        "FoFc_real",
        "FoFc_imag",
        "TwoFoFc_real",
        "TwoFoFc_imag",
        "mean_Iobs",
        "mean_dyn_scaled",
        "mean_born_scaled",
        "mean_q_dyn_over_born",
        "mean_row_scale",
        "mean_I_over_sigma",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in coefficients:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def _write_coefficients_hkl(path: Path, coefficients: Sequence[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("# GROND native row-level dynamical Fourier coefficients\n")
        handle.write("# h k l F_dyn_obs sigF_dyn_obs F_dyn_calc phase_calc_deg Fo_real Fo_imag FoFc_real FoFc_imag TwoFoFc_real TwoFoFc_imag n_rows\n")
        for row in coefficients:
            handle.write(
                f"{int(row['h']):5d} {int(row['k']):5d} {int(row['l']):5d} "
                f"{float(row['F_dyn_obs']):14.6g} {float(row['sigF_dyn_obs']):14.6g} "
                f"{float(row['F_dyn_calc']):14.6g} {float(row['phase_calc_deg']):12.5f} "
                f"{float(row['Fo_real']):14.6g} {float(row['Fo_imag']):14.6g} "
                f"{float(row['FoFc_real']):14.6g} {float(row['FoFc_imag']):14.6g} "
                f"{float(row['TwoFoFc_real']):14.6g} {float(row['TwoFoFc_imag']):14.6g} "
                f"{int(row['n_rows']):6d}\n"
            )


def _cif_token(value: object) -> str:
    if value is None:
        return "?"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "?"
        return f"{value:.8g}"
    if isinstance(value, int):
        return str(value)
    text = str(value)
    if not text:
        return "?"
    if re.search(r"\\s|['\"]|^[_;]", text):
        return "'" + text.replace("'", "''") + "'"
    return text


def _fraction_text(value: float) -> str:
    value = float(value) % 1.0
    for den in (1, 2, 3, 4, 6, 8, 12):
        num = round(value * den)
        if abs(value - num / den) < 1e-8:
            if num == 0:
                return ""
            return f"+{num}/{den}" if den != 1 else f"+{num}"
    return f"+{value:.6g}"


def _op_triplet(record: Mapping[str, object]) -> str:
    rot = np.asarray(record.get("rot", np.eye(3)), dtype=float)
    trans = np.asarray(record.get("trans", np.zeros(3)), dtype=float)
    axes = ("x", "y", "z")
    parts = []
    for row, shift in zip(rot, trans, strict=True):
        expr = ""
        for coeff, axis in zip(row, axes, strict=True):
            coeff_i = int(round(float(coeff)))
            if coeff_i == 0:
                continue
            if not expr:
                expr += "-" if coeff_i < 0 else ""
            else:
                expr += "-" if coeff_i < 0 else "+"
            expr += axis if abs(coeff_i) == 1 else f"{abs(coeff_i)}*{axis}"
        expr += _fraction_text(float(shift))
        parts.append(expr or "0")
    return ",".join(parts)


def _write_dynamic_cif(
    path: Path,
    *,
    model: Mapping[str, object],
    model_path: Path,
    row_table: Path,
    coefficients: Sequence[Mapping[str, object]],
    summary: Mapping[str, object],
    maps: Mapping[str, object],
    wavelength_A: float,
) -> None:
    cell = [float(v) for v in model["cell"]]
    lines = [
        "data_grond_native_dynamical",
        "_audit_creation_method 'GROND native row-level dynamical Fourier export'",
        "_diffrn_radiation_probe electron",
        f"_diffrn_radiation_wavelength {wavelength_A:.8g}",
        f"_cell_length_a {cell[0]:.8g}",
        f"_cell_length_b {cell[1]:.8g}",
        f"_cell_length_c {cell[2]:.8g}",
        f"_cell_angle_alpha {cell[3]:.8g}",
        f"_cell_angle_beta {cell[4]:.8g}",
        f"_cell_angle_gamma {cell[5]:.8g}",
        "_space_group_name_H-M_alt 'model supplied operations'",
        "_grond_dynamic_map_source_model " + _cif_token(str(model_path)),
        "_grond_dynamic_map_source_row_table " + _cif_token(str(row_table)),
        "_grond_dynamic_map_formula '|F_dyn_obs,row| = |F_model| * sqrt(I_obs / I_dyn_scaled); map phases are model phases'",
        "_grond_dynamic_map_coefficients " + _cif_token(str(summary.get("coefficients_csv", ""))),
        "_grond_dynamic_map_fo " + _cif_token(str(maps.get("fo", {}).get("map", ""))),
        "_grond_dynamic_map_2fofc " + _cif_token(str(maps.get("2fofc", {}).get("map", ""))),
        "_grond_dynamic_map_fofc " + _cif_token(str(maps.get("fofc", {}).get("map", ""))),
        f"_grond_dynamic_map_input_rows {int(summary.get('input_rows', 0))}",
        f"_grond_dynamic_map_valid_rows {int(summary.get('valid_positive_rows', 0))}",
        f"_grond_dynamic_map_merged_reflections {int(summary.get('merged_reflections', 0))}",
        "",
        "loop_",
        "_space_group_symop_id",
        "_space_group_symop_operation_xyz",
    ]
    for idx, op in enumerate(model.get("symmetry_ops") or [{"rot": np.eye(3), "trans": np.zeros(3)}], start=1):
        lines.append(f"{idx} {_cif_token(_op_triplet(op))}")
    lines.extend(
        [
            "",
            "loop_",
            "_atom_type_symbol",
            "_atom_type_scat_Cromer_Mann_a1",
            "_atom_type_scat_Cromer_Mann_b1",
            "_atom_type_scat_Cromer_Mann_a2",
            "_atom_type_scat_Cromer_Mann_b2",
            "_atom_type_scat_Cromer_Mann_a3",
            "_atom_type_scat_Cromer_Mann_b3",
            "_atom_type_scat_Cromer_Mann_a4",
            "_atom_type_scat_Cromer_Mann_b4",
            "_atom_type_scat_Cromer_Mann_c",
        ]
    )
    for sfac in model.get("sfac", []):
        raw = list(sfac.get("raw", []))
        while len(raw) < 9:
            raw.append(0.0)
        lines.append(" ".join([_cif_token(sfac.get("element", "?"))] + [_cif_token(float(v)) for v in raw[:9]]))
    lines.extend(
        [
            "",
            "loop_",
            "_atom_site_label",
            "_atom_site_type_symbol",
            "_atom_site_fract_x",
            "_atom_site_fract_y",
            "_atom_site_fract_z",
            "_atom_site_occupancy",
            "_atom_site_adp_type",
            "_atom_site_U_iso_or_equiv",
        ]
    )
    for atom in model.get("atoms", []):
        adp_type = "Uani" if atom.get("adp_type") == "anis" else "Uiso"
        uiso = float(atom.get("uiso", (float(atom.get("u11", 0.0)) + float(atom.get("u22", 0.0)) + float(atom.get("u33", 0.0))) / 3.0))
        lines.append(
            " ".join(
                [
                    _cif_token(atom.get("name", "?")),
                    _cif_token(atom.get("element", "?")),
                    _cif_token(float(atom.get("x", 0.0))),
                    _cif_token(float(atom.get("y", 0.0))),
                    _cif_token(float(atom.get("z", 0.0))),
                    _cif_token(float(atom.get("occ", 1.0))),
                    _cif_token(adp_type),
                    _cif_token(uiso),
                ]
            )
        )
    aniso_atoms = [atom for atom in model.get("atoms", []) if atom.get("adp_type") == "anis"]
    if aniso_atoms:
        lines.extend(
            [
                "",
                "loop_",
                "_atom_site_aniso_label",
                "_atom_site_aniso_U_11",
                "_atom_site_aniso_U_22",
                "_atom_site_aniso_U_33",
                "_atom_site_aniso_U_12",
                "_atom_site_aniso_U_13",
                "_atom_site_aniso_U_23",
            ]
        )
        for atom in aniso_atoms:
            lines.append(
                " ".join(
                    [
                        _cif_token(atom.get("name", "?")),
                        _cif_token(float(atom.get("u11", 0.0))),
                        _cif_token(float(atom.get("u22", 0.0))),
                        _cif_token(float(atom.get("u33", 0.0))),
                        _cif_token(float(atom.get("u12", 0.0))),
                        _cif_token(float(atom.get("u13", 0.0))),
                        _cif_token(float(atom.get("u23", 0.0))),
                    ]
                )
            )
    lines.extend(
        [
            "",
            "loop_",
            "_refln_index_h",
            "_refln_index_k",
            "_refln_index_l",
            "_refln_F_dyn_obs",
            "_refln_F_dyn_obs_sigma",
            "_refln_F_dyn_calc",
            "_refln_phase_calc",
            "_refln_F_dyn_difference",
            "_refln_F_dyn_2Fo_minus_Fc",
            "_refln_grond_row_count",
            "_refln_grond_mean_Iobs",
            "_refln_grond_mean_Idyn_scaled",
            "_refln_grond_mean_Iborn_scaled",
            "_refln_grond_mean_q_dyn_over_born",
            "_refln_grond_mean_row_scale",
        ]
    )
    for row in coefficients:
        lines.append(
            " ".join(
                [
                    str(int(row["h"])),
                    str(int(row["k"])),
                    str(int(row["l"])),
                    _cif_token(float(row["F_dyn_obs"])),
                    _cif_token(float(row["sigF_dyn_obs"])),
                    _cif_token(float(row["F_dyn_calc"])),
                    _cif_token(float(row["phase_calc_deg"])),
                    _cif_token(float(row["F_dyn_obs"]) - float(row["F_dyn_calc"])),
                    _cif_token(2.0 * float(row["F_dyn_obs"]) - float(row["F_dyn_calc"])),
                    str(int(row["n_rows"])),
                    _cif_token(float(row["mean_Iobs"])),
                    _cif_token(float(row["mean_dyn_scaled"])),
                    _cif_token(float(row["mean_born_scaled"])),
                    _cif_token(float(row["mean_q_dyn_over_born"])),
                    _cif_token(float(row["mean_row_scale"])),
                ]
            )
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_native_dynamic_map_package(
    *,
    row_table: Path,
    model_path: Path,
    out_dir: Path,
    options: NativeDynamicMapOptions | None = None,
) -> dict[str, object]:
    """Write native dynamical coefficients, CCP4 maps, and a GROND CIF."""

    options = options or NativeDynamicMapOptions()
    row_table = Path(row_table).expanduser().resolve()
    model_path = Path(model_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not row_table.exists():
        return {"status": "missing_row_table", "row_table": str(row_table)}
    if not model_path.exists():
        return {"status": "missing_model", "model": str(model_path)}
    rows = _read_rows(row_table)
    model = parse_model(model_path)
    coefficients, summary = _merge_native_amplitudes(rows, model, options)
    if not coefficients:
        return {
            "status": "no_coefficients",
            "row_table": str(row_table),
            "model": str(model_path),
            "summary": summary,
        }

    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "native_dynamical_coefficients.csv"
    hkl_path = out_dir / "native_dynamical_coefficients.hkl"
    cif_path = out_dir / "grond_native_dynamical.cif"
    _write_coefficients_csv(csv_path, coefficients)
    _write_coefficients_hkl(hkl_path, coefficients)
    maps: dict[str, dict[str, object]] = {}
    for kind, name in (("fo", "grond_native_dynamic_fo.map"), ("2fofc", "grond_native_dynamic_2fofc.map"), ("fofc", "grond_native_dynamic_fofc.map")):
        data, meta = _synthesize_map(coefficients, model, kind, options.grid_spacing_A, options.max_grid_points)
        map_path = out_dir / name
        _write_ccp4(map_path, data, model)
        maps[kind] = {"map": str(map_path), "exists": map_path.exists(), **meta}
    summary = {
        **summary,
        "row_table": str(row_table),
        "model": str(model_path),
        "coefficients_csv": str(csv_path),
        "coefficients_hkl": str(hkl_path),
        "cif": str(cif_path),
        "grid_spacing_A": float(options.grid_spacing_A),
        "max_grid_points": int(options.max_grid_points),
    }
    _write_dynamic_cif(
        cif_path,
        model=model,
        model_path=model_path,
        row_table=row_table,
        coefficients=coefficients,
        summary=summary,
        maps=maps,
        wavelength_A=_model_wavelength_A(model_path),
    )
    manifest = {
        "status": "ok",
        "definition": "Native row-level GROND dynamical Fourier maps using dynamically inferred |Fe_obs| and model phases.",
        "outputs": {
            "coefficients_csv": str(csv_path),
            "coefficients_hkl": str(hkl_path),
            "dynamic_cif": str(cif_path),
        },
        "summary": summary,
        "maps": maps,
    }
    manifest_path = out_dir / "native_dynamic_maps_manifest.json"
    manifest["outputs"]["manifest"] = str(manifest_path)
    manifest_path.write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    return manifest
