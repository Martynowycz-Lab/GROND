"""Native row-level dynamical Fourier-map export.

The maps here are built from GROND's native row table, not from SHELXL FCF
files.  Each observed row is converted to an electron structure-factor
amplitude using the row-level dynamical prediction and the current model phase:

    |F_dyn_obs,row| = |F_model| * sqrt(I_obs / I_dyn_scaled)

Rows related by the model space-group rotations are merged in amplitude space
with inverse-variance weights, while genuinely independent Friedel branches
remain separate through amplitude inference.  Before a real electrostatic-
potential map is synthesized, each signed coefficient is expanded over the
model symmetry orbit (with phases recalculated at every mate) and the two
Friedel branches are projected onto the required Hermitian relation using
their propagated amplitude uncertainties.  The resulting Fo, Fo-Fc, and
2Fo-Fc coefficients are then Fourier transformed onto a CCP4 grid.  This
follows the usual dynamical-refinement map convention of using a model phase
with a dynamically estimated observed electron amplitude.
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


def _normalize_row_tables(
    row_table: Path | str | Sequence[Path | str],
    labels: Sequence[str] | None = None,
) -> list[tuple[str, Path]]:
    """Return labeled native row-table paths for one- or multi-dataset maps."""

    if isinstance(row_table, (str, Path)):
        paths: list[Path | str] = [row_table]
    else:
        paths = list(row_table)
    if not paths:
        return []
    label_values = list(labels or [])
    out: list[tuple[str, Path]] = []
    for idx, item in enumerate(paths):
        path = Path(item).expanduser().resolve()
        label = str(label_values[idx]).strip() if idx < len(label_values) else ""
        out.append((label or path.parent.name or f"dataset_{idx + 1}", path))
    return out


def _read_labeled_rows(
    row_tables: Sequence[tuple[str, Path]],
) -> tuple[list[dict[str, str]], dict[str, object]]:
    """Read policy-eligible rows and apply each raw-row weight exactly once.

    Dataset-policy tables retain ``sigma_eff_original`` so downstream consumers
    can recover the unweighted uncertainty.  Always starting from that value
    prevents a copied or resumed table from applying ``raw_row_weight`` twice.
    This is the same contract used by ``combine_native_nuisance_runs.py``.
    """

    rows: list[dict[str, str]] = []
    per_dataset: list[dict[str, object]] = []
    totals = {
        "source_rows": 0,
        "included_rows": 0,
        "excluded_rows": 0,
        "excluded_explicit_role_rows": 0,
        "excluded_completeness_only_rows": 0,
        "excluded_nonpositive_raw_row_weight_rows": 0,
        "excluded_invalid_raw_row_weight_rows": 0,
        "rows_with_explicit_raw_row_weight": 0,
        "downweighted_rows": 0,
        "upweighted_rows": 0,
        "sigma_eff_reset_from_original_rows": 0,
        "raw_row_weight_applied_rows": 0,
    }
    for label, path in row_tables:
        dataset_counts = {key: 0 for key in totals}
        for row in _read_rows(path):
            totals["source_rows"] += 1
            dataset_counts["source_rows"] += 1
            copied = dict(row)
            if not str(copied.get("dataset", "")).strip():
                copied["dataset"] = label
            copied["_grond_row_table_label"] = label
            copied["_grond_row_table_path"] = str(path)

            role = str(copied.get("dataset_role", "primary") or "primary").strip().lower()
            raw_weight_text = str(copied.get("raw_row_weight", "")).strip()
            explicit_weight = raw_weight_text != ""
            if explicit_weight:
                totals["rows_with_explicit_raw_row_weight"] += 1
                dataset_counts["rows_with_explicit_raw_row_weight"] += 1
                try:
                    raw_weight = float(raw_weight_text)
                except (TypeError, ValueError):
                    raw_weight = float("nan")
            else:
                raw_weight = 1.0

            exclusion_key = ""
            if role == "exclude":
                exclusion_key = "excluded_explicit_role_rows"
            elif role == "completeness_only":
                exclusion_key = "excluded_completeness_only_rows"
            elif not math.isfinite(raw_weight):
                exclusion_key = "excluded_invalid_raw_row_weight_rows"
            elif raw_weight <= 0.0:
                exclusion_key = "excluded_nonpositive_raw_row_weight_rows"
            if exclusion_key:
                totals["excluded_rows"] += 1
                totals[exclusion_key] += 1
                dataset_counts["excluded_rows"] += 1
                dataset_counts[exclusion_key] += 1
                continue

            original_sigma = _as_float(copied, "sigma_eff_original")
            if math.isfinite(original_sigma) and original_sigma > 0.0:
                copied["sigma_eff"] = f"{original_sigma:.17g}"
                totals["sigma_eff_reset_from_original_rows"] += 1
                dataset_counts["sigma_eff_reset_from_original_rows"] += 1
            sigma = _as_float(copied, "sigma_eff")
            if abs(raw_weight - 1.0) > 1e-12:
                if math.isfinite(sigma) and sigma > 0.0:
                    copied["sigma_eff"] = f"{sigma / math.sqrt(raw_weight):.17g}"
                    totals["raw_row_weight_applied_rows"] += 1
                    dataset_counts["raw_row_weight_applied_rows"] += 1
                direction_key = "downweighted_rows" if raw_weight < 1.0 else "upweighted_rows"
                totals[direction_key] += 1
                dataset_counts[direction_key] += 1

            totals["included_rows"] += 1
            dataset_counts["included_rows"] += 1
            rows.append(copied)
        per_dataset.append({"label": label, "path": str(path), **dataset_counts})
    return rows, {
        **totals,
        "per_dataset": per_dataset,
        "definition": (
            "dataset_role exclude/completeness_only and nonpositive or invalid raw_row_weight are omitted; "
            "included sigma_eff is restored from sigma_eff_original when available, then divided by "
            "sqrt(raw_row_weight) exactly once"
        ),
    }


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


def _symmetry_equivalent_hkls(
    hkl: tuple[int, int, int],
    symmetry_ops: Sequence[Mapping[str, object]],
) -> list[tuple[int, int, int]]:
    """Return the signed reciprocal orbit generated by model rotations.

    Translation parts change phases but not Miller-index equivalence and are
    therefore deliberately absent here.  Both the supplied rotation and its
    inverse are admitted so the result remains complete for parsers that emit
    generators instead of a closed operation list.  Friedel inversion is not
    added: in an acentric crystal it is an independent dynamical observation.
    """

    vector = np.asarray(hkl, dtype=float)
    matrices: list[np.ndarray] = [np.eye(3, dtype=float)]
    for operation in symmetry_ops:
        rotation = np.asarray(operation.get("rot", np.eye(3)), dtype=float)
        if rotation.shape != (3, 3) or not np.all(np.isfinite(rotation)):
            raise ValueError("model symmetry rotation must be a finite 3x3 matrix")
        matrices.extend((rotation, np.linalg.inv(rotation)))
    equivalents: set[tuple[int, int, int]] = set()
    for matrix in matrices:
        transformed = vector @ matrix
        rounded = np.rint(transformed).astype(int)
        if not np.allclose(transformed, rounded, rtol=0.0, atol=1e-7):
            raise ValueError(f"model symmetry did not map Miller index {hkl} to integers")
        equivalents.add(tuple(int(value) for value in rounded))
    return sorted(equivalents)


def _canonical_model_symmetry_hkl(
    hkl: tuple[int, int, int],
    symmetry_ops: Sequence[Mapping[str, object]],
) -> tuple[int, int, int]:
    return min(_symmetry_equivalent_hkls(hkl, symmetry_ops))


def _friedel_orbit_summary(
    keys: Sequence[tuple[int, int, int]],
    symmetry_ops: Sequence[Mapping[str, object]],
) -> dict[str, int]:
    key_set = set(keys)
    visited: set[tuple[int, int, int]] = set()
    independent_pairs = 0
    self_related = 0
    unpaired = 0
    for key in sorted(key_set):
        if key in visited:
            continue
        mate = _canonical_model_symmetry_hkl(tuple(-value for value in key), symmetry_ops)
        if mate == key:
            self_related += 1
            visited.add(key)
        elif mate in key_set:
            independent_pairs += 1
            visited.update((key, mate))
        else:
            unpaired += 1
            visited.add(key)
    return {
        "independent_friedel_pairs_retained": int(independent_pairs),
        "self_friedel_symmetry_orbits": int(self_related),
        "unpaired_friedel_orbits": int(unpaired),
    }


def _merge_native_amplitudes(
    rows: Sequence[Mapping[str, str]],
    model: Mapping[str, object],
    options: NativeDynamicMapOptions,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not rows:
        return [], {
            "input_rows": 0,
            "valid_positive_rows": 0,
            "merged_reflections": 0,
            "rejected_rows": 0,
            "min_i_over_sigma": float(options.min_i_over_sigma),
            "d_min_A": None if options.d_min_A is None else float(options.d_min_A),
            "definition": (
                "|F_dyn_obs,row| = |F_model| * sqrt(I_obs / I_dyn_scaled), merged over model-"
                "symmetry equivalents with propagated amplitude weights while preserving Friedel opposites."
            ),
        }
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
    d_A = np.asarray([_as_float(row, "d_A", float("nan")) for row in rows], dtype=float)
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

    symmetry_ops = list(model.get("symmetry_ops") or [])
    grouped: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for idx in np.where(valid)[0]:
        signed_hkl = (int(h[idx]), int(k[idx]), int(ell[idx]))
        grouped[_canonical_model_symmetry_hkl(signed_hkl, symmetry_ops)].append(int(idx))

    merged: list[dict[str, object]] = []
    for hkl, indices in sorted(grouped.items()):
        idx = np.asarray(indices, dtype=int)
        fc_ref = complex(fcalc_res_grid(model, np.asarray([hkl[0]]), np.asarray([hkl[1]]), np.asarray([hkl[2]]))[0])
        fc_ref_abs = abs(fc_ref)
        if fc_ref_abs <= 1e-12:
            continue
        # Use each row's own model amplitude before merging.  Exact model
        # symmetry makes these equal; retaining the row-level values both
        # avoids a hidden assumption and permits an explicit consistency audit.
        row_fobs = fc_abs[idx] * np.sqrt(np.maximum(target[idx], 0.0) / np.maximum(dyn_scaled[idx], 1e-30))
        sigma_f = 0.5 * fc_abs[idx] * sigma[idx] / np.sqrt(np.maximum(target[idx] * dyn_scaled[idx], 1e-30))
        weights = 1.0 / np.maximum(sigma_f * sigma_f, 1e-30)
        fobs = float(np.sum(weights * row_fobs) / max(float(np.sum(weights)), 1e-30))
        sigf = float(1.0 / math.sqrt(max(float(np.sum(weights)), 1e-30)))
        phase = math.atan2(fc_ref.imag, fc_ref.real)
        phase_factor = complex(math.cos(phase), math.sin(phase))
        fo = fobs * phase_factor
        fofc = (fobs - fc_ref_abs) * phase_factor
        twofofc = (2.0 * fobs - fc_ref_abs) * phase_factor
        split_values = {str(rows[i].get("split", "")).strip() or "unknown" for i in indices}
        dataset_values = {
            str(rows[i].get("dataset", rows[i].get("_grond_row_table_label", ""))).strip() or "unknown" for i in indices
        }
        signed_hkl_values = sorted({(int(h[i]), int(k[i]), int(ell[i])) for i in indices})
        fc_group = fc_abs[idx]
        fc_group_mean = float(np.mean(fc_group))
        fc_relative_spread = (
            float((np.max(fc_group) - np.min(fc_group)) / fc_group_mean) if fc_group_mean > 1e-30 else 0.0
        )
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
                "datasets": ",".join(sorted(dataset_values)),
                "dataset_count": int(len(dataset_values)),
                "source_signed_hkl_count": int(len(signed_hkl_values)),
                "source_signed_hkls": ";".join(",".join(str(value) for value in key) for key in signed_hkl_values),
                "max_model_Fc_relative_spread": fc_relative_spread,
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
                "mean_born_scaled": float(np.nanmean(born_scaled[idx]))
                if np.any(np.isfinite(born_scaled[idx]))
                else float("nan"),
                "mean_q_dyn_over_born": float(np.nanmean(q_dyn_over_born[idx]))
                if np.any(np.isfinite(q_dyn_over_born[idx]))
                else float("nan"),
                "mean_row_scale": float(np.nanmean(scale_values))
                if np.any(np.isfinite(scale_values))
                else float("nan"),
                "mean_I_over_sigma": float(np.mean(isig[idx])),
            }
        )

    valid_signed_hkls = sorted({(int(h[i]), int(k[i]), int(ell[i])) for i in np.where(valid)[0]})
    friedel_summary = _friedel_orbit_summary(list(grouped), symmetry_ops)
    summary = {
        "input_rows": int(len(rows)),
        "valid_positive_rows": int(np.count_nonzero(valid)),
        "merged_reflections": int(len(merged)),
        "exact_signed_reflections_before_symmetry_merge": int(len(valid_signed_hkls)),
        "model_symmetry_orbits_after_merge": int(len(merged)),
        "signed_reflections_collapsed_by_model_symmetry": int(len(valid_signed_hkls) - len(merged)),
        **friedel_summary,
        "max_model_Fc_relative_spread_within_symmetry_orbit": float(
            max((float(row["max_model_Fc_relative_spread"]) for row in merged), default=0.0)
        ),
        "rejected_rows": int(len(rows) - np.count_nonzero(valid)),
        "min_i_over_sigma": float(options.min_i_over_sigma),
        "d_min_A": None if options.d_min_A is None else float(options.d_min_A),
        "merge_policy": "model_symmetry_inverse_variance_preserve_friedel",
        "weighting": "inverse propagated variance of F_dyn_obs; I_dyn_scaled is treated as a deterministic model prediction",
        "definition": (
            "|F_dyn_obs,row| = |F_model| * sqrt(I_obs / I_dyn_scaled), merged over model-symmetry "
            "equivalents with propagated amplitude weights; independent Friedel opposites remain separate."
        ),
    }
    return merged, summary


def _coefficient_amplitude(row: Mapping[str, object], kind: str, fc_abs: float) -> tuple[float, float]:
    fobs = float(row["F_dyn_obs"])
    sigf = max(float(row["sigF_dyn_obs"]), 1e-30)
    if kind == "fo":
        return fobs, sigf
    if kind == "fofc":
        return fobs - fc_abs, sigf
    if kind == "2fofc":
        return 2.0 * fobs - fc_abs, 2.0 * sigf
    raise ValueError(f"unknown native map coefficient kind {kind!r}")


def _prepare_real_map_coefficients(
    coefficients: Sequence[Mapping[str, object]],
    model: Mapping[str, object],
    kind: str,
) -> tuple[dict[tuple[int, int, int], complex], dict[str, object]]:
    """Expand model symmetry and project signed coefficients onto Hermitian form.

    The signed, symmetry-merged amplitudes remain distinct in the exported
    coefficient tables.  A real electrostatic potential cannot retain two
    independent Friedel amplitudes, however, because it requires
    C(-h)=conj(C(h)).  This routine performs the unique inverse-variance least-
    squares projection onto that constraint and reports the pre-projection
    disagreement instead of silently discarding it.
    """

    symmetry_ops = list(model.get("symmetry_ops") or [])
    expanded_candidates: dict[tuple[int, int, int], list[tuple[complex, float, tuple[int, int, int]]]] = defaultdict(
        list
    )
    for row in coefficients:
        source_key = (int(row["h"]), int(row["k"]), int(row["l"]))
        for mate in _symmetry_equivalent_hkls(source_key, symmetry_ops):
            fc = complex(
                fcalc_res_grid(
                    model,
                    np.asarray([mate[0]], dtype=int),
                    np.asarray([mate[1]], dtype=int),
                    np.asarray([mate[2]], dtype=int),
                )[0]
            )
            fc_abs = abs(fc)
            if fc_abs <= 1e-12:
                continue
            amplitude, sigma_coefficient = _coefficient_amplitude(row, kind, fc_abs)
            phase_factor = fc / fc_abs
            expanded_candidates[mate].append((amplitude * phase_factor, sigma_coefficient, source_key))

    expanded: dict[tuple[int, int, int], tuple[complex, float, frozenset[tuple[int, int, int]]]] = {}
    overlapping_symmetry_assignments = 0
    for hkl, candidates in expanded_candidates.items():
        weights = np.asarray([1.0 / max(item[1] * item[1], 1e-30) for item in candidates], dtype=float)
        values = np.asarray([item[0] for item in candidates], dtype=np.complex128)
        value = complex(np.sum(weights * values) / max(float(np.sum(weights)), 1e-30))
        sigma_value = float(1.0 / math.sqrt(max(float(np.sum(weights)), 1e-30)))
        source_keys = frozenset(item[2] for item in candidates)
        if len(source_keys) > 1:
            overlapping_symmetry_assignments += 1
        expanded[hkl] = (value, sigma_value, source_keys)

    projected: dict[tuple[int, int, int], complex] = {}
    visited: set[tuple[int, int, int]] = set()
    filled_singletons = 0
    paired_branches = 0
    symmetry_self_pairs = 0
    conflict_abs: list[float] = []
    conflict_z: list[float] = []
    for hkl in sorted(expanded):
        if hkl in visited:
            continue
        value, sigma_value, source_keys = expanded[hkl]
        opposite = tuple(-index for index in hkl)
        if opposite == hkl:
            projected[hkl] = complex(value.real, 0.0)
            visited.add(hkl)
            continue
        if opposite not in expanded:
            projected[hkl] = value
            projected[opposite] = np.conjugate(value)
            filled_singletons += 1
            visited.update((hkl, opposite))
            continue
        opposite_value, opposite_sigma, opposite_source_keys = expanded[opposite]
        weight = 1.0 / max(sigma_value * sigma_value, 1e-30)
        opposite_weight = 1.0 / max(opposite_sigma * opposite_sigma, 1e-30)
        hermitian_value = complex(
            (weight * value + opposite_weight * np.conjugate(opposite_value)) / (weight + opposite_weight)
        )
        projected[hkl] = hermitian_value
        projected[opposite] = np.conjugate(hermitian_value)
        delta = float(abs(value - np.conjugate(opposite_value)))
        conflict_abs.append(delta)
        conflict_z.append(delta / max(math.sqrt(sigma_value * sigma_value + opposite_sigma * opposite_sigma), 1e-30))
        if source_keys == opposite_source_keys:
            symmetry_self_pairs += 1
        else:
            paired_branches += 1
        visited.update((hkl, opposite))

    conflict_array = np.asarray(conflict_abs, dtype=float)
    conflict_z_array = np.asarray(conflict_z, dtype=float)
    meta = {
        "signed_model_symmetry_orbits": int(len(coefficients)),
        "symmetry_expanded_signed_coefficients": int(len(expanded)),
        "overlapping_symmetry_assignments": int(overlapping_symmetry_assignments),
        "friedel_singleton_orbits_filled_by_conjugacy": int(filled_singletons),
        "independent_friedel_coefficient_pairs_projected": int(paired_branches),
        "symmetry_self_friedel_coefficient_pairs": int(symmetry_self_pairs),
        "real_map_reciprocal_coefficients": int(len(projected)),
        "friedel_preprojection_absolute_rms": float(np.sqrt(np.mean(np.square(conflict_array))))
        if len(conflict_array)
        else 0.0,
        "friedel_preprojection_normalized_rms": float(np.sqrt(np.mean(np.square(conflict_z_array))))
        if len(conflict_z_array)
        else 0.0,
        "friedel_preprojection_normalized_max": float(np.max(conflict_z_array)) if len(conflict_z_array) else 0.0,
        "symmetry_expansion": (
            "model rotations only; coefficient phases are recalculated from the supplied model at every mate, "
            "thereby including symmetry-translation phase shifts"
        ),
        "friedel_policy": (
            "independent signed amplitudes are retained through dynamical inference, then inverse-variance "
            "projected onto C(-h)=conj(C(h)) only for real electrostatic-potential synthesis"
        ),
        "hermitian_projection_weighting": "inverse propagated coefficient variance; no likelihood or ML map weighting",
    }
    return projected, meta


def _synthesize_map(
    coefficients: Sequence[Mapping[str, object]],
    model: Mapping[str, object],
    kind: str,
    spacing_A: float,
    max_grid_points: int,
) -> tuple[np.ndarray, dict[str, object]]:
    direct, coefficient_meta = _prepare_real_map_coefficients(coefficients, model, kind)
    hkls = np.asarray(list(direct), dtype=int)
    shape = _grid_shape(model["cell"], hkls, spacing_A)
    grid_points = int(shape[0] * shape[1] * shape[2])
    if grid_points > int(max_grid_points):
        raise ValueError(
            f"native dynamical map grid {shape} has {grid_points} points, above --native-map-max-grid-points={max_grid_points}"
        )
    reciprocal = np.zeros(shape, dtype=np.complex128)
    for hkl, value in direct.items():
        h, k, ell = hkl
        reciprocal[h % shape[0], k % shape[1], ell % shape[2]] = value
    # ``fcalc_res_grid`` uses the crystallographic convention
    # F(h) = sum_j f_j exp(+2*pi*i*h.r_j).  Recovering the real-space
    # potential therefore requires the negative-exponent transform.  NumPy's
    # forward FFT has that sign convention and, unlike ``ifftn``, is already
    # unnormalised, so no grid-point factor is needed here.
    complex_grid = np.fft.fftn(reciprocal) / float(gemmi.UnitCell(*model["cell"]).volume)
    data = np.asarray(complex_grid.real, dtype=np.float32)
    meta = {
        "kind": kind,
        "grid_shape": list(shape),
        "grid_points": grid_points,
        **coefficient_meta,
        # Backward-compatible keys.  Conflicts are resolved explicitly by the
        # uncertainty-aware Hermitian projection above, never by dropping a mate.
        "filled_friedel_conjugates": int(coefficient_meta["friedel_singleton_orbits_filled_by_conjugacy"]),
        "friedel_conjugate_conflicts": int(coefficient_meta["independent_friedel_coefficient_pairs_projected"]),
        "imaginary_rms_before_real_projection": float(np.sqrt(np.mean(np.square(complex_grid.imag)))),
        "real_rms": float(np.sqrt(np.mean(np.square(data)))),
        "normalization": "sum(F exp(-2*pi*i*h.x)) / cell_volume via numpy fftn",
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
        "datasets",
        "dataset_count",
        "source_signed_hkl_count",
        "source_signed_hkls",
        "max_model_Fc_relative_spread",
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
        handle.write(
            "# h k l F_dyn_obs sigF_dyn_obs F_dyn_calc phase_calc_deg Fo_real Fo_imag "
            "FoFc_real FoFc_imag TwoFoFc_real TwoFoFc_imag n_rows dataset_count datasets\n"
        )
        for row in coefficients:
            handle.write(
                f"{int(row['h']):5d} {int(row['k']):5d} {int(row['l']):5d} "
                f"{float(row['F_dyn_obs']):14.6g} {float(row['sigF_dyn_obs']):14.6g} "
                f"{float(row['F_dyn_calc']):14.6g} {float(row['phase_calc_deg']):12.5f} "
                f"{float(row['Fo_real']):14.6g} {float(row['Fo_imag']):14.6g} "
                f"{float(row['FoFc_real']):14.6g} {float(row['FoFc_imag']):14.6g} "
                f"{float(row['TwoFoFc_real']):14.6g} {float(row['TwoFoFc_imag']):14.6g} "
                f"{int(row['n_rows']):6d} {int(row.get('dataset_count', 0)):6d} {str(row.get('datasets', ''))}\n"
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
    row_tables: Sequence[tuple[str, Path]],
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
        "_grond_scattering_factor_kind electron",
        f"_cell_length_a {cell[0]:.8g}",
        f"_cell_length_b {cell[1]:.8g}",
        f"_cell_length_c {cell[2]:.8g}",
        f"_cell_angle_alpha {cell[3]:.8g}",
        f"_cell_angle_beta {cell[4]:.8g}",
        f"_cell_angle_gamma {cell[5]:.8g}",
        "_space_group_name_H-M_alt 'model supplied operations'",
        "_grond_dynamic_map_source_model " + _cif_token(str(model_path)),
        "_grond_dynamic_map_source_row_table_count " + _cif_token(len(row_tables)),
        "_grond_dynamic_map_formula '|F_dyn_obs,row| = |F_model| * sqrt(I_obs / I_dyn_scaled); map phases are model phases'",
        "_grond_dynamic_map_coefficients " + _cif_token(str(summary.get("coefficients_csv", ""))),
        "_grond_dynamic_map_fo " + _cif_token(str(maps.get("fo", {}).get("map", ""))),
        "_grond_dynamic_map_2fofc " + _cif_token(str(maps.get("2fofc", {}).get("map", ""))),
        "_grond_dynamic_map_fofc " + _cif_token(str(maps.get("fofc", {}).get("map", ""))),
        f"_grond_dynamic_map_input_rows {int(summary.get('input_rows', 0))}",
        f"_grond_dynamic_map_valid_rows {int(summary.get('valid_positive_rows', 0))}",
        f"_grond_dynamic_map_merged_reflections {int(summary.get('merged_reflections', 0))}",
        "_grond_dynamic_map_symmetry_merge_policy " + _cif_token(summary.get("merge_policy", "")),
        "_grond_dynamic_map_friedel_policy "
        + _cif_token(
            "signed Friedel branches retained during inference; uncertainty-aware Hermitian projection for real maps"
        ),
        "_grond_dynamic_map_map_weighting 'propagated amplitude variance only; no ML weighting'",
        "",
        "loop_",
        "_grond_dynamic_map_row_table_dataset",
        "_grond_dynamic_map_row_table_path",
    ]
    for label, table in row_tables:
        lines.append(f"{_cif_token(label)} {_cif_token(str(table))}")
    lines.extend(
        [
            "",
            "loop_",
            "_space_group_symop_id",
            "_space_group_symop_operation_xyz",
        ]
    )
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
        uiso = float(
            atom.get(
                "uiso", (float(atom.get("u11", 0.0)) + float(atom.get("u22", 0.0)) + float(atom.get("u33", 0.0))) / 3.0
            )
        )
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
            "_refln_grond_dataset_count",
            "_refln_grond_datasets",
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
                    str(int(row.get("dataset_count", 0))),
                    _cif_token(row.get("datasets", "")),
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
    row_table: Path | str | Sequence[Path | str],
    row_table_labels: Sequence[str] | None = None,
    model_path: Path,
    out_dir: Path,
    options: NativeDynamicMapOptions | None = None,
) -> dict[str, object]:
    """Write native dynamical coefficients, CCP4 maps, and a GROND CIF."""

    options = options or NativeDynamicMapOptions()
    row_tables = _normalize_row_tables(row_table, labels=row_table_labels)
    model_path = Path(model_path).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if not row_tables:
        return {"status": "missing_row_table", "row_table": ""}
    missing_tables = [{"label": label, "path": str(path)} for label, path in row_tables if not path.exists()]
    if missing_tables:
        return {
            "status": "missing_row_table",
            "row_tables": [{"label": label, "path": str(path)} for label, path in row_tables],
            "missing": missing_tables,
        }
    if not model_path.exists():
        return {"status": "missing_model", "model": str(model_path)}
    rows, dataset_policy = _read_labeled_rows(row_tables)
    model = parse_model(model_path)
    coefficients, summary = _merge_native_amplitudes(rows, model, options)
    rejected_after_policy = int(summary.get("rejected_rows", 0))
    summary = {
        **summary,
        "input_rows": int(dataset_policy["source_rows"]),
        "policy_eligible_rows": int(dataset_policy["included_rows"]),
        "rejected_rows": int(dataset_policy["source_rows"]) - int(summary.get("valid_positive_rows", 0)),
        "rejected_after_policy_rows": rejected_after_policy,
        "dataset_policy": dataset_policy,
    }
    if not coefficients:
        return {
            "status": "no_coefficients",
            "row_tables": [{"label": label, "path": str(path)} for label, path in row_tables],
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
    for kind, name in (
        ("fo", "grond_native_dynamic_fo.map"),
        ("2fofc", "grond_native_dynamic_2fofc.map"),
        ("fofc", "grond_native_dynamic_fofc.map"),
    ):
        data, meta = _synthesize_map(coefficients, model, kind, options.grid_spacing_A, options.max_grid_points)
        map_path = out_dir / name
        _write_ccp4(map_path, data, model)
        maps[kind] = {"map": str(map_path), "exists": map_path.exists(), **meta}
    summary = {
        **summary,
        "row_table": str(row_tables[0][1]) if len(row_tables) == 1 else "",
        "row_tables": [{"label": label, "path": str(path)} for label, path in row_tables],
        "row_table_count": int(len(row_tables)),
        "datasets": [label for label, _path in row_tables],
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
        row_tables=row_tables,
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
