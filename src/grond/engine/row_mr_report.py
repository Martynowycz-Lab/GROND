#!/usr/bin/env python3
"""Write publication tables with JANA/PETS-style row and MR factors.

The input is one or more GROND ``native_dyn_calc.csv`` row tables.  The
unmerged ``Robs``, ``Rall``, ``wRobs`` and ``wRall`` factors are calculated on
individual row observations.  ``MRobs``, ``MRall``, ``MwRobs`` and ``MwRall``
are calculated after arithmetic merging of symmetry-related rows from the
supplied model symmetry, with merged sigmas propagated as sqrt(sum(sigma^2)) / n.
All factors are on amplitudes F = sqrt(I): R = sum|Fo - Fc| / sum Fo, and the
weighted factors are Jana2020's wR(all) for refinement on F,
wR = sqrt(sum w (Fo - Fc)^2 / sum w Fo^2) with w = 1/(sigma_F^2 + (u Fo)^2).
The merged MR and MwR factors follow Klar et al. (2023).  The intensity-residual
form printed by earlier versions, sqrt(sum w (Io - Ic)^2 / sum w Io^2) with the
same w, is kept as ``wRobs_I``, ``wRall_I``, ``MwRobs_I`` and ``MwRall_I``.
These are GROND-computed values using the JANA/PETS dynamical-refinement
reporting family; they are not output parsed from JANA or PETS.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

try:
    from .bloch_engine import parse_model
except ImportError:  # pragma: no cover - used when executed as a script.
    from bloch_engine import parse_model


def safe_float(value, default=float("nan")) -> float:
    """Convert value to float, or return default when it cannot be converted."""

    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def row_source_sigma(row: dict) -> float:
    """Return the immutable source sigma with historical row-table fallbacks."""

    for name in ("source_sigma", "xds_sigma", "sigma", "sigma_eff"):
        value = safe_float(row.get(name))
        if math.isfinite(value) and value > 0.0:
            return value
    return float("nan")


def standard_json_data(value):
    """Return recursively JSON-safe data, mapping non-finite numbers to null."""

    if isinstance(value, dict):
        return {str(key): standard_json_data(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [standard_json_data(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def read_csv_rows(path: Path, dataset: str) -> list[dict]:
    """Read one row table, filling in a missing dataset label and the direct-engine column spellings.

    A missing target_I is taken from iobs, and a missing source_sigma from xds_sigma, sigma or sigma_eff.
    """

    with path.open(newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    for row in rows:
        if not str(row.get("dataset") or "").strip():
            row["dataset"] = dataset
        # The standalone Bloch engine writes its immutable observation columns
        # as iobs/sigma.  Auto's post-nuisance row contract renames the same
        # quantities target_I/sigma_eff.  Accept the direct-engine spelling
        # explicitly so forward-only audits can use the canonical
        # JANA/PETS report without manufacturing a second CSV dialect.
        if not str(row.get("target_I") or "").strip() and str(
            row.get("iobs") or ""
        ).strip():
            row["target_I"] = row["iobs"]
        if not str(row.get("source_sigma") or "").strip():
            for sigma_name in ("xds_sigma", "sigma", "sigma_eff"):
                if str(row.get(sigma_name) or "").strip():
                    row["source_sigma"] = row[sigma_name]
                    break
    return rows


def read_json(path: Path) -> dict:
    """Read a JSON file, or return an empty dict if it does not exist."""

    if not path.exists():
        return {}
    return json.loads(path.read_text())


def parse_named_path(text: str) -> tuple[str, Path]:
    """Parse name=path, or name a bare path after its parent folder (or its stem)."""

    if "=" in text:
        name, path = text.split("=", 1)
        return name.strip(), Path(path).expanduser()
    path = Path(text).expanduser()
    return path.parent.name or path.stem, path


def parse_resolution_limit_path(text: str) -> tuple[str, Path]:
    """Parse a named resolution-limit summary or directory path."""

    name, path = parse_named_path(text)
    if path.is_dir():
        path = path / "summary.json"
    return name, path


def atom_formula(model: dict) -> str:
    """Return the asymmetric-unit formula from atom elements and rounded occupancies, C and H first."""

    counts: dict[str, int] = defaultdict(int)
    for atom in model.get("atoms", []):
        element = str(atom.get("element", "")).strip()
        if not element:
            continue
        occ = safe_float(atom.get("occ", 1.0), 1.0)
        if occ <= 0:
            continue
        counts[element.capitalize()] += max(1, int(round(occ)))
    if not counts:
        return "not determined"
    order = ["C", "H", "N", "O", "S", "P", "F", "Cl", "Br", "I"]
    keys = [k for k in order if k in counts] + sorted(k for k in counts if k not in order)
    return "".join(f"{k}{counts[k] if counts[k] != 1 else ''}" for k in keys)


def model_lattice_label(model: dict) -> str:
    """Return the report's lattice label: the SHELX LATT code, or 'from CIF/model symmetry'."""

    latt = model.get("shelx_latt")
    if latt is None:
        return "from CIF/model symmetry"
    return f"SHELX LATT {latt}"


def reciprocal_equivalent_keys(h: int, k: int, l: int, symmetry_ops: list[dict]) -> list[tuple[int, int, int]]:
    """Return the sorted HKLs related to (h, k, l) by the model rotations and their inverses."""

    hkl = np.asarray([int(h), int(k), int(l)], dtype=float)
    keys: list[tuple[int, int, int]] = []
    for op in symmetry_ops or [{"rot": np.eye(3).tolist()}]:
        rot = np.asarray(op.get("rot", np.eye(3)), dtype=float)
        for mat in (rot, np.linalg.inv(rot)):
            equiv = np.rint(hkl @ mat).astype(int)
            keys.append(tuple(int(v) for v in equiv))
    return sorted(set(keys))


def canonical_hkl(h: int, k: int, l: int, symmetry_ops: list[dict]) -> tuple[int, int, int]:
    """Return the smallest symmetry-equivalent HKL, used as the merge key."""

    return min(reciprocal_equivalent_keys(h, k, l, symmetry_ops))


def jana_adjusted_observations(obs: np.ndarray, sigma: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Apply the weak/negative-observation convention reported by Klar et al."""

    weak = obs < 0.01 * sigma
    adjusted = np.where(weak, 0.0, obs)
    return adjusted, weak


def row_sqrt_weights(obs: np.ndarray, sigma: np.ndarray, instability_u: float = 0.01) -> np.ndarray:
    """Return JANA-style amplitude weights 1/sqrt(sigma_F² + (u F)²), with F = sqrt(I) and sigma_F = sigma/(2F).

    Weak rows (I < 0.01 sigma) use sigma_F = 5 sqrt(sigma).
    """

    obs_adjusted, weak = jana_adjusted_observations(obs, sigma)
    sqrt_obs = np.sqrt(np.maximum(obs_adjusted, 0.0))
    sigma_sqrt = sigma / np.maximum(2.0 * sqrt_obs, 1e-30)
    sigma_sqrt[weak] = 5.0 * np.sqrt(sigma[weak])
    denom = np.sqrt(sigma_sqrt * sigma_sqrt + (float(instability_u) * sqrt_obs) ** 2)
    return 1.0 / np.maximum(denom, 1e-30)


def row_mr_r_factors(obs, sigma, calc, instability_u: float = 0.01, observed_sigma_cut: float = 3.0) -> dict:
    """Return Nall, Nobs, Robs, Rall, wRobs, wRall and the *_I forms of calc against obs in the JANA/PETS convention.

    Weak observations (I < 0.01 sigma) are set to zero first. R is sum|Fo - Fc| / sum Fo on amplitudes
    F = sqrt(max(I, 0)). wR is the amplitude form, Jana2020's wR(all) for refinement on F:
    sqrt(sum w (Fo - Fc)^2 / sum w Fo^2) with w = 1/(sigma_F^2 + (u Fo)^2), the square of row_sqrt_weights.
    wRobs_I and wRall_I keep the intensity-residual form of earlier versions, sqrt(sum w (Io - Ic)^2 / sum w Io^2)
    with the same w. Observed rows have I > observed_sigma_cut * sigma before the weak-row adjustment.
    """

    obs = np.asarray(obs, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    calc = np.asarray(calc, dtype=float)
    mask = np.isfinite(obs) & np.isfinite(sigma) & np.isfinite(calc) & (sigma > 0)
    obs = obs[mask]
    sigma = sigma[mask]
    calc = calc[mask]
    if len(obs) == 0:
        return {
            "Nall": 0,
            "Nobs": 0,
            "Robs": float("nan"),
            "Rall": float("nan"),
            "wRobs": float("nan"),
            "wRall": float("nan"),
            "wRobs_I": float("nan"),
            "wRall_I": float("nan"),
        }

    obs_original = obs.copy()
    obs, _weak = jana_adjusted_observations(obs, sigma)
    obs_pos = np.maximum(obs, 0.0)
    calc_pos = np.maximum(calc, 0.0)
    fobs = np.sqrt(obs_pos)
    fcalc = np.sqrt(calc_pos)
    observed = obs_original > float(observed_sigma_cut) * sigma

    def r_factor(use):
        """Return the amplitude R factor sum|Fo - Fc| / sum Fo over the rows in use, NaN if none."""

        if not np.any(use):
            return float("nan")
        return float(np.sum(np.abs(fobs[use] - fcalc[use])) / max(np.sum(fobs[use]), 1e-30))

    # row_sqrt_weights is w^(1/2) = (sigma_F^2 + (u Fo)^2)^(-1/2); squaring the weighted terms gives w.
    weights = row_sqrt_weights(obs_original, sigma, instability_u=instability_u)
    amplitude_residual = fobs - fcalc
    intensity_residual = obs - calc

    def wr_factor(residual, reference, use):
        """Return sqrt(sum (w^(1/2) residual)^2 / sum (w^(1/2) reference)^2) over the rows in use, NaN if none."""

        if not np.any(use):
            return float("nan")
        return float(
            np.sqrt(
                np.sum((weights[use] * residual[use]) ** 2) / max(np.sum((weights[use] * reference[use]) ** 2), 1e-30)
            )
        )

    every = np.ones_like(observed, dtype=bool)
    return {
        "Nall": int(len(obs)),
        "Nobs": int(np.count_nonzero(observed)),
        "Robs": r_factor(observed),
        "Rall": r_factor(every),
        "wRobs": wr_factor(amplitude_residual, fobs, observed),
        "wRall": wr_factor(amplitude_residual, fobs, every),
        "wRobs_I": wr_factor(intensity_residual, obs, observed),
        "wRall_I": wr_factor(intensity_residual, obs, every),
    }


def merged_arrays(hkl, obs, sigma, calc, symmetry_ops: list[dict]):
    """Merge rows by symmetry-equivalent HKL: mean obs and calc, sigma = sqrt(sum sigma²)/n, and multiplicity."""

    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, (h, k, l) in enumerate(hkl):
        groups[canonical_hkl(int(h), int(k), int(l), symmetry_ops)].append(i)

    merged_hkl = []
    merged_obs = []
    merged_sigma = []
    merged_calc = []
    multiplicity = []
    for key, idx_list in sorted(groups.items()):
        idx = np.asarray(idx_list, dtype=int)
        good = np.isfinite(obs[idx]) & np.isfinite(sigma[idx]) & np.isfinite(calc[idx]) & (sigma[idx] > 0)
        if not np.any(good):
            continue
        idx = idx[good]
        n = len(idx)
        merged_hkl.append(key)
        merged_obs.append(float(np.mean(obs[idx])))
        merged_calc.append(float(np.mean(calc[idx])))
        merged_sigma.append(float(np.sqrt(np.sum(sigma[idx] * sigma[idx])) / max(n, 1)))
        multiplicity.append(int(n))
    return {
        "hkl": np.asarray(merged_hkl, dtype=int) if merged_hkl else np.zeros((0, 3), dtype=int),
        "obs": np.asarray(merged_obs, dtype=float),
        "sigma": np.asarray(merged_sigma, dtype=float),
        "calc": np.asarray(merged_calc, dtype=float),
        "multiplicity": np.asarray(multiplicity, dtype=int),
    }


def compute_model_metrics(rows: list[dict], model: dict, calc_column: str, instability_u: float = 0.01) -> dict:
    """Return the row R factors and merged MR factors of one calc column.

    Merging uses rows with finite values and integral HKL, after the weak-observation adjustment.
    """

    hkl_float = np.asarray(
        [[safe_float(r.get("h")), safe_float(r.get("k")), safe_float(r.get("l"))] for r in rows],
        dtype=float,
    )
    obs = np.asarray([safe_float(r.get("target_I")) for r in rows], dtype=float)
    sigma = np.asarray([row_source_sigma(r) for r in rows], dtype=float)
    calc = np.asarray([safe_float(r.get(calc_column)) for r in rows], dtype=float)
    base = row_mr_r_factors(obs, sigma, calc, instability_u=instability_u)
    finite_row = np.isfinite(obs) & np.isfinite(sigma) & np.isfinite(calc) & (sigma > 0)
    rounded_hkl = np.rint(hkl_float)
    valid_hkl = np.all(np.isfinite(hkl_float), axis=1) & np.all(
        np.isclose(hkl_float, rounded_hkl, rtol=0.0, atol=1e-6),
        axis=1,
    )
    merge_support = finite_row & valid_hkl
    obs_for_merge, _weak = jana_adjusted_observations(obs[merge_support], sigma[merge_support])
    merged = merged_arrays(
        rounded_hkl[merge_support].astype(int),
        obs_for_merge,
        sigma[merge_support],
        calc[merge_support],
        model.get("symmetry_ops", []),
    )
    mr = row_mr_r_factors(merged["obs"], merged["sigma"], merged["calc"], instability_u=instability_u)
    mult = merged["multiplicity"]
    return {
        **base,
        "Ninput": int(len(rows)),
        "Nexcluded": int(len(rows) - base["Nall"]),
        "Nmerge_input": int(base["Nall"]),
        "Nmerge_used": int(np.count_nonzero(merge_support)),
        "Nmerge_excluded": int(base["Nall"] - np.count_nonzero(merge_support)),
        "MRobs": mr["Robs"],
        "MRall": mr["Rall"],
        "MwRobs": mr["wRobs"],
        "MwRall": mr["wRall"],
        "MwRobs_I": mr["wRobs_I"],
        "MwRall_I": mr["wRall_I"],
        "MNobs": mr["Nobs"],
        "MNall": mr["Nall"],
        "merge_groups": int(len(mult)),
        "merge_multiplicity_mean": float(np.mean(mult)) if len(mult) else float("nan"),
        "merge_multiplicity_max": int(np.max(mult)) if len(mult) else 0,
    }


def compute_row_mr_summary(rows: list[dict], model: dict, instability_u: float = 0.01) -> dict:
    """Calculate Born and Bloch factors on one explicitly matched row support."""

    obs = np.asarray([safe_float(row.get("target_I")) for row in rows], dtype=float)
    sigma = np.asarray([row_source_sigma(row) for row in rows], dtype=float)
    dynamic_calc = np.asarray([safe_float(row.get("dyn_scaled")) for row in rows], dtype=float)
    born_calc = np.asarray([safe_float(row.get("born_scaled")) for row in rows], dtype=float)
    matched_support = (
        np.isfinite(obs)
        & np.isfinite(sigma)
        & (sigma > 0)
        & np.isfinite(dynamic_calc)
        & np.isfinite(born_calc)
    )
    matched_rows = [row for row, keep in zip(rows, matched_support, strict=True) if keep]
    n_input = int(len(rows))
    n_matched = int(np.count_nonzero(matched_support))
    n_excluded = int(n_input - n_matched)

    born = compute_model_metrics(matched_rows, model, "born_scaled", instability_u=instability_u)
    dynamic = compute_model_metrics(matched_rows, model, "dyn_scaled", instability_u=instability_u)
    for metrics in (born, dynamic):
        metrics["Ninput"] = n_input
        metrics["Nexcluded"] = n_excluded
        metrics["Nmatched"] = n_matched
    return {
        "definition": {
            "style": "JANA/PETS-style dynamical-refinement row and merged-row residual factors (GROND implementation); R and wR as Jana2020 for refinement on F, MR and MwR after Klar et al. (2023)",
            "provenance": "computed by GROND from its final signed row table; not emitted by JANA or PETS",
            "matched_support": "Born and Bloch factors use the identical intersection of rows with finite Iobs, positive finite immutable source sigma(Iobs), finite born_scaled, and finite dyn_scaled",
            "matched_support_counts": "Ninput is the supplied row count; Nmatched/Nall is the common finite Born/Bloch support; Nexcluded is Ninput-Nmatched",
            "unmerged_metrics": "Nobs, Nall, Robs, Rall, wRobs, and wRall use individual observations from the matched row support",
            "merged_metrics": "MNobs, MNall, MRobs, MRall, MwRobs, and MwRall use post-refinement model-symmetry groups; Nmerge_excluded counts matched rows with missing, non-finite, or non-integral HKL indices",
            "observed_cutoff": "Iobs > 3 sigma(Iobs), used for Robs, wRobs, MRobs, and MwRobs",
            "all_reflection_metrics": "Rall, wRall, MRall, and MwRall use all finite observations after row or merged-row grouping.",
            "observed_weighted_metrics": "wRobs and MwRobs are optional extensions that apply the same weighted residual formula to the Iobs > 3 sigma(Iobs) subsets; the conventional core reporting family is Robs, Rall, wRall, MRobs, MRall, and MwRall.",
            "R_formula": "sum(abs(sqrt(Iobs,JANA) - sqrt(max(Icalc,0)))) / sum(sqrt(Iobs,JANA))",
            "wR_formula": "sqrt(sum((w * (sqrt(Iobs,JANA) - sqrt(max(Icalc,0))))^2) / sum((w * sqrt(Iobs,JANA))^2)); amplitude form, Jana2020 wR(all) for refinement on F; used for wRobs, wRall, MwRobs, and MwRall",
            "wR_I_formula": "sqrt(sum((w * (Iobs,JANA - Icalc))^2) / sum((w * Iobs,JANA)^2)); intensity-residual form printed by earlier GROND versions, kept as wRobs_I, wRall_I, MwRobs_I, and MwRall_I",
            "nonpositive_intensity_policy": "following Klar et al. SI, rows with Iobs < 0.01 sigma(Iobs) use Iobs=0 and sigma(sqrt(Iobs))=5 sqrt(sigma(Iobs))",
            "weight_formula": "w = (sigma(sqrt(Iobs,JANA))^2 + (u*sqrt(Iobs,JANA))^2)^(-1/2), with u=0.01 by default; w^2 is the least-squares weight 1/(sigma_F^2 + (u F)^2)",
            "sigma_contract": "source_sigma is the immutable imported experimental sigma(Iobs); xds_sigma is accepted as its historical column name and sigma_eff only as a legacy fallback during CSV normalization",
            "sigma_sqrt_formula": "sigma(sqrt(Iobs)) = source_sigma/(2 sqrt(Iobs)) for ordinary positive rows; weak rows use 5 sqrt(source_sigma)",
            "instability_u": float(instability_u),
            "merge": "arithmetic means over model-symmetry-related rows; sigma_M = sqrt(sum sigma_i^2) / n",
            "sigma_source": "immutable experimental source_sigma from the supplied GROND row table; raw_row_weight and fitted sigma_eff are not folded into these JANA/PETS-style factors",
            "dataset_policy_note": "Dataset-policy raw_row_weight and fitted sigma_eff may affect training or inclusion, but neither is multiplied into the reported JANA/PETS-style R/MR formula.",
            "friedel_policy": "Friedel opposites remain separate in acentric groups unless they are related by an explicit model-symmetry rotation.",
            "compatibility_alias": "row_mr.kinematic is a deprecated byte-for-byte alias of row_mr.born; it is not a conventional merged-refinement result.",
            "endpoint_exclusion": "SHELXL R1(F), wR2(F^2), GooF, and endpoint reflection counts are intentionally not part of this row-statistics block.",
        },
        "support": {
            "Ninput": n_input,
            "Nmatched": n_matched,
            "Nexcluded": n_excluded,
            "status": "matched" if n_matched > 0 else "empty_matched_support",
        },
        "dynamic": dynamic,
        "born": born,
        "kinematic": born,
    }


def compute_per_dataset_row_mr(rows: list[dict], model: dict, instability_u: float = 0.01) -> dict[str, dict]:
    """Return the same JANA/PETS-style factors separately for each dataset."""

    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row.get("dataset") or "dataset")].append(row)
    return {
        dataset: compute_row_mr_summary(dataset_rows, model, instability_u=instability_u)
        for dataset, dataset_rows in sorted(grouped.items())
    }


def fmt_float(value, digits: int = 3) -> str:
    """Format a number with the given decimals, or 'n/a' if it is not finite."""

    v = safe_float(value)
    if not math.isfinite(v):
        return "n/a"
    return f"{v:.{digits}f}"


def fmt_int_pair(a, b) -> str:
    """Format two integers as 'a, b'."""

    return f"{int(a)}, {int(b)}"


def table_rows(summary: dict) -> list[dict]:
    """Return the historical table shape retained for API compatibility."""

    out = []
    for label, key in (("dynamical", "dynamic"), ("kinematical", "kinematic")):
        m = summary[key]
        out.append(
            {
                "Regularization": label,
                "Nobs, Nall": fmt_int_pair(m["Nobs"], m["Nall"]),
                "MNobs, MNall": fmt_int_pair(m["MNobs"], m["MNall"]),
                "Robs": fmt_float(m["Robs"]),
                "Rall": fmt_float(m["Rall"]),
                "wRall": fmt_float(m["wRall"]),
                "MRobs": fmt_float(m["MRobs"]),
                "MRall": fmt_float(m["MRall"]),
                "MwRall": fmt_float(m["MwRall"]),
            }
        )
    return out


def jana_table_rows(summary: dict) -> list[dict]:
    """Return the explicit JANA/PETS-style presentation table."""

    out = []
    for label, key in (
        ("Bloch dynamical", "dynamic"),
        ("matched Born (kinematical control)", "born"),
    ):
        m = summary[key]
        out.append(
            {
                "Row model": label,
                "Nobs, Nall": fmt_int_pair(m["Nobs"], m["Nall"]),
                "Nmerge used, excluded": fmt_int_pair(m["Nmerge_used"], m["Nmerge_excluded"]),
                "MNobs, MNall": fmt_int_pair(m["MNobs"], m["MNall"]),
                "Robs": fmt_float(m["Robs"]),
                "Rall": fmt_float(m["Rall"]),
                "wRobs": fmt_float(m["wRobs"]),
                "wRall": fmt_float(m["wRall"]),
                "MRobs": fmt_float(m["MRobs"]),
                "MRall": fmt_float(m["MRall"]),
                "MwRobs": fmt_float(m["MwRobs"]),
                "MwRall": fmt_float(m["MwRall"]),
            }
        )
    return out


def per_dataset_table_rows(summaries: dict[str, dict]) -> list[dict]:
    """Format per-dataset JANA/PETS-style factors for CSV/Markdown audit tables."""

    out: list[dict] = []
    for dataset, summary in summaries.items():
        for row in jana_table_rows(summary):
            out.append({"Dataset": dataset, **row})
    return out


def row_d_spacing(row: dict, model: dict | None = None) -> float:
    """Return d(hkl), preferring the shared reporting-model cell when supplied."""

    if model is not None:
        try:
            cell = gemmi.UnitCell(*[float(value) for value in model["cell"]])
            hkl = [int(float(row[key])) for key in ("h", "k", "l")]
            value = float(cell.calculate_d(hkl))
            return value if math.isfinite(value) and value > 0.0 else float("nan")
        except (KeyError, TypeError, ValueError, OverflowError):
            return float("nan")
    return safe_float(row.get("d_A"))


def dmin_for_rows(rows: list[dict], model: dict | None = None) -> float:
    """Return the highest-resolution d spacing represented by these rows."""

    dvals = np.asarray([row_d_spacing(row, model) for row in rows], dtype=float)
    finite = dvals[np.isfinite(dvals) & (dvals > 0.0)]
    return float(np.min(finite)) if len(finite) else float("nan")


def rows_at_or_below_resolution(
    rows: list[dict],
    dmin: float,
    model: dict | None = None,
) -> list[dict]:
    """Return rows retained by a d-spacing cutoff."""

    cutoff = safe_float(dmin)
    if not math.isfinite(cutoff) or cutoff <= 0.0:
        return list(rows)
    return [row for row in rows if row_d_spacing(row, model) >= cutoff - 1e-12]


def load_resolution_limit_payloads(items: list[str]) -> list[dict]:
    """Load resolution-limit summaries passed to the publication table generator."""

    payloads = []
    for item in items:
        name, path = parse_resolution_limit_path(item)
        payload = read_json(path)
        if not payload:
            continue
        payload = dict(payload)
        payload["dataset"] = name
        payload["summary_path"] = str(path.resolve())
        payloads.append(payload)
    return payloads


def choose_resolution_limit_payload(payloads: list[dict]) -> dict:
    """Prefer a merged/combined resolution-limit summary, otherwise use the first."""

    for payload in payloads:
        scope = str(payload.get("scope", "")).lower()
        dataset = str(payload.get("dataset", "")).lower()
        if scope in {"merged", "combined"} or dataset in {"merged", "combined"}:
            return payload
    return payloads[0] if payloads else {}


def suggested_resolution_dmin(payloads: list[dict]) -> float:
    """Return the promoted publication dmin from resolution-limit summaries."""

    payload = choose_resolution_limit_payload(payloads)
    decision = payload.get("decision") or {}
    return safe_float(decision.get("promoted_min_d_spacing"))


def endpoint_summary_rows(
    requested_rows: list[dict],
    model: dict,
    resolution_payloads: list[dict],
    instability_u: float = 0.01,
    *,
    full_rows: list[dict] | None = None,
    requested_min_d_spacing: float = 0.0,
    selected_scale_contract: str = "",
) -> tuple[list[dict], dict]:
    """Build the row-contract summaries reported by auto mode.

    The rows are the requested resolution with the full fitted scale curve, then the
    promoted/suggested resolution. A multi-dataset run (``full``) also reports its selected
    rows at both resolutions.
    """

    endpoint_rows: list[dict] = []
    explicit_requested_dmin = safe_float(requested_min_d_spacing)
    selected_contract = str(selected_scale_contract).strip().lower()
    explicit_multi_contract = selected_contract == "full"
    resolution_model = (
        model
        if explicit_multi_contract or (math.isfinite(explicit_requested_dmin) and explicit_requested_dmin > 0.0)
        else None
    )
    requested_dmin = (
        explicit_requested_dmin
        if math.isfinite(explicit_requested_dmin) and explicit_requested_dmin > 0.0
        else dmin_for_rows(requested_rows, resolution_model)
    )
    suggested_dmin = suggested_resolution_dmin(resolution_payloads)
    if not math.isfinite(suggested_dmin):
        suggested_dmin = requested_dmin

    def append(label: str, rows: list[dict], dmin: float, scale_model: str) -> None:
        """Add one row-contract summary: the rows kept at dmin, their row/MR factors and the scale description."""

        kept = rows_at_or_below_resolution(rows, dmin, resolution_model)
        summary = compute_row_mr_summary(kept, model, instability_u=instability_u) if kept else {}
        endpoint_rows.append(
            {
                "label": label,
                "status": "ok",
                "dmin_A": dmin_for_rows(kept, resolution_model) if kept else safe_float(dmin),
                "requested_cutoff_A": safe_float(dmin),
                "row_count": len(kept),
                "scale_model": scale_model,
                "scale_params": "n/a",
                "row_mr": summary,
            }
        )

    full_control_rows = list(full_rows) if full_rows else list(requested_rows)
    append("requested_resolution_full_scales", full_control_rows, requested_dmin, "full fitted scale curve")
    if explicit_multi_contract:
        append(
            "requested_resolution_selected_full_scales",
            requested_rows,
            requested_dmin,
            "selected full fitted scale curve",
        )
        append(
            "suggested_resolution_selected_full_scales",
            requested_rows,
            suggested_dmin,
            "selected full fitted scale curve",
        )
    else:
        append("suggested_resolution_full_scales", requested_rows, suggested_dmin, "full fitted scale curve")
    metadata = {
        "requested_dmin_A": requested_dmin,
        "suggested_dmin_A": suggested_dmin,
        "full_control_rows_supplied": bool(full_rows),
        "selected_scale_contract": selected_contract or "legacy_unspecified",
        "d_spacing_source": "shared reporting-model cell",
        "resolution_limit_summary": (
            choose_resolution_limit_payload(resolution_payloads).get("summary_path") if resolution_payloads else ""
        ),
    }
    return endpoint_rows, metadata


def endpoint_table_rows(endpoint_rows: list[dict]) -> list[dict]:
    """Return the historical endpoint-table shape for API compatibility.

    The historical function and payload names retain ``endpoint`` for backward
    compatibility.  These rows contain JANA/PETS-style row factors, not SHELXL
    corrected-HKL endpoint statistics.
    """

    rows = []
    for item in endpoint_rows:
        dyn = (item.get("row_mr") or {}).get("dynamic") or {}
        rows.append(
            {
                "Endpoint": item.get("label", ""),
                "Status": item.get("status", "ok"),
                "dmin A": fmt_float(item.get("dmin_A")),
                "Scale params": item.get("scale_params", "n/a"),
                "Nobs, Nall": fmt_int_pair(dyn.get("Nobs", 0), dyn.get("Nall", 0)) if dyn else "n/a",
                "Robs": fmt_float(dyn.get("Robs")),
                "Rall": fmt_float(dyn.get("Rall")),
                "wRall": fmt_float(dyn.get("wRall")),
                "MRobs": fmt_float(dyn.get("MRobs")),
                "MRall": fmt_float(dyn.get("MRall")),
                "MwRall": fmt_float(dyn.get("MwRall")),
            }
        )
    return rows


def row_contract_table_rows(endpoint_rows: list[dict]) -> list[dict]:
    """Format explicit JANA/PETS-style resolution/scale row contracts."""

    rows = []
    for item in endpoint_rows:
        dyn = (item.get("row_mr") or {}).get("dynamic") or {}
        rows.append(
            {
                "Row contract": item.get("label", ""),
                "Status": item.get("status", "ok"),
                "dmin A": fmt_float(item.get("dmin_A")),
                "Scale params": item.get("scale_params", "n/a"),
                "Nobs, Nall": fmt_int_pair(dyn.get("Nobs", 0), dyn.get("Nall", 0)) if dyn else "n/a",
                "Nmerge used, excluded": (
                    fmt_int_pair(dyn["Nmerge_used"], dyn["Nmerge_excluded"])
                    if dyn.get("Nmerge_used") is not None and dyn.get("Nmerge_excluded") is not None
                    else "n/a"
                ),
                "Robs": fmt_float(dyn.get("Robs")),
                "Rall": fmt_float(dyn.get("Rall")),
                "wRobs": fmt_float(dyn.get("wRobs")),
                "wRall": fmt_float(dyn.get("wRall")),
                "MRobs": fmt_float(dyn.get("MRobs")),
                "MRall": fmt_float(dyn.get("MRall")),
                "MwRobs": fmt_float(dyn.get("MwRobs")),
                "MwRall": fmt_float(dyn.get("MwRall")),
            }
        )
    return rows


def markdown_table(rows: list[dict]) -> list[str]:
    """Return Markdown table lines for rows, with the first row's keys as header."""

    headers = list(rows[0].keys())
    lines = ["| " + " | ".join(headers) + " |", "|" + "|".join(["---"] * len(headers)) + "|"]
    for row in rows:
        lines.append("| " + " | ".join(str(row[h]) for h in headers) + " |")
    return lines


def tex_escape(text: str) -> str:
    """Escape LaTeX special characters in text."""

    repl = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(repl.get(ch, ch) for ch in str(text))


def summary_metadata(rows: list[dict], model: dict, run_summaries: list[dict]) -> dict:
    """Collect the report's crystal, resolution and dynamical-setup fields from the rows, model and run summaries."""

    dvals = np.asarray([safe_float(r.get("d_A")) for r in rows], dtype=float)
    finite_d = dvals[np.isfinite(dvals) & (dvals > 0)]
    cell = model.get("cell", [float("nan")] * 6)
    try:
        uc = gemmi.UnitCell(*[float(v) for v in cell])
        volume = float(uc.volume)
    except Exception:
        volume = float("nan")
    n_basis = []
    n_points = []
    thickness = []
    row_b = []
    min_peak = []
    geometry_percentile = []
    min_d_spacing = []
    sg_window = []
    topup_d_min = []
    max_support_sg = []
    group_width = []
    detector_bins = []
    frame_scale_counts = []
    detector_scale_counts = []
    thickness_model = []
    absorption_mode = []
    sg_jitter_mode = []
    matrix_convention = []
    for rec in run_summaries:
        cfg = rec.get("config", {})
        best_native = rec.get("best_native") or {}
        best = best_native.get("nuisance", {})
        scale_model = best_native.get("scale_model") if isinstance(best_native.get("scale_model"), dict) else {}
        if scale_model.get("kind") == "pspline":
            # Smooth frame scales are reported by their effective degrees of freedom, not the integer ceil(EDF).
            frame_scale_counts.append(
                f"P-spline EDF {fmt_float(safe_float(scale_model.get('edf')), 1)} "
                f"(n_basis {scale_model.get('n_basis')}, lambda {scale_model.get('lambda')})"
            )
        elif "frame_scale_count" in best_native:
            frame_scale_counts.append(safe_float(best_native.get("frame_scale_count")))
        if "detector_scale_count" in best_native:
            detector_scale_counts.append(
                safe_float(best_native.get("detector_scale_count"))
            )
        if "n_basis" in cfg:
            n_basis.append(str(cfg.get("n_basis")))
        if "n_points" in cfg:
            n_points.append(str(cfg.get("n_points")))
        for target, key in (
            (min_peak, "min_peak"),
            (geometry_percentile, "geometry_percentile"),
            (min_d_spacing, "min_d_spacing"),
            (sg_window, "sg_window"),
            (topup_d_min, "topup_d_min"),
            (max_support_sg, "max_support_sg"),
        ):
            if key in cfg:
                target.append(safe_float(cfg.get(key)))
        if "thickness_A" in best:
            thickness.append(safe_float(best.get("thickness_A")))
        if "row_resolution_b" in best:
            row_b.append(safe_float(best.get("row_resolution_b")))
        for target, key in (
            (group_width, "group_width_deg"),
            (detector_bins, "detector_scale_bins"),
            (thickness_model, "thickness_model"),
            (absorption_mode, "absorption_mode"),
            (sg_jitter_mode, "sg_jitter_mode"),
            (matrix_convention, "matrix_convention"),
        ):
            if key in best:
                target.append(str(best.get(key)))

    def uniq_text(values, digits=3):
        """Join the distinct values, numbers with the given decimals, or return 'n/a'."""

        if not values:
            return "n/a"
        out = []
        for value in values:
            text = value if isinstance(value, str) else fmt_float(value, digits)
            if text not in out:
                out.append(text)
        return ", ".join(out)

    return {
        "datasets": sorted(set(str(r.get("dataset", "")) for r in rows if r.get("dataset", ""))),
        "row_count": len(rows),
        "formula_asymmetric_unit": atom_formula(model),
        "lattice": model_lattice_label(model),
        "cell": cell,
        "volume": volume,
        "dmin_A": float(np.min(finite_d)) if len(finite_d) else float("nan"),
        "sin_theta_over_lambda_max": 0.5 / float(np.min(finite_d)) if len(finite_d) else float("nan"),
        "n_basis": ", ".join(sorted(set(n_basis), key=n_basis.index)) if n_basis else "n/a",
        "n_points": ", ".join(sorted(set(n_points), key=n_points.index)) if n_points else "n/a",
        "thickness_A_mean": float(np.mean(thickness)) if thickness else float("nan"),
        "row_resolution_b_values": ", ".join(fmt_float(v, 2) for v in row_b) if row_b else "n/a",
        "min_peak": uniq_text(min_peak, 1),
        "geometry_percentile": uniq_text(geometry_percentile, 2),
        "min_d_spacing": uniq_text(min_d_spacing, 3),
        "sg_window": uniq_text(sg_window, 5),
        "topup_d_min": uniq_text(topup_d_min, 3),
        "max_support_sg": uniq_text(max_support_sg, 5),
        "group_width_deg": uniq_text(group_width, 3),
        "detector_scale_bins": uniq_text(detector_bins, 0),
        "frame_scale_counts": uniq_text(frame_scale_counts, 0),
        "detector_scale_counts": uniq_text(detector_scale_counts, 0),
        "thickness_model": uniq_text(thickness_model),
        "absorption_mode": uniq_text(absorption_mode),
        "sg_jitter_mode": uniq_text(sg_jitter_mode),
        "matrix_convention": uniq_text(matrix_convention),
    }


def write_markdown(
    path: Path,
    title: str,
    meta: dict,
    row_mr_summary: dict,
    endpoint_summaries: list[dict] | None = None,
    *,
    canonical: bool = False,
) -> None:
    """Write the Markdown publication report: setup, row/MR factors and the row-contract summaries.

    canonical=True writes the JANA/PETS row-statistics tables.
    """

    cell = meta["cell"]
    row_rows = jana_table_rows(row_mr_summary) if canonical else table_rows(row_mr_summary)
    endpoint_rows = (
        row_contract_table_rows(endpoint_summaries or [])
        if canonical
        else endpoint_table_rows(endpoint_summaries or [])
    )
    row_heading = "## JANA/PETS-Style Row And MR Factors" if canonical else "## GROND Row/MR R And MR Factors"
    endpoint_heading = (
        "## Resolution And Scale Row-Contract Summary" if canonical else "## Resolution And Scale Endpoint Summary"
    )
    instability_u = safe_float((row_mr_summary.get("definition") or {}).get("instability_u"))
    instability_u_text = f"{instability_u:g}" if math.isfinite(instability_u) else "n/a"
    lines = [
        f"# {title}",
        "",
        "This table reports JANA/PETS-style row and post-refinement merged-row residual factors computed by GROND. `Robs`, `Rall`, and `wRall` are calculated on unmerged rows. `MRobs`, `MRall`, and `MwRall` are calculated after arithmetic merging of model-symmetry-related observations and predictions. These are not SHELXL endpoint statistics.",
        "",
        "## Measurement And Model",
        "",
        f"Datasets: `{', '.join(meta['datasets']) if meta['datasets'] else 'not supplied'}`.",
        f"Rows used for final row-space statistics: `{meta['row_count']}`.",
        f"Empirical formula from model asymmetric unit: `{meta['formula_asymmetric_unit']}`.",
        f"Symmetry source: `{meta['lattice']}`.",
        f"Unit cell: `a={fmt_float(cell[0], 4)} A, b={fmt_float(cell[1], 4)} A, c={fmt_float(cell[2], 4)} A, alpha={fmt_float(cell[3], 3)} deg, beta={fmt_float(cell[4], 3)} deg, gamma={fmt_float(cell[5], 3)} deg`.",
        f"Volume: `{fmt_float(meta['volume'], 2)} A^3`.",
        "",
        "## Dynamical Setup",
        "",
        f"Final N/Q: `{meta['n_basis']}/{meta['n_points']}`.",
        f"Geometry/reflection selection: `d >= {meta['min_d_spacing']} A`, `PEAK >= {meta['min_peak']}`, geometry percentile `{meta['geometry_percentile']}`.",
        f"Bloch support controls: `sg_window={meta['sg_window']} A^-1`, `max_support_sg={meta['max_support_sg']} A^-1`, `topup_d_min={meta['topup_d_min']} A`.",
        f"Mean fitted thickness: `{fmt_float(meta['thickness_A_mean'], 1)} A`.",
        f"Thickness model: `{meta['thickness_model']}`. Group width: `{meta['group_width_deg']} deg`. Detector scale bins: `{meta['detector_scale_bins']}`. Frame-scale parameters: `{meta['frame_scale_counts']}`; detector-scale parameters: `{meta['detector_scale_counts']}`.",
        f"Absorption mode: `{meta['absorption_mode']}`. Sg jitter mode: `{meta['sg_jitter_mode']}`. Matrix convention: `{meta['matrix_convention']}`.",
        f"Raw-row resolution-envelope B values: `{meta['row_resolution_b_values']}`.",
        f"dmin: `{fmt_float(meta['dmin_A'], 3)} A`; sin(theta_max)/lambda: `{fmt_float(meta['sin_theta_over_lambda_max'], 3)} A^-1`.",
        "",
        row_heading,
        "",
        *markdown_table(row_rows),
        "",
        f"Definitions: `Robs`, optional `wRobs`, `MRobs`, and optional `MwRobs` use only observed reflections satisfying `Iobs > 3 sigma(Iobs)`. `Rall`, `wRall`, `MRall`, and `MwRall` use all finite rows or merged rows. Born and Bloch use one common finite row support. `Nmerge excluded` counts supported rows whose HKL indices cannot enter symmetry merging. The conventional core family is `Robs/Rall/wRall/MRobs/MRall/MwRall`. The instability factor is `u = {instability_u_text}`. Weights use `w = [sigma(sqrt(Iobs))^2 + (u sqrt(Iobs))^2]^(-1/2)` and `wR = sqrt(sum((w (sqrt(Iobs) - sqrt(Icalc)))^2) / sum((w sqrt(Iobs))^2))`, the amplitude form of Jana2020's `wR(all)` for refinement on F; `MR`/`MwR` follow Klar et al. (2023). The intensity-residual form of earlier versions is kept in the JSON output as `wRobs_I`, `wRall_I`, `MwRobs_I`, and `MwRall_I`.",
        "",
    ]
    if endpoint_rows:
        lines.extend(
            [
                endpoint_heading,
                "",
                "Auto mode reports the rows at the requested resolution and at the suggested final resolution, with the full fitted scale curve. These are JANA/PETS-style row statistics, not SHELXL endpoint statistics.",
                "",
                *markdown_table(endpoint_rows),
                "",
            ]
        )
    path.write_text("\n".join(lines))


def write_latex(
    path: Path,
    title: str,
    meta: dict,
    row_mr_summary: dict,
    endpoint_summaries: list[dict] | None = None,
    *,
    canonical: bool = False,
) -> None:
    """Write the LaTeX publication report; canonical=True writes the JANA/PETS row-statistics tables."""

    rows = jana_table_rows(row_mr_summary) if canonical else table_rows(row_mr_summary)
    endpoint_rows = (
        row_contract_table_rows(endpoint_summaries or [])
        if canonical
        else endpoint_table_rows(endpoint_summaries or [])
    )
    instability_u = safe_float((row_mr_summary.get("definition") or {}).get("instability_u"))
    instability_u_text = f"{instability_u:g}" if math.isfinite(instability_u) else "n/a"
    cell = meta["cell"]
    body = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[margin=0.75in]{geometry}",
        r"\usepackage{booktabs}",
        r"\usepackage{array}",
        r"\usepackage[T1]{fontenc}",
        r"\begin{document}",
        rf"\section*{{{tex_escape(title)}}}",
        r"\noindent This table reports JANA/PETS-style row and post-refinement merged-row residual factors computed by GROND. Unmerged factors use final rows directly; MR factors use arithmetic means of model-symmetry-related observations and predictions. These are not SHELXL endpoint statistics.",
        "",
        r"\subsection*{Measurement and model}",
        r"\begin{tabular}{@{}ll@{}}\toprule",
        rf"Datasets & {tex_escape(', '.join(meta['datasets']) if meta['datasets'] else 'not supplied')} \\",
        rf"Rows & {meta['row_count']} \\",
        rf"Formula from model asymmetric unit & {tex_escape(meta['formula_asymmetric_unit'])} \\",
        rf"Symmetry source & {tex_escape(meta['lattice'])} \\",
        rf"Unit cell & $a={fmt_float(cell[0], 4)}$, $b={fmt_float(cell[1], 4)}$, $c={fmt_float(cell[2], 4)}$ \\",
        rf"Angles & $\alpha={fmt_float(cell[3], 3)}^\circ$, $\beta={fmt_float(cell[4], 3)}^\circ$, $\gamma={fmt_float(cell[5], 3)}^\circ$ \\",
        rf"Volume & {fmt_float(meta['volume'], 2)} \AA$^3$ \\",
        rf"$d_{{min}}$ & {fmt_float(meta['dmin_A'], 3)} \AA \\",
        rf"$\sin(\theta_{{max}})/\lambda$ & {fmt_float(meta['sin_theta_over_lambda_max'], 3)} \AA$^{{-1}}$ \\",
        r"\bottomrule\end{tabular}",
        "",
        r"\subsection*{Dynamical setup}",
        r"\begin{tabular}{@{}ll@{}}\toprule",
        rf"Final N/Q & {tex_escape(meta['n_basis'])}/{tex_escape(meta['n_points'])} \\",
        rf"Reflection selection & $d \ge {tex_escape(meta['min_d_spacing'])}$ \AA, PEAK $\ge {tex_escape(meta['min_peak'])}$, geometry percentile {tex_escape(meta['geometry_percentile'])} \\",
        rf"Bloch support controls & $S_g$ window {tex_escape(meta['sg_window'])} \AA$^{{-1}}$, max support $S_g$ {tex_escape(meta['max_support_sg'])} \AA$^{{-1}}$, top-up $d_{{min}}$ {tex_escape(meta['topup_d_min'])} \AA \\",
        rf"Mean fitted thickness & {fmt_float(meta['thickness_A_mean'], 1)} \AA \\",
        rf"Thickness/group/detector model & {tex_escape(meta['thickness_model'])}; group width {tex_escape(meta['group_width_deg'])} deg; detector bins {tex_escape(meta['detector_scale_bins'])} \\",
        rf"Scale parameters & frame {tex_escape(meta['frame_scale_counts'])}; detector {tex_escape(meta['detector_scale_counts'])} \\",
        rf"Absorption, jitter, matrix convention & {tex_escape(meta['absorption_mode'])}; {tex_escape(meta['sg_jitter_mode'])}; {tex_escape(meta['matrix_convention'])} \\",
        rf"Raw-row resolution-envelope B values & {tex_escape(meta['row_resolution_b_values'])} \\",
        r"\bottomrule\end{tabular}",
        "",
    ]
    if canonical:
        body.extend(
            [
                r"\subsection*{JANA/PETS-style row and MR factors}",
                r"\begin{tabular}{@{}lrrrrrrrrrrr@{}}\toprule",
                r"Row model & Nobs,Nall & Nmerge used,excluded & MNobs,MNall & Robs & Rall & wRobs & wRall & MRobs & MRall & MwRobs & MwRall \\ \midrule",
            ]
        )
        for row in rows:
            body.append(
                rf"{tex_escape(row['Row model'])} & {tex_escape(row['Nobs, Nall'])} & {tex_escape(row['Nmerge used, excluded'])} & {tex_escape(row['MNobs, MNall'])} & "
                rf"{row['Robs']} & {row['Rall']} & {row['wRobs']} & {row['wRall']} & {row['MRobs']} & {row['MRall']} & {row['MwRobs']} & {row['MwRall']} \\"
            )
    else:
        body.extend(
            [
                r"\subsection*{GROND row and MR factors}",
                r"\begin{tabular}{@{}lrrrrrrrr@{}}\toprule",
                r"Regularization & Nobs,Nall & MNobs,MNall & Robs & Rall & wRall & MRobs & MRall & MwRall \\ \midrule",
            ]
        )
        for row in rows:
            body.append(
                rf"{tex_escape(row['Regularization'])} & {tex_escape(row['Nobs, Nall'])} & {tex_escape(row['MNobs, MNall'])} & "
                rf"{row['Robs']} & {row['Rall']} & {row['wRall']} & {row['MRobs']} & {row['MRall']} & {row['MwRall']} \\"
            )
    body.extend(
        [
            r"\bottomrule\end{tabular}",
            "",
            r"\bigskip",
            rf"\noindent $R_{{obs}}$, optional $wR_{{obs}}$, $MR_{{obs}}$, and optional $MwR_{{obs}}$ use reflections satisfying $I_{{obs}} > 3\sigma(I_{{obs}})$. $R_{{all}}$, $wR_{{all}}$, $MR_{{all}}$, and $MwR_{{all}}$ use all finite rows or merged rows. Born and Bloch use one common finite row support; Nmerge excluded counts supported rows whose HKL indices cannot enter symmetry merging. The conventional core family is $R_{{obs}}/R_{{all}}/wR_{{all}}/MR_{{obs}}/MR_{{all}}/MwR_{{all}}$. The instability factor is $u={tex_escape(instability_u_text)}$. Weights are $w=[\sigma(\sqrt{{I_{{obs}}}})^2+(u\sqrt{{I_{{obs}}}})^2]^{{-1/2}}$ and $wR=\sqrt{{\sum (w(\sqrt{{I_{{obs}}}}-\sqrt{{I_{{calc}}}}))^2/\sum (w\sqrt{{I_{{obs}}}})^2}}$, the amplitude form of Jana2020's $wR(\mathrm{{all}})$ for refinement on $F$; $MR$ and $MwR$ follow Klar et al. (2023).",
        ]
    )
    if endpoint_rows:
        endpoint_subheading = (
            r"\subsection*{Resolution and scale row-contract summary}"
            if canonical
            else r"\subsection*{Resolution and scale endpoint summary}"
        )
        body.extend(
            [
                "",
                endpoint_subheading,
                r"\noindent Auto mode reports the rows at the requested resolution and at the suggested final resolution, with the full fitted scale curve. These are not SHELXL endpoint statistics.",
                "",
            ]
        )
        if canonical:
            body.extend(
                [
                    r"\begin{tabular}{@{}llrrrrrrrrrrrr@{}}\toprule",
                    r"Row contract & Status & $d_{min}$ & Scale params & Nobs,Nall & Nmerge used,excluded & Robs & Rall & wRobs & wRall & MRobs & MRall & MwRobs & MwRall \\ \midrule",
                ]
            )
            for row in endpoint_rows:
                body.append(
                    rf"{tex_escape(row['Row contract'])} & {tex_escape(row['Status'])} & {tex_escape(row['dmin A'])} & "
                    rf"{tex_escape(row['Scale params'])} & {tex_escape(row['Nobs, Nall'])} & {tex_escape(row['Nmerge used, excluded'])} & {row['Robs']} & {row['Rall']} & "
                    rf"{row['wRobs']} & {row['wRall']} & {row['MRobs']} & {row['MRall']} & {row['MwRobs']} & {row['MwRall']} \\"
                )
        else:
            body.extend(
                [
                    r"\begin{tabular}{@{}llrrrrrrrrr@{}}\toprule",
                    r"Endpoint & Status & $d_{min}$ & Scale params & Nobs,Nall & Robs & Rall & wRall & MRobs & MRall & MwRall \\ \midrule",
                ]
            )
            for row in endpoint_rows:
                body.append(
                    rf"{tex_escape(row['Endpoint'])} & {tex_escape(row['Status'])} & {tex_escape(row['dmin A'])} & "
                    rf"{tex_escape(row['Scale params'])} & {tex_escape(row['Nobs, Nall'])} & {row['Robs']} & {row['Rall']} & "
                    rf"{row['wRall']} & {row['MRobs']} & {row['MRall']} & {row['MwRall']} \\"
                )
        body.extend([r"\bottomrule\end{tabular}", ""])
    body.extend([r"\end{document}", ""])
    path.write_text("\n".join(body))



def main() -> None:
    """Compute row and MR factors for each --run and write the Markdown, LaTeX, CSV and JSON tables to --out-dir."""

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--run", action="append", default=[], help="Dataset run as name=/path/to/run_dir, or just /path/to/run_dir."
    )
    ap.add_argument(
        "--full-run",
        action="append",
        default=[],
        help="Matched full-scale control run as name=/path/to/run_dir for multi-dataset endpoint accounting.",
    )
    ap.add_argument(
        "--row-table",
        action="append",
        default=[],
        help="Row table as name=/path/to/native_dyn_calc.csv, or just a path.",
    )
    ap.add_argument("--res", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--title", default="GROND Publication Table")
    ap.add_argument(
        "--instability-u",
        type=float,
        default=0.01,
        help="JANA/PETS-style row-weight instability factor u used by wR and MwR.",
    )
    ap.add_argument(
        "--resolution-limit",
        action="append",
        default=[],
        help="Resolution-limit summary as name=/path/to/summary.json, or a resolution-limit directory.",
    )
    ap.add_argument(
        "--requested-min-d-spacing",
        type=float,
        default=0.0,
        help="Explicit reporting cutoff calculated from the shared --res cell; zero derives the cutoff from supplied rows.",
    )
    ap.add_argument(
        "--selected-scale-contract",
        default="",
        help="full for a multi-dataset run: also report the selected rows with every crystal's full fitted scales.",
    )
    args = ap.parse_args()
    if not math.isfinite(float(args.instability_u)) or args.instability_u < 0.0:
        ap.error("--instability-u must be finite and nonnegative")
    if not math.isfinite(float(args.requested_min_d_spacing)) or args.requested_min_d_spacing < 0.0:
        ap.error("--requested-min-d-spacing must be finite and nonnegative")
    if args.selected_scale_contract and args.selected_scale_contract != "full":
        ap.error("--selected-scale-contract must be full")

    out = Path(args.out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    model = parse_model(Path(args.res).expanduser())

    rows: list[dict] = []
    run_summaries: list[dict] = []
    for item in args.run:
        name, run_dir = parse_named_path(item)
        run_dir = run_dir.resolve()
        rows.extend(read_csv_rows(run_dir / "native_dyn_calc.csv", name))
        run_summaries.append(read_json(run_dir / "summary.json"))
    for item in args.row_table:
        name, row_table = parse_named_path(item)
        rows.extend(read_csv_rows(row_table.resolve(), name))

    full_rows: list[dict] = []
    for item in args.full_run:
        name, run_dir = parse_named_path(item)
        full_rows.extend(read_csv_rows(run_dir.resolve() / "native_dyn_calc.csv", name))

    # Rows marked dataset_role=completeness_only (or exclude) do not enter the GROND row/MR metrics.
    rows = [r for r in rows if str(r.get("dataset_role", "primary")) not in {"completeness_only", "exclude"}]
    full_rows = [r for r in full_rows if str(r.get("dataset_role", "primary")) not in {"completeness_only", "exclude"}]

    input_row_count = len(rows)
    full_input_row_count = len(full_rows)
    if args.requested_min_d_spacing > 0.0:
        rows = rows_at_or_below_resolution(rows, args.requested_min_d_spacing, model)
        full_rows = rows_at_or_below_resolution(full_rows, args.requested_min_d_spacing, model)

    if not rows:
        raise SystemExit("No row tables were supplied.")

    row_mr_summary = compute_row_mr_summary(rows, model, instability_u=args.instability_u)
    per_dataset_summary = compute_per_dataset_row_mr(rows, model, instability_u=args.instability_u)
    per_dataset_rows = per_dataset_table_rows(per_dataset_summary)
    meta = summary_metadata(rows, model, run_summaries)
    if args.requested_min_d_spacing > 0.0:
        shared_cell_dmin = dmin_for_rows(rows, model)
        meta["dmin_A"] = shared_cell_dmin
        meta["sin_theta_over_lambda_max"] = (
            0.5 / shared_cell_dmin if math.isfinite(shared_cell_dmin) and shared_cell_dmin > 0.0 else float("nan")
        )
    meta["reporting_resolution_contract"] = {
        "requested_min_d_spacing_A": float(args.requested_min_d_spacing),
        "cell_source": str(Path(args.res).expanduser().resolve()),
        "d_spacing_source": "shared reporting-model cell",
        "selected_input_rows": input_row_count,
        "selected_retained_rows": len(rows),
        "full_control_input_rows": full_input_row_count,
        "full_control_retained_rows": len(full_rows),
    }
    resolution_payloads = load_resolution_limit_payloads(args.resolution_limit)
    endpoint_summaries, endpoint_metadata = endpoint_summary_rows(
        rows,
        model,
        resolution_payloads,
        instability_u=args.instability_u,
        full_rows=full_rows,
        requested_min_d_spacing=args.requested_min_d_spacing,
        selected_scale_contract=args.selected_scale_contract,
    )
    payload = {
        "schema": "grond.jana_row_statistics.v1",
        "numeric_units": "fractions",
        "title": args.title,
        "model": str(Path(args.res).expanduser().resolve()),
        "reporting_contract": {
            "statistics_family": "JANA/PETS-style row and post-refinement merged-row residual factors",
            "calculated_by": "GROND",
            "not_endpoint_statistics": True,
            "shelxl_endpoint_statistics_location": "run-root TABLE1_FINAL_STATISTICS.md and authoritative endpoint .lst",
            "compatibility_note": "The historical resolution_scale_endpoints key and publication_table filenames are retained for readers of older GROND outputs.",
        },
        "metadata": meta,
        "row_mr": row_mr_summary,
        "per_dataset_row_mr": per_dataset_summary,
        "row_resolution_scale_contracts": {
            "definition": "JANA/PETS-style row statistics at the requested and the promoted suggested resolution, with full fitted scales",
            "metadata": endpoint_metadata,
            "contracts": endpoint_summaries,
            "table_rows": row_contract_table_rows(endpoint_summaries),
        },
        "resolution_scale_endpoints": {
            "definition": "backward-compatible alias for row_resolution_scale_contracts; contains row/MR statistics, not SHELXL endpoint statistics",
            "metadata": endpoint_metadata,
            "endpoints": endpoint_summaries,
            "table_rows": endpoint_table_rows(endpoint_summaries),
        },
    }
    json_path = out / "row_mr_metrics.json"
    md_path = out / "publication_table.md"
    tex_path = out / "publication_table.tex"
    json_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    write_markdown(md_path, args.title, meta, row_mr_summary, endpoint_summaries)
    write_latex(tex_path, args.title, meta, row_mr_summary, endpoint_summaries)
    canonical_json_path = out / "JANA_ROW_STATISTICS.json"
    canonical_md_path = out / "JANA_ROW_STATISTICS.md"
    canonical_tex_path = out / "JANA_ROW_STATISTICS.tex"
    canonical_csv_path = out / "JANA_ROW_STATISTICS.csv"
    per_dataset_json_path = out / "JANA_ROW_STATISTICS_PER_DATASET.json"
    per_dataset_md_path = out / "JANA_ROW_STATISTICS_PER_DATASET.md"
    per_dataset_csv_path = out / "JANA_ROW_STATISTICS_PER_DATASET.csv"
    canonical_json_path.write_text(json.dumps(standard_json_data(payload), indent=2, allow_nan=False) + "\n")
    write_markdown(
        canonical_md_path,
        args.title,
        meta,
        row_mr_summary,
        endpoint_summaries,
        canonical=True,
    )
    write_latex(
        canonical_tex_path,
        args.title,
        meta,
        row_mr_summary,
        endpoint_summaries,
        canonical=True,
    )
    with canonical_csv_path.open("w", newline="") as stream:
        combined_rows = jana_table_rows(row_mr_summary)
        writer = csv.DictWriter(stream, fieldnames=list(combined_rows[0].keys()))
        writer.writeheader()
        writer.writerows(combined_rows)
    per_dataset_json_path.write_text(
        json.dumps(
            standard_json_data(
                {
                    "schema": "grond.jana_row_statistics_per_dataset.v1",
                    "numeric_units": "fractions",
                    "definition": row_mr_summary["definition"],
                    "datasets": per_dataset_summary,
                    "table_rows": per_dataset_rows,
                }
            ),
            indent=2,
            allow_nan=False,
        )
        + "\n"
    )
    per_dataset_md_path.write_text(
        "\n".join(
            [
                f"# {args.title}: Per-Dataset JANA/PETS-Style Row Statistics",
                "",
                f"Residual factors are fractions (`0.146` means `14.6%`). Each dataset uses the same matched-support JANA/PETS-style GROND implementation with row-weight instability factor `u={args.instability_u:g}` and keeps unmerged row factors separate from post-refinement merged factors. `Nmerge excluded` surfaces supported rows whose HKL indices cannot enter symmetry merging.",
                "",
                *markdown_table(per_dataset_rows),
                "",
            ]
        )
    )
    with per_dataset_csv_path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(per_dataset_rows[0].keys()))
        writer.writeheader()
        writer.writerows(per_dataset_rows)
    result = {
        "summary": str(json_path.resolve()),
        "markdown": str(md_path.resolve()),
        "latex": str(tex_path.resolve()),
        "canonical_jana_json": str(canonical_json_path.resolve()),
        "canonical_jana_markdown": str(canonical_md_path.resolve()),
        "canonical_jana_latex": str(canonical_tex_path.resolve()),
        "canonical_jana_csv": str(canonical_csv_path.resolve()),
        "per_dataset_jana_json": str(per_dataset_json_path.resolve()),
        "per_dataset_jana_markdown": str(per_dataset_md_path.resolve()),
        "per_dataset_jana_csv": str(per_dataset_csv_path.resolve()),
        "dynamic": row_mr_summary["dynamic"],
        "born": row_mr_summary["born"],
        "kinematic": row_mr_summary["kinematic"],
        "per_dataset": per_dataset_summary,
        "row_resolution_scale_contracts": payload["row_resolution_scale_contracts"],
        "resolution_scale_endpoints": payload["resolution_scale_endpoints"],
    }
    (out / "publication_artifacts.json").write_text(json.dumps(result, indent=2, allow_nan=True) + "\n")
    print(json.dumps(result, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
