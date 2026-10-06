#!/usr/bin/env python3
"""Pre-merge row-space R-factor forensics for native Bloch tables.

This audit stays upstream of SHELXL/HKLF merging.  It asks whether high all-row
R factors are driven by weak rows, resolution, equivalent-row scatter, missed
row-specific modulation, or a small set of pathological observations.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__:
    from .bloch_engine import XDSData
    from .native_row_regularize import DEFAULT_HKL, DEFAULT_RES, native_metrics, parse_wght, shelx_weights
else:  # pragma: no cover - direct script execution
    from bloch_engine import XDSData
    from native_row_regularize import DEFAULT_HKL, DEFAULT_RES, native_metrics, parse_wght, shelx_weights


def read_rows(path: Path) -> list[dict[str, str]]:
    """Read a CSV file into a list of row dicts; raise if it has no rows."""

    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows read from {path}")
    return rows


def fcol(rows: list[dict[str, str]], name: str, default=float("nan")) -> np.ndarray:
    """Return one column as a float array, with default for missing or unparsable cells."""

    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, default)))
        except (TypeError, ValueError):
            out.append(float(default))
    return np.asarray(out, dtype=float)


def icol(rows: list[dict[str, str]], name: str, default=-1) -> np.ndarray:
    """Return one column as an int array, with default for missing or unparsable cells."""

    out = []
    for r in rows:
        try:
            out.append(int(float(r.get(name, default))))
        except (TypeError, ValueError):
            out.append(int(default))
    return np.asarray(out, dtype=int)


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV file with the first row's keys as header; an empty list writes an empty file."""

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def pct(v: float) -> float:
    """Return a fraction as a percentage, or NaN if it is not finite."""

    return 100.0 * float(v) if np.isfinite(v) else float("nan")


def safe_metric(obs, sig, pred, mask, wght_a, wght_b, n_params=0) -> dict:
    """Return the core native metrics (n, n_gt, R1, R2, wR2, GooF, CC) for the rows in mask."""

    out = native_metrics(obs, sig, pred, mask, wght_a, wght_b, n_params=n_params)
    return {
        "n": out["n"],
        "n_gt": out["n_gt"],
        "R1_all": out["R1_all"],
        "R1_gt": out["R1_gt"],
        "R2_all": out["R2_all"],
        "wR2_all": out["wR2_all"],
        "GooF": out.get("GooF", float("nan")),
        "CC": out["CC"],
    }


def p222_equivs(h: int, k: int, l: int) -> list[tuple[int, int, int]]:
    """Return the four P222 rotation mates of (h, k, l)."""

    return [(h, k, l), (h, -k, -l), (-h, k, -l), (-h, -k, l)]


def canonical_key(h: int, k: int, l: int, mode: str) -> tuple[int, int, int]:
    """Return one HKL's group key: itself (exact) or its smallest P222 mate, optionally with Friedel mates."""

    if mode == "exact":
        return (int(h), int(k), int(l))
    equivs = p222_equivs(int(h), int(k), int(l))
    if mode == "p222_friedel":
        equivs = equivs + [(-a, -b, -c) for a, b, c in equivs]
    elif mode != "p222":
        raise ValueError(f"unknown group mode {mode!r}")
    return min(equivs)


def group_indices(h, k, l, mode: str, valid: np.ndarray) -> dict[tuple[int, int, int], np.ndarray]:
    """Group the valid rows by canonical_key and return {key: row indices}."""

    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for i, ok in enumerate(valid):
        if ok:
            groups[canonical_key(int(h[i]), int(k[i]), int(l[i]), mode)].append(i)
    return {key: np.asarray(idx, dtype=int) for key, idx in groups.items()}


def weighted_group_mean(obs, sig, groups: dict[tuple[int, int, int], np.ndarray]) -> np.ndarray:
    """Return each row's inverse-variance weighted group mean of Iobs, floored at zero."""

    out = np.full_like(obs, np.nan, dtype=float)
    for idx in groups.values():
        s = np.maximum(sig[idx], 1e-12)
        w = 1.0 / np.maximum(s * s, 1e-30)
        mean = float(np.sum(w * obs[idx]) / max(np.sum(w), 1e-30))
        out[idx] = max(mean, 0.0)
    return out


def simple_group_mean(values, groups: dict[tuple[int, int, int], np.ndarray]) -> np.ndarray:
    """Return each row's plain group mean of the finite values."""

    out = np.full_like(values, np.nan, dtype=float)
    for idx in groups.values():
        good = np.isfinite(values[idx])
        if np.any(good):
            out[idx] = float(np.nanmean(values[idx][good]))
    return out


def group_summary(
    mode: str,
    obs,
    sig,
    pred,
    born,
    d,
    isigi,
    q,
    x,
    y,
    z,
    radius,
    h,
    k,
    l,
    groups: dict[tuple[int, int, int], np.ndarray],
    valid,
    wght_a,
    wght_b,
) -> tuple[dict, np.ndarray, list[dict]]:
    """Compare the model with the scalar group floor for one grouping; return (summary, multiplicity, group rows).

    The floor predicts every row by its group's weighted mean Iobs, the best any merged scalar can do. The
    summary holds both metric sets, the residual share above the floor and the within-group correlation of
    observed and predicted variation.
    """

    floor_pred = weighted_group_mean(obs, sig, groups)
    pred_mean = simple_group_mean(pred, groups)
    dyn = safe_metric(obs, sig, pred, valid, wght_a, wght_b)
    floor = safe_metric(obs, sig, floor_pred, valid & np.isfinite(floor_pred), wght_a, wght_b)
    grouped_pred = safe_metric(obs, sig, pred_mean, valid & np.isfinite(pred_mean), wght_a, wght_b)
    born_metrics = safe_metric(obs, sig, born, valid & np.isfinite(born), wght_a, wght_b) if born is not None else {}

    multi = np.zeros(len(obs), dtype=bool)
    mult = np.zeros(len(obs), dtype=int)
    for idx in groups.values():
        mult[idx] = len(idx)
        if len(idx) > 1:
            multi[idx] = True

    within_obs = []
    within_pred = []
    group_rows = []
    total_abs = 0.0
    total_floor_abs = 0.0
    total_excess_abs = 0.0
    total_weighted = 0.0
    total_floor_weighted = 0.0
    for key, idx in groups.items():
        obs_g = obs[idx]
        sig_g = np.maximum(sig[idx], 1e-12)
        pred_g = pred[idx]
        floor_g = floor_pred[idx]
        wg = shelx_weights(obs_g, sig_g, pred_g, wght_a, wght_b)
        abs_model = float(np.nansum(np.abs(obs_g - pred_g)))
        abs_floor = float(np.nansum(np.abs(obs_g - floor_g)))
        weighted_model = float(np.nansum(wg * (obs_g - pred_g) ** 2))
        weighted_floor = float(np.nansum(shelx_weights(obs_g, sig_g, floor_g, wght_a, wght_b) * (obs_g - floor_g) ** 2))
        total_abs += abs_model
        total_floor_abs += abs_floor
        total_excess_abs += max(abs_model - abs_floor, 0.0)
        total_weighted += weighted_model
        total_floor_weighted += weighted_floor
        if len(idx) > 1:
            om = obs_g - np.nanmean(obs_g)
            pm = pred_g - np.nanmean(pred_g)
            within_obs.extend(om.tolist())
            within_pred.extend(pm.tolist())
        residual = obs_g - pred_g
        q_g = q[idx] if q is not None else np.full(len(idx), np.nan)
        group_rows.append(
            {
                "mode": mode,
                "h": key[0],
                "k": key[1],
                "l": key[2],
                "n": int(len(idx)),
                "d_median": float(np.nanmedian(d[idx])),
                "I_over_sigma_median": float(np.nanmedian(isigi[idx])),
                "obs_mean": float(np.nanmean(obs_g)),
                "pred_mean": float(np.nanmean(pred_g)),
                "obs_pred_mean_ratio": float(np.nanmean(obs_g) / max(np.nanmean(pred_g), 1e-30)),
                "signed_residual_median": float(np.nanmedian(residual)),
                "abs_model_residual_sum": abs_model,
                "abs_floor_residual_sum": abs_floor,
                "excess_abs_residual_sum": max(abs_model - abs_floor, 0.0),
                "weighted_model_residual_sum": weighted_model,
                "weighted_floor_residual_sum": weighted_floor,
                "excess_weighted_residual_sum": max(weighted_model - weighted_floor, 0.0),
                "floor_abs_residual_fraction": float(abs_floor / max(abs_model, 1e-30)),
                "q_min": float(np.nanmin(q_g)) if np.any(np.isfinite(q_g)) else float("nan"),
                "q_median": float(np.nanmedian(q_g)) if np.any(np.isfinite(q_g)) else float("nan"),
                "q_max": float(np.nanmax(q_g)) if np.any(np.isfinite(q_g)) else float("nan"),
                "frame_span": float(np.nanmax(z[idx]) - np.nanmin(z[idx])) if np.count_nonzero(np.isfinite(z[idx])) > 1 else 0.0,
                "detector_radius_median": float(np.nanmedian(radius[idx])) if np.any(np.isfinite(radius[idx])) else float("nan"),
                "x_span": float(np.nanmax(x[idx]) - np.nanmin(x[idx])) if np.count_nonzero(np.isfinite(x[idx])) > 1 else 0.0,
                "y_span": float(np.nanmax(y[idx]) - np.nanmin(y[idx])) if np.count_nonzero(np.isfinite(y[idx])) > 1 else 0.0,
                "first_row": int(idx[0]),
                "first_h": int(h[idx[0]]),
                "first_k": int(k[idx[0]]),
                "first_l": int(l[idx[0]]),
            }
        )

    within_obs_arr = np.asarray(within_obs, dtype=float)
    within_pred_arr = np.asarray(within_pred, dtype=float)
    m = np.isfinite(within_obs_arr) & np.isfinite(within_pred_arr)
    corr = float("nan")
    beta = float("nan")
    variance_explained = float("nan")
    if np.count_nonzero(m) > 3 and np.std(within_obs_arr[m]) > 0 and np.std(within_pred_arr[m]) > 0:
        corr = float(np.corrcoef(within_obs_arr[m], within_pred_arr[m])[0, 1])
        beta = float(np.sum(within_obs_arr[m] * within_pred_arr[m]) / max(np.sum(within_pred_arr[m] ** 2), 1e-30))
        resid = within_obs_arr[m] - beta * within_pred_arr[m]
        variance_explained = float(1.0 - np.sum(resid * resid) / max(np.sum((within_obs_arr[m] - np.mean(within_obs_arr[m])) ** 2), 1e-30))

    summary = {
        "mode": mode,
        "groups": int(len(groups)),
        "multi_groups": int(sum(1 for idx in groups.values() if len(idx) > 1)),
        "multi_rows": int(np.count_nonzero(multi)),
        "max_multiplicity": int(max((len(idx) for idx in groups.values()), default=0)),
        "dyn": dyn,
        "born": born_metrics,
        "group_floor": floor,
        "group_mean_prediction": grouped_pred,
        "floor_abs_fraction_of_model_abs": float(total_floor_abs / max(total_abs, 1e-30)),
        "excess_abs_fraction_of_model_abs": float(total_excess_abs / max(total_abs, 1e-30)),
        "floor_weighted_fraction_of_model_weighted": float(total_floor_weighted / max(total_weighted, 1e-30)),
        "within_equiv_obs_pred_corr": corr,
        "within_equiv_obs_per_pred_beta": beta,
        "within_equiv_variance_explained": variance_explained,
    }
    return summary, mult, group_rows


def quantile_bins(values: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Split the finite values into n_bins equal-count bins of row indices, in increasing order."""

    good = np.where(np.isfinite(values))[0]
    if not len(good):
        return []
    order = good[np.argsort(values[good], kind="mergesort")]
    return [b for b in np.array_split(order, max(1, int(n_bins))) if len(b)]


def axis_bins(axis_name, values, obs, sig, pred, floor_pred, valid, wght_a, wght_b, n_bins) -> list[dict]:
    """Return model and group-floor metrics in quantile bins of one row axis."""

    rows = []
    for bi, idx in enumerate(quantile_bins(values, n_bins), start=1):
        mask = np.zeros(len(obs), dtype=bool)
        mask[idx] = True
        mask &= valid
        if not np.any(mask):
            continue
        model = safe_metric(obs, sig, pred, mask, wght_a, wght_b)
        floor = safe_metric(obs, sig, floor_pred, mask & np.isfinite(floor_pred), wght_a, wght_b)
        rows.append(
            {
                "axis": axis_name,
                "bin": bi,
                "n": model["n"],
                "n_gt": model["n_gt"],
                "axis_min": float(np.nanmin(values[mask])),
                "axis_median": float(np.nanmedian(values[mask])),
                "axis_max": float(np.nanmax(values[mask])),
                "R1_all": model["R1_all"],
                "R2_all": model["R2_all"],
                "wR2_all": model["wR2_all"],
                "floor_R1_all": floor["R1_all"],
                "floor_R2_all": floor["R2_all"],
                "floor_wR2_all": floor["wR2_all"],
                "R2_excess_over_floor": model["R2_all"] - floor["R2_all"],
                "wR2_excess_over_floor": model["wR2_all"] - floor["wR2_all"],
                "CC": model["CC"],
            }
        )
    return rows


def hkl_class(h: int, k: int, l: int) -> str:
    """Label one HKL as the origin, an axial row (odd or even index), a zone plane or general."""

    hz = int(h) == 0
    kz = int(k) == 0
    lz = int(l) == 0
    if hz and kz and lz:
        return "000"
    if kz and lz:
        absent = abs(int(h)) % 2 == 1
        return "h00_absent_odd" if absent else "h00_allowed_even"
    if hz and lz:
        absent = abs(int(k)) % 2 == 1
        return "0k0_absent_odd" if absent else "0k0_allowed_even"
    if hz and kz:
        absent = abs(int(l)) % 2 == 1
        return "00l_absent_odd" if absent else "00l_allowed_even"
    if hz:
        return "0kl_plane"
    if kz:
        return "h0l_plane"
    if lz:
        return "hk0_plane"
    return "general"


def hkl_class_bins(rows, h, k, l, obs, sig, pred, valid, wght_a, wght_b) -> list[dict]:
    """Return metrics and residual shares for each HKL class, most residual-enriched first."""

    classes = np.asarray([hkl_class(a, b, c) for a, b, c in zip(h, k, l, strict=False)])
    total_abs = np.nansum(np.abs(obs[valid] - pred[valid]))
    total_weighted = np.nansum(shelx_weights(obs[valid], sig[valid], pred[valid], wght_a, wght_b) * (obs[valid] - pred[valid]) ** 2)
    out = []
    for cls in sorted(set(classes[valid])):
        mask = valid & (classes == cls)
        m = safe_metric(obs, sig, pred, mask, wght_a, wght_b)
        residual = obs[mask] - pred[mask]
        weighted = shelx_weights(obs[mask], sig[mask], pred[mask], wght_a, wght_b) * residual * residual
        row_fraction = float(m["n"] / max(np.count_nonzero(valid), 1))
        abs_fraction = float(np.nansum(np.abs(residual)) / max(total_abs, 1e-30))
        weighted_fraction = float(np.nansum(weighted) / max(total_weighted, 1e-30))
        q = fcol([rows[i] for i in np.where(mask)[0]], "q_dyn_over_born") if "q_dyn_over_born" in rows[0] and np.any(mask) else np.asarray([], dtype=float)
        out.append(
            {
                "hkl_class": cls,
                "n": m["n"],
                "n_gt": m["n_gt"],
                "row_fraction": row_fraction,
                "R1_all": m["R1_all"],
                "R1_gt": m["R1_gt"],
                "R2_all": m["R2_all"],
                "wR2_all": m["wR2_all"],
                "CC": m["CC"],
                "abs_residual_fraction": abs_fraction,
                "weighted_residual_fraction": weighted_fraction,
                "abs_residual_enrichment": float(abs_fraction / max(row_fraction, 1e-30)),
                "weighted_residual_enrichment": float(weighted_fraction / max(row_fraction, 1e-30)),
                "median_residual": float(np.nanmedian(residual)) if len(residual) else float("nan"),
                "median_abs_residual": float(np.nanmedian(np.abs(residual))) if len(residual) else float("nan"),
                "q_median": float(np.nanmedian(q[np.isfinite(q)])) if np.count_nonzero(np.isfinite(q)) else float("nan"),
            }
        )
    return sorted(out, key=lambda r: (r["weighted_residual_enrichment"], r["abs_residual_enrichment"]), reverse=True)


def hkl_class_equiv_modulation(h, k, l, obs, sig, pred, valid, groups: dict[tuple[int, int, int], np.ndarray]) -> list[dict]:
    """Return, for each HKL class, how well the model's within-group variation tracks the observed one."""

    classes = np.asarray([hkl_class(a, b, c) for a, b, c in zip(h, k, l, strict=False)])
    out = []
    for cls in sorted(set(classes[valid])):
        obs_dev = []
        pred_dev = []
        group_count = 0
        row_count = 0
        model_abs = 0.0
        scalar_abs = 0.0
        excess_abs = 0.0
        obs_abs_dev = []
        pred_abs_dev = []
        for idx in groups.values():
            idx = idx[classes[idx] == cls]
            if len(idx) <= 1:
                continue
            group_count += 1
            row_count += int(len(idx))
            obs_g = obs[idx]
            sig_g = np.maximum(sig[idx], 1e-12)
            pred_g = pred[idx]
            weights = 1.0 / np.maximum(sig_g * sig_g, 1e-30)
            obs_mean = float(np.sum(weights * obs_g) / max(np.sum(weights), 1e-30))
            pred_mean = float(np.nanmean(pred_g))
            obs_centered = obs_g - float(np.nanmean(obs_g))
            pred_centered = pred_g - pred_mean
            obs_dev.extend(obs_centered.tolist())
            pred_dev.extend(pred_centered.tolist())
            obs_abs_dev.extend(np.abs(obs_centered).tolist())
            pred_abs_dev.extend(np.abs(pred_centered).tolist())
            model = float(np.nansum(np.abs(obs_g - pred_g)))
            scalar = float(np.nansum(np.abs(obs_g - obs_mean)))
            model_abs += model
            scalar_abs += scalar
            excess_abs += max(model - scalar, 0.0)

        obs_arr = np.asarray(obs_dev, dtype=float)
        pred_arr = np.asarray(pred_dev, dtype=float)
        m = np.isfinite(obs_arr) & np.isfinite(pred_arr)
        corr = float("nan")
        beta = float("nan")
        variance_explained = float("nan")
        if np.count_nonzero(m) > 3 and np.std(obs_arr[m]) > 0 and np.std(pred_arr[m]) > 0:
            corr = float(np.corrcoef(obs_arr[m], pred_arr[m])[0, 1])
            beta = float(np.sum(obs_arr[m] * pred_arr[m]) / max(np.sum(pred_arr[m] ** 2), 1e-30))
            resid = obs_arr[m] - beta * pred_arr[m]
            variance_explained = float(1.0 - np.sum(resid * resid) / max(np.sum((obs_arr[m] - np.mean(obs_arr[m])) ** 2), 1e-30))
        out.append(
            {
                "hkl_class": cls,
                "groups": group_count,
                "rows": row_count,
                "within_equiv_obs_pred_corr": corr,
                "within_equiv_obs_per_pred_beta": beta,
                "within_equiv_variance_explained": variance_explained,
                "model_abs_residual_sum": model_abs,
                "scalar_equiv_abs_residual_sum": scalar_abs,
                "model_to_scalar_abs_ratio": float(model_abs / max(scalar_abs, 1e-30)),
                "excess_abs_residual_sum": excess_abs,
                "median_abs_obs_deviation": float(np.nanmedian(obs_abs_dev)) if obs_abs_dev else float("nan"),
                "median_abs_pred_deviation": float(np.nanmedian(pred_abs_dev)) if pred_abs_dev else float("nan"),
            }
        )
    return sorted(
        out,
        key=lambda r: (
            0 if np.isfinite(r["within_equiv_obs_pred_corr"]) else 1,
            r["within_equiv_obs_pred_corr"] if np.isfinite(r["within_equiv_obs_pred_corr"]) else 999.0,
            -r["model_to_scalar_abs_ratio"],
        ),
    )


def trim_curve(obs, sig, pred, floor_pred, valid, wght_a, wght_b) -> list[dict]:
    """Return metrics after dropping the top 0 to 5% of rows by residual excess over the group floor."""

    abs_model = np.abs(obs - pred)
    abs_floor = np.abs(obs - floor_pred)
    excess = np.maximum(abs_model - abs_floor, 0.0)
    order = np.where(valid & np.isfinite(excess))[0][np.argsort(excess[valid & np.isfinite(excess)])[::-1]]
    rows = []
    total = np.count_nonzero(valid)
    for frac in [0.0, 0.0025, 0.005, 0.01, 0.02, 0.05]:
        drop_n = int(round(total * frac))
        keep = valid.copy()
        if drop_n:
            keep[order[:drop_n]] = False
        m = safe_metric(obs, sig, pred, keep, wght_a, wght_b)
        rows.append(
            {
                "drop_top_excess_fraction": frac,
                "dropped_rows": drop_n,
                "kept_rows": m["n"],
                "R1_all": m["R1_all"],
                "R2_all": m["R2_all"],
                "wR2_all": m["wR2_all"],
                "R1_gt": m["R1_gt"],
                "CC": m["CC"],
            }
        )
    return rows


def add_xds_axes(rows: list[dict[str, str]], hkl_path: Path):
    """Return XDS xcal, ycal, zcal and detector radius (px from the median) per row; NaN without the HKL file."""

    n = len(rows)
    x = np.full(n, np.nan, dtype=float)
    y = np.full(n, np.nan, dtype=float)
    z = np.full(n, np.nan, dtype=float)
    if not hkl_path or not hkl_path.exists():
        return x, y, z, np.full(n, np.nan, dtype=float)
    row_index = icol(rows, "row_index")
    data = XDSData(hkl_path)
    ok = (row_index >= 0) & (row_index < len(data.hkl))
    x[ok] = data.xcal[row_index[ok]]
    y[ok] = data.ycal[row_index[ok]]
    z[ok] = data.zcal[row_index[ok]]
    radius = np.sqrt((x - np.nanmedian(x)) ** 2 + (y - np.nanmedian(y)) ** 2)
    return x, y, z, radius


def verdict(summary: dict, primary_mode: str) -> list[str]:
    """Return plain-language findings from the primary grouping's summary."""

    s = summary["group_modes"][primary_mode]
    dyn = s["dyn"]
    scalar = s["group_floor"]
    out = []
    if dyn["R1_all"] - dyn["R1_gt"] > 0.04:
        out.append("all-row penalty is mainly outside the strong set; prioritize weak-row and high-resolution treatment before larger Bloch bases")
    if scalar["R2_all"] > dyn["R2_all"] + 0.02:
        out.append("row-specific Bloch predictions beat scalar symmetry-equivalent averaging; keep row-level modeling before any SHELX-style merge")
    elif s["excess_abs_fraction_of_model_abs"] > 0.5:
        out.append("model residual is well above scalar equivalent-row collapse; there is systematic row-space error left to model")
    else:
        out.append("scalar equivalent-row collapse explains much of the residual; do not overfit this with more physical knobs")
    corr = s.get("within_equiv_obs_pred_corr", float("nan"))
    if np.isfinite(corr) and corr >= 0.7:
        out.append("within-equivalent row modulation is strongly predicted by Bloch; the next fixes should target outlier/background/frame terms rather than pre-merging")
    elif np.isfinite(corr) and corr < 0.2:
        out.append("row-specific Bloch modulation poorly explains within-equivalent intensity scatter; investigate geometry/background/frame nuisance terms")
    if scalar["R2_all"] < dyn["R2_all"] * 0.5:
        out.append("a large gap to the pre-merge group floor remains; held-out additive/background correction is a plausible high-return fix")
    return out


def modulation_verdict(class_mod_rows: list[dict]) -> list[str]:
    """Return findings for HKL classes whose within-group modulation is anti-correlated with the model."""

    out = []
    for r in class_mod_rows:
        corr = r.get("within_equiv_obs_pred_corr", float("nan"))
        ratio = r.get("model_to_scalar_abs_ratio", float("nan"))
        rows = int(r.get("rows", 0))
        if rows >= 8 and np.isfinite(corr) and corr < -0.3 and np.isfinite(ratio) and ratio > 1.5:
            out.append(
                f"{r['hkl_class']} equivalent-row modulation is anti-correlated with Bloch prediction; test axial/sign-specific geometry, absorption, or Bloch-index convention before adding global beams"
            )
    return out


def main() -> None:
    """Run the pre-merge R-factor audit on a native row table.

    Writes the group, axis-bin, HKL-class and trim CSVs, summary.json and REPORT.md to --out-dir.
    """

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL))
    ap.add_argument("--res", default=str(DEFAULT_RES))
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--obs-column", default="target_I")
    ap.add_argument("--sigma-column", default="sigma_eff")
    ap.add_argument("--pred-column", default="dyn_scaled")
    ap.add_argument("--born-column", default="born_scaled")
    ap.add_argument("--group-modes", default="exact,p222,p222_friedel")
    ap.add_argument("--primary-group-mode", choices=["exact", "p222", "p222_friedel"], default="p222_friedel")
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--top-groups", type=int, default=100)
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.row_table))
    obs = fcol(rows, args.obs_column)
    sig = np.maximum(fcol(rows, args.sigma_column), 1e-12)
    pred = fcol(rows, args.pred_column)
    born = fcol(rows, args.born_column) if args.born_column in rows[0] else None
    d = fcol(rows, "d_A")
    q = fcol(rows, "q_dyn_over_born") if "q_dyn_over_born" in rows[0] else None
    h = icol(rows, "h")
    k = icol(rows, "k")
    l = icol(rows, "l")
    isigi = obs / np.maximum(sig, 1e-12)
    x, y, z, radius = add_xds_axes(rows, Path(args.hkl))
    wght_a, wght_b = parse_wght(Path(args.res))
    valid = np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & np.isfinite(d) & (sig > 0)

    group_modes = [m.strip() for m in args.group_modes.split(",") if m.strip()]
    group_summaries = {}
    group_rows_all = []
    multiplicities = {}
    floor_predictions = {}
    groups_by_mode = {}
    for mode in group_modes:
        groups = group_indices(h, k, l, mode, valid)
        groups_by_mode[mode] = groups
        summary, mult, group_rows = group_summary(mode, obs, sig, pred, born, d, isigi, q, x, y, z, radius, h, k, l, groups, valid, wght_a, wght_b)
        group_summaries[mode] = summary
        multiplicities[mode] = mult
        floor_predictions[mode] = weighted_group_mean(obs, sig, groups)
        group_rows_all.extend(group_rows)
        ranked = sorted(group_rows, key=lambda r: (r["excess_weighted_residual_sum"], r["excess_abs_residual_sum"]), reverse=True)
        write_csv(out / f"top_{mode}_groups.csv", ranked[: max(0, int(args.top_groups))])

    primary_floor = floor_predictions[args.primary_group_mode]
    axis_rows = []
    axes = {
        "resolution_d_A": d,
        "inverse_d2": 1.0 / np.maximum(d, 1e-12) ** 2,
        "I_over_sigma": isigi,
        "Iobs": obs,
        "q_dyn_over_born": q if q is not None else np.full(len(rows), np.nan),
        "frame_zcal": z,
        "detector_radius": radius,
        f"multiplicity_{args.primary_group_mode}": multiplicities[args.primary_group_mode].astype(float),
    }
    for name, values in axes.items():
        axis_rows.extend(axis_bins(name, values, obs, sig, pred, primary_floor, valid, wght_a, wght_b, args.bins))
    write_csv(out / "premerge_axis_bins.csv", axis_rows)
    class_rows = hkl_class_bins(rows, h, k, l, obs, sig, pred, valid, wght_a, wght_b)
    write_csv(out / "hkl_class_bins.csv", class_rows)
    class_mod_rows = hkl_class_equiv_modulation(h, k, l, obs, sig, pred, valid, groups_by_mode[args.primary_group_mode])
    write_csv(out / "hkl_class_equiv_modulation.csv", class_mod_rows)
    write_csv(out / "trim_top_excess_curve.csv", trim_curve(obs, sig, pred, primary_floor, valid, wght_a, wght_b))

    top_group_rows = sorted(
        [r for r in group_rows_all if r["mode"] == args.primary_group_mode],
        key=lambda r: (r["excess_weighted_residual_sum"], r["excess_abs_residual_sum"]),
        reverse=True,
    )[: max(0, int(args.top_groups))]

    recommendations = verdict({"group_modes": group_summaries}, args.primary_group_mode)
    recommendations.extend(modulation_verdict(class_mod_rows))
    summary = {
        "inputs": {
            "row_table": str(Path(args.row_table).resolve()),
            "hkl": str(Path(args.hkl).resolve()) if args.hkl else "",
            "res": str(Path(args.res).resolve()),
            "obs_column": args.obs_column,
            "sigma_column": args.sigma_column,
            "pred_column": args.pred_column,
            "born_column": args.born_column if born is not None else "",
        },
        "row_count": int(len(rows)),
        "valid_rows": int(np.count_nonzero(valid)),
        "wght": {"a": float(wght_a), "b": float(wght_b)},
        "primary_group_mode": args.primary_group_mode,
        "group_modes": group_summaries,
        "recommendations": recommendations,
        "artifacts": {
            "axis_bins": str((out / "premerge_axis_bins.csv").resolve()),
            "hkl_class_bins": str((out / "hkl_class_bins.csv").resolve()),
            "hkl_class_equiv_modulation": str((out / "hkl_class_equiv_modulation.csv").resolve()),
            "trim_curve": str((out / "trim_top_excess_curve.csv").resolve()),
            "top_primary_groups": str((out / f"top_{args.primary_group_mode}_groups.csv").resolve()),
        },
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    lines = [
        "# Pre-Merge Row-Space R-Factor Audit",
        "",
        f"Rows: `{len(rows)}`; valid rows: `{np.count_nonzero(valid)}`; primary equivalence: `{args.primary_group_mode}`.",
        "",
        "## Metrics Before SHELX-Style Merging",
        "",
        "| group mode | groups | multi rows | dyn R1all % | dyn R2all % | dyn wR2all % | scalar-equivalent R1all % | scalar-equivalent R2all % | scalar-equivalent wR2all % | within-row corr |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for mode in group_modes:
        s = group_summaries[mode]
        dyn = s["dyn"]
        floor = s["group_floor"]
        lines.append(
            f"| {mode} | {s['groups']} | {s['multi_rows']} | {pct(dyn['R1_all']):.2f} | {pct(dyn['R2_all']):.2f} | {pct(dyn['wR2_all']):.2f} | "
            f"{pct(floor['R1_all']):.2f} | {pct(floor['R2_all']):.2f} | {pct(floor['wR2_all']):.2f} | {s['within_equiv_obs_pred_corr']:.3f} |"
        )
    lines.extend(["", "## Candidate Fix Signals", ""])
    for item in summary["recommendations"]:
        lines.append(f"- {item}")
    lines.extend(
        [
            "",
            "## HKL Class Residuals",
            "",
            "| class | n | R1all % | R2all % | wR2all % | row % | weighted residual % | enrichment | q median |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in class_rows:
        lines.append(
            f"| {r['hkl_class']} | {r['n']} | {pct(r['R1_all']):.2f} | {pct(r['R2_all']):.2f} | {pct(r['wR2_all']):.2f} | "
            f"{pct(r['row_fraction']):.1f} | {pct(r['weighted_residual_fraction']):.1f} | {r['weighted_residual_enrichment']:.2f} | {r['q_median']:.3g} |"
        )
    lines.extend(
        [
            "",
            "## HKL Class Equivalent-Row Modulation",
            "",
            "| class | groups | rows | within corr | beta | model/scalar abs | excess abs residual |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in class_mod_rows:
        if r["rows"] <= 1:
            continue
        lines.append(
            f"| {r['hkl_class']} | {r['groups']} | {r['rows']} | {r['within_equiv_obs_pred_corr']:.3f} | "
            f"{r['within_equiv_obs_per_pred_beta']:.3f} | {r['model_to_scalar_abs_ratio']:.2f} | {r['excess_abs_residual_sum']:.1f} |"
        )
    lines.extend(
        [
            "",
            "## Top Pre-Merge Offenders",
            "",
            "| h | k | l | n | d median | I/sigma median | obs/pred mean | excess abs residual | frame span | q median |",
            "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for r in top_group_rows[:12]:
        lines.append(
            f"| {r['h']} | {r['k']} | {r['l']} | {r['n']} | {r['d_median']:.3f} | {r['I_over_sigma_median']:.2f} | "
            f"{r['obs_pred_mean_ratio']:.3f} | {r['excess_abs_residual_sum']:.1f} | {r['frame_span']:.1f} | {r['q_median']:.3g} |"
        )
    lines.extend(
        [
            "",
            "Artifacts:",
            f"- `{out / 'premerge_axis_bins.csv'}`",
            f"- `{out / 'hkl_class_bins.csv'}`",
            f"- `{out / 'hkl_class_equiv_modulation.csv'}`",
            f"- `{out / 'trim_top_excess_curve.csv'}`",
            f"- `{out / f'top_{args.primary_group_mode}_groups.csv'}`",
            f"- `{out / 'summary.json'}`",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str((out / "REPORT.md").resolve()), "summary": str((out / "summary.json").resolve())}, indent=2))


if __name__ == "__main__":
    main()
