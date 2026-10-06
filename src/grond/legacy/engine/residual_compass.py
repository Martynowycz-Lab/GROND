#!/usr/bin/env python3
"""Durable residual compass for native dynamical row tables.

This command is intentionally post-calculation.  It reads a native row table
such as `native_dyn_calc.csv`, keeps the Bloch predictions frozen, and reports
where residuals live: resolution, I/sigma, XDS CORR, detector position, frame,
and dynamical q tails.  It is meant to be run after every serious native
nuisance or structure-update pass before inventing another knob.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
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


def col(rows, name: str, default=float("nan")) -> np.ndarray:
    """Return one column as a float array, with default for missing or unparsable cells."""

    out = []
    for r in rows:
        try:
            out.append(float(r.get(name, default)))
        except (TypeError, ValueError):
            out.append(float(default))
    return np.asarray(out, dtype=float)


def str_col(rows, name: str, default: str = "") -> np.ndarray:
    """Return one column as a string array, with default for missing cells."""

    return np.asarray([str(r.get(name, default)) for r in rows])


def int_col(rows, name: str, default: int = -1) -> np.ndarray:
    """Return one column as an int array, with default for missing or unparsable cells."""

    out = []
    for r in rows:
        try:
            out.append(int(float(r.get(name, default))))
        except (TypeError, ValueError):
            out.append(int(default))
    return np.asarray(out, dtype=int)


def parse_models(text: str, rows: list[dict[str, str]]) -> list[tuple[str, str]]:
    """Parse --model name=column pairs, defaulting to dyn_scaled and born_scaled; raise on unknown columns."""

    available = set(rows[0].keys())
    if not text:
        models = []
        if "dyn_scaled" in available:
            models.append(("dyn", "dyn_scaled"))
        if "born_scaled" in available:
            models.append(("born", "born_scaled"))
        if not models:
            raise ValueError("No default model columns found; pass --model name=column.")
        return models
    models = []
    for part in text.split(","):
        if not part.strip():
            continue
        if "=" not in part:
            raise ValueError(f"--model entries must be name=column, got {part!r}")
        name, column = [x.strip() for x in part.split("=", 1)]
        if column not in available:
            raise ValueError(f"model column {column!r} is not in row table")
        models.append((name, column))
    if not models:
        raise ValueError("No model columns requested")
    return models


def quantile_bins(values: np.ndarray, n_bins: int) -> list[np.ndarray]:
    """Split the finite values into n_bins equal-count bins of row indices, in increasing order."""

    values = np.asarray(values, dtype=float)
    finite = np.isfinite(values)
    idx = np.where(finite)[0]
    if len(idx) == 0:
        return []
    order = idx[np.argsort(values[idx], kind="mergesort")]
    return [b for b in np.array_split(order, max(1, int(n_bins))) if len(b)]


def metric_row(
    axis: str,
    bin_no: int,
    idx: np.ndarray,
    axis_values: np.ndarray,
    model_name: str,
    obs,
    sig,
    pred,
    q,
    wght_a,
    wght_b,
    total_amp_residual: float,
    total_weighted_residual: float,
    total_rows: int,
):
    """Return one model's metrics and residual shares in one axis bin.

    Shares are relative to the model's totals over all rows; enrichment is a share divided by the bin's row
    fraction.
    """

    mask = np.zeros(len(obs), dtype=bool)
    mask[np.asarray(idx, dtype=int)] = True
    metrics = native_metrics(obs, sig, pred, mask, wght_a, wght_b, n_params=0)
    local = np.asarray(idx, dtype=int)
    valid = local[np.isfinite(obs[local]) & np.isfinite(sig[local]) & np.isfinite(pred[local])]
    ratio = np.full(len(valid), np.nan, dtype=float)
    if len(valid):
        ratio = obs[valid] / np.maximum(pred[valid], 1e-30)
    residual = obs[valid] - pred[valid] if len(valid) else np.asarray([], dtype=float)
    res_sig = residual / np.maximum(sig[valid], 1e-12) if len(valid) else np.asarray([], dtype=float)
    amp_residual = np.abs(np.sqrt(np.maximum(obs[valid], 0.0)) - np.sqrt(np.maximum(pred[valid], 0.0))) if len(valid) else np.asarray([], dtype=float)
    weights = shelx_weights(obs[valid], np.maximum(sig[valid], 1e-12), pred[valid], wght_a, wght_b) if len(valid) else np.asarray([], dtype=float)
    weighted_residual = weights * residual * residual if len(valid) else np.asarray([], dtype=float)
    row_fraction = float(len(valid) / max(int(total_rows), 1))
    amp_fraction = float(np.nansum(amp_residual) / max(float(total_amp_residual), 1e-30)) if len(valid) else float("nan")
    weighted_fraction = float(np.nansum(weighted_residual) / max(float(total_weighted_residual), 1e-30)) if len(valid) else float("nan")
    vals = axis_values[local]
    qv = q[local] if q is not None else np.asarray([], dtype=float)
    return {
        "axis": axis,
        "bin": int(bin_no),
        "model": model_name,
        "n": int(metrics["n"]),
        "n_gt": int(metrics["n_gt"]),
        "axis_min": float(np.nanmin(vals)) if len(vals) else float("nan"),
        "axis_median": float(np.nanmedian(vals)) if len(vals) else float("nan"),
        "axis_max": float(np.nanmax(vals)) if len(vals) else float("nan"),
        "R1_all": float(metrics["R1_all"]),
        "R1_gt": float(metrics["R1_gt"]),
        "wR2_all": float(metrics["wR2_all"]),
        "CC": float(metrics["CC"]),
        "median_residual_obs_minus_calc": float(np.nanmedian(residual)) if len(residual) else float("nan"),
        "median_abs_residual": float(np.nanmedian(np.abs(residual))) if len(residual) else float("nan"),
        "median_obs_calc_ratio": float(np.nanmedian(ratio)) if len(ratio) else float("nan"),
        "median_signed_residual_over_sigma": float(np.nanmedian(res_sig)) if len(res_sig) else float("nan"),
        "median_abs_residual_over_sigma": float(np.nanmedian(np.abs(res_sig))) if len(res_sig) else float("nan"),
        "row_fraction": row_fraction,
        "amp_residual_fraction": amp_fraction,
        "weighted_residual_fraction": weighted_fraction,
        "amp_residual_enrichment": float(amp_fraction / max(row_fraction, 1e-30)) if np.isfinite(amp_fraction) else float("nan"),
        "weighted_residual_enrichment": float(weighted_fraction / max(row_fraction, 1e-30)) if np.isfinite(weighted_fraction) else float("nan"),
        "q_p05": float(np.nanpercentile(qv[np.isfinite(qv)], 5)) if q is not None and np.count_nonzero(np.isfinite(qv)) else float("nan"),
        "q_median": float(np.nanmedian(qv[np.isfinite(qv)])) if q is not None and np.count_nonzero(np.isfinite(qv)) else float("nan"),
        "q_p95": float(np.nanpercentile(qv[np.isfinite(qv)], 95)) if q is not None and np.count_nonzero(np.isfinite(qv)) else float("nan"),
    }


def q_summary(q: np.ndarray, q_min: float, q_max: float, log_tail: float) -> dict:
    """Return percentiles of q = Idyn/Iborn and the fractions outside [q_min, q_max] or beyond |log q| > log_tail."""

    good = np.isfinite(q) & (q > 0)
    if not np.any(good):
        return {"n_positive_q": 0}
    qg = q[good]
    logq = np.log(qg)
    return {
        "n_positive_q": int(len(qg)),
        "q_p01": float(np.nanpercentile(qg, 1)),
        "q_p05": float(np.nanpercentile(qg, 5)),
        "q_median": float(np.nanmedian(qg)),
        "q_p95": float(np.nanpercentile(qg, 95)),
        "q_p99": float(np.nanpercentile(qg, 99)),
        "logq_median": float(np.nanmedian(logq)),
        "logq_mad": float(np.nanmedian(np.abs(logq - np.nanmedian(logq)))),
        "frac_q_below_min": float(np.mean(qg < float(q_min))),
        "frac_q_above_max": float(np.mean(qg > float(q_max))),
        "frac_abs_logq_tail": float(np.mean(np.abs(logq) > float(log_tail))),
        "q_min_threshold": float(q_min),
        "q_max_threshold": float(q_max),
        "abs_logq_tail_threshold": float(log_tail),
    }


def add_xds_axes(rows, hkl_path: Path):
    """Return XDS xcal, ycal, zcal and detector radius (px from the median) per row, by row_index."""

    row_index = int_col(rows, "row_index")
    data = XDSData(hkl_path)
    good = (row_index >= 0) & (row_index < len(data.hkl))
    x = np.full(len(rows), np.nan, dtype=float)
    y = np.full(len(rows), np.nan, dtype=float)
    z = np.full(len(rows), np.nan, dtype=float)
    if np.any(good):
        x[good] = data.xcal[row_index[good]]
        y[good] = data.ycal[row_index[good]]
        z[good] = data.zcal[row_index[good]]
    r = np.sqrt((x - np.nanmedian(x)) ** 2 + (y - np.nanmedian(y)) ** 2)
    return x, y, z, r


def optional_axis(rows: list[dict[str, str]], names: list[str]) -> tuple[str | None, np.ndarray | None]:
    """Return (column name, values) for the first usable candidate column, or (None, None).

    A column needs at least max(20, 1% of rows) finite values.
    """

    available = set(rows[0].keys()) if rows else set()
    for name in names:
        if name not in available:
            continue
        values = col(rows, name)
        if np.count_nonzero(np.isfinite(values)) >= max(20, len(rows) // 100):
            return name, values
    return None, None


def write_csv(path: Path, rows: list[dict]) -> None:
    """Write rows to a CSV file with the first row's keys as header; an empty list writes an empty file."""

    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        fieldnames = list(rows[0].keys())
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


def pct(v) -> float:
    """Return a fraction as a percentage, or NaN if it is not finite."""

    return 100.0 * float(v) if np.isfinite(v) else float("nan")


def valid_prediction_mask(obs, sig, pred):
    """Return the mask of rows with finite Iobs, sigma and prediction."""

    return np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred)


def residual_totals(obs, sig, pred, wght_a: float, wght_b: float) -> tuple[float, float]:
    """Return the total amplitude residual |√Iobs - √Icalc| and the SHELXL-weighted squared residual."""

    m = valid_prediction_mask(obs, sig, pred)
    if not np.any(m):
        return 0.0, 0.0
    amp = np.abs(np.sqrt(np.maximum(obs[m], 0.0)) - np.sqrt(np.maximum(pred[m], 0.0)))
    res = obs[m] - pred[m]
    w = shelx_weights(obs[m], np.maximum(sig[m], 1e-12), pred[m], wght_a, wght_b)
    return float(np.nansum(amp)), float(np.nansum(w * res * res))


def fit_global_scale(obs, sig, pred, wght_a: float, wght_b: float, n_iter: int = 6) -> tuple[np.ndarray, float]:
    """Fit one SHELXL-weighted scale of pred to obs and return (scaled prediction, scale).

    Starts from the 1/sigma² least-squares scale and refits n_iter times; NaN with fewer than two usable rows.
    """

    good = valid_prediction_mask(obs, sig, pred) & (pred > 0)
    if np.count_nonzero(good) < 2:
        return np.full_like(pred, np.nan, dtype=float), float("nan")
    x = pred[good]
    y = obs[good]
    s = np.maximum(sig[good], 1e-12)
    scale = float(np.sum(x * y / np.maximum(s * s, 1e-30)) / max(np.sum(x * x / np.maximum(s * s, 1e-30)), 1e-30))
    scale = max(scale, 1e-30)
    for _ in range(max(1, int(n_iter))):
        calc = x * scale
        w = shelx_weights(y, s, calc, wght_a, wght_b)
        scale = max(float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30)), 1e-30)
    scaled = np.full_like(pred, np.nan, dtype=float)
    scaled[np.isfinite(pred)] = pred[np.isfinite(pred)] * scale
    return scaled, scale


def pearson(x, y) -> float:
    """Return the Pearson correlation of the finite pairs, or NaN with fewer than three pairs or no spread."""

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    m = np.isfinite(x) & np.isfinite(y)
    if np.count_nonzero(m) < 3:
        return float("nan")
    xm = x[m] - np.mean(x[m])
    ym = y[m] - np.mean(y[m])
    den = math.sqrt(float(np.sum(xm * xm) * np.sum(ym * ym)))
    if den <= 0:
        return float("nan")
    return float(np.sum(xm * ym) / den)


def scale_pressure(rows, raw_col: str, scaled_col: str, axes: dict[str, tuple[np.ndarray, list[np.ndarray]]]) -> dict:
    """Summarize the row scale scaled/raw: percentiles, per-group medians and log-scale correlations with row axes."""

    raw = col(rows, raw_col)
    scaled = col(rows, scaled_col)
    good = np.isfinite(raw) & np.isfinite(scaled) & (raw > 0) & (scaled > 0)
    if not np.any(good):
        return {}
    scale = np.full_like(raw, np.nan, dtype=float)
    scale[good] = scaled[good] / raw[good]
    log_scale = np.full_like(scale, np.nan, dtype=float)
    log_scale[good] = np.log(scale[good])
    group = int_col(rows, "group")
    group_rows = []
    for g in sorted(set(int(v) for v in group[good])):
        m = good & (group == g)
        if np.count_nonzero(m):
            group_rows.append({"group": g, "n": int(np.count_nonzero(m)), "median_scale": float(np.nanmedian(scale[m]))})
    group_scales = np.asarray([r["median_scale"] for r in group_rows], dtype=float)
    corrs = {}
    for axis in ("inverse_d2", "I_over_sigma", "corr", "peak", "rlp", "detector_radius", "frame_zcal"):
        if axis in axes:
            corrs[axis] = pearson(log_scale, axes[axis][0])
    return {
        "raw_col": raw_col,
        "scaled_col": scaled_col,
        "n": int(np.count_nonzero(good)),
        "row_scale_p01": float(np.nanpercentile(scale[good], 1)),
        "row_scale_p05": float(np.nanpercentile(scale[good], 5)),
        "row_scale_median": float(np.nanmedian(scale[good])),
        "row_scale_p95": float(np.nanpercentile(scale[good], 95)),
        "row_scale_p99": float(np.nanpercentile(scale[good], 99)),
        "group_scale_p05": float(np.nanpercentile(group_scales, 5)) if len(group_scales) else float("nan"),
        "group_scale_median": float(np.nanmedian(group_scales)) if len(group_scales) else float("nan"),
        "group_scale_p95": float(np.nanpercentile(group_scales, 95)) if len(group_scales) else float("nan"),
        "log_scale_correlations": corrs,
        "group_rows": group_rows,
    }


def top_residual_rows(
    rows: list[dict[str, str]],
    obs,
    sig,
    pred,
    q,
    x,
    y,
    z,
    radius,
    wght_a: float,
    wght_b: float,
    limit: int,
) -> list[dict]:
    """Return the limit rows with the largest SHELXL-weighted residuals, with their shares and row context."""

    m = valid_prediction_mask(obs, sig, pred)
    res = np.full(len(rows), np.nan, dtype=float)
    res[m] = obs[m] - pred[m]
    amp = np.full(len(rows), np.nan, dtype=float)
    amp[m] = np.abs(np.sqrt(np.maximum(obs[m], 0.0)) - np.sqrt(np.maximum(pred[m], 0.0)))
    weighted = np.full(len(rows), np.nan, dtype=float)
    weighted[m] = shelx_weights(obs[m], np.maximum(sig[m], 1e-12), pred[m], wght_a, wght_b) * res[m] * res[m]
    amp_total = max(float(np.nansum(amp[m])), 1e-30)
    weighted_total = max(float(np.nansum(weighted[m])), 1e-30)
    order = np.where(m)[0][np.argsort(weighted[m])[::-1]]
    raw = col(rows, "dyn_raw") if "dyn_raw" in rows[0] else np.full(len(rows), np.nan)
    scale = np.full(len(rows), np.nan, dtype=float)
    good_scale = np.isfinite(raw) & np.isfinite(pred) & (raw > 0) & (pred > 0)
    scale[good_scale] = pred[good_scale] / raw[good_scale]
    out = []
    for rank, i in enumerate(order[: max(0, int(limit))], start=1):
        r = rows[int(i)]
        out.append(
            {
                "rank": rank,
                "row_index": r.get("row_index", ""),
                "h": r.get("h", ""),
                "k": r.get("k", ""),
                "l": r.get("l", ""),
                "split": r.get("split", ""),
                "group": r.get("group", ""),
                "detector_bin": r.get("detector_bin", ""),
                "d_A": r.get("d_A", ""),
                "target_I": float(obs[i]),
                "sigma_eff": float(sig[i]),
                "I_over_sigma": float(obs[i] / max(sig[i], 1e-12)),
                "dyn_scaled": float(pred[i]),
                "obs_minus_calc": float(res[i]),
                "residual_over_sigma": float(res[i] / max(sig[i], 1e-12)),
                "amp_residual_fraction": float(amp[i] / amp_total),
                "weighted_residual_fraction": float(weighted[i] / weighted_total),
                "q_dyn_over_born": float(q[i]) if q is not None and np.isfinite(q[i]) else float("nan"),
                "dyn_scale_over_raw": float(scale[i]) if np.isfinite(scale[i]) else float("nan"),
                "corr": r.get("corr", ""),
                "peak": r.get("peak", ""),
                "rlp": r.get("rlp", ""),
                "xcal": float(x[i]) if np.isfinite(x[i]) else float("nan"),
                "ycal": float(y[i]) if np.isfinite(y[i]) else float("nan"),
                "zcal": float(z[i]) if np.isfinite(z[i]) else float("nan"),
                "detector_radius": float(radius[i]) if np.isfinite(radius[i]) else float("nan"),
            }
        )
    return out


def concentration_summary(obs, sig, pred, wght_a: float, wght_b: float) -> dict:
    """Return the share of amplitude and weighted residual carried by the top 1, 5 and 10% of rows."""

    m = valid_prediction_mask(obs, sig, pred)
    if not np.any(m):
        return {}
    res = obs[m] - pred[m]
    amp = np.abs(np.sqrt(np.maximum(obs[m], 0.0)) - np.sqrt(np.maximum(pred[m], 0.0)))
    weighted = shelx_weights(obs[m], np.maximum(sig[m], 1e-12), pred[m], wght_a, wght_b) * res * res

    def frac_at(values, row_frac):
        """Return the share of the total carried by the largest row_frac of values."""

        n = max(1, int(math.ceil(len(values) * row_frac)))
        v = np.sort(values)[::-1]
        return float(np.sum(v[:n]) / max(float(np.sum(v)), 1e-30))

    return {
        "top_1pct_amp_fraction": frac_at(amp, 0.01),
        "top_5pct_amp_fraction": frac_at(amp, 0.05),
        "top_10pct_amp_fraction": frac_at(amp, 0.10),
        "top_1pct_weighted_fraction": frac_at(weighted, 0.01),
        "top_5pct_weighted_fraction": frac_at(weighted, 0.05),
        "top_10pct_weighted_fraction": frac_at(weighted, 0.10),
    }


def model_driver_rows(binned_rows: list[dict], model_name: str, key: str, limit: int = 8) -> list[dict]:
    """Return up to limit bins of one model with the largest value of key."""

    candidates = [r for r in binned_rows if r.get("model") == model_name and np.isfinite(float(r.get(key, float("nan"))))]
    return sorted(candidates, key=lambda r: float(r[key]), reverse=True)[:limit]


def main() -> None:
    """Bin each model's residuals along resolution, intensity, quality, detector, frame and q axes.

    Writes the bin, top-row and scale-group CSVs, summary.json and REPORT.md to --out-dir.
    """

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--row-table", required=True)
    ap.add_argument("--hkl", default=str(DEFAULT_HKL), help="INTEGRATE.HKL used to add detector x/y/frame axes from row_index.")
    ap.add_argument("--res", default=str(DEFAULT_RES), help="RES file used only for WGHT a/b values.")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--model", default="", help="Comma-separated name=column list. Default uses dyn=dyn_scaled,born=born_scaled when present.")
    ap.add_argument("--bins", type=int, default=8)
    ap.add_argument("--top-rows", type=int, default=100, help="Number of largest weighted-residual dyn rows to write.")
    ap.add_argument("--q-min", type=float, default=0.05)
    ap.add_argument("--q-max", type=float, default=20.0)
    ap.add_argument("--q-log-tail", type=float, default=math.log(4.0), help="Tail threshold on abs(log(q)). Default flags q outside a 4x band.")
    args = ap.parse_args()

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rows = read_rows(Path(args.row_table))
    models = parse_models(args.model, rows)
    obs = col(rows, "target_I")
    sig = col(rows, "sigma_eff")
    d = col(rows, "d_A")
    isigi = obs / np.maximum(sig, 1e-12)
    corr = col(rows, "corr")
    peak = col(rows, "peak")
    rlp = col(rows, "rlp")
    q = col(rows, "q_dyn_over_born") if "q_dyn_over_born" in rows[0] else None
    split = str_col(rows, "split")
    x, y, z, radius = add_xds_axes(rows, Path(args.hkl))
    wght_a, wght_b = parse_wght(Path(args.res))

    axes = {
        "resolution_d_A": (d, quantile_bins(d, args.bins)),
        "inverse_d2": (1.0 / np.maximum(d, 1e-12) ** 2, quantile_bins(1.0 / np.maximum(d, 1e-12) ** 2, args.bins)),
        "Iobs": (obs, quantile_bins(obs, args.bins)),
        "abs_Iobs": (np.abs(obs), quantile_bins(np.abs(obs), args.bins)),
        "I_over_sigma": (isigi, quantile_bins(isigi, args.bins)),
        "corr": (corr, quantile_bins(corr, args.bins)),
        "peak": (peak, quantile_bins(peak, args.bins)),
        "rlp": (rlp, quantile_bins(rlp, args.bins)),
        "detector_x": (x, quantile_bins(x, args.bins)),
        "detector_y": (y, quantile_bins(y, args.bins)),
        "detector_radius": (radius, quantile_bins(radius, args.bins)),
        "frame_zcal": (z, quantile_bins(z, args.bins)),
    }
    optional_axes = [
        ("local_background", ["local_background_mean", "background_mean", "bg_mean", "raw_background_mean"]),
        ("local_background_sigma", ["local_background_sigma", "background_sigma", "background_mad", "bg_sigma", "raw_background_sigma"]),
        ("foreground_pixels", ["n_foreground_pixels", "foreground_pixels", "n_fg"]),
        ("background_pixels", ["n_background_pixels", "background_pixels", "n_bg"]),
        ("profile_correlation", ["profile_corr", "raw_profile_corr", "bloch_profile_corr", "observed_profile_fraction"]),
        ("crowding", ["crowding", "overlap_score", "neighbor_score"]),
    ]
    for axis_name, names in optional_axes:
        source_name, values = optional_axis(rows, names)
        if values is not None:
            axes[f"{axis_name}:{source_name}"] = (values, quantile_bins(values, args.bins))
    if q is not None:
        axes["q_dyn_over_born"] = (q, quantile_bins(q, args.bins))
        logq = np.full_like(q, np.nan, dtype=float)
        good_q = np.isfinite(q) & (q > 0)
        logq[good_q] = np.log(q[good_q])
        axes["abs_log_q"] = (np.abs(logq), quantile_bins(np.abs(logq), args.bins))

    model_totals = {}
    for model_name, model_col in models:
        pred = col(rows, model_col)
        amp_total, weighted_total = residual_totals(obs, sig, pred, wght_a, wght_b)
        model_totals[model_name] = {"amp": amp_total, "weighted": weighted_total, "n": int(np.count_nonzero(valid_prediction_mask(obs, sig, pred)))}

    binned_rows = []
    for axis, (values, bins) in axes.items():
        for bi, idx in enumerate(bins, start=1):
            for model_name, model_col in models:
                totals = model_totals[model_name]
                binned_rows.append(
                    metric_row(
                        axis,
                        bi,
                        idx,
                        values,
                        model_name,
                        obs,
                        sig,
                        col(rows, model_col),
                        q,
                        wght_a,
                        wght_b,
                        totals["amp"],
                        totals["weighted"],
                        totals["n"],
                    )
                )
    write_csv(out / "residual_bins.csv", binned_rows)

    split_masks = {
        "work": split == "work",
        "free": split == "free",
        "all": np.ones(len(rows), dtype=bool),
    }
    model_summary = {}
    for model_name, model_col in models:
        pred = col(rows, model_col)
        model_summary[model_name] = {
            split_name: native_metrics(obs, sig, pred, mask, wght_a, wght_b, n_params=0)
            for split_name, mask in split_masks.items()
        }

    raw_scale_summary = {}
    for model_name, raw_col in (("dyn", "dyn_raw"), ("born", "born_raw")):
        if raw_col not in rows[0]:
            continue
        raw_pred = col(rows, raw_col)
        global_scaled, global_scale = fit_global_scale(obs, sig, raw_pred, wght_a, wght_b)
        raw_scale_summary[model_name] = {
            "raw_column": raw_col,
            "global_scale": float(global_scale),
            "global_scaled_metrics": {
                split_name: native_metrics(obs, sig, global_scaled, mask, wght_a, wght_b, n_params=1)
                for split_name, mask in split_masks.items()
            },
        }
        scaled_col = f"{model_name}_scaled"
        if scaled_col in rows[0]:
            raw_scale_summary[model_name]["scale_pressure"] = scale_pressure(rows, raw_col, scaled_col, axes)

    concentration = {}
    for model_name, model_col in models:
        concentration[model_name] = concentration_summary(obs, sig, col(rows, model_col), wght_a, wght_b)

    top_rows = []
    if args.top_rows > 0 and "dyn_scaled" in rows[0]:
        top_rows = top_residual_rows(rows, obs, sig, col(rows, "dyn_scaled"), q, x, y, z, radius, wght_a, wght_b, args.top_rows)
        write_csv(out / "top_weighted_residual_rows.csv", top_rows)

    scale_group_rows = []
    for model_name, data in raw_scale_summary.items():
        pressure = data.get("scale_pressure") or {}
        for row in pressure.get("group_rows", []):
            scale_group_rows.append({"model": model_name, **row})
    if scale_group_rows:
        write_csv(out / "scale_group_medians.csv", scale_group_rows)

    amp_driver_rows = model_driver_rows(binned_rows, "dyn", "amp_residual_enrichment", limit=12)
    weighted_driver_rows = model_driver_rows(binned_rows, "dyn", "weighted_residual_enrichment", limit=12)

    summary = {
        "inputs": {
            "row_table": str(Path(args.row_table).resolve()),
            "hkl": str(Path(args.hkl).resolve()),
            "res": str(Path(args.res).resolve()),
            "models": [{"name": n, "column": c} for n, c in models],
        },
        "row_count": int(len(rows)),
        "splits": {k: int(np.count_nonzero(v)) for k, v in split_masks.items()},
        "wght": {"a": float(wght_a), "b": float(wght_b)},
        "q_summary": q_summary(q, args.q_min, args.q_max, args.q_log_tail) if q is not None else {},
        "model_summary": model_summary,
        "raw_scale_summary": raw_scale_summary,
        "concentration": concentration,
        "top_weighted_residual_rows": top_rows[: min(25, len(top_rows))],
        "top_amp_driver_bins": amp_driver_rows,
        "top_weighted_driver_bins": weighted_driver_rows,
        "axis_count": {axis: int(len(bins)) for axis, (_values, bins) in axes.items()},
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    lines = [
        "# Native Residual Compass",
        "",
        f"Rows: `{len(rows)}`. Work/free: `{summary['splits']['work']}/{summary['splits']['free']}`. WGHT: `a={wght_a:g}`, `b={wght_b:g}`.",
        "",
        "## Overall Metrics",
        "",
        "| model | split | n | n_gt | R1all % | R1gt % | wR2all % | CC |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for model_name in model_summary:
        for split_name in ("work", "free", "all"):
            m = model_summary[model_name][split_name]
            lines.append(
                f"| {model_name} | {split_name} | {m['n']} | {m['n_gt']} | {pct(m['R1_all']):.2f} | {pct(m['R1_gt']):.2f} | {pct(m['wR2_all']):.2f} | {m['CC']:.4f} |"
            )
    if q is not None:
        qs = summary["q_summary"]
        lines.extend(
            [
                "",
                "## q Tail Summary",
                "",
                f"`q=Idyn/Iborn` positive rows: `{qs.get('n_positive_q', 0)}`. p01/p05/median/p95/p99: `{qs.get('q_p01', float('nan')):.4g}` / `{qs.get('q_p05', float('nan')):.4g}` / `{qs.get('q_median', float('nan')):.4g}` / `{qs.get('q_p95', float('nan')):.4g}` / `{qs.get('q_p99', float('nan')):.4g}`.",
                f"Tail fractions: q < `{args.q_min:g}` is `{100*qs.get('frac_q_below_min', float('nan')):.2f}%`; q > `{args.q_max:g}` is `{100*qs.get('frac_q_above_max', float('nan')):.2f}%`; abs(log q) > `{args.q_log_tail:.3f}` is `{100*qs.get('frac_abs_logq_tail', float('nan')):.2f}%`.",
            ]
        )
    lines.extend(["", "## Residual Concentration", ""])
    lines.append("| model | top 1% amp | top 5% amp | top 10% amp | top 1% weighted | top 5% weighted | top 10% weighted |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for model_name, c in concentration.items():
        lines.append(
            f"| {model_name} | {pct(c.get('top_1pct_amp_fraction', float('nan'))):.1f}% | {pct(c.get('top_5pct_amp_fraction', float('nan'))):.1f}% | {pct(c.get('top_10pct_amp_fraction', float('nan'))):.1f}% | "
            f"{pct(c.get('top_1pct_weighted_fraction', float('nan'))):.1f}% | {pct(c.get('top_5pct_weighted_fraction', float('nan'))):.1f}% | {pct(c.get('top_10pct_weighted_fraction', float('nan'))):.1f}% |"
        )
    if raw_scale_summary:
        lines.extend(["", "## Scale Factor Pressure", ""])
        lines.append("| model | scaled R1all % | scaled wR2all % | single-scale R1all % | single-scale wR2all % | row-scale p05/p50/p95 | strongest log-scale correlations |")
        lines.append("|---|---:|---:|---:|---:|---|---|")
        for model_name, data in raw_scale_summary.items():
            scaled = model_summary.get(model_name, {}).get("all", {})
            global_metrics = data["global_scaled_metrics"]["all"]
            pressure = data.get("scale_pressure") or {}
            corrs = pressure.get("log_scale_correlations", {})
            corr_text = ", ".join(
                f"{k}={v:+.2f}"
                for k, v in sorted(corrs.items(), key=lambda kv: abs(kv[1]) if np.isfinite(kv[1]) else -1.0, reverse=True)[:3]
                if np.isfinite(v)
            )
            scale_text = "n/a"
            if pressure:
                scale_text = f"{pressure['row_scale_p05']:.3g}/{pressure['row_scale_median']:.3g}/{pressure['row_scale_p95']:.3g}"
            lines.append(
                f"| {model_name} | {pct(scaled.get('R1_all', float('nan'))):.2f} | {pct(scaled.get('wR2_all', float('nan'))):.2f} | "
                f"{pct(global_metrics.get('R1_all', float('nan'))):.2f} | {pct(global_metrics.get('wR2_all', float('nan'))):.2f} | {scale_text} | {corr_text or 'n/a'} |"
            )
    lines.extend(["", "## Largest Dyn Driver Bins", ""])
    lines.append("These bins are ranked by residual enrichment, not by raw R factor. Enrichment above 1 means the bin contributes more residual than its row count alone predicts.")
    lines.append("")
    lines.append("| kind | axis | bin | n | axis median | R1all % | wR2all % | row % | residual % | enrichment |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
    for kind, key, rows_for_key in (
        ("amp", "amp_residual", amp_driver_rows[:8]),
        ("weighted", "weighted_residual", weighted_driver_rows[:8]),
    ):
        frac_key = f"{key}_fraction"
        enrich_key = f"{key}_enrichment"
        for r in rows_for_key:
            lines.append(
                f"| {kind} | {r['axis']} | {r['bin']} | {r['n']} | {float(r['axis_median']):.4g} | {pct(r['R1_all']):.2f} | {pct(r['wR2_all']):.2f} | "
                f"{pct(r['row_fraction']):.1f} | {pct(r[frac_key]):.1f} | {float(r[enrich_key]):.2f} |"
            )
    lines.extend(
        [
            "",
            "## Required Axes",
            "",
            "The binned CSV always includes resolution (`resolution_d_A`, `inverse_d2`), raw intensity (`Iobs`, `abs_Iobs`), I/sigma, XDS `CORR`, detector x, detector y, detector radius, frame `ZCAL`, and q-tail axes when q is present. If raw-box metadata are present, local background, background sigma, foreground/background pixel count, profile-correlation, and crowding axes are added automatically.",
            "",
            f"Binned diagnostics: `{out / 'residual_bins.csv'}`",
            f"Top dyn residual rows: `{out / 'top_weighted_residual_rows.csv'}`",
            f"Scale group medians: `{out / 'scale_group_medians.csv'}`",
            f"Machine summary: `{out / 'summary.json'}`",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"report": str((out / "REPORT.md").resolve()), "rows": len(rows), "bins_csv": str((out / "residual_bins.csv").resolve())}, indent=2))


if __name__ == "__main__":
    main()
