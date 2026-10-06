"""Matched row residuals, work-only scale fitting and explicit row-index loading."""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from grond.bloch.config import Config
from grond.bloch.splits import split_masks
from grond.io.xds import XDSData


def effective_sigma(data: XDSData, cfg: Config):
    """Return per-row sigmas inflated by the Config error model; NaN where the XDS sigma is not positive.

    sigma_eff^2 = sigma^2 + (a*I)^2 + (b_corr*I*c)^2 + (c_peak*I*p)^2, where a, b_corr and c_peak are the variance_*
    fields, c = max(0, 1 - CORR/100) and p = max(0, 1 - PEAK/100) (XDS percentages).
    """

    corr_term = np.maximum(0.0, 1.0 - data.corr / 100.0)
    peak_term = np.maximum(0.0, 1.0 - data.peak / 100.0)
    out = np.full(len(data.sigma), np.nan, dtype=float)
    valid = np.isfinite(data.sigma) & (np.asarray(data.sigma, dtype=float) > 0.0)
    out[valid] = np.sqrt(
        data.sigma[valid] ** 2
        + (cfg.variance_a * data.iobs[valid]) ** 2
        + (cfg.variance_b_corr * data.iobs[valid] * corr_term[valid]) ** 2
        + (cfg.variance_c_peak * data.iobs[valid] * peak_term[valid]) ** 2
    )
    return out


def scale_by_groups(obs, sig, pred, groups, fit_mask):
    """Fit one scale per group on the fit rows and return (scaled predictions, {group: scale}).

    Each scale minimises sum w*(obs - s*pred)^2 with w = 1/sigma^2 over rows in ``fit_mask`` with finite values and
    positive sigma and prediction; a negative scale is clipped to 0. A group with fewer than two such rows takes the
    global scale of all fit rows. Scaled predictions are NaN where ``pred`` is not finite.
    """

    obs = np.asarray(obs, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pred = np.asarray(pred, dtype=float)
    groups = np.asarray(groups, dtype=int)
    fit_mask = np.asarray(fit_mask, dtype=bool)
    if not (len(obs) == len(sig) == len(pred) == len(groups) == len(fit_mask)):
        raise ValueError("observation, sigma, prediction, group, and fit-mask lengths differ")
    scaled = np.full_like(pred, np.nan, dtype=float)
    scales = {}
    valid_measurement = np.isfinite(obs) & np.isfinite(sig) & (sig > 0.0)
    weights = np.zeros_like(sig, dtype=float)
    weights[valid_measurement] = 1.0 / (sig[valid_measurement] * sig[valid_measurement])
    global_fit = fit_mask & valid_measurement & np.isfinite(pred) & (pred > 0.0)
    if np.count_nonzero(global_fit) < 2:
        raise ValueError("at least two work rows with positive sigma and prediction are required to fit scales")
    x_global, y_global, w_global = pred[global_fit], obs[global_fit], weights[global_fit]
    global_scale = max(
        float(np.sum(w_global * x_global * y_global) / max(np.sum(w_global * x_global * x_global), 1e-30)),
        1e-30,
    )
    for g in np.unique(groups):
        fit = (groups == g) & fit_mask & valid_measurement & np.isfinite(pred) & (pred > 0)
        apply = groups == g
        if np.count_nonzero(fit) < 2:
            scales[int(g)] = global_scale
            finite_apply = apply & np.isfinite(pred)
            scaled[finite_apply] = pred[finite_apply] * global_scale
            continue
        x, y, w = pred[fit], obs[fit], weights[fit]
        scale = float(np.sum(w * x * y) / max(np.sum(w * x * x), 1e-30))
        scale = max(scale, 0.0)
        finite_apply = apply & np.isfinite(pred)
        scaled[finite_apply] = pred[finite_apply] * scale
        scales[int(g)] = scale
    return scaled, scales


def metrics(obs, sig, pred, mask):
    """Return n, n_gt, R1 (all rows and I > 2 sigma), wR2 and the Pearson CC of obs and pred on ``mask``.

    Only rows with finite obs, pred and positive sigma count. R1 uses F = sqrt(max(I, 0)); wR2 uses the weights
    1/(sigma^2 + (0.05*P)^2) with P = (max(obs, 0) + 2*max(pred, 0))/3. CC is NaN for fewer than three rows or a
    constant series.
    """

    mask = (
        np.asarray(mask, dtype=bool)
        & np.isfinite(obs)
        & np.isfinite(sig)
        & (np.asarray(sig, dtype=float) > 0.0)
        & np.isfinite(pred)
    )
    obs_m, sig_m, pred_m = obs[mask], sig[mask], pred[mask]
    if len(obs_m) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"),
                "wR2_all": float("nan"), "CC": float("nan")}
    gt = obs_m > 2.0 * sig_m
    fobs = np.sqrt(np.maximum(obs_m, 0.0))
    fpred = np.sqrt(np.maximum(pred_m, 0.0))
    eps = 1e-30

    def r1(local):
        """Return sum|Fobs - Fpred| / sum Fobs over the selected rows, or NaN when none is selected."""

        if not np.any(local):
            return float("nan")
        return float(np.sum(np.abs(fobs[local] - fpred[local])) / max(np.sum(fobs[local]), eps))

    p = (np.maximum(obs_m, 0.0) + 2.0 * np.maximum(pred_m, 0.0)) / 3.0
    w = 1.0 / np.maximum(sig_m * sig_m + (0.05 * p) ** 2, eps)
    wr2 = float(np.sqrt(np.sum(w * (obs_m - pred_m) ** 2) / max(np.sum(w * obs_m * obs_m), eps)))
    cc = (
        float(np.corrcoef(obs_m, pred_m)[0, 1])
        if len(obs_m) > 2 and np.std(pred_m) > 0 and np.std(obs_m) > 0
        else float("nan")
    )
    return {
        "n": int(np.count_nonzero(mask)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "wR2_all": wr2,
        "CC": cc,
    }


def evaluate_predictions(data, cfg, pred, fit_scales_on="work", model=None):
    """Scale dynamical and kinematical predictions per group on one split and score them on every split.

    ``pred`` holds ``dyn``, ``born`` and ``groups`` per row. Splits come from split_masks with the Config free fraction
    and seed (orbit-aware when ``model`` carries symmetry operations); the scales are fitted on ``fit_scales_on``. The
    ``delta_*`` entries are dynamical minus kinematical.
    """

    sig = effective_sigma(data, cfg)
    symmetry_ops = model.get("symmetry_ops") if model else None
    masks = split_masks(data.hkl, cfg.free_fraction, cfg.split_seed, symmetry_ops=symmetry_ops)
    group_indices = pred["groups"]
    fit_mask = masks[fit_scales_on]
    dyn_scaled, dyn_scales = scale_by_groups(data.iobs, sig, pred["dyn"], group_indices, fit_mask)
    born_scaled, born_scales = scale_by_groups(data.iobs, sig, pred["born"], group_indices, fit_mask)
    out = {"fit_scales_on": fit_scales_on, "splits": {}, "group_scale_count": int(len(dyn_scales))}
    for name, mask in masks.items():
        dyn_m = metrics(data.iobs, sig, dyn_scaled, mask)
        born_m = metrics(data.iobs, sig, born_scaled, mask)
        out["splits"][name] = {
            "dyn": dyn_m,
            "born": born_m,
            "delta_R1_all_dyn_minus_born": dyn_m["R1_all"] - born_m["R1_all"],
            "delta_R1_gt_dyn_minus_born": dyn_m["R1_gt"] - born_m["R1_gt"],
            "delta_wR2_all_dyn_minus_born": dyn_m["wR2_all"] - born_m["wR2_all"],
        }
    return out


def read_row_index_file(path: str | Path, *, row_count: int | None = None) -> list[int]:
    """Read unique zero-based row indices from text/CSV and reject invalid requested indices."""

    with open(path, newline="") as handle:
        sample = handle.read(4096)
        handle.seek(0)
        if "," in sample:
            reader = csv.DictReader(handle)
            field = (
                "row_index"
                if reader.fieldnames and "row_index" in reader.fieldnames
                else (reader.fieldnames[0] if reader.fieldnames else None)
            )
            if field is None:
                raise ValueError(f"No columns found in {path}")
            values = [int(row[field]) for row in reader]
        else:
            values = []
            for line_number, line in enumerate(handle, start=1):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                token = line.split()[0]
                if token.lower() in {"row_index", "index"}:
                    continue
                try:
                    values.append(int(token))
                except ValueError as exc:
                    raise ValueError(f"invalid row index {token!r} at {path}:{line_number}") from exc
    invalid = sorted({value for value in values if value < 0 or (row_count is not None and value >= row_count)})
    if invalid:
        allowed = "nonnegative" if row_count is None else f"in [0, {row_count})"
        raise ValueError(f"row indices in {path} must be {allowed}; invalid indices: {invalid[:8]}")
    return sorted(set(values))
