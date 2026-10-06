"""Held-out validation primitives shared by AUTO, its reports and the before/after comparator.

Definitions (checked against the independent reference in tests/reference/reference_lib.py):

* crystal-independent Laue-orbit keys, so one orbit is held out in every crystal at once;
* salted SHA-256 fold assignment (48-bit prefix mod K);
* amplitude R1 metrics (R_all on every row, R_obs on Io > 3 sigma);
* the paired, shell-stratified, merged-orbit bootstrap and the orbit sign test used for the hand;
* scale refits on training rows only (P-spline or free per-group scales) that predict held-out rows.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np

from grond.regularize.spline_scales import PSplineScaleFitter

INNER_CV_SALT = "GROND-INNER-20260925"
DEFAULT_FREE_SALT = "GROND-FREE-20260925"
BOOTSTRAP_SEED = 20260925
N_BOOT = 2000
PSPLINE_GRID = {"n_basis": (32,), "lambda": (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)}


# ----------------------------------------------------------------------------- keys and folds

def orbit_key(hkl, symmetry_ops=None, include_friedel: bool = True) -> str:
    """Canonical Laue-orbit key "a|b|c" (the largest equivalent index triple).

    For mmm with Friedel pairs this is "|h|||k|||l|", the key of the reference implementation, so their
    folds agree row for row when they share a salt.
    """
    h = np.asarray(hkl, dtype=float)
    rotations = [np.eye(3)]
    for operation in symmetry_ops or []:
        rotation = operation.get("rot") if isinstance(operation, dict) else operation[0]
        rotations.append(np.asarray(rotation, dtype=float))
    best = None
    for rotation in rotations:
        transformed = np.linalg.inv(rotation).T @ h
        rounded = np.rint(transformed).astype(int)
        if not np.allclose(transformed, rounded, rtol=0.0, atol=1e-7):
            raise ValueError("symmetry operation did not map a Miller index to integer values")
        for cand in ((tuple(int(v) for v in rounded),) + ((tuple(-int(v) for v in rounded),) if include_friedel else ())):
            if best is None or cand > best:
                best = cand
    return "|".join(str(v) for v in best)


def orbit_keys(hkls, symmetry_ops=None, include_friedel: bool = True) -> np.ndarray:
    """Return the orbit_key of every Miller index as an object array."""
    return np.asarray([orbit_key(h, symmetry_ops, include_friedel) for h in np.asarray(hkls)], dtype=object)


def hash_fold(keys: Iterable[str], salt: str, n_folds: int) -> np.ndarray:
    """Fold index int(sha256(salt|key)[:12], 16) mod n_folds for every key."""
    if int(n_folds) < 2:
        raise ValueError("n_folds must be at least 2")
    cache: dict[str, int] = {}
    out = []
    for key in keys:
        if key not in cache:
            cache[key] = int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:12], 16) % int(n_folds)
        out.append(cache[key])
    return np.asarray(out, dtype=int)


def free_mask(keys: Iterable[str], salt: str = DEFAULT_FREE_SALT, free_fraction: float = 0.10) -> np.ndarray:
    """Crystal-independent free set: fold 0 of K = round(1 / free_fraction) salted orbit folds."""
    if not 0.0 < float(free_fraction) < 1.0:
        raise ValueError("free_fraction must be between 0 and 1")
    n_folds = max(2, int(round(1.0 / float(free_fraction))))
    return hash_fold(list(keys), salt, n_folds) == 0


def load_orbit_exclusion(path: str | Path) -> dict[str, Any]:
    """Load a lockbox file (orbits never fitted, selected on or reported); verifies its SHA-256."""
    doc = json.loads(Path(path).read_text())
    listing = "\n".join(doc["lockbox_orbits"]).encode()
    if hashlib.sha256(listing).hexdigest() != doc["lockbox_sha256"]:
        raise ValueError(f"orbit exclusion list hash mismatch in {path}")
    return {"orbits": set(doc["lockbox_orbits"]), "sha256": doc["lockbox_sha256"],
            "shell_dmin": [float(s["d_min"]) for s in doc.get("shells", [])], "path": str(path)}


def shell_index(d, shell_dmin: list[float] | None = None, n_shells: int = 5) -> np.ndarray:
    """Resolution shell per row (0 = lowest resolution); equal-count shells when no boundaries are given."""
    d = np.asarray(d, dtype=float)
    if not shell_dmin:
        qs = np.quantile(d, np.linspace(1.0, 0.0, n_shells + 1)[1:-1])
        shell_dmin = [float(q) for q in qs] + [float(np.min(d))]
    out = np.full(len(d), len(shell_dmin) - 1, dtype=int)
    for s in range(len(shell_dmin) - 1, -1, -1):
        out[d >= shell_dmin[s]] = s
    return out


# ----------------------------------------------------------------------------- metrics

def r1(fo, fc, mask) -> float:
    """Return the amplitude R1 in percent, 100 sum |Fo - Fc| / sum Fo, over the masked rows; NaN when there are none."""
    mask = np.asarray(mask, dtype=bool)
    if not np.any(mask):
        return float("nan")
    denominator = float(np.sum(fo[mask]))
    if not math.isfinite(denominator) or denominator <= 0.0:
        return float("nan")
    return float(np.sum(np.abs(fo[mask] - fc[mask])) / denominator * 100.0)


def r_metrics(y, sig, pred, sel=None) -> dict[str, Any]:
    """Amplitude R1 in percent: R_all on every selected row, R_obs on Io > 3 sigma."""
    y = np.asarray(y, dtype=float)
    sig = np.asarray(sig, dtype=float)
    pred = np.asarray(pred, dtype=float)
    sel = np.ones(len(y), dtype=bool) if sel is None else np.asarray(sel, dtype=bool)
    sel = sel & np.isfinite(pred)
    fo = np.sqrt(np.maximum(y, 0.0))
    fc = np.sqrt(np.maximum(np.nan_to_num(pred), 0.0))
    obs = y > 3.0 * sig
    return {"r_all": r1(fo, fc, sel), "r_obs": r1(fo, fc, sel & obs), "n_all": int(sel.sum()), "n_obs": int((sel & obs).sum())}


def _reflection_group_shells(group_index, row_shells, group_count):
    """Assign each reflection group to the lower median of its observation shell labels.

    Equivalent observations can straddle a resolution-bin boundary when crystals
    have slightly different unit cells. One group must still be resampled as a
    whole. The median assignment is independent of row order; for an even tie,
    the lower label selects the lower-resolution shell.
    """
    group_shells = np.empty(group_count, dtype=int)
    for index in range(group_count):
        labels = np.sort(row_shells[group_index == index])
        group_shells[index] = labels[(len(labels) - 1) // 2]
    return group_shells


def paired_orbit_bootstrap(y, sig, pred_a, pred_b, orbit, shell, sel, n_boot: int = N_BOOT, seed: int = BOOTSTRAP_SEED):
    """Paired, shell-stratified merged-orbit bootstrap of R(a) - R(b) in pp for R_all and R_obs.

    pred_b may be None for a single-arm 95% CI of R(a). Whole reflection groups
    are resampled within shells. A group spanning shell labels uses their lower
    median, so the result does not depend on observation order. A ratio with no
    observed amplitude is undefined. If any resample has an undefined ratio,
    its interval is unavailable; draws are neither replaced nor discarded.
    """
    y = np.asarray(y, dtype=float)
    sig = np.asarray(sig, dtype=float)
    observed_amplitude = np.sqrt(np.maximum(y, 0.0))
    amplitude_a = np.sqrt(np.maximum(np.nan_to_num(np.asarray(pred_a, dtype=float)), 0.0))
    amplitude_b = None if pred_b is None else np.sqrt(np.maximum(np.nan_to_num(np.asarray(pred_b, dtype=float)), 0.0))
    sel = np.asarray(sel, dtype=bool) & np.isfinite(np.asarray(pred_a, dtype=float))
    if pred_b is not None:
        sel &= np.isfinite(np.asarray(pred_b, dtype=float))
    if not sel.any():
        raise ValueError("paired_orbit_bootstrap: no selected rows")
    reflection_groups, group_index = np.unique(np.asarray(orbit, dtype=object)[sel].astype(str), return_inverse=True)
    group_shells = _reflection_group_shells(group_index, np.asarray(shell)[sel], len(reflection_groups))
    rng = np.random.default_rng(seed)
    resample_counts = np.zeros((n_boot, len(reflection_groups)))
    for shell_id in np.unique(group_shells):
        members = np.where(group_shells == shell_id)[0]
        draws = rng.integers(0, len(members), size=(n_boot, len(members)))
        for resample_index in range(n_boot):
            resample_counts[resample_index] += np.bincount(members[draws[resample_index]], minlength=len(reflection_groups))
    result: dict[str, Any] = {}
    observed_mask = (y > 3.0 * sig)[sel]
    for metric_name, metric_mask in (("r_all", np.ones(int(sel.sum()), dtype=bool)), ("r_obs", observed_mask)):
        denominator = np.bincount(group_index, weights=np.where(metric_mask, observed_amplitude[sel], 0.0), minlength=len(reflection_groups))
        total_amplitude = float(denominator.sum())
        sampled_amplitude = resample_counts @ denominator
        undefined = ~np.isfinite(sampled_amplitude) | (sampled_amplitude <= 0.0)
        undefined_count = int(np.count_nonzero(undefined))
        if not math.isfinite(total_amplitude) or total_amplitude <= 0.0:
            point = {"value": float("nan")} if amplitude_b is None else {
                "delta": float("nan"), "a": float("nan"), "b": float("nan")}
            result[metric_name] = {**point, "ci95": [float("nan"), float("nan")],
                                  "ci_status": "no_positive_observed_amplitude", "n_undefined_resamples": int(n_boot)}
            continue
        numerator_a = np.bincount(group_index, weights=np.where(metric_mask, np.abs(observed_amplitude[sel] - amplitude_a[sel]), 0.0), minlength=len(reflection_groups))
        resampled_a = np.divide(100.0 * (resample_counts @ numerator_a), sampled_amplitude,
                                out=np.full(int(n_boot), np.nan), where=~undefined)
        point_a = 100.0 * numerator_a.sum() / total_amplitude
        if amplitude_b is None:
            result[metric_name] = {"value": float(point_a), "ci95": [float(np.percentile(resampled_a, 2.5)), float(np.percentile(resampled_a, 97.5))]}
        else:
            numerator_b = np.bincount(group_index, weights=np.where(metric_mask, np.abs(observed_amplitude[sel] - amplitude_b[sel]), 0.0), minlength=len(reflection_groups))
            resampled_b = np.divide(100.0 * (resample_counts @ numerator_b), sampled_amplitude,
                                    out=np.full(int(n_boot), np.nan), where=~undefined)
            point_b = 100.0 * numerator_b.sum() / total_amplitude
            difference = resampled_a - resampled_b
            result[metric_name] = {"delta": float(point_a - point_b), "ci95": [float(np.percentile(difference, 2.5)), float(np.percentile(difference, 97.5))],
                         "a": float(point_a), "b": float(point_b)}
        if undefined_count:
            result[metric_name].update(ci95=[float("nan"), float("nan")],
                                       ci_status="undefined_bootstrap_resamples", n_undefined_resamples=undefined_count)
    result["n_rows"] = int(sel.sum())
    result["n_orbits"] = int(len(reflection_groups))
    result["n_boot"] = int(n_boot)
    result["seed"] = int(seed)
    return result


def orbit_sign_test(y, pred_model, pred_inverse, orbit, sel) -> dict[str, Any]:
    """Per-orbit sign of sum(|Fo - Fc_inverse| - |Fo - Fc_model|); z > 0 favours the model hand."""
    y = np.asarray(y, dtype=float)
    pm = np.asarray(pred_model, dtype=float)
    pi = np.asarray(pred_inverse, dtype=float)
    sel = np.asarray(sel, dtype=bool) & np.isfinite(pm) & np.isfinite(pi)
    fo = np.sqrt(np.maximum(y, 0.0))
    fm = np.sqrt(np.maximum(np.nan_to_num(pm), 0.0))
    fi = np.sqrt(np.maximum(np.nan_to_num(pi), 0.0))
    uniq, inv = np.unique(np.asarray(orbit, dtype=object)[sel].astype(str), return_inverse=True)
    diff = (np.abs(fo - fi) - np.abs(fo - fm))[sel]
    s = np.bincount(inv, weights=diff, minlength=len(uniq))
    n_pos, n_neg = int((s > 0).sum()), int((s < 0).sum())
    z = (n_pos - n_neg) / math.sqrt(max(n_pos + n_neg, 1))
    return {"orbit_sign_z": float(z), "orbits_model_better": n_pos, "orbits_inverse_better": n_neg, "n_orbits": int(len(uniq)),
            "n_rows": int(sel.sum())}


# ----------------------------------------------------------------------------- scale refits on training rows

def _clamped(ang, train, clamp: bool):
    """Return the angles clipped to the range of the training rows when clamp is set, else unchanged."""
    ang = np.asarray(ang, dtype=float)
    return np.clip(ang, np.min(ang[train]), np.max(ang[train])) if clamp else ang


def pspline_fit_predict(y, sig, raw, ang, group, train, scale_cfg: dict | None = None, clamp: bool = True) -> dict[str, Any]:
    """P-spline frame scales fitted on train rows only; returns prediction for every row plus EDF."""
    cfg = dict(scale_cfg or {})
    train = np.asarray(train, dtype=bool)
    fitter = PSplineScaleFitter(n_basis=int(cfg.get("n_basis", 20)), objective=cfg.get("objective", "row_mr"),
                                floor_factor=float(cfg.get("floor_factor", 1e-12)))
    sol = fitter.fit(y, sig, raw, _clamped(ang, train, clamp), group=group, fit_mask=train,
                     lambda_smooth=float(cfg.get("lambda", 1.0)))
    return {"prediction": np.asarray(sol["prediction"], dtype=float), "edf": float(sol["edf"]), "lambda": float(sol["lambda"])}


def group_fit_predict(y, sig, raw, group, train) -> dict[str, Any]:
    """Free per-group (per-frame) scales: weighted LS s_g = sum(w y r) / sum(w r^2) on train rows, w = 1/sigma^2.

    Groups without training rows get the median fitted scale (reported as n_groups_clamped).
    """
    y = np.asarray(y, dtype=float)
    raw = np.asarray(raw, dtype=float)
    w = 1.0 / np.maximum(np.asarray(sig, dtype=float), 1e-12) ** 2
    group = np.asarray(group)
    train = np.asarray(train, dtype=bool)
    uniq, inv = np.unique(group, return_inverse=True)
    num = np.bincount(inv, weights=np.where(train, w * y * raw, 0.0), minlength=len(uniq))
    den = np.bincount(inv, weights=np.where(train, w * raw * raw, 0.0), minlength=len(uniq))
    ok = den > 0
    scale = np.full(len(uniq), np.nan)
    scale[ok] = num[ok] / den[ok]
    fill = float(np.median(scale[ok])) if ok.any() else 1.0
    scale[~ok] = fill
    return {"prediction": scale[inv] * raw, "n_scales": int(ok.sum()), "n_groups_clamped": int((~ok).sum())}


def select_pspline_by_inner_cv(y, sig, raw, ang, group, train, keys, grid: dict | None = None, salt: str = INNER_CV_SALT,
                               clamp: bool = True, objective: str = "row_mr") -> dict[str, Any]:
    """Pick (n_basis, lambda) by inner 3-fold orbit CV R_all restricted to train rows."""
    grid = grid or PSPLINE_GRID
    train = np.asarray(train, dtype=bool)
    inner = hash_fold(list(np.asarray(keys, dtype=object)), salt, 3)
    fo = np.sqrt(np.maximum(np.asarray(y, dtype=float), 0.0))
    scores = []
    for nb in grid["n_basis"]:
        for lam in grid["lambda"]:
            cfg = {"n_basis": int(nb), "lambda": float(lam), "objective": objective}
            oof = np.full(len(fo), np.nan)
            for h in range(3):
                tr = train & (inner != h)
                va = train & (inner == h)
                if not tr.any() or not va.any():
                    continue
                oof[va] = pspline_fit_predict(y, sig, raw, ang, group, tr, cfg, clamp)["prediction"][va]
            m = train & np.isfinite(oof)
            scores.append((r1(fo, np.sqrt(np.maximum(np.nan_to_num(oof), 0.0)), m), cfg))
    best = min(scores, key=lambda item: item[0])
    return {"scale": best[1], "inner_cv_r_all": float(best[0]),
            "grid": [{"n_basis": c["n_basis"], "lambda": c["lambda"], "inner_cv_r_all": float(s)} for s, c in scores]}


def cv_predictions(fit_predict, n_rows: int, folds: np.ndarray, eligible: np.ndarray | None = None) -> np.ndarray:
    """Out-of-fold predictions: for each fold f fit on eligible rows not in f and predict rows in f.

    fit_predict(train_mask) must return a prediction for every row. Rows outside eligible stay NaN.
    """
    folds = np.asarray(folds, dtype=int)
    eligible = np.ones(n_rows, dtype=bool) if eligible is None else np.asarray(eligible, dtype=bool)
    pred = np.full(n_rows, np.nan)
    for f in np.unique(folds[eligible]):
        train = eligible & (folds != f)
        val = eligible & (folds == f)
        if not train.any() or not val.any():
            continue
        pred[val] = np.asarray(fit_predict(train), dtype=float)[val]
    return pred
