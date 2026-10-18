"""Per-crystal correction, overlap scaling, symmetry keys, and endpoint merging. This preserves existing formal/overdispersion rules; it does not model all correlations between corrected observations."""

from __future__ import annotations

import math
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from grond.engine.bloch_engine import sha256
from grond.engine.endpoint_modulation import modulation_manifest_fields
from grond.refine.shelx_input import ABSOLUTE_HKL_KAPPA

ENDPOINT_Q_RATIO_BOUNDS = (0.05, 20.0)


def safe_float(v, default=float("nan")):
    """Return v as a float, or default when it cannot be converted."""

    try:
        return float(v)
    except (TypeError, ValueError):
        return default


def file_provenance(path: Path) -> dict[str, object]:
    """Return a stable source-artifact identity for a combined endpoint."""

    resolved = path.expanduser().resolve()
    exists = resolved.is_file()
    return {
        "path": str(resolved),
        "exists": exists,
        "size_bytes": int(resolved.stat().st_size) if exists else None,
        "sha256": sha256(resolved) if exists else "",
    }


def scale_contract_provenance(summary: dict, rows: list[dict]) -> dict[str, object]:
    """Describe the scale table actually materialized in one source run."""

    best_native = summary.get("best_native") or {}
    nuisance = best_native.get("nuisance") or {}
    row_model_fields = ("scale_replay_model", "scale_compression_model")
    model_names: list[str] = []
    source = ""
    for field in row_model_fields:
        values = sorted(
            {
                str(row.get(field, "")).strip()
                for row in rows
                if str(row.get(field, "")).strip()
            }
        )
        if values:
            model_names = values
            source = f"row_table.{field}"
            break

    if not model_names:
        replay_candidate = (summary.get("scale_contract_replay") or {}).get("candidate") or {}
        replay_name = str(replay_candidate.get("name", "")).strip()
        compression_candidate = (summary.get("scale_compression") or {}).get("selected_output") or {}
        compression_name = str(compression_candidate.get("name", "")).strip()
        candidates = (
            (replay_name, "summary.scale_contract_replay.candidate.name"),
            (str(nuisance.get("scale_replay_model", "")).strip(), "summary.best_native.nuisance.scale_replay_model"),
            (compression_name, "summary.scale_compression.selected_output.name"),
            (str(nuisance.get("scale_compression_model", "")).strip(), "summary.best_native.nuisance.scale_compression_model"),
        )
        for candidate, candidate_source in candidates:
            if candidate:
                model_names = [candidate]
                source = candidate_source
                break

    scale_model = best_native.get("scale_model") if isinstance(best_native.get("scale_model"), dict) else {}
    if not model_names and scale_model.get("kind") == "fixed_design":
        model_names = [str(scale_model.get("name", "fixed_design"))]
        source = "summary.best_native.scale_model"
    if not model_names and scale_model.get("kind") == "pspline":
        # Smooth frame scales have no group-width contract; parameter_count stays ceil(EDF), edf is exact.
        model_names = [f"pspline_n{scale_model.get('n_basis')}_lambda{safe_float(scale_model.get('lambda')):g}"]
        source = "summary.best_native.scale_model"

    if not model_names:
        group_width = safe_float(nuisance.get("group_width_deg"))
        detector_bins = safe_float(nuisance.get("detector_scale_bins"))
        if math.isfinite(group_width) or math.isfinite(detector_bins):
            width_text = f"{group_width:g}" if math.isfinite(group_width) else "unknown"
            detector_text = str(int(detector_bins)) if math.isfinite(detector_bins) else "unknown"
            model_names = [f"group_width_{width_text}_deg_det_{detector_text}"]
            source = "summary.best_native.nuisance"

    count_value = safe_float(best_native.get("scale_count"))
    count = int(count_value) if math.isfinite(count_value) and count_value >= 0.0 else None
    out = {
        "model": model_names[0] if len(model_names) == 1 else ("mixed" if model_names else "unknown"),
        "models": model_names,
        "model_source": source or "unavailable",
        "parameter_count": count,
        "fit_split": str(best_native.get("scale_fit_split", "")) or "unknown",
    }
    if scale_model.get("kind") in {"pspline", "fixed_design"}:
        out["edf"] = safe_float(scale_model.get("edf"))
    return out


def d_spacing_from_model(model: dict, h: int, k: int, ell: int) -> float:
    """Return d(hkl) from the parsed model cell without changing HKL signs."""

    try:
        a, b, c, alpha_deg, beta_deg, gamma_deg = [float(value) for value in model["cell"]]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("parsed model does not contain a usable six-value unit cell") from exc
    alpha, beta, gamma = np.deg2rad([alpha_deg, beta_deg, gamma_deg])
    direct_metric = np.asarray(
        [
            [a * a, a * b * math.cos(gamma), a * c * math.cos(beta)],
            [a * b * math.cos(gamma), b * b, b * c * math.cos(alpha)],
            [a * c * math.cos(beta), b * c * math.cos(alpha), c * c],
        ],
        dtype=float,
    )
    try:
        reciprocal_metric = np.linalg.inv(direct_metric)
    except np.linalg.LinAlgError as exc:
        raise ValueError("parsed model unit cell is singular") from exc
    hkl = np.asarray([int(h), int(k), int(ell)], dtype=float)
    inv_d2 = float(hkl @ reciprocal_metric @ hkl)
    if not math.isfinite(inv_d2) or inv_d2 <= 0.0:
        raise ValueError(f"cannot calculate a finite d spacing for HKL {(h, k, ell)}")
    return 1.0 / math.sqrt(inv_d2)


def filter_native_rows_by_resolution(rows: list[dict], model: dict, min_d_spacing: float) -> tuple[list[dict], dict]:
    """Apply a shared-model-cell endpoint/reporting d cutoff to native rows.

    Native tables may carry ``d_A`` calculated from each experiment's fitted
    cell.  A merged endpoint has one structure cell, so its resolution
    boundary must instead be calculated from signed HKL and that shared cell.
    """

    cutoff = float(min_d_spacing)
    kept: list[dict] = []
    input_d_spacings: list[float] = []
    retained_d_spacings: list[float] = []
    stored_membership_disagreements = 0
    for row in rows:
        try:
            h, k, ell = (int(float(row[key])) for key in ("h", "k", "l"))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError("resolution filtering needs integer h,k,l in every native row") from exc
        d_value = d_spacing_from_model(model, h, k, ell)
        input_d_spacings.append(d_value)
        stored_d = safe_float(row.get("d_A"))
        if cutoff > 0.0 and math.isfinite(stored_d) and stored_d > 0.0 and ((stored_d + 1e-12 >= cutoff) != (d_value + 1e-12 >= cutoff)):
            stored_membership_disagreements += 1
        if cutoff <= 0.0 or d_value + 1e-12 >= cutoff:
            kept.append(row)
            retained_d_spacings.append(d_value)
    return kept, {
        "requested_min_d_spacing_A": cutoff,
        "input_rows": len(rows),
        "kept_rows": len(kept),
        "retained_rows": len(kept),
        "dropped_rows": len(rows) - len(kept),
        "retained_fraction": float(len(kept) / len(rows)) if rows else float("nan"),
        "actual_input_min_d_spacing_A": min(input_d_spacings) if input_d_spacings else float("nan"),
        "actual_retained_min_d_spacing_A": min(retained_d_spacings) if retained_d_spacings else float("nan"),
        "actual_retained_max_d_spacing_A": max(retained_d_spacings) if retained_d_spacings else float("nan"),
        "d_spacings_calculated_from_model": len(rows),
        "stored_d_A_membership_disagreements": stored_membership_disagreements,
        "cell_A_deg": [float(value) for value in model["cell"]],
    }


def filter_hkl_rows_by_resolution(rows: list[tuple], model: dict, min_d_spacing: float) -> list[tuple]:
    """Filter a corrected-HKL source with the same crystallographic cutoff."""

    cutoff = float(min_d_spacing)
    if cutoff <= 0.0:
        return rows
    return [
        row
        for row in rows
        if d_spacing_from_model(model, int(row[0]), int(row[1]), int(row[2])) + 1e-12 >= cutoff
    ]


def hkl_resolution_stats(rows: list[tuple], model: dict) -> dict[str, object]:
    """Report the discrete resolution range represented by HKL rows."""

    spacings = [d_spacing_from_model(model, int(row[0]), int(row[1]), int(row[2])) for row in rows]
    return {
        "rows": len(rows),
        "actual_min_d_spacing_A": min(spacings) if spacings else float("nan"),
        "actual_max_d_spacing_A": max(spacings) if spacings else float("nan"),
        "cell_A_deg": [float(value) for value in model["cell"]],
    }


def selected_native_wght(summary: dict) -> tuple[float, float]:
    """Read the native row weights written by native_row_regularize."""

    block = summary.get("shelx_weight") or {}
    a = safe_float(
        block.get("selected_native_a", block.get("source_a", block.get("a"))),
        0.126534,
    )
    b = safe_float(
        block.get("selected_native_b", block.get("source_b", block.get("b"))),
        0.0,
    )
    return a, b


def apply_native_row_weight(row: dict, weight: float) -> dict:
    """Apply one dataset-policy raw-row weight to an unmodified sigma."""

    out = dict(row)
    sigma = safe_float(out.get("sigma_eff"))
    if math.isfinite(sigma) and sigma > 0.0 and weight > 0.0 and abs(weight - 1.0) > 1e-12:
        out["sigma_eff"] = f"{sigma / math.sqrt(weight):.17g}"
    return out


def canonical_p222_friedel(h, k, ell):
    """Return the smallest equivalent of (h, k, ell) under point group 222 and Friedel inversion (Laue mmm)."""

    variants = [
        (h, k, ell),
        (-h, -k, ell),
        (h, -k, -ell),
        (-h, k, -ell),
    ]
    variants += [(-a, -b, -c) for a, b, c in variants]
    return min(variants)


def reciprocal_equivalent_keys(
    h: int,
    k: int,
    ell: int,
    symmetry_ops: list[dict],
    *,
    merge_friedel: bool = True,
) -> list[tuple[int, int, int]]:
    """Return reciprocal equivalents from the model symmetry operations."""

    hkl = np.asarray([int(h), int(k), int(ell)], dtype=float)
    keys: list[tuple[int, int, int]] = []
    for op in symmetry_ops or [{"rot": np.eye(3).tolist()}]:
        rot = np.asarray(op.get("rot", np.eye(3)), dtype=float)
        for mat in (rot, np.linalg.inv(rot)):
            equiv = np.rint(hkl @ mat).astype(int)
            key = tuple(int(v) for v in equiv)
            keys.append(key)
            if merge_friedel:
                keys.append(tuple(-v for v in key))
    return sorted(set(keys))


def canonical_model_friedel(h: int, k: int, ell: int, symmetry_ops: list[dict]) -> tuple[int, int, int]:
    """Return the smallest equivalent of (h, k, ell) under the model symmetry operations and Friedel inversion."""

    return min(reciprocal_equivalent_keys(h, k, ell, symmetry_ops, merge_friedel=True))


def canonical_model_symmetry(h: int, k: int, ell: int, symmetry_ops: list[dict]) -> tuple[int, int, int]:
    """Return the smallest equivalent of (h, k, ell) under the model symmetry operations only.

    Unlike canonical_model_friedel, no Friedel inversion is added.
    """

    return min(reciprocal_equivalent_keys(h, k, ell, symmetry_ops, merge_friedel=False))


def endpoint_key_function(mode: str, symmetry_ops: list[dict]):
    """Return the HKL key used for endpoint scaling or merging."""

    if mode == "exact_signed_hkl":
        return lambda h, k, ell: (int(h), int(k), int(ell))
    if mode == "model_symmetry":
        return lambda h, k, ell: canonical_model_symmetry(int(h), int(k), int(ell), symmetry_ops)
    if mode == "model_symmetry_friedel":
        return lambda h, k, ell: canonical_model_friedel(int(h), int(k), int(ell), symmetry_ops)
    raise ValueError(f"unknown endpoint merge key {mode!r}")


def shelx_weights(obs, sig, pred, a, b):
    """Return SHELXL WGHT weights 1 / (sigma^2 + (a P)^2 + b P), P = (max(Iobs, 0) + 2 max(Icalc, 0)) / 3.

    a and b may be per-row arrays.
    """

    p = (np.maximum(obs, 0.0) + 2.0 * np.maximum(pred, 0.0)) / 3.0
    return 1.0 / np.maximum(
        sig * sig + (np.asarray(a, dtype=float) * p) ** 2 + np.asarray(b, dtype=float) * p,
        1e-30,
    )


def metrics(rows: list[dict], pred_col: str, wght_a: float, wght_b: float):
    """Return R1_all, R1_gt (target_I > 2 sigma_eff), wR2_all, CC and mean |sqrt(w) residual| of one prediction column.

    R values are fractions. Weights are SHELXL WGHT weights, with each row's native_wght_a and native_wght_b when
    recorded.
    """

    obs = np.asarray([safe_float(r["target_I"]) for r in rows], dtype=float)
    sig = np.asarray([safe_float(r["sigma_eff"]) for r in rows], dtype=float)
    pred = np.asarray([safe_float(r[pred_col]) for r in rows], dtype=float)
    a = np.asarray([safe_float(r.get("native_wght_a"), wght_a) for r in rows], dtype=float)
    b = np.asarray([safe_float(r.get("native_wght_b"), wght_b) for r in rows], dtype=float)
    m = np.isfinite(obs) & np.isfinite(sig) & np.isfinite(pred) & (sig > 0)
    obs = obs[m]
    sig = sig[m]
    pred = pred[m]
    a = a[m]
    b = b[m]
    if len(obs) == 0:
        return {"n": 0, "n_gt": 0, "R1_all": float("nan"), "R1_gt": float("nan"), "wR2_all": float("nan"), "CC": float("nan"), "mw_abs": float("nan")}
    gt = obs > 2.0 * sig
    fobs = np.sqrt(np.maximum(obs, 0.0))
    fcalc = np.sqrt(np.maximum(pred, 0.0))

    def r1(mask):
        """Return sum |Fobs - Fcalc| / sum Fobs over the masked rows, NaN when there are none."""

        if not np.any(mask):
            return float("nan")
        return float(np.sum(np.abs(fobs[mask] - fcalc[mask])) / max(np.sum(fobs[mask]), 1e-30))

    w = shelx_weights(obs, sig, pred, a, b)
    res = obs - pred
    wr2 = float(np.sqrt(np.sum(w * res * res) / max(np.sum(w * obs * obs), 1e-30)))
    cc = float(np.corrcoef(obs, pred)[0, 1]) if len(obs) > 2 and np.std(obs) > 0 and np.std(pred) > 0 else float("nan")
    return {
        "n": int(len(obs)),
        "n_gt": int(np.count_nonzero(gt)),
        "R1_all": r1(np.ones_like(gt, dtype=bool)),
        "R1_gt": r1(gt),
        "wR2_all": wr2,
        "CC": cc,
        "mw_abs": float(np.mean(np.sqrt(w) * np.abs(res))),
    }


def split_metrics(rows: list[dict], pred_col: str, wght_a: float, wght_b: float):
    """Return metrics of one prediction column on all rows and on the work and free rows (by the split column)."""

    out = {}
    for split in ("all", "work", "free"):
        use = rows if split == "all" else [r for r in rows if r.get("split") == split]
        out[split] = metrics(use, pred_col, wght_a, wght_b)
    return out


def native_metrics_by_split(rows: list[dict], wght_a: float, wght_b: float) -> dict[str, dict[str, dict]]:
    """Return native metrics in the single-run summary's split-first shape."""

    dyn = split_metrics(rows, "dyn_scaled", wght_a, wght_b)
    born = split_metrics(rows, "born_scaled", wght_a, wght_b)
    return {
        split: {"dyn": dyn[split], "born": born[split]}
        for split in ("all", "work", "free")
    }


def endpoint_correction_denominators(
    target: np.ndarray,
    dyn_raw: np.ndarray,
    dyn_scaled: np.ndarray,
    q: np.ndarray,
    lambda_q: float,
    selected: np.ndarray,
    *,
    q_policy: str = "clip",
    rescale: str = "median-iobs",
    fc2: np.ndarray | None = None,
    kappa: float = ABSOLUTE_HKL_KAPPA,
    restore: np.ndarray | None = None,
    scales: np.ndarray | None = None,
    reference_median: float | None = None,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Return per-row divisors d (Icorr = y/d), the rows kept, and an audit; shared by both endpoint writers.

    median-iobs (legacy): d = s * q^lambda with s = dyn_scaled/dyn_raw (or the caller's fitted ``scales``),
    later rescaled to the median IOBS.
    absolute: d = pred * q^(lambda-1) / (kappa |F|^2), i.e. Icorr = kappa |F|^2 y / pred for lambda = 1, the
    absolute-scale handoff (a perfect model gives Icorr = kappa |F|^2 on one scale for every
    crystal).  pred = dyn_scaled / restore is the scaled prediction in the IOBS units of y.  Rows with
    |F|^2 = 0 carry no information and are dropped.

    q_policy clip (legacy) clips q into [0.05, 20] (absolute: the divisor relative to its median) and keeps the
    row. exclude drops rows whose divisor relative to the median of the selected rows lies outside (0.05, 20)
    (rel = pred/|U_g|^2 relative to its training median) and counts them.

    ``reference_median`` replaces the median of the selected rows by another row set's (the work rows' divisor
    median), so held-out rows are clipped or excluded by exactly the rule the work rows met.
    """

    lo, hi = ENDPOINT_Q_RATIO_BOUNDS
    selected = np.asarray(selected, dtype=bool).copy()
    audit: dict[str, object] = {
        "q_policy": str(q_policy),
        "rescale": str(rescale),
        "q_ratio_bounds": [lo, hi],
        "n_rows_selected": int(np.count_nonzero(selected)),
        "n_excluded_q_rows": 0,
        "n_zero_fcalc_rows": 0,
    }
    if q_policy not in {"clip", "exclude"}:
        raise ValueError(f"unknown endpoint q policy {q_policy!r}; use clip or exclude")
    if rescale == "absolute":
        if fc2 is None:
            raise ValueError("absolute endpoint intensity scale needs |Fcalc|^2 of the current model for every row")
        fc2 = np.asarray(fc2, dtype=float)
        zero_f = selected & ~(np.isfinite(fc2) & (fc2 > 0.0))
        audit["n_zero_fcalc_rows"] = int(np.count_nonzero(zero_f))
        selected &= ~zero_f
        safe_fc2 = np.where(selected, fc2, 1.0)
        pred = dyn_scaled if restore is None else dyn_scaled / np.asarray(restore, dtype=float)
        denom = pred * np.power(np.maximum(q, 1e-30), float(lambda_q) - 1.0) / (float(kappa) * safe_fc2)
        audit["kappa"] = float(kappa)
    elif rescale in {"median-iobs", "none"}:
        if scales is None:
            scales = np.maximum(dyn_scaled / np.maximum(dyn_raw, 1e-30), 1e-30)
        q_used = np.clip(q, lo, hi) if q_policy == "clip" else q
        denom = scales * (q_used ** float(lambda_q))
    else:
        raise ValueError(f"unknown endpoint rescale mode {rescale!r}")
    denom = np.where(np.isfinite(denom) & (denom > 0.0), denom, np.nan)
    selected &= np.isfinite(denom)
    median = float(np.median(denom[selected])) if np.any(selected) else float("nan")
    if reference_median is not None:
        audit["divisor_median_own"] = median
        median = float(reference_median)
        audit["divisor_median_source"] = "reference"
    audit["divisor_median"] = median
    if np.any(selected):
        ratio = denom / median
        outside = selected & ~((ratio > lo) & (ratio < hi))
        if q_policy == "exclude":
            audit["n_excluded_q_rows"] = int(np.count_nonzero(outside))
            selected &= ~outside
        elif rescale == "absolute":
            audit["n_clipped_q_rows"] = int(np.count_nonzero(outside))
            denom = np.where(selected, median * np.clip(ratio, lo, hi), denom)
    audit["n_rows_kept"] = int(np.count_nonzero(selected))
    return np.maximum(np.nan_to_num(denom, nan=1.0), 1e-30), selected, audit


def endpoint_rows_and_corrections_from_native(
    rows: list[dict],
    lambda_q: float,
    *,
    q_policy: str = "clip",
    rescale: str = "median-iobs",
    model: dict | None = None,
    kappa: float = ABSOLUTE_HKL_KAPPA,
    audit: dict | None = None,
    reference: Mapping[str, object] | None = None,
):
    """Return corrected endpoint rows and their applied intensity multipliers.

    The defaults are the legacy contract (q clipped into [0.05, 20], median-IOBS rescale). ``rescale=
    "absolute"`` needs the current ``model``; its |Fcalc|^2 per row puts the rows on kappa*|F|^2.  ``audit``
    (a dict) receives the kept/excluded row counts.

    ``reference`` is the audit of another correction of the same crystal (its work rows): its divisor median and
    median-IOBS rescale factor are used instead of this row set's own, so every row is corrected exactly as it would
    have been among the reference rows (the held-out endpoint validation corrects the free rows this way).
    """

    hkl = [(int(r["h"]), int(r["k"]), int(r["l"])) for r in rows]
    target = np.asarray([safe_float(r["target_I"]) for r in rows], dtype=float)
    sig = np.asarray([safe_float(r["sigma_eff"]) for r in rows], dtype=float)
    restore = []
    for row in rows:
        mode = str(row.get("target_mode", "iobs"))
        value = row.get("source_to_target_factor")
        if value is None and mode != "iobs":
            raise ValueError(f"cannot restore {mode} endpoint units: row lacks source_to_target_factor")
        factor = 1.0 if value is None else safe_float(value)
        if not math.isfinite(factor) or factor <= 0.0:
            raise ValueError(f"invalid source_to_target_factor for endpoint export: {value!r}")
        restore.append(factor)
    restore = np.asarray(restore, dtype=float)
    target = target / restore
    sig = sig / restore
    dyn_raw = np.asarray([safe_float(r["dyn_raw"]) for r in rows], dtype=float)
    dyn_scaled = np.asarray([safe_float(r["dyn_scaled"]) for r in rows], dtype=float)
    q = np.asarray([safe_float(r["q_dyn_over_born"]) for r in rows], dtype=float)
    valid = (
        np.isfinite(target)
        & np.isfinite(sig)
        & (sig > 0.0)
        & np.isfinite(dyn_raw)
        & (dyn_raw > 0.0)
        & np.isfinite(dyn_scaled)
        & (dyn_scaled > 0.0)
        & np.isfinite(q)
        & (q > 0.0)
    )
    if q_policy == "clip" and rescale == "median-iobs":
        # Legacy contract, kept byte-for-byte.
        q_clip = np.clip(q, 0.05, 20.0)
        scales = np.maximum(dyn_scaled / np.maximum(dyn_raw, 1e-30), 1e-30)
        denom = np.maximum(scales * (q_clip ** float(lambda_q)), 1e-30)
        if audit is not None:
            audit.update({"q_policy": "clip", "rescale": "median-iobs", "n_rows_selected": int(np.count_nonzero(valid)),
                          "n_rows_kept": int(np.count_nonzero(valid)), "n_excluded_q_rows": 0})
    else:
        fc2 = None
        if rescale == "absolute":
            if model is None:
                raise ValueError("absolute endpoint intensity scale needs the current structure model")
            from grond.bloch.model import fcalc_res_grid

            hkl_array = np.asarray(hkl, dtype=float).reshape(-1, 3)
            fc2 = np.abs(fcalc_res_grid(model, hkl_array[:, 0], hkl_array[:, 1], hkl_array[:, 2])) ** 2
        denom, valid, denom_audit = endpoint_correction_denominators(
            target,
            dyn_raw,
            dyn_scaled,
            q,
            lambda_q,
            valid,
            q_policy=q_policy,
            rescale=rescale,
            fc2=fc2,
            kappa=kappa,
            restore=restore,
            reference_median=None if reference is None else float(reference["divisor_median"]),  # type: ignore[arg-type]
        )
        if audit is not None:
            audit.update(denom_audit)
    icorr = target / denom
    sigcorr = sig / denom
    if rescale == "median-iobs" and reference is not None:
        factor = float(reference["rescale_factor"])  # type: ignore[arg-type]
    elif rescale == "median-iobs":
        src = float(np.nanmedian(target[valid])) if np.any(valid) else 1.0
        dst = float(np.nanmedian(icorr[valid])) if np.any(valid) else 1.0
        factor = src / max(dst, 1e-30)
    else:
        factor = 1.0
    icorr *= factor
    sigcorr *= factor
    if audit is not None:
        audit["rescale_factor"] = float(factor)
    selected = np.where(valid)[0]
    endpoint_rows = [(*hkl[i], float(icorr[i]), float(sigcorr[i])) for i in selected]
    corrections = np.asarray(float(factor) / denom[selected], dtype=float)
    return endpoint_rows, corrections


def apply_hkl_weight(rows, weight: float):
    """Return HKL rows with sigma divided by sqrt(weight).

    weight <= 0 drops every row; weight 1 returns them unchanged.
    """

    if weight <= 0.0:
        return []
    if abs(weight - 1.0) < 1e-12:
        return rows
    factor = 1.0 / math.sqrt(max(weight, 1e-12))
    return [(h, k, ell, intensity, sig * factor) for h, k, ell, intensity, sig in rows]


def scale_hkl_rows(rows, scale: float):
    """Return HKL rows with I multiplied by scale and sigma by |scale|."""

    scale = float(scale)
    return [(h, k, ell, intensity * scale, sig * abs(scale)) for h, k, ell, intensity, sig in rows]


def audit_combined_endpoint_modulation(
    corrections_by_dataset: dict[str, np.ndarray | None],
    interdataset_scales: dict[str, float],
    *,
    endpoint_source: str,
) -> tuple[dict[str, object], dict[str, dict[str, object]]]:
    """Audit row corrections after applying relative interdataset scales."""

    combined_corrections: list[float] = []
    per_dataset: dict[str, dict[str, object]] = {}
    for name, source_corrections in corrections_by_dataset.items():
        dataset_scale = safe_float(interdataset_scales.get(name, 1.0), float("nan"))
        if source_corrections is None:
            fields = modulation_manifest_fields([])
            fields["modulation_audit"] = {
                **dict(fields["modulation_audit"]),
                "status": "unavailable_endpoint_source_hkl",
                "reason": "row-level source intensity multipliers are not recoverable from a prewritten HKL",
            }
        else:
            adjusted = np.asarray(source_corrections, dtype=float) * dataset_scale
            fields = modulation_manifest_fields(adjusted)
            combined_corrections.extend(float(value) for value in adjusted)
        per_dataset[name] = {**fields, "interdataset_scale": dataset_scale}

    combined = modulation_manifest_fields(combined_corrections)
    if endpoint_source == "hkl":
        combined["modulation_audit"] = {
            **dict(combined["modulation_audit"]),
            "status": "unavailable_endpoint_source_hkl",
            "reason": "combined modulation requires row-level native correction factors including interdataset scales",
        }
    return combined, per_dataset


def apply_sigma_holdout_floor(meta: dict) -> dict:
    """Avoid making a dataset tighter than its own held-out residual scatter."""

    out = dict(meta)
    if not str(out.get("status", "")).startswith("ok"):
        return out
    current = safe_float(out.get("sigma_scale"))
    holdout_before = safe_float(out.get("heldout_rms_z_before"))
    if not (math.isfinite(current) and current > 0.0 and math.isfinite(holdout_before) and holdout_before > 0.0):
        return out
    new_scale = max(current, holdout_before)
    if new_scale <= current + max(1e-12, 1e-9 * abs(current)):
        out["holdout_floor_applied"] = False
        return out
    out["sigma_scale_pre_holdout_floor"] = current
    out["sigma_scale"] = float(new_scale)
    out["holdout_floor_applied"] = True
    for key in ("train", "all", "heldout"):
        before_key = f"{key}_rms_z_before"
        after_key = f"{key}_rms_z_after"
        before = safe_float(out.get(before_key))
        if math.isfinite(before):
            out[after_key] = float(before / new_scale)
        med_before_key = f"{key}_median_abs_z_before"
        med_after_key = f"{key}_median_abs_z_after"
        med_before = safe_float(out.get(med_before_key))
        if math.isfinite(med_before):
            out[med_after_key] = float(med_before / new_scale)
    return out


def estimate_overlap_scales(rows_by_dataset: dict[str, list[tuple]], iterations: int = 6, canonicalizer=None) -> dict:
    """Estimate dataset scales from HKLs observed by distinct datasets.

    Replicates are first merged within each dataset.  Every scale update uses a
    leave-one-dataset-out reference, so a dataset can neither manufacture its
    own overlap nor pull its reference toward itself.
    """

    key_for = canonicalizer or canonical_p222_friedel
    names = list(rows_by_dataset)
    scales = {name: 1.0 for name in names}
    overlap_counts = {name: 0 for name in names}
    if len(names) <= 1:
        return {"iterations": 0, "overlap_counts": overlap_counts, "scales": scales}

    merged: dict[str, dict[tuple[int, int, int], tuple[float, float]]] = {}
    contributors: dict[tuple[int, int, int], set[str]] = {}
    for name, rows in rows_by_dataset.items():
        bins: dict[tuple[int, int, int], dict[str, float]] = {}
        for h, k, ell, intensity, sigma in rows:
            value = float(intensity)
            sigma_value = abs(float(sigma))
            if not (math.isfinite(value) and value > 0.0 and math.isfinite(sigma_value) and sigma_value > 0.0):
                continue
            key = key_for(int(h), int(k), int(ell))
            weight = 1.0 / (sigma_value * sigma_value)
            bucket = bins.setdefault(key, {"w": 0.0, "wI": 0.0})
            bucket["w"] += weight
            bucket["wI"] += weight * value
        merged[name] = {
            key: (bucket["wI"] / bucket["w"], math.sqrt(1.0 / bucket["w"]))
            for key, bucket in bins.items()
            if bucket["w"] > 0.0
        }
        for key in merged[name]:
            contributors.setdefault(key, set()).add(name)

    shared_keys = {key for key, datasets in contributors.items() if len(datasets) >= 2}
    adjacency = {name: set() for name in names}
    for key in shared_keys:
        datasets = contributors[key]
        for name in datasets:
            adjacency[name].update(datasets - {name})
    components: list[set[str]] = []
    unseen = set(names)
    while unseen:
        seed = unseen.pop()
        component = {seed}
        frontier = [seed]
        while frontier:
            current = frontier.pop()
            new = adjacency[current] & unseen
            unseen -= new
            component |= new
            frontier.extend(new)
        components.append(component)

    overlap_counts = {
        name: int(sum(key in shared_keys for key in merged[name]))
        for name in names
    }
    if not shared_keys:
        return {"iterations": 0, "overlap_counts": overlap_counts, "scales": scales}

    n_iterations = max(1, int(iterations))
    for component in components:
        component_names = sorted(component)
        if len(component_names) <= 1 or not any(adjacency[name] for name in component_names):
            continue
        local_index = {name: i for i, name in enumerate(component_names)}
        equations = []
        rhs = []
        base_weights = []
        for key in shared_keys:
            present = sorted(contributors[key] & component)
            for i, left in enumerate(present):
                left_i, left_sigma = merged[left][key]
                for right in present[i + 1 :]:
                    right_i, right_sigma = merged[right][key]
                    row = np.zeros(len(component_names), dtype=float)
                    row[local_index[left]] = 1.0
                    row[local_index[right]] = -1.0
                    equations.append(row)
                    rhs.append(math.log(right_i) - math.log(left_i))
                    relative_variance = (left_sigma / left_i) ** 2 + (right_sigma / right_i) ** 2
                    base_weights.append(1.0 / max(relative_variance, 1e-12))
        if not equations:
            continue
        design = np.asarray(equations, dtype=float)
        target = np.asarray(rhs, dtype=float)
        weights = np.asarray(base_weights, dtype=float)
        robust = np.ones_like(weights)
        solution = np.zeros(len(component_names), dtype=float)
        for _ in range(n_iterations):
            sqrt_w = np.sqrt(weights * robust)
            weighted_design = design * sqrt_w[:, None]
            weighted_target = target * sqrt_w
            gauge_weight = math.sqrt(float(np.median(weights)) * max(1, len(component_names)))
            weighted_design = np.vstack([weighted_design, np.ones(len(component_names)) * gauge_weight])
            weighted_target = np.append(weighted_target, 0.0)
            solution = np.linalg.lstsq(weighted_design, weighted_target, rcond=None)[0]
            residual = design @ solution - target
            mad = float(np.median(np.abs(residual - np.median(residual))))
            if mad <= 1e-12:
                break
            cutoff = 1.345 * 1.4826 * mad
            robust = np.minimum(1.0, cutoff / np.maximum(np.abs(residual), 1e-30))
        solution -= float(np.mean(solution))
        for name, value in zip(component_names, solution, strict=False):
            scales[name] = float(math.exp(value))
    return {
        "iterations": n_iterations,
        "overlap_counts": {name: int(count) for name, count in overlap_counts.items()},
        "scales": {name: float(value) for name, value in scales.items()},
    }


def merge_hkl_rows(rows, sigma_policy: str = "formal", canonicalizer=None):
    """Merge HKL rows sharing a key into inverse-variance weighted means; returns (sorted rows, metadata).

    The merged sigma is 1 / sqrt(sum w) with w = 1/sigma^2 (sigma at least 1e-6). With sigma_policy
    chisq_overdispersion the sigma of a reflection seen more than once is multiplied by sqrt(reduced chi^2) when
    that exceeds 1. canonicalizer maps (h, k, l) to the key; the default is the exact signed HKL.
    """

    key_for = canonicalizer or (lambda h, k, ell: (int(h), int(k), int(ell)))
    bins: dict[tuple[int, int, int], dict[str, float]] = {}
    for h, k, ell, intensity, sig in rows:
        key = key_for(int(h), int(k), int(ell))
        sig = max(abs(float(sig)), 1e-6)
        w = 1.0 / (sig * sig)
        b = bins.setdefault(key, {"w": 0.0, "wI": 0.0, "n": 0, "values": [], "weights": []})
        b["w"] += w
        b["wI"] += w * float(intensity)
        b["n"] += 1
        b["values"].append(float(intensity))
        b["weights"].append(float(w))
    out = []
    inflations = []
    for key, b in bins.items():
        mean = b["wI"] / max(b["w"], 1e-30)
        sigma = (1.0 / max(b["w"], 1e-30)) ** 0.5
        inflation = 1.0
        if sigma_policy == "chisq_overdispersion" and b["n"] > 1:
            values = np.asarray(b["values"], dtype=float)
            weights = np.asarray(b["weights"], dtype=float)
            chi2 = float(np.sum(weights * (values - mean) ** 2) / max(int(b["n"]) - 1, 1))
            if math.isfinite(chi2) and chi2 > 1.0:
                inflation = math.sqrt(chi2)
                sigma *= inflation
        inflations.append(inflation)
        out.append((*key, mean, sigma))
    out.sort(key=lambda x: (x[0], x[1], x[2]))
    inflations_arr = np.asarray(inflations, dtype=float) if inflations else np.asarray([], dtype=float)
    meta = {
        "policy": sigma_policy,
        "merged_rows": int(len(out)),
        "rows_with_multiple_observations": int(sum(1 for b in bins.values() if b["n"] > 1)),
        "inflated_rows": int(np.count_nonzero(inflations_arr > 1.0)) if inflations_arr.size else 0,
        "inflation_median": float(np.median(inflations_arr)) if inflations_arr.size else float("nan"),
        "inflation_p95": float(np.percentile(inflations_arr, 95)) if inflations_arr.size else float("nan"),
        "inflation_max": float(np.max(inflations_arr)) if inflations_arr.size else float("nan"),
    }
    return out, meta
