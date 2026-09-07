"""Native structure-update objective and frozen nuisance-scale application. Re-fitting scales inside every coordinate step can hide model changes; the existing frozen-block mode is deliberately preserved."""
from __future__ import annotations

import numpy as np

from grond.legacy.engine.bloch_engine import XDSData
from grond.regularize.observations import detector_bin_indices, group_indices, row_resolution_modulation
from grond.regularize.scales import fit_frame_detector_scales, fit_group_scales, shelx_weights


def choose_objective_rows(data: XDSData, indices, masks, target, sig, n_rows: int, n_bins: int, seed: int):
    candidates = np.asarray([i for i in indices if masks["work"][i] and target[i] > 0 and np.isfinite(target[i]) and np.isfinite(sig[i])], dtype=int)
    if n_rows <= 0 or n_rows >= len(candidates):
        return np.asarray(candidates, dtype=int)
    rng = np.random.default_rng(int(seed))
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    order = candidates[np.argsort(dvals[candidates])]
    bins = np.array_split(order, max(1, int(n_bins)))
    chosen = []
    remaining = int(n_rows)
    for _bi, b in enumerate(bins):
        if len(b) == 0:
            continue
        quota = int(round(n_rows * len(b) / max(len(candidates), 1)))
        quota = max(1, min(quota, len(b), remaining))
        remaining -= quota
        # Keep the row set representative but gently favor informative rows.
        isigi = np.maximum(target[b] / np.maximum(sig[b], 1e-12), 0.0)
        prob = np.sqrt(isigi + 1.0)
        prob = prob / np.sum(prob)
        chosen.extend(rng.choice(b, size=quota, replace=False, p=prob).tolist())
    if remaining > 0:
        pool = np.asarray([i for i in candidates if i not in set(chosen)], dtype=int)
        if len(pool):
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            chosen.extend(extra.tolist())
    return np.asarray(sorted(set(int(i) for i in chosen)), dtype=int)


def fit_scale_contract(data, target, sig, pred_full, cfg, fit_indices, wght_a, wght_b, detector_scale_bins: int = 0):
    groups = group_indices(data, cfg.group_width_deg, cfg.frame_offset)
    det_bins = detector_bin_indices(data, int(detector_scale_bins))
    fit_mask = np.zeros(len(target), dtype=bool)
    fit_mask[np.asarray(fit_indices, dtype=int)] = True
    if int(detector_scale_bins) > 1:
        scaled, group_scales, detector_scales = fit_frame_detector_scales(
            target,
            sig,
            pred_full,
            groups,
            det_bins,
            fit_mask,
            wght_a,
            wght_b,
        )
    else:
        scaled, group_scales = fit_group_scales(target, sig, pred_full, groups, fit_mask, wght_a, wght_b)
        detector_scales = {}
    return {
        "mode": "fit",
        "group_width_deg": float(cfg.group_width_deg),
        "frame_offset": float(cfg.frame_offset),
        "detector_scale_bins": int(detector_scale_bins),
        "groups": groups,
        "detector_bins": det_bins,
        "group_scales": {str(k): float(v) for k, v in group_scales.items()},
        "detector_scales": {str(k): float(v) for k, v in detector_scales.items()},
        "scale_count": int(len(group_scales) + len(detector_scales)),
        "scaled": scaled,
    }


def apply_scale_contract(pred_full, contract):
    scaled = np.full_like(pred_full, np.nan, dtype=float)
    groups = np.asarray(contract["groups"], dtype=int)
    group_scales = {int(k): float(v) for k, v in contract.get("group_scales", {}).items()}
    detector_scales = {int(k): float(v) for k, v in contract.get("detector_scales", {}).items()}
    det_bins = np.asarray(contract.get("detector_bins", np.zeros_like(groups)), dtype=int)
    for i, value in enumerate(pred_full):
        if not np.isfinite(value):
            continue
        scale = group_scales.get(int(groups[i]))
        if scale is None or not np.isfinite(scale) or scale <= 0:
            continue
        if int(contract.get("detector_scale_bins", 0)) > 1:
            dscale = detector_scales.get(int(det_bins[i]))
            if dscale is None or not np.isfinite(dscale) or dscale <= 0:
                continue
            scale *= dscale
        scaled[i] = float(value) * float(scale)
    return scaled


def apply_row_resolution_to_prediction(data, pred_full, row_resolution_b: float = 0.0):
    """Apply the raw-row resolution envelope used by dynamic gradients."""

    valid = np.isfinite(pred_full) & (pred_full > 0.0)
    factor, meta = row_resolution_modulation(data, valid, row_resolution_b)
    return pred_full * factor, factor, meta


def scaled_prediction(data, target, sig, pred, cfg, fit_indices, wght_a, wght_b, detector_scale_bins: int = 0, scale_contract=None, row_resolution_b: float = 0.0):
    full = np.full(len(target), np.nan, dtype=float)
    full[pred["indices"]] = pred["dyn"]
    full, row_factor, row_meta = apply_row_resolution_to_prediction(data, full, row_resolution_b)
    if scale_contract is not None:
        groups = np.asarray(scale_contract["groups"], dtype=int)
        scaled = apply_scale_contract(full, scale_contract)
        scales = dict(scale_contract.get("group_scales", {}))
        scale_count = int(scale_contract.get("scale_count", len(scales)))
    else:
        contract = fit_scale_contract(data, target, sig, full, cfg, fit_indices, wght_a, wght_b, detector_scale_bins=detector_scale_bins)
        groups = np.asarray(contract["groups"], dtype=int)
        scaled = contract["scaled"]
        scales = dict(contract["group_scales"])
        scale_count = int(contract["scale_count"])
    return full, scaled, scales, groups, scale_count


def residual_vector(data, target, sig, pred_scaled, indices, wght_a, wght_b):
    idx = np.asarray(indices, dtype=int)
    obs = target[idx]
    calc = pred_scaled[idx]
    sig_i = sig[idx]
    good = np.isfinite(obs) & np.isfinite(calc) & np.isfinite(sig_i)
    obs = obs[good]
    calc = calc[good]
    sig_i = np.maximum(sig_i[good], 1e-12)
    w = shelx_weights(obs, sig_i, calc, wght_a, wght_b)
    return np.sqrt(w) * (obs - calc)


def validation_score(eval_result, metric: str) -> float:
    split, value = metric.split("_", 1)
    if split not in eval_result["splits"]:
        raise ValueError(f"unknown validation split in metric {metric!r}")
    key = "wR2_all" if value == "wR2" else "R1_all"
    return float(eval_result["splits"][split][key])
