"""Observation units, row selection and frame-group geometry."""

from __future__ import annotations

import csv
import math
import re
from pathlib import Path

import numpy as np

from grond.engine.bloch_engine import (
    XDSData,
    sha256,
)

from .contracts import Nuisance


def nominal_frame_width_deg(data: XDSData) -> float:
    """Return the best available physical angular width of one XDS frame."""

    osc = abs(float(getattr(data, "oscillation_range", float("nan"))))
    if np.isfinite(osc) and osc > 0.0:
        return osc
    z = np.asarray(getattr(data, "zcal", []), dtype=float)
    z = z[np.isfinite(z)]
    if z.size >= 2:
        unique = np.unique(np.round(z, 6))
        if unique.size >= 2:
            dz = float(np.median(np.diff(unique)))
            if np.isfinite(dz) and dz > 0.0:
                o0 = float(data.omega_from_z(unique[0]))
                o1 = float(data.omega_from_z(unique[0] + dz))
                width = abs(o1 - o0)
                if np.isfinite(width) and width > 0.0:
                    return width
    return 1.0


def automatic_frame_offset_grid(datum_offset_frames: float) -> list[float]:
    """Return a compact profile-center grid around the XDS/GROND scan datum.

    The XDS header contains one global orientation while ``ZCAL`` may reflect
    locally refined integration geometry.  Center the grid on the nearest
    0.2-frame datum and retain symmetric residual probes instead of assuming
    that zero in the global header is also the row-profile center.
    """

    if not np.isfinite(datum_offset_frames):
        return [0.0, 0.2]
    center = round(float(datum_offset_frames) / 0.2) * 0.2
    return [round(center + residual, 10) for residual in (-0.4, 0.0, 0.4)]


def parse_group_width_grid(text: str, data: XDSData) -> list[float]:
    """Parse physical frame-group widths.

    In addition to numeric degree values, GROND accepts `frame`, `2frame`,
    `3frames`, and `auto`.  `auto` expands to one-, two-, and three-frame
    groups plus coarser one/two/three/five-degree checks that are not smaller
    than one physical frame.  This keeps the scale model portable across XDS
    runs with different oscillation widths without creating sub-frame scale
    factors unless the user asks for them explicitly.
    """

    frame_width = nominal_frame_width_deg(data)
    raw = [x for x in str(text).replace(",", " ").split() if x.strip()]
    if not raw:
        raw = ["auto"]
    values: list[float] = []
    for token in raw:
        t = token.strip().lower()
        if t == "auto":
            candidates = [frame_width, 2.0 * frame_width, 3.0 * frame_width, 1.0, 2.0, 3.0, 5.0]
            values.extend(v for v in candidates if v + 1e-9 >= frame_width)
            continue
        if t in {"frame", "frames"}:
            values.append(frame_width)
            continue
        m = re.fullmatch(r"([0-9]*\.?[0-9]+)(?:x)?frames?", t)
        if m:
            values.append(float(m.group(1)) * frame_width)
            continue
        if t.endswith("deg"):
            t = t[:-3]
        values.append(float(t))
    out: list[float] = []
    for value in values:
        value = float(value)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError(f"group widths must be positive finite values; got {value!r}")
        if not any(abs(value - seen) <= max(1e-9, 1e-7 * abs(seen)) for seen in out):
            out.append(value)
    return out


def target_factor(data: XDSData, mode: str):
    """Return the positive rowwise factor mapping XDS IOBS into a fit target."""

    if mode == "iobs":
        factor = np.ones_like(data.iobs, dtype=float)
    elif mode == "iobs_over_rlp":
        factor = 1.0 / np.maximum(data.rlp, 1e-12)
    elif mode == "iobs_times_peak":
        factor = np.maximum(data.peak, 0.0) / 100.0
    elif mode == "recorded_proxy":
        factor = np.maximum(data.peak, 0.0) / 100.0 / np.maximum(data.rlp, 1e-12)
    else:
        raise ValueError(f"unknown target mode {mode!r}")
    return factor


def target_arrays(data: XDSData, mode: str):
    """Return the reduced XDS row target and matching sigma convention."""

    factor = target_factor(data, mode)
    return data.iobs * factor, data.sigma * np.abs(factor)


def endpoint_observation_arrays(data: XDSData, target, sigma, mode: str):
    """Restore a diagnostic fit target to ordinary XDS IOBS endpoint units.

    Alternate target modes are useful for fitting nuisance terms against a
    reconstructed scalar observation.  SHELXL HKLF4 files, however, must stay
    in the ordinary XDS-complete intensity convention.  Undo the declared
    positive target factor after fitting while retaining any sigma inflation
    already present in ``sigma``.
    """

    factor = np.asarray(target_factor(data, mode), dtype=float)
    safe = np.isfinite(factor) & (factor > 0.0)
    endpoint_target = np.full_like(np.asarray(target, dtype=float), np.nan)
    endpoint_sigma = np.full_like(np.asarray(sigma, dtype=float), np.nan)
    endpoint_target[safe] = np.asarray(target, dtype=float)[safe] / factor[safe]
    endpoint_sigma[safe] = np.asarray(sigma, dtype=float)[safe] / factor[safe]
    return endpoint_target, endpoint_sigma, factor


def sigma_eff(data: XDSData, target, base_sigma, nui: Nuisance):
    """Return the effective sigma of every row: the base sigma plus the error-model terms in quadrature.

    The sigma_a, sigma_b_corr and sigma_c_peak terms are proportional to the target; the second and third also scale
    with (1 - CORR/100) and (1 - PEAK/100).
    """

    corr_term = np.maximum(0.0, 1.0 - data.corr / 100.0)
    peak_term = np.maximum(0.0, 1.0 - np.maximum(data.peak, 0.0) / 100.0)
    return np.sqrt(
        np.maximum(base_sigma, 1e-12) ** 2
        + (nui.sigma_a * target) ** 2
        + (nui.sigma_b_corr * target * corr_term) ** 2
        + (nui.sigma_c_peak * target * peak_term) ** 2
    )


def group_indices(data: XDSData, group_width_deg: float, frame_offset: float):
    """Return the frame group of every row: omega at ZCAL + frame_offset in group_width_deg bins.

    Bins start at the smallest omega.
    """

    omegas = np.asarray([data.omega_from_z(float(z) + float(frame_offset)) for z in data.zcal], dtype=float)
    return ((omegas - float(np.min(omegas))) // float(group_width_deg)).astype(int)


def geometry_tail_mask(data: XDSData, base_mask, percentile: float = 100.0):
    """Keep the broad geometry body and exclude only the worst geometry tail."""
    if percentile >= 100.0:
        return np.ones(len(data.iobs), dtype=bool), {}
    base = np.asarray(base_mask, dtype=bool)
    dx = data.xobs - data.xcal
    dy = data.yobs - data.ycal
    dz = data.zobs - data.zcal
    dr = np.sqrt(dx * dx + dy * dy)
    finite_dr = np.isfinite(dr)
    finite_dz = np.isfinite(dz)
    geom_base = base & finite_dr & finite_dz
    if not np.any(geom_base):
        return np.ones(len(data.iobs), dtype=bool), {"status": "no_finite_xds_observed_geometry"}
    dr_cut = float(np.nanpercentile(dr[geom_base], percentile))
    dz_cut = float(np.nanpercentile(np.abs(dz[geom_base]), percentile))
    finite_rlp = np.isfinite(data.rlp) & (data.rlp > 0)
    log_rlp = np.full(len(data.iobs), np.nan, dtype=float)
    log_rlp[finite_rlp] = np.log(data.rlp[finite_rlp])
    rlp_base = geom_base & np.isfinite(log_rlp)
    if np.any(rlp_base):
        med = float(np.nanmedian(log_rlp[rlp_base]))
        mad = float(np.nanmedian(np.abs(log_rlp[rlp_base] - med)))
        sigma = 1.4826 * mad if mad > 0 else float(np.nanstd(log_rlp[rlp_base]))
        if not np.isfinite(sigma) or sigma <= 0:
            sigma = 1.0
        rlp_z = np.abs((log_rlp - med) / sigma)
        rlp_cut = float(np.nanpercentile(rlp_z[rlp_base], percentile))
        rlp_keep = ~np.isfinite(rlp_z) | (rlp_z <= rlp_cut)
    else:
        rlp_cut = float("nan")
        rlp_keep = np.ones(len(data.iobs), dtype=bool)
    keep = (~finite_dr | (dr <= dr_cut)) & (~finite_dz | (np.abs(dz) <= dz_cut)) & rlp_keep
    return keep, {
        "status": "ok",
        "percentile": float(percentile),
        "observed_geometry_rows": int(np.count_nonzero(geom_base)),
        "missing_observed_geometry_rows": int(
            np.count_nonzero(base & ~finite_dr & ~finite_dz)
        ),
        "dr_cut_px": dr_cut,
        "abs_dz_cut_frame": dz_cut,
        "rlp_log_robust_z_cut": rlp_cut,
        "base_rows": int(np.count_nonzero(base)),
        "kept_base_rows": int(np.count_nonzero(base & keep)),
        "rejected_base_rows": int(np.count_nonzero(base & ~keep)),
    }


def row_selection(data: XDSData, target, positive_only: bool, min_d: float, min_peak: float, max_rows: int = 0, geometry_percentile: float = 100.0):
    """Return the indices of the eligible rows.

    Eligible rows have a finite target, positive sigma and RLP, pass the optional positive, PEAK and d cuts, and are
    not in the geometry tail. With max_rows > 0 at most max_rows rows are kept, taken in turn from frame x
    resolution strata, highest target/sigma first.
    """

    mask = (
        np.isfinite(target)
        & np.isfinite(data.sigma)
        & (np.asarray(data.sigma, dtype=float) > 0.0)
        & np.isfinite(data.zcal)
        & np.isfinite(data.rlp)
        & (np.asarray(data.rlp, dtype=float) > 0.0)
    )
    if positive_only:
        mask &= target > 0.0
    if min_peak > 0:
        mask &= data.peak >= min_peak
    if min_d > 0:
        d_spacings = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
        mask &= d_spacings >= min_d
    geom_keep, _geom_summary = geometry_tail_mask(data, mask, geometry_percentile)
    mask &= geom_keep
    indices = np.where(mask)[0]
    if max_rows > 0 and len(indices) > max_rows:
        n_bins = max(1, min(10, int(math.sqrt(int(max_rows)))))
        d_spacings = np.asarray(
            [data.cell.calculate_d(tuple(int(v) for v in data.hkl[row_index])) for row_index in indices],
            dtype=float,
        )
        frames = np.asarray(data.zcal[indices], dtype=float)
        signal_to_noise = np.asarray(target[indices], dtype=float) / np.asarray(data.sigma[indices], dtype=float)

        def rank_bins(values):
            """Return equal-count rank bins 0..n_bins-1 of values (stable order for ties)."""

            order = np.argsort(values, kind="mergesort")
            bins = np.empty(len(values), dtype=int)
            bins[order] = np.minimum(n_bins - 1, np.arange(len(values)) * n_bins // len(values))
            return bins

        frame_bins = rank_bins(frames)
        resolution_bins = rank_bins(1.0 / np.maximum(d_spacings, 1e-12))
        strata = {}
        for position, row_index in enumerate(indices):
            strata.setdefault((int(frame_bins[position]), int(resolution_bins[position])), []).append(
                (float(signal_to_noise[position]), int(row_index))
            )
        queues = [
            [row_index for _score, row_index in sorted(rows, key=lambda item: (item[0], -item[1]))]
            for _key, rows in sorted(strata.items())
        ]
        selected = []
        while len(selected) < int(max_rows) and any(queues):
            for queue in queues:
                if queue and len(selected) < int(max_rows):
                    selected.append(queue.pop())
        indices = np.asarray(selected, dtype=int)
    return indices


def row_contract_selection(data: XDSData, target, path: Path) -> tuple[np.ndarray, dict]:
    """Load one immutable row-membership contract and verify it against XDS.

    Continuation jobs normally rebuild their eligible observations from the raw
    XDS file. That is unsafe after a multi-dataset policy/scale promotion: the
    promoted result may intentionally retain a strict subset of otherwise
    eligible rows. A contract therefore selects *only* its recorded row
    indices and fails closed if it no longer describes the supplied XDS input.

    The source is a GROND ``native_dyn_calc.csv`` (policy annotations are
    allowed). ``row_index,h,k,l,target_I,xds_sigma,d_A`` bind both row identity
    and the raw observation rather than merely trusting a list of integers.
    """

    contract = Path(path).expanduser().resolve()
    if not contract.is_file():
        raise ValueError(f"row contract is missing: {contract}")
    required = ("row_index", "h", "k", "l", "target_I", "xds_sigma", "d_A")
    with contract.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = tuple(reader.fieldnames or ())
        missing = [field for field in required if field not in fields]
        if missing:
            raise ValueError(
                f"row contract {contract} is missing required column(s): {', '.join(missing)}"
            )
        rows = [dict(row) for row in reader]
    if not rows:
        raise ValueError(f"row contract {contract} contains no rows")

    seen: set[int] = set()
    errors: list[str] = []
    checked_target = checked_sigma = checked_d = 0
    for ordinal, row in enumerate(rows, start=2):
        try:
            row_index = int(str(row["row_index"]).strip())
            hkl = tuple(int(str(row[field]).strip()) for field in ("h", "k", "l"))
            target_i = float(str(row["target_I"]).strip())
            xds_sigma = float(str(row["xds_sigma"]).strip())
            d_expected = float(str(row["d_A"]).strip())
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"line {ordinal}: invalid row identity/observation: {exc}")
            if len(errors) >= 10:
                break
            continue
        if row_index < 0 or row_index >= len(data.hkl):
            errors.append(f"line {ordinal}: row_index {row_index} is outside XDS range [0, {len(data.hkl)})")
        elif row_index in seen:
            errors.append(f"line {ordinal}: duplicate row_index {row_index}")
        else:
            seen.add(row_index)
            observed_hkl = tuple(int(value) for value in np.asarray(data.hkl[row_index], dtype=int))
            if observed_hkl != hkl:
                errors.append(
                    f"line {ordinal}: row_index {row_index} HKL mismatch; contract={hkl}, XDS={observed_hkl}"
                )
            else:
                if not (
                    math.isfinite(target_i)
                    and math.isclose(float(target[row_index]), target_i, rel_tol=1e-11, abs_tol=1e-10)
                ):
                    errors.append(
                        f"line {ordinal}: row_index {row_index} target_I mismatch; "
                        f"contract={target_i:.17g}, XDS={float(target[row_index]):.17g}"
                    )
                else:
                    checked_target += 1
                if not (
                    math.isfinite(xds_sigma)
                    and math.isclose(float(data.sigma[row_index]), xds_sigma, rel_tol=1e-11, abs_tol=1e-10)
                ):
                    errors.append(
                        f"line {ordinal}: row_index {row_index} xds_sigma mismatch; "
                        f"contract={xds_sigma:.17g}, XDS={float(data.sigma[row_index]):.17g}"
                    )
                else:
                    checked_sigma += 1
                d_actual = float(data.cell.calculate_d(observed_hkl))
                if not (
                    math.isfinite(d_expected)
                    # d_A is derived from the already-bound integer HKL and
                    # unit cell, not an independent observation. Different
                    # libm/linear-algebra builds can differ at a few 1e-10 A;
                    # keep this strict enough to catch a changed cell while
                    # allowing harmless cross-platform roundoff.
                    and math.isclose(d_actual, d_expected, rel_tol=1e-8, abs_tol=1e-9)
                ):
                    errors.append(
                        f"line {ordinal}: row_index {row_index} d_A mismatch; "
                        f"contract={d_expected:.17g}, XDS={d_actual:.17g}"
                    )
                else:
                    checked_d += 1
        if len(errors) >= 10:
            break
    if errors:
        raise ValueError(
            "row contract does not bind exactly to the supplied XDS observations; "
            "refusing to continue with changed row membership:\n- " + "\n- ".join(errors)
        )

    indices = np.asarray(sorted(seen), dtype=int)
    if len(indices) != len(rows):
        raise ValueError(
            f"row contract {contract} selected {len(indices)} unique indices from {len(rows)} rows"
        )
    return indices, {
        "status": "verified_exact_row_contract",
        "path": str(contract),
        "sha256": sha256(contract),
        "requested_rows": int(len(rows)),
        "selected_rows": int(len(indices)),
        "row_index_min": int(indices[0]),
        "row_index_max": int(indices[-1]),
        "identity_columns": list(required),
        "target_rows_verified": int(checked_target),
        "sigma_rows_verified": int(checked_sigma),
        "d_spacing_rows_verified": int(checked_d),
        "selection_rule": "contract row_index membership; no raw-XDS eligibility rows were added or removed",
    }


def endpoint_resolution_modulation(data: XDSData, valid, endpoint_b: float):
    """Return exp(endpoint_b (s^2 - s^2_center)) per row, s = 1/(2d), and its metadata; all 1 when endpoint_b is 0.

    s^2_center is the median over the valid rows.
    """

    if abs(float(endpoint_b)) <= 0.0:
        return np.ones(len(data.hkl), dtype=float), {"endpoint_b": float(endpoint_b), "s2_center": float("nan")}
    d_spacings = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    s2 = 1.0 / np.maximum(4.0 * d_spacings * d_spacings, 1e-30)
    center_mask = np.asarray(valid, dtype=bool) & np.isfinite(s2)
    s2_center = float(np.nanmedian(s2[center_mask])) if np.any(center_mask) else float(np.nanmedian(s2))
    mod = np.exp(float(endpoint_b) * (s2 - s2_center))
    return mod, {"endpoint_b": float(endpoint_b), "s2_center": s2_center}


def row_resolution_modulation(data: XDSData, valid, row_resolution_b: float):
    """Resolution-envelope factor applied to raw dynamical row predictions.

    Positive values increase high-resolution dynamical intensities relative to
    low-resolution rows. In corrected-HKL diagnostics this is approximately the
    alternate sign of endpoint_b because the endpoint divides observations by
    the dynamical correction.
    """

    row_resolution_b = float(row_resolution_b)
    if abs(row_resolution_b) <= 0.0:
        return np.ones(len(data.hkl), dtype=float), {"row_resolution_b": row_resolution_b, "row_s2_center": float("nan")}
    d_spacings = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    s2 = 1.0 / np.maximum(4.0 * d_spacings * d_spacings, 1e-30)
    center_mask = np.asarray(valid, dtype=bool) & np.isfinite(s2)
    s2_center = float(np.nanmedian(s2[center_mask])) if np.any(center_mask) else float(np.nanmedian(s2))
    exponent = np.clip(row_resolution_b * (s2 - s2_center), -20.0, 20.0)
    mod = np.exp(exponent)
    return mod, {"row_resolution_b": row_resolution_b, "row_s2_center": s2_center}
