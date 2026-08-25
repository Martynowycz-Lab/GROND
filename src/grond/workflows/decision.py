from __future__ import annotations

import itertools
import json
import math
from collections.abc import Sequence
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from grond.hkl_transform import parse_hkl_transform_info
from grond.io.config import DatasetSpec


def _all_signed_axis_permutations(determinant: int) -> list[str]:
    """Return the complete, fixed-order signed-axis permutation universe."""

    transforms = []
    for axes in itertools.permutations(("h", "k", "l")):
        for signs in itertools.product((1, -1), repeat=3):
            spec = ",".join(
                ("" if sign > 0 else "-") + axis
                for axis, sign in zip(axes, signs, strict=True)
            )
            info = parse_hkl_transform_info(spec)
            if info.determinant == int(determinant):
                transforms.append(info.canonical)
    return sorted(set(transforms), key=lambda value: (value != "identity", value))


ALL_PROPER_SIGNED_PERMUTATIONS = _all_signed_axis_permutations(1)
ALL_IMPROPER_SIGNED_PERMUTATIONS = _all_signed_axis_permutations(-1)
# Backward-compatible names for callers that used the historical four-item
# "common" list.  Their value is now deliberately the complete 24-operator
# signed-axis universe for each determinant.
COMMON_PROPER_TRANSFORMS = list(ALL_PROPER_SIGNED_PERMUTATIONS)
COMMON_IMPROPER_TRANSFORMS = list(ALL_IMPROPER_SIGNED_PERMUTATIONS)
# Backward-compatible public name, now intentionally safe for automatic use.
COMMON_TRANSFORMS = COMMON_PROPER_TRANSFORMS
ALL_COMMON_TRANSFORMS = [*COMMON_PROPER_TRANSFORMS, *COMMON_IMPROPER_TRANSFORMS]


@dataclass
class PreflightDecision:
    """The automatic data-quality decision GROND makes before expensive work."""

    dataset: str
    hkl: str
    status: str
    row_count: int
    positive_fraction: float
    sigma_positive_fraction: float
    peak_p50: float
    peak_p90: float
    corr_p50: float
    d_min_observed_A: float
    observed_geometry_fraction: float
    geometry_dr_p99_px: float
    geometry_abs_dz_p99_frame: float
    recommended_min_peak: float
    recommended_geometry_percentile: float
    scout_max_rows: int
    warnings: list[str]


def _safe_percentile(values, p: float) -> float:
    """Return a finite percentile or NaN when no finite values exist."""

    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan")
    return float(np.nanpercentile(arr, p))


def _xds_observed_geometry_mask(
    xobs: np.ndarray,
    yobs: np.ndarray,
    zobs: np.ndarray,
) -> np.ndarray:
    """Identify real XDS centroids, excluding its all-zero missing sentinel."""

    finite = np.isfinite(xobs) & np.isfinite(yobs) & np.isfinite(zobs)
    missing_sentinel = (xobs == 0.0) & (yobs == 0.0) & (zobs == 0.0)
    return finite & ~missing_sentinel


def _parse_xds_rows(path: Path) -> tuple[dict[str, str], np.ndarray]:
    """Parse enough of an XDS HKL file for data-quality decisions."""

    if not path.exists():
        raise FileNotFoundError(path)
    header: dict[str, str] = {}
    rows: list[list[float]] = []
    with path.open(errors="replace") as f:
        for line in f:
            if line.startswith("!"):
                if "=" in line:
                    key, value = line[1:].split("=", 1)
                    header[key.strip()] = value.strip()
                continue
            parts = line.split()
            if len(parts) < 11:
                continue
            try:
                vals = [float(x) for x in parts[:21]]
            except ValueError:
                continue
            if len(vals) == 12:
                h, k, ell, iobs, sig, xd, yd, zd, rlp, peak, corr, psi = vals
                vals = [
                    h, k, ell, iobs, sig, xd, yd, zd, rlp, peak, corr,
                    float("nan"), float("nan"), float("nan"), float("nan"),
                    float("nan"), float("nan"), float("nan"), float("nan"),
                    psi, 0.0,
                ]
            if len(vals) >= 21:
                rows.append(vals[:21])
    if not rows:
        return header, np.zeros((0, 21), dtype=float)
    return header, np.asarray(rows, dtype=float)


def _d_spacings(header: dict[str, str], hkl: np.ndarray) -> np.ndarray:
    """Calculate d spacings from the XDS unit cell when it is available."""

    try:
        import gemmi

        vals = [float(x) for x in header.get("UNIT_CELL_CONSTANTS", "").split()[:6]]
        if len(vals) != 6:
            return np.full(len(hkl), np.nan)
        cell = gemmi.UnitCell(*vals)
        return np.asarray([cell.calculate_d(tuple(int(v) for v in row)) for row in hkl], dtype=float)
    except Exception:
        return np.full(len(hkl), np.nan)


def preflight_dataset(ds: DatasetSpec, *, requested_min_peak: float, requested_geometry_percentile: float, requested_scout_rows: int) -> PreflightDecision:
    """Inspect one dataset and recommend safe filters for automatic scouting."""

    header, rows = _parse_xds_rows(Path(ds.hkl))
    warnings: list[str] = []
    n = int(len(rows))
    if n == 0:
        return PreflightDecision(
            dataset=ds.name,
            hkl=ds.hkl,
            status="fatal",
            row_count=0,
            positive_fraction=0.0,
            sigma_positive_fraction=0.0,
            peak_p50=float("nan"),
            peak_p90=float("nan"),
            corr_p50=float("nan"),
            d_min_observed_A=float("nan"),
            observed_geometry_fraction=0.0,
            geometry_dr_p99_px=float("nan"),
            geometry_abs_dz_p99_frame=float("nan"),
            recommended_min_peak=0.0,
            recommended_geometry_percentile=100.0,
            scout_max_rows=0,
            warnings=["no numeric reflection rows found"],
        )
    iobs = rows[:, 3]
    sig = rows[:, 4]
    peak = rows[:, 9]
    corr = rows[:, 10]
    xcal, ycal, zcal = rows[:, 5], rows[:, 6], rows[:, 7]
    xobs, yobs, zobs = rows[:, 12], rows[:, 13], rows[:, 14]
    hkl = rows[:, :3].astype(int)
    dvals = _d_spacings(header, hkl)
    positive_fraction = float(np.mean(iobs > 0.0))
    sigma_positive_fraction = float(np.mean(sig > 0.0))
    geom = _xds_observed_geometry_mask(xobs, yobs, zobs)
    observed_geometry_fraction = float(np.mean(geom))
    dr = np.sqrt((xobs - xcal) ** 2 + (yobs - ycal) ** 2)
    dz = zobs - zcal

    usable = np.isfinite(iobs) & np.isfinite(sig) & (sig > 0)
    minimum_usable = max(100, math.ceil(0.05 * n))
    usable_count = int(np.count_nonzero(usable))
    if usable_count < minimum_usable:
        warnings.append("few rows have finite intensity and positive sigma")
    if positive_fraction < 0.5:
        warnings.append("less than half of rows have positive target intensity")
    if sigma_positive_fraction < 0.98:
        warnings.append("some rows have non-positive sigma")
    profile_fitting = str(header.get("PROFILE_FITTING", "")).strip().upper()
    if profile_fitting and profile_fitting not in {"FALSE", "F", "0", "NO"}:
        warnings.append(
            f"XDS rows declare PROFILE_FITTING={profile_fitting}; production GROND normally uses "
            "a separate PROFILE_FITTING=FALSE summation integration"
        )
    missing_geometry_count = int(np.count_nonzero(~geom))
    if missing_geometry_count:
        warnings.append(
            f"{missing_geometry_count} rows have absent XDS observed centroids; "
            "excluded from geometry-tail diagnostics"
        )
    if observed_geometry_fraction < 0.5:
        warnings.append("observed XDS geometry columns are mostly absent; geometry-tail filter disabled")

    requested_min_peak = float(requested_min_peak)
    base_d = np.ones(n, dtype=bool)
    if np.isfinite(dvals).any() and ds.min_d_spacing > 0:
        base_d &= dvals >= float(ds.min_d_spacing)
    positive = iobs > 0
    min_needed = max(250, min(2000, int(0.25 * np.count_nonzero(base_d & positive & usable))))
    recommended_min_peak = requested_min_peak
    for candidate in [requested_min_peak, 90.0, 75.0, 60.0, 50.0, 25.0, 0.0]:
        pass_rows = int(np.count_nonzero(base_d & positive & usable & (peak >= candidate)))
        if pass_rows >= min_needed:
            recommended_min_peak = float(candidate)
            break
    if recommended_min_peak < requested_min_peak:
        warnings.append(
            f"a PEAK floor of {recommended_min_peak:g} would retain more rows than the requested "
            f"{requested_min_peak:g}; suggestion only, automatic execution preserves the requested floor"
        )

    recommended_geometry_percentile = float(requested_geometry_percentile)
    if observed_geometry_fraction < 0.5:
        recommended_geometry_percentile = 100.0
    elif np.isfinite(dr[geom]).any():
        p50 = _safe_percentile(dr[geom], 50)
        p99 = _safe_percentile(dr[geom], 99)
        if np.isfinite(p50) and p50 > 0 and np.isfinite(p99) and p99 > 25 * p50:
            recommended_geometry_percentile = min(recommended_geometry_percentile, 97.5)
            warnings.append(
                "large detector-position tail detected; a stricter 97.5 geometry percentile is suggested, "
                "but automatic execution preserves the requested percentile"
            )

    row_budget = requested_scout_rows if requested_scout_rows > 0 else 1000
    surviving_rows = int(np.count_nonzero(base_d & positive & usable & (peak >= requested_min_peak)))
    scout_max_rows = min(int(row_budget), surviving_rows)
    status = "ok"
    if usable_count < minimum_usable:
        status = "fatal"
    elif scout_max_rows < 250:
        status = "bad"
        warnings.append("too few rows survive basic filters for reliable automatic decisions")

    return PreflightDecision(
        dataset=ds.name,
        hkl=ds.hkl,
        status=status,
        row_count=n,
        positive_fraction=positive_fraction,
        sigma_positive_fraction=sigma_positive_fraction,
        peak_p50=_safe_percentile(peak, 50),
        peak_p90=_safe_percentile(peak, 90),
        corr_p50=_safe_percentile(corr, 50),
        d_min_observed_A=float(np.nanmin(dvals)) if np.isfinite(dvals).any() else float("nan"),
        observed_geometry_fraction=observed_geometry_fraction,
        geometry_dr_p99_px=_safe_percentile(dr[geom], 99) if np.any(geom) else float("nan"),
        geometry_abs_dz_p99_frame=_safe_percentile(np.abs(dz[geom]), 99) if np.any(geom) else float("nan"),
        recommended_min_peak=recommended_min_peak,
        recommended_geometry_percentile=recommended_geometry_percentile,
        scout_max_rows=scout_max_rows,
        warnings=warnings,
    )


def apply_preflight(ds: DatasetSpec, decision: PreflightDecision) -> DatasetSpec:
    """Return a dataset copy without silently changing its declared row support.

    Preflight recommendations remain visible diagnostics.  A user who wants a
    different PEAK/partiality floor or geometry percentile must request that
    contract explicitly so prep files, command ledgers, and final reports all
    describe the rows that were actually used.
    """

    return replace(
        ds,
        min_peak=ds.min_peak,
        geometry_percentile=ds.geometry_percentile,
    )


def expand_transform_candidates(
    current: str,
    spec: str,
    *,
    parity_policy: str = "proper-only",
) -> list[str]:
    """Build the HKL transform list used by the transform scout.

    Automatic/common scouting is proper-only by default.  The historical
    mixed-parity search remains available as the explicit
    ``all-diagnostic`` policy, but downstream reports then remain conditional
    and cannot silently claim named absolute configuration.
    """

    spec = (spec or "auto").strip()
    if parity_policy not in {"proper-only", "all-diagnostic"}:
        raise ValueError(
            "transform parity policy must be 'proper-only' or 'all-diagnostic'"
        )
    if spec.lower() in {"off", "none"}:
        info = parse_hkl_transform_info(current)
        if parity_policy == "proper-only" and not info.proper:
            raise ValueError(
                f"improper fixed transform {info.canonical!r} (determinant -1) is not allowed "
                "by --transform-parity-policy=proper-only; canonicalize the model/HKLF4 pair "
                f"to {info.proper_canonical_transform!r}, or use all-diagnostic mode without "
                "an absolute-configuration claim"
            )
        return [info.canonical]
    out: list[str] = []
    raw: list[tuple[str, bool]] = []
    common = ALL_COMMON_TRANSFORMS if parity_policy == "all-diagnostic" else COMMON_PROPER_TRANSFORMS
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if item == "auto":
            raw.extend([(current, True), *((candidate, True) for candidate in common)])
        elif item == "common":
            raw.extend((candidate, True) for candidate in common)
        else:
            raw.append((item, False))
    if not raw:
        raw = [(current, True), *((candidate, True) for candidate in common)]
    for item, automatic in raw:
        item = "identity" if item in {"", "auto"} else item
        info = parse_hkl_transform_info(item)
        if parity_policy == "proper-only" and not info.proper:
            if not automatic:
                raise ValueError(
                    f"improper transform {info.canonical!r} (determinant -1) is not allowed by "
                    "--transform-parity-policy=proper-only; use a proper sign-paired operator "
                    f"{info.proper_canonical_transform!r}, or opt into all-diagnostic mode without "
                    "an absolute-configuration claim"
                )
            continue
        if info.canonical not in out:
            out.append(info.canonical)
    if not out:
        raise ValueError("no HKL transform candidates remain after applying the parity policy")
    return out


def safe_label(text: str) -> str:
    """Turn arbitrary option text into a filesystem-safe label."""

    return (
        text.replace(",", "_")
        .replace("-", "m")
        .replace("+", "p")
        .replace("/", "_")
        .replace(" ", "")
        .replace("identity", "identity")
    )


def safe_float(value, fallback: float = float("inf")) -> float:
    """Convert a value to float without throwing inside score logic."""

    try:
        out = float(value)
    except (TypeError, ValueError):
        return fallback
    return out if math.isfinite(out) else fallback


def score_native_summary(summary: dict) -> dict:
    """Extract the ranking metrics used to compare native Bloch stages."""

    best = summary.get("best_native") or {}
    metrics = best.get("metrics") or {}
    all_metrics = metrics.get("all") or {}
    free_metrics = metrics.get("free") or {}
    dyn = all_metrics.get("dyn") or {}
    born = all_metrics.get("born") or {}
    free_dyn = free_metrics.get("dyn") or {}
    dyn_w = safe_float(dyn.get("wR2_all"), float("inf"))
    free_w = safe_float(free_dyn.get("wR2_all"), dyn_w)
    dyn_r1 = safe_float(dyn.get("R1_all"), dyn_w)
    born_r1gt = safe_float(born.get("R1_gt"), 1.0)
    born_cc = safe_float(born.get("CC"), 0.0)
    dyn_cc = safe_float(dyn.get("CC"), 0.0)
    score = dyn_w + 0.25 * free_w + 0.10 * dyn_r1 + max(0.0, born_r1gt - 0.35) + max(0.0, 0.5 - born_cc)
    return {
        "score": score,
        "dyn_wR2_all": dyn_w,
        "free_dyn_wR2_all": free_w,
        "dyn_R1_all": dyn_r1,
        "dyn_R1_gt": safe_float(dyn.get("R1_gt"), float("nan")),
        "born_R1_gt": born_r1gt,
        "born_CC": born_cc,
        "dyn_CC": dyn_cc,
    }


def score_transform_summary(summary: dict) -> dict:
    """Score a transform without consuming coordinate-hand-sensitive Bloch terms.

    Transform selection and hand selection are distinct inference problems.
    The former therefore uses only independently scaled *work-row*
    matched-Born intensities, which are invariant to fractional-coordinate
    inversion for GROND's real electron-scattering potential.  Free rows are
    reserved for the later matched hand test.  Dynamical and all-row blocks
    remain in the scout summary for diagnostics but are deliberately absent
    from this ranking target.
    """

    best = summary.get("best_native") or {}
    metrics = best.get("metrics") or {}
    work_metrics = metrics.get("work") or {}
    all_metrics = metrics.get("all") or {}
    born = work_metrics.get("born") or {}
    all_born = all_metrics.get("born") or {}
    born_w = safe_float(born.get("wR2_all"), float("inf"))
    born_r1 = safe_float(born.get("R1_all"), float("inf"))
    born_r1gt = safe_float(born.get("R1_gt"), float("inf"))
    born_cc = safe_float(born.get("CC"), float("-inf"))
    score = (
        born_w
        + 0.10 * born_r1
        + max(0.0, born_r1gt - 0.35)
        + max(0.0, 0.5 - born_cc)
    )
    return {
        "score": score,
        "selection_target": "matched_born_work_only_hand_invariant_v1",
        "work_born_wR2_all": born_w,
        "work_born_R1_all": born_r1,
        "work_born_R1_gt": born_r1gt,
        "work_born_CC": born_cc,
        "diagnostic_all_born_wR2_all": safe_float(
            all_born.get("wR2_all"), float("nan")
        ),
        "diagnostic_all_born_R1_all": safe_float(
            all_born.get("R1_all"), float("nan")
        ),
        "coordinate_hand_sensitive_terms_used": False,
        "free_rows_used_for_transform_selection": False,
    }


def thickness_polish_nuisance(nui: dict, frame_offset_candidates: Sequence[float] | None = None) -> dict:
    """Narrow a nuisance grid around the best thickness and geometry values."""

    out = dict(nui)
    t = safe_float(nui.get("thickness_A"), 1200.0)
    delta = max(100.0, min(400.0, 0.18 * t))
    vals = sorted({max(100.0, t - delta), t, t + delta})
    out["thickness_A"] = ",".join(f"{v:.15g}" for v in vals)
    g = safe_float(nui.get("thickness_grad_frac"), 0.0)
    # A uniform-gradient distribution spans t0 * (1 +/- g). Keep g <= 1 so
    # the lower edge cannot cross into nonphysical negative thickness.
    g = min(1.0, max(0.0, g))
    gvals = sorted({max(0.0, g - 0.2), g, min(1.0, g + 0.2)})
    out["thickness_grad_frac"] = ",".join(f"{v:.15g}" for v in gvals)
    f = safe_float(nui.get("frame_offset"), 0.0)
    fvals = {f - 0.1, f, f + 0.1}
    for candidate in frame_offset_candidates or ():
        value = safe_float(candidate, float("nan"))
        if math.isfinite(value):
            fvals.add(value)
    out["frame_offset"] = ",".join(f"{v:.15g}" for v in sorted(fvals))
    return out


def should_stop_for_bad_data(summary: dict, *, max_wR2: float, min_improvement: float) -> tuple[bool, list[str]]:
    """Decide whether a branch is too poor to spend more compute on."""

    score = score_native_summary(summary)
    reasons: list[str] = []
    if not math.isfinite(score["dyn_wR2_all"]):
        reasons.append("dynamic all-row wR2 is not finite")
    if score["dyn_wR2_all"] > max_wR2:
        reasons.append(f"dynamic all-row wR2 {score['dyn_wR2_all']:.3f} exceeds stop threshold {max_wR2:.3f}")
    born_w = (((summary.get("best_native") or {}).get("metrics") or {}).get("all") or {}).get("born", {}).get("wR2_all")
    try:
        improvement = float(born_w) - score["dyn_wR2_all"]
    except Exception:
        improvement = float("nan")
    if math.isfinite(improvement) and improvement < min_improvement:
        reasons.append(f"Bloch improvement over Born wR2 is only {improvement:.4f}")
    return bool(reasons), reasons


def write_json(path: Path, payload) -> None:
    """Write JSON with parent-directory creation and NaN-tolerant encoding."""

    path.parent.mkdir(parents=True, exist_ok=True)
    if hasattr(payload, "__dataclass_fields__"):
        payload = asdict(payload)
    path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
