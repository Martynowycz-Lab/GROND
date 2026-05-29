from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np

from grond.io.config import DatasetSpec

COMMON_TRANSFORMS = [
    "identity",
    "l,-k,h",
    "-l,k,-h",
    "h,-k,-l",
    "-h,k,l",
    "k,h,-l",
    "-k,-h,l",
]


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
    sig = np.abs(rows[:, 4])
    peak = rows[:, 9]
    corr = rows[:, 10]
    xcal, ycal, zcal = rows[:, 5], rows[:, 6], rows[:, 7]
    xobs, yobs, zobs = rows[:, 12], rows[:, 13], rows[:, 14]
    hkl = rows[:, :3].astype(int)
    dvals = _d_spacings(header, hkl)
    positive_fraction = float(np.mean(iobs > 0.0))
    sigma_positive_fraction = float(np.mean(sig > 0.0))
    geom = np.isfinite(xobs) & np.isfinite(yobs) & np.isfinite(zobs)
    observed_geometry_fraction = float(np.mean(geom))
    dr = np.sqrt((xobs - xcal) ** 2 + (yobs - ycal) ** 2)
    dz = zobs - zcal

    usable = np.isfinite(iobs) & np.isfinite(sig) & (sig > 0)
    if np.count_nonzero(usable) < max(100, 0.05 * n):
        warnings.append("few rows have finite intensity and positive sigma")
    if positive_fraction < 0.5:
        warnings.append("less than half of rows have positive target intensity")
    if sigma_positive_fraction < 0.98:
        warnings.append("some rows have non-positive sigma")
    if observed_geometry_fraction < 0.5:
        warnings.append("observed XDS geometry columns are mostly absent; geometry-tail filter disabled")

    requested_min_peak = float(requested_min_peak)
    base_d = np.ones(n, dtype=bool)
    if np.isfinite(dvals).any() and ds.min_d_spacing > 0:
        base_d &= dvals >= float(ds.min_d_spacing)
    positive = iobs > 0
    min_needed = max(250, min(2000, int(0.25 * np.count_nonzero(base_d & positive))))
    recommended_min_peak = requested_min_peak
    for candidate in [requested_min_peak, 90.0, 75.0, 60.0, 50.0, 25.0, 0.0]:
        pass_rows = int(np.count_nonzero(base_d & positive & (peak >= candidate)))
        if pass_rows >= min_needed:
            recommended_min_peak = float(candidate)
            break
    if recommended_min_peak < requested_min_peak:
        warnings.append(f"lowered PEAK floor from {requested_min_peak:g} to {recommended_min_peak:g} to keep enough rows")

    recommended_geometry_percentile = float(requested_geometry_percentile)
    if observed_geometry_fraction < 0.5:
        recommended_geometry_percentile = 100.0
    elif np.isfinite(dr[geom]).any():
        p50 = _safe_percentile(dr[geom], 50)
        p99 = _safe_percentile(dr[geom], 99)
        if np.isfinite(p50) and p50 > 0 and np.isfinite(p99) and p99 > 25 * p50:
            recommended_geometry_percentile = min(recommended_geometry_percentile, 97.5)
            warnings.append("large detector-position tail detected; using stricter geometry percentile")

    row_budget = requested_scout_rows if requested_scout_rows > 0 else 1000
    scout_max_rows = max(100, min(int(row_budget), int(np.count_nonzero(base_d & positive & (peak >= recommended_min_peak)))))
    status = "ok"
    if scout_max_rows < 250:
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
    """Return a dataset copy with the preflight-recommended filters applied."""

    return DatasetSpec(
        name=ds.name,
        hkl=ds.hkl,
        xds_dir=ds.xds_dir,
        hkl_transform=ds.hkl_transform,
        min_d_spacing=ds.min_d_spacing,
        min_peak=decision.recommended_min_peak,
        target_mode=ds.target_mode,
        include_nonpositive=ds.include_nonpositive,
        geometry_percentile=decision.recommended_geometry_percentile,
    )


def expand_transform_candidates(current: str, spec: str) -> list[str]:
    """Build the signed HKL transform list used by the transform scout."""

    spec = (spec or "auto").strip()
    if spec.lower() in {"off", "none"}:
        return [current]
    out: list[str] = []
    raw: list[str] = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        if item == "auto":
            raw.extend([current, *COMMON_TRANSFORMS])
        elif item == "common":
            raw.extend(COMMON_TRANSFORMS)
        else:
            raw.append(item)
    if not raw:
        raw = [current, *COMMON_TRANSFORMS]
    for item in raw:
        item = "identity" if item in {"", "auto"} else item
        if item not in out:
            out.append(item)
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


def thickness_polish_nuisance(nui: dict) -> dict:
    """Narrow a nuisance grid around the best thickness and geometry values."""

    out = dict(nui)
    t = safe_float(nui.get("thickness_A"), 1200.0)
    delta = max(100.0, min(400.0, 0.18 * t))
    vals = sorted({max(100.0, t - delta), t, t + delta})
    out["thickness_A"] = ",".join(f"{v:.6g}" for v in vals)
    g = safe_float(nui.get("thickness_grad_frac"), 0.0)
    gvals = sorted({max(0.0, g - 0.2), g, min(1.5, g + 0.2)})
    out["thickness_grad_frac"] = ",".join(f"{v:.6g}" for v in gvals)
    f = safe_float(nui.get("frame_offset"), 0.0)
    out["frame_offset"] = ",".join(f"{v:.6g}" for v in sorted({f - 0.1, f, f + 0.1}))
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
