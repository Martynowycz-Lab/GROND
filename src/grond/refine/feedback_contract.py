"""Pinned feedback arguments, selected-endpoint identity, and cycle summaries. Reopening whichever RES sorts first is unsafe; the adopted model must match the recorded endpoint and health hash."""

from __future__ import annotations

import json
import math
from pathlib import Path

from .shelxl_diagnostics import shelx_res_health


def read_json(path: Path) -> dict:
    """Read a JSON file, allowing GROND summaries that contain NaN tokens."""

    return json.loads(path.read_text())


def finite(value) -> bool:
    """Return whether a value is a finite float."""

    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def grid_value(value, default: str = "0.0") -> str:
    """Format a pinned scalar as a one-value GROND grid string."""

    if value is None:
        return default
    if isinstance(value, str):
        return value
    if finite(value):
        return f"{float(value):.12g}"
    return default


def summary_value(summary: dict, key: str, default=None):
    """Return a config value first, then a best-nuisance value."""

    cfg = summary.get("config") or {}
    nui = ((summary.get("best_native") or {}).get("nuisance") or {})
    if key in cfg:
        return cfg[key]
    return nui.get(key, default)


def nuisance(summary: dict) -> dict:
    """Return the selected nuisance dictionary from a native summary."""

    out = dict((summary.get("best_native") or {}).get("nuisance") or {})
    if not out:
        raise SystemExit("source summary does not contain best_native.nuisance")
    return out


def _path_within(path: Path, root: Path) -> bool:
    """Return whether ``path`` resolves inside ``root`` without raising."""

    try:
        path.resolve().relative_to(root.resolve())
    except (OSError, ValueError):
        return False
    return True


def endpoint_res(run_dir: Path, summary: dict | None = None) -> Path:
    """Return the exact summary-selected no-EXTI B=0 endpoint RES.

    A production feedback cycle passes its freshly written summary.  This
    binds the adopted model to the selected lambda/B/EXTI record rather than
    to whichever old endpoint directory happens to sort first on a resumed
    output tree.  The summary-free branch is retained only for older direct
    callers and diagnostics.
    """

    if summary is not None:
        endpoint = summary.get("best_endpoint")
        endpoint = endpoint if isinstance(endpoint, dict) else {}
        shelxl = endpoint.get("shelxl")
        shelxl = shelxl if isinstance(shelxl, dict) else {}
        if str(shelxl.get("status", "")).strip().lower() != "ok":
            raise FileNotFoundError(
                "summary-selected endpoint does not have a successful SHELXL status: "
                f"{shelxl.get('status', 'missing')}"
            )
        try:
            lam = float(endpoint["lambda_q"])
            endpoint_b = float(endpoint.get("endpoint_b", 0.0))
        except (KeyError, TypeError, ValueError) as exc:
            raise FileNotFoundError("summary-selected endpoint lacks finite lambda_q/endpoint_b") from exc
        keep_exti = bool(endpoint.get("keep_exti", False))
        if not math.isfinite(lam) or not math.isfinite(endpoint_b):
            raise FileNotFoundError("summary-selected endpoint has non-finite lambda_q/endpoint_b")
        if abs(endpoint_b) > 1e-12 or keep_exti:
            raise FileNotFoundError(
                "feedback requires the summary-selected B=0 no-EXTI endpoint; "
                f"got B={endpoint_b:g}, keep_exti={keep_exti}"
            )
        lst_value = str(shelxl.get("lst") or "")
        if not lst_value:
            raise FileNotFoundError("summary-selected endpoint does not record its SHELXL LST path")
        candidate = Path(lst_value).expanduser().with_suffix(".res")
        lam_tag = str(lam).replace(".", "p")
        b_tag = str(endpoint_b).replace("-", "m").replace(".", "p")
        expected_dir = run_dir / f"endpoint_lam_{lam_tag}_B_{b_tag}_noexti"
        expected = expected_dir / "dyn_nuisance_corrected.res"
        if candidate.resolve() != expected.resolve() or not _path_within(candidate, run_dir):
            raise FileNotFoundError(
                "summary-selected endpoint path disagrees with lambda/B/EXTI identity: "
                f"recorded={candidate}, expected={expected}"
            )
        health = shelx_res_health(candidate, candidate.with_suffix(".lst"))
        recorded_health = shelxl.get("res_health")
        recorded_health = recorded_health if isinstance(recorded_health, dict) else {}
        recorded_path = str(recorded_health.get("path") or "")
        recorded_sha = str(recorded_health.get("sha256") or "")
        try:
            recorded_atom_count = int(recorded_health.get("atom_count", 0) or 0)
            recorded_sfac_count = int(recorded_health.get("sfac_count", 0) or 0)
        except (TypeError, ValueError, OverflowError) as exc:
            raise FileNotFoundError("summary-selected endpoint health has invalid model counts") from exc
        if (
            not recorded_health.get("ok")
            or recorded_health.get("safe_for_feedback") is False
            or recorded_atom_count <= 0
            or recorded_sfac_count <= 0
            or not recorded_path
            or not recorded_sha
        ):
            raise FileNotFoundError(
                "summary-selected endpoint lacks a complete safe path/hash-bound health record"
            )
        if Path(recorded_path).resolve() != candidate.resolve():
            raise FileNotFoundError(
                "summary-selected endpoint health path disagrees with its LST/model path: "
                f"health={recorded_path}, model={candidate}"
            )
        if recorded_sha != str(health.get("sha256") or ""):
            raise FileNotFoundError("summary-selected endpoint model hash no longer matches the recorded health hash")
        if not health.get("ok") or health.get("safe_for_feedback") is False:
            raise FileNotFoundError(f"summary-selected endpoint is not safe for feedback: {health}")
        return candidate

    direct = run_dir / "endpoint_lam_1p0_B_0p0_noexti" / "dyn_nuisance_corrected.res"
    direct_health = shelx_res_health(direct, direct.with_suffix(".lst"))
    if direct_health["ok"] and direct_health.get("safe_for_feedback") is not False:
        return direct
    matches = sorted(run_dir.glob("endpoint_lam_*_B_0p0_noexti/dyn_nuisance_corrected.res"))
    rejected = []
    for match in matches:
        health = shelx_res_health(match, match.with_suffix(".lst"))
        if health["ok"] and health.get("safe_for_feedback") is not False:
            return match
        rejected.append(health)
    detail = f"; rejected={rejected}" if rejected else ""
    raise FileNotFoundError(f"no usable no-EXTI B=0 endpoint RES found under {run_dir}{detail}")


def selected_metrics(summary: dict) -> dict:
    """Extract compact native and endpoint metrics for one cycle."""

    best = summary.get("best_native") or {}
    metrics = best.get("metrics") or {}
    all_dyn = ((metrics.get("all") or {}).get("dyn") or {})
    all_born = ((metrics.get("all") or {}).get("born") or {})
    work_dyn = ((metrics.get("work") or {}).get("dyn") or {})
    free_dyn = ((metrics.get("free") or {}).get("dyn") or {})
    work_born = ((metrics.get("work") or {}).get("born") or {})
    free_born = ((metrics.get("free") or {}).get("born") or {})
    endpoint = summary.get("best_endpoint") or {}
    shelxl = endpoint.get("shelxl") or {}
    post_sigma = endpoint.get("post_bloch_sigma") or {}
    q_control = best.get("q_control") or {}
    return {
        "native_dyn_R1_all": all_dyn.get("R1_all"),
        "native_dyn_R1_gt": all_dyn.get("R1_gt"),
        "native_dyn_wR2_all": all_dyn.get("wR2_all"),
        "native_dyn_CC": all_dyn.get("CC"),
        "native_dyn_work_R1_all": work_dyn.get("R1_all"),
        "native_dyn_work_wR2_all": work_dyn.get("wR2_all"),
        "native_dyn_work_CC": work_dyn.get("CC"),
        "native_dyn_free_R1_all": free_dyn.get("R1_all"),
        "native_dyn_free_wR2_all": free_dyn.get("wR2_all"),
        "native_dyn_free_CC": free_dyn.get("CC"),
        "native_born_R1_all": all_born.get("R1_all"),
        "native_born_wR2_all": all_born.get("wR2_all"),
        "native_born_work_wR2_all": work_born.get("wR2_all"),
        "native_born_free_wR2_all": free_born.get("wR2_all"),
        "endpoint_R1_gt": shelxl.get("R1_gt"),
        "endpoint_R1_all": shelxl.get("R1_all"),
        "endpoint_wR2_all": shelxl.get("wR2_all"),
        "endpoint_GooF": shelxl.get("GooF"),
        "endpoint_N_gt": shelxl.get("N_gt"),
        "endpoint_N_all": shelxl.get("N_all"),
        "endpoint_status": shelxl.get("status"),
        "post_bloch_sigma_policy": post_sigma.get("policy"),
        "post_bloch_sigma_status": post_sigma.get("status"),
        "post_bloch_sigma_scale": post_sigma.get("sigma_scale"),
        "post_bloch_sigma_heldout_rms_z_after": post_sigma.get("heldout_rms_z_after"),
        "post_bloch_sigma_all_rms_z_after": post_sigma.get("all_rms_z_after"),
        "q_control": q_control.get("mode", "real"),
        "q_control_shuffled_rows": q_control.get("shuffled_rows", 0),
        "q_control_active_bins": q_control.get("active_bins", 0),
        "q_control_logq_corr": q_control.get("logq_corr_original_shuffled"),
    }


def feedback_prediction_cache_dir(run_dir: Path) -> Path:
    """Return the hash-validated cache shared by all cycles in one feedback run."""

    return run_dir.parent / "cache" / "shared"
