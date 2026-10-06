"""Frozen row-table serialization and content identity. The emitted column order and formatting are part of the existing artifact contract, not presentation choices."""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from grond.legacy.engine.bloch_engine import XDSData, sha256

from .observations import target_factor


def write_row_table(path: Path, data: XDSData, target, best_eval, masks, target_mode: str = "iobs"):
    """Write the native row table CSV of one evaluated trial: one line per valid row.

    Each line holds the row identity, target and sigmas, Sg and profile diagnostics, raw and scaled Bloch (dyn) and
    Born predictions, and residuals (scaled prediction - target) / sigma_eff. split is free for masks['free'] rows,
    else work.
    """

    arrays = best_eval["arrays"]
    source_factors = np.asarray(target_factor(data, target_mode), dtype=float)
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        fields = [
            "row_index", "h", "k", "l", "d_A", "target_I", "target_mode", "source_to_target_factor", "source_sigma", "sigma_eff", "xds_sigma", "peak", "corr", "rlp",
            "group", "detector_bin", "basis_size", "basis_support_hash64", "center_sg_inv_A", "omega_bragg_deg", "omega_bragg_minus_center_deg", "omega_bragg_root_available",
            "sg_window_inv_A", "sg_window_raw_inv_A", "sg_window_half_frames", "sg_window_clipped",
            "sg_dSg_domega_inv_A_per_deg", "sg_dSg_dframe_inv_A", "sg_jitter_sigma_inv_A",
            "sg_jitter_d_inv_A", "sg_jitter_frame_fraction", "sg_jitter_kernel_points",
            "sg_profile_points_total", "sg_profile_weight_sum",
            "sg_profile_same_branch_continuation_used",
            "sg_profile_same_branch_continuation_segments",
            "max_sg_solve_residual",
            "split", "row_resolution_factor", "dyn_raw_unmodulated", "dyn_raw", "born_raw", "q_dyn_over_born", "dyn_scaled", "born_scaled",
            "dyn_residual_over_sigma", "born_residual_over_sigma",
        ]
        if "scale_angle_deg" in arrays:
            # P-spline scale coordinate omega(ZCAL); per-frame tables keep the historical columns.
            fields.append("scale_angle_deg")
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        default_float = np.full(len(target), np.nan, dtype=float)
        default_int = np.zeros(len(target), dtype=int)
        default_one_int = np.ones(len(target), dtype=int)
        default_bool = np.zeros(len(target), dtype=bool)
        for i in np.where(arrays["valid"])[0]:
            split = "free" if masks["free"][i] else "work"
            q = arrays["dyn_raw"][i] / max(arrays["born_raw"][i], 1e-30)
            row = {
                "row_index": int(i),
                "h": int(data.hkl[i, 0]), "k": int(data.hkl[i, 1]), "l": int(data.hkl[i, 2]),
                "d_A": float(dvals[i]),
                "target_I": float(target[i]),
                "target_mode": str(target_mode),
                "source_to_target_factor": float(source_factors[i]),
                # Immutable integration sigma transformed by the same known
                # factor as target_I; xds_sigma retains the original units.
                "source_sigma": float(data.sigma[i] * abs(source_factors[i])),
                "sigma_eff": float(arrays["sigma_eff"][i]),
                "xds_sigma": float(data.sigma[i]),
                "peak": float(data.peak[i]),
                "corr": float(data.corr[i]),
                "rlp": float(data.rlp[i]),
                "group": int(arrays["groups"][i]),
                "detector_bin": int(arrays.get("detector_bins", np.zeros(len(target), dtype=int))[i]),
                "basis_size": int(arrays.get("basis_size", np.zeros(len(target), dtype=int))[i]),
                "basis_support_hash64": format(
                    int(arrays.get("basis_support_hash64", np.zeros(len(target), dtype=np.uint64))[i]),
                    "016x",
                ),
                "center_sg_inv_A": float(arrays.get("center_sg", default_float)[i]),
                "omega_bragg_deg": float(arrays.get("omega_bragg", default_float)[i]),
                "omega_bragg_minus_center_deg": float(arrays.get("omega_delta", default_float)[i]),
                "omega_bragg_root_available": int(
                    bool(arrays.get("omega_bragg_root_available", default_bool)[i])
                ),
                "sg_window_inv_A": float(arrays.get("sg_window", default_float)[i]),
                "sg_window_raw_inv_A": float(arrays.get("sg_window_raw", default_float)[i]),
                "sg_window_half_frames": float(arrays.get("sg_window_half_frames", default_float)[i]),
                "sg_window_clipped": int(bool(arrays.get("sg_window_clipped", default_bool)[i])),
                "sg_dSg_domega_inv_A_per_deg": float(arrays.get("sg_dSg_domega", default_float)[i]),
                "sg_dSg_dframe_inv_A": float(arrays.get("sg_dSg_dframe", default_float)[i]),
                "sg_jitter_sigma_inv_A": float(arrays.get("sg_jitter_sigma", default_float)[i]),
                "sg_jitter_d_inv_A": float(arrays.get("sg_jitter_d_inv", default_float)[i]),
                "sg_jitter_frame_fraction": float(arrays.get("sg_jitter_frame_fraction", default_float)[i]),
                "sg_jitter_kernel_points": int(arrays.get("sg_jitter_kernel_points", default_int)[i]),
                "sg_profile_points_total": int(arrays.get("sg_profile_points_total", default_int)[i]),
                "sg_profile_weight_sum": float(arrays.get("sg_profile_weight_sum", default_float)[i]),
                "sg_profile_same_branch_continuation_used": int(
                    bool(arrays.get("sg_profile_same_branch_continuation_used", default_bool)[i])
                ),
                "sg_profile_same_branch_continuation_segments": int(
                    arrays.get("sg_profile_same_branch_continuation_segments", default_one_int)[i]
                ),
                "max_sg_solve_residual": float(arrays.get("sg_residual", default_float)[i]),
                "split": split,
                "row_resolution_factor": float(arrays.get("row_resolution_factor", np.ones(len(target), dtype=float))[i]),
                "dyn_raw_unmodulated": float(arrays.get("dyn_raw_unmodulated", arrays["dyn_raw"])[i]),
                "dyn_raw": float(arrays["dyn_raw"][i]),
                "born_raw": float(arrays["born_raw"][i]),
                "q_dyn_over_born": float(q),
                "dyn_scaled": float(arrays["dyn_scaled"][i]),
                "born_scaled": float(arrays["born_scaled"][i]),
                "dyn_residual_over_sigma": float((arrays["dyn_scaled"][i] - target[i]) / max(arrays["sigma_eff"][i], 1e-12)),
                "born_residual_over_sigma": float((arrays["born_scaled"][i] - target[i]) / max(arrays["sigma_eff"][i], 1e-12)),
            }
            if "scale_angle_deg" in arrays:
                row["scale_angle_deg"] = float(arrays["scale_angle_deg"][i])
            w.writerow(row)


def native_row_table_artifact(path: Path) -> dict[str, str]:
    """Return the byte-bound native row-table artifact recorded in summary.json."""

    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"native row table is missing: {resolved}")
    return {
        "native_row_table": str(resolved),
        "native_row_table_sha256": sha256(resolved),
    }
