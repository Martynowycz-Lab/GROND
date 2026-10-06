# ruff: noqa: E741
# h, k, l retain standard crystallographic Miller-index notation.
"""Persist and replay row geometry while preserving exact source/model contracts."""

from __future__ import annotations

import hashlib
import json
import os
import time
import zipfile
from pathlib import Path

import numpy as np

from grond.bloch.cache import row_plan_cache_validation_errors
from grond.bloch.constants import ROW_PLAN_CACHE_SCHEMA
from grond.bloch.hamiltonian import validate_passive_absorption
from grond.bloch.model import model_content_hash
from grond.bloch.numerics import _rotation_matrix_axis_angle, sha256, validate_row_indices


class RowCacheMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def row_plan_cache_mode(self) -> str:
        """Return the requested frozen row-plan cache mode."""

        mode = str(getattr(self.cfg, "row_plan_cache", "off") or "off").lower()
        if mode in {"0", "false", "no", "none"}:
            return "off"
        if mode in {"1", "true", "yes", "on"}:
            return "readwrite"
        if mode not in {"off", "exact", "readwrite", "rebuild"}:
            raise ValueError(f"unknown row_plan_cache={mode!r}; expected off, exact, readwrite, or rebuild")
        return mode

    def row_plan_cache_key(self, indices):
        """Return a geometry/config cache key for frozen row plans.

        ``exact`` keys the cache to the current structural model.  It is safe
        for repeated nuisance calculations at fixed coordinates because only
        thickness, scale, or other non-basis terms change.  ``readwrite``
        intentionally omits the current model and retains the older opt-in
        frozen-basis behavior for native finite-difference experiments.
        """

        indices = validate_row_indices(indices, row_count=len(self.data.iobs))
        source_hashes = self.data.verify_source_files()
        if self.geometry_path is not None:
            path = Path(self.geometry_path["path"])
            if sha256(path) != self.geometry_path["sha256"]:
                raise ValueError(f"loaded geometry path changed after reading: {path}; reload the Bloch port")
        cfg_fields = {
            "cache_schema": ROW_PLAN_CACHE_SCHEMA,
            "hkl_sha256": source_hashes["hkl"],
            "hkl_transform": self.data.hkl_transform_spec,
            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "n_basis": int(self.cfg.n_basis),
            "n_points": int(self.cfg.n_points),
            "sg_window": float(self.cfg.sg_window),
            "sg_window_mode": str(self.cfg.sg_window_mode),
            "sg_window_min": float(self.cfg.sg_window_min),
            "sg_window_max": float(self.cfg.sg_window_max),
            "sg_window_scale": float(self.cfg.sg_window_scale),
            "sg_window_frame_floor": float(self.cfg.sg_window_frame_floor),
            "sg_jitter_mode": str(self.cfg.sg_jitter_mode),
            "sg_jitter_sigma0": float(self.cfg.sg_jitter_sigma0),
            "sg_jitter_sigma_d_inv": float(self.cfg.sg_jitter_sigma_d_inv),
            "sg_jitter_sigma_frame": float(self.cfg.sg_jitter_sigma_frame),
            "sg_jitter_points": int(self.cfg.sg_jitter_points),
            "sg_jitter_student_df": float(self.cfg.sg_jitter_student_df),
            "sg_jitter_width_factor": float(self.cfg.sg_jitter_width_factor),
            "mosaicity_mode": str(self.cfg.mosaicity_mode),
            "mosaicity_sigma_deg": float(self.cfg.mosaicity_sigma_deg),
            "mosaicity_max_sigma_deg": float(self.cfg.mosaicity_max_sigma_deg),
            "mosaicity_points": int(self.cfg.mosaicity_points),
            "mosaicity_support_sigma_deg": float(self.cfg.mosaicity_support_sigma_deg),
            "mosaicity_support_points": int(self.cfg.mosaicity_support_points),
            "mosaicity_support_width_sigma": float(self.cfg.mosaicity_support_width_sigma),
            "profile_coordinate": str(self.cfg.profile_coordinate),
            "profile_basis_mode": str(self.cfg.profile_basis_mode),
            "profile_angle_max_half_width_deg": float(self.cfg.profile_angle_max_half_width_deg),
            "geometry_path_sha256": (str(self.geometry_path["sha256"]) if self.geometry_path is not None else ""),
            "profile_quadrature": str(self.cfg.profile_quadrature),
            "profile_accumulation": str(self.cfg.profile_accumulation),
            "topup_d_min": float(self.cfg.topup_d_min),
            "ug_d_min": float(self.cfg.ug_d_min),
            "max_support_sg": float(self.cfg.max_support_sg),
            "frame_offset": float(self.cfg.frame_offset),
            "matrix_convention": str(self.cfg.matrix_convention),
            "dose_b_mode": str(self.cfg.dose_b_mode),
            "dose_b_per_frame": float(self.cfg.dose_b_per_frame),
            "dose_b_ref_frame": float(self.dose_ref_frame),
            "cell": [float(v) for v in self.model["cell"]],
            "wavelength": float(self.data.wavelength),
            "scan_matrix_sha256": source_hashes.get("scan_matrix", ""),
            "pets_exact_geometry_sha256": source_hashes.get("pets_exact_geometry", ""),
        }
        if getattr(self.data, "zcal_frame_convention", "scan_start") != "scan_start":
            # Keys of files read in the legacy scan-start mapping are unchanged; XDS-convention rows get their own.
            cfg_fields["zcal_frame_convention"] = str(self.data.zcal_frame_convention)
        if self.row_plan_cache_mode() == "exact":
            cfg_fields.update(
                {
                    "basis_model_content_sha256": model_content_hash(self.model),
                    "basis_res_sha256": self.res_hash,
                    "basis_policy": "exact_fixed_model",
                }
            )
        text = json.dumps(cfg_fields, sort_keys=True, allow_nan=True)
        return hashlib.sha256(text.encode()).hexdigest()[:20], cfg_fields

    def row_plan_cache_path(self, key: str) -> Path:
        """Return the .npz path of a row-plan cache key under ``cache_dir/row_plans`` (requires cache_dir)."""

        if self.cache_dir is None:
            raise ValueError("row-plan cache requires cache_dir")
        return self.cache_dir / "row_plans" / f"row_plan_{key}.npz"

    def build_row_plan_cache(self, indices):
        """Build immutable row geometry and basis plans for later reuse."""

        indices = validate_row_indices(indices, row_count=len(self.data.iobs))
        key, meta = self.row_plan_cache_key(indices)
        meta = dict(meta)
        meta.update(
            {
                "key": key,
                "basis_model_content_sha256": model_content_hash(self.model),
                "basis_res_sha256": self.res_hash,
                "basis_policy": (
                    "exact_fixed_model" if self.row_plan_cache_mode() == "exact" else "frozen_from_cache_builder_model"
                ),
            }
        )
        unique_diff_index: dict[tuple[int, int, int], int] = {}
        unique_diffs: list[tuple[int, int, int]] = []
        basis_offsets = [0]
        basis_flat = []
        profile_offsets = [0]
        profile_weights_flat = []
        profile_omega_flat = []
        profile_tilts_flat = []
        sg_offsets = [0]
        sg_basis_flat = []
        diff_offsets = [0]
        diff_idx_flat = []
        center_sg = []
        omega_center = []
        omega_bragg = []
        omega_delta = []
        omega_bragg_root_available = []
        sg_window = []
        sg_domega = []
        sg_dframe = []
        sg_window_raw = []
        sg_window_half_frames = []
        sg_window_clipped = []
        sg_jitter_sigma = []
        sg_jitter_d_inv = []
        sg_jitter_frame_fraction = []
        sg_jitter_kernel_points = []
        sg_profile_points_total = []
        sg_profile_weight_sum = []
        sg_profile_same_branch_continuation_used = []
        sg_profile_same_branch_continuation_segments = []
        sg_residual_max = []
        dose_delta_b = []
        convention = str(getattr(self.cfg, "matrix_convention", "j_minus_i") or "j_minus_i")
        if convention not in {"j_minus_i", "i_minus_j"}:
            raise ValueError(f"unknown matrix_convention={convention!r}; expected j_minus_i or i_minus_j")

        t0 = time.perf_counter()
        for pos, row_idx in enumerate(indices):
            row_idx = int(row_idx)
            hkl = self.data.hkl[row_idx]
            if getattr(self.data, "pets_exact_a_center", None) is not None:
                orientations, profile_angles, profile_weights, semiangle, ordinal = self.pets_exact_orientation_samples(
                    row_idx
                )
                profile_omegas = np.asarray(profile_angles, dtype=float)
                profile_tilts = np.zeros((len(profile_omegas), 2), dtype=float)
                center = np.asarray(self.data.pets_exact_a_center[ordinal], dtype=float)
                c_sg = float(self.pets_exact_sg_for_hkls(np.asarray([hkl], dtype=int), center)[0])
                derivative_step = 1e-4
                plus = _rotation_matrix_axis_angle(np.asarray([1.0, 0.0, 0.0]), derivative_step) @ center
                minus = _rotation_matrix_axis_angle(np.asarray([1.0, 0.0, 0.0]), -derivative_step) @ center
                derivative = float(
                    (
                        self.pets_exact_sg_for_hkls(np.asarray([hkl], dtype=int), plus)[0]
                        - self.pets_exact_sg_for_hkls(np.asarray([hkl], dtype=int), minus)[0]
                    )
                    / (2.0 * derivative_step)
                )
                sg_window_i = 2.0 * abs(derivative) * semiangle
                basis, _pets_basis_meta = self.pets_exact_select_common_basis(hkl, orientations, row_idx)
                sg_basis_rows = [
                    np.asarray(
                        self.pets_exact_sg_for_hkls(basis, orientation),
                        dtype=float,
                    )
                    for orientation in orientations
                ]
                # These legacy fields are not used for exact PETS propagation,
                # but the cache validator requires finite fixed-width arrays.
                omega_c = 0.0
                omega_b = 0.0
                omega_b_available = False
                sg_resid_values = np.asarray([], dtype=float)
                sg_width_meta = {
                    "sg_dSg_domega_inv_A_per_deg": derivative,
                    "sg_dSg_dframe_inv_A": derivative * float(self.data.oscillation_range),
                    "sg_window_raw_inv_A": sg_window_i,
                    "sg_window_half_frames": semiangle / max(abs(float(self.data.oscillation_range)), 1e-30),
                    "sg_window_clipped": False,
                }
                jitter_meta = {
                    "sg_jitter_sigma_inv_A": 0.0,
                    "sg_jitter_d_inv_A": float("nan"),
                    "sg_jitter_frame_fraction": float("nan"),
                    "sg_jitter_kernel_points": 1,
                    "sg_profile_points_total": int(len(orientations)),
                    "sg_profile_weight_sum": float(np.sum(profile_weights)),
                    "sg_profile_same_branch_continuation_used": False,
                    "sg_profile_same_branch_continuation_segments": 1,
                }
            else:
                zcal = self.data.zcal[row_idx]
                omega_c = self.omega_from_z(zcal)
                c_sg = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega_c)[0])
                omega_b, omega_b_available = self.row_diagnostic_omega_bragg(hkl, omega_c)
                sg_window_i, sg_width_meta = self.row_sg_window(row_idx, hkl, omega_c)
                (
                    solved_omegas,
                    mosaic_tilts,
                    profile_weights,
                    sg_resid_values,
                    jitter_meta,
                    _mosaicity_meta,
                ) = self.row_profile_orientation_points(row_idx, hkl, omega_c, c_sg, sg_window_i)
                profile_omegas = np.asarray(solved_omegas, dtype=float)
                profile_tilts = np.asarray(mosaic_tilts, dtype=float).reshape(-1, 2)
                basis, _cone_basis_meta = self.select_basis_for_profile(
                    hkl,
                    omega_c,
                    sg_window_i,
                    solved_omegas,
                    mosaic_tilts,
                    row_idx=row_idx,
                )
                sg_basis_rows = []
                for omega, (along_deg, transverse_deg) in zip(solved_omegas, mosaic_tilts, strict=True):
                    sg_basis = self.sg_for_hkls_at_orientation(basis, omega, along_deg, transverse_deg)
                    sg_basis_rows.append(np.asarray(sg_basis, dtype=float))
            if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(
                basis[1], np.asarray(hkl, dtype=int)
            ):
                raise RuntimeError(
                    "Frozen row-plan basis contract violated: basis must start with direct beam then exact signed target hkl."
                )

            diffs = (
                basis[None, :, :] - basis[:, None, :]
                if convention == "j_minus_i"
                else basis[:, None, :] - basis[None, :, :]
            )
            row_diff_idx = []
            for diff in diffs.reshape(-1, 3):
                tup = tuple(int(v) for v in diff)
                idx = unique_diff_index.get(tup)
                if idx is None:
                    idx = len(unique_diffs)
                    unique_diff_index[tup] = idx
                    unique_diffs.append(tup)
                row_diff_idx.append(idx)

            basis_flat.extend(tuple(int(v) for v in b) for b in basis)
            basis_offsets.append(len(basis_flat))
            if not (len(profile_weights) == len(profile_omegas) == len(profile_tilts)):
                raise RuntimeError("profile weights, orientations, and tilts must have identical lengths")
            profile_weights_flat.extend(float(v) for v in profile_weights)
            profile_omega_flat.extend(float(value) for value in profile_omegas)
            profile_tilts_flat.extend((float(row[0]), float(row[1])) for row in profile_tilts)
            profile_offsets.append(len(profile_weights_flat))
            sg_basis_arr = np.asarray(sg_basis_rows, dtype=float)
            sg_basis_flat.extend(float(v) for v in sg_basis_arr.reshape(-1))
            sg_offsets.append(len(sg_basis_flat))
            diff_idx_flat.extend(int(v) for v in row_diff_idx)
            diff_offsets.append(len(diff_idx_flat))

            center_sg.append(c_sg)
            omega_center.append(float(omega_c))
            omega_bragg.append(float(omega_b))
            omega_delta.append(float(omega_b - omega_c))
            omega_bragg_root_available.append(bool(omega_b_available))
            sg_window.append(float(sg_window_i))
            sg_domega.append(float(sg_width_meta["sg_dSg_domega_inv_A_per_deg"]))
            sg_dframe.append(float(sg_width_meta["sg_dSg_dframe_inv_A"]))
            sg_window_raw.append(float(sg_width_meta["sg_window_raw_inv_A"]))
            sg_window_half_frames.append(float(sg_width_meta["sg_window_half_frames"]))
            sg_window_clipped.append(bool(sg_width_meta["sg_window_clipped"]))
            sg_jitter_sigma.append(float(jitter_meta["sg_jitter_sigma_inv_A"]))
            sg_jitter_d_inv.append(float(jitter_meta["sg_jitter_d_inv_A"]))
            sg_jitter_frame_fraction.append(float(jitter_meta["sg_jitter_frame_fraction"]))
            sg_jitter_kernel_points.append(int(jitter_meta["sg_jitter_kernel_points"]))
            sg_profile_points_total.append(int(jitter_meta["sg_profile_points_total"]))
            sg_profile_weight_sum.append(float(jitter_meta["sg_profile_weight_sum"]))
            sg_profile_same_branch_continuation_used.append(
                bool(jitter_meta["sg_profile_same_branch_continuation_used"])
            )
            sg_profile_same_branch_continuation_segments.append(
                int(jitter_meta["sg_profile_same_branch_continuation_segments"])
            )
            sg_residual_max.append(float(np.max(np.abs(sg_resid_values))) if len(sg_resid_values) else 0.0)
            dose_delta_b.append(float(self.row_delta_b(row_idx)))
            if (pos + 1) % 500 == 0 or pos + 1 == len(indices):
                elapsed = time.perf_counter() - t0
                rate = (pos + 1) / max(elapsed, 1e-12)
                eta = (len(indices) - (pos + 1)) / max(rate, 1e-12)
                print(f"built row plans {pos + 1}/{len(indices)} | rate={rate:.2f}/s | ETA={eta / 60:.1f}m", flush=True)

        unique_diffs_arr = np.asarray(unique_diffs, dtype=np.int32).reshape(-1, 3)
        unique_inv_d2 = self.inv_d2_for_hkls(unique_diffs_arr) if len(unique_diffs_arr) else np.zeros(0, dtype=float)
        if self.row_plan_cache_key(indices)[0] != key:
            raise ValueError("row-plan inputs changed during construction; refusing to return the result")
        return {
            "meta": meta,
            "indices": indices.astype(np.int64),
            "basis_offsets": np.asarray(basis_offsets, dtype=np.int64),
            "basis_flat": np.asarray(basis_flat, dtype=np.int32).reshape(-1, 3),
            "profile_offsets": np.asarray(profile_offsets, dtype=np.int64),
            "profile_weights_flat": np.asarray(profile_weights_flat, dtype=float),
            "profile_omega_flat": np.asarray(profile_omega_flat, dtype=float),
            "profile_tilts_flat": np.asarray(profile_tilts_flat, dtype=float).reshape(-1, 2),
            "sg_offsets": np.asarray(sg_offsets, dtype=np.int64),
            "sg_basis_flat": np.asarray(sg_basis_flat, dtype=float),
            "diff_offsets": np.asarray(diff_offsets, dtype=np.int64),
            "diff_idx_flat": np.asarray(diff_idx_flat, dtype=np.int64),
            "unique_diffs": unique_diffs_arr,
            "unique_inv_d2": np.asarray(unique_inv_d2, dtype=float),
            "center_sg": np.asarray(center_sg, dtype=float),
            "omega_center": np.asarray(omega_center, dtype=float),
            "omega_bragg": np.asarray(omega_bragg, dtype=float),
            "omega_delta": np.asarray(omega_delta, dtype=float),
            "omega_bragg_root_available": np.asarray(omega_bragg_root_available, dtype=bool),
            "sg_window": np.asarray(sg_window, dtype=float),
            "sg_domega": np.asarray(sg_domega, dtype=float),
            "sg_dframe": np.asarray(sg_dframe, dtype=float),
            "sg_window_raw": np.asarray(sg_window_raw, dtype=float),
            "sg_window_half_frames": np.asarray(sg_window_half_frames, dtype=float),
            "sg_window_clipped": np.asarray(sg_window_clipped, dtype=bool),
            "sg_jitter_sigma": np.asarray(sg_jitter_sigma, dtype=float),
            "sg_jitter_d_inv": np.asarray(sg_jitter_d_inv, dtype=float),
            "sg_jitter_frame_fraction": np.asarray(sg_jitter_frame_fraction, dtype=float),
            "sg_jitter_kernel_points": np.asarray(sg_jitter_kernel_points, dtype=np.int32),
            "sg_profile_points_total": np.asarray(sg_profile_points_total, dtype=np.int32),
            "sg_profile_weight_sum": np.asarray(sg_profile_weight_sum, dtype=float),
            "sg_profile_same_branch_continuation_used": np.asarray(
                sg_profile_same_branch_continuation_used,
                dtype=bool,
            ),
            "sg_profile_same_branch_continuation_segments": np.asarray(
                sg_profile_same_branch_continuation_segments,
                dtype=np.int32,
            ),
            "sg_residual_max": np.asarray(sg_residual_max, dtype=float),
            "dose_delta_b": np.asarray(dose_delta_b, dtype=float),
        }

    def save_row_plan_cache(self, path: Path, cache: dict) -> None:
        """Write a row-plan cache as compressed .npz via a temporary file and rename, with ``meta`` as JSON."""

        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_name(f"{path.stem}.{os.getpid()}.tmp.npz")
        save = {k: v for k, v in cache.items() if k != "meta"}
        np.savez_compressed(tmp, **save, meta_json=json.dumps(cache.get("meta", {}), sort_keys=True, allow_nan=True))
        os.replace(tmp, path)

    def load_row_plan_cache_file(self, path: Path) -> dict:
        """Read a row-plan cache .npz written by save_row_plan_cache, decoding ``meta`` from JSON."""

        with np.load(path, allow_pickle=False) as loaded:
            if "meta_json" not in loaded.files:
                raise ValueError(f"row-plan cache {path} has no metadata")
            cache = {k: loaded[k] for k in loaded.files if k != "meta_json"}
            cache["meta"] = json.loads(str(loaded["meta_json"]))
        return cache

    def load_or_build_row_plan_cache(self, indices):
        """Return the row-plan cache for ``indices``, or None when the cache is off.

        exact and readwrite reuse a valid file; a stale, incomplete or unreadable one is reported and rebuilt, and
        rebuild mode always rebuilds. ``status`` records hit, built or rebuilt.
        """

        mode = self.row_plan_cache_mode()
        if mode == "off":
            return None
        key, meta = self.row_plan_cache_key(indices)
        path = self.row_plan_cache_path(key)
        if mode in {"exact", "readwrite"} and path.exists():
            try:
                cache = self.load_row_plan_cache_file(path)
            except (OSError, ValueError, TypeError, EOFError, zipfile.BadZipFile) as exc:
                print(
                    f"Ignoring unreadable row-plan cache {path}: {type(exc).__name__}: {exc}",
                    flush=True,
                )
            else:
                errors = row_plan_cache_validation_errors(cache, indices, key, meta)
                if not errors:
                    if self.row_plan_cache_key(indices)[0] != key:
                        raise ValueError("row-plan inputs changed while loading the cache")
                    cache["status"] = {
                        "status": "hit",
                        "mode": mode,
                        "path": str(path),
                        "key": key,
                    }
                    return cache
                print(
                    "Ignoring stale or incomplete row-plan cache "
                    f"{path}: {'; '.join(errors[:4])}{'; ...' if len(errors) > 4 else ''}",
                    flush=True,
                )
        cache = self.build_row_plan_cache(indices)
        if self.row_plan_cache_key(indices)[0] != key:
            raise ValueError("row-plan inputs changed during construction; refusing to cache the result")
        self.save_row_plan_cache(path, cache)
        cache["status"] = {
            "status": "built" if mode in {"exact", "readwrite"} else "rebuilt",
            "mode": mode,
            "path": str(path),
            "key": key,
        }
        return cache

    def current_unique_ug(self, unique_diffs: np.ndarray) -> np.ndarray:
        """Compute current-model U_g only for the cache's unique differences."""

        unique_diffs = np.asarray(unique_diffs, dtype=int).reshape(-1, 3)
        out = np.zeros(len(unique_diffs), dtype=np.complex128)
        if len(unique_diffs) == 0:
            return out
        # Active differences may extend beyond the optional dense cache.
        return self.calculate_ug_array(unique_diffs)

    def cached_static_hamiltonian(
        self, basis, diff_idx, unique_diffs, unique_inv_d2, unique_ug, row_idx, absorb: bool, dose_delta_b=None
    ):
        """Return a row's static Hamiltonian from cached beam-difference lookups (see hamiltonian_static_part).

        ``diff_idx`` maps the n*n beam differences into ``unique_diffs``, ``unique_inv_d2`` and ``unique_ug``. Dose
        attenuation uses ``dose_delta_b`` when given, else the row's current value; absorption terms and the passivity
        check are those of hamiltonian_static_part.
        """

        n = len(basis)
        double_prec = bool(getattr(self.cfg, "double_precision", False))
        h_dtype = np.complex128 if double_prec else np.complex64
        H = np.zeros((n, n), dtype=h_dtype)
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        if absorb and mode in {"mean", "midgley_like", "midgley"}:
            H[np.diag_indices(n)] += 1j * float(getattr(self.cfg, "absorption_mu", 0.0))
        elif absorb and mode == "beam_dependent":
            H[np.diag_indices(n)] += 1j * self.beam_absorption_mu(basis)
        u_mat = np.asarray(unique_ug[diff_idx], dtype=h_dtype).reshape(n, n).copy()
        delta_b = float(self.row_delta_b(row_idx) if dose_delta_b is None else dose_delta_b)
        if delta_b != 0.0:
            inv_d2 = np.asarray(unique_inv_d2[diff_idx], dtype=float).reshape(n, n)
            u_mat *= np.exp(-0.25 * delta_b * inv_d2)
        if absorb:
            diffs = np.asarray(unique_diffs[diff_idx], dtype=int).reshape(n, n, 3)
            eta = self.offdiag_absorption_eta_array(diffs)
            if np.any(eta):
                u_mat = u_mat + 1j * eta * u_mat
        np.fill_diagonal(u_mat, 0.0 + 0.0j)
        H += u_mat / (2.0 * self.data.k)
        if absorb and self.absorption_enabled():
            validate_passive_absorption(H, mode=mode, row_index=row_idx)
        return H

    def row_prediction_from_cache(self, cache: dict, pos: int, unique_ug: np.ndarray):
        """Return (row index, result) for cache position ``pos``, predicted with the current U_g values.

        Profile-weighted thickness averages of the cached profile points give ``dyn`` and ``born`` (and ``dyn_inverse``
        with paired_inverse_hand); the other fields are the cached row geometry, as in row_prediction.
        """

        row_idx = int(cache["indices"][pos])
        b0, b1 = int(cache["basis_offsets"][pos]), int(cache["basis_offsets"][pos + 1])
        p0, p1 = int(cache["profile_offsets"][pos]), int(cache["profile_offsets"][pos + 1])
        s0, s1 = int(cache["sg_offsets"][pos]), int(cache["sg_offsets"][pos + 1])
        d0, d1 = int(cache["diff_offsets"][pos]), int(cache["diff_offsets"][pos + 1])
        basis = np.asarray(cache["basis_flat"][b0:b1], dtype=int)
        n = len(basis)
        weights = np.asarray(cache["profile_weights_flat"][p0:p1], dtype=float)
        sg_basis_rows = np.asarray(cache["sg_basis_flat"][s0:s1], dtype=float).reshape(len(weights), n)
        diff_idx = np.asarray(cache["diff_idx_flat"][d0:d1], dtype=np.int64)
        H_dyn_static = self.cached_static_hamiltonian(
            basis,
            diff_idx,
            cache["unique_diffs"],
            cache["unique_inv_d2"],
            unique_ug,
            row_idx,
            absorb=True,
            dose_delta_b=float(cache["dose_delta_b"][pos]) if "dose_delta_b" in cache else None,
        )
        if self.absorption_enabled():
            H_born_static = self.cached_static_hamiltonian(
                basis,
                diff_idx,
                cache["unique_diffs"],
                cache["unique_inv_d2"],
                unique_ug,
                row_idx,
                absorb=False,
                dose_delta_b=float(cache["dose_delta_b"][pos]) if "dose_delta_b" in cache else None,
            )
        else:
            H_born_static = H_dyn_static
        h10_born = H_born_static[1, 0]
        dyn_vals = []
        inverse_dyn_vals = []
        born_vals = []
        for sg_basis in sg_basis_rows:
            H = self.hamiltonian_from_static(H_dyn_static, sg_basis)
            if bool(getattr(self.cfg, "paired_inverse_hand", False)):
                forward, reverse = self.propagate_direct_to_target_thickness_average_pair(H)
                dyn_vals.append(forward)
                inverse_dyn_vals.append(reverse)
            else:
                dyn_vals.append(self.propagate_direct_to_target_thickness_average(H))
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            born_vals.append(self.born_thickness_average(h10_born, delta))
        dyn = float(np.sum(weights * np.asarray(dyn_vals, dtype=float)))
        born = float(np.sum(weights * np.asarray(born_vals, dtype=float)))
        support = [[int(v) for v in row] for row in basis.tolist()]
        result = {
            "dyn": dyn,
            "born": born,
            "basis_size": int(n),
            "support_hkls": support,
            "max_sg_solve_residual": float(cache["sg_residual_max"][pos]) if "sg_residual_max" in cache else 0.0,
            "center_sg_inv_A": float(cache["center_sg"][pos]),
            "omega_center_deg": float(cache["omega_center"][pos]),
            "omega_bragg_deg": float(cache["omega_bragg"][pos]),
            "omega_bragg_minus_center_deg": float(cache["omega_delta"][pos]),
            "omega_bragg_root_available": bool(
                cache.get(
                    "omega_bragg_root_available",
                    np.isfinite(cache["omega_bragg"]),
                )[pos]
            ),
            "sg_window_inv_A": float(cache["sg_window"][pos]),
            "sg_dSg_domega_inv_A_per_deg": float(cache["sg_domega"][pos]),
            "sg_dSg_dframe_inv_A": float(cache["sg_dframe"][pos]),
            "sg_window_raw_inv_A": float(cache["sg_window_raw"][pos]),
            "sg_window_half_frames": float(cache["sg_window_half_frames"][pos]),
            "sg_window_clipped": bool(cache["sg_window_clipped"][pos]),
            "sg_jitter_sigma_inv_A": float(cache["sg_jitter_sigma"][pos]),
            "sg_jitter_d_inv_A": float(cache["sg_jitter_d_inv"][pos]),
            "sg_jitter_frame_fraction": float(cache["sg_jitter_frame_fraction"][pos]),
            "sg_jitter_kernel_points": int(cache["sg_jitter_kernel_points"][pos]),
            "sg_profile_points_total": int(cache["sg_profile_points_total"][pos]),
            "sg_profile_weight_sum": float(cache["sg_profile_weight_sum"][pos]),
            "sg_profile_same_branch_continuation_used": bool(cache["sg_profile_same_branch_continuation_used"][pos]),
            "sg_profile_same_branch_continuation_segments": int(
                cache["sg_profile_same_branch_continuation_segments"][pos]
            ),
            "dose_delta_b_A2": float(cache["dose_delta_b"][pos]),
            "thickness_model": str(getattr(self.cfg, "thickness_model", "uniform_gradient")),
            "thickness_sigma_frac": float(getattr(self.cfg, "thickness_sigma_frac", 0.0)),
        }
        if inverse_dyn_vals:
            result["dyn_inverse"] = float(np.sum(weights * np.asarray(inverse_dyn_vals, dtype=float)))
        return row_idx, result

    def row_prediction_from_active_cache(self, pos: int):
        """Return row_prediction_from_cache for position ``pos`` of the active cache and its current U_g values."""

        return self.row_prediction_from_cache(self._active_row_plan_cache, int(pos), self._active_unique_ug)
