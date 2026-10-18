"""Construct exact signed-row plans, including PETS-owned rocking orientations."""

from __future__ import annotations

import numpy as np

from grond.bloch.basis import ranked_candidates
from grond.bloch.hamiltonian import ProfileHamiltonians
from grond.bloch.numerics import _rotation_matrix_axis_angle


class RowPlanMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def pets_exact_orientation_samples(self, row_idx):
        """Return the PETS-owned ``Rx(delta) @ A_zone`` rocking path."""

        centers = getattr(self.data, "pets_exact_a_center", None)
        semiangles = getattr(self.data, "pets_integration_semiangle_deg", None)
        if centers is None or semiangles is None:
            raise ValueError("PETS exact orientation samples require PETS geometry metadata")
        ordinal = int(round(float(self.data.zcal[row_idx]))) - 1
        if not 0 <= ordinal < len(centers):
            raise ValueError(f"PETS row {row_idx} uses zone ordinal {ordinal + 1}, outside 1..{len(centers)}")
        center = np.asarray(centers[ordinal], dtype=float)
        semiangle = float(semiangles[ordinal])
        count = max(1, int(self.cfg.n_points))
        offsets = np.asarray([0.0], dtype=float) if count == 1 else np.linspace(-semiangle, semiangle, count)
        matrices = np.asarray(
            [_rotation_matrix_axis_angle(np.asarray([1.0, 0.0, 0.0]), offset) @ center for offset in offsets],
            dtype=float,
        )
        weights = np.ones(len(offsets), dtype=float)
        accumulation = str(getattr(self.cfg, "profile_accumulation", "sum") or "sum").lower()
        if accumulation == "mean":
            weights /= float(np.sum(weights))
        elif accumulation != "sum":
            raise ValueError(f"unknown profile_accumulation={accumulation!r}")
        return matrices, offsets, weights, semiangle, ordinal

    def pets_exact_sg_for_hkls(self, hkls, orientation):
        """Return exact S_g = (k^2 - |s0 + g|^2)/(2k) in Å⁻¹ for HKL rows, with g = orientation @ h."""

        g = (np.asarray(orientation, dtype=float) @ np.asarray(hkls, dtype=float).T).T
        return (self.data.k**2 - np.sum((self.data.s0 + g) ** 2, axis=1)) / (2.0 * self.data.k)

    def pets_exact_select_common_basis(self, target_hkl, orientations, row_idx):
        """Select one scored beam union over the full PETS rocking wedge."""

        pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(self.pool_inv_d2, row_idx)
        coupling = np.abs(pool_ug) ** 2
        best_score = np.zeros(len(self.pool_hkl), dtype=float)
        minimum_abs_sg = np.full(len(self.pool_hkl), np.inf, dtype=float)
        for orientation in np.asarray(orientations, dtype=float):
            abs_sg = np.abs(self.pets_exact_sg_for_hkls(self.pool_hkl, orientation))
            minimum_abs_sg = np.minimum(minimum_abs_sg, abs_sg)
            best_score = np.maximum(best_score, coupling / (abs_sg + 1e-5))
        candidates = np.flatnonzero(minimum_abs_sg <= float(self.cfg.max_support_sg))
        basis = [(0, 0, 0), tuple(int(value) for value in target_hkl)]
        seen = set(basis)
        eligible_candidates = {tuple(int(value) for value in self.pool_hkl[index]) for index in candidates}
        eligible_candidate_count = int(len(candidates))
        full_union_count = int(len(seen | eligible_candidates))
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed:
            take = min(len(candidates), needed + 1)
            candidates = ranked_candidates(best_score, candidates, take)
            for index in candidates:
                candidate = tuple(int(value) for value in self.pool_hkl[index])
                if candidate in seen:
                    continue
                basis.append(candidate)
                seen.add(candidate)
                if len(basis) >= int(self.cfg.n_basis):
                    break
        return np.asarray(basis, dtype=int), {
            "pets_basis_mode": "one_common_scored_union_over_full_recorded_wedge",
            "pets_basis_orientation_count": int(len(orientations)),
            "pets_basis_candidate_count": eligible_candidate_count,
            "pets_basis_full_union_count": full_union_count,
            "pets_basis_ranked_candidate_count": int(len(candidates)),
            "pets_basis_cap": int(self.cfg.n_basis),
            "pets_basis_cap_truncated": bool(len(basis) < full_union_count),
        }

    def pets_exact_row_prediction_plan(self, row_idx):
        """Build one exact PETS direct-angle row plan without Sg root solving."""

        if str(getattr(self.cfg, "profile_normalization", "legacy")) != "legacy":
            raise ValueError("target_sg_equivalent_v1 describes XDS/DIALS rows, not PETS-owned exact rocking wedges")
        hkl = np.asarray(self.data.hkl[row_idx], dtype=int)
        orientations, offsets, profile_weights, semiangle, ordinal = self.pets_exact_orientation_samples(row_idx)
        center = np.asarray(self.data.pets_exact_a_center[ordinal], dtype=float)
        center_sg = float(self.pets_exact_sg_for_hkls(np.asarray([hkl]), center)[0])
        derivative_step = 1e-4
        plus = _rotation_matrix_axis_angle(np.asarray([1.0, 0.0, 0.0]), derivative_step) @ center
        minus = _rotation_matrix_axis_angle(np.asarray([1.0, 0.0, 0.0]), -derivative_step) @ center
        sg_derivative = float(
            (
                self.pets_exact_sg_for_hkls(np.asarray([hkl]), plus)[0]
                - self.pets_exact_sg_for_hkls(np.asarray([hkl]), minus)[0]
            )
            / (2.0 * derivative_step)
        )
        sg_window = 2.0 * abs(sg_derivative) * semiangle
        basis, basis_meta = self.pets_exact_select_common_basis(hkl, orientations, row_idx)
        if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(basis[1], hkl):
            raise RuntimeError("PETS frozen basis must start with direct beam then exact signed target HKL")
        dynamic_static = self.hamiltonian_static_part(basis, absorb=True, row_idx=row_idx)
        born_static = (
            self.hamiltonian_static_part(basis, absorb=False, row_idx=row_idx)
            if self.absorption_enabled()
            else dynamic_static
        )
        h10_born = born_static[1, 0]
        sg_profiles = np.empty((len(orientations), len(basis)), dtype=dynamic_static.real.dtype)
        born_values = []
        born_deltas = []
        for profile_index, orientation in enumerate(orientations):
            sg_basis = self.pets_exact_sg_for_hkls(basis, orientation)
            sg_profiles[profile_index] = sg_basis
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            born_deltas.append(delta)
            born_values.append(self.born_thickness_average(h10_born, delta))
        return {
            "_dyn_matrices": ProfileHamiltonians(dynamic_static, sg_profiles),
            "_profile_weights": np.asarray(profile_weights, dtype=float),
            "_born_vals": np.asarray(born_values, dtype=float),
            "_born_h10_abs2": float(abs(h10_born) ** 2),
            "_born_deltas": np.asarray(born_deltas, dtype=float),
            "_sg_resid": np.asarray([], dtype=float),
            "basis_size": int(len(basis)),
            "support_hkls": [[int(value) for value in row] for row in basis.tolist()],
            "center_sg_inv_A": center_sg,
            "omega_center_deg": float("nan"),
            "omega_bragg_deg": float("nan"),
            "omega_bragg_minus_center_deg": float("nan"),
            "sg_window_inv_A": sg_window,
            "dose_delta_b_A2": float(self.row_delta_b(row_idx)),
            "thickness_model": str(getattr(self.cfg, "thickness_model", "uniform_gradient")),
            "thickness_sigma_frac": float(getattr(self.cfg, "thickness_sigma_frac", 0.0)),
            "sg_profile_coordinate": "pets_exact_direct_angle",
            "sg_profile_quadrature": "uniform_linspace",
            "sg_profile_accumulation": str(getattr(self.cfg, "profile_accumulation", "sum")),
            "sg_profile_angle_half_width_deg": semiangle,
            "sg_profile_points_total": int(len(orientations)),
            "sg_profile_weight_sum": float(np.sum(profile_weights)),
            "sg_dSg_domega_inv_A_per_deg": sg_derivative,
            "sg_dSg_dframe_inv_A": sg_derivative * float(self.data.oscillation_range),
            "sg_window_raw_inv_A": sg_window,
            "sg_window_half_frames": semiangle / max(abs(float(self.data.oscillation_range)), 1e-30),
            "sg_window_clipped": False,
            "sg_jitter_sigma_inv_A": 0.0,
            "sg_jitter_d_inv_A": float("nan"),
            "sg_jitter_frame_fraction": float("nan"),
            "sg_jitter_kernel_points": 1,
            "sg_profile_same_branch_continuation_used": False,
            "sg_profile_same_branch_continuation_segments": 1,
            "pets_zone_ordinal": int(ordinal + 1),
            "pets_rocking_offsets_deg": [float(value) for value in offsets],
            **basis_meta,
        }

    def row_prediction_plan(self, row_idx):
        """Build one row's Bloch matrices and metadata before eigensolver evaluation."""

        if getattr(self.data, "pets_exact_a_center", None) is not None:
            return self.pets_exact_row_prediction_plan(row_idx)

        hkl = self.data.hkl[row_idx]
        zcal = self.data.zcal[row_idx]
        omega_center = self.omega_from_z(zcal)
        center_sg = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega_center)[0])
        omega_bragg, omega_bragg_root_available = self.row_diagnostic_omega_bragg(hkl, omega_center)
        sg_window_i, sg_width_meta = self.row_sg_window(row_idx, hkl, omega_center)
        (
            solved_omegas,
            mosaic_tilts,
            profile_weights,
            sg_resid,
            jitter_meta,
            mosaicity_meta,
        ) = self.row_profile_orientation_points(row_idx, hkl, omega_center, center_sg, sg_window_i)
        basis, cone_basis_meta = self.select_basis_for_profile(
            hkl,
            omega_center,
            sg_window_i,
            solved_omegas,
            mosaic_tilts,
            row_idx=row_idx,
        )
        if not np.array_equal(basis[0], np.asarray([0, 0, 0], dtype=int)) or not np.array_equal(
            basis[1], np.asarray(hkl, dtype=int)
        ):
            raise RuntimeError(
                "Frozen kernel basis contract violated: basis must start with direct beam then exact signed target hkl."
            )
        H_dyn_static = self.hamiltonian_static_part(basis, absorb=True, row_idx=row_idx)
        if self.absorption_enabled():
            H_born_static = self.hamiltonian_static_part(basis, absorb=False, row_idx=row_idx)
        else:
            H_born_static = H_dyn_static
        h10_born = H_born_static[1, 0]
        sg_profiles = np.empty((len(solved_omegas), len(basis)), dtype=H_dyn_static.real.dtype)
        born_vals, born_deltas = [], []
        for profile_index, (omega, (along_deg, transverse_deg)) in enumerate(zip(solved_omegas, mosaic_tilts, strict=True)):
            sg_basis = self.sg_for_hkls_at_orientation(basis, omega, along_deg, transverse_deg)
            sg_profiles[profile_index] = sg_basis
            delta = float(np.real(sg_basis[1] - sg_basis[0]))
            born_deltas.append(delta)
            born_vals.append(self.born_thickness_average(h10_born, delta))
        return {
            "_dyn_matrices": ProfileHamiltonians(H_dyn_static, sg_profiles),
            "_profile_weights": np.asarray(profile_weights, dtype=float),
            "_born_vals": np.asarray(born_vals, dtype=float),
            "_born_h10_abs2": float(abs(h10_born) ** 2),
            "_born_deltas": np.asarray(born_deltas, dtype=float),
            "_sg_resid": np.asarray(sg_resid, dtype=float),
            "basis_size": int(len(basis)),
            "support_hkls": [[int(v) for v in row] for row in basis.tolist()],
            "center_sg_inv_A": center_sg,
            "omega_center_deg": float(omega_center),
            "omega_bragg_deg": float(omega_bragg),
            "omega_bragg_minus_center_deg": float(omega_bragg - omega_center),
            "omega_bragg_root_available": bool(omega_bragg_root_available),
            "sg_window_inv_A": float(sg_window_i),
            "dose_delta_b_A2": float(self.row_delta_b(row_idx)),
            "thickness_model": str(getattr(self.cfg, "thickness_model", "uniform_gradient")),
            "thickness_sigma_frac": float(getattr(self.cfg, "thickness_sigma_frac", 0.0)),
            **sg_width_meta,
            **jitter_meta,
            **mosaicity_meta,
            **cone_basis_meta,
        }

    def finish_row_prediction_plan(self, plan, dyn_vals):
        """Return a row plan's public fields with the profile-weighted ``dyn`` and ``born`` intensities added.

        ``dyn_vals`` holds one dynamical target intensity per profile point. Keys starting with ``_`` are working arrays
        and are dropped; ``max_sg_solve_residual`` is the largest |S_g| root-solve residual.
        """

        profile_weights = np.asarray(plan["_profile_weights"], dtype=float)
        born_vals = np.asarray(plan["_born_vals"], dtype=float)
        sg_resid = np.asarray(plan["_sg_resid"], dtype=float)
        out = {k: v for k, v in plan.items() if not k.startswith("_")}
        out.update(
            {
                "dyn": float(np.sum(profile_weights * np.asarray(dyn_vals, dtype=float))),
                "born": float(np.sum(profile_weights * born_vals)),
                "max_sg_solve_residual": float(np.max(np.abs(sg_resid))) if len(sg_resid) else 0.0,
            }
        )
        return out

    def row_prediction(self, row_idx):
        """Predict one row: its plan record with profile-weighted, thickness-averaged ``dyn`` and ``born``.

        With paired_inverse_hand the same eigensystems also give the inverse-hand intensity as ``dyn_inverse``.
        """

        plan = self.row_prediction_plan(row_idx)
        if bool(getattr(self.cfg, "paired_inverse_hand", False)):
            pairs = [self.propagate_direct_to_target_thickness_average_pair(H) for H in plan["_dyn_matrices"]]
            out = self.finish_row_prediction_plan(plan, [pair[0] for pair in pairs])
            weights = np.asarray(plan["_profile_weights"], dtype=float)
            out["dyn_inverse"] = float(np.sum(weights * np.asarray([pair[1] for pair in pairs], dtype=float)))
            return out
        dyn_vals = [self.propagate_direct_to_target_thickness_average(H) for H in plan["_dyn_matrices"]]
        return self.finish_row_prediction_plan(plan, dyn_vals)
