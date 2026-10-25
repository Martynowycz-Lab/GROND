"""Construct exact signed-row plans."""

from __future__ import annotations

import numpy as np

from grond.bloch.hamiltonian import ProfileHamiltonians


class RowPlanMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def row_prediction_plan(self, row_idx):
        """Build one row's Bloch matrices and metadata before eigensolver evaluation."""

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
        basis = self.select_basis(hkl, omega_center, row_idx=row_idx)
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
            **sg_width_meta,
            **jitter_meta,
            **mosaicity_meta,
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
