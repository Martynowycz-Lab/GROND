"""Direct-plus-target beam support and profile/cone support unions.

The support score is a selection heuristic, not a guaranteed propagation-error
bound. Increasing a cap alone does not recover beams outside the source pool."""

from __future__ import annotations

import math

import numpy as np


class BasisMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def select_basis(self, target_hkl, omega_bragg, row_idx=None):
        sg_pool = self.sg_for_hkls(self.pool_hkl, omega_bragg)
        if row_idx is None:
            pool_ug = self.pool_ug
        else:
            pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(self.pool_inv_d2, row_idx)
        score = np.abs(pool_ug) ** 2 / (np.abs(sg_pool) + 1e-5)
        ok = np.abs(sg_pool) <= float(self.cfg.max_support_sg)
        basis = [(0, 0, 0), tuple(int(v) for v in target_hkl)]
        seen = set(basis)
        candidates = np.flatnonzero(ok)
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed == 0:
            return np.asarray(basis, dtype=int)
        take = min(len(candidates), needed + 1)
        if take < len(candidates):
            local = np.argpartition(score[candidates], -take)[-take:]
            candidates = candidates[local]
        candidates = candidates[np.lexsort((candidates, -score[candidates]))]
        for idx in candidates:
            h = tuple(int(v) for v in self.pool_hkl[idx])
            if h in seen:
                continue
            basis.append(h)
            seen.add(h)
            if len(basis) >= int(self.cfg.n_basis):
                break
        return np.asarray(basis, dtype=int)

    def cone_support_orientations(self, hkl, omega_center, sg_window_i):
        """Return a fixed orientation envelope for common-basis selection.

        The envelope depends on the declared maximum cone width, not the
        candidate width or evaluation quadrature.  Consequently sigma=0 and
        every tested finite sigma use exactly the same per-row beam basis.
        """

        sigma = float(getattr(self.cfg, "mosaicity_support_sigma_deg", 0.0))
        n = max(2, int(getattr(self.cfg, "mosaicity_support_points", 5)))
        width = float(getattr(self.cfg, "mosaicity_support_width_sigma", 4.0))
        derivative = abs(float(self.sg_derivative_per_deg(hkl, omega_center)))
        if not math.isfinite(derivative) or derivative <= 1e-12:
            raise ValueError("common-cone basis requires a finite nonzero dSg/domega")
        raw_half_width = 0.5 * float(sg_window_i) / derivative
        maximum = float(getattr(self.cfg, "profile_angle_max_half_width_deg", 2.0))
        half_width = min(raw_half_width, maximum)
        omega_points = np.linspace(
            float(omega_center) - half_width,
            float(omega_center) + half_width,
            n,
        )
        cone_axis = np.asarray([0.0], dtype=float) if sigma <= 0.0 else np.linspace(-width * sigma, width * sigma, n)
        cone_points = np.stack(np.meshgrid(cone_axis, cone_axis, indexing="ij"), axis=-1).reshape(-1, 2)
        return (
            omega_points,
            cone_points,
            {
                "mosaicity_basis_mode": "common_maximum_cone",
                "mosaicity_support_sigma_deg": sigma,
                "mosaicity_support_points_per_axis": int(n),
                "mosaicity_support_width_sigma": width,
                "mosaicity_support_orientation_count": int(len(omega_points) * len(cone_points)),
                "mosaicity_support_profile_half_width_deg": float(half_width),
            },
        )

    def select_basis_common_cone(self, target_hkl, omega_center, sg_window_i, row_idx=None):
        """Select one beam basis over the widest declared 2D mosaic cone."""

        if str(getattr(self.cfg, "mosaicity_mode", "none")).lower() != "gaussian_cone":
            return self.select_basis(target_hkl, omega_center, row_idx=row_idx), {}
        omega_points, cone_points, meta = self.cone_support_orientations(target_hkl, omega_center, sg_window_i)
        if row_idx is None:
            pool_ug = self.pool_ug
        else:
            pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(self.pool_inv_d2, row_idx)
        coupling = np.abs(pool_ug) ** 2
        best_score = np.zeros(len(self.pool_hkl), dtype=float)
        minimum_abs_sg = np.full(len(self.pool_hkl), np.inf, dtype=float)
        for omega in omega_points:
            for along_deg, transverse_deg in cone_points:
                sg_pool = self.sg_for_hkls_at_orientation(
                    self.pool_hkl,
                    float(omega),
                    float(along_deg),
                    float(transverse_deg),
                )
                abs_sg = np.abs(sg_pool)
                minimum_abs_sg = np.minimum(minimum_abs_sg, abs_sg)
                best_score = np.maximum(best_score, coupling / (abs_sg + 1e-5))
        ok = minimum_abs_sg <= float(self.cfg.max_support_sg)
        basis = [(0, 0, 0), tuple(int(v) for v in target_hkl)]
        seen = set(basis)
        candidates = np.flatnonzero(ok)
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed == 0:
            return np.asarray(basis, dtype=int), meta
        take = min(len(candidates), needed + 1)
        if take < len(candidates):
            local = np.argpartition(best_score[candidates], -take)[-take:]
            candidates = candidates[local]
        candidates = candidates[np.lexsort((candidates, -best_score[candidates]))]
        for idx in candidates:
            h = tuple(int(v) for v in self.pool_hkl[idx])
            if h in seen:
                continue
            basis.append(h)
            seen.add(h)
            if len(basis) >= int(self.cfg.n_basis):
                break
        return np.asarray(basis, dtype=int), meta

    def select_basis_for_profile(
        self,
        target_hkl,
        omega_center,
        sg_window_i,
        solved_omegas,
        mosaic_tilts,
        row_idx=None,
    ):
        """Select one frozen basis over the declared rocking-profile support.

        ``center`` preserves the historical row-center selection. The
        high-fidelity ``common_profile`` mode evaluates a fixed support grid
        spanning the complete solved rocking interval and retains every beam
        that enters the Sg support at any support orientation. The support
        grid size is independent of propagation quadrature Q.

        A two-axis Gaussian cone already owns a wider, candidate-invariant
        support construction and continues to use that implementation.
        """

        if str(getattr(self.cfg, "mosaicity_mode", "none")).lower() == "gaussian_cone":
            return self.select_basis_common_cone(
                target_hkl,
                omega_center,
                sg_window_i,
                row_idx=row_idx,
            )
        mode = str(getattr(self.cfg, "profile_basis_mode", "center") or "center").lower()
        if mode == "center":
            return self.select_basis(
                target_hkl,
                omega_center,
                row_idx=row_idx,
            ), {
                "profile_basis_mode": "center",
                "profile_basis_orientation_count": 1,
            }
        if mode != "common_profile":
            raise ValueError(f"unknown profile_basis_mode={mode!r}")

        omegas = np.asarray(solved_omegas, dtype=float).reshape(-1)
        tilts = np.asarray(mosaic_tilts, dtype=float).reshape(-1, 2)
        if len(omegas) == 0 or len(omegas) != len(tilts):
            raise ValueError("common-profile basis requires matching nonempty rocking angles and tilts")
        support_count = max(
            2,
            int(getattr(self.cfg, "mosaicity_support_points", 5)),
        )
        if float(np.max(omegas) - np.min(omegas)) <= 1e-12:
            support_omegas = np.asarray([float(omegas[0])], dtype=float)
        else:
            support_omegas = np.linspace(
                float(np.min(omegas)),
                float(np.max(omegas)),
                support_count,
            )
        support_tilts = np.zeros((len(support_omegas), 2), dtype=float)
        if row_idx is None:
            pool_ug = self.pool_ug
        else:
            pool_ug = self.pool_ug * self.dose_attenuation_for_inv_d2(
                self.pool_inv_d2,
                row_idx,
            )
        coupling = np.abs(pool_ug) ** 2
        best_score = np.zeros(len(self.pool_hkl), dtype=float)
        minimum_abs_sg = np.full(len(self.pool_hkl), np.inf, dtype=float)
        for omega, (along_deg, transverse_deg) in zip(
            support_omegas,
            support_tilts,
            strict=True,
        ):
            sg_pool = self.sg_for_hkls_at_orientation(
                self.pool_hkl,
                float(omega),
                float(along_deg),
                float(transverse_deg),
            )
            abs_sg = np.abs(sg_pool)
            minimum_abs_sg = np.minimum(minimum_abs_sg, abs_sg)
            best_score = np.maximum(
                best_score,
                coupling / (abs_sg + 1e-5),
            )
        candidates = np.flatnonzero(minimum_abs_sg <= float(self.cfg.max_support_sg))
        basis = [(0, 0, 0), tuple(int(value) for value in target_hkl)]
        seen = set(basis)
        eligible = {tuple(int(value) for value in self.pool_hkl[index]) for index in candidates}
        full_union_count = int(len(seen | eligible))
        needed = max(0, int(self.cfg.n_basis) - len(basis))
        if needed:
            take = min(len(candidates), needed + 1)
            if take < len(candidates):
                local = np.argpartition(best_score[candidates], -take)[-take:]
                candidates = candidates[local]
            candidates = candidates[np.lexsort((candidates, -best_score[candidates]))]
            for index in candidates:
                candidate = tuple(int(value) for value in self.pool_hkl[index])
                if candidate in seen:
                    continue
                basis.append(candidate)
                seen.add(candidate)
                if len(basis) >= int(self.cfg.n_basis):
                    break
        return np.asarray(basis, dtype=int), {
            "profile_basis_mode": "common_profile",
            "profile_basis_orientation_count": int(len(support_omegas)),
            "profile_basis_omega_min_deg": float(np.min(support_omegas)),
            "profile_basis_omega_max_deg": float(np.max(support_omegas)),
            "profile_basis_candidate_count": int(len(eligible)),
            "profile_basis_full_union_count": full_union_count,
            "profile_basis_cap": int(self.cfg.n_basis),
            "profile_basis_cap_truncated": bool(len(basis) < full_union_count),
        }
