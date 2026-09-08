"""Rocking quadrature and incoherent mosaic-orientation distributions.

Weights retain the existing sum/mean and relative angular-Jacobian convention.
They are not silently promoted to absolute detector exposure weights."""

from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

from grond.bloch.errors import MissingObservedCentroidError


class ProfileMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def row_sg_window(self, row_idx, hkl, omega_bragg):
        """Return an auditable per-row target-Sg window.

        The default remains the audited fixed window.  Adaptive modes are
        deliberately simple XDS metadata probes, not fitted scale parameters.
        They alter which part of the synthetic rocking curve is averaged.
        """
        mode = str(self.cfg.sg_window_mode or "fixed").lower()
        base = float(self.cfg.sg_window)
        deriv_deg = abs(float(self.sg_derivative_per_deg(hkl, omega_bragg)))
        deriv_frame = deriv_deg * abs(float(self.data.oscillation_range))
        dz = abs(float(self.data.zobs[row_idx] - self.data.zcal[row_idx]))
        if mode in {"zobs_residual", "zobs_plus_floor"} and not math.isfinite(dz):
            # A missing centroid is not a measured zero residual. The normal
            # fixed window needs no centroid and remains available explicitly.
            raise MissingObservedCentroidError(
                f"sg_window_mode={mode} requires an observed centroid for row {row_idx}, "
                f"HKL {tuple(int(v) for v in hkl)}; ZOBS/ZCAL is unavailable. "
                "Use sg_window_mode=fixed for integrations without observed centroids."
            )
        peak_frac = max(float(self.data.peak[row_idx]) / 100.0, 1e-6)
        rlp_ratio = max(float(self.data.rlp[row_idx]), 1e-12) / max(self.median_rlp, 1e-12)
        half_frames = float(self.cfg.sg_window_frame_floor)

        if mode == "fixed":
            raw = base
            half_frames_used = float("nan")
            source = "global"
        elif mode == "zobs_residual":
            half_frames_used = max(half_frames, dz)
            raw = 2.0 * deriv_frame * half_frames_used
            source = "2*abs(dSg/dframe)*max(frame_floor,abs(ZOBS-ZCAL))"
        elif mode == "zobs_plus_floor":
            half_frames_used = half_frames + dz
            raw = 2.0 * deriv_frame * half_frames_used
            source = "2*abs(dSg/dframe)*(frame_floor+abs(ZOBS-ZCAL))"
        elif mode == "rlp_peak":
            half_frames_used = float("nan")
            raw = base * math.sqrt(max(rlp_ratio, 1e-12) / peak_frac)
            source = "sg_window*sqrt((RLP/median_RLP)/(PEAK/100))"
        elif mode == "dials_bbox":
            z0 = float(self.data.alf0[row_idx])
            z1 = float(self.data.bet0[row_idx])
            if np.isfinite(z0) and np.isfinite(z1) and z1 > z0:
                half_frames_used = max(half_frames, 0.5 * (z1 - z0))
                source = "2*abs(dSg/dframe)*max(frame_floor,0.5*(DIALS bbox z1-z0))"
            else:
                half_frames_used = half_frames
                source = "2*abs(dSg/dframe)*frame_floor; DIALS bbox z-range unavailable"
            raw = 2.0 * deriv_frame * half_frames_used
        else:
            raise ValueError(f"unknown sg_window_mode={self.cfg.sg_window_mode!r}")

        scaled = float(raw) * float(self.cfg.sg_window_scale)
        clipped = float(np.clip(scaled, float(self.cfg.sg_window_min), float(self.cfg.sg_window_max)))
        return clipped, {
            "sg_window_mode": mode,
            "sg_window_raw_inv_A": float(raw),
            "sg_window_source": source,
            "sg_dSg_domega_inv_A_per_deg": float(deriv_deg),
            "sg_dSg_dframe_inv_A": float(deriv_frame),
            "sg_window_half_frames": float(half_frames_used),
            "sg_window_zobs_minus_zcal_frame": float(self.data.zobs[row_idx] - self.data.zcal[row_idx]),
            "sg_window_rlp_ratio": float(rlp_ratio),
            "sg_window_peak_fraction": float(peak_frac),
            "sg_window_clipped": bool(clipped != scaled),
        }

    def row_sg_jitter_sigma(self, row_idx, hkl):
        """Return the target-Sg jitter sigma for an observation.

        This is an orientation-distribution diagnostic, not a hidden scale.  It
        broadens the target-Sg quadrature by averaging nearby offsets and keeps
        the ordinary uniform-Sg profile as the base distribution.
        """
        mode = str(getattr(self.cfg, "sg_jitter_mode", "none") or "none").lower()
        if mode == "none":
            return 0.0, {
                "sg_jitter_mode": mode,
                "sg_jitter_sigma_inv_A": 0.0,
                "sg_jitter_d_inv_A": float("nan"),
                "sg_jitter_frame_fraction": float("nan"),
            }
        d = float(self.data.cell.calculate_d(tuple(int(v) for v in hkl)))
        d_inv = 1.0 / max(d, 1e-12)
        frame_frac = (float(self.data.zcal[row_idx]) - self.zcal_min) / self.zcal_span
        raw = (
            float(getattr(self.cfg, "sg_jitter_sigma0", 0.0))
            + float(getattr(self.cfg, "sg_jitter_sigma_d_inv", 0.0)) * d_inv
            + float(getattr(self.cfg, "sg_jitter_sigma_frame", 0.0)) * frame_frac
        )
        sigma = max(0.0, float(raw))
        return sigma, {
            "sg_jitter_mode": mode,
            "sg_jitter_sigma_inv_A": float(sigma),
            "sg_jitter_d_inv_A": float(d_inv),
            "sg_jitter_frame_fraction": float(frame_frac),
        }

    def row_profile_sg_points(self, row_idx, hkl, sg_window_i):
        """Build weighted target-Sg profile samples, optionally convolved by jitter."""
        n_base = max(1, int(self.cfg.n_points))
        half_width = 0.5 * float(sg_window_i)
        rule = str(getattr(self.cfg, "profile_quadrature", "uniform") or "uniform").lower()
        accumulation = str(getattr(self.cfg, "profile_accumulation", "sum") or "sum").lower()
        if accumulation not in {"sum", "mean"}:
            raise ValueError(f"unknown profile_accumulation={accumulation!r}; expected sum or mean")
        if n_base == 1:
            base = np.asarray([0.0], dtype=float)
            base_w = np.asarray([1.0], dtype=float)
        elif rule == "uniform":
            base = np.linspace(-half_width, half_width, n_base)
            base_w = np.ones(n_base, dtype=float)
        elif rule == "trapezoid":
            base = np.linspace(-half_width, half_width, n_base)
            base_w = np.ones(n_base, dtype=float)
            base_w[0] = 0.5
            base_w[-1] = 0.5
            base_w *= n_base / max(float(np.sum(base_w)), 1e-30)
        elif rule in {"gauss_legendre", "legendre"}:
            nodes, weights = leggauss(n_base)
            base = half_width * nodes
            base_w = np.asarray(weights, dtype=float)
            base_w *= n_base / max(float(np.sum(base_w)), 1e-30)
        else:
            raise ValueError(f"unknown profile_quadrature={rule!r}; expected uniform, trapezoid, or gauss_legendre")
        if accumulation == "mean":
            base_w = base_w / max(float(np.sum(base_w)), 1e-30)
        sigma, meta = self.row_sg_jitter_sigma(row_idx, hkl)
        meta["sg_profile_same_branch_continuation_used"] = False
        meta["sg_profile_same_branch_continuation_segments"] = 1
        mode = str(meta["sg_jitter_mode"]).lower()
        if mode == "none" or sigma <= 0.0:
            meta.update(
                {
                    "sg_profile_quadrature": rule,
                    "sg_profile_accumulation": accumulation,
                    "sg_profile_weight_sum": float(np.sum(base_w)),
                    "sg_jitter_kernel_points": 1,
                    "sg_profile_points_total": int(len(base)),
                    "sg_jitter_offset_abs_p95_inv_A": 0.0,
                }
            )
            return base, base_w, meta

        n_kernel = max(1, int(getattr(self.cfg, "sg_jitter_points", 5)))
        if mode == "gaussian":
            if n_kernel == 1:
                offsets = np.asarray([0.0], dtype=float)
                kernel_w = np.asarray([1.0], dtype=float)
            else:
                nodes, weights = hermgauss(n_kernel)
                offsets = math.sqrt(2.0) * sigma * nodes
                kernel_w = weights / math.sqrt(math.pi)
        elif mode == "student_t":
            if n_kernel == 1:
                offsets = np.asarray([0.0], dtype=float)
                kernel_w = np.asarray([1.0], dtype=float)
            else:
                width = max(float(getattr(self.cfg, "sg_jitter_width_factor", 3.0)), 1e-9) * sigma
                offsets = np.linspace(-width, width, n_kernel)
                nu = max(float(getattr(self.cfg, "sg_jitter_student_df", 4.0)), 1.01)
                x = offsets / max(sigma, 1e-12)
                kernel_w = (1.0 + (x * x) / nu) ** (-(nu + 1.0) / 2.0)
        else:
            raise ValueError(f"unknown sg_jitter_mode={getattr(self.cfg, 'sg_jitter_mode', None)!r}")

        kernel_w = np.asarray(kernel_w, dtype=float)
        kernel_w /= max(float(np.sum(kernel_w)), 1e-30)
        requested = (base[:, None] + offsets[None, :]).reshape(-1)
        weights = (base_w[:, None] * kernel_w[None, :]).reshape(-1)
        abs_offsets = np.abs(offsets)
        meta.update(
            {
                "sg_profile_quadrature": rule,
                "sg_profile_accumulation": accumulation,
                "sg_profile_weight_sum": float(np.sum(weights)),
                "sg_jitter_kernel_points": int(len(offsets)),
                "sg_profile_points_total": int(len(requested)),
                "sg_jitter_offset_abs_p95_inv_A": float(np.quantile(abs_offsets, 0.95)) if len(abs_offsets) else 0.0,
            }
        )
        return requested, weights, meta

    def row_mosaicity_angle_points(self):
        """Return a normalized projected angular mosaic distribution.

        These offsets are applied directly to the crystal rotation angle after
        each base target-Sg point has been solved.  Unlike the legacy Sg-jitter
        proxy, every finite angular sample is geometrically defined and no
        additional Ewald-root search is required.
        """

        mode = str(getattr(self.cfg, "mosaicity_mode", "none") or "none").lower()
        sigma = float(getattr(self.cfg, "mosaicity_sigma_deg", 0.0))
        if mode == "none" or sigma <= 0.0:
            return (
                np.asarray([0.0], dtype=float),
                np.asarray([1.0], dtype=float),
                {
                    "mosaicity_mode": "none" if sigma <= 0.0 else mode,
                    "mosaicity_sigma_deg": 0.0 if sigma <= 0.0 else sigma,
                    "mosaicity_kernel_points": 1,
                    "mosaicity_offset_abs_p95_deg": 0.0,
                },
            )
        if mode == "gaussian_cone":
            return (
                np.asarray([0.0], dtype=float),
                np.asarray([1.0], dtype=float),
                {
                    "mosaicity_mode": mode,
                    "mosaicity_sigma_deg": sigma,
                    "mosaicity_kernel_points": 1,
                    "mosaicity_offset_abs_p95_deg": 0.0,
                },
            )
        if mode != "gaussian_omega":
            raise ValueError(f"unknown mosaicity_mode={mode!r}")
        n = max(1, int(getattr(self.cfg, "mosaicity_points", 3)))
        if n == 1:
            offsets = np.asarray([0.0], dtype=float)
            weights = np.asarray([1.0], dtype=float)
        else:
            nodes, raw_weights = hermgauss(n)
            offsets = math.sqrt(2.0) * sigma * np.asarray(nodes, dtype=float)
            weights = np.asarray(raw_weights, dtype=float) / math.sqrt(math.pi)
            weights /= max(float(np.sum(weights)), 1e-30)
        return (
            offsets,
            weights,
            {
                "mosaicity_mode": mode,
                "mosaicity_sigma_deg": sigma,
                "mosaicity_kernel_points": int(len(offsets)),
                "mosaicity_offset_abs_p95_deg": float(np.quantile(np.abs(offsets), 0.95)),
            },
        )

    def row_mosaicity_cone_points(self):
        """Return tensor Gauss-Hermite points for a 2D Gaussian cone.

        The two rotations are incoherent mosaic orientations.  Intensities,
        not amplitudes, are averaged over this normalized distribution.
        """

        mode = str(getattr(self.cfg, "mosaicity_mode", "none") or "none").lower()
        sigma = float(getattr(self.cfg, "mosaicity_sigma_deg", 0.0))
        if mode != "gaussian_cone" or sigma <= 0.0:
            return (
                np.zeros((1, 2), dtype=float),
                np.asarray([1.0], dtype=float),
                {
                    "mosaicity_cone_points": 1,
                    "mosaicity_cone_quadrature": "zero_width",
                    "mosaicity_cone_weight_sum": 1.0,
                },
            )
        n = max(1, int(getattr(self.cfg, "mosaicity_points", 3)))
        if n == 1:
            offsets = np.asarray([0.0], dtype=float)
            weights = np.asarray([1.0], dtype=float)
        else:
            nodes, raw_weights = hermgauss(n)
            offsets = math.sqrt(2.0) * sigma * np.asarray(nodes, dtype=float)
            weights = np.asarray(raw_weights, dtype=float) / math.sqrt(math.pi)
            weights /= max(float(np.sum(weights)), 1e-30)
        support_sigma = float(getattr(self.cfg, "mosaicity_support_sigma_deg", 0.0))
        support_width = float(getattr(self.cfg, "mosaicity_support_width_sigma", 4.0))
        if float(np.max(np.abs(offsets))) > support_sigma * support_width + 1e-12:
            raise ValueError(
                "mosaicity quadrature extends beyond the declared common-cone "
                "basis envelope; increase mosaicity_support_sigma_deg or "
                "mosaicity_support_width_sigma"
            )
        points = np.stack(np.meshgrid(offsets, offsets, indexing="ij"), axis=-1).reshape(-1, 2)
        pair_weights = np.outer(weights, weights).reshape(-1)
        pair_weights /= max(float(np.sum(pair_weights)), 1e-30)
        return (
            points,
            pair_weights,
            {
                "mosaicity_cone_points": int(len(points)),
                "mosaicity_cone_quadrature": f"tensor_gauss_hermite_{n}x{n}",
                "mosaicity_cone_weight_sum": float(np.sum(pair_weights)),
                "mosaicity_cone_offset_radius_p95_deg": float(np.quantile(np.linalg.norm(points, axis=1), 0.95)),
            },
        )

    def row_profile_orientation_points(self, row_idx, hkl, omega_center, center_sg, sg_window_i):
        """Return explicit scan angles and two-axis mosaic orientations."""

        (
            base_omegas,
            base_weights,
            sg_resid,
            profile_meta,
            mosaicity_meta,
        ) = self.row_profile_omega_points(row_idx, hkl, omega_center, center_sg, sg_window_i)
        cone_points, cone_weights, cone_meta = self.row_mosaicity_cone_points()
        count_base = len(base_omegas)
        count_cone = len(cone_points)
        omegas = np.repeat(np.asarray(base_omegas, dtype=float), count_cone)
        tilts = np.tile(np.asarray(cone_points, dtype=float), (count_base, 1))
        weights = (
            np.asarray(base_weights, dtype=float)[:, None] * np.asarray(cone_weights, dtype=float)[None, :]
        ).reshape(-1)
        profile_meta = dict(profile_meta)
        profile_meta.update(
            {
                "sg_profile_points_total": int(len(weights)),
                "sg_profile_weight_sum": float(np.sum(weights)),
            }
        )
        mosaicity_meta = {**mosaicity_meta, **cone_meta}
        if str(getattr(self.cfg, "mosaicity_mode", "none")).lower() == "gaussian_cone":
            mosaicity_meta["mosaicity_kernel_points"] = int(count_cone)
            mosaicity_meta["mosaicity_offset_abs_p95_deg"] = float(
                cone_meta.get("mosaicity_cone_offset_radius_p95_deg", 0.0)
            )
        return omegas, tilts, weights, sg_resid, profile_meta, mosaicity_meta

    def row_profile_omega_points(self, row_idx, hkl, omega_center, center_sg, sg_window_i):
        """Build solved base-profile angles convolved with direct angular mosaicity."""

        coordinate = str(getattr(self.cfg, "profile_coordinate", "target_sg") or "target_sg").lower()
        if coordinate == "direct_angle":
            jitter_sigma, jitter_meta = self.row_sg_jitter_sigma(row_idx, hkl)
            if jitter_sigma > 0.0:
                raise ValueError("direct-angle profile quadrature cannot be combined with target-Sg jitter")
            n_base = max(1, int(self.cfg.n_points))
            derivative = abs(float(self.sg_derivative_per_deg(hkl, omega_center)))
            if not math.isfinite(derivative) or derivative <= 1e-12:
                raise ValueError("direct-angle profile requires a finite nonzero dSg/domega")
            raw_half_width_deg = 0.5 * float(sg_window_i) / derivative
            maximum = float(getattr(self.cfg, "profile_angle_max_half_width_deg", 2.0))
            half_width_deg = min(raw_half_width_deg, maximum)
            rule = str(getattr(self.cfg, "profile_quadrature", "uniform") or "uniform").lower()
            accumulation = str(getattr(self.cfg, "profile_accumulation", "sum") or "sum").lower()
            if n_base == 1:
                base_offsets = np.asarray([0.0], dtype=float)
                base_weights = np.asarray([1.0], dtype=float)
            elif rule == "uniform":
                base_offsets = np.linspace(-half_width_deg, half_width_deg, n_base)
                base_weights = np.ones(n_base, dtype=float)
            elif rule == "trapezoid":
                base_offsets = np.linspace(-half_width_deg, half_width_deg, n_base)
                base_weights = np.ones(n_base, dtype=float)
                base_weights[[0, -1]] = 0.5
                base_weights *= n_base / max(float(np.sum(base_weights)), 1e-30)
            elif rule in {"gauss_legendre", "legendre"}:
                nodes, weights = leggauss(n_base)
                base_offsets = half_width_deg * np.asarray(nodes, dtype=float)
                base_weights = np.asarray(weights, dtype=float)
                base_weights *= n_base / max(float(np.sum(base_weights)), 1e-30)
            else:
                raise ValueError(f"unknown profile_quadrature={rule!r}")
            if accumulation == "mean":
                base_weights /= max(float(np.sum(base_weights)), 1e-30)
            elif accumulation != "sum":
                raise ValueError(f"unknown profile_accumulation={accumulation!r}")
            angular_offsets, angular_weights, mosaicity_meta = self.row_mosaicity_angle_points()
            sample_omegas = (
                float(omega_center) + base_offsets[:, None] + np.asarray(angular_offsets, dtype=float)[None, :]
            ).reshape(-1)
            profile_weights = (base_weights[:, None] * np.asarray(angular_weights, dtype=float)[None, :]).reshape(-1)
            jitter_meta.update(
                {
                    "sg_profile_coordinate": "direct_angle",
                    "sg_profile_quadrature": rule,
                    "sg_profile_accumulation": accumulation,
                    "sg_profile_angle_jacobian": False,
                    "sg_profile_angle_half_width_deg": float(half_width_deg),
                    "sg_profile_angle_raw_half_width_deg": float(raw_half_width_deg),
                    "sg_profile_angle_half_width_clipped": bool(half_width_deg < raw_half_width_deg),
                    "sg_profile_weight_sum": float(np.sum(profile_weights)),
                    "sg_profile_points_total": int(len(profile_weights)),
                    "sg_profile_same_branch_continuation_used": False,
                    "sg_profile_same_branch_continuation_segments": 1,
                    "sg_profile_root_search": "not_used_direct_angle",
                    "sg_jitter_kernel_points": 1,
                    "sg_jitter_offset_abs_p95_inv_A": 0.0,
                }
            )
            return (
                np.asarray(sample_omegas, dtype=float),
                np.asarray(profile_weights, dtype=float),
                np.asarray([], dtype=float),
                jitter_meta,
                mosaicity_meta,
            )
        if coordinate != "target_sg":
            raise ValueError(f"unknown profile_coordinate={coordinate!r}")

        requested, base_weights, jitter_meta = self.row_profile_sg_points(row_idx, hkl, sg_window_i)
        physical_targets = float(center_sg) + np.asarray(requested, dtype=float)
        base_omegas = []
        root_residuals = []
        for target_sg in physical_targets:
            omega = self.solve_sg_profile_target(
                row_idx,
                hkl,
                omega_center,
                target_sg,
                jitter_meta,
            )
            base_omegas.append(float(omega))
            solved = float(self.sg_for_hkls(np.asarray([hkl], dtype=int), omega)[0])
            root_residuals.append(solved - float(target_sg))

        base_weights = np.asarray(base_weights, dtype=float)
        jacobian = 1.0 / np.maximum(
            np.abs([self.sg_derivative_per_deg(hkl, omega) for omega in base_omegas]),
            1e-12,
        )
        original_weight_sum = float(np.sum(base_weights))
        base_weights *= jacobian
        base_weights *= original_weight_sum / max(float(np.sum(base_weights)), 1e-30)
        angular_offsets, angular_weights, mosaicity_meta = self.row_mosaicity_angle_points()
        sample_omegas = (
            np.asarray(base_omegas, dtype=float)[:, None] + np.asarray(angular_offsets, dtype=float)[None, :]
        ).reshape(-1)
        profile_weights = (base_weights[:, None] * np.asarray(angular_weights, dtype=float)[None, :]).reshape(-1)
        jitter_meta.update(
            {
                "sg_profile_coordinate": "target_sg",
                "sg_profile_angle_jacobian": True,
                "sg_profile_jacobian_ratio": float(np.max(jacobian) / max(np.min(jacobian), 1e-30)),
                "sg_profile_weight_sum": float(np.sum(profile_weights)),
                "sg_profile_points_total": int(len(profile_weights)),
            }
        )
        return (
            np.asarray(sample_omegas, dtype=float),
            np.asarray(profile_weights, dtype=float),
            np.asarray(root_residuals, dtype=float),
            jitter_meta,
            mosaicity_meta,
        )
