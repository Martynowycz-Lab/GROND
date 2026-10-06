"""Lab-frame orientations and bounded Ewald-sphere root solving.

Root branches and finite-difference steps are preserved verbatim from the audited
engine; this extraction changes ownership, not acquisition conventions."""

from __future__ import annotations

import math

import numpy as np
from scipy.optimize import brentq

from grond.bloch.errors import InfeasibleSgJitterProfileError, LocalSgCrossingError
from grond.bloch.numerics import _rotation_matrix_axis_angle


class GeometryMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def geometry_path_at_z(self, z):
        """Return fixed geometry-only corrections at a nominal frame."""

        if getattr(self, "geometry_path", None) is None:
            return 0.0, 0.0, 0.0
        contract = self.geometry_path
        x = (float(z) - float(contract["frame_center"])) / float(contract["frame_half_range"])
        x = float(np.clip(x, -1.0, 1.0))
        coeff = contract["coefficients"]
        return tuple(
            float(np.polynomial.chebyshev.chebval(x, coeff[name]))
            for name in ("slew_frames", "tilt_x_deg", "tilt_y_deg")
        )

    def omega_from_z(self, z):
        """Return the rotation angle (degrees) of nominal frame ``z``, after frame_offset and any geometry-path slew."""

        nominal_z = float(z)
        if getattr(self, "geometry_path", None) is None:
            return self.data.omega_from_z(nominal_z + float(self.cfg.frame_offset))
        slew_frames, _, _ = self.geometry_path_at_z(nominal_z)
        return self.data.omega_from_z(nominal_z + float(self.cfg.frame_offset) + float(slew_frames))

    def nominal_z_from_corrected_omega(self, omega_deg):
        """Invert the strictly increasing slew map within its declared displacement bound."""

        base = float(self.data.z_from_omega(omega_deg)) - float(self.cfg.frame_offset)
        if getattr(self, "geometry_path", None) is None:
            return base
        extent = float(self.geometry_path["bounds"]["max_abs_slew_frames"]) + 1e-10

        def residual(z):
            slew_frames, _, _ = self.geometry_path_at_z(z)
            return z - base + float(slew_frames)

        return float(brentq(residual, base - extent, base + extent, xtol=1e-12))

    def gvecs(self, hkls, omega_deg):
        """Return lab-frame reciprocal vectors g = S R(omega) F A h (Å⁻¹) of HKL rows at ``omega_deg`` degrees.

        R is the right-handed rotation about the scan axis. With a geometry path, A is taken at the nominal frame of
        ``omega_deg`` and the path's two tilts (about tilt axis x, then y) rotate the result.
        """

        if getattr(self, "geometry_path", None) is None:
            A = self.data.A_for_omega(omega_deg)
        else:
            nominal_z = self.nominal_z_from_corrected_omega(omega_deg)
            A = self.data.A_for_omega(self.data.omega_from_z(nominal_z))
        theta = np.radians(float(omega_deg))
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        R = self.data.axis_eye + s * self.data.axis_cross + (1.0 - c) * self.data.axis_cross2
        transform = self.data.setting_rotation @ R @ self.data.fixed_rotation @ A
        g = (transform @ np.asarray(hkls, dtype=float).T).T
        if getattr(self, "geometry_path", None) is None:
            return g
        _, tilt_x_deg, tilt_y_deg = self.geometry_path_at_z(nominal_z)
        axes = self.geometry_path["tilt_axes_lab"]
        correction = _rotation_matrix_axis_angle(axes[1], tilt_y_deg) @ _rotation_matrix_axis_angle(axes[0], tilt_x_deg)
        return (correction @ g.T).T

    def mosaicity_tangent_axes(self):
        """Return deterministic along-scan and transverse lab-frame axes.

        A small crystal rotation about the incident-beam direction does not
        change excitation errors, so an incoherent orientation distribution
        needs only the two axes tangent to the beam sphere.  The first axis is
        the scan axis projected into that plane; the second completes a
        right-handed orthonormal pair.  This keeps the two fitted components
        physically interpretable and avoids an arbitrary detector convention.
        """

        beam = np.asarray(self.data.s0, dtype=float).copy()
        beam /= max(float(np.linalg.norm(beam)), 1e-30)
        scan = np.asarray(self.data.axis, dtype=float).copy()
        scan -= float(np.dot(scan, beam)) * beam
        if float(np.linalg.norm(scan)) <= 1e-12:
            seed = np.asarray([1.0, 0.0, 0.0], dtype=float)
            if abs(float(np.dot(seed, beam))) > 0.9:
                seed = np.asarray([0.0, 1.0, 0.0], dtype=float)
            scan = seed - float(np.dot(seed, beam)) * beam
        along = scan / float(np.linalg.norm(scan))
        transverse = np.cross(beam, along)
        transverse /= float(np.linalg.norm(transverse))
        return along, transverse

    def gvecs_at_orientation(
        self,
        hkls,
        omega_deg,
        mosaic_along_deg=0.0,
        mosaic_transverse_deg=0.0,
    ):
        """Return reciprocal vectors at one explicit mosaic orientation."""

        g = np.asarray(self.gvecs(hkls, omega_deg), dtype=float)
        along_deg = float(mosaic_along_deg)
        transverse_deg = float(mosaic_transverse_deg)
        if along_deg == 0.0 and transverse_deg == 0.0:
            return g
        along, transverse = self.mosaicity_tangent_axes()
        correction = _rotation_matrix_axis_angle(transverse, transverse_deg) @ _rotation_matrix_axis_angle(
            along, along_deg
        )
        return (correction @ g.T).T

    def sg_for_hkls(self, hkls, omega_deg):
        """Return excitation errors S_g = (k^2 - |s0 + g|^2)/(2k) in Å⁻¹ of HKL rows at ``omega_deg``.

        S_g is positive when s0 + g lies inside the Ewald sphere.
        """

        g = self.gvecs(hkls, omega_deg)
        return (self.data.k**2 - np.sum((self.data.s0 + g) ** 2, axis=1)) / (2.0 * self.data.k)

    def sg_for_hkls_at_orientation(
        self,
        hkls,
        omega_deg,
        mosaic_along_deg=0.0,
        mosaic_transverse_deg=0.0,
    ):
        """Return S_g (Å⁻¹) after extra mosaic rotations (degrees) about the along-scan and transverse axes.

        With both angles zero this is sg_for_hkls.
        """

        if float(mosaic_along_deg) == 0.0 and float(mosaic_transverse_deg) == 0.0:
            return self.sg_for_hkls(hkls, omega_deg)
        g = self.gvecs_at_orientation(
            hkls,
            omega_deg,
            mosaic_along_deg,
            mosaic_transverse_deg,
        )
        return (self.data.k**2 - np.sum((self.data.s0 + g) ** 2, axis=1)) / (2.0 * self.data.k)

    def _fixed_xds_analytic_geometry_data(self):
        """Return fixed XDS geometry, or ``None`` when exact algebra is invalid."""

        data = getattr(self, "data", None)
        header = getattr(data, "header", {})
        source_text = str(header.get("GROND_SOURCE", "") or "").strip().upper()
        if source_text:
            source_kind = source_text
        else:
            source_kind = "DIALS" if any(str(key).upper().startswith("DIALS_") for key in header) else "XDS"
        required = (
            "fixed_rotation",
            "setting_rotation",
            "axis",
            "s0",
            "A",
            "k",
        )
        if (
            source_kind != "XDS"
            or getattr(data, "A_at_scan_points", None) is not None
            or getattr(self, "geometry_path", None) is not None
            or any(not hasattr(data, name) for name in required)
        ):
            return None
        return data

    def _analytic_fixed_xds_omega_for_sg(
        self,
        hkl,
        omega0,
        target_sg,
        max_delta_deg,
    ):
        """Return the exact local Sg root for a fixed XDS orientation matrix.

        This path is valid only when the source is XDS, ``A`` does not vary
        over the scan, and no fitted geometry path is active.  It solves the
        sinusoidal Ewald equation directly and retains the same nearest-local-
        branch contract as the numerical fallback.
        """

        data = self._fixed_xds_analytic_geometry_data()
        if data is None:
            raise NotImplementedError

        hkl_array = np.asarray(hkl, dtype=float)
        g0 = np.asarray(data.fixed_rotation, dtype=float) @ np.asarray(data.A, dtype=float) @ hkl_array
        axis = np.asarray(data.axis, dtype=float)
        beam = np.asarray(data.setting_rotation, dtype=float).T @ np.asarray(data.s0, dtype=float)
        parallel = axis * float(np.dot(axis, g0))
        perpendicular = g0 - parallel
        cross = np.cross(axis, g0)
        constant = float(np.dot(beam, parallel))
        cosine = float(np.dot(beam, perpendicular))
        sine = float(np.dot(beam, cross))
        radius = math.hypot(cosine, sine)
        right_hand_side = -float(data.k) * float(target_sg) - 0.5 * float(np.dot(g0, g0)) - constant
        if radius <= 1e-15:
            raise LocalSgCrossingError("fixed XDS geometry has no orientation-dependent Sg crossing")
        ratio = right_hand_side / radius
        if ratio < -1.0 - 1e-12 or ratio > 1.0 + 1e-12:
            raise LocalSgCrossingError("fixed XDS geometry has no real Sg crossing")
        ratio = float(np.clip(ratio, -1.0, 1.0))
        phase = math.atan2(sine, cosine)
        offset = math.acos(ratio)
        omega0_radians = math.radians(float(omega0))
        candidates = []
        for base in (phase - offset, phase + offset):
            central_turn = round((omega0_radians - base) / (2.0 * math.pi))
            for turn in (central_turn - 1, central_turn, central_turn + 1):
                candidate = math.degrees(base + 2.0 * math.pi * turn)
                if abs(candidate - float(omega0)) <= float(max_delta_deg) + 1e-10:
                    candidates.append(candidate)
        if not candidates:
            raise LocalSgCrossingError("fixed XDS geometry has no crossing on the requested local branch")
        return float(min(candidates, key=lambda value: abs(value - float(omega0))))

    def _numerical_omega_for_sg(
        self,
        hkl,
        omega0,
        target_sg,
        max_delta_deg,
    ):
        """Numerical local-root fallback for scan-varying or fitted geometry."""

        hkl_array = np.asarray([hkl], dtype=int)

        def residual(omega):
            """Return S_g(omega) minus the target S_g for this row's HKL."""

            return float(self.sg_for_hkls(hkl_array, float(omega))[0]) - target_sg

        offsets = np.linspace(-max_delta_deg, max_delta_deg, 65)
        values = np.asarray([residual(omega0 + offset) for offset in offsets], dtype=float)
        close = np.where(np.abs(values) <= 1e-10)[0]
        if len(close):
            best = int(close[np.argmin(np.abs(offsets[close]))])
            return omega0 + float(offsets[best])
        brackets = []
        for i in range(len(offsets) - 1):
            if values[i] * values[i + 1] < 0.0:
                midpoint = 0.5 * (offsets[i] + offsets[i + 1])
                brackets.append((abs(midpoint), i))
        if not brackets:
            raise LocalSgCrossingError(
                f"no local Sg={target_sg:.6g} crossing for HKL {tuple(int(v) for v in hkl)} "
                f"within {max_delta_deg:.6g} degrees of omega={omega0:.6g}"
            )
        _, index = min(brackets)
        lower = omega0 + float(offsets[index])
        upper = omega0 + float(offsets[index + 1])
        omega = float(brentq(residual, lower, upper, xtol=1e-12, rtol=1e-12, maxiter=100))
        final_residual = residual(omega)
        if abs(final_residual) > 1e-9 or abs(omega - omega0) > max_delta_deg + 1e-10:
            raise ValueError(
                f"local Sg solver failed for HKL {tuple(int(v) for v in hkl)}: "
                f"residual={final_residual:.3g}, delta_omega={omega - omega0:.3g} degrees"
            )
        return omega

    def _fast_analytic_newton_omega_for_sg(
        self,
        hkl,
        omega0,
        target_sg,
        max_delta_deg,
    ):
        """Return the local angle (degrees) where one HKL reaches ``target_sg``: closed form, then Newton steps.

        The closed-form crossing uses A frozen at ``omega0``; up to three Newton steps on the full, possibly
        scan-varying, geometry remove the residual. Raises NotImplementedError without fixed-rotation fields or with a
        geometry path, and LocalSgCrossingError when no crossing lies within ``max_delta_deg`` or the final residual
        exceeds 1e-9.
        """

        data = getattr(self, "data", None)
        if data is None:
            raise NotImplementedError
        required = ("fixed_rotation", "setting_rotation", "axis", "s0", "k")
        if any(not hasattr(data, name) for name in required) or getattr(self, "geometry_path", None) is not None:
            raise NotImplementedError

        A_local = data.A_for_omega(omega0) if hasattr(data, "A_for_omega") else getattr(data, "A", None)
        if A_local is None:
            raise NotImplementedError

        hkl_array = np.asarray(hkl, dtype=float)
        g0 = np.asarray(data.fixed_rotation, dtype=float) @ np.asarray(A_local, dtype=float) @ hkl_array
        axis = np.asarray(data.axis, dtype=float)
        beam = np.asarray(data.setting_rotation, dtype=float).T @ np.asarray(data.s0, dtype=float)
        parallel = axis * float(np.dot(axis, g0))
        perpendicular = g0 - parallel
        cross = np.cross(axis, g0)
        constant = float(np.dot(beam, parallel))
        cosine = float(np.dot(beam, perpendicular))
        sine = float(np.dot(beam, cross))
        radius = math.hypot(cosine, sine)
        if radius <= 1e-15:
            raise LocalSgCrossingError("geometry has no orientation-dependent Sg crossing")
        right_hand_side = -float(data.k) * float(target_sg) - 0.5 * float(np.dot(g0, g0)) - constant
        ratio = right_hand_side / radius
        if ratio < -1.0 - 1e-12 or ratio > 1.0 + 1e-12:
            raise LocalSgCrossingError("geometry has no real Sg crossing")
        ratio = float(np.clip(ratio, -1.0, 1.0))
        phase = math.atan2(sine, cosine)
        offset = math.acos(ratio)
        omega0_radians = math.radians(float(omega0))
        candidates = []
        for base in (phase - offset, phase + offset):
            central_turn = round((omega0_radians - base) / (2.0 * math.pi))
            for turn in (central_turn - 1, central_turn, central_turn + 1):
                candidate = math.degrees(base + 2.0 * math.pi * turn)
                if abs(candidate - float(omega0)) <= float(max_delta_deg) + 1e-10:
                    candidates.append(candidate)
        if not candidates:
            raise LocalSgCrossingError("geometry has no crossing on the requested local branch")
        omega = float(min(candidates, key=lambda value: abs(value - float(omega0))))

        # Newton refinement for scan-varying geometry or residual polishing
        hkl_arr = np.asarray([hkl], dtype=int)
        for _ in range(3):
            sg_val = float(self.sg_for_hkls(hkl_arr, omega)[0])
            res = sg_val - target_sg
            if abs(res) <= 1e-10:
                break
            deriv = float(self.sg_derivative_per_deg(hkl, omega))
            if abs(deriv) < 1e-9:
                break
            step = res / deriv
            if abs(step) > float(max_delta_deg):
                break
            omega -= step

        final_residual = float(self.sg_for_hkls(hkl_arr, omega)[0]) - target_sg
        if abs(final_residual) > 1e-9 or abs(omega - float(omega0)) > float(max_delta_deg) + 1e-10:
            raise LocalSgCrossingError("fast solver failed tolerance")
        return omega

    def solve_omega_for_sg(self, hkl, omega0, target_sg, max_delta_deg=None):
        """Solve one local Ewald crossing without jumping to another branch."""

        hkl_array = np.asarray([hkl], dtype=int)
        omega0 = float(omega0)
        target_sg = float(target_sg)

        def residual(omega):
            """Return S_g(omega) minus the target S_g for this row's HKL."""

            return float(self.sg_for_hkls(hkl_array, float(omega))[0]) - target_sg

        initial = residual(omega0)
        if abs(initial) <= 1e-10:
            return omega0
        oscillation = abs(float(self.data.oscillation_range))
        if max_delta_deg is None:
            max_delta_deg = max(2.0 * oscillation, min(2.0, 10.0 * oscillation))
        max_delta_deg = float(max_delta_deg)
        if not math.isfinite(max_delta_deg) or max_delta_deg <= 0.0:
            raise ValueError("local Sg root-search range must be finite and positive")

        try:
            return self._fast_analytic_newton_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )
        except (NotImplementedError, LocalSgCrossingError):
            pass

        try:
            omega = self._analytic_fixed_xds_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )
        except NotImplementedError:
            return self._numerical_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )
        except LocalSgCrossingError:
            return self._numerical_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )

        final_residual = residual(omega)
        if abs(final_residual) > 1e-9:
            return self._numerical_omega_for_sg(
                hkl,
                omega0,
                target_sg,
                max_delta_deg,
            )
        return omega

    def row_diagnostic_omega_bragg(self, hkl, omega_center):
        """Return the reporting-only zero-Sg angle and its availability.

        Propagation uses the recorded scan-centered Sg profile and never this
        diagnostic. Coarse-sliced DIALS shoeboxes can legitimately span a
        crossing just outside the deliberately strict local-root window. In
        that case retain the observation and report the unavailable diagnostic
        as NaN instead of aborting or changing row membership.
        """

        if str(getattr(self.cfg, "profile_coordinate", "target_sg")) == "direct_angle":
            return float(omega_center), True
        try:
            return float(self.solve_omega_for_sg(hkl, omega_center, 0.0)), True
        except LocalSgCrossingError:
            return float("nan"), False

    def solve_sg_profile_target(self, row_idx, hkl, omega_center, target_sg, jitter_meta):
        """Solve one profile target while preserving the strict local-root contract.

        A nonzero jitter kernel is a nuisance candidate, not required geometry.
        If one of its convolved off-center samples has no crossing inside the
        audited local angular bound, the whole jitter candidate is infeasible.
        The caller can then retain ordinary or otherwise feasible candidates;
        it must not widen the bound or silently truncate and renormalize the
        requested orientation distribution.
        """

        try:
            return self.solve_omega_for_sg(hkl, omega_center, target_sg)
        except LocalSgCrossingError as exc:
            mode = str(jitter_meta.get("sg_jitter_mode", "none") or "none").lower()
            sigma = float(jitter_meta.get("sg_jitter_sigma_inv_A", 0.0) or 0.0)
            if mode == "none" or sigma <= 0.0:
                omega, segments = self.solve_same_branch_sg_profile_target(
                    hkl,
                    omega_center,
                    target_sg,
                    initial_error=exc,
                )
                jitter_meta["sg_profile_same_branch_continuation_used"] = True
                jitter_meta["sg_profile_same_branch_continuation_segments"] = max(
                    int(jitter_meta.get("sg_profile_same_branch_continuation_segments", 1)),
                    int(segments),
                )
                return omega
            raise InfeasibleSgJitterProfileError(
                "infeasible Sg-jitter quadrature target for "
                f"row {int(row_idx)}, HKL {tuple(int(v) for v in hkl)}: "
                f"mode={mode}, sigma={sigma:.6g} 1/A, target Sg={float(target_sg):.6g} 1/A; "
                f"{exc}. Reject this nuisance candidate; the local root-search bound is unchanged."
            ) from exc

    def solve_same_branch_sg_profile_target(
        self,
        hkl,
        omega_center,
        target_sg,
        *,
        initial_error,
    ):
        """Continue a required base-profile target along one monotonic Ewald branch.

        ``solve_omega_for_sg`` intentionally keeps a short angular bound because
        it is also used to validate the Bragg crossing assigned to an observed
        row.  A configured base Sg profile is different: its exact support can
        legitimately extend farther in angle when ``dSg/domega`` is small.  We
        reach that support through strict local solves for successively closer
        Sg targets, and reject any path that reverses angle or crosses a
        stationary point.  Active jitter remains subject to the single local
        bound in ``solve_sg_profile_target`` because it is an optional nuisance
        candidate rather than required base-profile support.
        """

        hkl_array = np.asarray([hkl], dtype=int)
        omega_center = float(omega_center)
        target_sg = float(target_sg)
        center_sg = float(self.sg_for_hkls(hkl_array, omega_center)[0])
        target_delta = target_sg - center_sg
        derivative0 = float(self.sg_derivative_per_deg(hkl, omega_center))
        derivative_tol = 1e-12
        if (
            not math.isfinite(center_sg)
            or not math.isfinite(target_delta)
            or not math.isfinite(derivative0)
            or abs(derivative0) <= derivative_tol
        ):
            raise LocalSgCrossingError(
                f"cannot continue Sg={target_sg:.6g} for HKL {tuple(int(v) for v in hkl)} "
                f"from omega={omega_center:.6g}: the local Ewald branch has "
                f"dSg/domega={derivative0:.6g} 1/A/degree"
            ) from initial_error

        expected_angle_direction = math.copysign(1.0, target_delta / derivative0)
        oscillation = abs(float(self.data.oscillation_range))
        strict_local_bound = max(2.0 * oscillation, min(2.0, 10.0 * oscillation))
        max_segments = 8
        max_total_delta_deg = max_segments * strict_local_bound
        last_reason = str(initial_error)
        for segments in (2, 4, max_segments):
            omega = omega_center
            branch_ok = True
            for step_index in range(1, segments + 1):
                intermediate_target = center_sg + target_delta * (step_index / segments)
                try:
                    solved = self.solve_omega_for_sg(hkl, omega, intermediate_target)
                except LocalSgCrossingError as exc:
                    last_reason = str(exc)
                    branch_ok = False
                    break
                angle_step = float(solved - omega)
                derivative = float(self.sg_derivative_per_deg(hkl, solved))
                total_angle_delta = abs(float(solved - omega_center))
                if (
                    not math.isfinite(derivative)
                    or abs(derivative) <= derivative_tol
                    or derivative * derivative0 <= 0.0
                    or angle_step * expected_angle_direction < -1e-10
                    or total_angle_delta > max_total_delta_deg + 1e-10
                ):
                    last_reason = (
                        "continuation would leave the monotonic local Ewald branch: "
                        f"omega={solved:.6g}, dSg/domega={derivative:.6g} 1/A/degree"
                    )
                    branch_ok = False
                    break
                interval_error = self.same_branch_interval_error(
                    hkl,
                    omega,
                    solved,
                    derivative0,
                    strict_local_bound,
                )
                if interval_error:
                    last_reason = interval_error
                    branch_ok = False
                    break
                omega = float(solved)
            if branch_ok:
                final_residual = float(self.sg_for_hkls(hkl_array, omega)[0]) - target_sg
                if abs(final_residual) <= 1e-9:
                    return omega, segments
                last_reason = f"final continuation residual was {final_residual:.6g} 1/A"

        raise LocalSgCrossingError(
            f"no same-branch Sg={target_sg:.6g} continuation for HKL "
            f"{tuple(int(v) for v in hkl)} from omega={omega_center:.6g} after "
            f"up to {max_segments} strict local segments and "
            f"{max_total_delta_deg:.6g} degrees total; {last_reason}"
        ) from initial_error

    def same_branch_interval_error(
        self,
        hkl,
        omega_a,
        omega_b,
        reference_derivative,
        strict_local_bound,
    ):
        """Return an explanation if one continuation interval is not monotonic.

        Endpoint derivative checks alone can miss a reversal and a second turn
        inside one successful root bracket.  Sample no more coarsely than the
        strict solver's 65-point grid, insert every DIALS scan-matrix boundary,
        and require both finite-difference slopes and midpoint derivatives to
        retain the reference branch sign.
        """

        omega_a = float(omega_a)
        omega_b = float(omega_b)
        if abs(omega_b - omega_a) <= 1e-12:
            return None
        grid_step = 2.0 * float(strict_local_bound) / 64.0
        sample_count = max(2, int(math.ceil(abs(omega_b - omega_a) / grid_step)) + 1)
        samples = list(np.linspace(min(omega_a, omega_b), max(omega_a, omega_b), sample_count))
        scan_matrices = getattr(self.data, "A_at_scan_points", None)
        starting_angle = getattr(self.data, "starting_angle", None)
        oscillation = getattr(self.data, "oscillation_range", None)
        if scan_matrices is not None and starting_angle is not None and oscillation is not None:
            for index in range(len(scan_matrices)):
                boundary = float(starting_angle) + index * float(oscillation)
                if samples[0] < boundary < samples[-1]:
                    samples.append(boundary)
        samples = np.asarray(sorted(set(samples)), dtype=float)
        hkl_array = np.asarray([hkl], dtype=int)
        values = np.asarray(
            [float(self.sg_for_hkls(hkl_array, omega)[0]) for omega in samples],
            dtype=float,
        )
        if not np.all(np.isfinite(values)):
            return "continuation encountered a non-finite Sg value inside a local segment"
        reference_sign = math.copysign(1.0, float(reference_derivative))
        slopes = np.diff(values) / np.diff(samples)
        slope_tol = 1e-12
        if np.any(~np.isfinite(slopes)) or np.any(reference_sign * slopes <= slope_tol):
            bad = int(np.where((~np.isfinite(slopes)) | (reference_sign * slopes <= slope_tol))[0][0])
            return (
                "continuation reverses or stalls inside a local segment: "
                f"omega interval {samples[bad]:.6g}..{samples[bad + 1]:.6g}, "
                f"finite-difference dSg/domega={slopes[bad]:.6g} 1/A/degree"
            )
        midpoints = 0.5 * (samples[:-1] + samples[1:])
        midpoint_derivatives = np.asarray(
            [float(self.sg_derivative_per_deg(hkl, omega)) for omega in midpoints],
            dtype=float,
        )
        if np.any(~np.isfinite(midpoint_derivatives)) or np.any(reference_sign * midpoint_derivatives <= slope_tol):
            bad = int(
                np.where((~np.isfinite(midpoint_derivatives)) | (reference_sign * midpoint_derivatives <= slope_tol))[
                    0
                ][0]
            )
            return (
                "continuation derivative reverses or stalls inside a local segment: "
                f"omega={midpoints[bad]:.6g}, "
                f"dSg/domega={midpoint_derivatives[bad]:.6g} 1/A/degree"
            )
        return None

    def _analytic_fixed_xds_sg_derivative_per_deg(self, hkl, omega_deg):
        """Exact dSg/domega for the same fixed-XDS contract as the root solver."""

        data = self._fixed_xds_analytic_geometry_data()
        if data is None:
            raise NotImplementedError
        g0 = (
            np.asarray(data.fixed_rotation, dtype=float)
            @ np.asarray(data.A, dtype=float)
            @ np.asarray(hkl, dtype=float)
        )
        theta = math.radians(float(omega_deg))
        cosine = math.cos(theta)
        sine = math.sin(theta)
        axis = np.asarray(data.axis, dtype=float)
        rotated = cosine * g0 + sine * np.cross(axis, g0) + (1.0 - cosine) * axis * float(np.dot(axis, g0))
        derivative = np.cross(axis, rotated)
        beam = np.asarray(data.setting_rotation, dtype=float).T @ np.asarray(data.s0, dtype=float)
        per_radian = -float(np.dot(beam, derivative)) / float(data.k)
        return per_radian * (math.pi / 180.0)

    def sg_derivative_per_deg(self, hkl, omega_deg):
        """Return dS_g/domega (Å⁻¹ per degree): analytic for fixed XDS geometry, else a central difference.

        The finite-difference step is 1e-4 degree.
        """

        try:
            return self._analytic_fixed_xds_sg_derivative_per_deg(hkl, omega_deg)
        except NotImplementedError:
            pass
        step = 1e-4
        hp = np.asarray([hkl], dtype=int)
        sp = float(self.sg_for_hkls(hp, float(omega_deg) + step)[0])
        sm = float(self.sg_for_hkls(hp, float(omega_deg) - step)[0])
        return (sp - sm) / (2.0 * step)
