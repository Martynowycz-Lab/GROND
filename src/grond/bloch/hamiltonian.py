"""Static many-beam coupling, optical loss and real excitation diagonals.

The propagation sign is exp(2*pi*i*H*t); positive semidefinite imaginary loss
is required for passive propagation. Physical defaults remain unchanged."""

from __future__ import annotations

import math

import numpy as np

from grond.bloch.errors import NonPassiveAbsorptionError


def validate_passive_absorption(H, *, mode="unknown", row_index=None):
    """Require nonnegative loss for exp(2*pi*i*H*t), without changing H.

    With H=A+iB, d||psi||^2/dt=-4*pi*psi^dagger*B*psi. Gershgorin
    bounds avoid another eigensolve when positive diagonal loss suffices.
    This check is performed on the static matrix once per row, because adding
    real excitation errors at the rocking points cannot change B.
    """

    matrix = np.asarray(H, dtype=np.complex128)
    loss = (matrix - matrix.conj().T) / (2j)
    magnitude = float(np.linalg.norm(loss, ord=np.inf))
    tolerance = 1e-12 + 1e-10 * magnitude
    diagonal = np.real(np.diag(loss))
    radii = np.sum(np.abs(loss), axis=1) - np.abs(np.diag(loss))
    bound = float(np.min(diagonal - radii))
    if bound >= -tolerance:
        return
    minimum = float(np.linalg.eigvalsh(loss)[0])
    if minimum < -tolerance:
        details = {
            "reason": "nonpassive_absorption",
            "absorption_mode": str(mode),
            "row_index": None if row_index is None else int(row_index),
            "basis_size": int(len(matrix)),
            "minimum_loss_eigenvalue_inv_A": minimum,
            "minimum_additional_uniform_mu_inv_A": -minimum,
            "tolerance_inv_A": tolerance,
        }
        raise NonPassiveAbsorptionError(
            f"absorption candidate {mode!r} is not passive: minimum loss eigenvalue "
            f"{minimum:.6g} 1/A. Use an elastic candidate or a physically supported "
            "positive absorptive potential; increasing a fit parameter merely to "
            "pass this check is not an absorption model.",
            details=details,
        )


class HamiltonianMixin:
    """One cohesive part of XDSBlochPort; initialized by the public facade."""

    def absorption_enabled(self):
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        return mode != "none" and (
            abs(float(getattr(self.cfg, "absorption_mu", 0.0))) > 0.0
            or abs(float(getattr(self.cfg, "absorption_frac", 0.0))) > 0.0
            or abs(float(getattr(self.cfg, "absorption_mu_g2", 0.0))) > 0.0
            or abs(float(getattr(self.cfg, "absorption_mu_theta2", 0.0))) > 0.0
        )

    def beam_absorption_mu(self, basis):
        """Beam-specific diagonal absorptive potential in Hamiltonian units.

        The returned values are added as ``+i*mu_i``. With the project
        convention ``exp(2*pi*i*H*t)``, positive ``mu_i`` damps amplitudes as
        ``exp(-2*pi*mu_i*t)``.  ``absorption_mu`` is the uniform term.  The
        optional slopes are global, auditable damping probes versus |g|^2 and
        the small Bragg angle squared; they are not row-wise fitted knobs.
        """
        inv_d2 = np.asarray(self.inv_d2_for_hkls(np.asarray(basis, dtype=int)), dtype=float)
        sin_theta = np.sqrt(np.maximum(inv_d2, 0.0)) / max(2.0 * float(self.data.k), 1e-30)
        theta = np.arcsin(np.clip(sin_theta, 0.0, 1.0))
        return (
            float(getattr(self.cfg, "absorption_mu", 0.0))
            + float(getattr(self.cfg, "absorption_mu_g2", 0.0)) * inv_d2
            + float(getattr(self.cfg, "absorption_mu_theta2", 0.0)) * theta * theta
        )

    def offdiag_absorption_eta(self, diff):
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        eta = float(getattr(self.cfg, "absorption_frac", 0.0))
        if eta == 0.0:
            return 0.0
        if mode in {"offdiag_fraction", "midgley_like", "midgley"}:
            return eta
        if mode == "offdiag_tds":
            inv_d2 = float(self.inv_d2_for_hkls(diff))
            b_tds = max(float(getattr(self.cfg, "absorption_tds_b", 0.0)), 0.0)
            return eta * (1.0 - math.exp(-0.25 * b_tds * inv_d2))
        return 0.0

    def offdiag_absorption_eta_array(self, diffs):
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        eta = float(getattr(self.cfg, "absorption_frac", 0.0))
        diffs = np.asarray(diffs, dtype=int)
        if eta == 0.0:
            return np.zeros(diffs.shape[:-1], dtype=float)
        if mode in {"offdiag_fraction", "midgley_like", "midgley"}:
            return np.full(diffs.shape[:-1], eta, dtype=float)
        if mode == "offdiag_tds":
            inv_d2 = self.inv_d2_for_hkls(diffs.reshape(-1, 3)).reshape(diffs.shape[:-1])
            b_tds = max(float(getattr(self.cfg, "absorption_tds_b", 0.0)), 0.0)
            return eta * (1.0 - np.exp(-0.25 * b_tds * inv_d2))
        return np.zeros(diffs.shape[:-1], dtype=float)

    def hamiltonian_static_part(self, basis, absorb: bool = True, row_idx=None):
        """Return the row-static Hamiltonian terms for a fixed beam basis.

        For a given row and support basis the off-diagonal Fourier-potential
        couplings do not depend on the target-Sg quadrature point.  Only the
        diagonal excitation errors change as the crystal is rocked through the
        profile.  Keeping this static part separate avoids rebuilding the same
        dense U(h_j-h_i) block for every profile sample.
        """

        n = len(basis)
        H = np.zeros((n, n), dtype=np.complex128)
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        if absorb and mode in {"mean", "midgley_like", "midgley"}:
            # Positive imaginary diagonal gives exp(2*pi*i*(S+i*mu)*t),
            # i.e. uniform elastic-beam damping exp(-2*pi*mu*t).
            H[np.diag_indices(n)] += 1j * float(getattr(self.cfg, "absorption_mu", 0.0))
        elif absorb and mode == "beam_dependent":
            H[np.diag_indices(n)] += 1j * self.beam_absorption_mu(basis)
        convention = str(getattr(self.cfg, "matrix_convention", "j_minus_i") or "j_minus_i")
        if convention not in {"j_minus_i", "i_minus_j"}:
            raise ValueError(f"unknown matrix_convention={convention!r}; expected j_minus_i or i_minus_j")
        # Production signed-row convention: use U(h_j - h_i).
        # The alternate sign is kept as an auditable raw-forward-model test.
        diffs = (
            basis[None, :, :] - basis[:, None, :]
            if convention == "j_minus_i"
            else basis[:, None, :] - basis[None, :, :]
        )
        u_mat = self.ug_array(diffs) if row_idx is None else self.ug_for_row_array(diffs, row_idx)
        if absorb:
            eta = self.offdiag_absorption_eta_array(diffs)
            if np.any(eta):
                # Absorptive Fourier coefficients are represented as a
                # same-phase imaginary potential. V(-g)=conj(U_g)+i*eta*conj(U_g),
                # not conj(V(g)), so the matrix is intentionally non-Hermitian.
                # The midgley_like mode is an optical-potential diagnostic
                # following Colmey, Doherty, Malik, and Midgley's absorption
                # analysis for electron-diffraction Bloch row modelling; it is gated rather
                # than assumed because absorption can be negligible for many
                # routine light-atom integrated-intensity regularizations.
                u_mat = u_mat + 1j * eta * u_mat
        np.fill_diagonal(u_mat, 0.0 + 0.0j)
        H += u_mat / (2.0 * self.data.k)
        if absorb and self.absorption_enabled():
            validate_passive_absorption(H, mode=mode, row_index=row_idx)
        return H

    def hamiltonian_from_static(self, static_H, sg_basis):
        H = np.array(static_H, dtype=np.complex128, copy=True)
        diag = np.diag_indices(H.shape[0])
        H[diag] += np.asarray(sg_basis, dtype=float)
        return H

    def hamiltonian(self, basis, sg_basis, absorb: bool = True, row_idx=None):
        static_H = self.hamiltonian_static_part(basis, absorb=absorb, row_idx=row_idx)
        return self.hamiltonian_from_static(static_H, sg_basis)
