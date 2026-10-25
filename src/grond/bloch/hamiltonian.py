"""Static many-beam coupling, optical loss and real excitation diagonals.

The propagation sign is exp(2*pi*i*H*t); positive semidefinite imaginary loss
is required for passive propagation. Physical defaults remain unchanged."""

from __future__ import annotations

import operator
from collections.abc import Sequence

import numpy as np

from grond.bloch.errors import NonPassiveAbsorptionError


def hamiltonian_with_excitation(static, excitation):
    """Copy a static Hamiltonian and add real excitation errors without changing its precision."""

    matrix = np.array(static, copy=True)
    matrix[np.diag_indices(len(matrix))] += np.asarray(excitation, dtype=matrix.real.dtype)
    return matrix


class ProfileHamiltonians(Sequence):
    """A frozen profile's Hamiltonians, constructed one at a time from static coupling and real Sg values.

    Storage is O(N² + P*N), not O(P*N²), for N beams and P profile points. Iteration/indexing produces the same
    dense matrices as the former list. Explicit np.asarray conversion is supported when a complete stack is wanted.
    """

    def __init__(self, static, excitation):
        self.static = np.array(static, copy=True)
        self.excitation = np.array(excitation, dtype=self.static.real.dtype, copy=True)
        if (self.static.ndim != 2 or self.static.shape[0] == 0 or self.static.shape[0] != self.static.shape[1]
                or self.excitation.ndim != 2 or self.excitation.shape[1] != self.static.shape[0]):
            raise ValueError("profile Hamiltonians need a square static matrix and one Sg per beam at every point")
        self.static.setflags(write=False)
        self.excitation.setflags(write=False)

    def __len__(self):
        return len(self.excitation)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return ProfileHamiltonians(self.static, self.excitation[index])
        return hamiltonian_with_excitation(self.static, self.excitation[operator.index(index)])

    @property
    def shape(self):
        return (len(self), *self.static.shape)

    @property
    def storage_nbytes(self):
        return self.static.nbytes + self.excitation.nbytes

    def diagonals(self):
        """All matrix diagonals in O(P*N) storage, including any static diagonal potential."""
        return np.diag(self.static)[None, :] + self.excitation

    def __array__(self, dtype=None, copy=None):
        if copy is False:
            raise ValueError("a full profile Hamiltonian stack requires a new allocation")
        output = np.empty(self.shape, dtype=self.static.dtype if dtype is None else dtype)
        for index, matrix in enumerate(self):
            output[index] = matrix
        return output


def validate_passive_absorption(H, *, mode="unknown", row_index=None):
    """Require nonnegative loss for exp(2*pi*i*H*t), without changing H.

    With H=A+iB, d||psi||^2/dt=-4*pi*psi^dagger*B*psi. Gershgorin
    bounds avoid another eigensolve when positive diagonal loss suffices.
    This check is performed on the static matrix once per row, because adding
    real excitation errors at the rocking points cannot change B.
    """

    matrix = np.asarray(H, dtype=np.complex128)
    if (matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[0] != matrix.shape[1]
            or not np.all(np.isfinite(matrix))):
        raise ValueError("passivity requires a nonempty finite square Hamiltonian")
    loss = (0.5 * matrix - 0.5 * matrix.conj().T) / 1j
    magnitude = float(np.linalg.norm(loss, ord=np.inf))
    if not np.isfinite(magnitude):
        raise ValueError("passivity requires a finite loss-matrix norm")
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
        """Return True when an absorption mode is set and the uniform absorption strength is nonzero."""

        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        return mode != "none" and abs(float(getattr(self.cfg, "absorption_mu", 0.0))) > 0.0

    def hamiltonian_static_part(self, basis, absorb: bool = True, row_idx=None):
        """Return the row-static Hamiltonian terms for a fixed beam basis.

        For a given row and support basis the off-diagonal Fourier-potential
        couplings do not depend on the target-Sg quadrature point.  Only the
        diagonal excitation errors change as the crystal is rocked through the
        profile.  Keeping this static part separate avoids rebuilding the same
        dense U(h_j-h_i) block for every profile sample.
        """

        n = len(basis)
        double_prec = bool(getattr(self.cfg, "double_precision", False))
        h_dtype = np.complex128 if double_prec else np.complex64
        H = np.zeros((n, n), dtype=h_dtype)
        mode = str(getattr(self.cfg, "absorption_mode", "none") or "none").lower()
        if absorb and mode in {"mean", "midgley_like", "midgley"}:
            # Positive imaginary diagonal gives exp(2*pi*i*(S+i*mu)*t),
            # i.e. uniform elastic-beam damping exp(-2*pi*mu*t).
            H[np.diag_indices(n)] += 1j * float(getattr(self.cfg, "absorption_mu", 0.0))
        # Production signed-row convention: use U(h_j - h_i).
        diffs = basis[None, :, :] - basis[:, None, :]
        u_mat = self.ug_array(diffs) if row_idx is None else self.ug_for_row_array(diffs, row_idx)
        np.fill_diagonal(u_mat, 0.0 + 0.0j)
        H += u_mat.astype(h_dtype, copy=False) / (2.0 * self.data.k)
        if absorb and self.absorption_enabled():
            validate_passive_absorption(H, mode=mode, row_index=row_idx)
        return H

    def hamiltonian_from_static(self, static_H, sg_basis):
        """Return a copy of ``static_H`` with the excitation errors ``sg_basis`` (Å⁻¹) added to the diagonal.

        The diagonal is added as float64 for a complex128 matrix and as float32 otherwise, so the precision is kept.
        """

        return hamiltonian_with_excitation(static_H, sg_basis)

    def hamiltonian(self, basis, sg_basis, absorb: bool = True, row_idx=None):
        """Return the full row Hamiltonian: static coupling and absorption part plus ``sg_basis`` on the diagonal."""

        static_H = self.hamiltonian_static_part(basis, absorb=absorb, row_idx=row_idx)
        return self.hamiltonian_from_static(static_H, sg_basis)
