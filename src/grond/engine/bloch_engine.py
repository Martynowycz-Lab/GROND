#!/usr/bin/env python3
"""The Bloch-wave calculator of one crystal (XDSBlochPort), used by every engine stage.

Conventions:

* signed observation HKLs are preserved;
* off-diagonal coupling uses U(h_j - h_i), not U(h_i - h_j);
* the rocking path is sampled in target Sg and converted back to XDS omega;
* all basis-beam Sg values are recomputed at each omega;
* Born and Bloch predictions are scaled and scored on identical row masks.
"""

from __future__ import annotations

import copy
import math
import multiprocessing as multiprocessing
import os
import sys
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

if not __package__:
    # Engine scripts run as files import this module by name; the package must be importable too.
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
from scipy.linalg import eig, expm

try:
    from grond.paths import runtime_cache_dir
except ImportError:  # Direct script execution from the legacy engine directory.

    def runtime_cache_dir(*parts: str) -> Path:
        """Return the GROND cache directory ($GROND_CACHE_DIR or ~/.cache/grond) joined with parts."""

        override = os.environ.get("GROND_CACHE_DIR", "").strip()
        root = Path(override).expanduser() if override else Path.home() / ".cache" / "grond"
        return root.joinpath(*parts)


DEFAULT_HKL = Path("INTEGRATE.HKL")
DEFAULT_RES = Path("model.res")


from grond.bloch.basis import (
    BasisMixin as BasisMixin,
)
from grond.bloch.cache import (
    dense_ug_memory_plan as dense_ug_memory_plan,
    enforce_dense_ug_memory_limit as enforce_dense_ug_memory_limit,
    row_plan_cache_validation_errors as row_plan_cache_validation_errors,
)
from grond.bloch.config import (
    Config as Config,
    frozen_kernel_contract as frozen_kernel_contract,
)
from grond.bloch.constants import (
    DEFAULT_MAX_SUPPORT_SG as DEFAULT_MAX_SUPPORT_SG,
    DENSE_UG_BUILD_BYTES_PER_VOXEL as DENSE_UG_BUILD_BYTES_PER_VOXEL,
    DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL as DENSE_UG_CACHE_LOAD_BYTES_PER_VOXEL,
    UG_CACHE_SCHEMA as UG_CACHE_SCHEMA,
)
from grond.bloch.errors import (
    InfeasibleSgJitterProfileError as InfeasibleSgJitterProfileError,
    LocalSgCrossingError as LocalSgCrossingError,
    NonPassiveAbsorptionError as NonPassiveAbsorptionError,
)
from grond.bloch.geometry import (
    GeometryMixin as GeometryMixin,
)
from grond.bloch.hamiltonian import (
    HamiltonianMixin as HamiltonianMixin,
    validate_passive_absorption as validate_passive_absorption,
)
from grond.bloch.metrics import (
    read_row_index_file as read_row_index_file,
)
from grond.bloch.model import (
    alternate_hand_model as alternate_hand_model,
    atom_dw as atom_dw,
    atom_dw_array as atom_dw_array,
    cell_orthogonalization_matrix as cell_orthogonalization_matrix,
    cif_number as cif_number,
    cif_symmetry_records as cif_symmetry_records,
    fcalc_res as fcalc_res,
    fcalc_res_grid as fcalc_res_grid,
    inv_d2_from_hkl_array as inv_d2_from_hkl_array,
    loop_rows as loop_rows,
    model_symmetry_ops as model_symmetry_ops,
    parse_cif as parse_cif,
    parse_model as parse_model,
    parse_res as parse_res,
    relativistic_gamma_from_wavelength as relativistic_gamma_from_wavelength,
    shelx_u_cart_matrix as shelx_u_cart_matrix,
    unique_symmetry_mates as unique_symmetry_mates,
)
from grond.bloch.numerics import (
    _rotation_matrix_axis_angle as _rotation_matrix_axis_angle,
    cell_from_real_matrix as cell_from_real_matrix,
    hermitian_eigh as hermitian_eigh,
    load_geometry_path_contract as load_geometry_path_contract,
    parse_hkl_transform as parse_hkl_transform,
    rotation_matrix as rotation_matrix,
    sha256 as sha256,
)
from grond.bloch.potential import (
    PotentialMixin as PotentialMixin,
)
from grond.bloch.prediction import (
    PredictionMixin as PredictionMixin,
)
from grond.bloch.profiles import (
    ProfileMixin as ProfileMixin,
)
from grond.bloch.row_cache import (
    RowCacheMixin as RowCacheMixin,
)
from grond.bloch.row_plans import (
    RowPlanMixin as RowPlanMixin,
)
from grond.bloch.splits import (
    build_transform_split_contract as build_transform_split_contract,
    candidate_invariant_split_membership as candidate_invariant_split_membership,
    metric_filter_signed_axis_transforms as metric_filter_signed_axis_transforms,
    validate_transform_split_contract as validate_transform_split_contract,
)
from grond.bloch.thickness import gamma_cf_point as gamma_cf_point, uniform_grid_amplitudes as uniform_grid_amplitudes
from grond.io.xds import (
    XDSData as XDSData,
)


class XDSBlochPort(
    PotentialMixin,
    GeometryMixin,
    ProfileMixin,
    BasisMixin,
    HamiltonianMixin,
    RowPlanMixin,
    RowCacheMixin,
    PredictionMixin,
):
    """Public compatibility facade for the signed-row Bloch forward model.

    Mixins own disjoint responsibilities and share the state initialized here.
    Dense propagation remains here to keep its explicit class-level fallbacks
    and existing standalone script semantics unchanged.
    """

    def __init__(
        self, data: XDSData, res_model, cfg: Config, cache_dir: Path | None = None, res_hash: str | None = None
    ):
        """Bind one reflection table, model and configuration, and build the shared Bloch state.

        Loads or builds the Ug cube (skipped in sparse_active storage), the beam pool, the thickness
        quadrature, the frame scale groups and the ZCAL range used by frame-dependent terms. A geometry-path
        contract must carry the hash of the current HKL file.
        """

        self.data = data
        source_hashes = self.data.verify_source_files()
        self.model = copy.deepcopy(res_model)
        self.cfg = cfg
        self.geometry_path = load_geometry_path_contract(getattr(cfg, "geometry_path_file", ""))
        if self.geometry_path is not None:
            expected_hkl_hash = str(self.geometry_path.get("source_hkl_sha256", ""))
            actual_hkl_hash = source_hashes["hkl"]
            if not expected_hkl_hash:
                raise ValueError("geometry path is missing its source HKL hash")
            if expected_hkl_hash != actual_hkl_hash:
                raise ValueError("geometry path source HKL hash does not match the current reflection table")
        self.gamma = relativistic_gamma_from_wavelength(data.wavelength) if cfg.gamma is None else float(cfg.gamma)
        self.res_hash = res_hash or "unknown"
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.model["cell"] = (data.cell.a, data.cell.b, data.cell.c, data.cell.alpha, data.cell.beta, data.cell.gamma)
        self.h_lim, self.k_lim, self.l_lim = data.cell.get_hkl_limits(float(self.cfg.ug_d_min))
        requested_ug_storage = str(getattr(self.cfg, "ug_storage_mode", "auto") or "auto").lower()
        row_plan_mode = str(getattr(self.cfg, "row_plan_cache", "off") or "off").lower()
        self.ug_storage_mode = (
            "sparse_active"
            if requested_ug_storage == "auto" and row_plan_mode != "off"
            else ("dense" if requested_ug_storage == "auto" else requested_ug_storage)
        )
        if self.ug_storage_mode == "sparse_active":
            self.ug_cube = None
            self.ug_cache_status = {
                "status": "sparse_active",
                "allocation": {
                    "dense_cube_skipped": True,
                    "reason": "row-plan cache evaluates only its exact active HKL differences",
                },
            }
        else:
            self.ug_cube = self.load_or_build_ug_cube()
        self.pool_hkl, self.pool_ug = self.build_pool()
        self.pool_inv_d2 = self.inv_d2_for_hkls(self.pool_hkl)
        self.thick, self.thick_w = self.thickness_grid()
        self.group_indices = self.make_groups()
        z_valid = np.asarray(self.data.zcal, dtype=float)
        z_valid = z_valid[np.isfinite(z_valid)]
        self.zcal_min = float(np.min(z_valid)) if z_valid.size else 0.0
        self.zcal_max = float(np.max(z_valid)) if z_valid.size else self.zcal_min
        self.zcal_span = max(self.zcal_max - self.zcal_min, 1e-12)
        self.dose_ref_frame = float(getattr(self.cfg, "dose_b_ref_frame", float("nan")))
        if not np.isfinite(self.dose_ref_frame):
            self.dose_ref_frame = float(np.min(z_valid)) if z_valid.size else 0.0

    @staticmethod
    def _nonhermitian_spectral_target_coupling(H):
        """Return a stable eigensystem coupling, or ``None`` when direct expm is safer."""

        H = np.asarray(H, dtype=np.complex128)
        try:
            vals, vecs = eig(H)
            condition = float(np.linalg.cond(vecs))
            if not math.isfinite(condition) or condition > 1e10:
                return None
            inv_vecs = np.linalg.inv(vecs)
            reconstruction = vecs @ np.diag(vals) @ inv_vecs
            relative_error = float(np.linalg.norm(reconstruction - H) / max(np.linalg.norm(H), np.finfo(float).eps))
            if not math.isfinite(relative_error) or relative_error > 1e-8:
                return None
        except np.linalg.LinAlgError:
            return None
        coupling = vecs[1, :] * inv_vecs[:, 0]
        return np.asarray(vals, dtype=np.complex128), np.asarray(coupling, dtype=np.complex128)

    @staticmethod
    def _uniform_scalar_absorption_decomposition(H, rtol=None):
        """Return ``(Hermitian H0, mu)`` when ``H = H0 + i*mu*I`` exactly.

        A uniform imaginary potential commutes with the elastic Hamiltonian.
        Its wave-amplitude effect is the scalar ``exp(-2*pi*mu*t)`` and its
        intensity effect is ``exp(-4*pi*mu*t)``.  Recognizing this case avoids
        a general complex eigensolve without approximating the physics.

        With ``rtol`` the anti-Hermitian part must equal ``i*mu*I`` within
        ``rtol*max|H|`` (single-precision matrices round U(g) and U(-g)
        independently) and the exact Hermitian part ``(H + H^H)/2`` is
        returned, as in the reference ``split_uniform_absorption`` (tests/reference/reference_lib.py).
        """

        matrix = np.asarray(H, dtype=np.complex128)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            return None
        diagonal_imaginary = np.imag(np.diag(matrix))
        if not np.all(np.isfinite(diagonal_imaginary)):
            return None
        if rtol is not None:
            anti = 0.5 * (matrix - matrix.conj().T)
            mu_diag = np.imag(np.diag(anti))
            mu = float(np.mean(mu_diag))
            bound = float(rtol) * max(float(np.max(np.abs(matrix))), 1e-30)
            off_diagonal = anti - np.diag(np.diag(anti))
            if float(np.max(np.abs(off_diagonal), initial=0.0)) > bound or float(np.max(np.abs(mu_diag - mu))) > bound:
                return None
            return 0.5 * (matrix + matrix.conj().T), mu
        mu = float(np.mean(diagonal_imaginary))
        elastic = matrix.copy()
        elastic[np.diag_indices(len(elastic))] -= 1j * mu
        if not np.allclose(elastic, elastic.conj().T, atol=1e-12, rtol=0.0):
            return None
        return elastic, mu

    @staticmethod
    def _nonhermitian_target_intensities(H, thicknesses):
        """Propagate a non-Hermitian matrix, falling back for defective eigensystems."""

        H = np.asarray(H, dtype=np.complex128)
        thicknesses = np.atleast_1d(np.asarray(thicknesses, dtype=float))
        spectral = XDSBlochPort._nonhermitian_spectral_target_coupling(H)

        if spectral is None:
            psi0 = np.zeros(H.shape[0], dtype=np.complex128)
            psi0[0] = 1.0 + 0.0j
            return np.asarray(
                [abs((expm((2j * np.pi * float(t)) * H) @ psi0)[1]) ** 2 for t in thicknesses],
                dtype=float,
            )

        vals, coupling = spectral
        phase = np.exp((2j * np.pi) * vals[:, None] * thicknesses[None, :])
        return np.asarray(np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2, dtype=float)

    def propagate_direct_to_target(self, H, thickness):
        """Return the target-beam intensity after propagating the direct beam through one thickness.

        Beam 0 is the direct beam and beam 1 the target; thickness is in Å and H in Å⁻¹.
        """

        if self.absorption_enabled():
            uniform = self._uniform_scalar_absorption_decomposition(H)
            if uniform is None:
                return float(self._nonhermitian_target_intensities(H, [thickness])[0])
            elastic, mu = uniform
            vals, vecs = hermitian_eigh(elastic)
            coupling = vecs[1, :] * np.conj(vecs[0, :])
            phase = np.exp(2j * np.pi * vals * thickness)
            return float(abs(np.sum(coupling * phase)) ** 2 * math.exp(-4.0 * math.pi * mu * float(thickness)))
        else:
            vals, vecs = hermitian_eigh(H)
            coupling = vecs[1, :] * np.conj(vecs[0, :])
        phase = np.exp(2j * np.pi * vals * thickness)
        return abs(np.sum(coupling * phase)) ** 2

    def propagate_direct_to_target_gamma_cf(self, H, cf):
        """Exact untruncated gamma thickness average of both hands from one Hermitian eigensystem.

        The gamma characteristic function replaces the thickness nodes (grond.bloch.thickness.gamma_cf_point).
        It needs H = H0 + i*mu*I; any other absorptive potential is refused rather than approximated. H0 is the
        exact Hermitian part (H + H^H)/2, as in tests/reference/reference_lib.py matrix_spectrum.
        """

        matrix = np.asarray(H, dtype=np.complex128)
        mu = 0.0
        if self.absorption_enabled():
            uniform = self._uniform_scalar_absorption_decomposition(matrix, rtol=2e-6)
            if uniform is None:
                raise ValueError(
                    "thickness_evaluator cf needs a uniform scalar absorptive potential H = H0 + i*mu*I; this "
                    "Hamiltonian has a non-uniform or off-diagonal absorptive part. Use thickness_evaluator table"
                )
            elastic, mu = uniform
        else:
            elastic = 0.5 * (matrix + matrix.conj().T)
        vals, vecs = hermitian_eigh(elastic)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
        return gamma_cf_point(vals, coupling, cf["shape"], cf["theta"], alpha=4.0 * np.pi * float(mu))

    def _uniform_table_intensities(self, vals, coupling, inverse=False):
        """|amplitude|^2 at the table evaluator's uniform nodes by exact factorised phases; None for other nodes.

        grond.bloch.thickness.uniform_grid_amplitudes equals the direct exp(2*pi*i*lam*t) sum to rounding and makes
        the 4,951-node support table cost one small matrix product per profile point. The legacy quadrature and any
        non-uniform node set keep the direct expression.
        """

        grid = self.thickness_uniform_grid()
        if grid is None:
            return None
        return np.abs(uniform_grid_amplitudes(vals, coupling, *grid, inverse=inverse)[0]) ** 2

    def elastic_spectrum_thickness_average(self, vals, coupling):
        """Thickness average of the target intensity from one elastic eigensystem (eigenvalues, V[1] conj(V[0]))."""

        per_t = self._uniform_table_intensities(vals, coupling)
        if per_t is None:
            phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
            per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
        return float(np.sum(self.thick_w * per_t))

    def propagate_direct_to_target_thickness_average(self, H):
        """Propagate one Bloch matrix and average over the configured thickness quadrature."""

        cf = self.thickness_cf_parameters()
        if cf is not None:
            return self.propagate_direct_to_target_gamma_cf(H, cf)[0]
        if self.absorption_enabled():
            uniform = self._uniform_scalar_absorption_decomposition(H)
            if uniform is None:
                per_t = self._nonhermitian_target_intensities(H, self.thick)
                return float(np.sum(self.thick_w * per_t))
            elastic, mu = uniform
            vals, vecs = hermitian_eigh(elastic)
            coupling = vecs[1, :] * np.conj(vecs[0, :])
            per_t = self._uniform_table_intensities(vals, coupling)
            if per_t is None:
                phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
                per_t = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2
            per_t *= np.exp(-4.0 * np.pi * mu * self.thick)
            return float(np.sum(self.thick_w * per_t))
        vals, vecs = hermitian_eigh(H)
        return self.elastic_spectrum_thickness_average(vals, vecs[1, :] * np.conj(vecs[0, :]))

    def propagate_direct_to_target_thickness_average_pair(self, H):
        """Return original- and inverse-hand intensities from one eigensystem.

        For elastic scattering the inverted-coordinate Hamiltonian is ``H*``.
        With a real incident direct-beam vector, propagation through ``H*`` at
        positive thickness has the same intensity as propagation through ``H``
        at negative thickness.  The two signs therefore share one Hermitian
        eigendecomposition.  A uniform absorptive potential ``H = H0 + i*mu*I``
        keeps the identity exactly: the inverse hand is ``conj(H0) + i*mu*I``
        and both hands carry the same scalar ``exp(-4*pi*mu*t)``.  Any other
        absorptive potential is rejected rather than silently approximated.
        """

        cf = self.thickness_cf_parameters()
        if cf is not None:
            return self.propagate_direct_to_target_gamma_cf(H, cf)
        elastic = np.asarray(H, dtype=np.complex128)
        mu = 0.0
        if self.absorption_enabled():
            # complex64 matrices: compare the anti-Hermitian part to i*mu*I at
            # single-precision tolerance.
            uniform = self._uniform_scalar_absorption_decomposition(elastic, rtol=2e-6)
            if uniform is None:
                raise ValueError(
                    "paired inverse-hand propagation requires a uniform scalar absorptive potential "
                    "H = H0 + i*mu*I; this Hamiltonian has a non-uniform or off-diagonal absorptive part. "
                    "Run explicit coordinate-inverted branches for that absorption model"
                )
            elastic, mu = uniform
        vals, vecs = hermitian_eigh(elastic)
        coupling = vecs[1, :] * np.conj(vecs[0, :])
        attenuation = np.exp(-4.0 * np.pi * mu * self.thick)
        forward = self._uniform_table_intensities(vals, coupling)
        if forward is not None:
            forward = forward * attenuation
            reverse = self._uniform_table_intensities(vals, coupling, inverse=True) * attenuation
        else:
            phase = np.exp((2j * np.pi) * vals[:, None] * self.thick[None, :])
            forward = np.abs(np.sum(coupling[:, None] * phase, axis=0)) ** 2 * attenuation
            reverse = np.abs(np.sum(coupling[:, None] * np.conj(phase), axis=0)) ** 2 * attenuation
        return (
            float(np.sum(self.thick_w * forward)),
            float(np.sum(self.thick_w * reverse)),
        )
