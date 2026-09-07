"""Validated forward-model parameters and the machine-readable numerical contract."""

from __future__ import annotations

import math
from dataclasses import dataclass

from grond.bloch.constants import COMPUTE_BACKENDS, DEFAULT_MAX_SUPPORT_SG, SITE_SYMMETRY_TOLERANCE_FRACTIONAL


@dataclass
class Config:
    n_basis: int = 20
    n_points: int = 20
    sg_window: float = 0.006
    sg_window_mode: str = "fixed"
    sg_window_min: float = 0.003
    sg_window_max: float = 0.012
    sg_window_scale: float = 1.0
    sg_window_frame_floor: float = 0.50
    sg_jitter_mode: str = "none"
    sg_jitter_sigma0: float = 0.0
    sg_jitter_sigma_d_inv: float = 0.0
    sg_jitter_sigma_frame: float = 0.0
    sg_jitter_points: int = 5
    sg_jitter_student_df: float = 4.0
    sg_jitter_width_factor: float = 3.0
    mosaicity_mode: str = "none"
    mosaicity_sigma_deg: float = 0.0
    mosaicity_max_sigma_deg: float = 0.5
    mosaicity_points: int = 3
    mosaicity_support_sigma_deg: float = 0.0
    mosaicity_support_points: int = 5
    mosaicity_support_width_sigma: float = 4.0
    geometry_path_file: str = ""
    profile_basis_mode: str = "center"
    profile_coordinate: str = "target_sg"
    profile_angle_max_half_width_deg: float = 2.0
    profile_quadrature: str = "uniform"
    profile_accumulation: str = "sum"
    topup_d_min: float = 0.8
    max_support_sg: float = DEFAULT_MAX_SUPPORT_SG
    thickness_A: float = 1240.0
    thickness_model: str = "uniform_gradient"
    thickness_grad_frac: float = 0.25
    thickness_sigma_frac: float = 0.0
    thickness_points: int = 7
    gamma: float | None = None
    group_width_deg: float = 3.0
    free_fraction: float = 0.10
    split_seed: int = 42
    variance_a: float = 0.0
    variance_b_corr: float = 0.0
    variance_c_peak: float = 0.0
    frame_offset: float = 0.0
    ug_d_min: float = 0.30
    absorption_mode: str = "none"
    absorption_mu: float = 0.0
    absorption_frac: float = 0.0
    absorption_mu_g2: float = 0.0
    absorption_mu_theta2: float = 0.0
    absorption_tds_b: float = 0.0
    dose_b_mode: str = "none"
    dose_b_per_frame: float = 0.0
    dose_b_ref_frame: float = float("nan")
    compute_backend: str = "scipy"
    jax_batch_size: int = 0
    jax_matrix_bucket_size: int = 0
    bethe_strong_count: int = 48
    bethe_variant: str = "schur_target"
    bethe_promote_denom: float = 0.005
    bethe_denom_eps: float = 1e-5
    matrix_convention: str = "j_minus_i"
    row_plan_cache: str = "off"
    ug_storage_mode: str = "auto"
    paired_inverse_hand: bool = False

    def __post_init__(self):
        integer_minima = {
            "n_basis": (self.n_basis, 2),
            "n_points": (self.n_points, 1),
            "sg_jitter_points": (self.sg_jitter_points, 1),
            "mosaicity_points": (self.mosaicity_points, 1),
            "mosaicity_support_points": (self.mosaicity_support_points, 2),
            "thickness_points": (self.thickness_points, 1),
            "jax_batch_size": (self.jax_batch_size, 0),
            "jax_matrix_bucket_size": (self.jax_matrix_bucket_size, 0),
            "bethe_strong_count": (self.bethe_strong_count, 2),
        }
        for name, (value, minimum) in integer_minima.items():
            if isinstance(value, bool) or int(value) != value or int(value) < minimum:
                raise ValueError(f"{name} must be an integer of at least {minimum}")

        ug_storage_mode = str(self.ug_storage_mode or "auto").lower()
        if ug_storage_mode not in {"auto", "dense", "sparse_active"}:
            raise ValueError("ug_storage_mode must be auto, dense, or sparse_active")

        positive_fields = {
            "sg_window": self.sg_window,
            "sg_window_min": self.sg_window_min,
            "sg_window_max": self.sg_window_max,
            "sg_window_scale": self.sg_window_scale,
            "sg_jitter_width_factor": self.sg_jitter_width_factor,
            "topup_d_min": self.topup_d_min,
            "max_support_sg": self.max_support_sg,
            "mosaicity_max_sigma_deg": self.mosaicity_max_sigma_deg,
            "mosaicity_support_width_sigma": self.mosaicity_support_width_sigma,
            "profile_angle_max_half_width_deg": self.profile_angle_max_half_width_deg,
            "thickness_A": self.thickness_A,
            "group_width_deg": self.group_width_deg,
            "ug_d_min": self.ug_d_min,
            "bethe_denom_eps": self.bethe_denom_eps,
        }
        for name, value in positive_fields.items():
            if not math.isfinite(float(value)) or float(value) <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if self.gamma is not None and (not math.isfinite(float(self.gamma)) or float(self.gamma) <= 1.0):
            raise ValueError("gamma must be greater than 1 when explicitly supplied")

        nonnegative_fields = {
            "sg_window_frame_floor": self.sg_window_frame_floor,
            "sg_jitter_sigma0": self.sg_jitter_sigma0,
            "sg_jitter_sigma_d_inv": self.sg_jitter_sigma_d_inv,
            "sg_jitter_sigma_frame": self.sg_jitter_sigma_frame,
            "mosaicity_sigma_deg": self.mosaicity_sigma_deg,
            "mosaicity_support_sigma_deg": self.mosaicity_support_sigma_deg,
            "thickness_grad_frac": self.thickness_grad_frac,
            "thickness_sigma_frac": self.thickness_sigma_frac,
            "variance_a": self.variance_a,
            "variance_b_corr": self.variance_b_corr,
            "variance_c_peak": self.variance_c_peak,
            "absorption_mu": self.absorption_mu,
            "absorption_frac": self.absorption_frac,
            "absorption_mu_g2": self.absorption_mu_g2,
            "absorption_mu_theta2": self.absorption_mu_theta2,
            "absorption_tds_b": self.absorption_tds_b,
            "dose_b_per_frame": self.dose_b_per_frame,
            "bethe_promote_denom": self.bethe_promote_denom,
        }
        for name, value in nonnegative_fields.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(self.mosaicity_sigma_deg) > float(self.mosaicity_max_sigma_deg):
            raise ValueError("mosaicity_sigma_deg exceeds the explicit mosaicity_max_sigma_deg bound")
        if float(self.mosaicity_support_sigma_deg) > float(self.mosaicity_max_sigma_deg):
            raise ValueError("mosaicity_support_sigma_deg exceeds the explicit mosaicity_max_sigma_deg bound")
        if (
            str(self.mosaicity_mode).lower() == "gaussian_cone"
            and float(self.mosaicity_sigma_deg) > float(self.mosaicity_support_sigma_deg) + 1e-15
        ):
            raise ValueError(
                "gaussian_cone requires mosaicity_support_sigma_deg to be at least "
                "mosaicity_sigma_deg so candidate widths share one enclosing beam basis"
            )
        if (
            str(self.mosaicity_mode).lower() == "gaussian_cone"
            and str(self.profile_coordinate).lower() != "direct_angle"
        ):
            raise ValueError(
                "gaussian_cone currently requires profile_coordinate=direct_angle; "
                "this keeps the physical scan interval explicit and avoids extra Sg roots"
            )

        choices = {
            "sg_window_mode": (
                self.sg_window_mode,
                {"fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak", "dials_bbox"},
            ),
            "sg_jitter_mode": (self.sg_jitter_mode, {"none", "gaussian", "student_t"}),
            "mosaicity_mode": (
                self.mosaicity_mode,
                {"none", "gaussian_omega", "gaussian_cone"},
            ),
            "profile_coordinate": (
                self.profile_coordinate,
                {"target_sg", "direct_angle"},
            ),
            "profile_basis_mode": (
                self.profile_basis_mode,
                {"center", "common_profile"},
            ),
            "profile_quadrature": (
                self.profile_quadrature,
                {"uniform", "trapezoid", "gauss_legendre", "legendre"},
            ),
            "profile_accumulation": (self.profile_accumulation, {"sum", "mean"}),
            "thickness_model": (
                self.thickness_model,
                {"single", "uniform_gradient", "gaussian", "gaussian_gradient"},
            ),
            "absorption_mode": (
                self.absorption_mode,
                {
                    "none",
                    "mean",
                    "beam_dependent",
                    "offdiag_fraction",
                    "offdiag_tds",
                    "midgley_like",
                    "midgley",
                },
            ),
            "dose_b_mode": (self.dose_b_mode, {"none", "linear_frame"}),
            "compute_backend": (
                self.compute_backend,
                set(COMPUTE_BACKENDS) | {"bethe", "scipy_bethe_promoted"},
            ),
            "bethe_variant": (
                self.bethe_variant,
                {"truncate", "schur0", "schur_target", "schur_mean"},
            ),
            "matrix_convention": (self.matrix_convention, {"j_minus_i", "i_minus_j"}),
            "row_plan_cache": (
                self.row_plan_cache,
                {"off", "exact", "readwrite", "rebuild"},
            ),
        }
        for name, (value, allowed) in choices.items():
            if str(value).lower() not in allowed:
                raise ValueError(f"unsupported {name} {value!r}; expected one of {sorted(allowed)}")

        if self.sg_window_min > self.sg_window_max:
            raise ValueError("sg_window_min must not exceed sg_window_max")
        fixed_width = float(self.sg_window) * float(self.sg_window_scale)
        if self.sg_window_mode == "fixed" and not self.sg_window_min <= fixed_width <= self.sg_window_max:
            raise ValueError(
                "fixed sg_window*sg_window_scale lies outside sg_window_min/sg_window_max; "
                "change the bounds explicitly instead of accepting silent clipping"
            )
        if self.thickness_grad_frac > 1.0:
            raise ValueError("thickness_grad_frac must not exceed 1")
        if not math.isfinite(float(self.sg_jitter_student_df)) or self.sg_jitter_student_df <= 1.0:
            raise ValueError("sg_jitter_student_df must be greater than 1")
        if not 0.0 < float(self.free_fraction) < 1.0:
            raise ValueError("free_fraction must be between 0 and 1")
        if not math.isfinite(float(self.frame_offset)):
            raise ValueError("frame_offset must be finite")
        if not (math.isnan(float(self.dose_b_ref_frame)) or math.isfinite(float(self.dose_b_ref_frame))):
            raise ValueError("dose_b_ref_frame must be finite or NaN")
        if bool(self.paired_inverse_hand):
            absorptive_values = (
                self.absorption_mu,
                self.absorption_frac,
                self.absorption_mu_g2,
                self.absorption_mu_theta2,
                self.absorption_tds_b,
            )
            if str(self.absorption_mode).lower() != "none" or any(
                abs(float(value)) > 0.0 for value in absorptive_values
            ):
                raise ValueError("paired_inverse_hand requires absorption_mode=none and zero absorptive coefficients")
            if str(self.compute_backend).lower() not in {"scipy", "numpy_scipy_cpu"}:
                raise ValueError("paired_inverse_hand currently requires the full SciPy Hermitian backend")


def frozen_kernel_contract(cfg: Config | None = None):
    """Machine-readable record of the audited Bloch convention.

    This is intentionally descriptive rather than configurable.  The XDS raw
    dynrow work should test observation contracts, not silently mutate the
    production Bloch kernel.
    """
    convention = str(getattr(cfg, "matrix_convention", "j_minus_i") if cfg is not None else "j_minus_i")
    offdiag_expr = "U(h_j - h_i) / (2*k)" if convention == "j_minus_i" else "U(h_i - h_j) / (2*k)"
    offdiag_sign = "basis[j] - basis[i]" if convention == "j_minus_i" else "basis[i] - basis[j]"
    return {
        "name": "audited_signed_row_simple_hermitian",
        "structure_factors": {
            "internal_occupancy": "physical site occupancy; parsed SHELX SOF multiplied by the site-symmetry order; coreCIF occupancy unchanged",
            "site_rounding_tolerance_fractional": SITE_SYMMETRY_TOLERANCE_FRACTIONAL,
            "displacements": "crystallographic Uij or Uiso in A^2; CIF Bij/Biso converted by division by 8*pi^2",
        },
        "geometry": {
            "g(alpha)": "S @ R(alpha) @ F @ A(alpha) @ h (S=F=identity for ordinary XDS)",
            "Sg": "(k^2 - |s0 + g(alpha)|^2) / (2*k)",
            "profile_mode": "target_sg_to_angle or explicit direct-angle integration",
            "profile_coordinate": (None if cfg is None else str(getattr(cfg, "profile_coordinate", "target_sg"))),
            "profile_quadrature": None if cfg is None else str(getattr(cfg, "profile_quadrature", "uniform")),
            "profile_accumulation": None if cfg is None else str(getattr(cfg, "profile_accumulation", "sum")),
            "profile_weight_mode": "rocking-profile samples are summed by default; legacy normalized averaging is available as profile_accumulation=mean",
            "profile_coordinate_jacobian": "target-Sg quadrature weights include relative abs(domega/dSg) so exposure sampling is uniform in angle",
            "profile_width_mode": "fixed unless cfg.sg_window_mode explicitly requests an auditable XDS metadata-derived per-row width",
            "orientation_distribution": (
                "optional target-Sg jitter, projected one-axis mosaic convolution, or "
                "an incoherent two-axis Gaussian orientation cone with one beam basis "
                "fixed over the widest declared cone; all disabled by default"
            ),
            "geometry_path": (
                "optional fixed geometry-only Chebyshev slew and two-axis tilt contract; "
                "never fitted from intensities inside the Bloch calculation"
            ),
            "basis_sg_recompute": "all basis Sg recomputed at every solved target-Sg angle",
            "bragg_root_search": "strict local angular bound around the recorded row geometry",
            "base_profile_root_search": "exact target-Sg continuation through strict local steps on one derivative-sign-certified Ewald branch",
            "jitter_root_search": "active optional jitter targets retain the strict single local angular bound and reject the candidate if unreachable",
        },
        "hamiltonian": {
            "matrix_convention": convention,
            "diagonal": "H_ii = S_i",
            "offdiagonal": offdiag_expr,
            "offdiagonal_sign": offdiag_sign,
            "default_convention_note": "j_minus_i is the production signed-row convention; i_minus_j is an explicit raw-forward-model diagnostic.",
            "dose_b": "optional frame-dependent isotropic Debye-Waller attenuation U_g -> U_g exp(-DeltaB/(4d_g^2)); default none",
            "hermitian_fill": "H_ji = conj(H_ij)",
            "propagation": "psi(t) = exp(2*pi*i*H*t) psi(0)",
            "paired_inverse_hand": (
                None
                if cfg is None
                else {
                    "enabled": bool(getattr(cfg, "paired_inverse_hand", False)),
                    "method": "I_inverse(t)=I_original(-t) with one elastic Hermitian eigensystem",
                    "absorption_allowed": False,
                }
            ),
            "absorption": "optional non-Hermitian optical-potential diagnostics; default none. Modes are mean, beam_dependent, offdiag_fraction, offdiag_tds, and midgley_like.",
            "absorption_passivity": "the static loss operator (H-H^dagger)/(2i) must be positive semidefinite within numerical tolerance; nonpassive grid candidates are rejected",
            "rejected_nonhermitian_modes": [
                "row-specific imaginary diagonal terms",
                "EXTI-like extinction inside the raw Bloch matrix",
                "per-row fitted absorptive form factors",
            ],
            "solver": "scipy.linalg.eigh/eig by default; optional experimental scipy expm/expm_multiply, promoted Bethe/Schur, and JAX batched paths only when explicitly requested and probed",
        },
        "basis": {
            "mode": "direct_target_topup",
            "profile_basis_mode": None if cfg is None else str(cfg.profile_basis_mode),
            "first_beams": ["direct (0,0,0)", "target signed hkl"],
            "topup_d_min_A": None if cfg is None else float(cfg.topup_d_min),
            "max_support_sg_inv_A": None if cfg is None else float(cfg.max_support_sg),
            "score": "abs(U_g)^2 / (abs(Sg) + 1e-5)",
            "rejected_modes": ["row contract measured-intensity ranking", "frame_measured_plus_topup"],
        },
        "row_identity": {
            "signed_hkl_required": True,
            "asu_mapping": False,
            "friedel_merge": False,
            "p222_canonicalization": False,
        },
    }
