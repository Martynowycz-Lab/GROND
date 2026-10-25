"""Validated forward-model parameters and the machine-readable numerical contract."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field

from grond.bloch.constants import DEFAULT_MAX_SUPPORT_SG, SITE_SYMMETRY_TOLERANCE_FRACTIONAL
from grond.bloch.thickness import (
    DEFAULT_TABLE_DT_A,
    DEFAULT_THICKNESS_MAX_A,
    DEFAULT_THICKNESS_MIN_A,
    THICKNESS_EVALUATORS,
    config_table_nodes,
    config_thickness_contract,
    resolved_config_evaluator,
    validate_support,
)


def _default_double_precision() -> bool:
    """Default numerical precision for the Bloch engine.

    complex64 is the default. Its intensity error depends on the matrix,
    thickness and propagation method; no universal accuracy bound is implied.
    Set GROND_PRECISION=float64
    (or "double") to make complex128 the default everywhere, e.g. for physics
    audits and the unit-test suite. An explicit Config(double_precision=...)
    always wins.
    """

    value = os.environ.get("GROND_PRECISION", "float32").strip().lower()
    if value in {"float64", "double", "complex128", "fp64"}:
        return True
    if value in {"float32", "single", "complex64", "fp32", ""}:
        return False
    raise ValueError(f"GROND_PRECISION must be float32 or float64, got {value!r}")


def _default_centring_coset() -> bool:
    """Default of Config.basis_centring_coset: GROND_BASIS_CENTRING_COSET=1 (or true/on/yes) turns it on."""

    value = os.environ.get("GROND_BASIS_CENTRING_COSET", "0").strip().lower()
    if value in {"1", "true", "on", "yes"}:
        return True
    if value in {"0", "false", "off", "no", ""}:
        return False
    raise ValueError(f"GROND_BASIS_CENTRING_COSET must be 0 or 1, got {value!r}")


def resolve_double_precision(double_flag: bool | None = None) -> bool:
    """Prediction precision of the row engines: complex128 if --double is given, else from GROND_PRECISION."""

    return True if double_flag else _default_double_precision()


def prediction_precision_record(double_flag: bool | None = None) -> dict:
    """The resolved prediction precision (resolve_double_precision) and what set it, for the run summary.

    hamiltonian_dtype is the dtype that double_precision selects for the Bloch matrices (bloch/hamiltonian.py,
    bloch/row_cache.py).
    """

    env = os.environ.get("GROND_PRECISION")
    double = resolve_double_precision(double_flag)
    source = "--double" if double_flag else ("GROND_PRECISION" if env is not None and env.strip() else "default")
    return {
        "double_precision": double,
        "hamiltonian_dtype": "complex128" if double else "complex64",
        "source": source,
        "double_flag": bool(double_flag),
        "GROND_PRECISION": env,
    }


@dataclass
class Config:
    """Forward-model settings of the Bloch engine, validated when the object is created.

    Lengths are in Å (thickness_*_A, topup_d_min, ug_d_min), excitation-error widths S_g in Å⁻¹ (sg_window*,
    max_support_sg), angles in degrees and frame_offset in frames. double_precision defaults from GROND_PRECISION.
    """

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
    geometry_path_file: str = ""
    profile_quadrature: str = "uniform"
    profile_accumulation: str = "sum"
    topup_d_min: float = 0.8
    max_support_sg: float = DEFAULT_MAX_SUPPORT_SG
    thickness_A: float = 1240.0
    thickness_model: str = "uniform_gradient"
    thickness_grad_frac: float = 0.25
    thickness_points: int = 7
    gamma: float | None = None
    group_width_deg: float = 3.0
    free_fraction: float = 0.10
    variance_a: float = 0.0
    variance_b_corr: float = 0.0
    variance_c_peak: float = 0.0
    frame_offset: float = 0.0
    ug_d_min: float = 0.30
    absorption_mode: str = "none"
    absorption_mu: float = 0.0
    dose_b_mode: str = "none"
    dose_b_per_frame: float = 0.0
    dose_b_ref_frame: float = float("nan")
    matrix_convention: str = "j_minus_i"
    row_plan_cache: str = "off"
    ug_storage_mode: str = "auto"
    paired_inverse_hand: bool = False
    double_precision: bool = field(default_factory=_default_double_precision)
    # Physical thickness distributions on a support (grond.bloch.thickness). The default "quadrature" keeps the
    # legacy node quadrature of uniform_gradient; "table", "cf" and "auto" evaluate uniform_gradient or
    # gamma (mean thickness_A, shape thickness_gamma_shape) on [thickness_min_A, thickness_max_A].
    thickness_gamma_shape: float = 1.0
    thickness_evaluator: str = "quadrature"
    thickness_min_A: float = DEFAULT_THICKNESS_MIN_A
    thickness_max_A: float = DEFAULT_THICKNESS_MAX_A
    thickness_table_dt_A: float = DEFAULT_TABLE_DT_A
    # Keep only top-up beams in the direct beam's lattice-centring coset (g . t integer for every centring
    # translation t of the model). Beams of other cosets have no potential coupling to it (U of their difference
    # is a centring absence), so the propagated amplitudes are unchanged while the matrices shrink. Default from
    # GROND_BASIS_CENTRING_COSET (off: legacy pool), so that every stage of a run inherits one setting.
    basis_centring_coset: bool = field(default_factory=_default_centring_coset)

    def __post_init__(self):
        """Raise ValueError for an out-of-range value or an unsupported combination of modes; fields are not changed."""

        integer_minima = {
            "n_basis": (self.n_basis, 2),
            "n_points": (self.n_points, 1),
            "sg_jitter_points": (self.sg_jitter_points, 1),
            "mosaicity_points": (self.mosaicity_points, 1),
            "thickness_points": (self.thickness_points, 1),
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
            "thickness_A": self.thickness_A,
            "group_width_deg": self.group_width_deg,
            "ug_d_min": self.ug_d_min,
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
            "thickness_grad_frac": self.thickness_grad_frac,
            "variance_a": self.variance_a,
            "variance_b_corr": self.variance_b_corr,
            "variance_c_peak": self.variance_c_peak,
            "absorption_mu": self.absorption_mu,
            "dose_b_per_frame": self.dose_b_per_frame,
        }
        for name, value in nonnegative_fields.items():
            if not math.isfinite(float(value)) or float(value) < 0.0:
                raise ValueError(f"{name} must be finite and nonnegative")
        if float(self.mosaicity_sigma_deg) > float(self.mosaicity_max_sigma_deg):
            raise ValueError("mosaicity_sigma_deg exceeds the explicit mosaicity_max_sigma_deg bound")

        choices = {
            "sg_window_mode": (self.sg_window_mode, {"fixed", "dials_bbox"}),
            "sg_jitter_mode": (self.sg_jitter_mode, {"none", "gaussian", "student_t"}),
            "mosaicity_mode": (self.mosaicity_mode, {"none", "gaussian_omega", "transverse_target"}),
            "profile_quadrature": (self.profile_quadrature, {"uniform"}),
            "profile_accumulation": (self.profile_accumulation, {"sum"}),
            "thickness_model": (self.thickness_model, {"uniform_gradient", "gamma"}),
            "thickness_evaluator": (self.thickness_evaluator, set(THICKNESS_EVALUATORS)),
            "absorption_mode": (self.absorption_mode, {"none", "mean", "midgley_like", "midgley"}),
            "dose_b_mode": (self.dose_b_mode, {"none", "linear_frame"}),
            "matrix_convention": (self.matrix_convention, {"j_minus_i"}),
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
        if not math.isfinite(float(self.thickness_gamma_shape)) or float(self.thickness_gamma_shape) <= 0.0:
            raise ValueError("thickness_gamma_shape must be finite and positive")
        if str(self.thickness_evaluator).lower() != "quadrature":
            validate_support(self.thickness_min_A, self.thickness_max_A, self.thickness_table_dt_A)
        resolved_thickness = resolved_config_evaluator(self)
        if resolved_thickness == "table":
            config_table_nodes(self)  # fails closed when the distribution has no mass on the support
        if not math.isfinite(float(self.sg_jitter_student_df)) or self.sg_jitter_student_df <= 1.0:
            raise ValueError("sg_jitter_student_df must be greater than 1")
        if not 0.0 < float(self.free_fraction) < 1.0:
            raise ValueError("free_fraction must be between 0 and 1")
        if not math.isfinite(float(self.frame_offset)):
            raise ValueError("frame_offset must be finite")
        if not (math.isnan(float(self.dose_b_ref_frame)) or math.isfinite(float(self.dose_b_ref_frame))):
            raise ValueError("dose_b_ref_frame must be finite or NaN")
        # The exact hand pair holds for H = H0 + i*mu*I: no absorption, or a uniform diagonal loss (mean/midgley
        # modes add i*mu to every beam).
        if (
            bool(self.paired_inverse_hand)
            and str(self.absorption_mode).lower() == "none"
            and abs(float(self.absorption_mu)) > 0.0
        ):
            raise ValueError(
                "paired_inverse_hand requires absorption_mode=none with absorption_mu 0, or a uniform absorption "
                "mode (mean, midgley_like, midgley) with absorption_mu >= 0"
            )


def frozen_kernel_contract(cfg: Config | None = None):
    """Machine-readable record of the audited Bloch convention.

    This is intentionally descriptive rather than configurable: work on
    observation contracts must not silently mutate the production Bloch kernel.
    """
    convention = str(getattr(cfg, "matrix_convention", "j_minus_i") if cfg is not None else "j_minus_i")
    contract = {
        "name": "audited_signed_row_simple_hermitian",
        "structure_factors": {
            "internal_occupancy": "physical site occupancy; parsed SHELX SOF multiplied by the site-symmetry order; coreCIF occupancy unchanged",
            "site_rounding_tolerance_fractional": SITE_SYMMETRY_TOLERANCE_FRACTIONAL,
            "displacements": "crystallographic Uij or Uiso in A^2; CIF Bij/Biso converted by division by 8*pi^2",
        },
        "geometry": {
            "g(alpha)": "S @ R(alpha) @ F @ A(alpha) @ h (S=F=identity for ordinary XDS)",
            "Sg": "(k^2 - |s0 + g(alpha)|^2) / (2*k)",
            "profile_mode": "target_sg_to_angle",
            "profile_quadrature": None if cfg is None else str(getattr(cfg, "profile_quadrature", "uniform")),
            "profile_accumulation": None if cfg is None else str(getattr(cfg, "profile_accumulation", "sum")),
            "profile_weight_mode": "rocking-profile samples are summed",
            "profile_coordinate_jacobian": "target-Sg quadrature weights include relative abs(domega/dSg) so exposure sampling is uniform in angle",
            "profile_width_mode": "fixed unless cfg.sg_window_mode explicitly requests an auditable XDS metadata-derived per-row width",
            "orientation_distribution": (
                "optional target-Sg jitter, projected one-axis mosaic convolution, or "
                "the re-centred transverse mosaic spread; all disabled by default"
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
            "offdiagonal": "U(h_j - h_i) / (2*k)",
            "offdiagonal_sign": "basis[j] - basis[i]",
            "default_convention_note": "j_minus_i is the production signed-row convention.",
            "dose_b": "optional frame-dependent isotropic Debye-Waller attenuation U_g -> U_g exp(-DeltaB/(4d_g^2)); default none",
            "hermitian_fill": "H_ji = conj(H_ij)",
            "propagation": "psi(t) = exp(2*pi*i*H*t) psi(0)",
            "paired_inverse_hand": (
                None
                if cfg is None
                else {
                    "enabled": bool(getattr(cfg, "paired_inverse_hand", False)),
                    "method": "I_inverse(t)=I_original(-t) with one elastic Hermitian eigensystem",
                    # uniform absorption H = H0 + i*mu*I splits exactly: eigh(H0) and exp(-4*pi*mu*t) for both hands
                    "absorption_allowed": "uniform_scalar_only",
                }
            ),
            "absorption": "optional uniform imaginary potential i*mu*I; default none. Modes are mean and midgley_like.",
            "absorption_passivity": "the static loss operator (H-H^dagger)/(2i) must be positive semidefinite within numerical tolerance; nonpassive grid candidates are rejected",
            "rejected_nonhermitian_modes": [
                "row-specific imaginary diagonal terms",
                "EXTI-like extinction inside the raw Bloch matrix",
                "per-row fitted absorptive form factors",
            ],
            "solver": "scipy.linalg.eigh/eig; scipy expm for an ill-conditioned non-Hermitian eigensystem",
        },
        "basis": {
            "mode": "direct_target_topup",
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
    if cfg is not None and str(getattr(cfg, "thickness_evaluator", "quadrature")).lower() != "quadrature":
        # Recorded only for the support evaluators, so legacy contracts are unchanged.
        contract["thickness_distribution"] = config_thickness_contract(cfg)
    return contract
