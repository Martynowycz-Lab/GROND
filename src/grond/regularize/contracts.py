"""Nuisance contracts and discrete-grid parsing. These values define existing experiments; this module does not choose new scientific defaults."""

from __future__ import annotations

from dataclasses import dataclass

TARGET_MODES = ("iobs", "iobs_over_rlp", "iobs_times_peak", "recorded_proxy")


TARGET_MODE_LABELS = {
    "iobs": "IOBS",
    "iobs_over_rlp": "IOBS/RLP",
    "iobs_times_peak": "IOBS*PEAK/100",
    "recorded_proxy": "IOBS*PEAK/(100*RLP)",
}


@dataclass(frozen=True)
class Nuisance:
    thickness_A: float
    thickness_grad_frac: float
    frame_offset: float
    group_width_deg: float
    detector_scale_bins: int
    sigma_a: float
    sigma_b_corr: float
    sigma_c_peak: float
    thickness_model: str = "uniform_gradient"
    thickness_sigma_frac: float = 0.0
    sg_jitter_mode: str = "none"
    sg_jitter_sigma0: float = 0.0
    sg_jitter_sigma_d_inv: float = 0.0
    sg_jitter_sigma_frame: float = 0.0
    sg_jitter_points: int = 1
    sg_jitter_student_df: float = 4.0
    sg_jitter_width_factor: float = 3.0
    mosaicity_mode: str = "none"
    mosaicity_sigma_deg: float = 0.0
    mosaicity_points: int = 1
    mosaicity_support_sigma_deg: float = 0.0
    mosaicity_support_points: int = 5
    mosaicity_support_width_sigma: float = 4.0
    geometry_path_file: str = ""
    absorption_mode: str = "none"
    absorption_mu: float = 0.0
    absorption_frac: float = 0.0
    absorption_mu_g2: float = 0.0
    absorption_mu_theta2: float = 0.0
    absorption_tds_b: float = 0.0
    dose_b_mode: str = "none"
    dose_b_per_frame: float = 0.0
    dose_b_ref_frame: float = float("nan")
    matrix_convention: str = "j_minus_i"
    row_resolution_b: float = 0.0
    native_wght_a: float = float("nan")
    native_wght_b: float = float("nan")
    scale_smoothness: float = 0.0
    detector_scale_ridge: float = 0.05
    detector_scale_smoothness: float = 0.10
    detector_scale_min_rows: int = 20


def kernel_variance_terms(nuisance: dict) -> dict[str, float]:
    """Translate fitted nuisance sigma names to the Bloch ``Config`` names."""

    return {
        "variance_a": float(nuisance["sigma_a"]),
        "variance_b_corr": float(nuisance["sigma_b_corr"]),
        "variance_c_peak": float(nuisance["sigma_c_peak"]),
    }


def parse_float_grid(text: str) -> list[float]:
    return [float(x) for x in str(text).replace(",", " ").split() if x.strip()]


def parse_int_grid(text: str) -> list[int]:
    return [int(round(float(x))) for x in str(text).replace(",", " ").split() if x.strip()]


def absorption_parameter_grid(args, mode: str) -> list[dict[str, float | str]]:
    valid_modes = {"none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"}
    if mode not in valid_modes:
        raise ValueError(f"unknown absorption mode {mode!r}")
    mu_values = parse_float_grid(args.absorption_mu_grid) if mode in {"mean", "beam_dependent", "midgley_like"} else [0.0]
    frac_values = parse_float_grid(args.absorption_frac_grid) if mode in {"offdiag_fraction", "offdiag_tds", "midgley_like"} else [0.0]
    mu_g2_values = parse_float_grid(args.absorption_mu_g2_grid) if mode == "beam_dependent" else [0.0]
    mu_theta2_values = parse_float_grid(args.absorption_mu_theta2_grid) if mode == "beam_dependent" else [0.0]
    tds_b_values = parse_float_grid(args.absorption_tds_b_grid) if mode == "offdiag_tds" else [0.0]
    out = []
    for mu in mu_values:
        for frac in frac_values:
            for mu_g2 in mu_g2_values:
                for mu_theta2 in mu_theta2_values:
                    for tds_b in tds_b_values:
                        out.append(
                            {
                                "absorption_mode": mode,
                                "absorption_mu": float(mu),
                                "absorption_frac": float(frac),
                                "absorption_mu_g2": float(mu_g2),
                                "absorption_mu_theta2": float(mu_theta2),
                                "absorption_tds_b": float(tds_b),
                            }
                        )
    return out


AUTO_EXACT_ROW_PLAN_MIN_REUSE = 16


def estimated_row_plan_reuse_per_geometry_key(
    args,
    thickness_models: list[str],
    absorption_modes: list[str],
) -> int:
    """Count prediction candidates that can share one immutable row plan.

    Thickness distributions and absorption coefficients change the propagated
    matrix but not the row basis, scan geometry, or profile quadrature stored in
    the cache.  Frame offsets, jitter, mosaicity, dose, and geometry-path terms
    do change the cache key and are deliberately excluded from this count.
    """

    thickness_candidates = 0
    for _thickness in parse_float_grid(args.thickness_grid):
        for model in thickness_models:
            if model not in {"single", "uniform_gradient", "gaussian", "gaussian_gradient"}:
                raise ValueError(f"unknown thickness_model {model!r}")
            grad_values = (
                parse_float_grid(args.thickness_grad_grid)
                if model in {"uniform_gradient", "gaussian_gradient"}
                else [0.0]
            )
            sigma_values = (
                parse_float_grid(args.thickness_sigma_frac_grid)
                if model in {"gaussian", "gaussian_gradient"}
                else [0.0]
            )
            thickness_candidates += len(grad_values) * len(sigma_values)
    absorption_candidates = sum(
        len(absorption_parameter_grid(args, mode)) for mode in absorption_modes
    )
    return int(thickness_candidates * absorption_candidates)


def resolve_fixed_model_row_plan_cache(
    requested_mode: str,
    *,
    estimated_reuse: int,
    minimum_reuse: int = AUTO_EXACT_ROW_PLAN_MIN_REUSE,
) -> dict[str, int | str]:
    """Resolve automatic caching only when its serial build can amortize."""

    requested = str(requested_mode or "off").lower()
    resolved = (
        "exact" if int(estimated_reuse) >= int(minimum_reuse) else "off"
    ) if requested == "auto" else requested
    if resolved not in {"off", "exact", "readwrite", "rebuild"}:
        raise ValueError(
            f"unknown row_plan_cache={requested!r}; expected auto, off, exact, readwrite, or rebuild"
        )
    return {
        "requested": requested,
        "resolved": resolved,
        "estimated_reuse_per_geometry_key": int(estimated_reuse),
        "auto_exact_minimum_reuse": int(minimum_reuse),
    }
