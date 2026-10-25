"""Nuisance contracts and discrete-grid parsing. These values define existing experiments; this module does not choose new scientific defaults."""

from __future__ import annotations

import math
from dataclasses import dataclass

TARGET_MODES = ("iobs", "iobs_over_rlp", "iobs_times_peak", "recorded_proxy")


@dataclass(frozen=True)
class Nuisance:
    """Nuisance values of one native trial; each trial record stores them as its nuisance dict.

    Thicknesses are in A and angles in degrees. sigma_a, sigma_b_corr and sigma_c_peak are the error-model terms of
    sigma_eff. A NaN native_wght_a or native_wght_b means the stage's SHELXL weights are used. group_width_deg sets
    the frame groups of the row table. detector_scale_bins, scale_smoothness and the detector_scale_* fields belong
    to the removed per-frame and detector scale models; they stay in the record at 0 and their defaults.
    """

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
    # Support thickness distributions (grond.bloch.thickness); "quadrature" is the legacy node quadrature.
    thickness_gamma_shape: float = 1.0
    thickness_evaluator: str = "quadrature"
    thickness_min_A: float = 100.0
    thickness_max_A: float = 10000.0
    thickness_table_dt_A: float = 2.0


THICKNESS_MODELS = ("single", "uniform_gradient", "gaussian", "gaussian_gradient", "gamma")


def thickness_shape_grids(args, thickness_model: str) -> tuple[list[float], list[float], list[float]]:
    """(grad, gaussian sigma fraction, gamma shape) grids of one thickness model; unused axes are one value."""

    if thickness_model not in THICKNESS_MODELS:
        raise ValueError(f"unknown thickness_model {thickness_model!r}")
    grad_values = parse_float_grid(args.thickness_grad_grid) if thickness_model in {"uniform_gradient", "gaussian_gradient"} else [0.0]
    sigma_values = parse_float_grid(args.thickness_sigma_frac_grid) if thickness_model in {"gaussian", "gaussian_gradient"} else [0.0]
    shape_values = (
        parse_float_grid(getattr(args, "thickness_gamma_shape_grid", "1.0")) if thickness_model == "gamma" else [1.0]
    )
    return grad_values, sigma_values, shape_values


def thickness_support_fields(args) -> dict[str, float | str]:
    """Nuisance/Config fields of the thickness evaluator and its physical support (nm options in A)."""

    return {
        "thickness_evaluator": str(getattr(args, "thickness_evaluator", "quadrature")),
        "thickness_min_A": 10.0 * float(getattr(args, "thickness_min_nm", 10.0)),
        "thickness_max_A": 10.0 * float(getattr(args, "thickness_max_nm", 1000.0)),
        "thickness_table_dt_A": float(getattr(args, "thickness_table_dt", 2.0)),
    }


def thickness_distribution_options(nuisance: dict) -> list[str]:
    """Native command options that pin a support thickness distribution from a nuisance record.

    Empty for the legacy quadrature, so commands of legacy stages stay byte-identical.
    """

    evaluator = str(nuisance.get("thickness_evaluator", "quadrature") or "quadrature")
    if evaluator == "quadrature" and str(nuisance.get("thickness_model", "")) != "gamma":
        return []
    return [
        "--thickness-gamma-shape-grid",
        f"{float(nuisance.get('thickness_gamma_shape', 1.0)):.17g}",
        "--thickness-evaluator",
        evaluator,
        "--thickness-min-nm",
        f"{float(nuisance.get('thickness_min_A', 100.0)) / 10.0:.17g}",
        "--thickness-max-nm",
        f"{float(nuisance.get('thickness_max_A', 10000.0)) / 10.0:.17g}",
        "--thickness-table-dt",
        f"{float(nuisance.get('thickness_table_dt_A', 2.0)):.17g}",
    ]


def kernel_variance_terms(nuisance: dict) -> dict[str, float]:
    """Translate fitted nuisance sigma names to the Bloch ``Config`` names."""

    return {
        "variance_a": float(nuisance["sigma_a"]),
        "variance_b_corr": float(nuisance["sigma_b_corr"]),
        "variance_c_peak": float(nuisance["sigma_c_peak"]),
    }


def parse_float_grid(text: str) -> list[float]:
    """Parse a comma- or space-separated list of floats."""

    return [float(x) for x in str(text).replace(",", " ").split() if x.strip()]


def parse_int_grid(text: str) -> list[int]:
    """Parse a comma- or space-separated list of numbers, rounded to integers."""

    return [int(round(float(x))) for x in str(text).replace(",", " ").split() if x.strip()]


@dataclass(frozen=True)
class ScaleModel:
    """P-spline frame scales fitted inside every native trial.

    Like ``scale_fit_split`` this is a stage evaluation contract, not a Nuisance
    field: a smooth cubic P-spline in scan angle (tests/reference/reference_lib.py
    fit_scales_cv) whose (n_basis, lambda) are chosen by inner 3-fold orbit CV over
    the scale-fit rows.
    """

    n_basis_grid: tuple[int, ...] = (32,)
    lambda_grid: tuple[float, ...] = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)
    objective: str = "row_mr"
    clamp: bool = True

    def __post_init__(self) -> None:
        """Refuse empty grids, n_basis below 4 and non-positive lambda."""

        if not self.n_basis_grid or not self.lambda_grid:
            raise ValueError("pspline scale model needs non-empty n_basis and lambda grids")
        if any(int(n) < 4 for n in self.n_basis_grid):
            raise ValueError("pspline n_basis values must be at least 4 (cubic B-splines)")
        if any(not math.isfinite(float(v)) or float(v) <= 0.0 for v in self.lambda_grid):
            raise ValueError("pspline lambda values must be finite and positive")

    def to_dict(self) -> dict[str, object]:
        """Return the scale-model contract as a JSON-safe dict."""

        return {
            "kind": "pspline",
            "n_basis_grid": [int(n) for n in self.n_basis_grid],
            "lambda_grid": [float(v) for v in self.lambda_grid],
            "objective": self.objective,
            "clamp": bool(self.clamp),
        }


def scale_model_from_args(args) -> ScaleModel:
    """P-spline scale contract from the --scale-spline-* options; absent options give their defaults."""

    return ScaleModel(
        n_basis_grid=tuple(parse_int_grid(getattr(args, "scale_spline_n_basis_grid", "32"))),
        lambda_grid=tuple(parse_float_grid(getattr(args, "scale_spline_lambda_grid", "0.3,1,3,10,30,100"))),
    )


def absorption_parameter_grid(args, mode: str) -> list[dict[str, float | str]]:
    """Return every uniform absorption parameter set of one mode as Nuisance fields; mode none has mu 0."""

    if mode not in {"none", "mean", "midgley_like"}:
        raise ValueError(f"unknown absorption mode {mode!r}")
    mu_values = parse_float_grid(args.absorption_mu_grid) if mode != "none" else [0.0]
    return [{"absorption_mode": mode, "absorption_mu": float(mu)} for mu in mu_values]


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
            grad_values, sigma_values, shape_values = thickness_shape_grids(args, model)
            thickness_candidates += len(grad_values) * len(sigma_values) * len(shape_values)
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
