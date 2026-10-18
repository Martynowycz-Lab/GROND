"""Finished-run inputs of the dynamical polish: row tables, the run's Bloch contract and frozen row plans.

A crystal is one final native row table (``native_dyn_calc.csv`` with its stage ``summary.json``).  The Bloch
configuration is rebuilt from the stage summary exactly as the engine built its trials (window mode, profile
sampling, mosaic cone, dose B, thickness quadrature, absorption, beam support), and two checks bind it:

* the engine's own row intensities for a sample of rows, recomputed with the rebuilt configuration and the
  stage model, must reproduce the table's ``dyn_raw_unmodulated``;
* the polish forward model at the start parameters must reproduce the engine's intensities for the start
  model on a sample of rows.

Row plans (beam basis, excitation errors per profile point, profile weights, dose B, uniform absorption) are
built once with the start model and then frozen: the basis membership and the rocking geometry do not depend
on the structure except through the beam ranking, which the polish holds fixed.
"""

from __future__ import annotations

import contextlib
import csv
import dataclasses
import json
import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from grond.bloch.config import Config
from grond.bloch.potential import PotentialMixin
from grond.bloch.thickness import (
    DEFAULT_TABLE_DT_A,
    DEFAULT_THICKNESS_MAX_A,
    DEFAULT_THICKNESS_MIN_A,
    config_distribution,
    config_support,
    distribution_weight_derivatives,
    distribution_weights,
    resolved_config_evaluator,
    support_grid,
)
from grond.polish.propagator import born_intensity, row_intensity

# Engine backends whose intensities are the full-basis eigensolution the polish differentiates.
EXACT_BACKENDS = {"scipy", "numpy_scipy_cpu", "scipy_expm", "expm", "scipy_expm_multiply", "expm_multiply"}
# Rows sampled per crystal for the configuration and forward-model identity checks.
CHECK_ROWS = 12
# Relative tolerance of the table reproduction (the run may have used complex64 matrices).
TABLE_CHECK_RTOL = 2e-3
# Relative tolerance of the polish forward model against the engine for the same model (both complex128).
IDENTITY_RTOL = 1e-7
# Relative tolerance of the polish's frame scales, refitted to the stage's own rows, against its dyn_scaled (the
# per-frame engine fit stops its SHELX-weight iteration at 1e-8).
SCALE_CHECK_RTOL = 1e-6


def _need(block: dict, key: str, where: str):
    """Return ``block[key]``, or raise ValueError naming ``where`` when the stage predates the polish contract."""

    if key not in block:
        raise ValueError(f"{where} lacks {key!r}; the stage predates this polish contract, rerun it")
    return block[key]


def stage_config(summary: dict) -> Config:
    """The engine Config of a native stage's selected trial (as populate_nuisance_trials built it)."""

    if summary.get("prediction_kind") == "frozen_dynamical_polish" or "diffbloch_polish" in summary:
        raise ValueError(
            "polished rows use a frozen plan, not the source stage recipe; "
            "replay with grond.polish.artifacts.replay_rows (legacy unbound outputs must be regenerated)"
        )
    c = _need(summary, "config", "stage summary")
    best = _need(summary, "best_native", "stage summary")
    n = _need(best, "nuisance", "stage summary best_native")
    where_c, where_n = "stage summary config", "stage summary best_native.nuisance"
    min_d = float(c.get("min_d_spacing", 0.0) or 0.0)
    topup = float(_need(c, "topup_d_min", where_c))
    effective_topup = min(topup, min_d) if min_d > 0.0 else topup
    ref_frame = c.get("dose_b_ref_frame", float("nan"))
    ref_frame = float("nan") if ref_frame is None else float(ref_frame)
    cfg = Config(
        n_basis=int(_need(c, "n_basis", where_c)),
        n_points=int(_need(c, "n_points", where_c)),
        sg_window=float(_need(c, "sg_window", where_c)),
        sg_window_mode=str(_need(c, "sg_window_mode", where_c)),
        sg_window_min=float(_need(c, "sg_window_min", where_c)),
        sg_window_max=float(_need(c, "sg_window_max", where_c)),
        sg_window_scale=float(_need(c, "sg_window_scale", where_c)),
        sg_window_frame_floor=float(_need(c, "sg_window_frame_floor", where_c)),
        sg_jitter_mode=str(_need(n, "sg_jitter_mode", where_n)),
        sg_jitter_sigma0=float(_need(n, "sg_jitter_sigma0", where_n)),
        sg_jitter_sigma_d_inv=float(_need(n, "sg_jitter_sigma_d_inv", where_n)),
        sg_jitter_sigma_frame=float(_need(n, "sg_jitter_sigma_frame", where_n)),
        sg_jitter_points=int(_need(c, "sg_jitter_points", where_c)),
        sg_jitter_student_df=float(_need(c, "sg_jitter_student_df", where_c)),
        sg_jitter_width_factor=float(_need(c, "sg_jitter_width_factor", where_c)),
        mosaicity_mode=str(n.get("mosaicity_mode", "none")),
        mosaicity_sigma_deg=float(n.get("mosaicity_sigma_deg", 0.0)),
        mosaicity_max_sigma_deg=float(c.get("mosaicity_max_sigma_deg", 0.5)),
        mosaicity_points=int(c.get("mosaicity_points", 3)),
        mosaicity_support_sigma_deg=float(c.get("mosaicity_support_sigma_deg", 0.0)),
        mosaicity_support_points=int(c.get("mosaicity_support_points", 5)),
        mosaicity_support_width_sigma=float(c.get("mosaicity_support_width_sigma", 4.0)),
        geometry_path_file=str(c.get("geometry_path", "") or ""),
        profile_basis_mode=str(_need(c, "profile_basis_mode", where_c)),
        profile_coordinate=str(_need(c, "profile_coordinate", where_c)),
        profile_normalization=str(c.get("profile_normalization", "legacy")),
        profile_angle_max_half_width_deg=float(_need(c, "profile_angle_max_half_width_deg", where_c)),
        profile_quadrature=str(_need(c, "profile_quadrature", where_c)),
        profile_accumulation=str(_need(c, "profile_accumulation", where_c)),
        topup_d_min=effective_topup,
        max_support_sg=float(_need(c, "max_support_sg", where_c)),
        thickness_A=float(_need(n, "thickness_A", where_n)),
        thickness_model=str(_need(n, "thickness_model", where_n)),
        thickness_grad_frac=float(_need(n, "thickness_grad_frac", where_n)),
        thickness_sigma_frac=float(n.get("thickness_sigma_frac", 0.0)),
        thickness_points=int(_need(c, "thickness_points", where_c)),
        thickness_evaluator=str(n.get("thickness_evaluator", c.get("thickness_evaluator", "quadrature"))),
        thickness_gamma_shape=float(n.get("thickness_gamma_shape", c.get("thickness_gamma_shape", 1.0))),
        thickness_min_A=float(n.get("thickness_min_A", c.get("thickness_min_A", DEFAULT_THICKNESS_MIN_A))),
        thickness_max_A=float(n.get("thickness_max_A", c.get("thickness_max_A", DEFAULT_THICKNESS_MAX_A))),
        thickness_table_dt_A=float(n.get("thickness_table_dt_A", c.get("thickness_table_dt_A", DEFAULT_TABLE_DT_A))),
        group_width_deg=float(_need(n, "group_width_deg", where_n)),
        frame_offset=float(_need(n, "frame_offset", where_n)),
        absorption_mode=str(n.get("absorption_mode", "none")),
        absorption_mu=float(n.get("absorption_mu", 0.0)),
        absorption_frac=float(n.get("absorption_frac", 0.0)),
        absorption_mu_g2=float(n.get("absorption_mu_g2", 0.0)),
        absorption_mu_theta2=float(n.get("absorption_mu_theta2", 0.0)),
        absorption_tds_b=float(n.get("absorption_tds_b", 0.0)),
        dose_b_mode=str(n.get("dose_b_mode", "none")),
        dose_b_per_frame=float(n.get("dose_b_per_frame", 0.0)),
        dose_b_ref_frame=ref_frame,
        compute_backend=str(c.get("compute_backend", "scipy")),
        matrix_convention=str(c.get("matrix_convention", "j_minus_i")),
        row_plan_cache="off",
        paired_inverse_hand=False,
        free_fraction=float(c.get("free_fraction", 0.10)),
        split_seed=int(c.get("split_seed", 42)),
        double_precision=bool(c.get("double_precision", False)),
    )
    return cfg


def thickness_quadrature(cfg: Config) -> dict:
    """Nodes t_i = t0 * (1 + grad * a_i + sigma * b_i) and weights of the engine's thickness grid."""

    evaluator = resolved_config_evaluator(cfg)
    if evaluator == "table":
        # The forward engine may omit zero-weight nodes for one fixed distribution. A fit must retain the
        # entire declared support: changing the mean or width can put positive mass on those nodes later.
        family, params = config_distribution(cfg)
        nodes = support_grid(*config_support(cfg))
        weights = distribution_weights(family, params, nodes)
        return {
            "model": str(cfg.thickness_model), "evaluator": "table", "nodes": nodes, "weights": weights,
            "t0": float(cfg.thickness_A), "grad": float(cfg.thickness_grad_frac),
            "gamma_shape": float(cfg.thickness_gamma_shape),
            "refine_grad": cfg.thickness_model == "uniform_gradient" and cfg.thickness_grad_frac > 0,
        }
    if evaluator != "quadrature":
        raise ValueError(
            f"dynamical polish does not yet support the {evaluator!r} thickness evaluator "
            f"for {cfg.thickness_model!r}; use a verified table evaluator or legacy node quadrature"
        )
    model = str(cfg.thickness_model).lower()
    probe = SimpleNamespace(cfg=cfg)
    nodes, weights = PotentialMixin.thickness_grid(probe)
    t0 = float(cfg.thickness_A)
    grad = float(cfg.thickness_grad_frac)
    unit = np.asarray(nodes, dtype=float) / t0
    if model == "uniform_gradient" and grad > 0.0 and len(unit) > 1:
        a = (unit - 1.0) / grad
        refine_grad = True
    elif model == "gaussian_gradient" and grad > 0.0 and len(unit) > 1:
        # nodes are t0 (1 + grad xg + sqrt2 sigma xh); only t0 is refined here
        a = np.zeros_like(unit)
        refine_grad = False
    else:
        a = np.zeros_like(unit)
        refine_grad = False
    if model not in {"single", "uniform_gradient", "gaussian", "gaussian_gradient"}:
        raise ValueError(f"dynamical polish has no thickness parameterization for thickness_model={model!r}")
    return {
        "model": model,
        "t0": t0,
        "grad": grad,
        "refine_grad": refine_grad,
        "base": unit - grad * a,  # the part that does not scale with grad
        "a": a,
        "weights": np.asarray(weights, dtype=float),
    }


def thickness_nodes(quad: dict, t0: float, grad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(nodes, dnodes/dt0, dnodes/dgrad) for thickness centre t0 and gradient fraction grad."""

    if quad.get("evaluator") == "table":
        nodes = np.asarray(quad["nodes"], dtype=float)
        return nodes, np.zeros_like(nodes), np.zeros_like(nodes)
    unit = quad["base"] + float(grad) * quad["a"]
    nodes = float(t0) * unit
    if np.any(nodes <= 0.0):
        raise ValueError("thickness quadrature reached a non-positive node; the polish keeps the engine grid positive")
    return nodes, unit, float(t0) * quad["a"]


def thickness_weights(quad: dict, t0: float, grad: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Weights and derivatives with respect to nominal mean/centre and wedge gradient.

    Table support nodes stay fixed; changing the distribution changes their
    weights, including support renormalization. Gamma shape remains fixed.
    Legacy quadrature moves nodes and keeps weights fixed.
    """
    if quad.get("evaluator") != "table":
        weights = np.asarray(quad["weights"], dtype=float)
        return weights, np.zeros_like(weights), np.zeros_like(weights)
    parameters = ({"mean": t0, "shape": quad["gamma_shape"]}
                  if quad["model"] == "gamma" else {"t0": t0, "grad": grad})
    weights, derivatives = distribution_weight_derivatives(quad["model"], parameters, quad["nodes"])
    return weights, derivatives.get("mean", derivatives.get("t0")), derivatives.get("grad", np.zeros_like(weights))


@dataclass
class PolishCrystal:
    """One crystal's final native row table, its stage contract and the engine Config rebuilt from it.

    Row arrays follow table order: ``y`` is target_I, ``sigma_source`` and ``sigma_eff`` the source and effective
    sigmas, ``table_dyn`` the unmodulated Bloch rows and ``table_pred`` the scaled prediction (dyn_scaled).
    ``fit_weight`` is 0 for a crystal that is predicted but never fitted.
    """

    name: str
    row_table: Path
    summary_path: Path
    summary: dict
    hkl: Path
    hkl_transform: str
    stage_res: Path
    stage_alternate_hand: bool
    cfg: Config
    fieldnames: list[str]
    table_rows: list[dict]
    rows: np.ndarray
    hkl_rows: np.ndarray
    d: np.ndarray
    split: np.ndarray
    y: np.ndarray
    sigma_source: np.ndarray
    sigma_eff: np.ndarray
    angle: np.ndarray
    group: np.ndarray
    row_factor: np.ndarray
    table_dyn: np.ndarray
    table_pred: np.ndarray
    fit_weight: float
    role: str
    scale: dict
    table_dyn_raw: np.ndarray = field(default_factory=lambda: np.zeros(0))
    endpoint_weight: float = 1.0
    extra: dict = field(default_factory=dict)

    @property
    def fit_mask(self) -> np.ndarray:
        """Work rows that fit this crystal's own frame scales (the stage's scale-fit rows)."""

        valid = np.isfinite(self.y) & np.isfinite(self.sigma_source) & (self.sigma_source > 0.0)
        return (self.split == "work") & valid

    @property
    def in_objective(self) -> bool:
        """False for an endpoint-only (completeness_only) crystal: predicted and written to the HKL, never fitted."""

        return self.fit_weight > 0.0

    @property
    def free_mask(self) -> np.ndarray:
        """Held-out (free) rows of this crystal."""

        return self.split == "free"


def _floats(rows: list[dict], name: str) -> np.ndarray:
    """Return one column of the table rows as floats, NaN where missing or not numeric."""

    out = np.full(len(rows), np.nan)
    for i, row in enumerate(rows):
        with contextlib.suppress(TypeError, ValueError):
            out[i] = float(row.get(name, "nan"))
    return out


def scale_contract(summary: dict, table_rows: list[dict]) -> dict:
    """The frame-scale family of a stage and its fitted hyperparameters."""

    replay = sorted({str(r.get("scale_replay_model", "") or "") for r in table_rows} - {""})
    if replay:
        raise ValueError(
            f"row table replays a compact scale contract ({', '.join(replay)}); the polish refits the stage's own "
            "frame-scale family and cannot reproduce a replayed topology"
        )
    best = summary.get("best_native") or {}
    info = best.get("scale_model") if isinstance(best.get("scale_model"), dict) else {}
    kind = str(info.get("kind") or (summary.get("config") or {}).get("scale_model") or "per-frame")
    nui = best.get("nuisance") or {}
    if int(float(nui.get("detector_scale_bins", 0) or 0)) > 1:
        raise ValueError("the polish has no detector-scale field; the stage used detector scale bins")
    if kind == "pspline":
        return {
            "kind": "pspline",
            "n_basis": int(_need(info, "n_basis", "best_native.scale_model")),
            "lambda": float(_need(info, "lambda", "best_native.scale_model")),
            "born_lambda": float(info.get("born_lambda", info["lambda"])),
            "born_n_basis": int(info.get("born_n_basis", info["n_basis"])),
            "objective": str(info.get("objective", "row_mr")),
            "clamp": bool(info.get("clamp", True)),
        }
    if kind == "per-frame":
        if abs(float(nui.get("scale_smoothness", 0.0) or 0.0)) > 0.0:
            raise ValueError("the polish refits free per-frame scales; the stage used a scale-smoothness penalty")
        # The engine fits per-frame scales with iterated SHELX weights 1/(sigma^2 + (a P)^2 + b P) of the stage's
        # native WGHT (regularize.scales.fit_group_scales); the polish needs the same a and b.
        weight = _need(best, "native_weight", "stage summary best_native")
        return {
            "kind": "per-frame",
            "wght_a": float(_need(weight, "a", "stage summary best_native.native_weight")),
            "wght_b": float(_need(weight, "b", "stage summary best_native.native_weight")),
        }
    raise ValueError(f"unknown stage scale model {kind!r}")


def load_crystal(name: str, row_table: Path) -> PolishCrystal:
    """Load one crystal from its final native row table and the stage summary.json beside it.

    Rebuilds the stage Config and scale contract, and refuses a stage the polish cannot reproduce: a backend that is not
    the full eigensolution, split labels other than work/free, a null q_control, a row modulation beyond
    row_resolution_factor, an internally inverted (--alternate-hand) model, or a missing HKL or model file.
    """

    row_table = Path(row_table).expanduser().resolve()
    summary_path = row_table.parent / "summary.json"
    if not row_table.is_file() or not summary_path.is_file():
        raise FileNotFoundError(f"{name}: row table or its summary.json is missing ({row_table})")
    summary = json.loads(summary_path.read_text())
    with row_table.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        table_rows = list(reader)
    if not table_rows:
        raise ValueError(f"{name}: {row_table} has no rows")
    required = ("row_index", "h", "k", "l", "d_A", "target_I", "source_sigma", "sigma_eff", "split", "dyn_raw_unmodulated", "dyn_raw",
                "dyn_scaled", "group")
    missing = [key for key in required if key not in fieldnames]
    if missing:
        raise ValueError(f"{name}: {row_table} lacks the columns {missing}")
    inputs = _need(summary, "inputs", f"{name} stage summary")
    config = summary.get("config") or {}
    cfg = stage_config(summary)
    backend = str(cfg.compute_backend).lower()
    if backend not in EXACT_BACKENDS and not backend.startswith("jax"):
        raise ValueError(
            f"{name}: stage compute_backend={backend!r} is not the full eigensolution the polish differentiates"
        )
    split = np.asarray([str(r.get("split", "")).strip().lower() for r in table_rows], dtype=object)
    unknown = sorted(set(split) - {"work", "free"})
    if unknown:
        raise ValueError(f"{name}: row table has split labels {unknown}; only work and free rows are supported")
    angle = _floats(table_rows, "scale_angle_deg") if "scale_angle_deg" in fieldnames else None
    weight = _floats(table_rows, "raw_row_weight") if "raw_row_weight" in fieldnames else np.ones(len(table_rows))
    role = str(table_rows[0].get("dataset_role", "primary") or "primary")
    fit_weight = float(weight[0]) if np.isfinite(weight[0]) else 1.0
    if role in {"exclude", "completeness_only"}:
        fit_weight = 0.0
    # The combine's endpoint weight (completeness_only crystals keep a positive one; excluded crystals none).
    endpoint = _floats(table_rows[:1], "endpoint_weight")[0] if "endpoint_weight" in fieldnames else float("nan")
    endpoint_weight = float(endpoint) if np.isfinite(endpoint) else fit_weight
    if role == "exclude":
        endpoint_weight = 0.0
    factor = _floats(table_rows, "row_resolution_factor") if "row_resolution_factor" in fieldnames else np.ones(len(table_rows))
    factor = np.where(np.isfinite(factor), factor, 1.0)
    q_control = ((summary.get("best_native") or {}).get("q_control") or {}).get("mode", "real")
    if str(q_control) != "real":
        raise ValueError(f"{name}: the stage used q_control={q_control!r} (a null control), not the real Bloch rows")
    # dyn_raw must be the row-resolution factor times the unmodulated Bloch row: the only modulation the polish carries.
    dyn_raw = _floats(table_rows, "dyn_raw")
    modulation = relative_mismatch(dyn_raw, factor * _floats(table_rows, "dyn_raw_unmodulated"))
    if not (math.isfinite(modulation) and modulation <= 1e-12):
        raise ValueError(
            f"{name}: dyn_raw is not row_resolution_factor x dyn_raw_unmodulated (relative difference {modulation:.3g}); "
            "the stage applied a row modulation the polish does not model"
        )
    if angle is None:
        angle = np.full(len(table_rows), np.nan)
    crystal = PolishCrystal(
        name=name,
        row_table=row_table,
        summary_path=summary_path,
        summary=summary,
        hkl=Path(str(_need(inputs, "hkl", "stage summary inputs"))).expanduser(),
        hkl_transform=str(inputs.get("hkl_transform", "identity") or "identity"),
        stage_res=Path(str(_need(inputs, "res", "stage summary inputs"))).expanduser(),
        stage_alternate_hand=bool(config.get("alternate_hand", False)),
        cfg=cfg,
        fieldnames=fieldnames,
        table_rows=table_rows,
        rows=np.asarray([int(float(r["row_index"])) for r in table_rows], dtype=int),
        hkl_rows=np.asarray([[int(float(r["h"])), int(float(r["k"])), int(float(r["l"]))] for r in table_rows], dtype=int),
        d=_floats(table_rows, "d_A"),
        split=split,
        y=_floats(table_rows, "target_I"),
        sigma_source=_floats(table_rows, "source_sigma"),
        sigma_eff=_floats(table_rows, "sigma_eff"),
        angle=angle,
        group=np.asarray([int(float(r["group"])) for r in table_rows], dtype=int),
        row_factor=factor,
        table_dyn=_floats(table_rows, "dyn_raw_unmodulated"),
        table_pred=_floats(table_rows, "dyn_scaled"),
        fit_weight=fit_weight,
        role=role,
        scale=scale_contract(summary, table_rows),
        table_dyn_raw=dyn_raw,
        endpoint_weight=endpoint_weight,
    )
    if crystal.stage_alternate_hand:
        # The engine inverts the model internally, but its SHELXL endpoint refines the RES as given: the stage's rows
        # and its publication model are in opposite hands, which neither identity check can see.
        raise ValueError(
            f"{name}: the stage computed its rows with --alternate-hand (an internally inverted model), so its SHELXL "
            "endpoint is in the other hand; polish a stage run on a materialised alternate-hand model (as AUTO does)"
        )
    if crystal.scale["kind"] == "pspline" and not np.all(np.isfinite(crystal.angle)):
        raise ValueError(f"{name}: P-spline stage rows lack scale_angle_deg")
    if not crystal.hkl.is_file():
        raise FileNotFoundError(f"{name}: stage HKL {crystal.hkl} is not readable")
    if not crystal.stage_res.is_file():
        raise FileNotFoundError(f"{name}: stage model {crystal.stage_res} is not readable")
    return crystal


# ----------------------------------------------------------------------------- worker side

_PORTS: dict = {}


def _thread_limits() -> None:
    """Default the BLAS and OpenMP thread counts to 1 for this process and the workers it starts."""

    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ.setdefault(name, "1")


def _port(spec: dict):
    """One XDSBlochPort per worker and (data, model, configuration)."""

    from grond.legacy.engine.bloch_engine import XDSBlochPort, XDSData, alternate_hand_model, parse_model

    key = json.dumps(spec, sort_keys=True, default=str)
    port = _PORTS.get(key)
    if port is None:
        data = XDSData(Path(spec["hkl"]), hkl_transform=spec["hkl_transform"])
        model = parse_model(Path(spec["res"]))
        if spec.get("alternate_hand"):
            model = alternate_hand_model(model)
        cfg = Config(**spec["cfg"])
        port = XDSBlochPort(data, model, cfg)
        if len(_PORTS) > 4:
            _PORTS.clear()
        _PORTS[key] = port
    return port


def config_dict(cfg: Config) -> dict:
    """Return a Config as a plain dict, the form a worker spec carries."""

    out = dataclasses.asdict(cfg)
    return out


def _uniform_absorption(matrix: np.ndarray, row_index: int, scale: float) -> float:
    """Validate the supported Hermitian-plus-uniform-loss Hamiltonian and return its loss."""

    diagonal = np.diagonal(matrix, axis1=-2, axis2=-1)
    mu_diagonal = diagonal.imag
    mu = float(np.mean(mu_diagonal))
    anti = 0.5 * (matrix - np.conj(np.swapaxes(matrix, -2, -1)))
    indices = np.arange(matrix.shape[-1])
    anti[..., indices, indices] = 0.0
    if (float(np.max(np.abs(anti), initial=0.0)) > 1e-9 * scale
            or float(np.max(np.abs(mu_diagonal - mu), initial=0.0)) > 1e-9 * scale):
        raise ValueError(
            f"row {row_index}: the stage Hamiltonian is not Hermitian plus a uniform i*mu; the polish supports "
            "absorption_mode none or a uniform absorptive potential only"
        )
    return mu


def compact_row_plan(port, row_index: int, *, predict: bool = False) -> dict:
    """Freeze one row's basis and profile geometry without stacking its dense Hamiltonians.

    Both scan and PETS geometry require O(N^2 + P*N) memory. Optional reference predictions use the native
    propagator one profile at a time.
    """

    if port.data.pets_exact_a_center is not None:
        plan = port.row_prediction_plan(row_index)
        basis = np.asarray(plan["support_hkls"], dtype=np.int32)
        matrices = plan["_dyn_matrices"]
        weights = np.asarray(plan["_profile_weights"], dtype=float)
        static = np.asarray(matrices.static, dtype=np.complex128)
        diagonal = matrices.diagonals().astype(np.complex128)
        sg = diagonal.real.copy()
        off_diagonal = np.abs(static)
        np.fill_diagonal(off_diagonal, 0.0)
        scale = max(float(np.max(off_diagonal)), float(np.max(np.abs(diagonal))), 1e-30)
        mu = _uniform_absorption(static, row_index, scale)
        reference = float(sum(weight * port.propagate_direct_to_target_thickness_average(np.asarray(matrix, dtype=np.complex128))
                              for weight, matrix in zip(weights, matrices, strict=True))) if predict else float("nan")
    else:
        hkl = port.data.hkl[row_index]
        centre = port.omega_from_z(port.data.zcal[row_index])
        centre_sg = float(port.sg_for_hkls(np.asarray([hkl]), centre)[0])
        width, _metadata = port.row_sg_window(row_index, hkl, centre)
        angles, tilts, weights, _residuals, _jitter, _mosaic = port.row_profile_orientation_points(
            row_index, hkl, centre, centre_sg, width
        )
        basis, _metadata = port.select_basis_for_profile(hkl, centre, width, angles, tilts, row_idx=row_index)
        basis = np.asarray(basis, dtype=np.int32)
        if len(basis) < 2 or not np.array_equal(basis[0], [0, 0, 0]) or not np.array_equal(basis[1], hkl):
            raise ValueError("frozen basis must start with the direct beam and exact signed target HKL")
        static = port.hamiltonian_static_part(basis, absorb=True, row_idx=row_index)
        sg = np.asarray([port.sg_for_hkls_at_orientation(basis, angle, along, transverse)
                         for angle, (along, transverse) in zip(angles, tilts, strict=True)], dtype=static.real.dtype)
        scale = max(float(np.max(np.abs(static))), float(np.max(np.abs(sg + np.diag(static)))), 1e-30)
        mu = _uniform_absorption(static, row_index, scale)
        reference = float(sum(weight * port.propagate_direct_to_target_thickness_average(
            port.hamiltonian_from_static(static, excitation))
            for weight, excitation in zip(weights, sg, strict=True))) if predict else float("nan")
        sg = (sg + np.diag(static).real).astype(float)
    return {"support_hkls": basis, "sg": sg, "weights": np.asarray(weights, dtype=float),
            "mu": mu if port.absorption_enabled() else 0.0,
            "dose_delta_b_A2": float(port.row_delta_b(row_index)), "native_intensity": reference}


def build_plan_chunk(task: dict) -> dict:
    """Build and save the frozen row plans of one chunk; engine check intensities for sampled rows."""

    _thread_limits()
    port = _port(task["polish_spec"])
    rows = [int(r) for r in task["rows"]]
    sample_engine = set(int(r) for r in task.get("engine_check_rows", []))
    sample_table = set(int(r) for r in task.get("table_check_rows", []))
    stage_port = _port(task["stage_spec"]) if sample_table else None
    basis_all, sg_all, w_all = [], [], []
    n_beams, n_prof, mu_all, delta_b, engine_dyn, stage_dyn = [], [], [], [], [], []
    for r in rows:
        plan = compact_row_plan(port, r, predict=r in sample_engine)
        basis, sg, weights, mu = (plan[key] for key in ("support_hkls", "sg", "weights", "mu"))
        basis_all.append(basis)
        sg_all.append(sg.ravel())
        w_all.append(weights)
        n_beams.append(len(basis))
        n_prof.append(len(weights))
        mu_all.append(mu)
        delta_b.append(float(plan.get("dose_delta_b_A2", 0.0) or 0.0))
        engine_dyn.append(plan["native_intensity"])
        stage_dyn.append(compact_row_plan(stage_port, r, predict=True)["native_intensity"]
                         if r in sample_table else float("nan"))
    out = Path(task["out"])
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(".tmp.npz")
    np.savez(
        tmp,
        rows=np.asarray(rows, dtype=np.int64),
        positions=np.asarray(task["positions"], dtype=np.int64),
        n_beams=np.asarray(n_beams, dtype=np.int64),
        n_prof=np.asarray(n_prof, dtype=np.int64),
        basis=np.concatenate(basis_all).astype(np.int32),
        sg=np.concatenate(sg_all),
        w=np.concatenate(w_all),
        mu=np.asarray(mu_all),
        delta_b=np.asarray(delta_b),
        engine_dyn=np.asarray(engine_dyn),
        stage_dyn=np.asarray(stage_dyn),
        thick=np.asarray(port.thick, dtype=float),
        thick_w=np.asarray(port.thick_w, dtype=float),
    )
    os.replace(tmp, out)
    return {"out": str(out), "rows": len(rows), "max_beams": int(max(n_beams)) if n_beams else 0}


def _chunk_views(chunk):
    """Yield (i, basis, sg, weights) per row of a saved plan chunk; ``sg`` has shape (profile points, beams)."""

    ob = osg = ow = 0
    for i in range(len(chunk["rows"])):
        n, p = int(chunk["n_beams"][i]), int(chunk["n_prof"][i])
        yield i, chunk["basis"][ob : ob + n], chunk["sg"][osg : osg + p * n].reshape(p, n), chunk["w"][ow : ow + p]
        ob += n
        osg += p * n
        ow += p


def evaluate_chunk(task: dict) -> dict:
    """Row intensities, Born controls, thickness derivatives and structure Jacobian rows of one chunk."""

    _thread_limits()
    chunk = dict(np.load(task["chunk"]))
    u_table = np.load(task["u"], mmap_mode="r")
    lookup = np.load(task["lookup"], mmap_mode="r")
    lim = np.asarray(task["lookup_lim"], dtype=np.int64)
    d_re = np.load(task["d_re"], mmap_mode="r") if task.get("d_re") else None
    d_im = np.load(task["d_im"], mmap_mode="r") if task.get("d_im") else None
    grad_positions = set(int(v) for v in task.get("grad_positions", []))
    a_matrix = np.asarray(task["a_matrix"], dtype=float)
    k = float(task["k"])
    j_minus_i = str(task["matrix_convention"]) == "j_minus_i"
    t_nodes = np.asarray(task["t_nodes"], dtype=float)
    t_weights = np.asarray(task["t_weights"], dtype=float)
    dt_dt0 = np.asarray(task["dt_dt0"], dtype=float)
    dt_dgrad = np.asarray(task["dt_dgrad"], dtype=float)
    dw_dt0 = np.asarray(task.get("dw_dt0", np.zeros_like(t_weights)), dtype=float)
    dw_dgrad = np.asarray(task.get("dw_dgrad", np.zeros_like(t_weights)), dtype=float)
    keep_response = bool(task.get("return_thickness_response", False))
    weight_gradient = keep_response or bool(np.any(dw_dt0) or np.any(dw_dgrad))
    n_rows = len(chunk["rows"])
    n_g = len(u_table)
    intensity = np.zeros(n_rows)
    born = np.zeros(n_rows)
    d_t0 = np.zeros(n_rows)
    d_grad = np.zeros(n_rows)
    response = np.zeros((n_rows, len(t_nodes))) if keep_response else None
    grad_rows = [i for i in range(n_rows) if int(chunk["positions"][i]) in grad_positions]
    gamma_g = np.zeros((len(grad_rows), n_g), dtype=np.complex128) if (grad_rows and d_re is not None) else None
    slot = {i: j for j, i in enumerate(grad_rows)}
    shift = 2 * lim
    for i, basis, sg, weights in _chunk_views(chunk):
        basis = basis.astype(np.int64)
        diffs = basis[None, :, :] - basis[:, None, :] if j_minus_i else basis[:, None, :] - basis[None, :, :]
        shifted = diffs + lim
        n = len(basis)
        off = ~np.eye(n, dtype=bool)
        inside = np.all((shifted >= 0) & (shifted <= shift), axis=-1)
        if not np.all(inside[off]):
            raise ValueError(f"row {int(chunk['rows'][i])}: a beam difference lies outside the Fourier-potential table")
        idx = np.full((n, n), -1, dtype=np.int64)
        idx[off] = lookup[tuple(shifted[off].T)]
        if np.any(idx[off] < 0):
            raise ValueError(f"row {int(chunk['rows'][i])}: a beam difference lies outside the Fourier-potential table")
        delta_b = float(chunk["delta_b"][i])
        if delta_b != 0.0:
            g_cart = diffs.reshape(-1, 3).astype(float) @ a_matrix.T
            att = np.exp(-0.25 * delta_b * np.sum(g_cart * g_cart, axis=1)).reshape(n, n)
        else:
            att = np.ones((n, n))
        coupling = np.zeros((n, n), dtype=np.complex128)
        coupling[off] = np.asarray(u_table[idx[off]]) * att[off] / (2.0 * k)
        want_grad = i in slot and gamma_g is not None
        result = row_intensity(
            coupling, sg, weights, t_nodes, t_weights, float(chunk["mu"][i]), adjoint=want_grad,
            return_weight_gradient=weight_gradient,
        )
        value, dt, gamma = result[:3]
        intensity[i] = value
        d_t0[i] = float(dt @ dt_dt0)
        d_grad[i] = float(dt @ dt_dgrad)
        if weight_gradient:
            d_t0[i] += float(result[3] @ dw_dt0)
            d_grad[i] += float(result[3] @ dw_dgrad)
        if keep_response:
            response[i] = result[3]
        born[i] = born_intensity(abs(coupling[1, 0]) ** 2, sg[:, 1] - sg[:, 0], weights, t_nodes, t_weights)
        if want_grad:
            contrib = gamma[off] * att[off] / (2.0 * k)
            flat = idx[off]
            gamma_g[slot[i]] += np.bincount(flat, weights=contrib.real, minlength=n_g) + 1j * np.bincount(
                flat, weights=contrib.imag, minlength=n_g
            )
    jac = None
    if gamma_g is not None:
        jac = gamma_g.real @ np.asarray(d_re) + gamma_g.imag @ np.asarray(d_im)
    output = {
        "crystal": str(task.get("crystal", "")),
        "positions": np.asarray(chunk["positions"], dtype=np.int64),
        "intensity": intensity,
        "born": born,
        "d_t0": d_t0,
        "d_grad": d_grad,
        "grad_positions": np.asarray([int(chunk["positions"][i]) for i in grad_rows], dtype=np.int64),
        "jacobian": jac,
        "engine_dyn": np.asarray(chunk["engine_dyn"]),
        "stage_dyn": np.asarray(chunk["stage_dyn"]),
        "thick": np.asarray(chunk["thick"]),
        "thick_w": np.asarray(chunk["thick_w"]),
    }
    if keep_response:
        output["thickness_response"] = response
        output["thickness_response_nodes"] = t_nodes.copy()
    return output


def pool_context_name() -> str:
    """forkserver on Linux, spawn elsewhere: never fork the polish process itself.

    The parent has run torch (the dU_g/dp autograd) and its OpenMP thread pool before every evaluation; a forked
    child that enters an OpenMP region of that state can hang.  The fork server and spawned workers start clean.
    """

    import platform

    return "forkserver" if platform.system() == "Linux" else "spawn"


def run_tasks(function, tasks: list[dict], workers: int) -> list[dict]:
    """Run worker tasks in a process pool (pool_context_name) or serially for one worker."""

    if int(workers) <= 1 or len(tasks) <= 1:
        return [function(task) for task in tasks]
    import multiprocessing

    # Workers inherit the environment at start: one BLAS/OpenMP thread each (setting it inside a worker would be
    # too late, numpy is already loaded there).
    _thread_limits()
    context = multiprocessing.get_context(pool_context_name())
    with context.Pool(processes=min(int(workers), len(tasks))) as pool:
        return list(pool.imap(function, tasks, chunksize=1))


def relative_mismatch(a: np.ndarray, b: np.ndarray) -> float:
    """Return max |a - b| / |b| over entries where both are finite, or NaN when there are none."""

    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.any(ok):
        return float("nan")
    return float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300)))


def sample_rows(n: int, count: int = CHECK_ROWS) -> np.ndarray:
    """Return up to ``count`` evenly spaced, unique positions in 0..n-1 for the identity checks."""

    if n <= 0:
        return np.zeros(0, dtype=int)
    return np.unique(np.linspace(0, n - 1, min(n, count)).round().astype(int))


def ball_radius(crystal: PolishCrystal) -> float:
    """Largest |g| of a beam difference: top-up pool radius plus the finest target row."""

    d_min_rows = float(np.nanmin(crystal.d)) if np.any(np.isfinite(crystal.d)) else float(crystal.cfg.topup_d_min)
    return (1.0 / float(crystal.cfg.topup_d_min) + 1.0 / min(d_min_rows, float(crystal.cfg.topup_d_min))) * (1.0 + 1e-9)
