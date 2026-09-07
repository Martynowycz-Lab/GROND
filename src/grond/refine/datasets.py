"""Independent crystal state for shared native structure updates. Each crystal keeps its observations, scale design, nuisance parameters, and weights; concatenation does not make their errors independent."""

from __future__ import annotations

import copy
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from grond.io.config import DATASET_NAME_RE
from grond.legacy.engine.bloch_engine import (
    Config,
    XDSData,
    split_masks,
)
from grond.legacy.engine.native_structure_update import (
    apply_row_resolution_to_prediction,
    fit_scale_contract,
    prediction_for_model,
    residual_vector,
    scaled_prediction,
)
from grond.regularize.contracts import Nuisance
from grond.regularize.observations import row_selection, sigma_eff, target_arrays
from grond.regularize.scales import native_metrics


@dataclass
class DatasetState:
    name: str
    hkl: Path
    data: XDSData
    target: np.ndarray
    base_sig: np.ndarray
    sig: np.ndarray
    target_indices: np.ndarray
    objective_indices: np.ndarray
    masks: dict
    validation_fit_indices: np.ndarray
    nuisance: Nuisance
    cfg: Config
    objective_cfg: Config
    wght_a: float
    wght_b: float
    detector_scale_bins: int
    min_d_spacing: float
    min_peak: float
    target_mode: str
    dataset_weight: float
    dataset_role: str


def parse_bool(text: str | None, default: bool = False) -> bool:
    if text is None or str(text).strip() == "":
        return default
    return str(text).strip().lower() in {"1", "true", "yes", "y", "on"}


def parse_float(row: dict, key: str, default: float) -> float:
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return float(default)
    return float(val)


def parse_int(row: dict, key: str, default: int) -> int:
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return int(default)
    return int(float(val))


def parse_text(row: dict, key: str, default: str) -> str:
    val = row.get(key, "")
    if val is None or str(val).strip() == "":
        return str(default)
    return str(val).strip()


def choose_objective_rows(data: XDSData, indices, masks, target, sig, n_rows: int, n_bins: int, seed: int):
    candidates = np.asarray(
        [i for i in indices if masks["work"][i] and target[i] > 0 and np.isfinite(target[i]) and np.isfinite(sig[i])],
        dtype=int,
    )
    if n_rows <= 0 or n_rows >= len(candidates):
        return np.asarray(candidates, dtype=int)
    rng = np.random.default_rng(int(seed))
    dvals = np.asarray([data.cell.calculate_d(tuple(int(v) for v in h)) for h in data.hkl], dtype=float)
    order = candidates[np.argsort(dvals[candidates])]
    bins = np.array_split(order, max(1, int(n_bins)))
    chosen: list[int] = []
    remaining = int(n_rows)
    chosen_set: set[int] = set()
    for b in bins:
        if len(b) == 0:
            continue
        quota = int(round(n_rows * len(b) / max(len(candidates), 1)))
        quota = max(1, min(quota, len(b), remaining))
        if quota <= 0:
            continue
        remaining -= quota
        isigi = np.maximum(target[b] / np.maximum(sig[b], 1e-12), 0.0)
        prob = np.sqrt(isigi + 1.0)
        prob = prob / np.sum(prob)
        picked = rng.choice(b, size=quota, replace=False, p=prob).tolist()
        chosen.extend(int(x) for x in picked)
        chosen_set.update(int(x) for x in picked)
    if remaining > 0:
        pool = np.asarray([int(i) for i in candidates if int(i) not in chosen_set], dtype=int)
        if len(pool):
            extra = rng.choice(pool, size=min(remaining, len(pool)), replace=False)
            chosen.extend(int(x) for x in extra.tolist())
    return np.asarray(sorted(set(chosen)), dtype=int)


def read_datasets_csv(path: Path) -> list[dict]:
    with path.open(newline="") as f:
        rows = [dict(r) for r in csv.DictReader(f)]
    rows = [r for r in rows if parse_bool(r.get("include"), True)]
    if not rows:
        raise ValueError(f"No included datasets in {path}")
    names = [str(row.get("name") or row.get("dataset") or "").strip() for row in rows]
    for name in names:
        if not DATASET_NAME_RE.fullmatch(name):
            raise ValueError(
                f"dataset name {name!r} must contain only letters, numbers, dots, underscores, or hyphens"
            )
    folded = [name.casefold() for name in names]
    if len(set(folded)) != len(folded):
        raise ValueError("dataset names in the CSV must be unique without regard to letter case")
    return rows


def stable_name_seed(name: str) -> int:
    total = 0
    for ch in str(name):
        total = (total * 131 + ord(ch)) % 100000
    return total


def build_dataset_state(row: dict, args, wght_a: float, wght_b: float, symmetry_ops=None) -> DatasetState:
    name = row.get("name") or row.get("dataset") or Path(row["hkl"]).parent.parent.name
    if not DATASET_NAME_RE.fullmatch(str(name)):
        raise ValueError(f"unsafe dataset name {name!r}")
    hkl = Path(row["hkl"])
    hkl_transform = row.get("hkl_transform") or args.hkl_transform
    data = XDSData(hkl, hkl_transform=hkl_transform)
    target_mode = row.get("target_mode") or args.target_mode
    target, base_sig = target_arrays(data, target_mode)
    min_d = parse_float(row, "min_d_spacing", args.min_d_spacing)
    effective_topup_d_min = min(float(args.topup_d_min), float(min_d)) if min_d > 0 else float(args.topup_d_min)
    min_peak = parse_float(row, "min_peak", args.min_peak)
    geometry_percentile = parse_float(row, "geometry_percentile", args.geometry_percentile)
    include_nonpositive = parse_bool(row.get("include_nonpositive"), args.include_nonpositive)
    target_indices = row_selection(data, target, not include_nonpositive, min_d, min_peak, 0, geometry_percentile)
    masks = split_masks(
        data.hkl,
        args.free_fraction,
        args.split_seed,
        symmetry_ops=symmetry_ops,
    )
    selected_mask = np.zeros(len(target), dtype=bool)
    selected_mask[target_indices] = True
    masks = {k: v & selected_mask for k, v in masks.items()}
    nuisance = Nuisance(
        thickness_A=parse_float(row, "thickness_A", parse_float(row, "thickness", args.thickness)),
        thickness_grad_frac=parse_float(row, "thickness_grad_frac", args.thickness_grad_frac),
        frame_offset=parse_float(row, "frame_offset", args.frame_offset),
        group_width_deg=parse_float(row, "group_width_deg", args.group_width_deg),
        detector_scale_bins=parse_int(row, "detector_scale_bins", args.detector_scale_bins),
        sigma_a=parse_float(row, "sigma_a", args.sigma_a),
        sigma_b_corr=parse_float(row, "sigma_b_corr", args.sigma_b_corr),
        sigma_c_peak=parse_float(row, "sigma_c_peak", args.sigma_c_peak),
        thickness_model=parse_text(row, "thickness_model", args.thickness_model),
        thickness_sigma_frac=parse_float(row, "thickness_sigma_frac", args.thickness_sigma_frac),
        sg_jitter_mode=parse_text(row, "sg_jitter_mode", args.sg_jitter_mode),
        sg_jitter_sigma0=parse_float(row, "sg_jitter_sigma0", args.sg_jitter_sigma0),
        sg_jitter_sigma_d_inv=parse_float(row, "sg_jitter_sigma_d_inv", args.sg_jitter_sigma_d_inv),
        sg_jitter_sigma_frame=parse_float(row, "sg_jitter_sigma_frame", args.sg_jitter_sigma_frame),
        sg_jitter_points=parse_int(row, "sg_jitter_points", args.sg_jitter_points),
        sg_jitter_student_df=parse_float(row, "sg_jitter_student_df", args.sg_jitter_student_df),
        sg_jitter_width_factor=parse_float(row, "sg_jitter_width_factor", args.sg_jitter_width_factor),
        absorption_mode=parse_text(row, "absorption_mode", args.absorption_mode),
        absorption_mu=parse_float(row, "absorption_mu", args.absorption_mu),
        absorption_frac=parse_float(row, "absorption_frac", args.absorption_frac),
        absorption_mu_g2=parse_float(row, "absorption_mu_g2", args.absorption_mu_g2),
        absorption_mu_theta2=parse_float(row, "absorption_mu_theta2", args.absorption_mu_theta2),
        absorption_tds_b=parse_float(row, "absorption_tds_b", args.absorption_tds_b),
        dose_b_mode=parse_text(row, "dose_b_mode", args.dose_b_mode),
        dose_b_per_frame=parse_float(row, "dose_b_per_frame", args.dose_b_per_frame),
        dose_b_ref_frame=parse_float(row, "dose_b_ref_frame", args.dose_b_ref_frame),
        row_resolution_b=parse_float(row, "row_resolution_b", args.row_resolution_b),
        native_wght_a=parse_float(row, "native_wght_a", args.native_wght_a),
        native_wght_b=parse_float(row, "native_wght_b", args.native_wght_b),
    )
    use_wght_a = float(nuisance.native_wght_a) if np.isfinite(float(nuisance.native_wght_a)) else float(wght_a)
    use_wght_b = float(nuisance.native_wght_b) if np.isfinite(float(nuisance.native_wght_b)) else float(wght_b)
    sig = sigma_eff(data, target, base_sig, nuisance)
    dataset_weight = max(0.0, parse_float(row, "dataset_weight", parse_float(row, "raw_row_weight", 1.0)))
    dataset_role = parse_text(row, "dataset_role", "primary")
    if dataset_weight > 0.0 and dataset_weight < 1.0:
        sig = sig / math.sqrt(max(dataset_weight, 1e-12))
    objective_rows = int(parse_float(row, "objective_rows", args.objective_rows_per_dataset))
    objective_indices = choose_objective_rows(data, target_indices, masks, target, sig, objective_rows, args.objective_bins, args.split_seed + stable_name_seed(name))
    if args.scale_fit_split == "all":
        validation_fit_indices = np.asarray(target_indices, dtype=int)
    else:
        validation_fit_indices = np.asarray([i for i in target_indices if masks["work"][i]], dtype=int)
    cfg = Config(
        n_basis=args.n_basis,
        n_points=args.n_points,
        sg_window=args.sg_window,
        sg_window_mode=args.sg_window_mode,
        sg_window_min=args.sg_window_min,
        sg_window_max=args.sg_window_max,
        sg_window_scale=args.sg_window_scale,
        sg_window_frame_floor=args.sg_window_frame_floor,
        sg_jitter_mode=nuisance.sg_jitter_mode,
        sg_jitter_sigma0=nuisance.sg_jitter_sigma0,
        sg_jitter_sigma_d_inv=nuisance.sg_jitter_sigma_d_inv,
        sg_jitter_sigma_frame=nuisance.sg_jitter_sigma_frame,
        sg_jitter_points=nuisance.sg_jitter_points,
        sg_jitter_student_df=nuisance.sg_jitter_student_df,
        sg_jitter_width_factor=nuisance.sg_jitter_width_factor,
        profile_quadrature=args.profile_quadrature,
        profile_accumulation=args.profile_accumulation,
        topup_d_min=effective_topup_d_min,
        max_support_sg=args.max_support_sg,
        thickness_A=nuisance.thickness_A,
        thickness_model=nuisance.thickness_model,
        thickness_grad_frac=nuisance.thickness_grad_frac,
        thickness_sigma_frac=nuisance.thickness_sigma_frac,
        thickness_points=args.thickness_points,
        group_width_deg=nuisance.group_width_deg,
        frame_offset=nuisance.frame_offset,
        free_fraction=args.free_fraction,
        split_seed=args.split_seed,
        variance_a=nuisance.sigma_a,
        variance_b_corr=nuisance.sigma_b_corr,
        variance_c_peak=nuisance.sigma_c_peak,
        compute_backend=args.compute_backend,
        jax_batch_size=args.jax_batch_size,
        jax_matrix_bucket_size=args.jax_matrix_bucket_size,
        bethe_strong_count=args.bethe_strong_count,
        bethe_variant=args.bethe_variant,
        bethe_promote_denom=args.bethe_promote_denom,
        bethe_denom_eps=args.bethe_denom_eps,
        ug_d_min=args.ug_d_min,
        absorption_mode=nuisance.absorption_mode,
        absorption_mu=nuisance.absorption_mu,
        absorption_frac=nuisance.absorption_frac,
        absorption_mu_g2=nuisance.absorption_mu_g2,
        absorption_mu_theta2=nuisance.absorption_mu_theta2,
        absorption_tds_b=nuisance.absorption_tds_b,
        dose_b_mode=nuisance.dose_b_mode,
        dose_b_per_frame=nuisance.dose_b_per_frame,
        dose_b_ref_frame=nuisance.dose_b_ref_frame,
        row_plan_cache=args.row_plan_cache,
    )
    objective_cfg = copy.deepcopy(cfg)
    objective_cfg.group_width_deg = args.objective_group_width_deg
    return DatasetState(
        name=str(name),
        hkl=hkl,
        data=data,
        target=target,
        base_sig=base_sig,
        sig=sig,
        target_indices=np.asarray(target_indices, dtype=int),
        objective_indices=np.asarray(objective_indices, dtype=int),
        masks=masks,
        validation_fit_indices=np.asarray(validation_fit_indices, dtype=int),
        nuisance=nuisance,
        cfg=cfg,
        objective_cfg=objective_cfg,
        wght_a=float(use_wght_a),
        wght_b=float(use_wght_b),
        detector_scale_bins=int(nuisance.detector_scale_bins),
        min_d_spacing=float(min_d),
        min_peak=float(min_peak),
        target_mode=str(target_mode),
        dataset_weight=float(dataset_weight),
        dataset_role=str(dataset_role),
    )


def predict_state(state: DatasetState, model, indices, cache_root: Path, tag: str, workers: int, chunksize: int, objective: bool = False):
    cfg = state.objective_cfg if objective else state.cfg
    ds_cache = cache_root / state.name
    return prediction_for_model(state.data, model, cfg, np.asarray(indices, dtype=int), ds_cache, tag, workers, chunksize)


def fit_frozen_scale_contract(state: DatasetState, model, cache_root: Path, tag: str, workers: int, chunksize: int):
    pred = predict_state(state, model, state.objective_indices, cache_root, f"{tag}_{state.name}_scale_anchor", workers, chunksize, objective=True)
    full = np.full(len(state.target), np.nan, dtype=float)
    full[pred["indices"]] = pred["dyn"]
    full, _, _ = apply_row_resolution_to_prediction(state.data, full, state.nuisance.row_resolution_b)
    contract = fit_scale_contract(
        state.data,
        state.target,
        state.sig,
        full,
        state.objective_cfg,
        state.objective_indices,
        state.wght_a,
        state.wght_b,
        detector_scale_bins=state.detector_scale_bins,
    )
    contract["mode"] = "frozen_block"
    contract["anchor_prediction_seconds"] = float(pred.get("seconds", 0.0))
    return contract


def state_residual_vector(state: DatasetState, model, cache_root: Path, tag: str, workers: int, chunksize: int, scale_contract=None):
    pred = predict_state(state, model, state.objective_indices, cache_root, f"{tag}_{state.name}_moving", workers, chunksize, objective=True)
    _, scaled, _, _, _ = scaled_prediction(
        state.data,
        state.target,
        state.sig,
        pred,
        state.objective_cfg,
        state.objective_indices,
        state.wght_a,
        state.wght_b,
        detector_scale_bins=state.detector_scale_bins,
        scale_contract=scale_contract,
        row_resolution_b=state.nuisance.row_resolution_b,
    )
    return residual_vector(state.data, state.target, state.sig, scaled, state.objective_indices, state.wght_a, state.wght_b)


def evaluate_state(state: DatasetState, model, cache_root: Path, tag: str, workers: int, chunksize: int):
    pred = predict_state(state, model, state.target_indices, cache_root, f"{tag}_{state.name}_validation", workers, chunksize, objective=False)
    _, scaled, _, _, scale_count = scaled_prediction(
        state.data,
        state.target,
        state.sig,
        pred,
        state.cfg,
        state.validation_fit_indices,
        state.wght_a,
        state.wght_b,
        detector_scale_bins=state.detector_scale_bins,
        row_resolution_b=state.nuisance.row_resolution_b,
    )
    valid = np.isfinite(scaled)
    splits = {}
    for split, mask in state.masks.items():
        splits[split] = native_metrics(state.target, state.sig, scaled, mask & valid, state.wght_a, state.wght_b, n_params=int(scale_count))
    return {
        "name": state.name,
        "splits": splits,
        "scaled": scaled,
        "valid": valid,
        "scale_count": int(scale_count),
        "prediction_seconds": float(pred.get("seconds", 0.0)),
    }


def score_eval(eval_result: dict, metric: str) -> float:
    split, value = metric.split("_", 1)
    key = "wR2_all" if value == "wR2" else "R1_all"
    return float(eval_result["splits"][split][key])


def compact_eval(eval_result: dict) -> dict:
    return {
        "splits": eval_result["splits"],
        "datasets": [
            {k: v for k, v in ds.items() if k not in {"scaled", "valid"}}
            for ds in eval_result.get("datasets", [])
        ],
    }


def write_dataset_rows(out: Path, states: list[DatasetState]):
    with (out / "datasets_used.csv").open("w", newline="") as f:
        fields = [
            "name",
            "hkl",
            "target_rows",
            "objective_rows",
            "work_rows",
            "free_rows",
            "min_d_spacing",
            "min_peak",
            "target_mode",
            "dataset_role",
            "dataset_weight",
            "thickness",
            "thickness_grad_frac",
            "frame_offset",
            "group_width_deg",
            "sigma_a",
            "sigma_b_corr",
            "sigma_c_peak",
            "native_wght_a",
            "native_wght_b",
            "thickness_model",
            "thickness_sigma_frac",
            "sg_jitter_mode",
            "sg_jitter_sigma0",
            "sg_jitter_sigma_d_inv",
            "sg_jitter_sigma_frame",
            "sg_jitter_points",
            "sg_jitter_student_df",
            "sg_jitter_width_factor",
            "absorption_mode",
            "absorption_mu",
            "absorption_frac",
            "absorption_mu_g2",
            "absorption_mu_theta2",
            "absorption_tds_b",
            "dose_b_mode",
            "dose_b_per_frame",
            "dose_b_ref_frame",
            "row_resolution_b",
            "hkl_transform",
        ]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for s in states:
            w.writerow(
                {
                    "name": s.name,
                    "hkl": str(s.hkl),
                    "target_rows": int(len(s.target_indices)),
                    "objective_rows": int(len(s.objective_indices)),
                    "work_rows": int(np.count_nonzero(s.masks["work"])),
                    "free_rows": int(np.count_nonzero(s.masks["free"])),
                    "min_d_spacing": s.min_d_spacing,
                    "min_peak": s.min_peak,
                    "target_mode": s.target_mode,
                    "dataset_role": s.dataset_role,
                    "dataset_weight": s.dataset_weight,
                    "thickness": s.nuisance.thickness_A,
                    "thickness_grad_frac": s.nuisance.thickness_grad_frac,
                    "frame_offset": s.nuisance.frame_offset,
                    "group_width_deg": s.nuisance.group_width_deg,
                    "sigma_a": s.nuisance.sigma_a,
                    "sigma_b_corr": s.nuisance.sigma_b_corr,
                    "sigma_c_peak": s.nuisance.sigma_c_peak,
                    "native_wght_a": s.nuisance.native_wght_a,
                    "native_wght_b": s.nuisance.native_wght_b,
                    "thickness_model": s.nuisance.thickness_model,
                    "thickness_sigma_frac": s.nuisance.thickness_sigma_frac,
                    "sg_jitter_mode": s.nuisance.sg_jitter_mode,
                    "sg_jitter_sigma0": s.nuisance.sg_jitter_sigma0,
                    "sg_jitter_sigma_d_inv": s.nuisance.sg_jitter_sigma_d_inv,
                    "sg_jitter_sigma_frame": s.nuisance.sg_jitter_sigma_frame,
                    "sg_jitter_points": s.nuisance.sg_jitter_points,
                    "sg_jitter_student_df": s.nuisance.sg_jitter_student_df,
                    "sg_jitter_width_factor": s.nuisance.sg_jitter_width_factor,
                    "absorption_mode": s.nuisance.absorption_mode,
                    "absorption_mu": s.nuisance.absorption_mu,
                    "absorption_frac": s.nuisance.absorption_frac,
                    "absorption_mu_g2": s.nuisance.absorption_mu_g2,
                    "absorption_mu_theta2": s.nuisance.absorption_mu_theta2,
                    "absorption_tds_b": s.nuisance.absorption_tds_b,
                    "dose_b_mode": s.nuisance.dose_b_mode,
                    "dose_b_per_frame": s.nuisance.dose_b_per_frame,
                    "dose_b_ref_frame": s.nuisance.dose_b_ref_frame,
                    "row_resolution_b": s.nuisance.row_resolution_b,
                    "hkl_transform": s.data.hkl_transform_spec,
                }
            )
