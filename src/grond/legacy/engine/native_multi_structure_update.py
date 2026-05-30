#!/usr/bin/env python3
"""Shared-model native Bloch row structure update across many XDS datasets.

This is the structure-moving counterpart to native_multi_crystal_update.py. It
keeps the row contract unmerged and dataset-local, but updates one shared atomic
model against the concatenated native dynamic residual vector. Each dataset has
its own thickness, gradient, frame offset, sigma model, and group scales.

The default mode is intentionally conservative:

* scales are frozen inside each coordinate/ADP microcycle;
* nuisance parameters are read from a CSV and held fixed during atom moves;
* acceptance is gated on a validation metric over the full selected rows;
* start-geometry bond and ADP-similarity restraints are included.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import os
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

import numpy as np
from scipy.optimize import least_squares

from bloch_engine import Config, XDSData, parse_model, sha256, split_masks
from native_row_regularize import (
    Nuisance,
    native_metrics,
    parse_wght,
    row_selection,
    sigma_eff,
    target_arrays,
)
from native_structure_update import (
    adp_similarity_residuals,
    bond_restraint_residuals,
    build_start_restraints,
    CctbxRiguEvaluator,
    apply_row_resolution_to_prediction,
    fit_scale_contract,
    inspect_cctbx_bridge,
    model_from_vector,
    parameter_bounds,
    parameter_layout,
    parameter_prior,
    prediction_for_model,
    residual_vector,
    scaled_prediction,
    update_res_coordinates,
    validation_score,
    write_model_delta_csv,
    write_parameter_csv,
    write_restraints_csv,
)


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


class EarlyStopOptimization(Exception):
    """Stop a structure block after a real plateau while preserving its best vector."""

    def __init__(self, reason: str, x, metric: float, eval_no: int):
        super().__init__(reason)
        self.reason = reason
        self.x = np.asarray(x, dtype=float).copy()
        self.metric = float(metric)
        self.eval_no = int(eval_no)


def early_stop_min_evals(args, params: list[dict]) -> int:
    """Return the minimum residual calls before a plateau gate is allowed to fire."""

    explicit = int(getattr(args, "early_stop_min_evals", 0))
    if explicit > 0:
        return explicit
    window = max(1, int(getattr(args, "early_stop_window", 0)))
    # One finite-difference optimizer iteration is roughly n_parameters + 1 calls.
    # Waiting for two sweeps keeps ADP/XYZ blocks from stopping on the first local
    # derivative stencil while still preventing multi-day flat tails.
    return max(2 * (len(params) + 1), window)


def make_early_stop_result(exc: EarlyStopOptimization, eval_count: int):
    """Build a small scipy-like result object for a block stopped by our gate."""

    return SimpleNamespace(
        x=exc.x,
        success=False,
        status=0,
        message=f"early stopped: {exc.reason}",
        cost=float("nan"),
        optimality=float("nan"),
        nfev=int(eval_count),
        early_stop=True,
        early_stop_reason=exc.reason,
        early_stop_metric=float(exc.metric),
        early_stop_eval=int(exc.eval_no),
    )
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
    return rows


def stable_name_seed(name: str) -> int:
    total = 0
    for ch in str(name):
        total = (total * 131 + ord(ch)) % 100000
    return total


def build_dataset_state(row: dict, args, wght_a: float, wght_b: float) -> DatasetState:
    name = row.get("name") or row.get("dataset") or Path(row["hkl"]).parent.parent.name
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
    masks = split_masks(data.hkl, args.free_fraction, args.split_seed)
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


def evaluate_all(states: list[DatasetState], model, cache_root: Path, tag: str, workers: int, chunksize: int):
    per_dataset = [evaluate_state(s, model, cache_root, tag, workers, chunksize) for s in states]
    joint = {"datasets": per_dataset, "splits": {}}
    for split in ("all", "work", "free"):
        obs_parts, sig_parts, calc_parts = [], [], []
        for state, ev in zip(states, per_dataset):
            mask = state.masks[split] & ev["valid"]
            obs_parts.append(state.target[mask])
            sig_parts.append(state.sig[mask])
            calc_parts.append(ev["scaled"][mask])
        obs = np.concatenate(obs_parts) if obs_parts else np.asarray([])
        sig = np.concatenate(sig_parts) if sig_parts else np.asarray([])
        calc = np.concatenate(calc_parts) if calc_parts else np.asarray([])
        joint["splits"][split] = native_metrics(obs, sig, calc, np.ones_like(obs, dtype=bool), states[0].wght_a, states[0].wght_b, n_params=sum(ev["scale_count"] for ev in per_dataset))
    return joint


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


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--datasets-csv", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--hkl-transform", default="identity")
    ap.add_argument("--n-basis", type=int, default=20)
    ap.add_argument("--n-points", type=int, default=10)
    ap.add_argument("--sg-window", type=float, default=0.004)
    ap.add_argument("--sg-window-mode", choices=["fixed", "zobs_residual", "zobs_plus_floor", "rlp_peak"], default="fixed")
    ap.add_argument("--sg-window-min", type=float, default=0.003)
    ap.add_argument("--sg-window-max", type=float, default=0.012)
    ap.add_argument("--sg-window-scale", type=float, default=1.0)
    ap.add_argument("--sg-window-frame-floor", type=float, default=0.50)
    ap.add_argument("--topup-d-min", type=float, default=0.8)
    ap.add_argument("--max-support-sg", type=float, default=0.005)
    ap.add_argument("--min-d-spacing", type=float, default=0.70)
    ap.add_argument("--min-peak", type=float, default=90.0)
    ap.add_argument("--geometry-percentile", type=float, default=100.0, help="Keep this broad geometry percentile after PEAK/d-spacing filters.")
    ap.add_argument("--target-mode", choices=["iobs", "recorded_proxy"], default="iobs")
    ap.add_argument("--include-nonpositive", action="store_true")
    ap.add_argument("--objective-rows-per-dataset", type=int, default=300)
    ap.add_argument("--objective-bins", type=int, default=10)
    ap.add_argument("--objective-group-width-deg", type=float, default=3.0)
    ap.add_argument("--objective-scale-mode", choices=["frozen_block", "refit_each_eval"], default="frozen_block")
    ap.add_argument("--scale-fit-split", choices=["work", "all"], default="all")
    ap.add_argument("--block-schedule", default="adp_ueq,xyz")
    ap.add_argument("--atom-selection", choices=["nonh", "all", "labels"], default="nonh")
    ap.add_argument("--atom-labels", default="")
    ap.add_argument("--max-shift-frac", type=float, default=0.006)
    ap.add_argument("--xyz-restraint-sigma", type=float, default=0.006)
    ap.add_argument("--adp-shift", type=float, default=0.006)
    ap.add_argument("--adp-restraint-sigma", type=float, default=0.003)
    ap.add_argument("--adp-min", type=float, default=0.002)
    ap.add_argument("--adp-max", type=float, default=0.150)
    ap.add_argument("--restraints", choices=["none", "start_geometry", "cctbx_rigu"], default="start_geometry")
    ap.add_argument("--bond-restraint-sigma", type=float, default=0.020)
    ap.add_argument("--adp-similarity-sigma", type=float, default=0.005)
    ap.add_argument("--bond-detection-tolerance", type=float, default=0.45)
    ap.add_argument("--bond-detection-max-distance", type=float, default=1.95)
    ap.add_argument("--max-nfev", type=int, default=20)
    ap.add_argument("--diff-step", type=float, default=1e-4)
    ap.add_argument("--thickness", type=float, default=800.0)
    ap.add_argument("--thickness-model", choices=["single", "uniform_gradient", "gaussian", "gaussian_gradient"], default="uniform_gradient")
    ap.add_argument("--thickness-grad-frac", type=float, default=0.4)
    ap.add_argument("--thickness-sigma-frac", type=float, default=0.0)
    ap.add_argument("--thickness-points", type=int, default=5)
    ap.add_argument("--group-width-deg", type=float, default=1.5)
    ap.add_argument("--frame-offset", type=float, default=0.0)
    ap.add_argument("--detector-scale-bins", type=int, default=0)
    ap.add_argument("--sigma-a", type=float, default=0.06)
    ap.add_argument("--sigma-b-corr", type=float, default=0.0)
    ap.add_argument("--sigma-c-peak", type=float, default=0.0)
    ap.add_argument("--compute-backend", choices=["scipy", "scipy_bethe", "scipy_expm", "scipy_expm_multiply", "jax", "jax_auto", "jax_cpu", "jax_mps", "jax_cuda"], default="scipy", help="Bloch propagation backend. SciPy eigensolver is the production reference; scipy_bethe uses an opt-in promoted Bethe/Schur strong-beam reduction; SciPy expm/expm_multiply and JAX are experimental.")
    ap.add_argument("--jax-batch-size", type=int, default=0, help="JAX matrix batch size. 0 chooses a conservative size from matrix dimension and device count.")
    ap.add_argument("--profile-quadrature", choices=["uniform", "trapezoid", "gauss_legendre"], default="uniform", help="Rocking-profile Sg quadrature rule.")
    ap.add_argument("--profile-accumulation", choices=["sum", "mean"], default="sum", help="Accumulate rocking-profile samples as summed detector sub-exposures or legacy normalized means.")
    ap.add_argument("--sg-jitter-mode", choices=["none", "gaussian", "student_t"], default="none")
    ap.add_argument("--sg-jitter-sigma0", type=float, default=0.0)
    ap.add_argument("--sg-jitter-sigma-d-inv", type=float, default=0.0)
    ap.add_argument("--sg-jitter-sigma-frame", type=float, default=0.0)
    ap.add_argument("--sg-jitter-points", type=int, default=1)
    ap.add_argument("--sg-jitter-student-df", type=float, default=4.0)
    ap.add_argument("--sg-jitter-width-factor", type=float, default=3.0)
    ap.add_argument("--absorption-mode", choices=["none", "mean", "beam_dependent", "offdiag_fraction", "offdiag_tds", "midgley_like"], default="none")
    ap.add_argument("--absorption-mu", type=float, default=0.0)
    ap.add_argument("--absorption-frac", type=float, default=0.0)
    ap.add_argument("--absorption-mu-g2", type=float, default=0.0)
    ap.add_argument("--absorption-mu-theta2", type=float, default=0.0)
    ap.add_argument("--absorption-tds-b", type=float, default=0.0)
    ap.add_argument("--dose-b-mode", choices=["none", "linear_frame"], default="none")
    ap.add_argument("--dose-b-per-frame", type=float, default=0.0)
    ap.add_argument("--dose-b-ref-frame", type=float, default=float("nan"))
    ap.add_argument("--row-resolution-b", type=float, default=0.0, help="Raw-row dynamical resolution-envelope B from per-dataset nuisance regularization.")
    ap.add_argument("--native-wght-a", type=float, default=float("nan"))
    ap.add_argument("--native-wght-b", type=float, default=float("nan"))
    ap.add_argument("--ug-d-min", type=float, default=0.55)
    ap.add_argument("--free-fraction", type=float, default=0.10)
    ap.add_argument("--split-seed", type=int, default=42)
    ap.add_argument("--acceptance-metric", choices=["all_wR2", "work_wR2", "free_wR2", "all_R1"], default="all_wR2")
    ap.add_argument("--acceptance-min-delta", type=float, default=0.0)
    ap.add_argument("--free-wR2-tolerance", type=float, default=0.002)
    ap.add_argument("--allow-free-worse", action="store_true")
    ap.add_argument("--early-stop-window", type=int, default=96, help="Stop a structure block after this many residual calls without a meaningful objective improvement; 0 disables.")
    ap.add_argument("--early-stop-min-evals", type=int, default=0, help="Minimum residual calls before early stopping; 0 uses an automatic parameter-count-aware floor.")
    ap.add_argument("--early-stop-min-improvement", type=float, default=1e-5, help="Minimum RMS improvement counted as progress for structure-block early stopping.")
    ap.add_argument("--early-stop-metric", choices=["total_rms", "data_rms"], default="total_rms", help="RMS trace column used by the structure-block plateau stop.")
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--chunksize", type=int, default=8)
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = out / "cache"
    res_path = Path(args.res)
    base_model = parse_model(res_path)
    wght_a, wght_b = parse_wght(res_path)
    rows = read_datasets_csv(Path(args.datasets_csv))
    states = [build_dataset_state(row, args, wght_a, wght_b) for row in rows]
    write_dataset_rows(out, states)

    block_schedule = [x.strip() for x in re.split(r"[,\s]+", args.block_schedule) if x.strip()]
    allowed_blocks = {"xyz", "adp_ueq", "xyz_adp_ueq"}
    unknown = [b for b in block_schedule if b not in allowed_blocks]
    if unknown:
        raise ValueError(f"unknown model-update block(s): {unknown}; allowed {sorted(allowed_blocks)}")
    restraints = []
    if args.restraints == "start_geometry":
        restraints = build_start_restraints(base_model, args.atom_selection, args.atom_labels, args.bond_detection_tolerance, args.bond_detection_max_distance)
    write_restraints_csv(out / "restraints.csv", restraints)
    cctbx_rigu = CctbxRiguEvaluator(res_path, out) if args.restraints == "cctbx_rigu" else None
    cctbx_bridge = cctbx_rigu.status if cctbx_rigu is not None else inspect_cctbx_bridge(res_path, out)
    restraint_count = int(cctbx_rigu.count) if cctbx_rigu is not None else int(len(restraints))

    current_model = copy.deepcopy(base_model)
    start_eval = evaluate_all(states, current_model, cache_root, "start", args.workers, args.chunksize)
    current_eval = start_eval
    last_candidate_eval = start_eval
    trace = []
    block_results = []
    accepted_any = False
    acceptance_reason = "no model-update blocks accepted"

    for block_no, block_mode in enumerate(block_schedule, start=1):
        block_tag = f"block_{block_no:02d}_{block_mode}"
        try:
            params, x0 = parameter_layout(current_model, args.atom_selection, args.atom_labels, block_mode)
        except ValueError as exc:
            block_results.append({"block": block_no, "mode": block_mode, "skipped": True, "reason": str(exc)})
            continue
        has_xyz = any(p.get("kind", "xyz") == "xyz" for p in params)
        has_adp = any(p.get("kind") == "adp_ueq" for p in params)
        bounds = parameter_bounds(params, x0, args.max_shift_frac, args.adp_shift, args.adp_min, args.adp_max)
        scale_contracts = {}
        if args.objective_scale_mode == "frozen_block":
            for state in states:
                scale_contracts[state.name] = fit_frozen_scale_contract(state, current_model, cache_root, block_tag, args.workers, args.chunksize)
        eval_counter = {"n": 0}
        early_stop_state = {
            "enabled": int(args.early_stop_window) > 0,
            "window": max(1, int(args.early_stop_window)),
            "min_evals": early_stop_min_evals(args, params),
            "min_improvement": max(0.0, float(args.early_stop_min_improvement)),
            "best_metric": float("inf"),
            "best_eval": 0,
            "best_x": np.asarray(x0, dtype=float).copy(),
            "initial_metric": float("nan"),
            "triggered": False,
            "reason": "",
        }

        def residuals(x, *, params=params, x0=x0, block_tag=block_tag, block_no=block_no, block_mode=block_mode, current_model=current_model, has_xyz=has_xyz, has_adp=has_adp):
            eval_counter["n"] += 1
            model = model_from_vector(current_model, params, x)
            parts = []
            data_rms_by_dataset = {}
            for state in states:
                contract = scale_contracts.get(state.name)
                res = state_residual_vector(state, model, cache_root, block_tag, args.workers, args.chunksize, scale_contract=contract)
                parts.append(res)
                data_rms_by_dataset[state.name] = float(np.sqrt(np.mean(res * res))) if len(res) else float("nan")
            data_resid = np.concatenate(parts) if parts else np.zeros(0, dtype=float)
            prior = parameter_prior(params, x, x0, args.xyz_restraint_sigma, args.adp_restraint_sigma)
            geom = bond_restraint_residuals(model, restraints, args.bond_restraint_sigma) if has_xyz else np.zeros(0, dtype=float)
            adp = adp_similarity_residuals(model, restraints, args.adp_similarity_sigma) if has_adp else np.zeros(0, dtype=float)
            cctbx_geom = cctbx_rigu.residuals(model) if cctbx_rigu is not None and (has_xyz or has_adp) else np.zeros(0, dtype=float)
            total = np.concatenate([data_resid, prior, geom, adp, cctbx_geom])
            rms = float(np.sqrt(np.mean(total * total))) if len(total) else float("nan")
            data_rms = float(np.sqrt(np.mean(data_resid * data_resid))) if len(data_resid) else float("nan")
            trace.append(
                {
                    "block": block_no,
                    "mode": block_mode,
                    "eval": eval_counter["n"],
                    "total_rms": rms,
                    "data_rms": data_rms,
                    "prior_rms": float(np.sqrt(np.mean(prior * prior))) if len(prior) else 0.0,
                    "geometry_rms": float(np.sqrt(np.mean(geom * geom))) if len(geom) else 0.0,
                    "adp_similarity_rms": float(np.sqrt(np.mean(adp * adp))) if len(adp) else 0.0,
                    "cctbx_rigu_rms": float(np.sqrt(np.mean(cctbx_geom * cctbx_geom))) if len(cctbx_geom) else 0.0,
                    "max_abs_shift": float(np.max(np.abs(x - x0))) if len(x) else 0.0,
                    "dataset_data_rms": data_rms_by_dataset,
                }
            )
            print(
                f"multi block {block_no:02d} {block_mode} eval {eval_counter['n']:03d}: "
                f"data_rms={trace[-1]['data_rms']:.4f} total_rms={rms:.4f} max_shift={trace[-1]['max_abs_shift']:.5f}",
                flush=True,
            )
            if early_stop_state["enabled"]:
                stop_metric = rms if args.early_stop_metric == "total_rms" else data_rms
                if math.isfinite(stop_metric):
                    if not math.isfinite(float(early_stop_state["initial_metric"])):
                        early_stop_state["initial_metric"] = float(stop_metric)
                    if stop_metric < float(early_stop_state["best_metric"]) - float(early_stop_state["min_improvement"]):
                        early_stop_state["best_metric"] = float(stop_metric)
                        early_stop_state["best_eval"] = int(eval_counter["n"])
                        early_stop_state["best_x"] = np.asarray(x, dtype=float).copy()
                    stale = int(eval_counter["n"]) - int(early_stop_state["best_eval"])
                    if int(eval_counter["n"]) >= int(early_stop_state["min_evals"]) and stale >= int(early_stop_state["window"]):
                        reason = (
                            f"{args.early_stop_metric} plateau for {stale} residual calls "
                            f"after eval {early_stop_state['best_eval']} "
                            f"(best {float(early_stop_state['best_metric']):.6g})"
                        )
                        early_stop_state["triggered"] = True
                        early_stop_state["reason"] = reason
                        raise EarlyStopOptimization(
                            reason,
                            early_stop_state["best_x"],
                            float(early_stop_state["best_metric"]),
                            int(early_stop_state["best_eval"]),
                        )
            return total

        x_scale = np.asarray([max(float(args.adp_shift if p.get("kind") == "adp_ueq" else args.max_shift_frac), 1e-5) for p in params], dtype=float)
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=bounds,
                jac="2-point",
                diff_step=args.diff_step,
                loss="linear",
                max_nfev=args.max_nfev,
                verbose=1,
                x_scale=x_scale,
            )
        except EarlyStopOptimization as exc:
            result = make_early_stop_result(exc, eval_counter["n"])
            print(f"multi block {block_no:02d} {block_mode}: {result.message}", flush=True)
        candidate_model = model_from_vector(current_model, params, result.x)
        candidate_eval = evaluate_all(states, candidate_model, cache_root, f"{block_tag}_candidate", args.workers, args.chunksize)
        last_candidate_eval = candidate_eval
        start_metric = score_eval(current_eval, args.acceptance_metric)
        cand_metric = score_eval(candidate_eval, args.acceptance_metric)
        free_start = score_eval(current_eval, "free_wR2")
        free_cand = score_eval(candidate_eval, "free_wR2")
        metric_ok = cand_metric <= start_metric - float(args.acceptance_min_delta)
        free_ok = args.allow_free_worse or args.acceptance_metric == "free_wR2" or free_cand <= free_start + float(args.free_wR2_tolerance)
        block_accepted = bool(metric_ok and free_ok)
        reason = (
            f"accepted: {args.acceptance_metric} {start_metric:.6g} -> {cand_metric:.6g}"
            if block_accepted
            else f"rejected: {args.acceptance_metric} {start_metric:.6g} -> {cand_metric:.6g}; free_wR2 {free_start:.6g} -> {free_cand:.6g}; metric_ok={metric_ok}; free_ok={free_ok}"
        )
        candidate_res = out / f"{block_tag}_candidate.res"
        accepted_res = out / f"{block_tag}_accepted.res"
        update_res_coordinates(res_path, candidate_res, candidate_model)
        update_res_coordinates(res_path, accepted_res, candidate_model if block_accepted else current_model)
        write_parameter_csv(out / f"{block_tag}_candidate_parameters.csv", params, x0, result.x)
        write_parameter_csv(out / f"{block_tag}_accepted_parameters.csv", params, x0, result.x if block_accepted else x0)
        block_results.append(
            {
                "block": block_no,
                "mode": block_mode,
                "parameter_count": int(len(params)),
                "accepted": block_accepted,
                "reason": reason,
                "current_metric": start_metric,
                "candidate_metric": cand_metric,
                "free_wR2_start": free_start,
                "free_wR2_candidate": free_cand,
                "least_squares": {
                    "success": bool(result.success),
                    "status": int(result.status),
                    "message": str(result.message),
                    "cost": float(result.cost),
                    "optimality": float(result.optimality),
                    "nfev": int(result.nfev),
                    "early_stop": bool(getattr(result, "early_stop", False)),
                    "early_stop_reason": str(getattr(result, "early_stop_reason", "")),
                    "early_stop_eval": int(getattr(result, "early_stop_eval", 0)),
                    "early_stop_metric": float(getattr(result, "early_stop_metric", float("nan"))),
                },
                "candidate_res": str(candidate_res),
                "accepted_res": str(accepted_res),
            }
        )
        print(f"multi block {block_no:02d} {block_mode}: {reason}", flush=True)
        if block_accepted:
            current_model = candidate_model
            current_eval = candidate_eval
            accepted_any = True
            acceptance_reason = reason

    final_eval = current_eval
    candidate_eval = last_candidate_eval
    updated_res = out / "bloch_multi_updated_model.res"
    accepted_res = out / "accepted_model.res"
    update_res_coordinates(res_path, updated_res, current_model)
    update_res_coordinates(res_path, accepted_res, current_model)
    write_model_delta_csv(out / "model_delta.csv", base_model, current_model)
    with (out / "trace.jsonl").open("w") as f:
        for rec in trace:
            f.write(json.dumps(rec, allow_nan=True) + "\n")

    summary = {
        "inputs": {
            "datasets_csv": str(Path(args.datasets_csv).resolve()),
            "res": str(res_path.resolve()),
            "res_sha256": sha256(res_path),
        },
        "config": vars(args),
        "row_counts": {
            "datasets": [
                {
                    "name": s.name,
                    "target_rows": int(len(s.target_indices)),
                    "objective_rows": int(len(s.objective_indices)),
                    "work_rows": int(np.count_nonzero(s.masks["work"])),
                    "free_rows": int(np.count_nonzero(s.masks["free"])),
                    "nuisance": asdict(s.nuisance),
                }
                for s in states
            ],
            "total_target_rows": int(sum(len(s.target_indices) for s in states)),
            "total_objective_rows": int(sum(len(s.objective_indices) for s in states)),
        },
        "restraints": {
            "mode": args.restraints,
            "count": restraint_count,
            "bond_restraint_sigma": float(args.bond_restraint_sigma),
            "adp_similarity_sigma": float(args.adp_similarity_sigma),
            "cctbx_bridge": cctbx_bridge,
        },
        "blocks": block_results,
        "acceptance": {
            "accepted_any": bool(accepted_any),
            "metric": args.acceptance_metric,
            "start_metric": score_eval(start_eval, args.acceptance_metric),
            "accepted_metric": score_eval(final_eval, args.acceptance_metric),
            "free_wR2_start": score_eval(start_eval, "free_wR2"),
            "free_wR2_accepted": score_eval(final_eval, "free_wR2"),
            "reason": acceptance_reason,
        },
        "start_eval": compact_eval(start_eval),
        "candidate_eval": compact_eval(candidate_eval),
        "accepted_eval": compact_eval(final_eval),
        "accepted_res": str(accepted_res),
        "updated_res": str(updated_res),
        "wall_seconds": time.perf_counter() - t0,
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=True) + "\n")

    with (out / "dataset_metrics.csv").open("w", newline="") as f:
        fields = ["stage", "dataset", "split", "n", "R1_gt", "R1_all", "wR2_all", "GooF", "CC"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for stage, ev in [("start", start_eval), ("candidate", candidate_eval), ("accepted", final_eval)]:
            for ds_ev in ev["datasets"]:
                for split, m in ds_ev["splits"].items():
                    w.writerow({"stage": stage, "dataset": ds_ev["name"], "split": split, **{k: m.get(k) for k in fields if k in m}})

    lines = [
        "# Native Multi-Dataset Row-Parameter Regularization",
        "",
        f"Datasets: `{len(states)}`. Target rows: `{summary['row_counts']['total_target_rows']}`. Objective rows: `{summary['row_counts']['total_objective_rows']}`. N/Q: `{args.n_basis}/{args.n_points}`.",
        "",
        f"Scale mode: `{args.objective_scale_mode}`. Atom selection: `{args.atom_selection}`. Blocks: `{','.join(block_schedule)}`.",
        "",
        "## Joint Validation",
        "",
        "| split | start R1all % | candidate R1all % | accepted R1all % | start wR2all % | candidate wR2all % | accepted wR2all % | start CC | candidate CC | accepted CC |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for split in ("work", "free", "all"):
        s = start_eval["splits"][split]
        c = candidate_eval["splits"][split]
        a = final_eval["splits"][split]
        lines.append(
            f"| {split} | {100*s['R1_all']:.2f} | {100*c['R1_all']:.2f} | {100*a['R1_all']:.2f} | "
            f"{100*s['wR2_all']:.2f} | {100*c['wR2_all']:.2f} | {100*a['wR2_all']:.2f} | "
            f"{s['CC']:.4f} | {c['CC']:.4f} | {a['CC']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## Blocks",
            "",
            "| block | mode | params | accepted | decision |",
            "|---:|---|---:|---|---|",
        ]
    )
    for br in block_results:
        lines.append(f"| {br['block']} | {br['mode']} | {br['parameter_count']} | {str(br['accepted']).lower()} | {br['reason']} |")
    lines.extend(
        [
            "",
            "## Files",
            "",
            f"Accepted RES: `{accepted_res}`",
            f"Model delta: `{out / 'model_delta.csv'}`",
            f"Dataset metrics: `{out / 'dataset_metrics.csv'}`",
            f"Trace: `{out / 'trace.jsonl'}`",
        ]
    )
    (out / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(json.dumps({"out_dir": str(out.resolve()), "report": str((out / "REPORT.md").resolve()), "accepted_any": accepted_any, "accepted_metric": summary["acceptance"]["accepted_metric"]}, indent=2, allow_nan=True))


if __name__ == "__main__":
    main()
