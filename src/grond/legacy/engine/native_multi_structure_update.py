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

import copy
import csv
import json
import math
import os
import re
import sys
import time
from dataclasses import asdict
from pathlib import Path
from types import SimpleNamespace

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ[_name] = "1"

if not __package__:
    # Direct legacy execution must import this checkout, not another installed GROND.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

import numpy as np
from scipy.optimize import least_squares

from grond.refine.arguments_multidataset import build_parser as build_multi_update_parser
from grond.refine.datasets import (
    DatasetState as DatasetState,
    build_dataset_state as build_dataset_state,
    choose_objective_rows as choose_objective_rows,
    compact_eval as compact_eval,
    evaluate_state as evaluate_state,
    fit_frozen_scale_contract as fit_frozen_scale_contract,
    parse_bool as parse_bool,
    parse_float as parse_float,
    parse_int as parse_int,
    parse_text as parse_text,
    predict_state as predict_state,
    read_datasets_csv as read_datasets_csv,
    score_eval as score_eval,
    stable_name_seed as stable_name_seed,
    state_residual_vector as state_residual_vector,
    write_dataset_rows as write_dataset_rows,
)
from grond.regularize.contracts import Nuisance as Nuisance
from grond.regularize.scales import native_metrics as native_metrics

if __package__:
    from ...io.config import DATASET_NAME_RE
    from .bloch_engine import DEFAULT_MAX_SUPPORT_SG, parse_model, sha256
    from .native_row_regularize import (
        TARGET_MODES,
        parse_wght,
    )
    from .native_structure_update import (
        CctbxRiguEvaluator,
        absolute_finite_difference_jacobian,
        adp_similarity_residuals,
        bond_restraint_residuals,
        build_start_restraints,
        inspect_cctbx_bridge,
        model_from_vector,
        parameter_absolute_steps,
        parameter_bounds,
        parameter_layout,
        parameter_prior,
        special_position_coordinate_policy,
        update_res_coordinates,
        write_model_delta_csv,
        write_parameter_csv,
        write_restraints_csv,
    )
else:  # pragma: no cover - direct script execution
    try:
        from grond.io.config import DATASET_NAME_RE
    except ImportError:
        DATASET_NAME_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,63}\Z")
    from bloch_engine import DEFAULT_MAX_SUPPORT_SG, parse_model, sha256
    from native_row_regularize import (
        TARGET_MODES,
        parse_wght,
    )
    from native_structure_update import (
        CctbxRiguEvaluator,
        absolute_finite_difference_jacobian,
        adp_similarity_residuals,
        bond_restraint_residuals,
        build_start_restraints,
        inspect_cctbx_bridge,
        model_from_vector,
        parameter_absolute_steps,
        parameter_bounds,
        parameter_layout,
        parameter_prior,
        special_position_coordinate_policy,
        update_res_coordinates,
        write_model_delta_csv,
        write_parameter_csv,
        write_restraints_csv,
    )


from grond.legacy.engine.bloch_engine import Config as Config, XDSData as XDSData


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


# Keep the public evaluation dispatcher here so documented legacy hooks remain local.
def evaluate_all(states: list[DatasetState], model, cache_root: Path, tag: str, workers: int, chunksize: int):
    per_dataset = [evaluate_state(s, model, cache_root, tag, workers, chunksize) for s in states]
    joint = {"datasets": per_dataset, "splits": {}}
    for split in ("all", "work", "free"):
        obs_parts, sig_parts, calc_parts = [], [], []
        wght_a_parts, wght_b_parts = [], []
        for state, ev in zip(states, per_dataset, strict=True):
            mask = state.masks[split] & ev["valid"]
            obs_parts.append(state.target[mask])
            sig_parts.append(state.sig[mask])
            calc_parts.append(ev["scaled"][mask])
            wght_a_parts.append(np.full(np.count_nonzero(mask), state.wght_a, dtype=float))
            wght_b_parts.append(np.full(np.count_nonzero(mask), state.wght_b, dtype=float))
        obs = np.concatenate(obs_parts) if obs_parts else np.asarray([])
        sig = np.concatenate(sig_parts) if sig_parts else np.asarray([])
        calc = np.concatenate(calc_parts) if calc_parts else np.asarray([])
        wght_a = np.concatenate(wght_a_parts) if wght_a_parts else np.asarray([])
        wght_b = np.concatenate(wght_b_parts) if wght_b_parts else np.asarray([])
        joint["splits"][split] = native_metrics(
            obs,
            sig,
            calc,
            np.ones_like(obs, dtype=bool),
            wght_a,
            wght_b,
            n_params=sum(ev["scale_count"] for ev in per_dataset),
        )
    return joint


def main():
    ap = build_multi_update_parser(
        description=__doc__,
        max_support_sg=DEFAULT_MAX_SUPPORT_SG,
        target_modes=TARGET_MODES,
    )
    args = ap.parse_args()

    t0 = time.perf_counter()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    cache_root = out / "cache"
    res_path = Path(args.res)
    base_model = parse_model(res_path)
    wght_a, wght_b = parse_wght(res_path)
    rows = read_datasets_csv(Path(args.datasets_csv))
    states = [
        build_dataset_state(row, args, wght_a, wght_b, symmetry_ops=base_model.get("symmetry_ops"))
        for row in rows
    ]
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
        coordinate_policy = special_position_coordinate_policy(current_model, args.atom_selection, args.atom_labels, block_mode)
        try:
            params, x0 = parameter_layout(current_model, args.atom_selection, args.atom_labels, block_mode)
        except ValueError as exc:
            block_results.append({"block": block_no, "mode": block_mode, "skipped": True, "reason": str(exc), "coordinate_constraints": coordinate_policy})
            continue
        has_xyz = any(p.get("kind", "xyz") == "xyz" for p in params)
        has_adp = any(p.get("kind") == "adp_ueq" for p in params)
        bounds = parameter_bounds(params, x0, args.max_shift_frac, args.adp_shift, args.adp_min, args.adp_max)
        absolute_steps = parameter_absolute_steps(
            params,
            xyz_step_frac=args.xyz_diff_step_frac,
            adp_step_a2=args.adp_diff_step_a2,
            legacy_step=args.diff_step,
        )
        scale_contracts = {}
        if args.objective_scale_mode == "frozen_block":
            for state in states:
                scale_contracts[state.name] = fit_frozen_scale_contract(state, current_model, cache_root, block_tag, args.workers, args.chunksize)
        eval_counter = {"n": 0}
        residual_cache = {"x": None, "value": None}
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
            residual_cache["x"] = np.asarray(x, dtype=float).copy()
            residual_cache["value"] = np.asarray(total, dtype=float).copy()
            return total

        def jacobian(x):
            cached = residual_cache.get("value")
            cached_x = residual_cache.get("x")
            if cached_x is None or not np.array_equal(np.asarray(cached_x, dtype=float), np.asarray(x, dtype=float)):
                cached = None
            return absolute_finite_difference_jacobian(
                residuals,
                x,
                bounds,
                absolute_steps,
                f0=cached,
            )

        x_scale = np.asarray([max(float(args.adp_shift if p.get("kind") == "adp_ueq" else args.max_shift_frac), 1e-5) for p in params], dtype=float)
        try:
            result = least_squares(
                residuals,
                x0,
                bounds=bounds,
                jac=jacobian,
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
                "coordinate_constraints": coordinate_policy,
                "finite_difference": {
                    "scheme": "bounded_two_point_absolute",
                    "xyz_step_frac": float(args.diff_step if args.diff_step is not None else args.xyz_diff_step_frac),
                    "adp_step_a2": float(args.diff_step if args.diff_step is not None else args.adp_diff_step_a2),
                    "legacy_diff_step_shorthand_used": bool(args.diff_step is not None),
                    "minimum_step": float(np.min(absolute_steps)),
                    "maximum_step": float(np.max(absolute_steps)),
                },
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
