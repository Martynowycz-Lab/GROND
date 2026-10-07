"""Run the dynamical polish on a finished GROND run and write its SHELXL-compatible outputs."""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np

from grond.polish import crystals as C
from grond.polish.outputs import (
    SUMMARY_NAME,
    run_shelxl_outputs,
    write_polished_res,
    write_polished_tables,
    write_report,
)
from grond.polish.refine import PolishOptions, PolishProblem, leak_flags, solve_polish
from grond.polish.structure import StructureModel

SCHEMA = "grond.diffbloch-polish.v1"


def run_row_tables(run_dir: Path, result: dict) -> dict[str, Path]:
    """The final row tables of the run's corrected-HKL endpoint (``combined_input_runs`` for several crystals).

    They include endpoint-only (completeness_only) crystals, which the polish predicts and writes to the HKL but
    never fits.  Every crystal of the raw-row result (``publication_runs``) must use the same table.
    """

    from grond.workflows.endpoint_selection import _recorded_native_row_tables

    tables = _recorded_native_row_tables(result, include_endpoint_only=True)
    if not tables:
        raise ValueError(f"{run_dir}: grond_summary.json records no final native row tables")
    fitted = _recorded_native_row_tables(result)
    differing = sorted(name for name, path in fitted.items() if Path(tables.get(name, "")) != Path(path))
    if differing:
        raise ValueError(
            f"{run_dir}: the raw-row runs and the endpoint input runs record different row tables for {differing}; "
            "the polish needs one row table per crystal"
        )
    if list(tables) == ["dataset"]:
        prep = run_dir / "run_prep.json"
        if prep.is_file():
            datasets = json.loads(prep.read_text()).get("datasets") or []
            if len(datasets) == 1 and datasets[0].get("name"):
                tables = {str(datasets[0]["name"]): tables["dataset"]}
    out = {}
    for name, path in tables.items():
        path = Path(path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(f"{run_dir}: recorded row table for {name} is missing: {path}")
        out[str(name)] = path
    return out


def publication_model(run_dir: Path, result: dict) -> Path:
    """The run's SHELXL publication model (hash-verified), the start of the polish."""

    from grond.refine.feedback_contract import endpoint_res
    from grond.workflows.shared_refinement import multi_combined_endpoint_model

    if result.get("combined"):
        model = multi_combined_endpoint_model(Path(str(result["combined"])).expanduser())
        if model is None:
            raise FileNotFoundError(
                f"{run_dir}: the combined SHELXL endpoint is missing or fails its hash check; pass --start-res explicitly"
            )
        return model
    primary = Path(str(result.get("primary_run") or "")).expanduser()
    if not primary.is_dir():
        raise FileNotFoundError(f"{run_dir}: grond_summary.json records no primary run; pass --start-res explicitly")
    return endpoint_res(primary, json.loads((primary / "summary.json").read_text()))


def endpoint_contract(run_dir: Path, result: dict, crystals: list[C.PolishCrystal]) -> dict:
    """The run's own corrected-HKL endpoint settings that the SHELXL re-refinement replays.

    The handoff itself is always the same (work rows only, kappa*|F|^2, pathological correction ratios
    excluded, promoted WGHT, converged SHELXL, ISOR-only NPD recovery).  The merge key, the per-dataset,
    merge and post-Bloch sigma calibrations and the resolution cutoff are the run's: from ``run_config.json``
    and the combined endpoint summary for several crystals, from the stage summary for one.  A run that does
    not record them is refused.
    """

    def need(block, key: str, where: str):
        """Return ``block[key]``, or raise ValueError when the run does not record that endpoint setting."""

        if not isinstance(block, dict) or key not in block:
            raise ValueError(f"{where} does not record {key!r}; the polish cannot replay this run's endpoint settings")
        return block[key]

    if result.get("combined"):
        combined = Path(str(result["combined"])).expanduser() / "combined_summary.json"
        config = run_dir / "run_config.json"
        for path in (combined, config):
            if not path.is_file():
                raise FileNotFoundError(f"{path} is missing; the polish replays the run's combined endpoint settings from it")
        endpoint = need(json.loads(combined.read_text()), "combined_endpoint", str(combined))
        run_config = json.loads(config.read_text())
        multi = need(run_config, "multi_dataset_sigma", str(config))
        post = need(run_config, "post_bloch_sigma", str(config))
        where_m, where_p = f"{config} multi_dataset_sigma", f"{config} post_bloch_sigma"
        return {
            "source": [str(config), str(combined)],
            "merge_key": str(need(endpoint, "merge_key", str(combined))),
            "dataset_sigma_policy": str(need(multi, "resolved_dataset_policy", where_m)),
            "dataset_sigma_holdout_fraction": float(need(multi, "dataset_holdout_fraction", where_m)),
            "dataset_sigma_min_scale": float(need(multi, "dataset_min_scale", where_m)),
            "dataset_sigma_max_scale": float(need(multi, "dataset_max_scale", where_m)),
            "dataset_sigma_holdout_floor": bool(need(multi, "dataset_holdout_floor", where_m)),
            "merge_sigma_policy": str(need(multi, "resolved_merge_policy", where_m)),
            "post_bloch_sigma_policy": str(need(post, "resolved_final_endpoint", where_p)),
            "post_bloch_sigma_holdout_fraction": float(need(post, "holdout_fraction", where_p)),
            "post_bloch_sigma_min_scale": float(need(post, "min_scale", where_p)),
            "post_bloch_sigma_max_scale": float(need(post, "max_scale", where_p)),
            "min_d_spacing": float(need(endpoint, "requested_min_d_spacing_A", str(combined)) or 0.0),
        }
    if len(crystals) != 1:
        raise ValueError(f"{run_dir}: {len(crystals)} row tables but no combined endpoint in grond_summary.json")
    config = crystals[0].summary.get("config")
    where = f"{crystals[0].summary_path} config"
    return {
        "source": [str(crystals[0].summary_path)],
        "merge_key": "exact_signed_hkl",
        "dataset_sigma_policy": "none",
        "dataset_sigma_holdout_fraction": 0.2,
        "dataset_sigma_min_scale": 0.25,
        "dataset_sigma_max_scale": 4.0,
        "dataset_sigma_holdout_floor": False,
        "merge_sigma_policy": "formal",
        "post_bloch_sigma_policy": str(need(config, "post_bloch_sigma_policy", where)),
        "post_bloch_sigma_holdout_fraction": float(need(config, "post_bloch_sigma_holdout_fraction", where)),
        "post_bloch_sigma_min_scale": float(need(config, "post_bloch_sigma_min_scale", where)),
        "post_bloch_sigma_max_scale": float(need(config, "post_bloch_sigma_max_scale", where)),
        "min_d_spacing": float(need(config, "min_d_spacing", where) or 0.0),
    }


def orbit_exclusion(crystals: list[C.PolishCrystal], structure: StructureModel, exclude_orbits_json) -> dict:
    """The lockbox orbits every stage dropped; with ``exclude_orbits_json`` they must be exactly that list.

    The polish cannot remove rows from a finished run (its scales, split and endpoint were made with them), so a
    requested exclusion that a stage did not apply is refused, and no row of any table may lie in an excluded orbit.
    """

    recorded = {
        c.name: str(((c.summary.get("row_selection") or {}).get("split_scheme") or {}).get("excluded_orbits_sha256") or "")
        for c in crystals
    }
    if not exclude_orbits_json:
        return {"requested": "", "stage_excluded_orbits_sha256": recorded}
    from grond.bloch.splits import load_split_exclusion
    from grond.validation import heldout as H

    exclusion = load_split_exclusion(exclude_orbits_json)
    differing = {name: sha or "none" for name, sha in recorded.items() if sha != exclusion["sha256"]}
    if differing:
        raise ValueError(
            f"--exclude-orbits-json {exclusion['path']} (sha256 {exclusion['sha256']}) is not the orbit exclusion the run's "
            f"stages applied ({differing}); the polish cannot drop rows from a finished run. Polish a run made with "
            "this exclusion"
        )
    symmetry = structure.model.get("symmetry_ops")
    for crystal in crystals:
        keys = H.orbit_keys(crystal.hkl_rows, symmetry, include_friedel=True)
        inside = int(sum(str(key) in exclusion["orbits"] for key in keys))
        if inside:
            raise ValueError(f"{crystal.name}: {inside} row(s) of {crystal.row_table} lie in orbits of {exclusion['path']}")
    return {"requested": exclusion["path"], "sha256": exclusion["sha256"], "stage_excluded_orbits_sha256": recorded}


def _jsonable(value):
    """Return a JSON-ready copy of nested results; non-finite floats become None.

    Arrays and tuples become lists, NumPy scalars Python numbers and paths strings.
    """

    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, (np.floating, float)):
        return float(value) if math.isfinite(float(value)) else None
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, Path):
        return str(value)
    return value


def physics_record(problem: PolishProblem) -> dict:
    """Return, per crystal, the Bloch settings, scale model, row counts and dataset role recorded in the summary."""

    out = {}
    for crystal in problem.crystals:
        cfg = crystal.cfg
        out[crystal.name] = {
            "sg_window_mode": cfg.sg_window_mode,
            "sg_window": cfg.sg_window,
            "sg_window_max": cfg.sg_window_max,
            "n_points": cfg.n_points,
            "profile_coordinate": cfg.profile_coordinate,
            "profile_normalization": cfg.profile_normalization,
            "mosaicity_mode": cfg.mosaicity_mode,
            "mosaicity_sigma_deg": cfg.mosaicity_sigma_deg,
            "mosaicity_points": cfg.mosaicity_points,
            "sg_jitter_mode": cfg.sg_jitter_mode,
            "dose_b_mode": cfg.dose_b_mode,
            "dose_b_per_frame": cfg.dose_b_per_frame,
            "thickness_model": cfg.thickness_model,
            "thickness_points": cfg.thickness_points,
            "thickness_bounds_A": list(problem.thickness_bounds[crystal.name]),
            "absorption_mode": cfg.absorption_mode,
            "absorption_mu": cfg.absorption_mu,
            "max_support_sg": cfg.max_support_sg,
            "n_basis": cfg.n_basis,
            "topup_d_min": cfg.topup_d_min,
            "matrix_convention": cfg.matrix_convention,
            "scale_model": crystal.scale,
            "fit_rows": int(np.count_nonzero(problem.scales[crystal.name].fit)) if crystal.in_objective else 0,
            "scale_fit_rows": int(np.count_nonzero(problem.scales[crystal.name].fit)),
            "free_rows": int(np.count_nonzero(crystal.free_mask)),
            "dataset_role": crystal.role,
            "dataset_fit_weight": crystal.fit_weight,
            "endpoint_weight": crystal.endpoint_weight,
            "in_objective": crystal.in_objective,
            "fourier_coefficients": problem.geometry[crystal.name]["n_g"],
        }
    return out


def run_diffbloch_polish(
    run_dir: str | Path,
    out_dir: str | Path,
    *,
    start_res: str | Path | None = None,
    options: PolishOptions | None = None,
    python: str,
    shelxl: str,
    run_shelxl: bool = True,
    shelxl_iterations: int = 15,
    shelxl_ls_cycles: int = 10,
    exclude_orbits_json: str | Path | None = None,
    log=print,
) -> dict:
    """Polish a finished GROND run into a new ``out_dir`` and return the summary written there.

    Loads the run's final row tables (excluded crystals dropped) and starts from the publication model unless
    ``start_res`` is given. Writes polished_model.res, the polished row tables, the SHELXL re-refinement (unless
    ``run_shelxl`` is False), final_polish_summary.json and FINAL_POLISH.md. An ``out_dir`` that already holds a polish
    result is refused.
    """

    t_start = time.perf_counter()
    run_dir = Path(run_dir).expanduser().resolve()
    out_dir = Path(out_dir).expanduser().resolve()
    if (out_dir / SUMMARY_NAME).exists():
        raise FileExistsError(f"{out_dir} already holds a polish result; choose a new output directory")
    out_dir.mkdir(parents=True, exist_ok=True)
    options = options or PolishOptions()
    summary_path = run_dir / "grond_summary.json"
    if not summary_path.is_file():
        raise FileNotFoundError(f"{run_dir} is not a finished GROND run (no grond_summary.json)")
    result = json.loads(summary_path.read_text())
    tables = run_row_tables(run_dir, result)
    loaded = [C.load_crystal(name, path) for name, path in tables.items()]
    # Excluded crystals are dropped.  Endpoint-only (completeness_only) crystals are predicted at the polished
    # model and written to the SHELXL HKL, as in the run's endpoint, but never fitted or scored.
    not_fitted = {c.name: c.role for c in loaded if not c.in_objective}
    crystals = [c for c in loaded if c.in_objective or c.endpoint_weight > 0.0]
    if not any(c.in_objective for c in crystals):
        raise ValueError(f"{run_dir}: no crystal carries a positive dataset-policy row weight")
    contract = endpoint_contract(run_dir, result, loaded) if run_shelxl else {}
    start = Path(start_res).expanduser().resolve() if start_res else publication_model(run_dir, result)
    structure = StructureModel(start, hydrogen_mode=options.hydrogen_mode)
    if structure.n_params == 0:
        raise ValueError(f"{start}: no refinable coordinates or displacement parameters")
    exclusion = orbit_exclusion(crystals, structure, exclude_orbits_json)
    log(
        f"dynamical polish: {sum(c.in_objective for c in crystals)} fitted crystal(s) "
        f"({len(crystals)} in the endpoint), {structure.n_params} structural parameters, start {start}"
    )
    problem = PolishProblem(crystals, structure, out_dir / "work", options)
    solution = solve_polish(problem, start, log=log)
    fit = solution["fit"]
    x = fit["x"]
    p_final = x[: problem.n_struct]
    final_model = structure.model_at(p_final)
    shifts = structure.shifts(p_final)
    heavy = [row["shift_A"] for row in shifts if row["role"] == "refined" and row["element"] not in {"H", "D"}]
    polished_res = out_dir / "polished_model.res"
    free = (solution["heldout"].get("free") or {})
    rem = [
        "GROND dynamical polish (DiffBloch-style): xyz and U refined against unmerged work rows with the run's Bloch",
        f"physics; start {start.name}; free rows never fitted. Optimizer {fit['status']}.",
    ]
    write_polished_res(start, polished_res, final_model, set(structure.moved_atoms()), rem)
    thickness = {}
    for crystal in crystals:
        t0, grad = problem.thickness(problem.x0(), crystal.name)
        t1, g1 = problem.thickness(x, crystal.name)
        thickness[crystal.name] = {
            "model": problem.quad[crystal.name]["model"],
            "evaluator": problem.quad[crystal.name].get("evaluator", "quadrature"),
            "fixed_gamma_shape": problem.quad[crystal.name].get("gamma_shape") if crystal.cfg.thickness_model == "gamma" else None,
            "start_t0_A": t0,
            "final_t0_A": t1,
            "start_grad": grad,
            "final_grad": g1,
            "refined": sorted(problem.thick_index[crystal.name]),
        }
    runs = write_polished_tables(problem, solution, out_dir / "tables")
    shelxl_block: dict = {"status": "skipped"}
    if run_shelxl:
        shelxl_block = run_shelxl_outputs(
            polished_res,
            runs,
            out_dir / "shelxl",
            python=python,
            shelxl=shelxl,
            iterations=shelxl_iterations,
            ls_cycles=shelxl_ls_cycles,
            contract=contract,
            log=log,
        )
    esd = fit["esd"]
    parameters = []
    for i, label in enumerate(problem.labels):
        entry = dict(label)
        entry.update({"start": float(problem.x0()[i]), "final": float(x[i]), "shift": float(x[i] - problem.x0()[i]), "esd": float(esd[i])})
        parameters.append(entry)
    npd = structure.npd_atoms(p_final)
    trajectory = fit["trajectory"]
    summary = {
        "schema": SCHEMA,
        "status": "complete",
        "run_dir": str(run_dir),
        "start_model": str(start),
        "polished_model": str(polished_res),
        "frozen_predictions": {name: str(path / "frozen" / "manifest.json") for name, path in runs.items()},
        "crystals": {c.name: {"row_table": str(c.row_table), "stage_summary": str(c.summary_path), "hkl": str(c.hkl)} for c in crystals},
        "crystals_not_fitted": not_fitted,
        "crystals_endpoint_only": [c.name for c in crystals if not c.in_objective],
        "endpoint_contract": contract,
        "orbit_exclusion": exclusion,
        "contract": {
            "objective": "sum over work rows of w (y - s f I_Bloch)^2 + kappa c^T P c; stage scale-fit weights times dataset weight",
            "refined": ["xyz (site-symmetry tangent space)", "Uij/Uiso", "thickness centre and uniform-gradient fraction", "frame scales (profiled)"],
            "fixed": ["occupancies", "cell", "orientation and frame offset", "mosaic cone", "dose B", "Sg jitter", "absorption",
                      "beam basis and rocking geometry (frozen at the start model)"],
            "hydrogen_mode": options.hydrogen_mode,
            "hydrogen_treatment": (
                "independent coordinates and ADPs, subject to explicit fixed tokens and site symmetry"
                if options.hydrogen_mode == "independent"
                else "positions translate with the parent; negative Uiso keeps its parent-Ueq multiplier"
            ),
            "free_rows": "predicted and reported only; never fitted, never used to stop, never written to the SHELXL HKL",
            "gradient": "exact: Daleckii-Krein adjoint of the Bloch propagator; torch forward-mode autograd for dU_g/dp",
        },
        "physics": physics_record(problem),
        "checks": solution["checks"],
        "optimizer": {
            "status": fit["status"],
            "converged": fit["converged"],
            "projected_gradient_cosine": fit["projected_gradient_cosine"],
            "iterations": int(max(t["iteration"] for t in trajectory)),
            "trajectory": trajectory,
            "options": asdict(options),
        },
        "evaluations": int(problem.evaluations),
        "heldout": solution["heldout"],
        "thickness": thickness,
        "model": {
            "n_structure_parameters": int(problem.n_struct),
            "n_thickness_parameters": int(problem.n_x - problem.n_struct),
            "max_shift_A": float(max(heavy)) if heavy else 0.0,
            "rms_shift_A": float(math.sqrt(np.mean(np.square(heavy)))) if heavy else 0.0,
            "atom_shifts": shifts,
            "npd_atoms": npd,
            "npd_atoms_start": fit["npd_start"],
            "npd_step_rejections": int(fit["npd_rejections"]),
            "restraints_not_applied": structure.instructions["restraints"],
            "parameters": parameters,
        },
        "shelxl": shelxl_block,
        "leak_flags": {
            **leak_flags(problem),
            "free_rows_in_shelxl_hkl": bool((shelxl_block.get("refined") or {}).get("free_rows_in_hkl", False)),
        },
        "timing": {
            "wall_seconds": time.perf_counter() - t_start,
            "plan_seconds": solution["plans"]["seconds"],
            "evaluation_seconds": problem.eval_seconds,
        },
        "free_r_obs_polished_minus_start_pp": ((free.get("polished_minus_start") or {}).get("r_obs") or {}).get("delta"),
    }
    summary = _jsonable(summary)
    (out_dir / SUMMARY_NAME).write_text(json.dumps(summary, indent=2) + "\n")
    write_report(summary, out_dir / "FINAL_POLISH.md")
    log(f"dynamical polish: {out_dir / SUMMARY_NAME}")
    return summary
