"""Held-out selection for the AUTO physics feature gates (mosaic cone and dose B).

The other feature gates (row resolution, absorption, Sg jitter, sigma model) compare one grid run
with the active run by ``score_native_summary``: all-row wR2 + 0.25 * free wR2 + 0.10 * all-row R1 plus
Born sanity penalties, with scales fitted on every row under ``--production-scales``.  That score is
in-sample, and its free term sees free rows that were inside the scale fit.

The mosaic-cone and dose gates instead compare every candidate of one grid run on identical work rows
by orbit-grouped cross-validation (grond.validation.heldout): 3 salted crystal-independent orbit folds,
scales refitted on the training orbits of each fold, amplitude R on the held-out orbits.  Free rows (the
engine split, plus the crystal-independent orbit-hash free set when that scheme is requested) and
excluded (lockbox) orbits are never used.  Every candidate is scored with P-spline frame scales at
every (n_basis, lambda) of the run's own scale contract (--scale-spline-n-basis-grid x
--scale-spline-lambda-grid) and keeps its lowest cross-validated R_all over all development rows, which
chooses the physics variant jointly with the scale setting; that setting's out-of-fold predictions enter
the bootstrap and it is recorded per candidate.  The lowest cross-validated R_all is selected and replaces the active physics only when the paired merged-orbit
bootstrap of candidate minus active puts the R_obs change below 0 and the R_all change at most +0.05 pp.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np

from grond.cli_support.values import read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.regularize.contracts import ScaleModel, scale_model_from_args
from grond.regularize.selection import nuisance_reference_matches
from grond.regularize.trial_rows import load_trial_rows_npz
from grond.validation.heldout import (
    BOOTSTRAP_SEED,
    DEFAULT_FREE_SALT,
    INNER_CV_SALT,
    N_BOOT,
    cv_predictions,
    free_mask,
    hash_fold,
    load_orbit_exclusion,
    paired_orbit_bootstrap,
    pspline_fit_predict,
    r_metrics,
    shell_index,
)
from grond.workflows.model_selection import best_nuisance, dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import dose_b_per_frame_grid, feature_grid, stage_n_points

HELDOUT_GATE_FAMILIES = ("mosaicity", "dose")
FAMILY_FIELDS = {
    "mosaicity": ("mosaicity_mode", "mosaicity_sigma_deg", "mosaicity_points"),
    "dose": ("dose_b_mode", "dose_b_per_frame"),
}
GATE_CV_FOLDS = 3
GATE_CV_SALT = INNER_CV_SALT
GATE_RULE = {"r_obs_delta_ci95_upper_lt_pp": 0.0, "r_all_delta_ci95_upper_le_pp": 0.05}
MIN_GATE_ROWS = 50


def uses_heldout_gate(args: argparse.Namespace, family: str) -> bool:
    """Return true when this family is scored by orbit cross-validation instead of the in-sample score."""

    return family in HELDOUT_GATE_FAMILIES


def _family_matches(family: str, reference: dict, nuisance: dict) -> bool:
    """Whether a candidate carries the reference values of the gated family."""

    if family == "mosaicity":
        mode_ref, mode = str(reference.get("mosaicity_mode", "none")), str(nuisance.get("mosaicity_mode", "none"))
        values = ("mosaicity_sigma_deg",)
    else:
        mode_ref, mode = str(reference.get("dose_b_mode", "none")), str(nuisance.get("dose_b_mode", "none"))
        values = ("dose_b_per_frame",)
        # The engine applies no dose term for mode none or for linear_frame at 0 per frame (row_delta_b),
        # so both are the same no-dose physics; a mode-none reference carries no per-frame value.
        try:
            dose_ref = 0.0 if mode_ref == "none" else float(reference.get("dose_b_per_frame", 0.0))
            dose = 0.0 if mode == "none" else float(nuisance.get("dose_b_per_frame", 0.0))
        except (TypeError, ValueError):
            return False
        if dose_ref == 0.0 and dose == 0.0:
            return True
    if mode_ref != mode:
        return False
    for key in values:
        try:
            if not math.isclose(float(reference.get(key, 0.0)), float(nuisance.get(key, 0.0)), rel_tol=1e-12, abs_tol=1e-15):
                return False
        except (TypeError, ValueError):
            return False
    return True


def baseline_candidate_index(family: str, active_nuisance: dict, candidates: list[dict]) -> int:
    """Index of the candidate that reproduces the active nuisance; fail closed when absent."""

    other = {key: value for key, value in active_nuisance.items() if key not in FAMILY_FIELDS[family]}
    for index, nuisance in enumerate(candidates):
        if _family_matches(family, active_nuisance, nuisance) and nuisance_reference_matches(other, nuisance):
            return index
    raise RuntimeError(
        f"held-out {family} gate grid does not contain the active nuisance; the candidates cannot be compared with it"
    )


def _family_view(family: str, nuisance: dict) -> dict:
    """Return the nuisance fields that belong to one gate family (mosaicity or dose)."""

    return {key: nuisance.get(key) for key in FAMILY_FIELDS[family]}


def _gate_scale_settings(contract: ScaleModel) -> list[dict]:
    """Every (n_basis, lambda) setting the gate scores a candidate at."""

    return [
        {"n_basis": int(n_basis), "lambda": float(lam), "objective": str(contract.objective)}
        for n_basis in contract.n_basis_grid
        for lam in contract.lambda_grid
    ]


def score_trial_candidates(
    rows: dict,
    *,
    family: str,
    active_nuisance: dict,
    scale_contract: ScaleModel | None = None,
    n_folds: int = GATE_CV_FOLDS,
    salt: str = GATE_CV_SALT,
    n_boot: int = N_BOOT,
    seed: int = BOOTSTRAP_SEED,
    extra_free: np.ndarray | None = None,
    excluded_orbits: set[str] | frozenset[str] = frozenset(),
) -> dict:
    """Cross-validate every candidate of one gate run on identical work rows and apply the adoption rule.

    extra_free marks further held-out rows (the orbit-hash free set); excluded_orbits are dropped entirely.
    scale_contract is the run's P-spline contract (scale_model_from_args); without it the default contract is
    used.  Each candidate keeps the (n_basis, lambda) of its lowest cross-validated R_all over all development
    rows, recorded as its ``scale``.
    """

    contract = ScaleModel() if scale_contract is None else scale_contract
    settings = _gate_scale_settings(contract)
    clamp = bool(contract.clamp)
    candidates = list(rows["nuisance"])
    base = baseline_candidate_index(family, active_nuisance, candidates)
    target = np.asarray(rows["target"], dtype=float)
    sigma = np.asarray(rows["sigma"], dtype=float)
    all_keys = np.asarray(rows["orbit_key"]).astype(str)
    excluded = np.isin(all_keys, sorted(excluded_orbits)) if excluded_orbits else np.zeros(len(all_keys), dtype=bool)
    held_out = np.asarray(rows["free"], dtype=bool) | (
        np.zeros(len(all_keys), dtype=bool) if extra_free is None else np.asarray(extra_free, dtype=bool)
    )
    sel = (
        np.asarray(rows["work"], dtype=bool)
        & ~held_out
        & ~excluded
        & np.all(np.asarray(rows["valid"], dtype=bool), axis=0)
        & np.isfinite(target)
        & np.isfinite(sigma)
        & (sigma > 0.0)
    )
    if int(sel.sum()) < MIN_GATE_ROWS:
        raise RuntimeError(f"held-out {family} gate has only {int(sel.sum())} common work rows")
    y, sig = target[sel], sigma[sel]
    group = np.asarray(rows["group"], dtype=int)[sel]
    angle = np.asarray(rows["angle_deg"], dtype=float)[sel]
    keys = all_keys[sel]
    folds = hash_fold(list(keys), salt, n_folds)
    if len(np.unique(folds)) < n_folds:
        raise RuntimeError(f"held-out {family} gate rows do not populate all {n_folds} orbit folds")
    shell = shell_index(np.asarray(rows["d_A"], dtype=float)[sel])
    predictions: list[np.ndarray] = []
    scored: list[dict] = []
    for index, nuisance in enumerate(candidates):
        raw = np.asarray(rows["dyn_raw"][index], dtype=float)[sel]
        trials = []
        for setting in settings:
            def fit(train, raw=raw, setting=setting):
                """Fit P-spline scales on the ``train`` rows at this (n_basis, lambda); return the prediction."""
                return pspline_fit_predict(y, sig, raw, angle, group, train, setting, clamp=clamp)["prediction"]
            oof = cv_predictions(fit, len(y), folds)
            trials.append((setting, oof, r_metrics(y, sig, oof)))
        # Lowest cross-validated R_all over the scale grid; the first setting wins ties (grid order).
        chosen = min(range(len(trials)), key=lambda i: (_finite_or_inf(trials[i][2]["r_all"]), i))
        setting, oof, metrics = trials[chosen]
        predictions.append(oof)
        entry = {"index": index, **_family_view(family, nuisance), "cv_r_all": metrics["r_all"],
                 "cv_r_obs": metrics["r_obs"], "n_rows": metrics["n_all"], "n_obs": metrics["n_obs"],
                 "scale": dict(setting)}
        entry["scale_grid"] = [{"n_basis": s["n_basis"], "lambda": s["lambda"], "cv_r_all": m["r_all"]}
                               for s, _oof, m in trials]
        scored.append(entry)
    order = sorted(range(len(scored)), key=lambda i: (scored[i]["cv_r_all"], i != base, i))
    best = order[0]
    if not math.isfinite(float(scored[best]["cv_r_all"])):
        raise RuntimeError(f"held-out {family} gate produced no finite cross-validated R")
    paired = None
    reasons: list[str] = []
    if best == base or _family_matches(family, candidates[base], candidates[best]):
        reasons.append("active_physics_has_the_lowest_cv_r_all")
    else:
        paired = paired_orbit_bootstrap(y, sig, predictions[best], predictions[base], keys, shell,
                                        np.ones(len(y), dtype=bool), n_boot=n_boot, seed=seed)
        if paired["r_obs"].get("ci_status"):
            reasons.append("r_obs_ci_unavailable")
        elif not paired["r_obs"]["ci95"][1] < GATE_RULE["r_obs_delta_ci95_upper_lt_pp"]:
            reasons.append("r_obs_gain_ci_includes_zero")
        if paired["r_all"].get("ci_status"):
            reasons.append("r_all_ci_unavailable")
        elif not paired["r_all"]["ci95"][1] <= GATE_RULE["r_all_delta_ci95_upper_le_pp"]:
            reasons.append("r_all_change_ci_above_tolerance")
    accepted = paired is not None and not reasons
    return {
        "accepted": bool(accepted),
        "baseline_index": int(base),
        "selected_index": int(best),
        "selected_nuisance": dict(candidates[best]),
        "candidates": scored,
        "paired_orbit_bootstrap_selected_minus_active": paired,
        "rejection_reasons": reasons,
        "rule": dict(GATE_RULE),
        "heldout_scheme": {"kind": "orbit_cv_work_rows", "n_folds": int(n_folds), "salt": salt,
                           "free_rows_used": False, "n_rows": int(len(y)), "n_orbits": int(len(np.unique(keys))),
                           "n_free_rows_skipped": int(np.count_nonzero(held_out)),
                           "n_excluded_rows": int(np.count_nonzero(excluded))},
        "scale_model": "pspline",
        "scale_config": {**contract.to_dict(), "selection": "per_candidate_min_cv_r_all_over_grid",
                         "selected": dict(scored[best]["scale"]), "baseline": dict(scored[base]["scale"])},
    }


def _finite_or_inf(value: object) -> float:
    """Return ``value`` as a float, or +inf when it is missing or not finite (sorts last)."""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return math.inf
    return number if math.isfinite(number) else math.inf


def gate_row_policy(args: argparse.Namespace, rows: dict) -> dict:
    """Further rows the gate must not score: the orbit-hash free set and excluded (lockbox) orbits."""

    policy: dict = {}
    if str(getattr(args, "split_scheme", "legacy")) == "orbit-hash":
        keys = [str(key) for key in np.asarray(rows["orbit_key"])]
        salt = str(getattr(args, "split_salt", "") or DEFAULT_FREE_SALT)
        policy["extra_free"] = free_mask(keys, salt, float(getattr(args, "free_fraction", 0.10)))
    exclusion = str(getattr(args, "exclude_orbits_json", "") or "")
    if exclusion:
        policy["excluded_orbits"] = frozenset(load_orbit_exclusion(exclusion)["orbits"])
    return policy


def run_heldout_feature_gate(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    family: str,
    active_model: str,
    active_run: Path,
    active_nui: dict,
) -> tuple[dict, Path, dict]:
    """Run one physics gate grid on work-row scales, cross-validate every candidate, and promote the winner."""

    from grond.cli_support.execution import run_logged

    gate_dir = root / "runs" / ds.name / f"adaptive_{family}_gate"
    trial_rows = gate_dir / "trial_rows.npz"
    record: dict = {"family": family, "run": str(gate_dir), "selection": "orbit_cv_work_rows", "scale_fit_split": "work"}
    try:
        if family == "dose":
            record["dose_grid"] = dose_b_per_frame_grid(args, ds)[1]
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                gate_dir,
                args,
                n_basis=args.screen_n_basis,
                n_points=stage_n_points(args, ds, screen=True),
                stage=f"adaptive_{family}_gate",
                pinned=feature_grid(args, family, active_nui, ds=ds),
                make_maps=False,
                endpoint=False,
                scale_fit_split_override="work",
                allow_pinned_jitter_fallback=False,
                trial_rows_npz=trial_rows,
            ),
            logs,
            f"{ds.name}_adaptive_{family}_gate",
            dry_run=args.dry_run,
        )
        if args.dry_run:
            return active_nui, active_run, {**record, "status": "planned"}
        rows = load_trial_rows_npz(trial_rows)
        # The run's own scale contract: the grids the engine stages and the held-out hand refit use.
        scale_contract = scale_model_from_args(args)
        scored = score_trial_candidates(
            rows, family=family, active_nuisance=dict(active_nui), scale_contract=scale_contract,
            **gate_row_policy(args, rows)
        )
    except Exception as exc:
        return active_nui, active_run, {**record, "status": "failed", "error": str(exc)}
    record.update({"status": "accepted" if scored["accepted"] else "rejected", **scored})
    if not scored["accepted"]:
        return active_nui, active_run, record
    # Replay the winner once on the full row contract so the promoted run, its rows and its
    # nuisance are one candidate, with the active run's scale-fit split.
    selected_dir = root / "runs" / ds.name / f"adaptive_{family}_selected"
    active_split = str((read_json(active_run / "summary.json").get("config") or {}).get("scale_fit_split") or "")
    try:
        run_logged(
            native_stage_cmd(
                python,
                ds,
                active_model,
                dataset_baseline_hkl(prep, ds),
                selected_dir,
                args,
                n_basis=args.screen_n_basis,
                n_points=stage_n_points(args, ds, screen=True),
                stage=f"adaptive_{family}_selected",
                pinned=scored["selected_nuisance"],
                make_maps=False,
                endpoint=False,
                scale_fit_split_override=active_split or None,
                allow_pinned_jitter_fallback=False,
            ),
            logs,
            f"{ds.name}_adaptive_{family}_selected",
            dry_run=False,
        )
        materialized = best_nuisance(selected_dir)
        if not _family_matches(family, scored["selected_nuisance"], materialized):
            raise RuntimeError(
                f"replayed {family} candidate changed the selected values: "
                f"{_family_view(family, scored['selected_nuisance'])} -> {_family_view(family, materialized)}"
            )
    except Exception as exc:
        return active_nui, active_run, {**record, "status": "failed", "error": f"selected-candidate replay: {exc}"}
    record["selected_run"] = str(selected_dir)
    return materialized, selected_dir, record
