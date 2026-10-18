"""AUTO nested nuisance stage (``--nuisance-selection nested``) and the per-macro-cycle thickness update.

Nested nuisance stage (per crystal, after the screen and the early hand decision)
    Mosaic cone x dose B x thickness distribution x P-spline lambda are selected jointly by inner orbit CV
    (the reference implementation is tests/reference/reference_lib.py select_thickness_table).  The sequential
    alternative (thickness polish, then the cone and dose gates with the thickness frozen) can reject a
    parameter only because the thickness was fixed without it.  Here, per crystal:

    1. an anchor native run of the active (screen) nuisance at screen fidelity writes the row identity, targets,
       sigmas, scan angles, orbit keys and engine split of every row (``trial_rows.npz``) and records the exact
       spectral Config it used;
    2. ``grond.legacy.engine.nested_nuisance_stage`` computes the spectra of every physics variant (cone sigma grid
       x dose total-B grid; window, points and support of the resolved AUTO physics at ``--nested-n-basis``) on work
       rows only and picks (variant, family in ``--nested-families``, its parameters, P-spline lambda) by inner
       3-fold orbit CV R_all (grond.regularize.nested_selection);
    3. the pick (thickness distribution with its resolved evaluator and support, cone, dose, no absorption) is
       replayed once at screen fidelity (``*_nested_selected``) and pinned for every later stage: the in-sample
       row-resolution, Sg-jitter and sigma gates (unchanged), the SHELXL feedback, the final, hand-pair and
       publication stages.  The cone and dose gates do not run; the stage fails closed on any error.

    Everything is recorded in ``decisions/<ds>/nested_nuisance.json`` (request, the full CV R_all grid, the pick,
    edges, evaluator, cost).  ``--nuisance-selection sequential`` restores the thickness polish and the cone/dose
    gates exactly.

Per-macro-cycle thickness update (``--feedback-thickness-update``, default on under nested)
    Before each SHELXL macro-cycle whose entering model was not selected on yet (every cycle after the first), and
    once for the final model, the thickness distribution and lambda are re-selected on the spectra of that
    structure with the physics variant fixed (grond.legacy.engine.nested_nuisance_stage mode ``reselect``), and the
    cycle's correction uses them.  The loop trajectory records the thickness of every cycle, and the final,
    hand-pair and publication stages use the thickness selected for the kept model.
"""

from __future__ import annotations

import argparse
import math
from collections.abc import Mapping
from pathlib import Path

from grond.bloch.constants import BETHE_BACKENDS
from grond.cli_support.values import read_json, try_read_json
from grond.io.config import DatasetSpec, PrepConfig
from grond.paths import engine_script
from grond.regularize.contracts import scale_model_from_args
from grond.regularize.nested_selection import (
    DEFAULT_GAMMA_MEAN_GRID,
    DEFAULT_GAMMA_SHAPE_GRID,
    DEFAULT_WEDGE_GRAD_GRID,
    DEFAULT_WEDGE_T0_GRID,
    SelectionGrids,
    ThicknessSupport,
)
from grond.validation.heldout import DEFAULT_FREE_SALT, INNER_CV_SALT
from grond.workflows.decision import write_json
from grond.workflows.model_selection import best_nuisance, dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import (
    auto_mosaicity_grid,
    dose_b_per_frame_grid,
    feature_is_enabled,
    parse_strict_grid,
    stage_n_points,
)

NUISANCE_SELECTIONS = ("nested", "sequential")
# Feature gates the nested stage replaces (it selects the cone and dose jointly with the thickness).
NESTED_GATE_FAMILIES = ("mosaicity", "dose")
FAMILY_NAMES = {"uniform_gradient": "wedge", "wedge": "wedge", "gamma": "gamma"}
CONTRACT_SCHEMA = "grond.nested-nuisance-contract.v1"
# Nuisance fields the materialized replay must reproduce exactly.
REPLAY_FIELDS = ("thickness_model", "thickness_evaluator", "mosaicity_mode", "dose_b_mode")
REPLAY_NUMBERS = ("thickness_A", "mosaicity_sigma_deg", "dose_b_per_frame", "thickness_min_A", "thickness_max_A",
                  "thickness_table_dt_A")


# ----------------------------------------------------------------------------- policy


def nuisance_selection(args: argparse.Namespace) -> str:
    """Requested nuisance selection; namespaces without the option (older callers) are sequential."""

    value = str(getattr(args, "nuisance_selection", "sequential") or "sequential")
    if value not in NUISANCE_SELECTIONS:
        raise ValueError(f"unknown --nuisance-selection {value!r}; choices are {', '.join(NUISANCE_SELECTIONS)}")
    return value


def nested_selection_enabled(args: argparse.Namespace) -> bool:
    """Whether AUTO runs the nested nuisance stage (never under --fast, which skips all nuisance stages)."""

    return nuisance_selection(args) == "nested" and not bool(getattr(args, "fast", False))


def feedback_thickness_update_mode(args: argparse.Namespace) -> tuple[str, str]:
    """(on/off, reason): on by default under nested; an explicit on without a nested pick is refused."""

    requested = getattr(args, "feedback_thickness_update", None)
    nested = nuisance_selection(args) == "nested"
    if requested is None:
        return ("on", "default_nested") if nested else ("off", "not_applicable_sequential_nuisance_selection")
    requested = str(requested)
    if requested not in {"on", "off"}:
        raise ValueError(f"unknown --feedback-thickness-update {requested!r}; choices are on, off")
    if requested == "on" and not nested:
        raise ValueError(
            "--feedback-thickness-update on re-selects the nested thickness pick each macro-cycle; it needs "
            "--nuisance-selection nested"
        )
    from grond.workflows.run_presets import preset_source

    return requested, preset_source(args, "feedback_thickness_update") or "explicit"


def feedback_thickness_update_enabled(args: argparse.Namespace) -> bool:
    """Return whether the macro-cycles re-select the thickness: the update is on and the nested stage runs."""

    return feedback_thickness_update_mode(args)[0] == "on" and nested_selection_enabled(args)


def nested_families(args: argparse.Namespace) -> tuple[str, ...]:
    """Return the ``--nested-families`` thickness families as engine names; raise when empty or unknown."""

    tokens = [t.strip() for t in str(getattr(args, "nested_families", "uniform_gradient,gamma") or "").split(",") if t.strip()]
    if not tokens:
        raise ValueError("--nested-families is empty; give uniform_gradient and/or gamma")
    unknown = [t for t in tokens if t not in FAMILY_NAMES]
    if unknown:
        raise ValueError(f"--nested-families has unknown families {unknown}; choices are uniform_gradient, gamma")
    return tuple(dict.fromkeys(FAMILY_NAMES[t] for t in tokens))


def nested_grids(args: argparse.Namespace) -> SelectionGrids:
    """Thickness grids of the nested selection (nested_selection defaults) and the run's P-spline scale grid."""

    contract = scale_model_from_args(args)
    return SelectionGrids(
        families=nested_families(args),
        gamma_mean=DEFAULT_GAMMA_MEAN_GRID,
        gamma_shape=DEFAULT_GAMMA_SHAPE_GRID,
        wedge_t0=DEFAULT_WEDGE_T0_GRID,
        wedge_grad=DEFAULT_WEDGE_GRAD_GRID,
        mu=(0.0,),
        n_basis=tuple(int(v) for v in contract.n_basis_grid),
        lam=tuple(float(v) for v in contract.lambda_grid),
    )


# Backends whose Hermitian eigensystem the exact gamma (cf) evaluator needs downstream (grond.bloch.config).
CF_BACKENDS = ("scipy", "numpy_scipy_cpu")
# Backends whose rows are not the exact eigensystems the spectra hold (grond.legacy.engine.spectra_stage).
UNREPRESENTED_BACKENDS = frozenset(BETHE_BACKENDS)


def nested_evaluator(args: argparse.Namespace) -> tuple[str, str]:
    """(evaluator requested from the selection, reason).

    ``auto`` resolves a gamma pick with less than 1e-6 of its mass outside the support to the exact ``cf`` average,
    which the engine computes only on the SciPy backend; with any other --compute-backend the stage requests
    ``table`` (the same support model to its 2 A interpolation error), so no pick can be one the later stages refuse.
    Response tables (--nested-response-tables) keep no spectra, so they always request ``table``.
    """

    if bool(getattr(args, "nested_response_tables", False)):
        return "table", "response_tables_keep_no_spectra"
    backend = str(getattr(args, "compute_backend", "scipy") or "scipy").lower()
    if backend in CF_BACKENDS:
        return "auto", "cf_available_scipy_backend"
    return "table", f"cf_needs_scipy_backend_not_{backend}"


def validate_nested_nuisance_policy(args: argparse.Namespace, prep: PrepConfig | None = None) -> dict:
    """Refuse, before any calculation, settings the nested stage or the thickness update cannot honour."""

    selection = nuisance_selection(args)
    update, update_reason = feedback_thickness_update_mode(args)
    record = {"nuisance_selection": selection, "feedback_thickness_update": update,
              "feedback_thickness_update_reason": update_reason}
    if selection != "nested":
        return record
    if bool(getattr(args, "fast", False)):
        return {**record, "status": "not_applicable_fast", "reason": "--fast skips every nuisance stage"}
    if prep is not None and len(prep.datasets) == 1 and not bool(getattr(args, "auto_decide", True)):
        return {**record, "status": "not_applicable_fixed_workflow", "reason": "--no-auto-decide runs no nuisance stages"}
    if scale_model_from_args(args).kind != "pspline":
        raise ValueError(
            "--nuisance-selection nested scores every candidate with P-spline frame scales refitted by inner orbit "
            "CV; --scale-model per-frame needs --nuisance-selection sequential"
        )
    if int(getattr(args, "nested_n_basis", 0) or 0) < 2:
        raise ValueError("--nested-n-basis must be an integer of at least 2 beams")
    requested_points = str(getattr(args, "nested_n_points", "screen") or "screen").strip().lower()
    if requested_points not in {"screen", "final"}:
        try:
            if int(requested_points) < 2:
                raise ValueError
        except ValueError:
            raise ValueError("--nested-n-points must be screen, final or an integer of at least 2") from None
    if feature_is_enabled(args, "absorption"):
        raise ValueError(
            "--nuisance-selection nested keeps absorption off (uniform mu is degenerate with the thickness "
            "distribution; the spectra need H = H0 + i*mu*I); an absorption grid needs "
            "--nuisance-selection sequential"
        )
    backend = str(getattr(args, "compute_backend", "scipy") or "scipy").lower()
    if backend in UNREPRESENTED_BACKENDS:
        raise ValueError(
            f"--nuisance-selection nested scores candidates on the exact eigensystems of the full row Hamiltonians; "
            f"--compute-backend {backend} predicts rows from a Bethe/Schur-reduced Hamiltonian, so the pick would not "
            "describe its rows; use --compute-backend scipy or --nuisance-selection sequential"
        )
    nested_grids(args)
    auto_mosaicity_grid(args)
    if prep is not None:
        fixed = [ds.name for ds in prep.datasets if _fixed_thickness(args, ds.name)]
        if fixed:
            raise ValueError(
                f"--dataset-thickness-nm/--crystal-parameters-from fix the thickness of {', '.join(fixed)}; the nested "
                "nuisance selection selects the thickness distribution jointly with the cone and dose; use "
                "--nuisance-selection sequential"
            )
    evaluator, evaluator_reason = nested_evaluator(args)
    record.update(nested_n_basis=int(args.nested_n_basis), families=list(nested_families(args)), evaluator=evaluator,
                  evaluator_reason=evaluator_reason)
    return record


def nested_spectra_points(args: argparse.Namespace, ds) -> int:
    """Profile points of the nested spectra: the final points (AUTO default), the screen points, or an explicit count.

    A namespace without the option (internal callers) keeps the first nested build's screen points.
    """

    requested = str(getattr(args, "nested_n_points", "screen") or "screen").strip().lower()
    if requested == "screen":
        return int(stage_n_points(args, ds, screen=True))
    if requested == "final":
        return int(stage_n_points(args, ds, screen=False))
    return int(requested)


def _fixed_thickness(args: argparse.Namespace, name: str) -> dict:
    """Return the thickness fields a fresh AUTO crystal-parameter record pins for ``name``, else {}."""

    record = (getattr(args, "crystal_parameters", {}) or {}).get(name, {}) or {}
    return dict(record.get("fixed_thickness") or {}) if record.get("fresh_auto") else {}


# ----------------------------------------------------------------------------- variants and request


def nested_variants(args: argparse.Namespace, ds: DatasetSpec) -> list[dict]:
    """Physics variants: every cone width of the AUTO cone grid x every dose of the AUTO dose grid."""

    modes, sigmas = auto_mosaicity_grid(args)
    cones: list[tuple[str, float]] = []
    for mode in modes.split(","):
        if mode == "none":
            cones.append(("none", 0.0))
        else:
            cones += [(mode, float(s)) for s in parse_strict_grid(sigmas, "mosaicity sigma")
                      if float(s) > 0.0 or mode == "gaussian_cone"]
    dose_modes = [m for m in str(getattr(args, "dose_b_mode_grid", "none") or "none").replace(",", " ").split() if m]
    doses: list[tuple[str, float, dict]] = [("none", 0.0, {"dose_b_total_A2": 0.0})] if "none" in dose_modes else []
    if "linear_frame" in dose_modes:
        per_frame, conversion = dose_b_per_frame_grid(args, ds)
        values = parse_strict_grid(per_frame, "dose per-frame B")
        totals = conversion.get("total_b_grid_A2")
        for i, value in enumerate(values):
            params = {"dose_b_total_A2": float(totals[i])} if totals else {"dose_b_per_frame": float(value)}
            if float(value) == 0.0 and doses:
                continue  # linear_frame at 0 per frame is the no-dose physics already listed
            doses.append(("linear_frame", float(value), params))
    if not doses:
        raise ValueError(f"--dose-b-mode-grid {args.dose_b_mode_grid!r} gives no dose candidate")
    points = int(getattr(args, "mosaicity_points", 3))
    variants = []
    for mode, sigma in cones:
        for dose_mode, per_frame, dose_params in doses:
            config = {"mosaicity_mode": mode, "mosaicity_sigma_deg": float(sigma),
                      "dose_b_mode": dose_mode, "dose_b_per_frame": float(per_frame)}
            if mode != "none":
                config["mosaicity_points"] = points
            if mode == "gaussian_cone":
                for field in ("mosaicity_support_sigma_deg", "mosaicity_support_points", "mosaicity_support_width_sigma"):
                    config[field] = getattr(args, field)
            total = dose_params.get("dose_b_total_A2", dose_params.get("dose_b_per_frame"))
            variants.append({"label": f"cone{sigma:g}_dose{total:g}", "config": config,
                             "params": {"mosaicity_sigma_deg": float(sigma), **dose_params}})
    return variants


def selection_row_policy(args: argparse.Namespace) -> dict:
    """The orbit-hash free set and the orbit exclusion the stage removes on top of the engine split."""

    policy: dict = {"extra_free": None, "exclude_orbits_json": ""}
    if str(getattr(args, "split_scheme", "legacy")) == "orbit-hash":
        policy["extra_free"] = {"salt": str(getattr(args, "split_salt", "") or DEFAULT_FREE_SALT),
                                "fraction": float(getattr(args, "free_fraction", 0.10))}
    exclusion = str(getattr(args, "exclude_orbits_json", "") or "")
    if exclusion:
        policy["exclude_orbits_json"] = str(Path(exclusion).expanduser().resolve())
    return policy


def nested_request(args: argparse.Namespace, ds: DatasetSpec, *, model: str, rows_npz: Path, base_config: dict) -> dict:
    """Build the request the nested nuisance stage engine reads for one crystal."""

    from grond.legacy.engine.nested_nuisance_stage import REQUEST_SCHEMA

    contract = scale_model_from_args(args)
    return {
        "schema": REQUEST_SCHEMA,
        "mode": "select",
        "dataset": ds.name,
        "hkl": str(ds.hkl),
        "hkl_transform": str(ds.hkl_transform or "identity"),
        "res": str(model),
        "rows_npz": str(rows_npz),
        "rows_trial_index": 0,
        **selection_row_policy(args),
        "base_config": dict(base_config),
        "variants": nested_variants(args, ds),
        "grids": nested_grids(args).to_dict(),
        "support": ThicknessSupport().to_dict(),
        "evaluator": nested_evaluator(args)[0],
        "score_with": "table",
        **({"response_tables": True} if getattr(args, "nested_response_tables", False) else {}),
        "clamp": bool(contract.clamp),
        "objective": str(contract.objective),
        "salt": INNER_CV_SALT,
        "workers": int(getattr(args, "workers", 1) or 1),
    }


def nested_contract_path(root: Path, ds_name: str) -> Path:
    """Return the path of one crystal's nested nuisance contract (decisions/<name>/nested_nuisance.json)."""

    return Path(root) / "decisions" / ds_name / "nested_nuisance.json"


def stage_command(python: str, request_path: Path, result_path: Path, workers: int) -> list[str]:
    """Return the command line that runs the nested nuisance stage engine on one request."""

    return [python, str(engine_script("nested_nuisance_stage.py")), "--request", str(request_path), "--out",
            str(result_path), "--workers", str(max(1, int(workers)))]


def nested_gate_skip_record(family: str) -> dict:
    """Return the record of a feature gate skipped because the nested stage selects that family."""

    return {"family": family, "status": "skipped",
            "reason": "selected jointly with the thickness by the nested nuisance stage (--nuisance-selection nested)"}


def _replay_drift(pinned: Mapping[str, object], replayed: Mapping[str, object]) -> list[str]:
    """List the nuisance fields a materialized replay did not reproduce, as ``key old -> new`` strings."""

    drift = [f"{key} {pinned.get(key)!r} -> {replayed.get(key)!r}" for key in REPLAY_FIELDS
             if key in pinned and str(replayed.get(key)) != str(pinned.get(key))]
    numbers = list(REPLAY_NUMBERS) + (["thickness_gamma_shape"] if pinned.get("thickness_model") == "gamma"
                                      else ["thickness_grad_frac"])
    for key in numbers:
        if key not in pinned:
            continue
        try:
            same = math.isclose(float(replayed.get(key)), float(pinned[key]), rel_tol=1e-9, abs_tol=1e-15)
        except (TypeError, ValueError):
            same = False
        if not same:
            drift.append(f"{key} {pinned.get(key)!r} -> {replayed.get(key)!r}")
    return drift


# ----------------------------------------------------------------------------- the stage


def run_nested_nuisance_stage(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    active_model: str,
    active_nui: dict,
    run_prefix: str,
) -> tuple[dict, Path, dict]:
    """Anchor rows, nested selection and the pinned replay for one crystal; returns (nuisance, run, record).

    Any failure raises: the stage replaces the thickness polish and the cone/dose gates, so continuing with the
    screen nuisance would be a silent fallback.
    """

    from grond.cli_support.execution import run_logged
    from grond.legacy.engine.nested_nuisance_stage import spectral_base_config

    policy = validate_nested_nuisance_policy(args, prep)
    root = Path(root)
    stage_root = root / "runs" / ds.name / f"{run_prefix}_nested_nuisance"
    anchor = stage_root / "anchor_rows"
    rows_npz = anchor / "trial_rows.npz"
    screen_points = stage_n_points(args, ds, screen=True)
    spectra_points = nested_spectra_points(args, ds)
    record: dict = {
        "family": "nested_nuisance",
        "selection": "nested_inner_orbit_cv_work_rows",
        "selection_rows": "work",
        "score_rows": "work",
        "scale_fit_split": "work",
        "free_rows_in_selection": False,
        "heldout_scheme": {"kind": "inner_orbit_cv_work_rows", "n_folds": 3, "salt": INNER_CV_SALT,
                           "free_rows_used": False},
        "anchor_run": str(anchor),
        "contract": str(nested_contract_path(root, ds.name)),
        "nested_n_basis": int(args.nested_n_basis),
        "nested_n_points": int(spectra_points),
    }
    run_logged(
        native_stage_cmd(
            python, ds, active_model, dataset_baseline_hkl(prep, ds), anchor, args,
            n_basis=args.screen_n_basis, n_points=screen_points, stage=f"{run_prefix}_nested_rows",
            pinned=active_nui, make_maps=False, endpoint=False, scale_fit_split_override="work",
            allow_pinned_jitter_fallback=False, trial_rows_npz=rows_npz,
        ),
        logs,
        f"{ds.name}_{run_prefix}_nested_rows",
        dry_run=args.dry_run,
    )
    if args.dry_run:
        return active_nui, anchor, {**record, "status": "planned"}
    base = spectral_base_config(anchor, n_basis=int(args.nested_n_basis), n_points=int(spectra_points))
    request = nested_request(args, ds, model=active_model, rows_npz=rows_npz, base_config=base)
    request_path = stage_root / "request.json"
    result_path = stage_root / "result.json"
    write_json(request_path, request)
    run_logged(stage_command(python, request_path, result_path, int(args.workers)), logs,
               f"{ds.name}_{run_prefix}_nested_nuisance", dry_run=False)
    result = read_json(result_path)
    pinned = {**active_nui, **result["pinned_nuisance"]}
    selected = root / "runs" / ds.name / f"{run_prefix}_nested_selected"
    run_logged(
        native_stage_cmd(
            python, ds, active_model, dataset_baseline_hkl(prep, ds), selected, args,
            n_basis=args.screen_n_basis, n_points=screen_points, stage=f"{run_prefix}_nested_selected",
            pinned=pinned, make_maps=False, endpoint=False, scale_fit_split_override="work",
            allow_pinned_jitter_fallback=False,
        ),
        logs,
        f"{ds.name}_{run_prefix}_nested_selected",
        dry_run=False,
    )
    materialized = best_nuisance(selected)
    drift = _replay_drift(result["pinned_nuisance"], materialized)
    if drift:
        raise RuntimeError(f"{ds.name} nested nuisance replay did not reproduce the pick: " + "; ".join(drift))
    contract = {
        "schema": CONTRACT_SCHEMA,
        "status": "selected",
        "dataset": ds.name,
        "policy": policy,
        "request_path": str(request_path),
        "result_path": str(result_path),
        "anchor_run": str(anchor),
        "materialized_run": str(selected),
        "request": request,
        "result": result,
        "pinned_nuisance": result["pinned_nuisance"],
        "materialized_nuisance": materialized,
    }
    write_json(nested_contract_path(root, ds.name), contract)
    t = result["thickness"]
    params = ", ".join(f"{k} {t[k]:g}" for k in ("mean", "shape", "t0", "grad") if k in t)
    print(f"nested nuisance {ds.name}: {result['pick']['variant']}, {t['family']} {params}, lambda {t['scale_lambda']:g}, "
          f"evaluator {t['evaluator']}, inner CV R_all {t['inner_r_all']:.3f}", flush=True)
    if result["edges"]:
        print(f"WARN nested nuisance {ds.name}: the pick is on the grid edge of {', '.join(result['edges'])} "
              f"(recorded in {nested_contract_path(root, ds.name)}); the grids are not extended automatically", flush=True)
    support_bound = [name for name in ("mean", "t0") if name in (result.get("at_physical_bound") or [])]
    if support_bound:
        print(f"WARN nested nuisance {ds.name}: the thickness {support_bound[0]} is at the support bound "
              f"({t[support_bound[0]]:g} A, {t.get('mass_outside_support', float('nan')):.0%} of the distribution outside "
              f"the support); with {spectra_points} profile points a coarse screen favours the thin-crystal limit",
              flush=True)
    record.update(
        status="selected",
        run=str(selected),
        selected_nuisance=result["pinned_nuisance"],
        thickness=result["thickness"],
        variant=result["pick"]["variant"],
        variant_params=result["pick"].get("variant_params"),
        edges=result["edges"],
        at_physical_bound=result["at_physical_bound"],
        grid_size=result["grid"]["size"],
        rows=result["rows"],
        cost=result["cost"],
    )
    return materialized, selected, record


def nested_contract(root: Path, ds_name: str) -> dict:
    """The recorded nested nuisance contract of a crystal; fails closed when it is missing or not selected."""

    path = nested_contract_path(root, ds_name)
    payload = try_read_json(path)
    if not payload or payload.get("status") != "selected":
        raise RuntimeError(f"{ds_name}: the feedback thickness update needs a selected nested nuisance contract at {path}")
    return payload


# ----------------------------------------------------------------------------- feedback thickness update


def nested_feedback_args(args: argparse.Namespace, root: Path, ds: DatasetSpec) -> list[str]:
    """Options of the single-dataset SHELXL feedback loop for the per-macro-cycle thickness update."""

    if not feedback_thickness_update_enabled(args):
        return []
    if not args.dry_run:
        nested_contract(root, ds.name)
    return ["--feedback-thickness-update", "on", "--nested-nuisance-contract", str(nested_contract_path(root, ds.name))]


class MultiFeedbackThickness:
    """Per-crystal thickness per structure for the shared multi-crystal macro-cycles (run_logged stages)."""

    def __init__(self, prep: PrepConfig, root: Path, logs: Path, python: str, workers: int):
        """Set up one thickness tracker per crystal from its selected nested nuisance contract.

        Re-selections are written under runs/shared_shelxl_feedback/thickness_updates/<crystal>.
        """

        from grond.cli_support.execution import run_logged
        from grond.legacy.engine.nested_nuisance_stage import FeedbackThickness

        self.logs = logs
        self.crystals = {}
        for ds in prep.datasets:
            out_dir = Path(root) / "runs" / "shared_shelxl_feedback" / "thickness_updates" / ds.name

            def run_stage(request_path, result_path, label, name=ds.name):
                """Run one thickness re-selection of this crystal as a logged stage."""
                run_logged(stage_command(python, Path(request_path), Path(result_path), workers), logs,
                           f"{name}_feedback_thickness_{label}", dry_run=False)

            nested_contract(root, ds.name)
            self.crystals[ds.name] = FeedbackThickness(nested_contract_path(root, ds.name), out_dir, run_stage)

    def pinned(self, name: str, nuisance: Mapping[str, object], model: str, base_run: Path | None, label: str) -> tuple[dict, dict]:
        """(nuisance with the thickness selected for ``model``, its record); re-selects an unknown model."""

        from grond.legacy.engine.nested_nuisance_stage import apply_thickness

        tracker = self.crystals[name]
        if not tracker.known(model):
            if base_run is None:
                raise RuntimeError(
                    f"{name}: the macro-cycles start from a model the nested nuisance stage did not select on ({model})"
                )
            tracker.update(model, base_run, label)
        return apply_thickness(dict(nuisance), tracker.fields(model)), tracker.record(model)

    def drift(self, name: str, model: str, recorded: Mapping[str, object]) -> list[str]:
        """Thickness fields a cycle's native run did not reproduce for ``model`` (nested_nuisance_stage.thickness_drift)."""

        from grond.legacy.engine.nested_nuisance_stage import thickness_drift

        return thickness_drift(self.crystals[name].fields(model), dict(recorded))

    def final(self, model: str) -> dict:
        """Return each crystal's thickness record and fields for ``model`` (crystals that know it)."""

        return {name: {"record": tracker.record(model), "fields": tracker.fields(model)}
                for name, tracker in self.crystals.items() if tracker.known(model)}

    def summary(self) -> dict:
        """Return each crystal's contract path and its list of thickness updates."""

        return {name: {"contract": str(tracker.contract_path), "updates": tracker.updates}
                for name, tracker in self.crystals.items()}


def feedback_final_nuisance(nuisance: Mapping[str, dict], structure_feedback: Mapping[str, object]) -> dict[str, dict]:
    """Per-crystal nuisance with the thickness selected for the kept feedback model (unchanged without an update)."""

    from grond.legacy.engine.nested_nuisance_stage import apply_thickness

    update = structure_feedback.get("thickness_update") if isinstance(structure_feedback, Mapping) else None
    final = (update or {}).get("final") if isinstance(update, Mapping) else None
    if isinstance(update, Mapping) and update.get("enabled"):
        # With the update on, a crystal without a thickness for the kept model would be corrected with the thickness
        # of another structure; refuse instead of keeping the earlier pick.
        missing = sorted(set(nuisance) - set(final or {}))
        if missing:
            raise RuntimeError(
                f"the feedback thickness update recorded no thickness for the kept model of {', '.join(missing)}"
            )
    if not final:
        return dict(nuisance)
    out = dict(nuisance)
    for name, entry in final.items():
        if name in out:
            out[name] = apply_thickness(out[name], entry["fields"])
    return out
