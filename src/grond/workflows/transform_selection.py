"""Scout proper HKL transforms with geometry checks and a fixed observation split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from grond.cli_support.execution import (
    TRANSFORM_RERANK_TOP,
    TRANSFORM_SCOUT_WORKERS_PER_JOB,
    run_logged,
    transform_scout_job_count,
    transform_scout_semaphore,
)
from grond.cli_support.values import read_json
from grond.hashing import sha256_file
from grond.hkl_transform import (
    absolute_configuration_claim_contract,
    parse_hkl_transform_info,
    source_orientation_contract,
)
from grond.io.config import DatasetSpec, PrepConfig
from grond.workflows.decision import (
    ALL_PROPER_SIGNED_PERMUTATIONS,
    expand_transform_candidates,
    safe_float,
    safe_label,
    score_transform_summary,
    write_json,
)
from grond.workflows.hand_scout import _dataset_decision_identity
from grond.workflows.model_selection import dataset_baseline_hkl
from grond.workflows.native_commands import native_stage_cmd
from grond.workflows.nuisance_policy import transform_scout_nuisance
from grond.workflows.parity_evidence import replay_transform_work_born
from grond.workflows.transform_metadata import metadata_prefilter_transforms

TRANSFORM_SCOUT_SG_WINDOW = 0.004  # 1/A; the window the transform ranking was validated with
# Transform scouts: coarse stage at N=8/Q=3 on at most 1000 rows, then the top candidates at N=24/Q=10 on the same rows.
SCOUT_MAX_ROWS = 1000
SCOUT_N_BASIS = 8
SCOUT_N_POINTS = 3
TRANSFORM_RERANK_N_BASIS = 24
TRANSFORM_RERANK_N_POINTS = 10
# Maximum relative Frobenius mismatch of the reciprocal-cell metrics before a signed-axis transform is rejected.
TRANSFORM_METRIC_TOLERANCE = 0.02


def transform_scout_window_overrides() -> dict:
    """native_stage_cmd window overrides of a transform scout.

    The scouts use the fixed 0.004 1/A window the transform ranking was validated with; the AUTO physics window
    (0.010 or the per-row DIALS bbox) is for the refinement stages, not a Q3/Q10 Born scout.
    """

    return {"sg_window_override": TRANSFORM_SCOUT_SG_WINDOW, "sg_window_mode_override": "fixed"}


def dataset_with_transform(ds: DatasetSpec, transform: str) -> DatasetSpec:
    """Return a dataset copy using a different HKL transform."""

    return DatasetSpec(
        name=ds.name,
        hkl=ds.hkl,
        xds_dir=ds.xds_dir,
        baseline_hkl=ds.baseline_hkl,
        hkl_transform=transform,
        min_d_spacing=ds.min_d_spacing,
        min_peak=ds.min_peak,
        target_mode=ds.target_mode,
        include_nonpositive=ds.include_nonpositive,
        geometry_percentile=ds.geometry_percentile,
        split_contract=ds.split_contract,
        physical_crystal_id=ds.physical_crystal_id,
        acquisition_fingerprint=ds.acquisition_fingerprint,
    )


def dataset_with_split_contract(ds: DatasetSpec, split_contract: str) -> DatasetSpec:
    """Return a dataset copy bound to one immutable raw-row work/free split."""

    return DatasetSpec(
        name=ds.name,
        hkl=ds.hkl,
        xds_dir=ds.xds_dir,
        baseline_hkl=ds.baseline_hkl,
        hkl_transform=ds.hkl_transform,
        min_d_spacing=ds.min_d_spacing,
        min_peak=ds.min_peak,
        target_mode=ds.target_mode,
        include_nonpositive=ds.include_nonpositive,
        geometry_percentile=ds.geometry_percentile,
        split_contract=str(split_contract),
        physical_crystal_id=ds.physical_crystal_id,
        acquisition_fingerprint=ds.acquisition_fingerprint,
    )


def _transform_candidate_generation_contract(
    requested_spec: str,
    generated_candidates: list[str],
    retained_candidates: list[str],
    metric_filter: Mapping[str, object],
    *,
    parity_policy: str,
) -> dict[str, object]:
    """Describe whether candidate generation is calibration-grade automatic."""

    requested = str(requested_spec or "auto").strip()
    tokens = [token.strip().lower() for token in requested.split(";") if token.strip()]
    automatic_only = bool(tokens) and all(token in {"auto", "common"} for token in tokens)
    canonical_candidates = [parse_hkl_transform_info(candidate).canonical for candidate in generated_candidates]
    required = [parse_hkl_transform_info(candidate).canonical for candidate in ALL_PROPER_SIGNED_PERMUTATIONS]
    exact_full24_proper_set = bool(
        len(canonical_candidates) == len(required) and set(canonical_candidates) == set(required)
    )
    canonical_retained = [parse_hkl_transform_info(candidate).canonical for candidate in retained_candidates]
    metric_filter_valid = bool(
        metric_filter.get("policy") == "source_model_reciprocal_cell_metric_only_v1"
        and metric_filter.get("uses_intensities") is False
        and metric_filter.get("uses_atomic_coordinates_or_hand") is False
        and metric_filter.get("uses_atom_types_or_known_chemistry") is False
        and metric_filter.get("uses_known_configuration") is False
        and list(metric_filter.get("retained_transforms") or []) == canonical_retained
    )
    calibration_grade = bool(
        parity_policy == "proper-only"
        and automatic_only
        and exact_full24_proper_set
        and metric_filter_valid
        and canonical_retained
        and all(parse_hkl_transform_info(candidate).proper for candidate in canonical_retained)
    )
    return {
        "policy": (
            "automatic_full24_proper_metric_filtered_v1" if calibration_grade else "custom_or_restricted_candidates_v1"
        ),
        "requested_spec": requested,
        "automatic_tokens_only": automatic_only,
        "full_proper_signed_permutation_universe": required,
        "exact_full24_proper_set": exact_full24_proper_set,
        "generated_candidates_before_metadata": canonical_candidates,
        "candidate_count_before_metadata": len(canonical_candidates),
        "metric_filter": dict(metric_filter),
        "retained_candidates_after_metric_filter": canonical_retained,
        "metric_filter_contract_valid": metric_filter_valid,
        "calibration_grade_candidate_generation": calibration_grade,
    }


def dataset_transform_candidates(
    prep: PrepConfig, ds: DatasetSpec, args: argparse.Namespace, *, scouted: bool
) -> dict[str, object]:
    """The cell-metric-filtered transform candidates of one dataset (cells only: no intensities, atoms or hand).

    ``scouted`` is False when AUTO keeps the declared transform (--trust-hkl-transform, --no-auto-decide): no split
    contract is written and the native stages key the declared transform's model setting.
    """

    hkl_path = str(Path(ds.hkl).expanduser().resolve())
    declared = parse_hkl_transform_info(ds.hkl_transform or "identity").canonical
    if not scouted:
        source = "trusted" if getattr(args, "trust_hkl_transform", False) else "declared"
        return {"source": source, "hkl": hkl_path, "hkl_transform": declared, "generated": [declared],
                "retained": [declared], "metric_filter": {}}
    from grond.engine.bloch_engine import metric_filter_signed_axis_transforms

    if not Path(hkl_path).is_file():
        raise ValueError(f"transform candidates: the HKL file of dataset {ds.name!r} does not exist: {ds.hkl}")
    generated = expand_transform_candidates(
        ds.hkl_transform,
        parity_policy=str(getattr(args, "transform_parity_policy", "proper-only")),
    )
    retained, metric_filter = metric_filter_signed_axis_transforms(
        Path(ds.hkl),
        Path(prep.structure),
        generated,
        relative_tolerance=TRANSFORM_METRIC_TOLERANCE,
        dataset_label=ds.name,
    )
    return {"source": "scout", "hkl": hkl_path, "hkl_transform": declared, "generated": list(generated),
            "retained": list(retained), "metric_filter": metric_filter}


def _recorded_contract_candidate_sets(datasets: list[DatasetSpec]) -> dict[str, dict]:
    """The shared-closure ``candidate_set`` blocks of the split contracts the datasets already carry."""

    from grond.bloch.splits import SHARED_CANDIDATE_POLICY

    out: dict[str, dict] = {}
    for ds in datasets:
        text = str(ds.split_contract or "").strip()
        if not text:
            continue
        path = Path(text).expanduser()
        if not path.is_file():
            raise ValueError(f"split candidates: the split contract of dataset {ds.name!r} does not exist: {text}")
        block = read_json(path).get("candidate_set")
        if isinstance(block, Mapping) and block.get("policy") == SHARED_CANDIDATE_POLICY:
            out[ds.name] = dict(block)
    return out


METRIC_REJECTION_MARKER = "rejected every transform candidate"


def exclude_metric_incompatible_datasets(
    prep: PrepConfig, args: argparse.Namespace, root: Path
) -> tuple[PrepConfig, dict[str, object]]:
    """Drop crystals whose cell no transform candidate reconciles with the model (multi-crystal runs).

    One mis-indexed or badly calibrated crystal must not stop a multi-crystal run: it is excluded before any scout
    or split contract, loudly and with its recorded reason, and the run fails only when no crystal is left.
    """

    from dataclasses import replace as dc_replace

    kept, excluded = [], {}
    for ds in prep.datasets:
        try:
            dataset_transform_candidates(prep, ds, args, scouted=True)
        except ValueError as exc:
            if METRIC_REJECTION_MARKER not in str(exc):
                raise
            excluded[ds.name] = {"hkl": str(ds.hkl), "reason": str(exc)}
            print(f"WARN dataset {ds.name} excluded: {exc}", flush=True)
            continue
        kept.append(ds)
    record = {"policy": "exclude", "excluded": excluded, "kept": [ds.name for ds in kept]}
    if excluded:
        write_json(Path(root) / "decisions" / "metric_excluded_datasets.json", record)
    if not kept:
        raise RuntimeError("every dataset was rejected by the unit-cell metric filter; nothing is left to refine")
    return (dc_replace(prep, datasets=kept) if excluded else prep), record


def exclude_metric_incompatible_datasets_at_start(
    prep: PrepConfig, args: argparse.Namespace, root: Path
) -> PrepConfig:
    """The start-of-run exclusion: before the split is announced, the conventional control or any scout.

    Every later stage (split plan, leak-free start, run_multi or, when one crystal is left, run_single) then sees
    the same crystals.  Trusted transforms and single-crystal runs are not filtered here and fail loudly.
    """

    if len(prep.datasets) < 2 or bool(getattr(args, "trust_hkl_transform", False)):
        return prep
    prep, _record = exclude_metric_incompatible_datasets(prep, args, root)
    return prep


def plan_split_candidates(
    prep: PrepConfig,
    args: argparse.Namespace,
    datasets: list[DatasetSpec] | None = None,
    *,
    scouted: bool | None = None,
) -> dict[str, object]:
    """One split candidate set per crystal, fixed from cell metrics before any scout reads an intensity.

    Every crystal's contract is built from the shared closure of all crystals' retained candidates
    (grond.bloch.splits.shared_candidate_closure), so a Laue orbit is free or excluded in every crystal or in none,
    whichever transform each crystal's scout later selects.  Each crystal still scouts only its own retained
    candidates.

    A declared or trusted dataset that already carries a shared-closure split contract (GROND scout's
    scouted_prep.json) keeps that contract in every native stage, so the plan is the closure the contracts recorded:
    it is recomputed from their retained candidates and must reproduce every contract's shared-group hash.
    """

    from grond.bloch.model import parse_model
    from grond.bloch.splits import shared_candidate_closure

    datasets = list(prep.datasets if datasets is None else datasets)
    if scouted is None:
        scouted = not bool(getattr(args, "trust_hkl_transform", False))
    candidates = {ds.name: dataset_transform_candidates(prep, ds, args, scouted=scouted) for ds in datasets}
    retained = {name: list(entry["retained"]) for name, entry in candidates.items()}
    recorded = {} if scouted else _recorded_contract_candidate_sets(datasets)
    for name, block in recorded.items():
        retained[name] = [str(value) for value in block.get("dataset_retained_candidates") or []]
    union = shared_candidate_closure(retained, parse_model(Path(prep.structure)).get("symmetry_ops"))
    if recorded:
        mismatched = sorted(
            name
            for name, block in recorded.items()
            if str(block.get("shared_group_sha256") or "") != str(union["shared_group_sha256"])
        )
        if mismatched:
            raise ValueError(
                f"the split contracts of dataset(s) {', '.join(mismatched)} were built from a shared candidate closure "
                "that the run's datasets do not reproduce (another scout, another model or a dataset without its "
                "contract); re-scout every dataset (no --trust-hkl-transform) or give every dataset the contract of "
                "the same scout"
            )
        union["closure_from_split_contracts"] = sorted(recorded)
    for name, entry in candidates.items():
        union["datasets"][name].update({"hkl": entry["hkl"], "candidate_source": entry["source"]})
    return {"split_candidate_set": "shared-closure", "datasets": candidates, "union": union}


def _materialize_transform_hand_split_contract(
    prep: PrepConfig,
    ds: DatasetSpec,
    candidates: list[str],
    args: argparse.Namespace,
    root: Path,
    *,
    candidate_set: dict | None = None,
) -> tuple[DatasetSpec, dict[str, object]]:
    """Write and bind the candidate-union raw-row split used by all scouts.

    ``candidates`` is the crystal's split candidate set (its share of the shared closure, or its own retained
    candidates under --split-candidate-set per-crystal); ``candidate_set`` records the shared closure.
    """

    from grond.engine.bloch_engine import (
        build_transform_split_contract,
        validate_transform_split_contract,
    )

    path = (root / "decisions" / ds.name / "transform_hand_split_contract.json").resolve()
    scheme = str(getattr(args, "split_scheme", "legacy"))
    scheme_kwargs: dict[str, object] = {}
    if scheme != "legacy":
        from grond.bloch.splits import load_split_exclusion

        exclusion_path = str(getattr(args, "exclude_orbits_json", "") or "").strip()
        scheme_kwargs = {
            "scheme": scheme,
            "salt": str(args.split_salt),
            "excluded_orbits": load_split_exclusion(exclusion_path)["ordered_orbits"] if exclusion_path else None,
        }
    elif str(getattr(args, "exclude_orbits_json", "") or "").strip():
        raise ValueError("--exclude-orbits-json requires --split-scheme orbit-hash")
    payload = build_transform_split_contract(
        Path(ds.hkl),
        Path(prep.structure),
        candidates,
        dataset=ds.name,
        free_fraction=float(args.free_fraction),
        seed=int(args.split_seed),
        include_friedel=True,
        **scheme_kwargs,
        **({"candidate_set": candidate_set} if candidate_set is not None else {}),
    )
    validation = validate_transform_split_contract(
        payload,
        source_hkl=Path(ds.hkl),
        current_transform=candidates[0],
    )
    validation.pop("masks", None)
    write_json(path, payload)
    binding = {
        **validation,
        "path": str(path),
        "sha256": sha256_file(path),
    }
    return dataset_with_split_contract(ds, str(path)), binding


CROSS_CRYSTAL_SPLIT_CHECK_SCHEMA = "grond.cross_crystal_split_check.v1"
_CONFLICTS_IN_MESSAGE = 20


def _screen_row_orbits(table: Path, symmetry_ops) -> tuple[dict[str, set[str]], int]:
    """Laue-orbit keys of the work and free rows of one native row table (h,k,l are in the model setting)."""

    import csv

    from grond.validation import heldout

    hkls: list[tuple[int, int, int]] = []
    splits: list[str] = []
    with table.open(newline="") as handle:
        for row in csv.DictReader(handle):
            hkls.append((int(float(row["h"])), int(float(row["k"])), int(float(row["l"]))))
            splits.append(str(row.get("split", "")).strip().lower())
    unknown = sorted(set(splits) - {"work", "free"})
    if unknown:
        raise RuntimeError(f"cross-crystal split check: {table} has split labels {unknown}; expected work or free")
    keys = heldout.orbit_keys(hkls, symmetry_ops, include_friedel=True) if hkls else []
    orbits: dict[str, set[str]] = {"work": set(), "free": set()}
    for key, split in zip(keys, splits, strict=True):
        orbits[split].add(str(key))
    return orbits, len(hkls)


def _contract_excluded_orbits(ds: DatasetSpec, symmetry_ops) -> set[str]:
    """Final-transform orbit keys of the rows an orbit-hash split contract excluded (membership 2)."""

    from grond.io.xds import XDSData
    from grond.validation import heldout

    if not str(ds.split_contract or "").strip():
        return set()
    payload = json.loads(Path(ds.split_contract).expanduser().read_text())
    membership = [int(value) for value in payload.get("membership") or []]
    if 2 not in membership:
        return set()
    data = XDSData(Path(ds.hkl), hkl_transform=ds.hkl_transform)
    if len(data.hkl) != len(membership):
        raise RuntimeError(
            f"cross-crystal split check: split contract {ds.split_contract} has {len(membership)} rows but "
            f"{ds.hkl} has {len(data.hkl)}"
        )
    rows = [hkl for hkl, member in zip(data.hkl, membership, strict=True) if member == 2]
    return {str(key) for key in heldout.orbit_keys(rows, symmetry_ops, include_friedel=True)}


def run_cross_crystal_split_check(
    prep: PrepConfig,
    args: argparse.Namespace,
    screen_runs: Mapping[str, Path],
    root: Path,
) -> dict[str, object]:
    """Refuse a multi-crystal run in which one Laue orbit is held out in one crystal and fitted in another.

    Under --split-candidate-set shared-closure (default) every crystal's contract is built from one shared
    candidate closure, so this check is an assertion that passes by construction.  Under per-crystal each
    crystal's orbit-hash contract is built from that crystal's own metric-filtered transform
    candidates, and a component is free when any of its orbit keys is free.  Crystals that keep
    different, not Laue-equivalent candidates can therefore put the same orbit on opposite sides, and
    the combined SHELXL HKL would refine on rows another crystal reports as held out.  After every
    transform is selected, the work and free rows of the screen row tables are keyed with
    grond.validation.heldout.orbit_key (model Laue group + Friedel, final-transform model setting) and
    pooled: no orbit may be free in one crystal and work in another, and no orbit excluded in one
    crystal (the lockbox listing, or the contract's candidate-union exclusion) may be used in another.
    The record is decisions/cross_crystal_split_check.json.  The legacy per-dataset split has no
    crystal-independent free set; it is not checked and nothing is written, so legacy runs are unchanged.
    """

    scheme = str(getattr(args, "split_scheme", "legacy") or "legacy")
    if scheme != "orbit-hash":
        return {"status": "not_applicable_legacy_split", "split_scheme": scheme}
    from grond.bloch.model import parse_model
    from grond.bloch.splits import load_split_exclusion

    path = root / "decisions" / "cross_crystal_split_check.json"
    exclusion_path = str(getattr(args, "exclude_orbits_json", "") or "").strip()
    exclusion = load_split_exclusion(exclusion_path) if exclusion_path else None
    record: dict[str, object] = {
        "schema": CROSS_CRYSTAL_SPLIT_CHECK_SCHEMA,
        "split_scheme": scheme,
        "split_salt": str(getattr(args, "split_salt", "")),
        "free_fraction": float(getattr(args, "free_fraction", 0.10)),
        "excluded_orbits_sha256": str(exclusion["sha256"]) if exclusion else "",
        "listed_excluded_orbit_count": len(exclusion["orbits"]) if exclusion else 0,
        "orbit_key": (
            "grond.validation.heldout.orbit_key (model Laue group + Friedel) of the screen row-table h,k,l, "
            "which are in the selected transform's model setting"
        ),
        "rule": (
            "refuse when a Laue orbit is free in one crystal and work in another, or excluded in one crystal "
            "and used (work or free) in another"
        ),
        "split_candidate_set": "shared-closure",
    }
    if bool(getattr(args, "dry_run", False)):
        write_json(path, {**record, "status": "skipped_dry_run"})
        return {**record, "status": "skipped_dry_run"}
    symmetry_ops = parse_model(Path(prep.structure)).get("symmetry_ops") or []
    lockbox = set(exclusion["orbits"]) if exclusion else set()
    crystals: dict[str, dict[str, object]] = {}
    status_by_orbit: dict[str, dict[str, set[str]]] = {}
    excluded_by_orbit: dict[str, set[str]] = {}
    for ds in prep.datasets:
        table = Path(screen_runs[ds.name]) / "native_dyn_calc.csv"
        if not table.is_file():
            raise RuntimeError(
                f"cross-crystal split check needs the screen row table of {ds.name}, which is missing: {table}"
            )
        orbits, n_rows = _screen_row_orbits(table, symmetry_ops)
        excluded = _contract_excluded_orbits(ds, symmetry_ops) | lockbox
        for split, keys in orbits.items():
            for key in keys:
                status_by_orbit.setdefault(key, {"work": set(), "free": set()})[split].add(ds.name)
        for key in excluded:
            excluded_by_orbit.setdefault(key, set()).add(ds.name)
        crystals[ds.name] = {
            "row_table": str(table),
            "hkl_transform": str(ds.hkl_transform),
            "split_contract": str(ds.split_contract or ""),
            "rows": n_rows,
            "work_orbits": len(orbits["work"]),
            "free_orbits": len(orbits["free"]),
            # Orbits the candidate-union contract excluded in addition to the listed (lockbox) orbits.
            "candidate_union_excluded_orbits": len(excluded - lockbox),
        }
    free_work = [
        {"orbit": key, "free_in": sorted(status["free"]), "work_in": sorted(status["work"])}
        for key, status in sorted(status_by_orbit.items())
        if status["free"] and status["work"]
    ]
    excluded_used = []
    for key, excluded_in in sorted(excluded_by_orbit.items()):
        status = status_by_orbit.get(key) or {"work": set(), "free": set()}
        used_in = sorted(status["work"] | status["free"])
        if used_in:
            excluded_used.append({"orbit": key, "excluded_in": sorted(excluded_in), "used_in": used_in})
    record.update(
        {
            "status": "conflict" if free_work or excluded_used else "consistent",
            "crystals": crystals,
            "orbit_count": len(status_by_orbit),
            "orbits_in_more_than_one_crystal": sum(
                1 for status in status_by_orbit.values() if len(status["work"] | status["free"]) > 1
            ),
            "free_work_conflict_count": len(free_work),
            "excluded_used_conflict_count": len(excluded_used),
            "free_work_conflicts": free_work,
            "excluded_used_conflicts": excluded_used,
        }
    )
    write_json(path, record)
    if free_work or excluded_used:
        listed = [
            f"{item['orbit']} (free in {','.join(item['free_in'])}; work in {','.join(item['work_in'])})"
            for item in free_work
        ] + [
            f"{item['orbit']} (excluded in {','.join(item['excluded_in'])}; used in {','.join(item['used_in'])})"
            for item in excluded_used
        ]
        more = len(listed) - _CONFLICTS_IN_MESSAGE
        raise RuntimeError(
            f"cross-crystal free-set conflict: {len(free_work)} Laue orbit(s) are free in one crystal and work in "
            f"another and {len(excluded_used)} are excluded in one crystal and used in another (keys in the "
            "selected transforms' model setting). The combined SHELXL HKL would refine on rows that another "
            "crystal reports as held out. This happens when the crystals keep different, not Laue-equivalent "
            "transform candidates (candidate_generation in decisions/<crystal>/transform_decision.json). "
            f"Conflicting orbits: {'; '.join(listed[:_CONFLICTS_IN_MESSAGE])}"
            + (f"; and {more} more" if more > 0 else "")
            + f". Full list: {path}. Build the contracts from one shared candidate set (--split-candidate-set "
            "shared-closure, the default), give every crystal the same Laue-equivalent candidate set with "
            "--transform-candidates, or supply verified transforms with --trust-hkl-transform."
        )
    return record


def _transform_record_metadata(
    transform: str,
    *,
    source_hkl: str = "",
) -> dict[str, object]:
    """Return flat determinant metadata suitable for scout tables."""

    info = parse_hkl_transform_info(transform)
    record = {
        "transform": info.canonical,
        "transform_determinant": info.determinant,
        "transform_parity": info.parity,
        "transform_is_proper": info.proper,
        "sign_paired_transform": info.sign_paired_transform,
        "proper_canonical_transform": info.proper_canonical_transform,
        "coordinate_inversion_required_for_proper_canonicalization": (
            info.coordinate_inversion_required_for_proper_canonicalization
        ),
    }
    if source_hkl:
        record["source_orientation_contract"] = source_orientation_contract(source_hkl, info.canonical)
    return record


def _selected_transform_claim_contract(
    transform: str,
    *,
    selection_source: str,
    parity_policy: str,
    source_hkl: str = "",
) -> dict[str, object]:
    """Bind one selected transform to GROND's fail-closed claim policy."""

    contract = absolute_configuration_claim_contract(transform)
    contract.update(
        {
            "selection_source": selection_source,
            "transform_parity_policy": parity_policy,
        }
    )
    if source_hkl:
        orientation = source_orientation_contract(source_hkl, transform)
        contract["source_orientation_contract"] = orientation
        if not bool(orientation.get("absolute_claim_geometry_ready")):
            contract["configuration_status"] = "parity_unresolved"
            contract["absolute_configuration_claim_allowed"] = False
            contract["reason"] = (
                f"source orientation contract is {orientation.get('status', 'missing')}; "
                "a positive right-handed raw real-space axis matrix and positive transformed "
                "determinant are required before named absolute configuration can be claimed"
            )
    return contract


def transform_scout_records(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    scout_root: Path,
    transforms: list[str],
    *,
    scout_stage: str,
    n_basis: int,
    n_points: int,
    max_rows: int,
) -> list[dict]:
    """Run one transform-scout stage and return scored decision records."""

    from grond.workflows.crystal_parameters import crystal_arguments

    args = crystal_arguments(args, ds.name)
    window_overrides = transform_scout_window_overrides()
    n_basis = getattr(args, "fixed_n_basis", None) or n_basis
    n_points = getattr(args, "fixed_n_points", None) or n_points
    frozen_nuisance = {**transform_scout_nuisance(args), **getattr(args, "fixed_thickness", {})}
    frozen_nuisance_sha256 = hashlib.sha256(
        json.dumps(
            frozen_nuisance,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        ).encode("utf-8")
    ).hexdigest()

    def run_one(rank: int, transform: str) -> dict:
        """Run the scout stage for one transform and return its scored record.

        The score (work-row matched-Born terms only) is replayed from the hash-bound row table and must equal the
        summary's score. A failed run returns status ``failed`` with an infinite score.
        """
        cand = dataset_with_transform(ds, transform)
        out_dir = scout_root / safe_label(transform)
        base_record = {
            **_transform_record_metadata(transform, source_hkl=ds.hkl),
            "out_dir": str(out_dir),
            "scout_stage": scout_stage,
            "n_basis": int(n_basis),
            "n_points": int(n_points),
            "max_rows": int(max_rows),
            "rank_in": rank,
            "frozen_nuisance_contract": frozen_nuisance,
            "frozen_nuisance_contract_sha256": frozen_nuisance_sha256,
            "engine_trial_selection_metric_requested": "work_born_wR2",
        }
        try:
            cmd = native_stage_cmd(
                python,
                cand,
                prep.structure,
                dataset_baseline_hkl(prep, cand),
                out_dir,
                args,
                n_basis=n_basis,
                n_points=n_points,
                stage=f"scout_transform_{scout_stage}_{safe_label(transform)}",
                pinned=frozen_nuisance,
                make_maps=False,
                endpoint=False,
                max_rows_override=max_rows,
                workers_override=TRANSFORM_SCOUT_WORKERS_PER_JOB,
                scale_fit_split_override="work",
                selection_metric_override="work_born_wR2",
                **window_overrides,
            )
            if args.dry_run:
                run_logged(cmd, logs, f"{ds.name}_scout_transform_{scout_stage}_{safe_label(transform)}", dry_run=True)
            else:
                semaphore = transform_scout_semaphore(args)
                with semaphore:
                    run_logged(
                        cmd, logs, f"{ds.name}_scout_transform_{scout_stage}_{safe_label(transform)}", dry_run=False
                    )
            if args.dry_run:
                return {**base_record, "status": "dry_run", "score": float("inf")}
            summary_path = out_dir / "summary.json"
            summary = read_json(summary_path)
            summary_score = score_transform_summary(summary)
            replay = replay_transform_work_born(summary_path)
            replay_score = float(replay["score"])
            if not math.isclose(
                replay_score,
                float(summary_score["score"]),
                rel_tol=1e-10,
                abs_tol=1e-12,
            ):
                raise ValueError(
                    "transform work-Born score does not replay from the bound "
                    f"row artifact: summary={summary_score['score']}, "
                    f"replay={replay_score}"
                )
            return {
                **base_record,
                "status": "ok",
                **summary_score,
                "score": replay_score,
                "artifact_score_replay": replay,
                "engine_trial_selection_metric": str(summary.get("selection_metric") or ""),
                "nuisance_trial_count": int(safe_float(summary.get("nuisance_trial_count"), 0.0)),
                "prediction_geometry_trial_count": int(safe_float(summary.get("prediction_geometry_trial_count"), 0.0)),
            }
        except Exception as exc:
            return {**base_record, "status": "failed", "error": str(exc), "score": float("inf")}

    if len(transforms) <= 1:
        return [run_one(rank, transform) for rank, transform in enumerate(transforms, start=1)]
    records: list[dict] = []
    max_workers = transform_scout_job_count(args, len(transforms))
    with ThreadPoolExecutor(
        max_workers=max_workers, thread_name_prefix=f"grond-{ds.name}-{scout_stage}-transform"
    ) as pool:
        futures = [pool.submit(run_one, rank, transform) for rank, transform in enumerate(transforms, start=1)]
        for future in as_completed(futures):
            records.append(future.result())
    return sorted(records, key=lambda row: int(row.get("rank_in", 0)))


def ranked_transform_records(records: list[dict], *, dry_run: bool) -> list[dict]:
    """Return transform scout records in promotion order."""

    if dry_run:
        return [r for r in records if r.get("status") == "dry_run"]
    ok = [r for r in records if r.get("status") == "ok" and float(r.get("score", float("inf"))) < float("inf")]
    # Metadata overlap is allowed to affect scheduling only.  Exact scout ties
    # use a canonical transform spelling, never the metadata-derived input
    # order, so all-row overlap cannot silently make the selected decision.
    priority = {
        parse_hkl_transform_info(candidate).canonical: rank
        for rank, candidate in enumerate(ALL_PROPER_SIGNED_PERMUTATIONS)
    }
    return sorted(
        ok,
        key=lambda r: (
            float(r["score"]),
            priority.get(
                parse_hkl_transform_info(str(r.get("transform") or "identity")).canonical,
                len(priority),
            ),
            parse_hkl_transform_info(str(r.get("transform") or "identity")).canonical,
        ),
    )


def choose_transform_by_scout(
    prep: PrepConfig,
    ds: DatasetSpec,
    args: argparse.Namespace,
    python: str,
    logs: Path,
    root: Path,
    *,
    scout_max_rows: int | None = None,
    split_plan: Mapping[str, object] | None = None,
) -> tuple[DatasetSpec, dict]:
    """Choose the HKL alignment using cheap all-candidate scouts, then rerank top hits.

    ``split_plan`` (``plan_split_candidates`` over every crystal of the run) fixes this crystal's retained
    candidates and its split candidate set; without it the plan covers this dataset alone.  Retained candidates that
    are one indexing under the model point group's proper rotations are scouted once.
    """

    parity_policy = str(getattr(args, "transform_parity_policy", "proper-only"))
    if getattr(args, "trust_hkl_transform", False):
        selected_info = parse_hkl_transform_info(ds.hkl_transform or "identity")
        if not selected_info.proper and parity_policy != "all-diagnostic":
            raise ValueError(
                f"trusted transform {selected_info.canonical!r} is improper (determinant -1) "
                "and cannot be accepted under --transform-parity-policy=proper-only; invert the model "
                f"coordinates and declare the proper transform {selected_info.proper_canonical_transform!r}, "
                "or explicitly use all-diagnostic mode without a named absolute-configuration claim"
            )
        selected_transform = selected_info.canonical
        selected = {
            **_transform_record_metadata(selected_transform, source_hkl=ds.hkl),
            "status": "trusted_input",
            "score": None,
            "reason": "user supplied --trust-hkl-transform; transform scouting was skipped",
        }
        claim_contract = _selected_transform_claim_contract(
            selected_transform,
            selection_source="trusted_input",
            parity_policy=parity_policy,
            source_hkl=ds.hkl,
        )
        payload = {
            "records": [],
            "coarse_records": [],
            "rerank_records": [],
            "coarse_selected": selected,
            "selected": selected,
            "rerank_top": 0,
            "metadata_transform_decision": {},
            "dataset": ds.name,
            "input_identity": _dataset_decision_identity(prep, ds),
            "transform_parity_policy": parity_policy,
            "selected_transform_claim_contract": claim_contract,
            "scout_parallelism": {
                "max_concurrent_scout_jobs": 0,
                "workers_per_scout_job": 0,
                "coarse_candidate_count": 0,
                "coarse_parallel_jobs": 0,
                "reranked_candidate_count": 0,
                "reranked_parallel_jobs": 0,
            },
            "rule": (
                "trusted user-supplied HKL transform; no Bloch transform scout was run; "
                "an improper trusted transform is diagnostic-only and forbids a named "
                "absolute-configuration claim"
            ),
            "selection_target": "trusted_input_no_ranking",
            "known_configuration_used_for_candidate_generation_or_ranking": False,
            "metadata_candidate_elimination_used": False,
        }
        write_json(root / "decisions" / ds.name / "transform_decision.json", payload)
        return dataset_with_transform(ds, selected_transform), payload

    from grond.bloch.model import parse_model
    from grond.bloch.splits import contract_candidate_set, transform_equivalence_classes

    plan = split_plan if split_plan is not None else plan_split_candidates(prep, args, [ds], scouted=True)
    plan_datasets = plan.get("datasets") if isinstance(plan.get("datasets"), Mapping) else {}
    entry = plan_datasets.get(ds.name) if isinstance(plan_datasets, Mapping) else None
    if not isinstance(entry, Mapping) or entry.get("source") != "scout":
        raise ValueError(f"the split candidate plan has no scouted transform candidates for dataset {ds.name!r}")
    if str(entry.get("hkl") or "") != str(Path(ds.hkl).expanduser().resolve()) or str(
        entry.get("hkl_transform") or ""
    ) != parse_hkl_transform_info(ds.hkl_transform or "identity").canonical:
        raise ValueError(
            f"the split candidate plan of dataset {ds.name!r} was built for another HKL file or declared transform"
        )
    generated_candidates = list(entry["generated"])
    candidates = list(entry["retained"])
    metric_filter_contract = dict(entry["metric_filter"])
    union = dict(plan["union"])
    union_entry = union["datasets"][ds.name]
    # The leak-free start removed the rows of the closure recorded at the start of the run; the contracts must use it.
    # read_json, not try_read_json: a record that exists but does not parse must stop the run, not skip the check.
    recorded = read_json(root / "decisions" / "split_summary.json")
    recorded_set = recorded.get("split_candidate_set") if isinstance(recorded, Mapping) else None
    if isinstance(recorded_set, Mapping) and str(recorded_set.get("shared_group_sha256") or "") != str(
        union.get("shared_group_sha256") or ""
    ):
        raise ValueError(
            f"dataset {ds.name!r}: the split candidate closure differs from the one recorded at the start of the run "
            "(decisions/split_summary.json); the model cell or the transform candidate options changed"
        )
    equivalence = transform_equivalence_classes(candidates, parse_model(Path(prep.structure)).get("symmetry_ops"))
    scouted_candidates = list(equivalence["representatives"])
    if int(equivalence["collapsed_candidate_count"]):
        print(
            f"transform scout {ds.name}: {len(candidates)} metric-compatible candidates form "
            f"{len(scouted_candidates)} point-group classes; scouting {', '.join(scouted_candidates)}",
            flush=True,
        )
    candidate_generation = _transform_candidate_generation_contract(
        "auto",
        generated_candidates,
        candidates,
        metric_filter_contract,
        parity_policy=parity_policy,
    )
    candidate_generation["transform_canonicalization"] = equivalence
    candidate_generation["scouted_candidates"] = scouted_candidates
    candidate_generation["split_candidates"] = list(union_entry["split_candidates"])
    candidate_generation["split_candidate_set_policy"] = str(union.get("policy") or "")
    split_contract: dict[str, object] = {}
    if not args.dry_run:
        ds, split_contract = _materialize_transform_hand_split_contract(
            prep,
            ds,
            list(union_entry["split_candidates"]),
            args,
            root,
            candidate_set=contract_candidate_set(union, ds.name),
        )
    candidates = scouted_candidates
    metadata_transform_decision: dict = {}
    metadata_candidate_elimination_used = False
    baseline_hkl = dataset_baseline_hkl(prep, ds)
    if baseline_hkl:
        _metadata_candidates, metadata_transform_decision = metadata_prefilter_transforms(
            ds.hkl, baseline_hkl, candidates
        )
        # The metadata overlap may order the expensive proper candidates but may not eliminate any of them: its
        # all-row overlap is not part of the held-out-safe transform selection target.
        ranked_names = [
            str(record.get("transform"))
            for record in (metadata_transform_decision.get("records") or [])
            if isinstance(record, Mapping)
        ]
        candidates = [
            *[name for name in ranked_names if name in candidates],
            *[name for name in candidates if name not in ranked_names],
        ]
        metadata_transform_decision["prefilter_status"] = metadata_transform_decision.get("status")
        metadata_transform_decision["status"] = "diagnostic_order_only"
        metadata_transform_decision["selected_transforms"] = list(candidates)
        metadata_transform_decision["candidate_elimination_used"] = metadata_candidate_elimination_used
    scout_root = root / "decisions" / ds.name / "transform_scout"
    coarse_max_rows = SCOUT_MAX_ROWS if scout_max_rows is None else scout_max_rows
    coarse_records = transform_scout_records(
        prep,
        ds,
        args,
        python,
        logs,
        scout_root,
        candidates,
        scout_stage="coarse",
        n_basis=SCOUT_N_BASIS,
        n_points=SCOUT_N_POINTS,
        max_rows=coarse_max_rows,
    )
    ranked_coarse = ranked_transform_records(coarse_records, dry_run=args.dry_run)
    if not ranked_coarse and not args.dry_run:
        write_json(
            root / "decisions" / ds.name / "transform_decision.json",
            {"records": coarse_records, "coarse_records": coarse_records, "selected": None},
        )
        raise RuntimeError(f"all transform scouts failed for dataset {ds.name}; see {scout_root}")

    coarse_selected = ranked_coarse[0] if ranked_coarse else {"transform": ds.hkl_transform}
    rerank_top = TRANSFORM_RERANK_TOP
    rerank_records: list[dict] = []
    selected = coarse_selected
    if len(candidates) > 1 and rerank_top > 0 and ranked_coarse:
        top_records = ranked_coarse[: min(rerank_top, len(ranked_coarse))]
        top_transforms = [str(r["transform"]) for r in top_records]
        rerank_max_rows = int(coarse_max_rows)
        rerank_records = transform_scout_records(
            prep,
            ds,
            args,
            python,
            logs,
            scout_root / "updated",
            top_transforms,
            scout_stage="updated",
            n_basis=TRANSFORM_RERANK_N_BASIS,
            n_points=TRANSFORM_RERANK_N_POINTS,
            max_rows=rerank_max_rows,
        )
        ranked_rerank = ranked_transform_records(rerank_records, dry_run=args.dry_run)
        if ranked_rerank:
            selected = ranked_rerank[0]

    records = [*coarse_records, *rerank_records]
    selected_transform = str(selected["transform"])
    claim_contract = _selected_transform_claim_contract(
        selected_transform,
        selection_source="automatic_transform_scout",
        parity_policy=parity_policy,
        source_hkl=ds.hkl,
    )
    payload = {
        "records": records,
        "coarse_records": coarse_records,
        "rerank_records": rerank_records,
        "coarse_selected": coarse_selected,
        "selected": selected,
        "rerank_top": rerank_top,
        "metadata_transform_decision": metadata_transform_decision,
        "metadata_candidate_elimination_used": metadata_candidate_elimination_used,
        "transform_parity_policy": parity_policy,
        "candidate_parity": [_transform_record_metadata(candidate, source_hkl=ds.hkl) for candidate in candidates],
        "candidate_generation": candidate_generation,
        "ranking_tie_break": {
            "policy": "fixed_full24_proper_priority_then_canonical_v1",
            "full24_proper_priority": [
                parse_hkl_transform_info(candidate).canonical for candidate in ALL_PROPER_SIGNED_PERMUTATIONS
            ],
            "metadata_order_used_for_ties_or_rerank_admission": False,
        },
        "transform_hand_split_contract": split_contract,
        "split_candidate_union": union,
        "dataset": ds.name,
        "input_identity": _dataset_decision_identity(prep, ds),
        "selected_transform_claim_contract": claim_contract,
        "scout_parallelism": {
            "max_concurrent_scout_jobs": transform_scout_job_count(args),
            "workers_per_scout_job": TRANSFORM_SCOUT_WORKERS_PER_JOB,
            "coarse_candidate_count": len(candidates),
            "coarse_parallel_jobs": transform_scout_job_count(args, len(candidates)),
            "reranked_candidate_count": len(rerank_records),
            "reranked_parallel_jobs": transform_scout_job_count(args, len(rerank_records)) if rerank_records else 0,
        },
        "rule": (
            "proper-only metadata/Bloch transform scouting by default, followed by higher N/Q "
            "rerank of the top candidates; transform rankings use independently scaled matched-Born "
            "work rows only, reserve free rows, and never use coordinate-hand-sensitive Bloch terms; "
            "default metadata overlap may order but never eliminate candidates; all-diagnostic is an "
            "explicit mixed-parity mode that cannot by itself establish named absolute configuration"
        ),
        "selection_target": "matched_born_work_only_hand_invariant_v1",
        "selection_nuisance_contract": transform_scout_nuisance(args),
        "known_configuration_used_for_candidate_generation_or_ranking": False,
    }
    write_json(root / "decisions" / ds.name / "transform_decision.json", payload)
    return dataset_with_transform(ds, selected["transform"]), payload
