"""The run's split record: the orbit-hash free set, the excluded (lockbox) orbits and their counts.

AUTO writes ``decisions/split_summary.json`` and prints the summary at the start of every run, before any fit, and
``grond splits make`` writes the same record for a new dataset before its first run.  The counts are those every
native stage will use: crystals whose transform is scouted get the candidate-union contract of their split candidate
set (``plan_split_candidates``), and declared or trusted transforms get the split contract they carry (GROND scout's
scouted_prep.json) or else the plain Laue-orbit split.  Rows are the source rows of each file, before the per-stage
row selection.

Roles of the three row sets:

* free: held out inside the run.  Never fitted, never selected on (except the two decisions HELDOUT_VALIDATION.md flags)
  and never written to a GROND SHELXL HKL.  AUTO reports held-out R on these rows with orbit-bootstrap 95% CIs.
* lockbox (``--exclude-orbits-json``): never seen by any GROND stage.  Dropped from every fit, selection and report of
  every run, and scored once, after the final model is fixed, outside AUTO.
* inner CV folds: salted orbit folds of the work rows inside one stage (P-spline lambda, the cone and dose gates, the
  hand decision's out-of-fold evidence).  They select, they do not validate.

The free and lockbox rows are still in the supplied conventional HKL, which the conventional reference controls (the one-cycle replay
of the supplied RES/HKL and each stage's ``baseline_exti`` copy) use as given; those controls feed no AUTO stage (the
leak-free start refines the supplied model on work rows only).
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np

from grond.io.config import DatasetSpec, PrepConfig
from grond.validation import heldout

SPLIT_RECORD_SCHEMA = "grond.split_record.v1"
SPLIT_ROLES = {
    "free": (
        "held out inside the run: never fitted, selected on (except the flagged hand and best-free keep decisions) "
        "or written to a GROND SHELXL HKL; AUTO reports held-out R on these rows with orbit-bootstrap 95% CIs "
        "(HELDOUT_VALIDATION.md)"
    ),
    "lockbox": (
        "orbits listed in --exclude-orbits-json: never seen by any GROND stage (dropped from every fit, selection and "
        "report); score them once, after the final model is fixed, outside AUTO"
    ),
    "conventional_reference": (
        "free and lockbox rows remain in the supplied conventional HKL, which the conventional reference controls "
        "(the one-cycle replay of the supplied RES/HKL and each stage's baseline_exti copy) use as given; those "
        "controls feed no AUTO stage (the leak-free start refines the supplied model on work rows only)"
    ),
    "inner_cv_folds": (
        "salted orbit folds of the work rows inside one stage (P-spline lambda, cone and dose gates, the hand "
        "decision's out-of-fold evidence): selection, not validation"
    ),
}


def _file_sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _crystal_split(
    ds: DatasetSpec,
    entry: Mapping[str, object],
    union_entry: Mapping[str, object],
    args: argparse.Namespace,
    symmetry_ops,
    exclusion: Mapping[str, object] | None,
) -> tuple[dict, dict[str, str]]:
    """Per-crystal counts plus the status ("work", "free", "excluded") of every model-setting Laue-orbit key."""

    from grond.bloch.splits import (
        candidate_invariant_split_membership,
        orbit_hash_realised_free_fraction,
        orbit_hash_split_masks,
        split_masks,
        validate_transform_split_contract,
    )
    from grond.hkl_transform import parse_hkl_transform_info
    from grond.io.xds import XDSData

    scheme = str(getattr(args, "split_scheme", "orbit-hash") or "orbit-hash")
    free_fraction = float(getattr(args, "free_fraction", 0.10))
    salt = str(getattr(args, "split_salt", heldout.DEFAULT_FREE_SALT))
    if not Path(ds.hkl).expanduser().is_file():
        raise ValueError(f"split record: the HKL file of dataset {ds.name!r} does not exist: {ds.hkl}")
    data = XDSData(Path(ds.hkl), hkl_transform="identity")
    raw = np.asarray(data.raw_hkl, dtype=np.int64)
    group_keys: list[list[str]] | None = None
    mixed = 0
    if entry.get("source") == "scout":
        result = candidate_invariant_split_membership(
            raw,
            symmetry_ops,
            list(union_entry["split_candidates"]),
            free_fraction=free_fraction,
            seed=int(getattr(args, "split_seed", 42)),
            include_friedel=True,
            scheme=scheme,
            salt=salt,
            excluded_orbits=set(exclusion["ordered_orbits"]) if exclusion else None,
        )
        membership = np.asarray(result["membership"], dtype=int)
        groups = np.asarray(result["orbit_group_by_row"], dtype=int)
        group_keys = result.get("group_orbit_keys")
        mixed = int(result.get("mixed_status_group_count") or 0)
        split_source = "candidate_union_contract"
    elif str(ds.split_contract or "").strip():
        validation = validate_transform_split_contract(
            json.loads(Path(ds.split_contract).expanduser().read_text()), source_hkl=Path(ds.hkl)
        )
        masks = validation["masks"]
        membership = np.where(masks.get("excluded", np.zeros(len(raw), dtype=bool)), 2, np.where(masks["free"], 1, 0))
        groups = np.asarray(json.loads(Path(ds.split_contract).expanduser().read_text())["orbit_group_by_row"], dtype=int)
        split_source = "prep_split_contract"
    else:
        matrix = np.asarray(parse_hkl_transform_info(str(entry["hkl_transform"])).matrix, dtype=np.int64)
        model_hkl = raw @ matrix.T
        keys = heldout.orbit_keys(model_hkl, symmetry_ops, True)
        if scheme == "orbit-hash":
            masks, excluded = orbit_hash_split_masks(
                model_hkl, free_fraction, salt, symmetry_ops, excluded_orbits=exclusion["orbits"] if exclusion else None
            )
            membership = np.where(excluded, 2, np.where(masks["free"], 1, 0))
        else:
            masks = split_masks(model_hkl, free_fraction, int(getattr(args, "split_seed", 42)), symmetry_ops)
            membership = np.where(masks["free"], 1, 0)
        unique = sorted({str(key) for key in keys})
        index = {key: position for position, key in enumerate(unique)}
        groups = np.asarray([index[str(key)] for key in keys], dtype=int)
        group_keys = [[key] for key in unique]
        split_source = "declared_transform_laue_orbits"
    rows = int(len(membership))
    work, free, excluded_rows = (int(np.count_nonzero(membership == value)) for value in (0, 1, 2))
    group_status = {int(group): int(member) for group, member in zip(groups, membership, strict=True)}
    components = len(group_status)
    free_components = sum(1 for value in group_status.values() if value == 1)
    excluded_components = sum(1 for value in group_status.values() if value == 2)
    record = {
        "hkl": str(Path(ds.hkl).expanduser().resolve()),
        "hkl_sha256": _file_sha256(Path(ds.hkl)),
        "candidate_source": str(entry.get("source") or ""),
        "declared_transform": str(entry.get("hkl_transform") or ""),
        "retained_candidates": list(entry.get("retained") or []),
        "split_candidates": list(union_entry.get("split_candidates") or []),
        "split_source": split_source,
        "rows": rows,
        "work_rows": work,
        "free_rows": free,
        "excluded_rows": excluded_rows,
        "components": components,
        "free_components": free_components,
        "excluded_components": excluded_components,
        "mixed_status_components": mixed,
        "free_fraction_of_used_rows": (free / (work + free)) if work + free else None,
    }
    if scheme == "orbit-hash":
        record["realised_free_fraction"] = orbit_hash_realised_free_fraction(
            free_rows=free,
            rows=rows,
            free_components=free_components,
            components=components,
            requested=free_fraction,
            mixed_status_group_count=mixed,
        )
    status_names = {0: "work", 1: "free", 2: "excluded"}
    key_status: dict[str, str] = {}
    if group_keys is not None:
        for group, keys in enumerate(group_keys):
            if group in group_status:
                for key in keys:
                    key_status[str(key)] = status_names[group_status[group]]
    return record, key_status


def build_split_record(
    prep: PrepConfig,
    args: argparse.Namespace,
    *,
    plan: Mapping[str, object] | None = None,
) -> tuple[dict, dict]:
    """The split record of a prep under ``args`` (and the split candidate plan it used)."""

    from grond.bloch.model import parse_model
    from grond.bloch.splits import load_split_exclusion
    from grond.workflows.transform_selection import plan_split_candidates, transform_canonicalization_policy

    scheme = str(getattr(args, "split_scheme", "orbit-hash") or "orbit-hash")
    exclusion_path = str(getattr(args, "exclude_orbits_json", "") or "").strip()
    if exclusion_path and scheme != "orbit-hash":
        raise ValueError("--exclude-orbits-json requires --split-scheme orbit-hash")
    exclusion = load_split_exclusion(exclusion_path) if exclusion_path else None
    plan = dict(plan) if plan is not None else plan_split_candidates(prep, args)
    symmetry_ops = parse_model(Path(prep.structure)).get("symmetry_ops")
    union = dict(plan["union"])
    crystals: dict[str, dict] = {}
    status_by_key: dict[str, dict[str, str]] = {}
    for ds in prep.datasets:
        record, key_status = _crystal_split(
            ds, plan["datasets"][ds.name], union["datasets"][ds.name], args, symmetry_ops, exclusion
        )
        crystals[ds.name] = record
        for key, status in key_status.items():
            status_by_key.setdefault(key, {})[ds.name] = status
    if scheme == "orbit-hash":
        free_fraction = float(getattr(args, "free_fraction", 0.10))
        settings = {
            "split_salt": str(getattr(args, "split_salt", heldout.DEFAULT_FREE_SALT)),
            "n_folds": max(2, int(round(1.0 / free_fraction))),
            "free_rule": "fold 0 of round(1/free_fraction) sha256(salt|model Laue-orbit key) folds",
        }
    else:
        settings = {"split_seed": int(getattr(args, "split_seed", 42)), "free_rule": "per-dataset shuffle"}
    totals = {
        name: sum(int(record[name]) for record in crystals.values())
        for name in ("rows", "work_rows", "free_rows", "excluded_rows")
    }
    pooled: dict[str, object] = {
        **totals,
        "free_fraction_of_used_rows": (
            totals["free_rows"] / (totals["work_rows"] + totals["free_rows"])
            if totals["work_rows"] + totals["free_rows"]
            else None
        ),
    }
    if scheme == "orbit-hash":
        conflicts = sorted(key for key, per in status_by_key.items() if len(set(per.values())) > 1)
        pooled.update(
            {
                "orbit_keys": len(status_by_key),
                "work_orbit_keys": sum(1 for per in status_by_key.values() if set(per.values()) == {"work"}),
                "free_orbit_keys": sum(1 for per in status_by_key.values() if set(per.values()) == {"free"}),
                "excluded_orbit_keys": sum(1 for per in status_by_key.values() if set(per.values()) == {"excluded"}),
                "orbit_keys_in_more_than_one_crystal": sum(1 for per in status_by_key.values() if len(per) > 1),
                "cross_crystal_status_conflicts": len(conflicts),
                "cross_crystal_status_conflict_examples": conflicts[:20],
            }
        )
    candidate_set: dict[str, object] = {"policy": str(union.get("policy") or "")}
    for field in ("shared_group", "shared_group_order", "laue_coset_count", "pseudo_symmetric", "shared_group_sha256"):
        if field in union:
            candidate_set[field] = union[field]
    record = {
        "schema": SPLIT_RECORD_SCHEMA,
        "split_scheme": scheme,
        "free_fraction": float(getattr(args, "free_fraction", 0.10)),
        **settings,
        "excluded_orbits": (
            {"path": str(exclusion["path"]), "sha256": str(exclusion["sha256"]), "orbit_count": len(exclusion["orbits"])}
            if exclusion
            else None
        ),
        "split_candidate_set": candidate_set,
        "transform_canonicalization": transform_canonicalization_policy(args),
        "model": {"path": str(Path(prep.structure).expanduser().resolve()), "sha256": _file_sha256(Path(prep.structure))},
        "crystals": crystals,
        "pooled": pooled,
        "roles": dict(SPLIT_ROLES),
    }
    return record, plan


def format_split_summary(record: Mapping[str, object]) -> list[str]:
    """Console lines of a split record."""

    lines = []
    scheme = str(record.get("split_scheme") or "")
    candidate_set = record.get("split_candidate_set") if isinstance(record.get("split_candidate_set"), Mapping) else {}
    policy = str(candidate_set.get("policy") or "")
    closure = (
        f"shared candidate closure ({int(candidate_set.get('laue_coset_count') or 1)} Laue coset(s))"
        if "shared" in policy
        else "per-crystal candidate sets"
    )
    if scheme == "orbit-hash":
        lines.append(
            f"split: orbit-hash free set, salt {record.get('split_salt')}, fraction {float(record['free_fraction']):g} "
            f"(fold 0 of {record.get('n_folds')} salted Laue-orbit folds); {closure}; transform canonicalisation "
            f"{record.get('transform_canonicalization')}"
        )
    else:
        lines.append(
            f"split: legacy per-dataset shuffle, seed {record.get('split_seed')}, fraction "
            f"{float(record['free_fraction']):g}; {closure}"
        )
    excluded = record.get("excluded_orbits")
    if isinstance(excluded, Mapping):
        lines.append(
            f"split: excluded (lockbox) orbits {excluded['path']}: {excluded['orbit_count']} orbits, "
            f"sha256 {str(excluded['sha256'])[:16]}"
        )
    else:
        lines.append("split: no excluded (lockbox) orbits")
    crystals = record.get("crystals") if isinstance(record.get("crystals"), Mapping) else {}
    for name, crystal in crystals.items():
        used = crystal.get("free_fraction_of_used_rows")
        lines.append(
            f"split: {name}: {crystal['rows']} source rows = work {crystal['work_rows']} + free {crystal['free_rows']} "
            f"({'n/a' if used is None else f'{100.0 * float(used):.1f}%'} of used rows) + excluded "
            f"{crystal['excluded_rows']}; {crystal['components']} orbit classes; "
            f"{len(crystal.get('split_candidates') or [])} split candidates ({crystal['split_source']})"
        )
        realised = crystal.get("realised_free_fraction")
        for warning in (realised or {}).get("warnings", []) if isinstance(realised, Mapping) else []:
            lines.append(f"WARN split {name}: {warning}")
    pooled = record.get("pooled") if isinstance(record.get("pooled"), Mapping) else {}
    if pooled:
        used = pooled.get("free_fraction_of_used_rows")
        text = (
            f"split: pooled: {pooled['rows']} rows = work {pooled['work_rows']} + free {pooled['free_rows']} "
            f"({'n/a' if used is None else f'{100.0 * float(used):.1f}%'}) + excluded {pooled['excluded_rows']}"
        )
        if "orbit_keys" in pooled:
            text += (
                f"; {pooled['orbit_keys']} Laue orbits (free {pooled['free_orbit_keys']}, excluded "
                f"{pooled['excluded_orbit_keys']}), {pooled['orbit_keys_in_more_than_one_crystal']} in more than one "
                f"crystal, {pooled['cross_crystal_status_conflicts']} cross-crystal status conflicts"
            )
        lines.append(text)
        if int(pooled.get("cross_crystal_status_conflicts") or 0):
            lines.append(
                "WARN split: some Laue orbits are free in one crystal and work or excluded in another "
                f"(e.g. {', '.join(pooled.get('cross_crystal_status_conflict_examples') or [])}); the cross-crystal "
                "check will refuse this run. Use --split-candidate-set shared-closure."
            )
    return lines


def announce_run_split(prep: PrepConfig, args: argparse.Namespace, root: Path) -> dict:
    """Write decisions/split_summary.json, print the summary and return the split candidate plan."""

    from grond.workflows.decision import write_json

    record, plan = build_split_record(prep, args)
    path = Path(root) / "decisions" / "split_summary.json"
    write_json(path, record)
    for line in format_split_summary(record):
        print(line, flush=True)
    print(f"split: record {path}", flush=True)
    return plan
