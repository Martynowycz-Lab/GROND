"""``GROND splits make`` and ``GROND splits inspect``: the free set and the lockbox of a new run.

``make`` writes, for a prep file or HKL files plus a model, the split record AUTO will use (the orbit-hash free set:
salt, fraction, per-crystal and pooled counts, realised fraction) and, optionally, a lockbox: an optional set of
orbits held back from every stage, written as JSON with the fraction, the resolution shells, the orbit listing and the
SHA-256 of that listing.  Pass the lockbox to every AUTO run with ``--exclude-orbits-json`` and score it once, after
the final model is fixed.
``inspect`` prints and verifies a split record, a lockbox, a split contract or a finished run directory (with
``--lockbox``, it also checks that no row of the run lies in a lockbox orbit).

The candidate sets and the free rule are exactly AUTO's (``grond.workflows.transform_selection.plan_split_candidates``
and ``grond.workflows.split_record``), so the counts are the ones the run will report.  Only cell metrics are read to
fix the candidate sets; no intensity enters the free set or the lockbox.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import hashlib
import json
import secrets
from pathlib import Path

import numpy as np

from grond.io.config import DatasetSpec, PrepConfig, load_prep, resolve_hkl
from grond.validation import heldout

LOCKBOX_FILE = "lockbox.json"
SPLIT_RECORD_FILE = "split_record.json"
# Run directories of the conventional reference controls, which refine the supplied conventional HKL as given.
CONVENTIONAL_REFERENCE_DIRS = frozenset({"fresh_conventional_baseline", "baseline_exti"})


def _prep_for_splits(args: argparse.Namespace) -> PrepConfig:
    """Return the prep of ``splits make``: the ``--prep`` file, or one built from ``--hkl`` and ``--structure``."""

    if str(getattr(args, "prep", "") or "").strip():
        if args.hkl:
            raise SystemExit("Give either --prep or --hkl, not both.")
        prep = load_prep(Path(args.prep))
        if args.structure:
            prep.structure = str(Path(args.structure).expanduser().resolve())
        return prep
    if not args.hkl:
        raise SystemExit("Provide --prep, or one or more --hkl files (INTEGRATE.HKL, a GROND DIALS bridge) with --structure.")
    if not args.structure:
        raise SystemExit("Provide --structure with --hkl.")
    names = list(args.dataset_name or [])
    if names and len(names) != len(args.hkl):
        raise SystemExit("Give one --dataset-name per --hkl, or none.")
    datasets = []
    for index, value in enumerate(args.hkl):
        hkl, xds_dir = resolve_hkl(value)
        datasets.append(
            DatasetSpec(
                name=names[index] if names else f"dataset_{index + 1}",
                hkl=hkl,
                xds_dir=xds_dir,
                hkl_transform=args.hkl_transform,
            )
        )
    return PrepConfig(structure=str(Path(args.structure).expanduser().resolve()), datasets=datasets)


def _run_like_args(args: argparse.Namespace, *, exclude_orbits_json: str) -> argparse.Namespace:
    """The AUTO options the free set depends on (orbit-hash only)."""

    return argparse.Namespace(
        auto_decide=True,
        trust_hkl_transform=bool(args.trust_hkl_transform),
        transform_candidates=str(args.transform_candidates),
        transform_metric_tolerance=float(args.transform_metric_tolerance),
        transform_parity_policy=str(args.transform_parity_policy),
        split_candidate_set=str(args.split_candidate_set),
        transform_canonicalization=str(args.transform_canonicalization),
        split_scheme="orbit-hash",
        split_salt=str(args.split_salt),
        free_fraction=float(args.free_fraction),
        split_seed=42,
        exclude_orbits_json=str(exclude_orbits_json or ""),
    )


def _max_key(keys) -> str:
    """Return the largest orbit key, comparing the ``h|k|l`` keys as integer tuples."""

    return max(keys, key=lambda key: tuple(int(value) for value in str(key).split("|")))


def build_lockbox(
    prep: PrepConfig,
    plan: dict,
    *,
    fraction: float,
    salt: str,
    n_shells: int = 5,
    d_min: float = 0.0,
) -> dict:
    """A lockbox of whole orbit classes (the lockbox JSON format read by --exclude-orbits-json).

    Candidate orbits are the model-setting Laue orbits of every source row of every crystal (with d >= d_min).  An
    orbit class is one Laue orbit, or under a pseudo-symmetric shared candidate closure the orbits the split contracts
    merge, so the lockbox is excluded identically in every crystal whatever transform the scout selects.  Classes are
    put in equal-count resolution shells by 1/d^2 and, within each shell, the round(fraction x n) classes with the
    smallest int(sha256(salt|key)[:16], 16) are taken (key = the class's largest orbit key).  Every orbit key of a
    chosen class is listed.  d is computed with the model cell, so the shell boundaries do not depend on crystal
    order, and the candidate rows are every source row (no PEAK, d or geometry row filter).
    """

    import gemmi

    from grond.bloch.model import parse_model
    from grond.bloch.splits import laue_proper_rotations, row_orbit_key_sets
    from grond.hkl_transform import parse_hkl_transform_info
    from grond.io.xds import XDSData

    if not 0.0 < float(fraction) < 1.0:
        raise ValueError("--lockbox-fraction must be between 0 and 1")
    if int(n_shells) < 1:
        raise ValueError("--lockbox-shells must be at least 1")
    model = parse_model(Path(prep.structure))
    symmetry_ops = model.get("symmetry_ops")
    cell = gemmi.UnitCell(*[float(value) for value in model["cell"]])
    union = plan["union"]
    parent: dict[str, str] = {}

    def find(key: str) -> str:
        """Return the root of an orbit key's class (union-find with path halving)."""
        while parent[key] != key:
            parent[key] = parent[parent[key]]
            key = parent[key]
        return key

    row_keys: dict[str, list[frozenset[str]]] = {}
    for ds in prep.datasets:
        entry = plan["datasets"][ds.name]
        raw = np.asarray(XDSData(Path(ds.hkl), hkl_transform="identity").raw_hkl, dtype=np.int64)
        if entry["source"] == "scout":
            rows, transforms = raw, list(union["datasets"][ds.name]["split_candidates"])
        else:
            # Declared or trusted transform: its model-setting rows under the shared group (the Laue group itself
            # unless the dataset's split contract recorded a pseudo-symmetric closure).
            declared = np.asarray(parse_hkl_transform_info(str(entry["hkl_transform"])).matrix, dtype=np.int64)
            rows, transforms = raw @ declared.T, list(union.get("shared_group") or ["identity"])
        key_sets = row_orbit_key_sets(rows, transforms, symmetry_ops)
        row_keys[ds.name] = key_sets
        for keys in key_sets:
            ordered = sorted(keys)
            for key in ordered:
                parent.setdefault(key, key)
            for key in ordered[1:]:
                left, right = find(ordered[0]), find(key)
                if left != right:
                    parent[right] = left
    members: dict[str, set[str]] = {}
    for key in parent:
        members.setdefault(find(key), set()).add(key)
    classes = {}
    representative_of_root = {root: _max_key(keys) for root, keys in members.items()}
    for root, keys in members.items():
        representative = representative_of_root[root]
        d = float(cell.calculate_d([int(value) for value in representative.split("|")]))
        if float(d_min) > 0.0 and d < float(d_min):
            continue
        classes[representative] = {"keys": sorted(keys), "d": d}
    if not classes:
        raise ValueError("no candidate orbits remain for the lockbox")
    representatives = sorted(classes)
    d = np.asarray([classes[key]["d"] for key in representatives], dtype=float)
    order = np.argsort(1.0 / d**2, kind="mergesort")
    shell = np.empty(len(representatives), dtype=int)
    shell[order] = np.minimum(int(n_shells) - 1, np.arange(len(representatives)) * int(n_shells) // len(representatives))

    def rank_value(key: str) -> int:
        """Return the lockbox rank of a class key: int(sha256(salt|key)[:16], 16)."""
        return int(hashlib.sha256(f"{salt}|{key}".encode()).hexdigest()[:16], 16)

    chosen: list[str] = []
    shells = []
    for index in range(int(n_shells)):
        in_shell = [representatives[i] for i in np.where(shell == index)[0]]
        if not in_shell:
            continue
        in_shell.sort(key=rank_value)
        n_take = int(round(float(fraction) * len(in_shell)))
        chosen.extend(in_shell[:n_take])
        shell_d = [classes[key]["d"] for key in in_shell]
        shells.append({"shell": index, "d_max": max(shell_d), "d_min": min(shell_d), "n_orbits": len(in_shell), "n_lockbox": n_take})
    listing = sorted({key for representative in chosen for key in classes[representative]["keys"]})
    if not listing:
        raise ValueError("the lockbox is empty at this fraction; use more reflections or a larger --lockbox-fraction")
    listed = set(listing)
    rows = {
        name: {
            "candidate": sum(1 for keys in key_sets if representative_of_root[find(min(keys))] in classes),
            "lockbox": sum(1 for keys in key_sets if keys & listed),
        }
        for name, key_sets in row_keys.items()
    }
    laue = laue_proper_rotations(symmetry_ops)
    return {
        "kind": "grond_lockbox",
        "tool": "GROND splits make",
        "created_utc": datetime.datetime.now(datetime.UTC).isoformat(),
        "laue_class": (
            f"model Laue group ({len(laue)} proper rotations) + Friedel; orbit key grond.validation.heldout.orbit_key "
            "in the model setting; orbit classes of the split candidate set are listed whole"
        ),
        "salt": str(salt),
        "fraction": float(fraction),
        "n_shells": int(n_shells),
        "d_min_A": float(d_min),
        "candidate_filters": "every source row of every crystal" + (f", d >= {float(d_min):g} A" if float(d_min) > 0 else ""),
        "split_candidate_set": {
            "policy": str(union.get("policy") or ""),
            "shared_group_sha256": str(union.get("shared_group_sha256") or ""),
            "laue_coset_count": int(union.get("laue_coset_count") or 1),
        },
        "model": {"path": str(Path(prep.structure).expanduser().resolve()), "sha256": _sha256(Path(prep.structure))},
        "n_candidate_orbits": sum(len(classes[key]["keys"]) for key in representatives),
        "n_candidate_classes": len(representatives),
        "n_lockbox_orbits": len(listing),
        "n_lockbox_classes": len(chosen),
        "rows": rows,
        "shells": shells,
        "lockbox_sha256": hashlib.sha256("\n".join(listing).encode()).hexdigest(),
        "lockbox_orbits": listing,
    }


def _sha256(path: Path) -> str:
    """Return the SHA-256 hex digest of a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def cmd_splits_make(args: argparse.Namespace) -> int:
    """Write the split record of a new run and, with --lockbox-fraction, its lockbox."""

    from grond.workflows.decision import write_json
    from grond.workflows.split_record import build_split_record, format_split_summary
    from grond.workflows.transform_selection import plan_split_candidates

    out_dir = Path(args.out_dir).expanduser().resolve()
    record_path = out_dir / SPLIT_RECORD_FILE
    lockbox_path = out_dir / LOCKBOX_FILE
    make_lockbox = float(args.lockbox_fraction or 0.0) > 0.0
    if make_lockbox and str(args.exclude_orbits_json or "").strip():
        raise SystemExit("Give either --exclude-orbits-json (an existing lockbox) or --lockbox-fraction (a new one), not both.")
    for path in (record_path, *((lockbox_path,) if make_lockbox else ())):
        if path.exists():
            raise SystemExit(f"{path} exists; a split record and its lockbox are never overwritten. Use a new --out-dir.")
    prep = _prep_for_splits(args)
    plan_args = _run_like_args(args, exclude_orbits_json="")
    try:
        plan = plan_split_candidates(prep, plan_args)
        exclusion = str(args.exclude_orbits_json or "")
        lockbox = None
        if make_lockbox:
            salt = str(args.lockbox_salt or "").strip() or (
                f"GROND-LOCKBOX-{datetime.date.today():%Y%m%d}-{secrets.token_hex(8)}"
            )
            lockbox = build_lockbox(
                prep, plan, fraction=float(args.lockbox_fraction), salt=salt, n_shells=int(args.lockbox_shells),
                d_min=float(args.lockbox_d_min),
            )
        out_dir.mkdir(parents=True, exist_ok=True)
        if lockbox is not None:
            write_json(lockbox_path, lockbox)
            exclusion = str(lockbox_path)
        try:
            record, _plan = build_split_record(prep, _run_like_args(args, exclude_orbits_json=exclusion), plan=plan)
        except ValueError:
            # The lockbox was written by this call a moment ago; never leave one behind without its split record
            # (a rerun into the same --out-dir would otherwise be refused).
            if lockbox is not None:
                lockbox_path.unlink()
            raise
    except ValueError as exc:
        raise SystemExit(f"GROND splits make: {exc}") from exc
    record["tool"] = "GROND splits make"
    record["prep"] = str(Path(args.prep).expanduser().resolve()) if args.prep else ""
    write_json(record_path, record)
    for line in format_split_summary(record):
        print(line, flush=True)
    if lockbox is not None:
        print(
            f"lockbox: {lockbox_path}: {lockbox['n_lockbox_orbits']} orbits in {lockbox['n_lockbox_classes']} classes "
            f"({len(lockbox['shells'])} shells, fraction {lockbox['fraction']:g}, salt {lockbox['salt']}), sha256 "
            f"{lockbox['lockbox_sha256'][:16]}",
            flush=True,
        )
    print(f"split record: {record_path}", flush=True)
    flags = [f"--free-fraction {float(args.free_fraction):g}", f"--split-salt {args.split_salt}"]
    if exclusion:
        flags.append(f"--exclude-orbits-json {exclusion}")
    print(f"pass to every AUTO run of this dataset: {' '.join(flags)}", flush=True)
    return 0


def _row_table_orbit_keys(table: Path, symmetry_ops) -> list[str]:
    """Return the orbit key (Friedel mates included) of every row of a native row table."""

    with table.open(newline="") as handle:
        hkls = [(int(float(row["h"])), int(float(row["k"])), int(float(row["l"]))) for row in csv.DictReader(handle)]
    return [str(key) for key in heldout.orbit_keys(hkls, symmetry_ops, True)] if hkls else []


def _inspect_run(run_dir: Path, lockbox_path: str) -> tuple[dict, list[str], bool]:
    """Check a finished run's split record, split contracts and cross-crystal check; return result, lines, ok.

    With ``lockbox_path`` it also checks that the run honoured that lockbox and counts rows of row tables and written
    HKL files that lie in a lockbox orbit (the conventional reference HKLs are listed, not scanned).
    """

    from grond.bloch.model import parse_model
    from grond.bloch.splits import validate_transform_split_contract
    from grond.workflows.leak_free_start import read_hklf4_rows
    from grond.workflows.split_record import format_split_summary

    lines: list[str] = []
    ok = True
    summary_path = run_dir / "decisions" / "split_summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.is_file() else {}
    if summary:
        lines.extend(format_split_summary(summary))
    else:
        lines.append(f"split: {summary_path} is missing (a scout directory, or a run that predates the split record)")
    contracts = {}
    for path in sorted((run_dir / "decisions").glob("*/transform_hand_split_contract.json")):
        payload = json.loads(path.read_text())
        validation = validate_transform_split_contract(payload)
        validation.pop("masks", None)
        contracts[path.parent.name] = {
            "path": str(path),
            "rows": validation["row_count"],
            "work_rows": validation["work_row_count"],
            "free_rows": validation["free_row_count"],
            "excluded_rows": int(validation.get("excluded_row_count") or 0),
            "excluded_orbits_sha256": str(validation.get("excluded_orbits_sha256") or ""),
            "candidate_set_policy": str((validation.get("candidate_set") or {}).get("policy") or "per_crystal"),
        }
        lines.append(
            f"contract {path.parent.name}: {validation['row_count']} rows = work {validation['work_row_count']} + free "
            f"{validation['free_row_count']} + excluded {int(validation.get('excluded_row_count') or 0)} "
            f"({contracts[path.parent.name]['candidate_set_policy']})"
        )
    check_path = run_dir / "decisions" / "cross_crystal_split_check.json"
    check = json.loads(check_path.read_text()) if check_path.is_file() else {}
    if check:
        lines.append(f"cross-crystal split check: {check.get('status')}")
        ok = ok and check.get("status") in {"consistent", "skipped_dry_run"}
    result: dict = {"run_dir": str(run_dir), "split_summary": summary, "contracts": contracts, "cross_crystal_check": check}
    if lockbox_path:
        loaded = heldout.load_orbit_exclusion(lockbox_path)
        recorded = {str((summary.get("excluded_orbits") or {}).get("sha256") or "")} if summary else set()
        recorded |= {entry["excluded_orbits_sha256"] for entry in contracts.values()}
        recorded.discard("")
        honoured = recorded == {loaded["sha256"]}
        prep_path = run_dir / "run_prep.json"
        structure = (summary.get("model") or {}).get("path") if summary else ""
        if not structure and prep_path.is_file():
            structure = json.loads(prep_path.read_text()).get("structure", "")
        symmetry_ops = parse_model(Path(structure)).get("symmetry_ops") if structure else None
        leaked = 0
        tables = sorted((run_dir / "runs").glob("**/native_dyn_calc.csv")) if (run_dir / "runs").is_dir() else []
        for table in tables:
            leaked += sum(1 for key in _row_table_orbit_keys(table, symmetry_ops) if key in loaded["orbits"])
        # Every SHELXL HKL the run wrote (model setting).  The conventional reference controls hold the supplied
        # conventional HKL as given, by design; they are listed, not scanned.
        hkl_files, reference_hkls, hkl_leaks = [], [], {}
        for hkl_path in sorted(run_dir.rglob("*.hkl")):
            relative = hkl_path.relative_to(run_dir)
            if set(relative.parts) & CONVENTIONAL_REFERENCE_DIRS:
                reference_hkls.append(str(relative))
                continue
            hkl_files.append(str(relative))
            rows = [hkl for hkl, _line in read_hklf4_rows(hkl_path)]
            keys = heldout.orbit_keys(rows, symmetry_ops, True) if rows else []
            count = sum(1 for key in keys if str(key) in loaded["orbits"])
            if count:
                hkl_leaks[str(relative)] = count
        result["lockbox"] = {
            "path": str(lockbox_path),
            "sha256": loaded["sha256"],
            "recorded_exclusion_sha256": sorted(recorded),
            "exclusion_recorded": honoured,
            "row_tables_scanned": len(tables),
            "lockbox_rows_in_row_tables": leaked,
            "shelxl_hkls_scanned": len(hkl_files),
            "lockbox_rows_in_shelxl_hkls": hkl_leaks,
            "conventional_reference_hkls_not_scanned": reference_hkls,
        }
        lines.append(
            f"lockbox {lockbox_path}: sha256 {loaded['sha256'][:16]}; recorded by the run: {'yes' if honoured else 'no'}; "
            f"{leaked} lockbox rows in {len(tables)} row tables; {sum(hkl_leaks.values())} lockbox rows in "
            f"{len(hkl_files)} SHELXL HKL files ({len(reference_hkls)} conventional reference HKLs not scanned)"
        )
        for relative, count in hkl_leaks.items():
            lines.append(f"FAIL lockbox: {count} lockbox rows in {relative}")
        ok = ok and honoured and leaked == 0 and not hkl_leaks and symmetry_ops is not None
    return result, lines, ok


def cmd_splits_inspect(args: argparse.Namespace) -> int:
    """Print and verify a split record, lockbox, split contract or AUTO run directory."""

    from grond.bloch.splits import validate_transform_split_contract
    from grond.workflows.split_record import SPLIT_RECORD_SCHEMA, format_split_summary

    path = Path(args.path).expanduser().resolve()
    ok = True
    if path.is_dir():
        result, lines, ok = _inspect_run(path, str(args.lockbox or ""))
    elif path.is_file():
        payload = json.loads(path.read_text())
        if isinstance(payload, dict) and "lockbox_orbits" in payload:
            loaded = heldout.load_orbit_exclusion(path)
            result = {k: v for k, v in payload.items() if k != "lockbox_orbits"}
            lines = [
                f"lockbox {path}: {len(loaded['orbits'])} orbits, sha256 {loaded['sha256'][:16]} (verified), fraction "
                f"{payload.get('fraction')}, {len(payload.get('shells') or [])} shells, salt {payload.get('salt')}"
            ]
        elif isinstance(payload, dict) and payload.get("schema") == SPLIT_RECORD_SCHEMA:
            result, lines = payload, format_split_summary(payload)
        elif isinstance(payload, dict) and payload.get("kind") == "grond_transform_hand_split_contract":
            validation = validate_transform_split_contract(payload)
            validation.pop("masks", None)
            result = validation
            lines = [
                f"split contract {path}: {validation['row_count']} rows = work {validation['work_row_count']} + free "
                f"{validation['free_row_count']} + excluded {int(validation.get('excluded_row_count') or 0)}; "
                f"{len(validation['candidate_transforms'])} split candidates (verified)"
            ]
        else:
            raise SystemExit(f"{path} is not a split record, lockbox, split contract or run directory")
    else:
        raise SystemExit(f"{path} does not exist")
    if getattr(args, "json", False):
        print(json.dumps(result, indent=2, default=str))
    else:
        for line in lines:
            print(line, flush=True)
    if not ok:
        print("FAIL: the run's split does not verify (see above)", flush=True)
        return 2
    return 0
