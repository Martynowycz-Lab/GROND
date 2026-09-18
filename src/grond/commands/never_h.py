"""Run a matched H-free branch with no nuisance or scale re-selection."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from grond.cli_support.tools import external_tool_statuses
from grond.legacy.engine.native_row_regularize import shelx_res_health, write_endpoint_ins
from grond.legacy.engine.shelx_hkl import write_fixed_width_hkl
from grond.macromolecule import sha256_file
from grond.runtime import auto_worker_count
from grond.workflows.matched_prediction import predict_dataset
from grond.workflows.matched_reference import (
    ReferencePaths,
    frozen_merge_contract,
    load_reference,
    merge_changed_predictions,
    save_json,
)
from grond.workflows.model_alignment import reindex_hkl, require_no_hydrogen, write_aligned_res


def recorded_run(command, directory: Path, label: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    save_json(directory / f"{label}.command.json", {"command": command, "cwd": str(directory)})
    started = time.time()
    with (directory / f"{label}.log").open("w") as handle:
        process = subprocess.run(command, cwd=directory, stdout=handle, stderr=subprocess.STDOUT)
    if process.returncode:
        raise RuntimeError(f"{label} exited {process.returncode}; see {directory / (label + '.log')}")
    if label in {"shelxl", "zero_cycle_LIST6"}:
        model = directory / "model.res"
        if not model.is_file() or model.stat().st_size == 0:
            tail = "\n".join((directory / f"{label}.log").read_text(errors="replace").splitlines()[-18:])
            raise RuntimeError(f"SHELXL wrote no usable model despite exit code 0. {directory / (label + '.log')}\n{tail}")
    save_json(directory / f"{label}.timing.json", {"seconds": time.time() - started})


def _run(args, out: Path) -> int:
    paths = ReferencePaths(args.path_map)
    reference = load_reference(Path(args.reference_run).expanduser().resolve(), paths)
    source = Path(args.structure).expanduser().resolve()
    baseline = Path(args.baseline_hkl).expanduser().resolve() if args.baseline_hkl else source.with_suffix(".hkl")
    require_no_hydrogen(source)
    if not baseline.is_file():
        raise FileNotFoundError(f"Supply the conventional HKL matching the H-free model: {baseline}")
    package = Path(__file__).resolve().parents[1]
    source_digest = hashlib.sha256()
    for file in sorted(package.rglob("*.py")):
        source_digest.update(str(file.relative_to(package)).encode())
        source_digest.update(file.read_bytes())
    identity = {"reference_model_sha256": reference["model_sha256"], "input_model_sha256": sha256_file(source),
                "input_hkl_sha256": sha256_file(baseline), "package_source_sha256": source_digest.hexdigest()}
    previous = out / "RUN_MANIFEST.json"
    if previous.exists():
        old = json.loads(previous.read_text())
        if any(old.get(key) != value for key, value in identity.items()):
            raise ValueError("Inputs or code changed. Choose a new --out-dir.")
    tools = external_tool_statuses(args)
    shelxl = tools.get("shelxl", {}).get("resolved")
    maptool = tools.get("shelx2map", {}).get("resolved")
    if not args.dry_run and (not shelxl or (args.maps and not maptool)):
        raise RuntimeError("never-h needs SHELXL and, for maps, shelx2map. Configure the missing executable.")
    conventional = out / "conventional"
    conventional.mkdir(exist_ok=True)
    aligned = conventional / "model.res"
    alignment = write_aligned_res(source, Path(reference["model"]), aligned)
    reindex_hkl(baseline, conventional / "model.hkl", alignment["fractional_matrix"])
    save_json(out / "ALIGNMENT.json", alignment)
    frozen = frozen_merge_contract(reference)
    workers = auto_worker_count(args.workers, 0.85)
    manifest = {
        "command": [sys.executable, "-m", "grond", "never-h", "--reference-run", reference["root"],
                    "--structure", str(source), "--baseline-hkl", str(baseline), "--out-dir", str(out),
                    "--workers", str(workers), "--cycles", str(args.cycles), "--map-grid-spacing", str(args.map_grid_spacing),
                    *(["--shelxl", shelxl] if shelxl else []),
                    *[part for mapping in args.path_map for part in ("--path-map", mapping)],
                    *(["--no-maps"] if not args.maps else [])],
        **identity,
        "tools": tools,
        "environment": {key: os.environ.get(key, "") for key in ("PYTHONPATH", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "SHELX2MAP")},
        "reference_run": reference["root"], "reference_model": reference["model"],
        "reference_model_sha256": reference["model_sha256"], "input_model": str(source),
        "selected_reference_endpoint": reference["selected_endpoint"],
        "input_model_sha256": sha256_file(source), "input_hkl": str(baseline), "input_hkl_sha256": sha256_file(baseline),
        "cycles": args.cycles, "workers": workers, "maps": args.maps,
        "datasets": [{key: value for key, value in ds.items() if key not in {"rows", "summary"}} for ds in reference["datasets"]],
        "frozen": ["beam membership", "geometry", "thickness distribution", "jitter", "absorption", "dose terms",
                   "observation membership", "frame/detector/intercrystal scales", "empirical uncertainty factors", "merge policy"],
        "updated": ["H-free coordinates and ADPs", "SHELXL overall scale", "Bloch/Born predictions", "q and propagated uncertainties"],
        "hydrogen_history": "H-free input supplied by the user; absence of H is checked here, earlier solution history is external",
    }
    if previous.exists():
        old = json.loads(previous.read_text())
        for key in ("reference_model_sha256", "input_model_sha256", "input_hkl_sha256", "datasets", "cycles"):
            # Canonical JSON comparison handles the dose-reference NaN sentinel.
            if json.dumps(old[key], sort_keys=True) != json.dumps(manifest[key], sort_keys=True):
                raise ValueError("Inputs or settings changed. Choose a new --out-dir.")
    save_json(previous, manifest)
    if args.dry_run:
        save_json(out / "STATUS.json", {"status": "dry_run", "datasets": len(reference["datasets"])})
        print(f"Matched never-H plan: {previous}", flush=True)
        return 0

    snapshots = out / "reference_inputs"
    snapshots.mkdir(exist_ok=True)
    for original, name in ((Path(reference["model"]), "full_H.res"), (source, "input_never_H.res"), (baseline, "input_never_H.hkl")):
        shutil.copy2(original, snapshots / name)
    reference["model"] = str(snapshots / "full_H.res")

    def status(stage, **extra):
        save_json(out / "STATUS.json", {"status": "running", "stage": stage, "pid": os.getpid(), "updated": time.time(), **extra})
        print(stage, flush=True)

    for ds in reference["datasets"]:
        status(f"Verify full-H replay: {ds['name']}")
        predict_dataset(ds, Path(reference["model"]), Path(reference["model"]), out / "reference_replay" / ds["name"], workers=min(workers, 4), smoke=True)

    def calculate(model, directory):
        rows = {}
        for ds in reference["datasets"]:
            status(f"{directory.name}: {ds['name']}", model=str(model))
            rows[ds["name"]] = predict_dataset(ds, model, Path(reference["model"]), directory / ds["name"], workers=workers)
        return rows

    current = aligned
    records = []
    warning = ""
    for cycle in range(1, args.cycles + 1):
        directory = out / f"cycle_{cycle:02d}"
        stamp = directory / "cycle.json"
        if stamp.exists():
            record = json.loads(stamp.read_text())
            candidate = Path(record["model"])
            if record["input_sha256"] == sha256_file(current) and candidate.is_file() and record["model_sha256"] == sha256_file(candidate):
                require_no_hydrogen(candidate)
                current = candidate
                records.append(record)
                continue
        rows = calculate(current, directory)
        hkl = directory / "model.hkl"
        write_fixed_width_hkl(hkl, merge_changed_predictions(reference, frozen, rows))
        write_endpoint_ins(current, directory / "model.ins", hkl_name="model.hkl", keep_exti=False, wght=(0.0, 0.0), ls_cycles=1)
        status(f"SHELXL cycle {cycle}")
        recorded_run([shelxl, "model", "-t1"], directory, "shelxl")
        candidate = directory / "model.res"
        require_no_hydrogen(candidate)
        health = shelx_res_health(candidate, directory / "model.lst")
        if not health.get("safe_for_feedback"):
            warning = f"Stopped before adopting unsafe SHELXL model at cycle {cycle}; retained the previous model."
            save_json(directory / "rejected_model.json", health)
            break
        record = {"cycle": cycle, "input_sha256": sha256_file(current), "model": str(candidate),
                  "model_sha256": sha256_file(candidate), "hkl_sha256": sha256_file(hkl), "health": health}
        save_json(stamp, record)
        records.append(record)
        current = candidate
    final = out / "final"
    final_rows = calculate(current, final)
    shutil.copy2(current, final / "model.res")
    write_fixed_width_hkl(final / "model.hkl", merge_changed_predictions(reference, frozen, final_rows))
    maps = {}
    if args.maps:
        from grond.workflows.matched_maps import build_matched_maps
        status("Build matched maps")
        maps = build_matched_maps(conventional, final, out / "maps", shelxl, maptool, recorded_run, args.map_grid_spacing)
    result = {"status": "complete_with_warning" if warning else "complete", "warning": warning,
              "model": str(final / "model.res"), "cycles": records, "maps": maps,
              "reference_replay_passed": True, "commands": str(out / "RUN_MANIFEST.json")}
    save_json(out / "RESULT.json", result)
    save_json(out / "STATUS.json", {"status": result["status"], "warning": warning, "updated": time.time()})
    print(f"Matched never-H result: {out / 'RESULT.json'}", flush=True)
    return 0


def cmd_never_h(args) -> int:
    import fcntl

    if args.cycles < 0 or not 0 < args.map_grid_spacing <= 1:
        raise SystemExit("--cycles must be nonnegative and --map-grid-spacing must be in (0, 1] Angstrom.")
    out = Path(args.out_dir).expanduser().resolve()
    out.mkdir(parents=True, exist_ok=True)
    with (out / ".run.lock").open("w") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit(f"A never-H job is already running in {out}") from exc
        try:
            return _run(args, out)
        except Exception as exc:
            save_json(out / "STATUS.json", {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "updated": time.time()})
            raise SystemExit(f"never-h failed: {exc}. Details: {out / 'STATUS.json'}") from exc
