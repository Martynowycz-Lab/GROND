"""Figure maps for a finished run: every map boxed around one reference model placement.

``grond figure-maps --run-dir RUN --out-dir NEW`` reads RUN and writes only to
NEW.  It collects the run's maps (native dynamical Fo, 2Fo-Fc and Fo-Fc; the
corrected-endpoint shelx2map maps, weighted and raw; conventional shelx2map maps
when the run has a conventional .res/.fcf).  Maps that ``grond finish`` has not
written are calculated in NEW with the same code.  Each map's model is placed on
the reference model (``--reference-model``; by default the full-H reference a
never-h directory recorded, else the run's own model), and
every map is boxed around the reference placement plus a margin, so maps from a
full-H and a never-H run overlay directly.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path

from grond.cli_support.tools import external_tool_statuses
from grond.cli_support.values import try_read_json
from grond.macromolecule import sha256_file
from grond.maps.model_box import (
    DEFAULT_MODEL_BOX_MARGIN_A,
    native_package_entries,
    shelx2map_record_entries,
    write_model_box_maps,
)
from grond.maps.placement import PlacementOptions
from grond.workflows.finish_maps import (
    _finish_native_row_tables,
    _publication_endpoint_from_result,
    run_finish_endpoint_maps,
    run_finish_native_dynamic_maps,
    shelx2map_model_maps,
)

CONVENTIONAL_MAP_DEFINITION = (
    "Conventional SHELXL model maps from the run's fresh conventional baseline (.res/.fcf/.hkl), made with the same "
    "shelx2map settings as the corrected-endpoint maps: -w 1 primary maps and a raw -w 0 Fo-Fc diagnostic."
)


def _inside(path: Path, parent: Path) -> bool:
    """Return whether ``path`` resolves to a location inside ``parent``."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def _snapshot(directory: Path) -> dict[str, tuple[int, int]]:
    """Size and modification time of every file, to prove the run was only read."""

    snapshot = {}
    for root, _dirs, files in os.walk(directory):
        for name in files:
            path = Path(root) / name
            try:
                stat = path.stat()
            except OSError:
                continue
            snapshot[str(path.relative_to(directory))] = (stat.st_size, stat.st_mtime_ns)
    return snapshot


def _native_manifest(path: Path) -> dict[str, object]:
    """Return the native dynamical map manifest at ``path`` when its status is ok, else {}."""

    manifest = try_read_json(path)
    return manifest if manifest.get("status") == "ok" else {}


def _conventional_inputs(result: Mapping[str, object], args: argparse.Namespace) -> dict[str, Path] | None:
    """Return the conventional model's res/fcf/hkl paths and their source, or None when none are known.

    ``--conventional-res`` (and ``--conventional-fcf``) take precedence over the run's recorded conventional baseline
    outputs.
    """

    res_text = str(getattr(args, "conventional_res", "") or "").strip()
    if res_text:
        res = Path(res_text).expanduser().resolve()
        fcf_text = str(getattr(args, "conventional_fcf", "") or "").strip()
        fcf = Path(fcf_text).expanduser().resolve() if fcf_text else res.with_suffix(".fcf")
        return {"res": res, "fcf": fcf, "hkl": res.with_suffix(".hkl"), "source": "argument"}
    baseline = result.get("conventional_baseline") if isinstance(result.get("conventional_baseline"), Mapping) else {}
    outputs = baseline.get("outputs") if isinstance(baseline.get("outputs"), Mapping) else {}
    paths = {}
    for key in ("res", "fcf", "hkl"):
        item = outputs.get(key) if isinstance(outputs, Mapping) else None
        text = str(item.get("path", "") if isinstance(item, Mapping) else "")
        if text:
            paths[key] = Path(text).expanduser()
    if "res" in paths and "fcf" in paths:
        return {**paths, "source": "conventional_baseline"}
    return None


def _grond_run_sources(run_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict[str, object]:
    """Collect the map families and map entries of a GROND run directory, and its primary model.

    Maps the run has not written (native dynamical, endpoint and conventional shelx2map) are calculated under
    ``out_dir/source_maps`` when ``--generate-missing`` is on; the run directory is only read.
    """

    result = try_read_json(run_dir / "grond_summary.json")
    finish = try_read_json(run_dir / "finish_manifest.json")
    finish_maps = finish.get("maps") if isinstance(finish.get("maps"), Mapping) else {}
    generate = bool(getattr(args, "generate_missing", True))
    source_dir = out_dir / "source_maps"
    families: dict[str, dict[str, object]] = {}
    entries: list[dict[str, object]] = []

    native = _native_manifest(run_dir / "native_dynamic_maps" / "native_dynamic_maps_manifest.json")
    if native:
        families["native_dynamic"] = {"origin": "run", "manifest": str(run_dir / "native_dynamic_maps")}
    elif generate:
        try:
            native = run_finish_native_dynamic_maps(run_dir, result, args, out_dir=source_dir / "native_dynamic_maps")
        except (OSError, RuntimeError, ValueError) as exc:
            native = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        families["native_dynamic"] = {
            "error": native.get("error", ""),
            "origin": "generated_in_out_dir",
            "status": native.get("status"),
            "manifest": str((native.get("outputs") or {}).get("manifest", "")) if isinstance(native, Mapping) else "",
            "row_tables": [str(path) for _label, path in _finish_native_row_tables(result)],
            "row_model_contract": native.get("row_model_contract"),
        }
    if native.get("status") == "ok":
        entries.extend(native_package_entries(native))

    endpoint_record = finish_maps.get("shelx2map") if isinstance(finish_maps.get("shelx2map"), Mapping) else {}
    tools = external_tool_statuses(args)
    shelx2map_ok = bool(tools.get("shelx2map", {}).get("available"))
    if endpoint_record.get("status") == "ok":
        families["endpoint_shelx2map"] = {"origin": "run", "endpoint_res": endpoint_record.get("endpoint_res")}
    elif generate and shelx2map_ok:
        try:
            endpoint_record = run_finish_endpoint_maps(result, args, out_dir=source_dir / "endpoint_shelx2map")
        except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
            endpoint_record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        families["endpoint_shelx2map"] = {
            "error": endpoint_record.get("error", ""),
            "origin": "generated_in_out_dir",
            "status": endpoint_record.get("status"),
            "source_endpoint_res": endpoint_record.get("source_endpoint_res", endpoint_record.get("endpoint_res")),
        }
    elif generate:
        families["endpoint_shelx2map"] = {"origin": "not_generated", "status": "missing_shelx2map"}
    if endpoint_record.get("status") == "ok":
        entries.extend(shelx2map_record_entries(endpoint_record, family="grond_corrected", model_label="endpoint"))

    conventional = _conventional_inputs(result, args) if getattr(args, "conventional", True) else None
    if conventional is not None and generate and shelx2map_ok:
        missing = [key for key in ("res", "fcf") if not Path(conventional[key]).is_file()]
        if missing:
            families["conventional_shelx2map"] = {"origin": "not_generated", "status": "missing_inputs",
                                                  "missing": missing}
        else:
            try:
                record = shelx2map_model_maps(
                    {"endpoint_hkl": str(conventional.get("hkl", "")), "endpoint_fcf": str(conventional["fcf"])},
                    Path(conventional["res"]),
                    args,
                    out_dir=source_dir / "conventional_shelx2map",
                    prefix="conventional",
                    definition=CONVENTIONAL_MAP_DEFINITION,
                )
            except (OSError, RuntimeError, ValueError, subprocess.SubprocessError) as exc:
                record = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
            families["conventional_shelx2map"] = {
                "error": record.get("error", ""),
                "origin": "generated_in_out_dir",
                "status": record.get("status"),
                "inputs": conventional.get("source"),
                "source_res": str(conventional["res"]),
                "source_fcf": str(conventional["fcf"]),
            }
            if record.get("status") == "ok":
                entries.extend(shelx2map_record_entries(record, family="conventional", model_label="conventional"))
    elif conventional is None:
        families["conventional_shelx2map"] = {"origin": "not_available"}

    endpoint = _publication_endpoint_from_result(result)
    primary_text = str(endpoint.get("endpoint_res", "") or "").strip()
    primary = Path(primary_text).expanduser() if primary_text else None
    if primary is None or not primary.is_file():
        primary = Path(str(entries[0]["model"])) if entries else None
    return {"kind": "grond_run", "families": families, "entries": entries, "primary_model": primary}


def _in_run_dir(recorded: str, run_dir: Path | None) -> str:
    """A path never-h recorded, found in ``run_dir`` when that directory was copied or moved.

    never-h writes absolute paths (of the machine it ran on).  A path outside ``run_dir``
    whose last components (at least a directory and the file name, e.g.
    ``final/model.res``) name a file inside it is taken from ``run_dir``, so the
    directory that was passed is the one read.
    """

    path = Path(str(recorded))
    if run_dir is None or _inside(path, run_dir):
        return str(path)
    for count in range(min(len(path.parts) - 1, 4), 1, -1):
        candidate = Path(run_dir).joinpath(*path.parts[-count:])
        if candidate.is_file():
            return str(candidate)
    return str(path)


def never_h_map_entries(map_manifest: Path, run_dir: Path | None = None) -> list[dict[str, object]]:
    """Model-box entries for the maps listed in a never-h maps/MAP_MANIFEST.json."""

    entries = []
    for record in json.loads(Path(map_manifest).read_text()):
        label = f"never_h_{record['type']}_{record['support']}_w{record['weight']}_{record['kind']}"
        entry = {"label": label, "map": _in_run_dir(record["map"], run_dir),
                 "model": _in_run_dir(record["model"], run_dir),
                 "model_label": f"never_h_{record['type']}", "family": f"never_h_{record['type']}",
                 "kind": record["kind"], "weighting": f"shelx2map -w {record['weight']}"}
        if entry["map"] != str(record["map"]):
            entry["recorded_map"] = str(record["map"])
        entries.append(entry)
    return entries


def never_h_model_boxes(maps_dir: Path, full_h_model: Path, margin_A: float = DEFAULT_MODEL_BOX_MARGIN_A) -> dict:
    """Box every never-h map around the full-H reference placement (never-h itself calls this)."""

    maps_dir = Path(maps_dir)
    return write_model_box_maps(
        never_h_map_entries(maps_dir / "MAP_MANIFEST.json"),
        Path(full_h_model),
        margin_A=margin_A,
        out_dir=maps_dir / "model_box",
        models_dir=maps_dir / "model_box",
        sigma_scaled=True,
        write_all_models=True,
    )


def _never_h_sources(run_dir: Path) -> dict[str, object]:
    """Collect the map entries, primary model and recorded full-H reference of a never-h run directory."""

    entries = never_h_map_entries(run_dir / "maps" / "MAP_MANIFEST.json", run_dir)
    result = try_read_json(run_dir / "RESULT.json")
    primary_text = str(result.get("model", "") or "")
    primary = Path(_in_run_dir(primary_text, run_dir)) if primary_text else run_dir / "final" / "model.res"
    full_h = run_dir / "reference_inputs" / "full_H.res"
    return {
        "kind": "never_h_run",
        "families": {"never_h_maps": {"origin": "run", "manifest": str(run_dir / "maps" / "MAP_MANIFEST.json")}},
        "entries": entries,
        "primary_model": primary if primary.is_file() else None,
        "recorded_full_h_reference": str(full_h) if full_h.is_file() else "",
    }


def run_figure_maps(args: argparse.Namespace) -> dict[str, object]:
    """Collect, generate if missing, place and box every map of a run; write only to the out dir."""

    run_dir = Path(args.run_dir).expanduser().resolve()
    out_dir = Path(args.out_dir).expanduser().resolve()
    if not run_dir.is_dir():
        raise SystemExit(f"Run directory not found: {run_dir}")
    if _inside(out_dir, run_dir) or _inside(run_dir, out_dir):
        raise SystemExit("--out-dir must be outside the run directory (and must not contain it); the run is only read.")
    if out_dir.exists() and any(out_dir.iterdir()):
        raise SystemExit(f"--out-dir must be new or empty; it will not overwrite an earlier result: {out_dir}")
    margin = float(getattr(args, "margin_A", DEFAULT_MODEL_BOX_MARGIN_A))
    if not margin >= 0.0:
        raise SystemExit("--margin-A must be a non-negative number of Angstroms.")
    before = _snapshot(run_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if (run_dir / "grond_summary.json").is_file():
        sources = _grond_run_sources(run_dir, out_dir, args)
    elif (run_dir / "maps" / "MAP_MANIFEST.json").is_file():
        sources = _never_h_sources(run_dir)
    else:
        raise SystemExit(
            f"{run_dir} is neither a GROND run (grond_summary.json) nor a never-H run (maps/MAP_MANIFEST.json)."
        )
    entries = sources["entries"]
    reference_text = str(getattr(args, "reference_model", "") or "").strip()
    recorded_full_h = str(sources.get("recorded_full_h_reference", "") or "")
    if reference_text:
        reference, reference_source = Path(reference_text).expanduser().resolve(), "argument"
    elif recorded_full_h:
        # never-h boxes its own maps around this full-H model; default to it so
        # figure-maps puts the never-H maps on the full-H placement as well.
        reference, reference_source = Path(recorded_full_h), "recorded_full_h_reference"
    else:
        reference, reference_source = sources.get("primary_model"), "run_model"
    manifest: dict[str, object] = {
        "schema_version": 1,
        "command": [sys.executable, "-m", "grond", *getattr(args, "argv", [])] if getattr(args, "argv", None) else [],
        "run_dir": str(run_dir),
        "run_kind": sources["kind"],
        "out_dir": str(out_dir),
        "margin_A": margin,
        "rmsd_max_A": float(args.rmsd_max_A),
        "reference_model": str(reference) if reference else "",
        "reference_is_run_model": reference_source == "run_model",
        "reference_source": reference_source,
        "run_model": str(sources.get("primary_model") or ""),
        "sources": sources["families"],
        "recorded_full_h_reference": sources.get("recorded_full_h_reference", ""),
    }
    if not entries or reference is None or not Path(reference).is_file():
        manifest["status"] = "no_maps" if not entries else "missing_reference_model"
    else:
        boxes = write_model_box_maps(
            entries,
            Path(reference),
            margin_A=margin,
            out_dir=out_dir / "model_box",
            models_dir=out_dir / "models",
            sigma_scaled=True,
            placement=PlacementOptions(
                rmsd_max_A=float(args.rmsd_max_A),
                allow_improper_space_group_operations=bool(getattr(args, "allow_improper_space_group_operations", False)),
            ),
            write_all_models=True,
        )
        for entry in entries:
            item = boxes["maps"].get(entry["label"])
            if isinstance(item, dict):
                item.update({key: entry[key] for key in ("family", "kind", "weighting", "recorded_map") if key in entry})
                item["unit_cell_map"] = str(Path(str(entry["map"])).resolve())
                unit_cell_map = Path(str(entry["map"]))
                # A missing map is already reported as failed; do not abort the manifest.
                item["unit_cell_map_sha256"] = sha256_file(unit_cell_map) if unit_cell_map.is_file() else ""
        manifest["model_box"] = boxes
        manifest["status"] = boxes["status"]
        manifest["warnings"] = list(boxes.get("warnings") or [])
    after = _snapshot(run_dir)
    manifest["run_dir_unchanged"] = before == after
    if before != after:
        changed = sorted(set(before) ^ set(after) | {key for key in before if key in after and before[key] != after[key]})
        manifest["run_dir_changed_files"] = changed[:50]
        manifest["status"] = "failed_run_dir_changed"
    (out_dir / "FIGURE_MAPS_MANIFEST.json").write_text(json.dumps(manifest, indent=2, allow_nan=True) + "\n")
    _write_readme(out_dir, manifest)
    return manifest


def _write_readme(out_dir: Path, manifest: Mapping[str, object]) -> None:
    """Write README.md in ``out_dir``: what each output file is, one table row per map, and the placements."""

    boxes = manifest.get("model_box") if isinstance(manifest.get("model_box"), Mapping) else {}
    lines = [
        "# GROND figure maps",
        "",
        f"Run: `{manifest.get('run_dir')}`  ",
        f"Reference model: `{manifest.get('reference_model')}`"
        + (" (the run's own model)" if manifest.get("reference_is_run_model") else "")
        + (" (the full-H reference recorded by never-h)"
           if manifest.get("reference_source") == "recorded_full_h_reference" else ""),
        f"Margin: {manifest.get('margin_A')} A. Status: {manifest.get('status')}.",
        "",
        "- `model_box/<map>_model_box.map`: the map boxed around the whole reference model plus the margin, in the",
        "  map's own units. `<map>_model_box_sigma.map` holds (value - unit-cell mean) / unit-cell RMS, so a",
        "  contour of 1.0 is one unit-cell sigma in any viewer (switch off per-file normalisation, for example",
        "  PyMOL `set normalize_ccp4_maps, off`).",
        "- `models/model_box_reference.pdb|res`: the reference placement all boxes are cut around.",
        "- `models/<model>_placed.pdb|res`: each map's own model moved onto the reference by the recorded",
        "  space-group operation (see `FIGURE_MAPS_MANIFEST.json` -> `model_box.placements`).",
        "- `models/<model>_own.pdb|res`: a model whose placement was refused (RMSD limit or a hand change, which",
        "  is never applied); its maps are boxed around the model itself and do not overlay the reference.",
        "- `source_maps/`: unit-cell maps calculated here because the run had not written them.",
        "",
        "| map | family | weighting | box grid | unit-cell RMS |",
        "|---|---|---|---|---|",
    ]
    for label, item in (boxes.get("maps") or {}).items():  # type: ignore[union-attr]
        box = item.get("box_map") or {}
        source = item.get("source") or {}
        lines.append(
            f"| {label} | {item.get('family', '')} | {item.get('weighting', '')} | "
            f"{'x'.join(str(v) for v in box.get('grid_size', []))} | {source.get('unit_cell_rms', float('nan')):.4g} |"
        )
    lines.append("")
    for key, report in (boxes.get("placements") or {}).items():  # type: ignore[union-attr]
        if report.get("status") == "aligned":
            lines.append(
                f"- {key}: {report['operation']['triplet']}, RMSD {report['rmsd_A']:.4f} A over "
                f"{report['matched_atoms']} non-H atoms ({report['matching']})"
            )
        else:
            lines.append(f"- {key}: {report.get('status')}: {report.get('error', '')}")
    (out_dir / "README.md").write_text("\n".join(lines) + "\n")
