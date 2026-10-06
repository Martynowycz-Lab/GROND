"""CCP4 maps boxed around a whole model, cut from the periodic unit-cell map.

A unit-cell map shows a molecule that crosses a cell edge in pieces.  For
figures GROND also writes, beside every unit-cell map, a map whose box covers
the whole model plus a margin (default 3 A).  The box values are read from the
full-cell map through its periodicity (and, for a partial map, the stored
symmetry operators), so they are identical to the unit-cell map wherever the
two overlap.  The unit-cell maps are kept unchanged.

Box maps are CCP4 files in P1 with the full-cell sampling in the header and a
non-zero start index, which PyMOL, ChimeraX and Coot read as a box in the
crystal frame.  A companion ``*_sigma`` file (optional) stores
``(value - unit-cell mean) / unit-cell RMS`` so a contour of 1.0 is one sigma of
the whole unit cell, not of the box.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.model import parse_model, shelx_u_cart_matrix
from grond.macromolecule import sha256_file
from grond.maps.ccp4 import read_ccp4_map
from grond.maps.placement import (
    PlacementOptions,
    PlacementRefused,
    assemble_molecules,
    place_model_on_reference,
    spacegroup_of,
)

DEFAULT_MODEL_BOX_MARGIN_A = 3.0
MODEL_BOX_SUFFIX = "_model_box"


def model_structure(model: Mapping[str, object], *, name: str = "GROND") -> gemmi.Structure:
    """Return a one-model gemmi Structure (Cartesian positions, B from Ueq, ANISOU)."""

    cell = gemmi.UnitCell(*[float(v) for v in model["cell"][:6]])  # type: ignore[index]
    structure = gemmi.Structure()
    structure.name = name
    structure.cell = cell
    group = spacegroup_of(model)
    structure.spacegroup_hm = group.hm if group else "P 1"
    chain = gemmi.Chain("A")
    residue = None
    names: set[str] = set()
    serial = 0

    def new_residue():
        """Return a new empty UNL hetero residue with the next sequence number."""

        nonlocal serial
        serial += 1
        created = gemmi.Residue()
        created.name = "UNL"
        created.seqid = gemmi.SeqId(serial, " ")
        created.het_flag = "H"
        return created

    for atom in model.get("atoms") or []:  # type: ignore[union-attr]
        label = str(atom["name"])[:4]
        if residue is None or label.upper() in names:
            if residue is not None:
                chain.add_residue(residue)
            residue = new_residue()
            names = set()
        names.add(label.upper())
        element = str(atom.get("element", "X")).strip().capitalize() or "X"
        created = gemmi.Atom()
        created.name = label
        created.element = gemmi.Element(element)
        created.pos = cell.orthogonalize(gemmi.Fractional(float(atom["x"]), float(atom["y"]), float(atom["z"])))
        created.occ = float(min(max(float(atom.get("occ", 1.0)), 0.0), 1.0))
        if atom.get("adp_type") == "anis":
            tensor = shelx_u_cart_matrix(atom, model["cell"])
            created.aniso = gemmi.SMat33f(
                float(tensor[0, 0]), float(tensor[1, 1]), float(tensor[2, 2]),
                float(tensor[0, 1]), float(tensor[0, 2]), float(tensor[1, 2]),
            )
            ueq = float(np.trace(tensor) / 3.0)
        else:
            ueq = float(atom.get("uiso", 0.0))
        created.b_iso = float(8.0 * math.pi * math.pi * max(ueq, 0.0))
        residue.add_atom(created)
    if residue is not None:
        chain.add_residue(residue)
    gemmi_model = gemmi.Model("1")
    gemmi_model.add_chain(chain)
    structure.add_model(gemmi_model)
    return structure


def write_model_pdb(model: Mapping[str, object], path: Path, *, remarks: Sequence[str] = ()) -> Path:
    """Write a PDB file of a small-molecule model for viewers (not for refinement)."""

    structure = model_structure(model, name=Path(path).stem)
    lines = [f"REMARK 250 {remark}"[:80] for remark in remarks]
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(line.ljust(80) + "\n" for line in lines) + structure.make_pdb_string())
    return path


def write_placed_res(source_res: Path, destination: Path, model: Mapping[str, object], remarks: Sequence[str]) -> Path:
    """Copy a SHELX model with new coordinates; Q-peaks are dropped, REM lines record the move."""

    from grond.bloch.model import is_shelx_q_peak
    from grond.refine.native_output import update_res_coordinates

    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    update_res_coordinates(Path(source_res), destination, model)
    lines = destination.read_text(errors="replace").splitlines()
    output: list[str] = []
    skip_continuation = False
    for line in lines:
        words = line.split()
        if skip_continuation:
            skip_continuation = line.rstrip().endswith("=")
            continue
        if words and is_shelx_q_peak(words[0]) and len(words) >= 5:
            skip_continuation = line.rstrip().endswith("=")
            continue
        output.append(line)
        if words and words[0].upper() == "TITL":
            output.extend(f"REM {remark}"[:79] for remark in remarks)
    destination.write_text("\n".join(output) + "\n")
    return destination


def model_fractional_box(model: Mapping[str, object], margin_A: float) -> dict[str, object]:
    """Fractional box around every atom plus ``margin_A`` (gemmi calculate_fractional_box)."""

    if not math.isfinite(float(margin_A)) or float(margin_A) < 0.0:
        raise ValueError("model box margin must be a non-negative number of Angstroms")
    structure = model_structure(model)
    box = structure.calculate_fractional_box(margin=float(margin_A))
    minimum = [float(box.minimum.x), float(box.minimum.y), float(box.minimum.z)]
    maximum = [float(box.maximum.x), float(box.maximum.y), float(box.maximum.z)]
    return {
        "fractional_min": minimum,
        "fractional_max": maximum,
        "margin_A": float(margin_A),
        "atoms": int(len(model.get("atoms") or [])),  # type: ignore[arg-type]
        "definition": "gemmi Structure.calculate_fractional_box(margin): atom extent plus margin*(a*, b*, c*)",
    }


def load_unit_cell_map(path: Path) -> tuple[gemmi.Ccp4Map, dict[str, object]]:
    """Read a CCP4 map and expand it to the full cell (stored symmetry, then periodicity)."""

    path = Path(path)
    ccp4 = read_ccp4_map(path)
    if any(abs(ccp4.header_float(word)) > 1e-6 for word in (50, 51, 52)):
        raise ValueError(f"{path}: nonzero Cartesian map origin; cannot place a model box")
    source_group = ccp4.grid.spacegroup.xhm() if ccp4.grid.spacegroup else "unknown"
    source_shape = list(ccp4.grid.shape)
    ccp4.setup(float("nan"))
    values = np.array(ccp4.grid, copy=False)
    if not np.isfinite(values).all():
        raise ValueError(f"{path}: map does not cover the full cell after symmetry expansion")
    ccp4.grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    return ccp4, {
        "source_map": str(path.resolve()),
        "source_sha256": sha256_file(path),
        "source_spacegroup": source_group,
        "source_shape": source_shape,
        "unit_cell_shape": list(ccp4.grid.shape),
        "unit_cell": [float(v) for v in (ccp4.grid.unit_cell.a, ccp4.grid.unit_cell.b, ccp4.grid.unit_cell.c,
                                         ccp4.grid.unit_cell.alpha, ccp4.grid.unit_cell.beta,
                                         ccp4.grid.unit_cell.gamma)],
        "grid_spacing_A": [float(ccp4.grid.unit_cell.a / ccp4.grid.nu), float(ccp4.grid.unit_cell.b / ccp4.grid.nv),
                           float(ccp4.grid.unit_cell.c / ccp4.grid.nw)],
        "unit_cell_mean": float(np.mean(values, dtype=np.float64)),
        "unit_cell_rms": float(np.std(values, dtype=np.float64)),
        "unit_cell_min": float(np.min(values)),
        "unit_cell_max": float(np.max(values)),
    }


def shift_map_origin(ccp4: gemmi.Ccp4Map, shift_fractional: Sequence[float]) -> dict[str, object]:
    """Move a full-cell map by a fractional shift in place: new(x) = old(x - shift).

    A shift that is a whole number of grid steps is applied exactly by rolling
    the grid.  Otherwise the band-limited map is moved by a Fourier phase shift.
    """

    shift = np.asarray(shift_fractional, dtype=float)
    values = np.array(ccp4.grid, copy=True).astype(np.float64)
    shape = np.asarray(values.shape)
    steps = shift * shape
    if np.allclose(steps, np.rint(steps), atol=1e-6):
        moved = np.roll(values, tuple(int(v) for v in np.rint(steps)), axis=(0, 1, 2))
        method = "grid_roll"
    else:
        frequencies = np.meshgrid(*(np.fft.fftfreq(n) * n for n in values.shape), indexing="ij")
        phase = np.exp(-2j * np.pi * sum(freq * value for freq, value in zip(frequencies, shift, strict=True)))
        moved = np.fft.ifftn(np.fft.fftn(values) * phase).real
        method = "fourier_phase_shift"
    ccp4.grid.array[:] = moved.astype(np.float32)
    return {"shift_fractional": [float(v) for v in shift], "method": method}


def write_unit_cell_map(ccp4: gemmi.Ccp4Map, path: Path) -> dict[str, object]:
    """Write a map's full-cell grid as a P1 CCP4 file; return its path, SHA-256 and grid shape."""

    grid = gemmi.FloatGrid(*ccp4.grid.shape)
    grid.unit_cell = ccp4.grid.unit_cell
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    grid.array[:] = np.array(ccp4.grid, copy=False)
    result = gemmi.Ccp4Map()
    result.grid = grid
    result.update_ccp4_header()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    result.write_ccp4_map(str(path))
    return {"map": str(path.resolve()), "sha256": sha256_file(path), "shape": list(grid.shape)}


def box_grid_range(box: Mapping[str, object], shape: Sequence[int]) -> tuple[list[int], list[int]]:
    """First grid index and size covering the fractional box (rounded outward)."""

    minimum = np.asarray(box["fractional_min"], dtype=float)
    maximum = np.asarray(box["fractional_max"], dtype=float)
    counts = np.asarray(shape, dtype=float)
    start = np.floor(minimum * counts + 1e-9).astype(int)
    stop = np.ceil(maximum * counts - 1e-9).astype(int)
    return start.tolist(), (stop - start + 1).tolist()


def write_box_map(
    ccp4: gemmi.Ccp4Map,
    box: Mapping[str, object],
    destination: Path,
    *,
    sigma_scaled: bool = False,
    unit_cell_mean: float = 0.0,
    unit_cell_rms: float = 1.0,
) -> dict[str, object]:
    """Write the box of a full-cell map (P1, full-cell sampling, start index in the header)."""

    shape = list(ccp4.grid.shape)
    start, size = box_grid_range(box, shape)
    grid = gemmi.FloatGrid(*shape)
    grid.unit_cell = ccp4.grid.unit_cell
    grid.spacegroup = gemmi.find_spacegroup_by_name("P 1")
    values = np.array(ccp4.grid, copy=False)
    if sigma_scaled:
        if not unit_cell_rms > 0.0:
            raise ValueError("unit-cell RMS must be positive for a sigma-scaled box")
        values = (values.astype(np.float64) - unit_cell_mean) / unit_cell_rms
    grid.array[:] = values.astype(np.float32)
    boxed = gemmi.Ccp4Map()
    boxed.grid = grid
    boxed.update_ccp4_header()
    # set_extent keeps points inside the box, so pass one just half a step wider
    # than the outward-rounded index range.
    crafted = gemmi.FractionalBox()
    crafted.extend(gemmi.Fractional(*((s - 0.5) / n for s, n in zip(start, shape, strict=True))))
    crafted.extend(gemmi.Fractional(*((s + z - 0.5) / n for s, z, n in zip(start, size, shape, strict=True))))
    boxed.set_extent(crafted)
    if list(boxed.grid.shape) != list(size) or [boxed.header_i32(word) for word in (5, 6, 7)] != list(start):
        raise RuntimeError("model box extent does not match the requested grid range")
    data = np.array(boxed.grid, copy=False)
    # set_extent leaves the full-cell statistics in the header; describe the box itself.
    for word, value in ((20, np.min(data)), (21, np.max(data)), (22, np.mean(data, dtype=np.float64)),
                        (55, np.std(data, dtype=np.float64))):
        boxed.set_header_float(word, float(value))
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    boxed.write_ccp4_map(str(destination))
    return {
        "map": str(destination.resolve()),
        "sha256": sha256_file(destination),
        "values": "unit_cell_sigma" if sigma_scaled else "absolute",
        "grid_start": [int(v) for v in start],
        "grid_size": [int(v) for v in size],
        "grid_sampling": [int(v) for v in shape],
        "box_fractional_min": [float(s / n) for s, n in zip(start, shape, strict=True)],
        "box_fractional_max": [float((s + z - 1) / n) for s, z, n in zip(start, size, shape, strict=True)],
        "box_mean": float(np.mean(data, dtype=np.float64)),
        "box_rms": float(np.std(data, dtype=np.float64)),
    }


def atom_box_clearance_A(model: Mapping[str, object], record: Mapping[str, object]) -> float:
    """Smallest distance (A, along cell-plane normals) from an atom to the box face."""

    cell = gemmi.UnitCell(*[float(v) for v in model["cell"][:6]])  # type: ignore[index]
    reciprocal = np.asarray([cell.reciprocal().a, cell.reciprocal().b, cell.reciprocal().c], dtype=float)
    xyz = np.asarray([[float(a["x"]), float(a["y"]), float(a["z"])] for a in model.get("atoms") or []])  # type: ignore[union-attr]
    if not len(xyz):
        return float("nan")
    low = np.asarray(record["box_fractional_min"], dtype=float)
    high = np.asarray(record["box_fractional_max"], dtype=float)
    clearance = np.minimum(xyz - low, high - xyz) / reciprocal
    return float(np.min(clearance))


def native_package_entries(package: Mapping[str, object], model_label: str = "native_dynamic_model") -> list[dict]:
    """Model-box entries for a native dynamical map package manifest."""

    summary = package.get("summary") if isinstance(package.get("summary"), Mapping) else {}
    model = str(summary.get("model", "") or "")  # type: ignore[union-attr]
    maps = package.get("maps") if isinstance(package.get("maps"), Mapping) else {}
    entries = []
    for kind in ("fo", "2fofc", "fofc"):
        item = maps.get(kind) if isinstance(maps, Mapping) else None
        if isinstance(item, Mapping) and item.get("map") and Path(str(item["map"])).is_file() and model:
            entries.append({"label": f"grond_native_dynamic_{kind}", "map": str(item["map"]), "model": model,
                            "model_label": model_label, "family": "native_dynamic", "kind": kind,
                            "weighting": "none (propagated-variance merge only)"})
    return entries


def shelx2map_record_entries(record: Mapping[str, object], *, family: str, model_label: str) -> list[dict]:
    """Model-box entries for a shelx2map map record (finish endpoint or conventional maps)."""

    model = str(record.get("endpoint_res", "") or "")
    maps = record.get("maps") if isinstance(record.get("maps"), Mapping) else {}
    entries = []
    for label, item in (maps or {}).items():  # type: ignore[union-attr]
        if not isinstance(item, Mapping) or item.get("status") not in {"ok", None}:
            continue
        path = Path(str(item.get("map", "")))
        if not path.is_file() or not model:
            continue
        weight = item.get("shelx2map_weight")
        entries.append({"label": f"{family}_{label}", "map": str(path), "model": model, "model_label": model_label,
                        "family": family, "kind": str(label).replace("_unweighted_diagnostic", ""),
                        "weighting": f"shelx2map -w {weight:g}" if isinstance(weight, (int, float)) else "shelx2map"})
    return entries


def _stem_for(label: str) -> str:
    """Return a file-name stem for a map label: other characters become ``_``; ``map`` when nothing is left."""

    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(label)).strip("_") or "map"


def write_model_box_maps(
    entries: Sequence[Mapping[str, object]],
    reference_model: Path,
    *,
    margin_A: float = DEFAULT_MODEL_BOX_MARGIN_A,
    out_dir: Path | None = None,
    models_dir: Path | None = None,
    sigma_scaled: bool = False,
    placement: PlacementOptions | None = None,
    write_all_models: bool = False,
    write_res: bool = True,
) -> dict[str, object]:
    """Write a model box for every map, all cut around the reference model's placement.

    The shared box covers the reference and every model placed on it, plus
    ``margin_A``; a refused model's maps are boxed around that model alone.

    ``entries`` hold ``label``, ``map`` (unit-cell CCP4 map), ``model`` (the model
    whose phases made the map) and optionally ``model_label``.  Each model is
    first placed on the reference (space-group operation, lattice translations,
    allowed origin shift; never a hand change).  A map whose model needed an
    origin shift is moved with it, and the moved unit-cell map is written too.
    Boxes go next to each map unless ``out_dir`` is given.  ``write_res=False``
    writes the models as PDB only (used inside run directories, so no extra
    SHELX .res file appears next to an endpoint).
    """

    placement = placement or PlacementOptions()
    reference_model = Path(reference_model).expanduser().resolve()
    reference_raw = parse_model(reference_model)
    reference, assembly = assemble_molecules(reference_raw)
    models_dir = Path(models_dir) if models_dir is not None else (Path(out_dir) if out_dir is not None else None)
    record: dict[str, object] = {
        "status": "ok",
        "reference_model": str(reference_model),
        "reference_model_sha256": sha256_file(reference_model),
        "reference_assembly": assembly,
        "box": {},
        "margin_A": float(margin_A),
        "sigma_scaled": bool(sigma_scaled),
        "placements": {},
        "maps": {},
        "models": {},
        "warnings": [],
        "definition": (
            "each map is boxed around the reference model and every model placed on it, plus the margin; values "
            "come from the periodic full-cell map, so a molecule crossing a cell edge is whole; unit-cell maps are "
            "unchanged"
        ),
    }

    def write_models(key: str, source: Path, model: Mapping[str, object], directory: Path, remarks: list[str]) -> dict:
        """Write one model as PDB (and as .res for a SHELX source when ``write_res``); return the written paths."""

        written: dict[str, str] = {}
        stem = _stem_for(key)
        pdb = write_model_pdb(model, directory / f"{stem}.pdb", remarks=remarks)
        written["pdb"] = str(pdb.resolve())
        if write_res and source.suffix.lower() in {".res", ".ins"}:
            res = write_placed_res(source, directory / f"{stem}.res", model, remarks)
            written["res"] = str(res.resolve())
            written["res_sha256"] = sha256_file(res)
        return written

    reference_directory = models_dir
    if reference_directory is None:
        first_map = next((Path(str(entry["map"])) for entry in entries if entry.get("map")), None)
        reference_directory = first_map.parent if first_map is not None else reference_model.parent
    reference_remarks = [f"GROND model-box reference {reference_model.name} sha256 {record['reference_model_sha256'][:16]}"]
    if not assembly["already_contiguous"]:
        reference_remarks.append("fragments made contiguous by lattice translations only")
    record["models"]["reference"] = write_models(
        "model_box_reference", reference_model, reference, reference_directory, reference_remarks
    )

    placements: dict[str, tuple[dict | None, dict[str, object], dict[str, object] | None, dict | None]] = {}
    for entry in entries:
        model_path = Path(str(entry["model"])).expanduser().resolve()
        key = str(entry.get("model_label") or model_path.stem)
        if key in placements:
            continue
        own_box = None
        own = None
        directory = models_dir if models_dir is not None else Path(str(entry["map"])).parent
        try:
            moving = parse_model(model_path)
            aligned, report = place_model_on_reference(moving, reference, placement)
            report["model"] = str(model_path)
            report["model_sha256"] = sha256_file(model_path)
        except PlacementRefused as exc:
            report = {**exc.report, "error": str(exc), "model": str(model_path)}
            aligned = None
            record["warnings"].append(f"{key}: {exc}")
            # Never apply the refused operation.  Box this model's maps around the
            # model itself instead, so its figure maps still show it whole.
            own, own_assembly = assemble_molecules(moving)
            own_box = model_fractional_box(own, margin_A)
            report["fallback"] = {
                "box_reference": "own_model",
                "assembly": own_assembly,
                "box": own_box,
                "note": "placement refused; these maps are boxed around their own model and do not overlay the "
                "reference",
            }
            record["models"][key] = write_models(
                f"{key}_own", model_path, own, directory,
                [f"GROND model box around {model_path.name} itself: placement on {reference_model.name} refused",
                 str(exc)[:70]],
            )
        except (OSError, ValueError, RuntimeError) as exc:
            report = {"status": "failed", "error": f"{type(exc).__name__}: {exc}", "model": str(model_path)}
            aligned = None
            record["warnings"].append(f"{key}: {exc}")
        placements[key] = (aligned, report, own_box, own)
        record["placements"][key] = report
        if aligned is not None and (write_all_models or not report.get("identity")):
            operation = report["operation"]
            remarks = [
                f"GROND placed {model_path.name} on {reference_model.name}: {operation['triplet']}",
                f"RMSD {report['rmsd_A']:.4f} A over {report['matched_atoms']} non-H atoms ({report['matching']})",
                "coordinates moved for figures only; not a refinement input",
            ]
            record["models"][key] = write_models(f"{key}_placed", model_path, aligned, directory, remarks)

    # One box for every placed map, so they overlay: it covers the reference and
    # every placed model (a placed model can have atoms the reference lacks, for
    # example H atoms when the reference is H-free).
    placed = {key: value[0] for key, value in placements.items() if value[0] is not None}
    covered = {**reference, "atoms": [*reference["atoms"], *(atom for model in placed.values() for atom in model["atoms"])]}
    box = {**model_fractional_box(covered, margin_A), "models_covered": ["reference", *placed]}
    record["box"] = box

    for entry in entries:
        label = str(entry["label"])
        map_path = Path(str(entry["map"])).expanduser()
        model_path = Path(str(entry["model"])).expanduser().resolve()
        key = str(entry.get("model_label") or model_path.stem)
        aligned, report, own_box, own = placements[key]
        destination_dir = Path(out_dir) if out_dir is not None else map_path.parent
        item: dict[str, object] = {"label": label, "model_label": key}
        if aligned is None and own_box is None:
            item.update(status="skipped_placement_failed", error=str(report.get("error", "")))
            record["maps"][label] = item
            continue
        entry_box = box if aligned is not None else own_box
        item["box_reference"] = "reference_model" if aligned is not None else "own_model_placement_refused"
        try:
            ccp4, source = load_unit_cell_map(map_path)
            item["source"] = source
            operation = report.get("operation") if aligned is not None else None
            if operation and operation.get("map_transform_required"):
                item["origin_shift"] = shift_map_origin(ccp4, operation["map_origin_shift_fractional"])
                moved = write_unit_cell_map(ccp4, destination_dir / f"{_stem_for(label)}_placed_unit_cell.map")
                item["placed_unit_cell_map"] = moved
                source = {**source, **load_unit_cell_map(Path(moved["map"]))[1]}
            name = _stem_for(label) + MODEL_BOX_SUFFIX
            absolute = write_box_map(ccp4, entry_box, destination_dir / f"{name}.map")
            absolute["min_atom_clearance_A"] = (
                min(atom_box_clearance_A(reference, absolute), atom_box_clearance_A(aligned, absolute))
                if aligned is not None
                else atom_box_clearance_A(own, absolute)
            )
            item["box_map"] = absolute
            if sigma_scaled:
                item["box_map_sigma"] = write_box_map(
                    ccp4, entry_box, destination_dir / f"{name}_sigma.map", sigma_scaled=True,
                    unit_cell_mean=float(source["unit_cell_mean"]), unit_cell_rms=float(source["unit_cell_rms"]),
                )
            item["status"] = "ok" if aligned is not None else "ok_own_model_box"
        except (OSError, ValueError, RuntimeError) as exc:
            item.update(status="failed", error=f"{type(exc).__name__}: {exc}")
            record["warnings"].append(f"{label}: {exc}")
        record["maps"][label] = item

    statuses = [str(item.get("status")) for item in record["maps"].values()]  # type: ignore[union-attr]
    written = {"ok", "ok_own_model_box"}
    if not statuses or all(status not in written for status in statuses):
        record["status"] = "failed"
    elif any(status not in written for status in statuses):
        record["status"] = "partial"
    elif any(status == "ok_own_model_box" for status in statuses):
        record["status"] = "ok_with_refused_placements"
    return record
