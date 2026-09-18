"""Make conventional, corrected-HKL and native maps on the same Fourier support."""

from __future__ import annotations

import csv
import math
import shutil
from collections import defaultdict
from pathlib import Path

import gemmi
import numpy as np

from grond.bloch.model import fcalc_res, parse_model
from grond.macromolecule import sha256_file
from grond.maps import native_dynamic as native
from grond.regularize.multidataset import canonical_model_friedel
from grond.workflows.finish_maps import _shelx2map_compatible_fcf, _write_finish_list6_ins
from grond.workflows.matched_reference import save_json
from grond.workflows.model_alignment import require_no_hydrogen


def coordinate_cif(model_path: Path, destination: Path) -> None:
    model = parse_model(model_path)
    wavelength = next(float(line.split()[1]) for line in model_path.read_text().splitlines() if line.strip().upper().startswith("CELL "))
    native._write_dynamic_cif(destination, model=model, model_path=model_path, row_tables=[], coefficients=[],
                              summary={}, maps={}, wavelength_A=wavelength)
    doc = gemmi.cif.read_file(str(destination))
    block = doc.sole_block()
    block.name = "H_free_coordinates"
    # This is a coordinate file, not a claim that its conventional model was dynamically refined.
    block.set_pair("_audit_creation_method", "'GROND matched-map coordinate export'")
    for item in list(block):
        if item.pair and item.pair[0].startswith("_grond_dynamic_map_"):
            item.erase()
    doc.write_file(str(destination))


def _fcf_rows(path):
    document = gemmi.cif.read_file(str(path))
    loop = document.sole_block().find_loop("_refln_index_h").get_loop()
    rows = [list(loop.values[i:i + loop.width()]) for i in range(0, len(loop.values), loop.width())]
    return document, loop, rows


def _native_fcf(row_tables, model_path, template, output):
    rows, _ = native._read_labeled_rows([(p.parent.name, p) for p in row_tables])
    model = parse_model(model_path)
    coefficients, metadata = native._merge_native_amplitudes(rows, model, native.NativeDynamicMapOptions())
    projected, projection = native._prepare_real_map_coefficients(coefficients, model, "fo")
    groups = defaultdict(list)
    for row in coefficients:
        key = canonical_model_friedel(int(row["h"]), int(row["k"]), int(row["l"]), model["symmetry_ops"])
        groups[key].append(float(row["sigF_dyn_obs"]))
    doc, loop, _ = _fcf_rows(template)
    tags = list(loop.tags)
    result = []
    for hkl, sigmas in sorted(groups.items()):
        fo = abs(projected[hkl])
        sigf = 1 / math.sqrt(sum(1 / max(s * s, 1e-30) for s in sigmas))
        fc = fcalc_res(model, hkl)
        values = dict(zip(("_refln_index_h", "_refln_index_k", "_refln_index_l"), hkl, strict=True))
        values.update(_refln_F_squared_meas=fo * fo, _refln_F_squared_sigma=2 * fo * sigf,
                      _refln_F_calc=abs(fc), _refln_phase_calc=math.degrees(np.angle(fc)))
        result.append([str(values[tag]) for tag in tags])
    if not result:
        raise ValueError("No native dynamical map coefficients")
    loop.set_all_values([list(column) for column in zip(*result, strict=True)])
    doc.sole_block().set_pair("_shelx_F_calc_maximum", str(max(float(r[tags.index("_refln_F_calc")]) for r in result)))
    doc.write_file(str(output))
    save_json(output.with_suffix(".json"), {"amplitudes": metadata, "real_map_projection": projection,
                                           "map_weighting": "shelx2map -w 0 and -w 1 applied to the same inferred native amplitudes"})
    return coefficients


def _dense_map(source, destination, model, spacing):
    raw = gemmi.read_ccp4_map(str(source))
    raw.setup(float("nan"))
    values = np.asarray(raw.grid).copy()
    if not np.isfinite(values).all() or np.std(values) <= 0:
        raise ValueError(f"Empty or nonfinite map: {source}")
    frequencies = [np.rint(np.fft.fftfreq(n) * n).astype(int) for n in values.shape]
    hkl = np.stack(np.meshgrid(*frequencies, indexing="ij"), axis=-1).reshape(-1, 3)
    shape = native._grid_shape(model["cell"], hkl, spacing)
    if math.prod(shape) > 32_000_000:
        raise ValueError("Requested map grid exceeds 32 million points; use a coarser --map-grid-spacing")
    transform = np.zeros(shape, dtype=complex)
    transform[tuple((hkl % shape).T)] = np.fft.ifftn(values).ravel()
    dense = np.fft.fftn(transform).real.astype(np.float32)
    native._write_ccp4(destination, dense, model)
    return {"shape": list(shape), "mean": float(dense.mean()), "sd": float(dense.std()),
            "minimum": float(dense.min()), "maximum": float(dense.max())}


def build_matched_maps(conventional: Path, final: Path, out: Path, shelxl: str, maptool: str, run, spacing: float) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    fcfs, models = {}, {}
    for name, source in (("conventional", conventional), ("grond_endpoint", final)):
        directory = out / name
        directory.mkdir(exist_ok=True)
        _write_finish_list6_ins(source / "model.res", directory / "model.ins")
        shutil.copy2(source / "model.hkl", directory / "model.hkl")
        run([shelxl, "model", "-t1"], directory, "zero_cycle_LIST6")
        require_no_hydrogen(directory / "model.res")
        coordinate_cif(directory / "model.res", directory / "model.cif")
        fcfs[name], _ = _shelx2map_compatible_fcf(directory / "model.fcf")
        models[name] = directory / "model.res"
    directory = out / "native_dynamical"
    directory.mkdir(exist_ok=True)
    shutil.copy2(final / "model.res", directory / "model.res")
    coordinate_cif(directory / "model.res", directory / "model.cif")
    row_tables = sorted(final.glob("*/native_dyn_calc.csv"))
    native_coefficients = _native_fcf(row_tables, directory / "model.res", fcfs["grond_endpoint"], directory / "model.fcf")
    fcfs["native_dynamical"] = directory / "model.fcf"
    models["native_dynamical"] = directory / "model.res"
    reference = parse_model(models["native_dynamical"])
    def key(hkl):
        return canonical_model_friedel(*map(int, hkl), reference["symmetry_ops"])
    sets = {name: {key(row[:3]) for row in _fcf_rows(path)[2]} for name, path in fcfs.items()}
    common = set.intersection(*sets.values())
    if not common:
        raise ValueError("The three map types have no common reflection support")
    save_json(out / "COMMON_SUPPORT.json", {"count": len(common), "families": sorted(common), "full_counts": {k: len(v) for k, v in sets.items()}})
    records = []
    for name, fcf in fcfs.items():
        model = parse_model(models[name])
        for support in ("full", "common"):
            doc, loop, rows = _fcf_rows(fcf)
            if support == "common":
                rows = [row for row in rows if key(row[:3]) in common]
            loop.set_all_values([list(column) for column in zip(*rows, strict=True)])
            dmin = min(gemmi.UnitCell(*model["cell"]).calculate_d(tuple(map(int, row[:3]))) for row in rows)
            doc.sole_block().set_pair("_reflns_d_resolution_high", str(dmin))
            selected = fcf.parent / f"{support}.fcf"
            doc.write_file(str(selected))
            for weight in (0, 1):
                for kind, mode in (("fofc", "0"), ("2fofc", "2")):
                    label = f"{support}_w{weight}_{kind}"
                    raw = fcf.parent / f"{label}_raw.map"
                    command = [maptool, "-w", str(weight), "-m", mode, "-o", str(raw), selected.stem]
                    run(command, fcf.parent, label)
                    output = fcf.parent / f"{label}.map"
                    stats = _dense_map(raw, output, model, spacing)
                    if name == "native_dynamical" and support == "full" and weight == 0 and kind == "fofc":
                        expected, _ = native._synthesize_map(native_coefficients, model, "fofc", spacing, 32_000_000)
                        observed = gemmi.read_ccp4_map(str(output))
                        observed.setup(float("nan"))
                        actual = np.asarray(observed.grid)
                        if actual.shape != expected.shape:
                            raise ValueError("Native and shelx2map grids differ; cannot verify the Fourier convention")
                        a, b = actual.astype(float).ravel(), expected.astype(float).ravel()
                        a -= a.mean()
                        b -= b.mean()
                        correlation = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
                        save_json(directory / "FOURIER_CHECK.json", {"correlation": correlation, "required": 0.999,
                                                                  "comparison": "shelx2map w0 versus direct native Fourier synthesis"})
                        if correlation < 0.999:
                            raise ValueError(f"Native Fourier convention check failed: correlation={correlation}")
                    records.append({"type": name, "support": support, "weight": weight, "kind": kind,
                                    "map": str(output), "model": str(models[name]), "cif": str(fcf.parent / "model.cif"),
                                    "map_sha256": sha256_file(output), "model_sha256": sha256_file(models[name]),
                                    "reflections": len(rows), "d_min_A": dmin, "command": command, **stats})
    save_json(out / "MAP_MANIFEST.json", records)
    with (out / "MAPS.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (out / "README.md").write_text(
        "Use common_w0_fofc.map or common_w1_fofc.map with the model.cif in the same directory. "
        "All three map types share the reflection families in COMMON_SUPPORT.json. Full-support and 2Fo-Fc variants are also retained. "
        "w0 and w1 mean shelx2map's unweighted and weighted coefficients; they do not change the refined coordinates. "
        "Native amplitudes come from the H-free Bloch prediction and H-free model phases, with inverse-variance symmetry/Friedel combination. "
        "These are model-conditioned native dynamical maps, not an independent direct-dynamical least-squares refinement. "
        "Map SD is display normalization, not a calibrated hydrogen-detection Z score. Commands and input hashes are in MAP_MANIFEST.json.\n"
    )
    return {"manifest": str(out / "MAP_MANIFEST.json"), "maps": len(records), "common_families": len(common)}
