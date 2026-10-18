"""Recalculate scattering with fixed geometry, beam membership, scales and errors."""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np

from grond.hashing import sha256_file
from grond.legacy.engine.bloch_engine import Config, XDSBlochPort, XDSData, parse_model
from grond.regularize.multidataset import native_metrics_by_split
from grond.regularize.observations import row_contract_selection, target_arrays
from grond.regularize.prediction_cache import external_geometry_fingerprints
from grond.workflows.matched_reference import save_json, write_rows


def prediction_port(dataset: dict, model: Path, reference_model: Path, cache: Path) -> XDSBlochPort:
    """Build a Bloch prediction port for ``model`` on a reference dataset's fixed rows and beam pool.

    The HKL, row table and scan geometry must still match their recorded hashes. For a model other than the reference,
    the beam pool must be identical and only the pool structure factors that rank the beam support come from the
    reference model; every prediction still comes from ``model``.
    """

    cfg = Config(**dataset["config"])
    for field, path in (("hkl_sha256", dataset["hkl"]), ("row_table_sha256", dataset["row_table"])):
        if sha256_file(Path(path)) != dataset["provenance"][field]:
            raise ValueError(f"Reference file changed during the run: {path}")
    transform = dataset["summary"]["config"]["hkl_transform"]
    data = XDSData(Path(dataset["hkl"]), hkl_transform=transform)
    if external_geometry_fingerprints(data, cfg) != dataset["provenance"]["geometry"]:
        raise ValueError("Reference scan geometry changed during the run")
    target, _ = target_arrays(data, dataset["summary"]["config"]["target_mode"])
    row_contract_selection(data, target, Path(dataset["row_table"]))
    reference = XDSBlochPort(data, parse_model(reference_model), cfg, cache_dir=cache, res_hash=sha256_file(reference_model))
    if model.resolve() == reference_model.resolve():
        return reference
    current = XDSBlochPort(data, parse_model(model), cfg, cache_dir=cache, res_hash=sha256_file(model))
    if not np.array_equal(current.pool_hkl, reference.pool_hkl):
        raise ValueError("The beam pool changed despite fixed geometry and resolution")
    # Only the support ranking comes from full H. Every matrix element, Born
    # amplitude and propagated wave still comes from the current H-free model.
    current.pool_ug = reference.pool_ug.copy()
    return current


def update_predictions(dataset: dict, pred: dict, *, check_reference: bool = False) -> list[dict]:
    """Replace the dynamical and Born predictions of a reference dataset's rows; scales and sigmas are kept.

    Beam membership, Sg window, jitter and Bragg angle must not change. Scaled values reuse each row's recorded scale.
    ``check_reference`` also requires the new raw values to equal the recorded ones.
    """

    old_by_index = {int(row["row_index"]): row for row in dataset["rows"]}
    output = []
    for pos, index in enumerate(pred["indices"]):
        old = old_by_index[int(index)]
        new = dict(old)
        if int(pred["basis_size"][pos]) != int(old["basis_size"]) or format(int(pred["basis_support_hash64"][pos]), "016x") != old["basis_support_hash64"]:
            raise ValueError(f"{dataset['name']}: beam membership changed at observation {index}")
        for saved, calculated in (("sg_window_inv_A", "sg_window"), ("sg_jitter_sigma_inv_A", "sg_jitter_sigma"), ("omega_bragg_deg", "omega_bragg")):
            if not np.isclose(float(old[saved]), float(pred[calculated][pos]), rtol=1e-9, atol=1e-10, equal_nan=True):
                raise ValueError(f"{dataset['name']}: {saved} changed at observation {index}")
        dyn = float(pred["dyn"][pos]) * float(old.get("row_resolution_factor", 1))
        born = float(pred["born"][pos])
        if not np.isfinite([dyn, born]).all() or min(dyn, born) <= 0:
            raise ValueError(f"{dataset['name']}: nonpositive prediction at observation {index}")
        new["dyn_raw_unmodulated"] = float(pred["dyn"][pos])
        for kind, raw in (("dyn", dyn), ("born", born)):
            previous = float(old[kind + "_raw"])
            if previous <= 0:
                raise ValueError("The reference has no recoverable scale for a zero prediction")
            if check_reference and not np.isclose(raw, previous, rtol=2e-7, atol=1e-12):
                raise ValueError(f"{dataset['name']}: full-H replay differs for {kind}, observation {index}: {raw} versus {previous}")
            new[kind + "_raw"] = raw
            new[kind + "_scaled"] = raw * float(old[kind + "_scaled"]) / previous
            new[kind + "_residual_over_sigma"] = (new[kind + "_scaled"] - float(old["target_I"])) / float(old["sigma_eff"])
        new["q_dyn_over_born"] = dyn / max(born, 1e-30)
        output.append(new)
    return output


def predict_dataset(dataset: dict, model: Path, reference_model: Path, out: Path, *, workers: int, smoke: bool = False) -> list[dict]:
    """Predict a reference dataset's rows for ``model`` and write native_dyn_calc.csv and summary.json in ``out``.

    ``smoke`` predicts only a few rows (spread, largest basis, brightest) and checks them against the reference. Returns
    the updated rows.
    """

    port = prediction_port(dataset, model, reference_model, out.parent / "ug_cache")
    rows = dataset["rows"]
    positions = np.arange(len(rows))
    if smoke:
        positions = np.unique(np.r_[np.linspace(0, len(rows) - 1, min(16, len(rows)), dtype=int),
                                    np.argmax([int(row["basis_size"]) for row in rows]),
                                    np.argmax([float(row["target_I"]) for row in rows])])
    indices = np.array([int(rows[i]["row_index"]) for i in positions])
    pred = port.predict_rows(indices, workers=workers, chunksize=4)
    updated = update_predictions(dataset, pred, check_reference=smoke)
    out.mkdir(parents=True, exist_ok=True)
    write_rows(out / "native_dyn_calc.csv", updated)
    summary = copy.deepcopy(dataset["summary"])
    summary["inputs"] = {**summary.get("inputs", {}), "res": str(model), "res_sha256": sha256_file(model)}
    summary["config"].update(res=str(model), out_dir=str(out))
    summary["artifacts"] = {"native_row_table": {"path": str(out / "native_dyn_calc.csv"),
                                                 "sha256": sha256_file(out / "native_dyn_calc.csv")}}
    nuisance = summary["best_native"]["nuisance"]
    summary["best_native"]["metrics"] = native_metrics_by_split(updated, float(nuisance.get("native_wght_a", 0)), float(nuisance.get("native_wght_b", 0)))
    summary["matched_reference"] = {"source_row_table": dataset["row_table"], "reference_model": str(reference_model),
                                    "reference_model_sha256": sha256_file(reference_model), "rows": len(updated),
                                    "beam_membership_verified": True, "scales_refitted": False,
                                    "nuisance_refitted": False, "full_H_replay": smoke,
                                    "prediction_seconds": float(pred["seconds"])}
    save_json(out / "summary.json", summary)
    return updated
